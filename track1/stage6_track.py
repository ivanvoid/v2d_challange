#!/usr/bin/env python3
"""Stage 6 — object pose for every frame: anchors, bidirectional tracking, re-finding a lost object,
global selection.

1. anchors   fixed windows of --anchor_every frames; per window the eligible frame of best quality
               quality q = visible (mask area / median, capped at 1) * (1 - body occlusion)^2
                           * sharpness^0.3 (Laplacian variance in the object box, relative)
                           * stillness (exp(-mask_change / 0.1))
               eligible  = mask area >= --min_anchor_vis x median, not touching the image border,
                           body occlusion <= --max_anchor_occ
             (no anchor in a window without a frame of quality >= --min_anchor_quality) + the stage-5e
             reference frame. Each anchor is registered from scratch; distinct orientations (>= 45 deg
             apart) within --hyp_margin of the best fit are kept (max --max_hyps).
2. tracking  every anchor hypothesis is tracked FORWARD to the next anchor and BACKWARD to the previous
             one (first/last anchors also run to the video ends); a chain arriving at an anchor is handed
             over (continued) unless the anchor has an orientation within --handover_deg. A low-fit frame
             is re-tracked once from the previous pose with more refiner iterations (a local retry only).
3. re-find   a visible stretch of >= --lost_after frames where NO candidate fits (IoU < --iou_ok) means
             the object was lost there: the best frame of the stretch is registered from scratch (a new
             anchor) and tracked both ways across the stretch; up to --recover_rounds rounds.
4. selection Viterbi over all frames: one candidate per frame minimising
                 sum_f -w_f * fit_f  +  rot_change / --deg_per_unit  +  trans_change / (--size_per_unit x size)
                                     (+ --flip_cost if > 60 deg in one frame)
             fit = IoU - colour error / 255 - --depth_weight * min(depth error / --depth_scale_cm, 1)
               IoU      pixels where the body is IN FRONT of the object (MoGe depth) do not count against a
                        pose: a handle under the holding hand is not penalised, one in empty space is
               depth    median |rendered - MoGe depth| on valid depth pixels only; not used for "thin"
                        objects (a hoop rim carries the depth of the wall behind it)
             translation cost relative to the object's size (a 1 m hoop vs a 0.2 m roller).
States per frame: observed (IoU >= --iou_ok), tracked (worse, but within --lost_after frames of an
observed frame), lost (further from any observed frame), occluded (mask smaller than --vis_frac of its
median: hidden, or missing in segmentation).
Writes out/episode_XXXXXX/poses/<frame:06d>.json (object-to-camera, original mesh frame) and
track/states.json (per frame: state, chosen source, fit, best alternative; anchors incl. re-found ones).

    /opt/venvs/foundation_pose/bin/python track1/stage6_track.py --episode 28 --out_root out \
        --weights weights --debug_log logs/stages/06_track/episode_000028.debug.log
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logutil import log, to_debug  # noqa: E402

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask, Transform3d  # noqa: E402
from v2d.common.datatypes import Image as V2dImage  # noqa: E402
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker, nvdiffrast_render  # noqa: E402
from v2d.mesh.lib.mesh import Mesh  # noqa: E402


def rot_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--debug_log", required=True)
    ap.add_argument("--min_anchor_vis", type=float, default=0.6, help="anchor mask area >= this x median")
    ap.add_argument("--hyp_margin", type=float, default=0.10)
    ap.add_argument("--max_hyps", type=int, default=3)
    ap.add_argument("--anchor_every", type=int, default=60, help="fixed anchor windows (frames)")
    ap.add_argument("--anchor_select", choices=("quality", "vis"), default="quality",
                    help="quality: v2 windows + best-quality frame (skip window if all poor); vis: plain v2 "
                         "(largest visible object)")
    ap.add_argument("--max_anchor_occ", type=float, default=0.25, help="quality mode: max body occlusion")
    ap.add_argument("--min_anchor_quality", type=float, default=0.3, help="no anchor in a window without such a frame")
    ap.add_argument("--track_iter", type=int, default=5)
    ap.add_argument("--retrack_iter", type=int, default=10)
    ap.add_argument("--register_iter", type=int, default=10)
    ap.add_argument("--iou_ok", type=float, default=0.5)
    ap.add_argument("--deg_per_unit", type=float, default=30.0, help="rotation change costing one unit of fit")
    ap.add_argument("--size_per_unit", type=float, default=0.25,
                    help="translation change costing one unit of fit, as a fraction of the object's size")
    ap.add_argument("--depth_weight", type=float, default=0.2, help="fit penalty for depth disagreement (max)")
    ap.add_argument("--depth_scale_cm", type=float, default=10.0, help="depth error giving the full penalty")
    ap.add_argument("--lost_after", type=int, default=30,
                    help="frames without any fitting candidate before the object counts as lost (re-find)")
    ap.add_argument("--recover_rounds", type=int, default=2)
    ap.add_argument("--flip_cost", type=float, default=3.0, help="extra cost for a > 60 deg change in one frame")
    ap.add_argument("--vis_frac", type=float, default=0.15)
    ap.add_argument("--handover_deg", type=float, default=45.0,
                    help="an arriving chain is continued unless an anchor orientation is within this")
    args = ap.parse_args()
    ov_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage6_overrides.json")
    overrides = json.load(open(ov_path)).get(str(args.episode), {}) if os.path.exists(ov_path) else {}
    for k, v in overrides.items():
        if not k.startswith("_"):
            setattr(args, k, v)

    ep = f"episode_{args.episode:06d}"
    if overrides:
        log(f"{ep}: per-episode overrides from stage6_overrides.json: "
            f"{ {k: v for k, v in overrides.items() if not k.startswith('_')} }")
    W = os.path.join(os.path.abspath(args.out_root), ep)
    meta = json.load(open(os.path.join(W, "track1_meta.json")))
    n, off = meta["n_frames"], meta["frame_offset"]
    intr = CameraIntrinsics.load(os.path.join(W, "intrinsics_stable.json"))
    K = intr.to_matrix()
    ref = json.load(open(os.path.join(W, "mesh", "choice.json")))["chosen"]
    oc = os.path.join(W, "mesh", "object_choice.json")          # stage 5e: mesh may come from another video
    if os.path.exists(oc) and json.load(open(oc)).get("track_ref_frame") is not None:
        ref = int(json.load(open(oc))["track_ref_frame"])
    poses_dir, track_dir = os.path.join(W, "poses"), os.path.join(W, "track")
    os.makedirs(poses_dir, exist_ok=True)
    os.makedirs(track_dir, exist_ok=True)
    t0 = time.time()

    import logging
    logging.disable(logging.INFO)                           # FoundationPose logs every crop/render step
    with to_debug(args.debug_log, "load tracker"):
        tr = FoundationPoseTracker(Mesh.load(os.path.join(W, "mesh_scaled.obj")),
                                   os.path.join(os.path.abspath(args.weights), "foundation_pose"))
    est = tr._est
    tfc = est.get_tf_to_centered_mesh().cpu().numpy()
    tfc_inv = np.linalg.inv(tfc)
    import paths
    obj = next(json.loads(l) for l in open(paths.META) if json.loads(l)["episode_index"] == args.episode)["object"]
    thin = bool(json.load(open(paths.PRIORS)).get(obj, {}).get("thin"))
    mv = np.array([[float(x) for x in l.split()[1:4]] for l in open(os.path.join(W, "mesh_scaled.obj"))
                   if l.startswith("v ")])
    size_cm = float(np.ptp(mv, axis=0).max()) * 100
    cm_unit = args.size_per_unit * size_cm
    log(f"{ep}: {obj}, size {size_cm:.0f} cm -> translation {cm_unit:.1f} cm costs one unit; depth term "
        f"{'off (thin object)' if thin else f'on (weight {args.depth_weight})'}")

    # ---- per-frame quality: visibility, border, body occlusion, sharpness, stillness
    import cv2
    body = torch.load(os.path.join(W, "body", "mhr_mesh.pt"), map_location="cpu", weights_only=False)
    bv, bf = body["pred_vertices"].numpy(), body["faces"].numpy()
    areas, border = np.zeros(n), np.zeros(n, bool)
    occ, sharp, change = np.zeros(n), np.zeros(n), np.zeros(n)
    occluder = [None] * n          # packed bits: body pixels IN FRONT of the object, around it (even frames)
    m_prev = None
    for f in range(n):
        m = np.asarray(Image.open(os.path.join(W, "masks", "1", f"{f:06d}.png"))) > 0
        areas[f] = m.sum()
        if m_prev is not None:
            u = (m | m_prev).sum()
            change[f] = 1.0 - (m & m_prev).sum() / u if u else 1.0
        m_prev = m
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        border[f] = x0 <= 1 or y0 <= 1 or x1 >= m.shape[1] - 1 or y1 >= m.shape[0] - 1
        gray = cv2.cvtColor(cv2.imread(os.path.join(W, "frames", f"{f:06d}.png")), cv2.COLOR_BGR2GRAY)
        sharp[f] = cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F).var()
        if f % 2 == 0 or f == n - 1:                       # body occlusion (projected body in front)
            px = cv2.imread(os.path.join(W, "depth", f"{f:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
            d = 1.0 / np.maximum(px / 65535.0, 1e-6) - 1.0
            uvw = bv[f] @ K.T
            uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
            sil = np.zeros(m.shape, np.uint8)
            cv2.fillPoly(sil, list(np.round(uv[bf]).astype(np.int32)), 1)
            pad = int(0.15 * max(x1 - x0, y1 - y0)) + 5
            reg = np.zeros(m.shape, bool)
            reg[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = True
            in_front = (sil > 0) & (d < float(np.median(d[m])) - 0.05) & ~m
            front = in_front & reg
            occ[f] = front.sum() / (m.sum() + front.sum())
            pad2 = int(0.5 * max(x1 - x0, y1 - y0)) + 10       # wider: a hidden handle can stick out further
            reg2 = np.zeros(m.shape, bool)
            reg2[max(0, y0 - pad2):y1 + pad2, max(0, x0 - pad2):x1 + pad2] = True
            occluder[f] = np.packbits(in_front & reg2)
        else:
            occ[f] = -1
    for f in range(n):                                     # odd frames: copy the previous even frame
        if occ[f] < 0:
            occ[f] = occ[f - 1]
            occluder[f] = occluder[f - 1]
    med_area = float(np.median(areas[areas > 0])) if (areas > 0).any() else 1.0
    vis = areas / med_area
    smax = np.percentile(sharp[sharp > 0], 95) if (sharp > 0).any() else 1.0
    quality = (np.minimum(vis, 1.0) * (1 - occ) ** 2 * np.clip(sharp / smax, 0, 1) ** 0.3
               * np.exp(-change / 0.1))
    eligible = (vis >= args.min_anchor_vis) & ~border
    if args.anchor_select == "quality":
        eligible &= occ <= args.max_anchor_occ
    log(f"{ep}: frame quality computed ({time.time() - t0:.0f}s): {eligible.sum()}/{n} frames eligible as "
        f"anchors; mask change median {np.median(change):.3f}, 90th pct {np.percentile(change, 90):.3f}")

    def load(f):
        rgb = np.asarray(Image.open(os.path.join(W, "frames", f"{f:06d}.png")).convert("RGB"))
        return rgb, DepthImage.load(os.path.join(W, "depth", f"{f:06d}.png")), \
            Mask.load(os.path.join(W, "masks", "1", f"{f:06d}.png"))

    HW = (intr.height, intr.width)

    def score(P_list, rgb, depth, mask, f=None):
        Pc = np.stack([P @ tfc_inv for P in P_list])
        with torch.no_grad():
            col, dep, _ = nvdiffrast_render(K, intr.height, intr.width,
                                            torch.as_tensor(Pc, device="cuda", dtype=torch.float),
                                            glctx=tr._glctx, mesh_tensors=est.mesh_tensors, get_normal=False)
        col, dep = col.cpu().numpy(), dep.cpu().numpy()
        m, d = mask.mask.astype(bool), depth.depth
        hid = (np.unpackbits(occluder[f])[:HW[0] * HW[1]].reshape(HW).astype(bool)
               if f is not None and occluder[f] is not None else None)
        out = []
        for i in range(len(P_list)):
            sil = dep[i] > 0.001
            both = sil & m
            seen = sil & ~hid if hid is not None else sil    # model pixels behind the body are not judged
            iou = float(both.sum() / max((seen | m).sum(), 1))
            ce = float(np.abs(col[i][both] * 255 - rgb[both]).mean()) if both.any() else 255.0
            valid = both & (d > 0)
            dcm = float(np.median(np.abs(dep[i][valid] - d[valid])) * 100) if valid.sum() >= 30 else None
            fit = iou - ce / 255.0
            if dcm is not None and not thin:
                fit -= args.depth_weight * min(dcm / args.depth_scale_cm, 1.0)
            out.append({"iou": iou, "colour_err": ce, "depth_cm": dcm, "fit": fit})
        return out

    # ---- 1. anchors: fixed windows (v2). Per window:
    #      quality  the eligible frame with the best quality (visible, unoccluded, sharp, still);
    #               no anchor if that quality < --min_anchor_quality
    #      vis      the eligible frame with the largest visible object (plain v2)
    #      (motion-adaptive spacing — v3 — gave more flips and is not used)
    anchor_frames, skipped = {ref}, []
    for s_ in range(0, n, args.anchor_every):
        idx = [f for f in range(s_, min(n, s_ + args.anchor_every)) if eligible[f]]
        if args.anchor_select == "quality":
            best_f = max(idx, key=lambda f: quality[f]) if idx else None
            ok = best_f is not None and quality[best_f] >= args.min_anchor_quality
        else:
            best_f = max(idx, key=lambda f: vis[f]) if idx else None
            ok = best_f is not None
        if ok:
            anchor_frames.add(best_f)
        else:
            skipped.append(s_ + off)
    log(f"{ep}: {len(anchor_frames)} anchor candidates (windows of {args.anchor_every} frames, select by "
        f"{args.anchor_select}" + (f"; no anchor in windows starting at video frames {skipped}" if skipped else "")
        + ")")
    def register_anchor(a, why="anchor"):
        """Register from scratch on frame a; distinct orientations within --hyp_margin of the best fit."""
        rgb, depth, mask = load(a)
        with to_debug(args.debug_log, f"register {why} {a}"):
            tr.register(V2dImage(data=rgb), depth, mask, intr, iteration=args.register_iter)
        distinct = []
        for h in est.poses[:40].cpu().numpy():
            P = h @ tfc
            if all(rot_deg(P, g) >= 45 for g in distinct):
                distinct.append(P)
            if len(distinct) == 6:
                break
        sc = score(distinct, rgb, depth, mask, a)
        best = max(s_["fit"] for s_ in sc)
        keep = sorted([i for i in range(len(distinct)) if sc[i]["fit"] >= best - args.hyp_margin],
                      key=lambda i: -sc[i]["fit"])[:args.max_hyps]
        if sc[keep[0]]["iou"] < 0.4:
            log(f"  {why} video frame {a + off}: best IoU {sc[keep[0]]['iou']:.2f} too low, dropped")
            return None
        fits_txt = ", ".join("%.2f" % sc[i]["fit"] for i in keep)
        log(f"  {why} video frame {a + off}: quality {quality[a]:.2f} (visible {vis[a]:.2f}, occluded "
            f"{occ[a]:.2f}, mask change {change[a]:.3f}); {len(distinct)} orientations, kept {len(keep)} "
            f"(fits {fits_txt}; IoU {sc[keep[0]]['iou']:.2f})")
        return {"frame": a, "hyps": [{"P": distinct[i], **sc[i]} for i in keep], "refound": why != "anchor"}

    anchors = []
    for a in sorted(anchor_frames):
        r = register_anchor(a)
        if r is not None:
            anchors.append(r)
    if not anchors:
        raise SystemExit(f"{ep}: no usable anchor")
    log(f"{ep}: {len(anchors)} anchors at video frames {[a['frame'] + off for a in anchors]} "
        f"({time.time() - t0:.0f}s)")

    # ---- 2. bidirectional tracking between anchors, with orientation hand-over
    #      Forward pass in anchor order, backward pass in reverse. A tracking chain that arrives at the
    #      next anchor is CONTINUED from there (as an extra start pose) unless it matches one of that
    #      anchor's registered orientations (< --handover_deg). So a consistent orientation can run
    #      through anchors instead of being forced to switch to the anchor's own (possibly flipped) pick.
    cands: dict[int, list[dict]] = {f: [] for f in range(n)}
    for ai, a in enumerate(anchors):
        for hi, h in enumerate(a["hyps"]):
            cands[a["frame"]].append({"P": h["P"], **{k: h[k] for k in ("iou", "colour_err", "depth_cm", "fit")},
                                      "src": f"anchor {a['frame'] + off}#{hi}"})
    n_steps, t1 = 0, time.time()

    def run_chain(P0, seq, src):
        nonlocal n_steps
        tr.reset_to_pose(Transform3d.from_matrix(P0))
        P_prev = P0
        for f in seq:
            rgb, depth, mask = load(f)
            with to_debug(args.debug_log, f"track {f}"):
                P = tr.track_one(V2dImage(data=rgb), depth, intr, iteration=args.track_iter).to_matrix()
            s = score([P], rgb, depth, mask, f)[0]
            if s["iou"] < args.iou_ok and vis[f] >= args.vis_frac:
                tr.reset_to_pose(Transform3d.from_matrix(P_prev))
                with to_debug(args.debug_log, f"retrack {f}"):
                    P2 = tr.track_one(V2dImage(data=rgb), depth, intr, iteration=args.retrack_iter).to_matrix()
                s2 = score([P2], rgb, depth, mask, f)[0]
                if s2["fit"] > s["fit"]:
                    P, s = P2, s2
                tr.reset_to_pose(Transform3d.from_matrix(P))
            cands[f].append({"P": P, **s, "src": src})
            P_prev = P
            n_steps += 1
        return P_prev

    for direction in ("fwd", "bwd"):
        order = list(range(len(anchors))) if direction == "fwd" else list(range(len(anchors) - 1, -1, -1))
        carried: list[tuple[np.ndarray, str]] = []          # chains arriving at the current anchor
        for ai in order:
            a = anchors[ai]
            if direction == "fwd":
                nb = anchors[ai + 1]["frame"] if ai + 1 < len(anchors) else n
                seq = range(a["frame"] + 1, nb)
            else:
                nb = anchors[ai - 1]["frame"] if ai > 0 else -1
                seq = range(a["frame"] - 1, nb, -1)
            starts = [(h["P"], f"{direction} from {a['frame'] + off}#{hi}", False) for hi, h in enumerate(a["hyps"])]
            handed = 0
            for P_arr, src_arr in carried:
                if all(rot_deg(P_arr, P_s) >= args.handover_deg for P_s, _, _ in starts):
                    root = src_arr.split(" via ")[0]
                    starts.append((P_arr, f"{root} via {a['frame'] + off}", True))
                    handed += 1
            starts = starts[:len(a["hyps"]) + 2]
            carried = []
            for P0, src, is_handed in starts:
                # a handed-over chain also tracks the anchor frame itself, so it has a candidate there
                chain = ([a["frame"]] if is_handed else []) + list(seq)
                P_end = run_chain(P0, chain, src) if chain else P0
                carried.append((P_end, src))
            if len(seq) == 0:
                continue
            log(f"  {direction} from anchor {a['frame'] + off}: {len(seq)} frames x {len(starts)} start(s)"
                f"{f' ({handed} handed over from the previous anchor)' if handed else ''} — "
                f"{n_steps} tracked so far, {n_steps / (time.time() - t1):.1f} frames/s")

    # ---- 3. re-find a lost object: a visible stretch where NO candidate fits gets a fresh registration
    def lost_runs():
        bad = [vis[f] >= args.vis_frac and max(c["iou"] for c in cands[f]) < args.iou_ok for f in range(n)]
        runs, f = [], 0
        while f < n:
            if bad[f]:
                g = f
                while g + 1 < n and bad[g + 1]:
                    g += 1
                if g - f + 1 >= args.lost_after:
                    runs.append((f, g))
                f = g + 1
            else:
                f += 1
        return runs

    tried = set()
    for rnd in range(args.recover_rounds):
        runs = lost_runs()
        if not runs:
            break
        log(f"{ep}: re-find round {rnd + 1}: no candidate fits on {len(runs)} stretch(es): "
            + ", ".join(f"video frames {a + off}-{b + off}" for a, b in runs))
        for a0, b0 in runs:
            pool = [f for f in range(a0, b0 + 1) if eligible[f] and f not in tried] or \
                   [f for f in range(a0, b0 + 1) if vis[f] >= 0.3 and f not in tried]
            if not pool:
                log(f"  video frames {a0 + off}-{b0 + off}: no frame clear enough to re-find the object")
                continue
            f0 = max(pool, key=lambda f: quality[f])
            tried.add(f0)
            r = register_anchor(f0, why="re-find")
            if r is None:
                continue
            anchors.append(r)
            for hi, h in enumerate(r["hyps"]):
                cands[f0].append({"P": h["P"], **{k: h[k] for k in ("iou", "colour_err", "depth_cm", "fit")},
                                  "src": f"refind {f0 + off}#{hi}"})
                run_chain(h["P"], range(f0 + 1, min(n, b0 + 1 + args.lost_after)), f"refind fwd {f0 + off}#{hi}")
                run_chain(h["P"], range(f0 - 1, max(-1, a0 - 1 - args.lost_after), -1), f"refind bwd {f0 + off}#{hi}")
    left = lost_runs()
    if args.recover_rounds:
        log(f"{ep}: after re-finding: {sum(b - a + 1 for a, b in left)} visible frames still without a fitting "
            f"candidate" + (f" ({', '.join(f'{a + off}-{b + off}' for a, b in left)})" if left else ""))
    anchors.sort(key=lambda a: a["frame"])

    # ---- 4. global selection (Viterbi)
    w = np.clip(vis / 0.5, 0.05, 1.0)
    cost = [np.array([-w[0] * c["fit"] for c in cands[0]])]
    back = [None]
    for f in range(1, n):
        prev, cur = cands[f - 1], cands[f]
        T = np.zeros((len(prev), len(cur)))
        for i, p in enumerate(prev):
            for j, c in enumerate(cur):
                r = rot_deg(p["P"], c["P"])
                tcm = float(np.linalg.norm(p["P"][:3, 3] - c["P"][:3, 3]) * 100)
                T[i, j] = r / args.deg_per_unit + tcm / cm_unit + (args.flip_cost if r > 60 else 0.0)
        tot = cost[-1][:, None] + T
        back.append(tot.argmin(0))
        cost.append(tot.min(0) - w[f] * np.array([c["fit"] for c in cur]))
    path = [int(np.argmin(cost[-1]))]
    for f in range(n - 1, 0, -1):
        path.append(int(back[f][path[-1]]))
    path = path[::-1]

    # states: observed / tracked (within --lost_after frames of an observed frame) / lost / occluded
    chosen = [cands[f][path[f]] for f in range(n)]
    observed = np.array([vis[f] >= args.vis_frac and chosen[f]["iou"] >= args.iou_ok for f in range(n)])
    since = np.full(n, 10 ** 6)                       # frames to the nearest observed frame
    last = -10 ** 6
    for f in range(n):
        last = f if observed[f] else last
        since[f] = f - last
    last = 10 ** 6
    for f in range(n - 1, -1, -1):
        last = f if observed[f] else last
        since[f] = min(since[f], last - f)

    rows, P_prev, overrides, switches, last_src = [], None, 0, 0, None
    for f in range(n):
        c = chosen[f]
        P = c["P"]
        best_alt = max(cands[f], key=lambda x: x["fit"])
        if best_alt["fit"] > c["fit"] + 0.05:
            overrides += 1
        src_track = c["src"].split("#")[0] + "#" + c["src"].split("#")[1] if "#" in c["src"] else c["src"]
        if last_src is not None and src_track.split(" ")[-1] != last_src.split(" ")[-1]:
            switches += 1
        last_src = src_track
        state = ("occluded" if vis[f] < args.vis_frac else "observed" if observed[f]
                 else "tracked" if since[f] <= args.lost_after else "lost")
        rows.append({"frame": f, "video_frame": f + off, "state": state, "src": c["src"],
                     "n_candidates": len(cands[f]), "iou": c["iou"], "colour_err": c["colour_err"],
                     "depth_cm": c["depth_cm"], "fit": c["fit"], "best_alt_fit": best_alt["fit"],
                     "best_alt_src": best_alt["src"], "area_frac": float(vis[f]),
                     "quality": float(quality[f]), "occlusion": float(occ[f]), "mask_change": float(change[f]),
                     "rot_step": 0.0 if P_prev is None else rot_deg(P, P_prev),
                     "trans_step_cm": 0.0 if P_prev is None else float(np.linalg.norm(P[:3, 3] - P_prev[:3, 3]) * 100),
                     "frames_from_observed": int(min(since[f], 10 ** 6)),
                     "recovery": c["src"] if c["src"].startswith("refind") else None})
        Transform3d.from_matrix(P).save(os.path.join(poses_dir, f"{f:06d}.json"))
        P_prev = P
    json.dump({"episode": args.episode, "reference_frame": ref, "object": obj, "size_cm": size_cm, "thin": thin,
               "params": {**vars(args), "max_rot_step": 45.0, "cm_unit": cm_unit},
               "anchors": [{"frame": a["frame"], "video_frame": a["frame"] + off, "refound": a.get("refound", False),
                            "fits": [h["fit"] for h in a["hyps"]]} for a in anchors],
               "frames": rows}, open(os.path.join(track_dir, "states.json"), "w"), indent=1)
    ious = np.array([r["iou"] for r in rows])
    big = sum(r["rot_step"] > 45 for r in rows)
    states = {s: sum(r["state"] == s for r in rows) for s in ("observed", "tracked", "lost", "occluded")}
    run = longest = 0
    for r in rows:
        run = run + 1 if r["state"] == "lost" else 0
        longest = max(longest, run)
    jumps = sum(r["trans_step_cm"] > 0.5 * size_cm for r in rows)
    log(f"{ep}: stage 6 OK — {n} poses, {n_steps} tracking steps in {time.time() - t0:.0f}s; IoU median "
        f"{np.median(ious):.2f}, frames IoU<0.5: {(ious < 0.5).sum()}; {states}; longest lost stretch {longest}; "
        f"rotation steps > 45 deg: {big}; translation steps > half the object size: {jumps}; "
        f"global selection kept a worse-fitting candidate on {overrides} frames (continuity), source switches {switches}")


if __name__ == "__main__":
    main()
