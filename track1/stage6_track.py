#!/usr/bin/env python3
"""Stage 6 — object pose for every frame: multi-anchor bidirectional tracking + global selection.

1. anchors   smart-sampled where the object is clearly visible and still:
               quality q = visible (mask area / median, capped at 1) * (1 - body occlusion)^2
                           * sharpness^0.3 (Laplacian variance in the object box, relative)
                           * stillness (exp(-mask_change / 0.1))
               eligible  = mask area >= --min_anchor_vis x median, not touching the image border,
                           body occlusion <= 0.25
             spacing follows the motion: the timeline is cut into stretches whose summed mask
             change (1 - IoU of consecutive masks) reaches --motion_budget (stretch length kept in
             [--min_gap, --max_gap]); each stretch's best eligible frame is an anchor, plus the
             stage-5 reference frame. Static stretches -> few anchors, fast motion -> many.
             Each anchor is registered independently; distinct orientations (>= 45 deg apart)
             within --hyp_margin of the best fit are kept (max --max_hyps).
2. tracking  every anchor hypothesis is tracked FORWARD to the next anchor and BACKWARD to the
             previous one (first/last anchors also run to the video ends). So most frames get
             candidates from both sides; nothing is tracked further than one anchor gap.
             Low-fit frames are re-tracked once with more refiner iterations (5 -> 10).
3. selection Viterbi over all frames: choose one candidate per frame minimising
                 sum_f  -w_f * fit_f(candidate)                      (w_f = visibility weight)
               + sum_f  rot_change/--deg_per_unit + trans_change_cm/--cm_per_unit (+ --flip_cost if > 60 deg)
             i.e. the best-fitting sequence that does not jump or flip.
             Continuous smoothing over the whole sequence (pose graph) is stage 7.
Writes out/episode_XXXXXX/poses/<frame:06d>.json (object-to-camera, original mesh frame) and
track/states.json (per frame: chosen source, fit, best alternative, steps; anchors).

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
    ap.add_argument("--motion_budget", type=float, default=2.0, help="summed mask change per anchor stretch")
    ap.add_argument("--min_gap", type=int, default=15)
    ap.add_argument("--max_gap", type=int, default=120)
    ap.add_argument("--max_anchors", type=int, default=16)
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
    ap.add_argument("--cm_per_unit", type=float, default=10.0, help="translation change costing one unit of fit")
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

    # ---- per-frame quality: visibility, border, body occlusion, sharpness, stillness
    import cv2
    body = torch.load(os.path.join(W, "body", "mhr_mesh.pt"), map_location="cpu", weights_only=False)
    bv, bf = body["pred_vertices"].numpy(), body["faces"].numpy()
    areas, border = np.zeros(n), np.zeros(n, bool)
    occ, sharp, change = np.zeros(n), np.zeros(n), np.zeros(n)
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
            front = (sil > 0) & reg & (d < float(np.median(d[m])) - 0.05) & ~m
            occ[f] = front.sum() / (m.sum() + front.sum())
        else:
            occ[f] = -1
    for f in range(n):                                     # odd frames: copy the previous even frame
        if occ[f] < 0:
            occ[f] = occ[f - 1]
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

    def score(P_list, rgb, depth, mask):
        Pc = np.stack([P @ tfc_inv for P in P_list])
        with torch.no_grad():
            col, dep, _ = nvdiffrast_render(K, intr.height, intr.width,
                                            torch.as_tensor(Pc, device="cuda", dtype=torch.float),
                                            glctx=tr._glctx, mesh_tensors=est.mesh_tensors, get_normal=False)
        col, dep = col.cpu().numpy(), dep.cpu().numpy()
        m, d = mask.mask.astype(bool), depth.depth
        out = []
        for i in range(len(P_list)):
            sil = dep[i] > 0.001
            both = sil & m
            iou = float(both.sum() / max((sil | m).sum(), 1))
            ce = float(np.abs(col[i][both] * 255 - rgb[both]).mean()) if both.any() else 255.0
            dcm = float(np.median(np.abs(dep[i][both] - d[both])) * 100) if (both & (d > 0)).any() else None
            out.append({"iou": iou, "colour_err": ce, "depth_cm": dcm, "fit": iou - ce / 255.0})
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
    anchors = []
    for a in sorted(anchor_frames):
        rgb, depth, mask = load(a)
        with to_debug(args.debug_log, f"register anchor {a}"):
            tr.register(V2dImage(data=rgb), depth, mask, intr, iteration=args.register_iter)
        distinct = []
        for h in est.poses[:40].cpu().numpy():
            P = h @ tfc
            if all(rot_deg(P, g) >= 45 for g in distinct):
                distinct.append(P)
            if len(distinct) == 6:
                break
        sc = score(distinct, rgb, depth, mask)
        best = max(s_["fit"] for s_ in sc)
        keep = sorted([i for i in range(len(distinct)) if sc[i]["fit"] >= best - args.hyp_margin],
                      key=lambda i: -sc[i]["fit"])[:args.max_hyps]
        if sc[keep[0]]["iou"] < 0.4:
            log(f"  anchor video frame {a + off}: best IoU {sc[keep[0]]['iou']:.2f} too low, dropped")
            continue
        anchors.append({"frame": a, "hyps": [{"P": distinct[i], **sc[i]} for i in keep]})
        fits_txt = ", ".join("%.2f" % sc[i]["fit"] for i in keep)
        log(f"  anchor video frame {a + off}: quality {quality[a]:.2f} (visible {vis[a]:.2f}, occluded "
            f"{occ[a]:.2f}, mask change {change[a]:.3f}); {len(distinct)} orientations, kept {len(keep)} "
            f"(fits {fits_txt}; IoU {sc[keep[0]]['iou']:.2f})")
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
            s = score([P], rgb, depth, mask)[0]
            if s["iou"] < args.iou_ok and vis[f] >= args.vis_frac:
                tr.reset_to_pose(Transform3d.from_matrix(P_prev))
                with to_debug(args.debug_log, f"retrack {f}"):
                    P2 = tr.track_one(V2dImage(data=rgb), depth, intr, iteration=args.retrack_iter).to_matrix()
                s2 = score([P2], rgb, depth, mask)[0]
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

    # ---- 3. global selection (Viterbi)
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
                T[i, j] = r / args.deg_per_unit + tcm / args.cm_per_unit + (args.flip_cost if r > 60 else 0.0)
        tot = cost[-1][:, None] + T
        back.append(tot.argmin(0))
        cost.append(tot.min(0) - w[f] * np.array([c["fit"] for c in cur]))
    path = [int(np.argmin(cost[-1]))]
    for f in range(n - 1, 0, -1):
        path.append(int(back[f][path[-1]]))
    path = path[::-1]

    rows, P_prev, overrides, switches, last_src = [], None, 0, 0, None
    for f in range(n):
        c = cands[f][path[f]]
        P = c["P"]
        best_alt = max(cands[f], key=lambda x: x["fit"])
        if best_alt["fit"] > c["fit"] + 0.05:
            overrides += 1
        src_track = c["src"].split("#")[0] + "#" + c["src"].split("#")[1] if "#" in c["src"] else c["src"]
        if last_src is not None and src_track.split(" ")[-1] != last_src.split(" ")[-1]:
            switches += 1
        last_src = src_track
        state = "occluded" if vis[f] < args.vis_frac else ("observed" if c["iou"] >= args.iou_ok else "uncertain")
        rows.append({"frame": f, "video_frame": f + off, "state": state, "src": c["src"],
                     "n_candidates": len(cands[f]), "iou": c["iou"], "colour_err": c["colour_err"],
                     "depth_cm": c["depth_cm"], "fit": c["fit"], "best_alt_fit": best_alt["fit"],
                     "best_alt_src": best_alt["src"], "area_frac": float(vis[f]),
                     "quality": float(quality[f]), "occlusion": float(occ[f]), "mask_change": float(change[f]),
                     "rot_step": 0.0 if P_prev is None else rot_deg(P, P_prev),
                     "trans_step_cm": 0.0 if P_prev is None else float(np.linalg.norm(P[:3, 3] - P_prev[:3, 3]) * 100),
                     "recovery": None})
        Transform3d.from_matrix(P).save(os.path.join(poses_dir, f"{f:06d}.json"))
        P_prev = P
    json.dump({"episode": args.episode, "reference_frame": ref, "params": {**vars(args), "max_rot_step": 45.0},
               "anchors": [{"frame": a["frame"], "video_frame": a["frame"] + off,
                            "fits": [h["fit"] for h in a["hyps"]]} for a in anchors],
               "frames": rows}, open(os.path.join(track_dir, "states.json"), "w"), indent=1)
    ious = np.array([r["iou"] for r in rows])
    big = sum(r["rot_step"] > 45 for r in rows)
    states = {s: sum(r["state"] == s for r in rows) for s in ("observed", "uncertain", "occluded")}
    log(f"{ep}: stage 6 OK — {n} poses, {n_steps} tracking steps in {time.time() - t0:.0f}s; IoU median "
        f"{np.median(ious):.2f}, frames IoU<0.5: {(ious < 0.5).sum()}; {states}; rotation steps > 45 deg: {big}; "
        f"global selection kept a worse-fitting candidate on {overrides} frames (continuity), source switches {switches}")


if __name__ == "__main__":
    main()
