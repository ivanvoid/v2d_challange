#!/usr/bin/env python3
"""Stage 5e — one mesh per physical object, chosen across all of its videos.

Each object appears in 3 videos (episodes_metadata.jsonl 'object'). Stage 5 built 3 candidate
meshes per video from that video's frames only; a mesh can fit the frames it was built from
and be wrong elsewhere (a pan without its handle, a deformed stool). Here every candidate of
the object is tested on frames of ALL its videos:

  frames     the stage-5c check frames of each video (5 per video, evenly taken from check.json)
  trial      one candidate at one relative size in one video: registered independently on every frame;
             per frame the best of FoundationPose's distinct orientations (>= 45 deg apart) by
             silhouette IoU. Colour is NOT used: a wrong texture must not decide the shape.
             trial score = (median IoU + median coverage) / 2 (coverage = share of the mask the mesh covers,
             so a missing part such as a handle counts)
  tested     each video's top --top_per_video (2) candidates by its own 5c check (6 of 9): outlines alone cannot
             see a wrong 3D shape (pan: a handle bent up into a hook tied the clean pan; 5c had it 3rd of 3)
  pass 1     every tested candidate, sizes --coarse (x0.9 / 1.0 / 1.1) in every video, so a good shape whose
             first size was a bit off is not dropped; candidate score = mean over videos of its best trial
  pass 2     the best --finalists: sizes --fine around each video's best pass-1 size (pass-1 trials reused)
  choice     the candidate with the best score after pass 2; each video keeps its own best size. Finalists
             within --tie (0.03, about the score's own noise) are reported as a near-tie and flagged on the
             review page (the outline score cannot see 3D shape: iron, a handle-less blob 0.90 vs the clean iron 0.87)
             (each video has its own depth scale; stage 8 maps everything to the body)
The winning trial's per-frame IoU, coverage, poses and render tiles are kept and used for every report and
for the tracking reference frame (nothing is refitted for the sheet).
Writes, for every video of the object:
  out/<ep>/mesh_scaled.obj (+ material files)   the chosen mesh at that video's size (the only writer)
  out/<ep>/mesh/object_choice.json              source candidate, winning trial, size, tracking ref frame
and logs/stages/05e_object/<object>.json + review sheet. stage5_validate.py then checks the exported mesh on
held-out frames and writes out/<ep>/mesh/validate.json, the final acceptance summary.

Run with the foundation_pose venv:
    /opt/venvs/foundation_pose/bin/python track1/stage5_object.py --object black_pan --out_root out \
        --weights weights --stage_dir logs/stages/05e_object
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logutil import log, to_debug  # noqa: E402
from viz import contrast_bg  # noqa: E402

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask  # noqa: E402
from v2d.common.datatypes import Image as V2dImage  # noqa: E402
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker, nvdiffrast_render  # noqa: E402
from v2d.mesh.lib.mesh import Mesh  # noqa: E402

import paths  # noqa: E402
META = str(paths.META)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TILE = 200                                       # review tile half-width (photo | render)


def rot_deg(a, b):
    c = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def scale_obj(src, dst, s):
    out = []
    for line in open(src):
        if line.startswith("v "):
            p = line.split()
            xyz = [float(x) * s for x in p[1:4]]
            out.append("v " + " ".join(f"{x:.6f}" for x in xyz) + ("" if len(p) <= 4 else " " + " ".join(p[4:])) + "\n")
        else:
            out.append(line)
    open(dst, "w").writelines(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--stage_dir", required=True)
    ap.add_argument("--frames_per_video", type=int, default=5)
    ap.add_argument("--coarse", type=float, nargs="+", default=[0.9, 1.0, 1.1],
                    help="pass 1: relative sizes tried for every candidate in every video")
    ap.add_argument("--fine", type=float, nargs="+", default=[0.9, 0.95, 1.05, 1.1],
                    help="pass 2: factors around each video's best pass-1 size, for the finalists")
    ap.add_argument("--finalists", type=int, default=2)
    ap.add_argument("--tie", type=float, default=0.03,
                    help="finalists within this of the best score are reported as a near-tie (flagged for review)")
    ap.add_argument("--top_per_video", type=int, default=2,
                    help="candidates tested per video: the best by that video's own 5c check")
    ap.add_argument("--n_hyp", type=int, default=5)
    args = ap.parse_args()

    eps = sorted(json.loads(l)["episode_index"] for l in open(META) if json.loads(l)["object"] == args.object)
    os.makedirs(args.stage_dir, exist_ok=True)
    dbg = os.path.join(args.stage_dir, f"{args.object}.debug.log")
    t0 = time.time()

    # ---- frames of every video
    videos = {}
    for e in eps:
        W = os.path.join(os.path.abspath(args.out_root), f"episode_{e:06d}")
        choice = os.path.join(W, "mesh", "choice.json")
        if not os.path.exists(choice):
            log(f"  episode {e}: no stage-5 result, skipped")
            continue
        c = json.load(open(choice))
        chk = json.load(open(os.path.join(W, "mesh", f"cand_{c['chosen']:06d}", "check.json")))
        fr = [p["frame"] for p in chk["frames"] if not p["own"]] or [p["frame"] for p in chk["frames"]]
        fr = [fr[int(i)] for i in np.linspace(0, len(fr) - 1, min(args.frames_per_video, len(fr))).round()]
        intr = CameraIntrinsics.load(os.path.join(W, "intrinsics_stable.json"))
        data = []
        for f in fr:
            m = Mask.load(os.path.join(W, "masks", "1", f"{f:06d}.png"))
            if not m.mask.astype(bool).any():
                continue
            data.append({"frame": f, "rgb": np.asarray(Image.open(os.path.join(W, "frames", f"{f:06d}.png")).convert("RGB")),
                         "depth": DepthImage.load(os.path.join(W, "depth", f"{f:06d}.png")), "mask": m})
        # only this video's best --top_per_video by its own 5c check: a mesh can fit outlines everywhere and still
        # have a wrong 3D shape (pan: a handle bent up into a hook tied the clean pan at 0.87; 5c had it 3rd of 3)
        ranked = sorted(c["candidates"], key=lambda x: -x["score"])
        cands = [os.path.join(W, "mesh", f"cand_{x['candidate']:06d}") for x in ranked[:args.top_per_video]
                 if os.path.exists(os.path.join(W, "mesh", f"cand_{x['candidate']:06d}", "mesh_scaled.obj"))]
        for x in ranked[args.top_per_video:]:
            log(f"  episode {e}: cand_{x['candidate']:06d} (video frame {x['video_frame']}) not tested — 5c rank "
                f"{ranked.index(x) + 1} of {len(ranked)} in its own video (score {x['score']:.2f})")
        off = json.load(open(os.path.join(W, "track1_meta.json")))["frame_offset"]
        videos[e] = {"W": W, "intr": intr, "data": data, "cands": cands, "off": off}
    cand_list = [(e, cd) for e, v in videos.items() for cd in v["cands"]]
    log(f"{args.object}: videos {list(videos)}, {len(cand_list)} candidate meshes, "
        f"{sum(len(v['data']) for v in videos.values())} test frames")

    def eval_frame(tr, v, d):
        """Register; best distinct orientation by IoU. Returns (iou, coverage, pose, tile)."""
        with to_debug(dbg, "register"):
            tr.register(V2dImage(data=d["rgb"]), d["depth"], d["mask"], v["intr"], iteration=5)
        est = tr._est
        distinct = []
        for h in est.poses[:40].cpu().numpy():
            if all(rot_deg(h, g) >= 45 for g in distinct):
                distinct.append(h)
            if len(distinct) == args.n_hyp:
                break
        K = v["intr"].to_matrix()
        with torch.no_grad():
            col, dep, _ = nvdiffrast_render(K, v["intr"].height, v["intr"].width,
                                            torch.as_tensor(np.stack(distinct), device="cuda", dtype=torch.float),
                                            glctx=tr._glctx, mesh_tensors=est.mesh_tensors, get_normal=False)
        dep = dep.cpu().numpy() > 0.001
        m = d["mask"].mask.astype(bool)
        ious = [(dep[i] & m).sum() / max((dep[i] | m).sum(), 1) for i in range(len(distinct))]
        b = int(np.argmax(ious))
        cov = float((dep[b] & m).sum() / max(m.sum(), 1))            # share of the mask the mesh covers
        return float(ious[b]), cov, distinct[b], make_tile(d, col[b].cpu().numpy(), dep[b], m)

    def make_tile(d, col, sil, m):
        """Photo crop with the mask (green) and model (blue) outlines | the textured render on a contrasting bg."""
        img = d["rgb"].copy()
        for mm, c in ((m, (40, 230, 60)), (sil, (60, 120, 255))):
            cnts, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, c, 2)
        rc = np.clip(col * (255 if col.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
        rc[~sil] = contrast_bg(rc[sil])
        ys, xs = np.nonzero(m | sil)
        pad = int(0.3 * max(np.ptp(xs), np.ptp(ys))) + 20
        y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
        t = Image.new("RGB", (2 * TILE, TILE))
        t.paste(Image.fromarray(img[y0:y1, x0:x1]).resize((TILE, TILE)), (0, 0))
        t.paste(Image.fromarray(rc[y0:y1, x0:x1]).resize((TILE, TILE)), (TILE, 0))
        return t

    trials = {}             # (candidate path, episode, rel) -> trial; kept so pass 2 reuses pass 1

    def run_trials(path, rels_per_video):
        """Every (video, size) of one candidate that was not tried yet. One tracker load per candidate."""
        todo = [(e, r) for e, rels in rels_per_video.items() for r in rels if (path, e, round(r, 4)) not in trials]
        if not todo:
            return
        with to_debug(dbg, f"load {path}"):
            tr = FoundationPoseTracker(Mesh.load(os.path.join(path, "mesh_scaled.obj")),
                                       os.path.join(os.path.abspath(args.weights), "foundation_pose"))
        for e, rel in todo:
            v = videos[e]
            if abs(getattr(tr, "_current_scale", 1.0) - rel) > 1e-6:
                with to_debug(dbg, f"rescale {rel}"):
                    tr.rescale_to(rel)
            res = [eval_frame(tr, v, d) for d in v["data"]]
            ious, covs = [r[0] for r in res], [r[1] for r in res]
            t = {"rel": rel, "ious": ious, "coverages": covs, "poses": [r[2].tolist() for r in res],
                 "tiles": [r[3] for r in res], "frames": [d["frame"] for d in v["data"]],
                 "median_iou": float(np.median(ious)) if res else 0.0,
                 "median_coverage": float(np.median(covs)) if res else 0.0}
            t["score"] = (t["median_iou"] + t["median_coverage"]) / 2
            trials[(path, e, round(rel, 4))] = t
        del tr
        torch.cuda.empty_cache()

    def best_trials(path):
        """Per video, the best trial of this candidate so far."""
        out = {}
        for (p, e, _), t in trials.items():
            if p == path and (e not in out or t["score"] > out[e]["score"]):
                out[e] = t
        return out

    def summary(c):
        bt = best_trials(c["path"])
        c["score"] = float(np.mean([t["score"] for t in bt.values()]))
        c["per_video"] = {str(e): {"rel": t["rel"], "score": t["score"], "median_iou": t["median_iou"],
                                   "median_coverage": t["median_coverage"],
                                   "tried": {f"{r:.3f}": round(tt["score"], 4) for (p, ee, r), tt in sorted(trials.items())
                                             if p == c["path"] and ee == e}}
                          for e, t in bt.items()}
        return c

    def fmt(c):
        return ", ".join(f"ep{k} x{p['rel']:.2f} IoU {p['median_iou']:.2f} cov {p['median_coverage']:.2f}"
                         for k, p in c["per_video"].items()) + f" -> score {c['score']:.2f}"

    # ---- pass 1: every candidate, coarse sizes, every video
    cands = []
    for e, path in cand_list:
        run_trials(path, {ev: args.coarse for ev in videos})
        c = summary({"source_episode": e, "candidate": os.path.basename(path), "path": path,
                     "video_frame": int(os.path.basename(path)[5:]) + videos[e]["off"]})
        cands.append(c)
        log(f"  pass 1 {c['candidate']} (video frame {c['video_frame']}) of ep {e}: {fmt(c)}")

    # ---- pass 2: finalists, finer sizes around each video's best
    finalists = sorted(cands, key=lambda c: -c["score"])[:args.finalists]
    for c in finalists:
        run_trials(c["path"], {int(k): sorted({round(p["rel"] * f, 4) for f in args.fine})
                               for k, p in c["per_video"].items()})
        summary(c)
        c["finalist"] = True
        log(f"  pass 2 {c['candidate']} (video frame {c['video_frame']}) of ep {c['source_episode']}: {fmt(c)}")
    # Near-ties are recorded, not resolved by a hidden rule: the outline score's own noise is about 0.02 (the same
    # mesh at x0.99 vs x1.00) and it cannot see 3D shape (iron: a handle-less blob 0.90 vs the clean iron 0.87).
    # A 3D-agreement tie-break (shape_consensus.py) was tried and dropped: its own noise (+-0.005) was as large
    # as the differences it would decide. The review page flags these objects.
    win = max(finalists, key=lambda c: c["score"])
    near_tie = [c for c in finalists if c is not win and c["score"] >= win["score"] - args.tie]
    if near_tie:
        log(f"  NEAR-TIE (within {args.tie}): chosen {win['candidate']} (video frame {win['video_frame']}) of ep "
            f"{win['source_episode']} {win['score']:.3f} vs " + ", ".join(
                f"{c['candidate']} (video frame {c['video_frame']}) of ep {c['source_episode']} {c['score']:.3f}"
                for c in near_tie) + " — the outline score cannot separate these; check the gallery")
    win_trial = best_trials(win["path"])

    # ---- write the chosen mesh to every video (stage 5e is the only writer of out/<ep>/mesh_scaled.obj)
    src_dir = win["path"]
    for e, v in videos.items():
        W, t = v["W"], win_trial[e]
        scale_obj(os.path.join(src_dir, "mesh_scaled.obj"), os.path.join(W, "mesh_scaled.obj"), t["rel"])
        for name in os.listdir(src_dir):
            if name.startswith("material") and name.endswith((".mtl", ".png")):
                shutil.copy(os.path.join(src_dir, name), os.path.join(W, name))
        vv = np.array([[float(x) for x in l.split()[1:4]] for l in open(os.path.join(W, "mesh_scaled.obj"))
                       if l.startswith("v ")])
        ref = t["frames"][int(np.argmax(t["ious"]))] if t["frames"] else None   # from the WINNING trial
        json.dump({"object": args.object, "source_episode": win["source_episode"], "source_candidate": win["candidate"], "source_video_frame": win["video_frame"],
                   "rel_scale": t["rel"], "extent_m": float(np.ptp(vv, axis=0).max()),
                   "score_iou_coverage": t["score"], "median_iou": t["median_iou"],
                   "median_coverage": t["median_coverage"], "track_ref_frame": ref,
                   "own_mesh": win["source_episode"] == e,
                   "frames": [{"frame": f, "video_frame": f + v["off"], "iou": i, "coverage": cv, "pose": po}
                              for f, i, cv, po in zip(t["frames"], t["ious"], t["coverages"], t["poses"])]},
                  open(os.path.join(W, "mesh", "object_choice.json"), "w"), indent=1)
    rep = {"object": args.object, "videos": list(videos), "coarse": args.coarse, "fine": args.fine,
           "candidates": [{k: v for k, v in c.items() if k != "path"} for c in cands],
           "tie": args.tie, "near_tie": [{"source_episode": c["source_episode"], "candidate": c["candidate"],
                                          "video_frame": c["video_frame"], "score": c["score"]} for c in near_tie],
           "chosen": {"source_episode": win["source_episode"], "candidate": win["candidate"],
                      "rel_per_video": {str(e): t["rel"] for e, t in win_trial.items()}, "score": win["score"]},
           "seconds": round(time.time() - t0)}
    json.dump(rep, open(os.path.join(args.stage_dir, f"{args.object}.json"), "w"), indent=1)

    # ---- review sheet: the winning trial's tiles (no refitting)
    font, small = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 15)
    rows = []
    for e, v in videos.items():
        t = win_trial[e]
        tiles = []
        for f, iou, cv, tile in zip(t["frames"], t["ious"], t["coverages"], t["tiles"]):
            tt = Image.new("RGB", (2 * TILE + 10, TILE + 26), (20, 20, 20))
            tt.paste(tile, (5, 26))
            ImageDraw.Draw(tt).text((5, 4), f"frame {f + v['off']}  IoU {iou:.2f}  cov {cv:.2f}", fill=(255, 255, 255),
                                    font=small)
            tiles.append(tt)
        w = (2 * TILE + 10)
        row = Image.new("RGB", (w * max(1, len(tiles)) + 230, TILE + 26), (0, 0, 0))
        ImageDraw.Draw(row).text((8, 10), f"episode {e}\nsize x{t['rel']:.2f}\nscore {t['score']:.2f}\n"
                                          f"(IoU {t['median_iou']:.2f}\n+ cov {t['median_coverage']:.2f}) / 2",
                                 fill=(255, 220, 0), font=font)
        for i, tt in enumerate(tiles):
            row.paste(tt, (230 + w * i, 0))
        rows.append(row)
    head = 90 + 24 * len(cands)
    sheet = Image.new("RGB", (max(max(r.width for r in rows), 1600), head + sum(r.height + 10 for r in rows)), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 10), f"{args.object}: chosen mesh {win['candidate']} (video frame {win['video_frame']}) from episode "
                      f"{win['source_episode']} - per tile: "
                      f"photo (green = SAM2 mask, blue = model) | textured render", fill=(255, 255, 255), font=font)
    dr.text((10, 45), "score = mean over videos of the best trial's (median IoU + median coverage) / 2; "
                      "* = finalist (finer sizes tried)", fill=(170, 170, 170), font=small)
    for i, c in enumerate(sorted(cands, key=lambda c: -c["score"])):
        mark = "  <== chosen" if c["path"] == win["path"] else ""
        dr.text((10, 70 + 24 * i), f"{'*' if c.get('finalist') else ' '} {c['candidate']} (video frame "
                f"{c['video_frame']}) of ep {c['source_episode']}: "
                f"{fmt(c)}{mark}", fill=(120, 255, 120) if mark else (220, 220, 220), font=small)
    y = head
    for r in rows:
        sheet.paste(r, (0, y)); y += r.height + 10
    os.makedirs(os.path.join(args.stage_dir, "review"), exist_ok=True)
    sheet.save(os.path.join(args.stage_dir, "review", f"{args.object}.png"))
    log(f"{args.object}: stage 5e OK — mesh {win['candidate']} (video frame {win['video_frame']}) of episode "
        f"{win['source_episode']} for videos "
        f"{list(videos)} (score {win['score']:.2f}; sizes " +
        ", ".join(f"ep{e} x{t['rel']:.2f}" for e, t in win_trial.items()) + f") in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
