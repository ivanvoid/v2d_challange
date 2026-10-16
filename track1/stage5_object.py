#!/usr/bin/env python3
"""Stage 5e — one mesh per physical object, chosen across all of its videos.

Each object appears in 3 videos (episodes_metadata.jsonl 'object'). Stage 5 built 2 candidate
meshes per video from that video's frames only; a mesh can fit the frames it was built from
and be wrong elsewhere (a pan without its handle, a deformed stool). Here every candidate of
the object is tested on frames of ALL its videos:

  frames     the stage-5 check frames of each video (5 per video, evenly taken from check.json)
  pass 1     every candidate, at its own metric size, registered independently on every frame;
             per frame the best of FoundationPose's distinct orientations (>= 45 deg apart) by
             silhouette IoU. Colour is NOT used here: a wrong texture must not decide the shape.
             score = mean over videos of the median IoU in that video (a mesh must fit all three)
  pass 2     the best 2 candidates: per video, relative scale in --rels; per video the best scale
             is kept (each video has its own depth scale; stage 8 maps everything to the body)
  choice     the candidate with the best score after pass 2
Writes, for every video of the object:
  out/<ep>/mesh_scaled.obj (+ material files)   the chosen mesh at that video's scale
  out/<ep>/mesh_scaled_stage5d.obj              the previous per-video mesh (kept)
  out/<ep>/mesh/object_choice.json              source candidate, scores, scale, tracking ref frame
and logs/stages/05e_object/<object>.json + review sheet.

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

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask  # noqa: E402
from v2d.common.datatypes import Image as V2dImage  # noqa: E402
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker, nvdiffrast_render  # noqa: E402
from v2d.mesh.lib.mesh import Mesh  # noqa: E402

import paths  # noqa: E402
META = str(paths.META)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


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
    ap.add_argument("--rels", type=float, nargs="+", default=[0.8, 0.9, 1.0, 1.1, 1.25])
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
        cands = [os.path.join(W, "mesh", d) for d in sorted(os.listdir(os.path.join(W, "mesh")))
                 if d.startswith("cand_") and os.path.exists(os.path.join(W, "mesh", d, "mesh_scaled.obj"))]
        off = json.load(open(os.path.join(W, "track1_meta.json")))["frame_offset"]
        videos[e] = {"W": W, "intr": intr, "data": data, "cands": cands, "off": off}
    cand_list = [(e, cd) for e, v in videos.items() for cd in v["cands"]]
    log(f"{args.object}: videos {list(videos)}, {len(cand_list)} candidate meshes, "
        f"{sum(len(v['data']) for v in videos.values())} test frames")

    def eval_frame(tr, v, d):
        """Register; best distinct orientation by IoU. Returns (iou, pose_centred, colour_err)."""
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
        return float(ious[b]), distinct[b], col[b].cpu().numpy(), dep[b]

    def score_cand(path, rel_per_video=None, keep_render=False):
        with to_debug(dbg, f"load {path}"):
            tr = FoundationPoseTracker(Mesh.load(os.path.join(path, "mesh_scaled.obj")),
                                       os.path.join(os.path.abspath(args.weights), "foundation_pose"))
        per = {}
        for e, v in videos.items():
            rel = (rel_per_video or {}).get(e, 1.0)
            if rel != 1.0:
                with to_debug(dbg, f"rescale {rel}"):
                    tr.rescale_to(rel)
            res = [eval_frame(tr, v, d) for d in v["data"]]
            per[e] = {"ious": [r[0] for r in res], "median": float(np.median([r[0] for r in res])) if res else 0.0}
            if keep_render:
                per[e]["renders"] = [(r[2], r[3]) for r in res]
            if rel != 1.0:
                with to_debug(dbg, "rescale back"):
                    tr.rescale_to(1.0)
        del tr
        torch.cuda.empty_cache()
        return per

    # ---- pass 1: every candidate at its own size on every video
    pass1 = []
    for e, path in cand_list:
        per = score_cand(path)
        s = float(np.mean([p["median"] for p in per.values()]))
        pass1.append({"source_episode": e, "candidate": os.path.basename(path), "path": path, "score": s,
                      "per_video": {str(k): {"median_iou": p["median"], "ious": p["ious"]} for k, p in per.items()}})
        log(f"  pass 1 {os.path.basename(path)} of ep {e}: " + ", ".join(
            f"ep{k} {p['median']:.2f}" for k, p in per.items()) + f" -> score {s:.2f}")

    # ---- pass 2: best 2, per-video relative scale
    pass2 = []
    for c in sorted(pass1, key=lambda c: -c["score"])[:2]:
        best_rel, best_med = {}, {}
        for rel in args.rels:
            per = score_cand(c["path"], {e: rel for e in videos})
            for e, p in per.items():
                if e not in best_med or p["median"] > best_med[e]:
                    best_med[e], best_rel[e] = p["median"], rel
        s = float(np.mean(list(best_med.values())))
        pass2.append({**c, "rel_per_video": {str(k): v for k, v in best_rel.items()},
                      "median_per_video": {str(k): v for k, v in best_med.items()}, "score2": s})
        log(f"  pass 2 {c['candidate']} of ep {c['source_episode']}: " + ", ".join(
            f"ep{k} x{best_rel[k]:.2f} IoU {best_med[k]:.2f}" for k in best_med) + f" -> score {s:.2f}")
    win = max(pass2, key=lambda c: c["score2"])

    # ---- write the chosen mesh to every video
    src_dir = win["path"]
    src_scale = json.load(open(os.path.join(src_dir, "scale.json")))["scale"]
    for e, v in videos.items():
        W, rel = v["W"], win["rel_per_video"][str(e)]
        keep = os.path.join(W, "mesh_scaled_stage5d.obj")
        if not os.path.exists(keep) and os.path.exists(os.path.join(W, "mesh_scaled.obj")):
            shutil.copy(os.path.join(W, "mesh_scaled.obj"), keep)
        scale_obj(os.path.join(src_dir, "mesh_scaled.obj"), os.path.join(W, "mesh_scaled.obj"), rel)
        for name in os.listdir(src_dir):
            if name.startswith("material") and name.endswith((".mtl", ".png")):
                shutil.copy(os.path.join(src_dir, name), os.path.join(W, name))
        vv = np.array([[float(x) for x in l.split()[1:4]] for l in open(os.path.join(W, "mesh_scaled.obj"))
                       if l.startswith("v ")])
        ious = win["per_video"][str(e)]["ious"]
        ref = v["data"][int(np.argmax(ious))]["frame"] if v["data"] else None
        json.dump({"object": args.object, "source_episode": win["source_episode"], "source_candidate": win["candidate"],
                   "rel_scale": rel, "extent_m": float(np.ptp(vv, axis=0).max()),
                   "median_iou": win["median_per_video"][str(e)], "track_ref_frame": ref,
                   "own_mesh": win["source_episode"] == e},
                  open(os.path.join(W, "mesh", "object_choice.json"), "w"), indent=1)
    rep = {"object": args.object, "videos": list(videos), "pass1": [{k: v for k, v in c.items() if k != "path"}
                                                                   for c in pass1],
           "pass2": [{k: v for k, v in c.items() if k != "path"} for c in pass2],
           "chosen": {"source_episode": win["source_episode"], "candidate": win["candidate"],
                      "rel_per_video": win["rel_per_video"], "score": win["score2"]},
           "seconds": round(time.time() - t0)}
    json.dump(rep, open(os.path.join(args.stage_dir, f"{args.object}.json"), "w"), indent=1)

    # ---- review sheet: chosen mesh on every test frame of every video (best orientation by IoU)
    per = score_cand(src_dir, {e: win["rel_per_video"][str(e)] for e in videos}, keep_render=True)
    font, small = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 15)
    rows = []
    for e, v in videos.items():
        tiles = []
        for d, iou, (col, sil) in zip(v["data"], per[e]["ious"], per[e]["renders"]):
            img = d["rgb"].copy()
            img[sil] = (0.3 * img[sil] + 0.7 * col[sil] * 255).astype(np.uint8)
            m = d["mask"].mask.astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
            ys, xs = np.nonzero(m | sil)
            pad = int(0.8 * max(np.ptp(xs), np.ptp(ys))) + 20
            crop = Image.fromarray(img[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad])
            crop = crop.resize((260, max(1, int(260 * crop.height / crop.width))))
            t = Image.new("RGB", (270, crop.height + 26), (20, 20, 20))
            t.paste(crop, (5, 26))
            ImageDraw.Draw(t).text((5, 4), f"frame {d['frame'] + v['off']}  IoU {iou:.2f}", fill=(255, 255, 255),
                                   font=small)
            tiles.append(t)
        h = max(t.height for t in tiles) if tiles else 50
        row = Image.new("RGB", (270 * max(1, len(tiles)) + 220, h), (0, 0, 0))
        ImageDraw.Draw(row).text((8, 10), f"episode {e}\nx{win['rel_per_video'][str(e)]:.2f}\nmedian IoU\n"
                                          f"{win['median_per_video'][str(e)]:.2f}", fill=(255, 220, 0), font=font)
        for i, t in enumerate(tiles):
            row.paste(t, (220 + 270 * i, 0))
        rows.append(row)
    head = 60 + 26 * len(pass1)
    sheet = Image.new("RGB", (max(r.width for r in rows), head + sum(r.height + 10 for r in rows)), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 10), f"{args.object}: chosen mesh {win['candidate']} from episode {win['source_episode']} "
                      f"(textured, green = SAM2 mask)", fill=(255, 255, 255), font=font)
    for i, c in enumerate(sorted(pass1, key=lambda c: -c["score"])):
        mark = "  <== chosen" if c["path"] == win["path"] else ""
        dr.text((10, 50 + 26 * i), f"{c['candidate']} of ep {c['source_episode']}: score {c['score']:.2f} (" +
                ", ".join(f"ep{k} {p['median_iou']:.2f}" for k, p in c["per_video"].items()) + ")" + mark,
                fill=(120, 255, 120) if mark else (220, 220, 220), font=small)
    y = head
    for r in rows:
        sheet.paste(r, (0, y)); y += r.height + 10
    os.makedirs(os.path.join(args.stage_dir, "review"), exist_ok=True)
    sheet.save(os.path.join(args.stage_dir, "review", f"{args.object}.png"))
    log(f"{args.object}: stage 5e OK — mesh {win['candidate']} of episode {win['source_episode']} for videos "
        f"{list(videos)} (score {win['score2']:.2f}) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
