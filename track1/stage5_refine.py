#!/usr/bin/env python3
"""Stage 5d — refine the chosen mesh's scale over many frames, and audit orientation.

Why: stage 5b fits the scale on ONE frame (the one the mesh was built from); SAM3D often guesses
the unseen depth of an object wrong, so the mesh can fit that frame and be too big from every
other viewpoint. Stage 5c also registers each check frame independently, and FoundationPose can
pick a 180-degree flipped orientation when the silhouette is nearly symmetric (an iron standing
on its heel).

What it does, with the chosen mesh (out/episode_XXXXXX/mesh_scaled.obj, textured):
  1. scale: for relative scales --rels (default 0.7 0.8 0.9 1.0 1.1), register the mesh on every
     check frame (stage-5c frames + its own) and measure each frame's best orientation
  2. orientation: per frame, FoundationPose's top hypotheses are reduced to distinct orientations
     (>= 45 deg apart); each is rendered with texture and scored
        fit = IoU - colour_error / 255        (colour error = mean |render - image| inside the overlap)
     "FP pick flipped" is recorded when the best distinct orientation is > 90 deg from FP's pick
  3. keeps the scale with the best median fit; rewrites out/episode_XXXXXX/mesh_scaled.obj
     (vertex lines scaled, texture kept; the previous file is kept as mesh_scaled_stage5c.obj)
Writes mesh/refine.json and the review sheet --sheet (per frame: image | FP pick | best orientation).

Run with the foundation_pose venv (needs ~10 GB GPU):
    /opt/venvs/foundation_pose/bin/python track1/stage5_refine.py --episode 6 --out_root out --weights weights \
        --debug_log logs/stages/05_mesh/episode_000006.debug.log \
        --sheet logs/stages/05_mesh/review/episode_000006_refine.png
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

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def rot_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def scale_obj(src: str, dst: str, s: float) -> None:
    """Scale the 'v' lines of an OBJ about the origin; keep everything else (uv, faces, material)."""
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
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--debug_log", required=True)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--rels", type=float, nargs="+", default=[0.7, 0.8, 0.9, 1.0, 1.1])
    ap.add_argument("--n_hyp", type=int, default=5, help="distinct orientations evaluated per frame")
    args = ap.parse_args()

    ep = f"episode_{args.episode:06d}"
    W = os.path.join(os.path.abspath(args.out_root), ep)
    off = json.load(open(os.path.join(W, "track1_meta.json")))["frame_offset"]
    intr = CameraIntrinsics.load(os.path.join(W, "intrinsics_stable.json"))
    K = intr.to_matrix()
    choice = json.load(open(os.path.join(W, "mesh", "choice.json")))
    cand = choice["chosen"]
    check = json.load(open(os.path.join(W, "mesh", f"cand_{cand:06d}", "check.json")))
    frames = [p["frame"] for p in check["frames"]]
    base_mesh = os.path.join(W, "mesh", f"cand_{cand:06d}", "mesh_scaled.obj")  # always refine from stage 5c's mesh

    data = {}
    for f in frames:
        data[f] = {"rgb": np.asarray(Image.open(os.path.join(W, "frames", f"{f:06d}.png")).convert("RGB")),
                   "mask": Mask.load(os.path.join(W, "masks", "1", f"{f:06d}.png")),
                   "depth": DepthImage.load(os.path.join(W, "depth", f"{f:06d}.png"))}

    with to_debug(args.debug_log, "stage5_refine load"):
        tr = FoundationPoseTracker(Mesh.load(base_mesh), os.path.join(os.path.abspath(args.weights), "foundation_pose"))

    def evaluate(f: int) -> dict:
        d = data[f]
        with to_debug(args.debug_log, f"register {f}"):
            tr.register(V2dImage(data=d["rgb"]), d["depth"], d["mask"], intr, iteration=5)
        est = tr._est
        hyp = est.poses[:40].cpu().numpy()          # centred-mesh poses, best FP score first
        distinct = []
        for h in hyp:
            if all(rot_deg(h, g) >= 45 for g in distinct):
                distinct.append(h)
            if len(distinct) == args.n_hyp:
                break
        with torch.no_grad():
            col, dep, _ = nvdiffrast_render(K, intr.height, intr.width,
                                            torch.as_tensor(np.stack(distinct), device="cuda", dtype=torch.float),
                                            glctx=tr._glctx, mesh_tensors=est.mesh_tensors, get_normal=False)
        col, dep = col.cpu().numpy(), dep.cpu().numpy()
        m, rgb, dd = d["mask"].mask.astype(bool), d["rgb"], d["depth"].depth
        res = []
        for i in range(len(distinct)):
            sil = dep[i] > 0.001
            both = sil & m
            ce = float(np.abs(col[i][both] * 255 - rgb[both]).mean()) if both.any() else 255.0
            iou = float(both.sum() / max((sil | m).sum(), 1))
            res.append({"iou": iou, "over": float((sil & ~m).sum() / max(sil.sum(), 1)),
                        "under": float((m & ~sil).sum() / max(m.sum(), 1)),
                        "area_ratio": float(sil.sum() / max(m.sum(), 1)), "colour_err": ce,
                        "depth_cm": float(np.median(np.abs(dep[i][both] - dd[both])) * 100) if both.any() else None,
                        "fit": iou - ce / 255.0, "rot_from_fp_deg": rot_deg(distinct[i], distinct[0]),
                        "pose_centred": distinct[i].tolist()})
        best = max(range(len(res)), key=lambda i: res[i]["fit"])
        return {"frame": f, "video_frame": f + off, "own": f == cand, "hyps": res, "best": best,
                "fp_flipped": res[best]["rot_from_fp_deg"] > 90 and res[best]["fit"] > res[0]["fit"] + 0.02,
                "_render": (col[[0, best]], dep[[0, best]])}

    t0 = time.time()
    trials = []
    for rel in args.rels:
        with to_debug(args.debug_log, f"rescale {rel}"):
            tr.rescale_to(rel)
        per = [evaluate(f) for f in frames]
        fits = [p["hyps"][p["best"]]["fit"] for p in per]
        ious = [p["hyps"][p["best"]]["iou"] for p in per]
        areas = [p["hyps"][p["best"]]["area_ratio"] for p in per]
        trials.append({"rel": rel, "median_fit": float(np.median(fits)), "median_iou": float(np.median(ious)),
                       "median_area_ratio": float(np.median(areas)), "frames": per})
        log(f"  scale x{rel:.2f}: median IoU {np.median(ious):.2f}, median mesh/mask area {np.median(areas):.2f}, "
            f"FP pick flipped on {sum(p['fp_flipped'] for p in per)}/{len(per)} frames")
    best = max(trials, key=lambda t: t["median_fit"])
    rel = best["rel"]
    if rel in (min(args.rels), max(args.rels)):
        log(f"  WARNING: best relative scale x{rel:.2f} is at the edge of {args.rels}")

    # Write the refined mesh (keep the stage-5c one for comparison).
    final = os.path.join(W, "mesh_scaled.obj")
    keep = os.path.join(W, "mesh_scaled_stage5c.obj")
    if not os.path.exists(keep):
        shutil.copy(final, keep)
    scale_obj(base_mesh, final, rel)
    sc = json.load(open(os.path.join(W, "mesh", f"cand_{cand:06d}", "scale.json")))["scale"]
    v = np.array([[float(x) for x in l.split()[1:4]] for l in open(final) if l.startswith("v ")])
    extent = float(np.ptp(v, axis=0).max())

    summary = {"chosen_candidate": cand, "rel_scale": rel, "total_scale": sc * rel, "extent_m": extent,
               "trials": [{k: t[k] for k in ("rel", "median_fit", "median_iou", "median_area_ratio")} for t in trials],
               "frames": [{k: v_ for k, v_ in p.items() if k != "_render"} for p in best["frames"]],
               "fp_flipped_frames": [p["video_frame"] for p in best["frames"] if p["fp_flipped"]],
               "seconds": round(time.time() - t0)}
    json.dump(summary, open(os.path.join(W, "mesh", "refine.json"), "w"), indent=1)

    # Sheet at the chosen scale: image + mask | FoundationPose pick | best distinct orientation.
    font, small = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 15)
    tiles = []
    for p in best["frames"]:
        d = data[p["frame"]]
        m = d["mask"].mask.astype(bool)
        ys, xs = np.nonzero(m)
        pad = int(0.8 * max(np.ptp(xs), np.ptp(ys))) + 20
        y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad
        cols, deps = p["_render"]
        panels = []
        for j, name in enumerate(("image + mask", "FoundationPose pick", "best orientation")):
            img = d["rgb"].copy()
            if j:
                k = j - 1
                sil = deps[k] > 0.001
                img[sil] = (cols[k][sil] * 255).astype(np.uint8)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
            crop = Image.fromarray(img[y0:y1, x0:x1])
            crop = crop.resize((320, max(1, int(320 * crop.height / crop.width))))
            h = p["hyps"][0 if j == 1 else p["best"]] if j else None
            label = name if h is None else (f"{name}: IoU {h['iou']:.2f} over {h['over']:.2f} col {h['colour_err']:.0f}"
                                            + (f" ({h['rot_from_fp_deg']:.0f}deg)" if j == 2 else ""))
            cv_ = Image.new("RGB", (330, crop.height + 30), (20, 20, 20))
            cv_.paste(crop, (5, 30))
            ImageDraw.Draw(cv_).text((5, 6), label, fill=(255, 255, 255), font=small)
            panels.append(cv_)
        hh = max(pn.height for pn in panels)
        tile = Image.new("RGB", (1010, hh + 38), (0, 0, 0))
        flag = "  FP PICK WAS FLIPPED" if p["fp_flipped"] else ""
        ImageDraw.Draw(tile).text((8, 6), f"video frame {p['video_frame']}{' (OWN)' if p['own'] else ''}{flag}",
                                  fill=(255, 80, 80) if flag else (255, 220, 0), font=font)
        for j, pn in enumerate(panels):
            tile.paste(pn, (j * 335, 38))
        tiles.append(tile)
    tw, th = max(t.width for t in tiles), max(t.height for t in tiles)
    head = 50 + 30 * (len(trials) + 1)
    sheet = Image.new("RGB", (tw * 2, th * ((len(tiles) + 1) // 2) + head), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 10), f"{ep} — stage 5d multi-frame scale + orientation audit — chosen x{rel:.2f} "
                      f"(extent {extent:.2f} m)", fill=(255, 255, 255), font=font)
    for i, t in enumerate(trials):
        dr.text((10, 50 + 30 * i), f"scale x{t['rel']:.2f}: median IoU {t['median_iou']:.2f}, median area ratio "
                                   f"{t['median_area_ratio']:.2f}, median fit {t['median_fit']:.3f}"
                                   + ("   <== chosen" if t is best else ""),
                fill=(120, 255, 120) if t is best else (220, 220, 220), font=small)
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 2) * tw, head + (i // 2) * th))
    os.makedirs(os.path.dirname(os.path.abspath(args.sheet)), exist_ok=True)
    sheet.save(args.sheet)
    log(f"{ep}: stage 5d OK — scale x{rel:.2f} (median IoU {best['median_iou']:.2f}, area ratio "
        f"{best['median_area_ratio']:.2f}), extent {extent:.2f} m, FoundationPose pick flipped on "
        f"{len(summary['fp_flipped_frames'])}/{len(best['frames'])} frames {summary['fp_flipped_frames']}, "
        f"{summary['seconds']}s")


if __name__ == "__main__":
    main()
