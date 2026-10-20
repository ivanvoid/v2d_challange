#!/usr/bin/env python3
"""Stage 5a — choose the frames to build the object mesh from.

Every 3rd frame of the contact span (scored_first..scored_last) is measured:
  status     stage-4 mask status (masks_quality.json): only 'ok' frames are eligible — SAM2 sure, both
             propagation passes agree, no leak, right size / depth / colour (falls back to 'low_conf' if
             fewer than --n_candidates frames are 'ok')
  area       object mask area (bigger = more detail for SAM3D)
  extent     real-world size of the mask (stage 4 size_m) relative to the 90th percentile over eligible
             frames: ~1 = the object looks COMPLETE (a pan with its handle, not only its body)
  occlusion  fraction of the object region covered by the body IN FRONT of it: body
             silhouette pixels (SAM3D-Body mesh, projected) inside the dilated object box
             whose MoGe depth is >= 5 cm nearer than the object's median depth; > --max_occ excluded
  sharpness  variance of the Laplacian inside the object box (motion blur check)
  border     mask touching the image edge (object cut off): excluded
  score = sqrt(area / max_area) * min(1, extent)^2 * (1 - occlusion)^2 * (sharpness / max_sharpness)^0.3

Candidates are taken greedily by score, each at least --min_sep frames from the others AND with a
clearly different mask shape (IoU < --max_shape_iou between the masks cropped to their box and resized
to 64x64): different views of the object, not the same pose twice.
Writes out/episode_XXXXXX/mesh/candidates.json (per-frame table + chosen frames).

Run with the sam3d_body venv:
    /opt/venvs/sam3d_body/bin/python track1/stage5_pick.py --episode 28 --out_root out
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from logutil import log


def load_depth(path: str) -> np.ndarray:
    px = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return 1.0 / np.maximum(px / 65535.0, 1e-6) - 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--n_candidates", type=int, default=3)
    ap.add_argument("--min_sep", type=int, default=30)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--max_occ", type=float, default=0.15)
    ap.add_argument("--max_shape_iou", type=float, default=0.8)
    args = ap.parse_args()

    ep = f"episode_{args.episode:06d}"
    W = os.path.join(os.path.abspath(args.out_root), ep)
    meta = json.load(open(os.path.join(W, "track1_meta.json")))
    off, first, last = meta["frame_offset"], meta["scored_first"], meta["scored_last"]
    intr = json.load(open(os.path.join(W, "intrinsics_stable.json")))
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
    H, Wd = intr["height"], intr["width"]
    body = torch.load(os.path.join(W, "body", "mhr_mesh.pt"), map_location="cpu", weights_only=False)
    bv, bf = body["pred_vertices"].numpy(), body["faces"].numpy()
    qp = os.path.join(W, "masks_quality.json")
    quality = json.load(open(qp))["frames"] if os.path.exists(qp) else {}

    rows = []
    for l in range(first - off, last - off + 1, args.step):
        m = cv2.imread(os.path.join(W, "masks", "1", f"{l:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None or not (m > 0).any():
            continue
        m = m > 0
        ys, xs = np.nonzero(m)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        pad = int(0.15 * max(x1 - x0, y1 - y0)) + 5
        X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(Wd, x1 + pad), min(H, y1 + pad)
        d = load_depth(os.path.join(W, "depth", f"{l:06d}.png"))
        oz = float(np.median(d[m]))
        uvw = bv[l] @ K.T
        uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
        sil = np.zeros((H, Wd), np.uint8)
        cv2.fillPoly(sil, list(np.round(uv[bf]).astype(np.int32)), 1)
        region = np.zeros((H, Wd), bool)
        region[Y0:Y1, X0:X1] = True
        front = (sil > 0) & region & (d < oz - 0.05) & ~m
        occ = float(front.sum() / (m.sum() + front.sum()))
        gray = cv2.cvtColor(cv2.imread(os.path.join(W, "frames", f"{l:06d}.png")), cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F).var())
        border = bool(x0 <= 1 or y0 <= 1 or x1 >= Wd - 1 or y1 >= H - 1)
        qf = quality.get(str(l + off), {})
        shape = cv2.resize(m[y0:y1, x0:x1].astype(np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST) > 0
        rows.append({"frame": l, "video_frame": l + off, "area": int(m.sum()), "occlusion": occ,
                     "sharpness": sharp, "border": border, "object_depth_m": oz,
                     "status": qf.get("status", "unknown"), "size_m": qf.get("size_m"),
                     "box": [int(x0), int(y0), int(x1), int(y1)], "_shape": shape})
    if not rows:
        raise SystemExit(f"{ep}: no non-empty object mask in the scored span")
    def eligible(statuses):
        return [r for r in rows if r["status"] in statuses and not r["border"] and r["occlusion"] <= args.max_occ]
    pool = eligible(("ok",))
    if len(pool) < args.n_candidates:
        log(f"  only {len(pool)} 'ok' frames without occlusion / border; also using 'low_conf' frames")
        pool = eligible(("ok", "low_conf"))
    if not pool:
        log("  no clear frame at all; falling back to every non-empty frame")
        pool = rows
    amax = max(r["area"] for r in pool)
    smax = max(r["sharpness"] for r in pool)
    sizes = [r["size_m"] for r in pool if r["size_m"]]
    full = float(np.percentile(sizes, 90)) if sizes else None
    for r in rows:
        r["extent"] = (r["size_m"] / full) if (full and r["size_m"]) else 1.0
        r["eligible"] = r in pool
        r["score"] = (np.sqrt(r["area"] / amax) * min(1.0, r["extent"]) ** 2 * (1 - r["occlusion"]) ** 2 *
                      (r["sharpness"] / smax) ** 0.3) if r["eligible"] else 0.0
    def shape_iou(a, b):
        return float((a["_shape"] & b["_shape"]).sum() / max((a["_shape"] | b["_shape"]).sum(), 1))
    chosen = []
    for r in sorted(pool, key=lambda r: -r["score"]):
        if all(abs(r["frame"] - c["frame"]) >= args.min_sep and shape_iou(r, c) < args.max_shape_iou for c in chosen):
            chosen.append(r)
        if len(chosen) == args.n_candidates:
            break
    for r in rows:
        r.pop("_shape", None)
    os.makedirs(os.path.join(W, "mesh"), exist_ok=True)
    json.dump({"episode": args.episode, "rows": rows, "chosen": [c["frame"] for c in chosen],
               "eligible_frames": [r["frame"] for r in pool], "params": vars(args)},
              open(os.path.join(W, "mesh", "candidates.json"), "w"), indent=1)
    log(f"{ep}: scored {len(rows)} frames; mesh will be built from:")
    for c in chosen:
        log(f"  video frame {c['video_frame']}: score {c['score']:.2f}, status {c['status']}, area {c['area']} px, "
            f"extent {c['extent']:.2f} of the full object, occluded {100 * c['occlusion']:.0f}%, "
            f"sharpness {c['sharpness']:.0f}")


if __name__ == "__main__":
    main()
