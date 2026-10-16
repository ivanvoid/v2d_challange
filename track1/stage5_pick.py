#!/usr/bin/env python3
"""Stage 5a — choose the frames to build the object mesh from.

Every 3rd frame of the scored span is scored on
  area       object mask area (bigger = more detail for SAM3D)
  occlusion  fraction of the object region covered by the body IN FRONT of it: body
             silhouette pixels (SAM3D-Body mesh, projected) inside the dilated object box
             whose MoGe depth is >= 5 cm nearer than the object's median depth
  sharpness  variance of the Laplacian inside the object box (motion blur check)
  border     mask touching the image edge (object cut off)
  score = sqrt(area / max_area) * (1 - occlusion)^2 * (sharpness / max_sharpness)^0.3 * (0.2 if border)

The best --n_candidates frames at least --min_sep frames apart are kept.
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
    ap.add_argument("--n_candidates", type=int, default=2)
    ap.add_argument("--min_sep", type=int, default=30)
    ap.add_argument("--step", type=int, default=3)
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
        rows.append({"frame": l, "video_frame": l + off, "area": int(m.sum()), "occlusion": occ,
                     "sharpness": sharp, "border": border, "object_depth_m": oz,
                     "box": [int(x0), int(y0), int(x1), int(y1)]})
    if not rows:
        raise SystemExit(f"{ep}: no non-empty object mask in the scored span")
    amax = max(r["area"] for r in rows)
    smax = max(r["sharpness"] for r in rows)
    for r in rows:
        r["score"] = (np.sqrt(r["area"] / amax) * (1 - r["occlusion"]) ** 2 *
                      (r["sharpness"] / smax) ** 0.3 * (0.2 if r["border"] else 1.0))
    chosen = []
    for r in sorted(rows, key=lambda r: -r["score"]):
        if all(abs(r["frame"] - c["frame"]) >= args.min_sep for c in chosen):
            chosen.append(r)
        if len(chosen) == args.n_candidates:
            break
    os.makedirs(os.path.join(W, "mesh"), exist_ok=True)
    json.dump({"episode": args.episode, "rows": rows, "chosen": [c["frame"] for c in chosen]},
              open(os.path.join(W, "mesh", "candidates.json"), "w"), indent=1)
    log(f"{ep}: scored {len(rows)} frames; mesh will be built from:")
    for c in chosen:
        log(f"  video frame {c['video_frame']}: score {c['score']:.2f}, area {c['area']} px, "
            f"occluded {100 * c['occlusion']:.0f}%, sharpness {c['sharpness']:.0f}, "
            f"{'TOUCHES BORDER' if c['border'] else 'inside image'}")


if __name__ == "__main__":
    main()
