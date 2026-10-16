#!/usr/bin/env python3
"""Stage 8a — put the object into the body's metric scale.

The body (SAM3D-Body + MHR fit) and the object (FoundationPose on MoGe depth) are both metric,
but their depth scales differ: on the body's own pixels MoGe depth / body depth = ratio < 1
(stage 2: 0.87-0.95). The scorer aligns everything with ONE Sim(3) fitted on the body, so the
object must live in the body's scale. Scaling the object's camera-frame position AND size by
s = 1/ratio keeps its image projection identical and moves it to the body's scale.

Variants tested (on every 2nd scored frame):
  none       s = 1
  global     s = 1 / median(ratio)                         position and size
  per_frame  position by a smoothed per-frame s_f, size by the global s
Score per variant: median hand-object gap (closest MHR hand vertex to the object surface, cm)
+ 3 x mean penetration (cm). The best variant is written.

Writes out/episode_XXXXXX/poses_final/<frame:06d>.json (scale field = object size factor)
and track/scale_match.json (ratio per frame, variant scores, choice).

    /opt/venvs/sam3d_body/bin/python track1/stage8_scale.py --episode 28 --out_root out \
        --model weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import trimesh
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logutil import log  # noqa: E402
from qa_episode import MHR, hand_vertex_ids, signed_dist, surface_query  # noqa: E402


def load_depth(path):
    px = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return 1.0 / np.maximum(px / 65535.0, 1e-6) - 1.0


def moge_over_body(v, K, depth):
    """Median MoGe depth / body depth over the front-most body vertex per pixel."""
    uvw = v @ K.T
    uv = np.round(uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)).astype(np.int64)
    h, w = depth.shape
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h) & (v[:, 2] > 0)
    if ok.sum() < 50:
        return np.nan
    lin, z = uv[ok, 1] * w + uv[ok, 0], v[ok, 2]
    o = np.lexsort((z, lin))
    lin, z = lin[o], z[o]
    first = np.r_[True, lin[1:] != lin[:-1]]
    return float(np.median(depth.reshape(-1)[lin[first]] / z[first]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pen_weight", type=float, default=3.0)
    args = ap.parse_args()

    ep = f"episode_{args.episode:06d}"
    W = os.path.join(os.path.abspath(args.out_root), ep)
    meta = json.load(open(os.path.join(W, "track1_meta.json")))
    n, off, first, last = meta["n_frames"], meta["frame_offset"], meta["scored_first"], meta["scored_last"]
    it = json.load(open(os.path.join(W, "intrinsics_stable.json")))
    K = np.array([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]], [0, 0, 1]])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # body on the scored span (video frames) from the one-identity fit
    fit = np.load(os.path.join(W, "body", "mhr_fit.npz"))
    mhr = MHR(args.model, dev, chunk=32, precision="float32")
    vf = np.arange(first, last + 1)
    loc = vf - off
    pose = torch.as_tensor(fit["pose"][vf], dtype=torch.float64, device=dev)
    z = torch.as_tensor(np.concatenate([fit["scales"], fit["shape"]])[None], dtype=torch.float64, device=dev)
    verts = (mhr.run(pose, z)[0] / 1000.0).cpu().numpy()
    hand_ids = hand_vertex_ids(mhr, pose, z)

    # 1. depth ratio per scored frame
    ratio = np.array([moge_over_body(verts[k], K, load_depth(os.path.join(W, "depth", f"{l:06d}.png")))
                      for k, l in enumerate(loc)])
    good = np.isfinite(ratio)
    ratio_f = np.interp(np.arange(len(ratio)), np.flatnonzero(good), ratio[good])
    ratio_s = gaussian_filter1d(median_filter(ratio_f, 15, mode="nearest"), 5, mode="nearest")
    s_global = float(1.0 / np.median(ratio_f))

    # object: smoothed poses (stage 7), mesh
    mesh = trimesh.load(os.path.join(W, "mesh_scaled.obj"), force="mesh", process=False)
    tree, spts, snrm = surface_query(mesh)
    poses = []
    for f in range(n):
        d = json.load(open(os.path.join(W, "poses_smoothed", f"{f:06d}.json")))
        w_, x, y, z_ = d["rotation"]
        poses.append((Rotation.from_quat([x, y, z_, w_]).as_matrix(), np.asarray(d["translation"], float)))

    def variant(name):
        if name == "none":
            return np.ones(n), 1.0
        if name == "global":
            return np.full(n, s_global), s_global
        s_loc = np.full(n, s_global)
        s_loc[loc] = 1.0 / ratio_s
        s_loc[:loc[0]] = s_loc[loc[0]]
        s_loc[loc[-1] + 1:] = s_loc[loc[-1]]
        return s_loc, s_global

    results = {}
    eval_k = np.arange(0, len(loc), 2)
    for name in ("none", "global", "per_frame"):
        s_pos, s_size = variant(name)
        gaps, pens = [], []
        for k in eval_k:
            l = loc[k]
            R, t = poses[l]
            t2 = t * s_pos[l]
            hp = verts[k, hand_ids]
            local = ((hp - t2) @ R) / s_size
            sd = signed_dist(tree, spts, snrm, local) * s_size
            gaps.append(np.abs(sd).min() * 100)
            pens.append(np.maximum(-sd, 0).mean() * 100)
        gaps, pens = np.array(gaps), np.array(pens)
        results[name] = {"gap_median_cm": float(np.median(gaps)), "gap_frac_under_2cm": float((gaps < 2).mean()),
                         "pen_mean_cm": float(pens.mean()),
                         "score": float(np.median(gaps) + args.pen_weight * pens.mean()),
                         "gaps": gaps.tolist(), "size_factor": s_size}
        log(f"  {name:9s}: hand-object gap median {np.median(gaps):5.1f} cm, frames < 2 cm "
            f"{100 * (gaps < 2).mean():3.0f}%, penetration {pens.mean():.3f} cm, object size x{s_size:.3f}")
    best = min(results, key=lambda k: results[k]["score"])
    s_pos, s_size = variant(best)

    out = os.path.join(W, "poses_final")
    os.makedirs(out, exist_ok=True)
    for f in range(n):
        d = json.load(open(os.path.join(W, "poses_smoothed", f"{f:06d}.json")))
        d["translation"] = (np.asarray(d["translation"]) * s_pos[f]).tolist()
        d["scale"] = [s_size] * 3
        json.dump(d, open(os.path.join(out, f"{f:06d}.json"), "w"), indent=1)
    json.dump({"episode": args.episode, "ratio_moge_over_body": ratio.tolist(), "ratio_smoothed": ratio_s.tolist(),
               "video_frames": vf.tolist(), "eval_video_frames": vf[eval_k].tolist(), "s_global": s_global,
               "variants": results, "chosen": best},
              open(os.path.join(W, "track", "scale_match.json"), "w"), indent=1)
    log(f"{ep}: stage 8a OK — MoGe/body depth ratio median {np.median(ratio_f):.3f} (range "
        f"{np.percentile(ratio_f, 5):.3f}-{np.percentile(ratio_f, 95):.3f}); chose '{best}' "
        f"(object x{s_size:.3f}); hand-object gap {results['none']['gap_median_cm']:.1f} -> "
        f"{results[best]['gap_median_cm']:.1f} cm")


if __name__ == "__main__":
    main()
