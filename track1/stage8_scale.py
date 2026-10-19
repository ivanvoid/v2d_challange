#!/usr/bin/env python3
"""Stage 8a — place the object where the hands touch it (depth along the camera ray).

The body (SAM3D-Body + MHR fit) and the object (FoundationPose on stabilised MoGe depth) are both
metric, but their depth scales disagree. The scorer aligns everything with ONE Sim(3) fitted on the
body, so what matters is where the object sits relative to the hands. Scaling the object's
camera-frame position AND size by k keeps its image projection identical and only moves it along
the camera ray. Experiment p1 (docs/EXPERIMENTS.md) showed that hand contact on scored frames is
the right guide (moving 8 episodes to the body's scale made CD-O worse by 5.9 cm).

Scored frames are the hand-object CONTACT frames (submission/scored_spans.json "frames"), so the
placement is chosen on them only:
  k        grid 0.80 .. 1.25 (step 0.025); k = 1 is MoGe's scale, k = 1/median(ratio) the body's
  mode     fixed     s_f = k
           follow    s_f = k * median(ratio) / ratio_f   (object follows the body's depth trend;
                     ratio_f = MoGe / body depth on the body's pixels, median 15 + Gaussian 5)
  score    median over scored frames of max(closest hand-object distance - 1 cm, 0) (0 if a hand
           point is inside; gaps up to --contact_tol_cm count as touching) + pen_weight x mean
           penetration (cm), both exactly as the scorer computes PEN:
           the BUDGETED mesh (4096 faces/vertices, as packed), winding-number inside test
           (pen_fast.PenetrationMesh), 512 hand points
  choice   among scores within --tie of the best, the smallest change (|log k|, "fixed" first):
           every 1 % of k moves the object ~4 cm, so it only moves on clear evidence
The best (mode, k) is written; the full table, the margin to the runner-up and the old three
variants (none / global / per_frame) are logged in track/scale_match.json.

Writes out/episode_XXXXXX/poses_final/<frame:06d>.json (scale field = object size factor)
and track/scale_match.json.

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
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
from logutil import log  # noqa: E402
from pen_fast import PenetrationMesh  # noqa: E402
from qa_episode import MHR, hand_vertex_ids  # noqa: E402

sys.path.insert(0, str(paths.KIT))
from v2dlb.mesh_budget import budget_mesh  # noqa: E402


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
    ap.add_argument("--contact_tol_cm", type=float, default=1.0, help="gaps up to this count as touching")
    ap.add_argument("--tie", type=float, default=0.05, help="scores within this of the best are ties")
    ap.add_argument("--k_min", type=float, default=0.80)
    ap.add_argument("--k_max", type=float, default=1.25)
    ap.add_argument("--k_step", type=float, default=0.025)
    ap.add_argument("--hand_points", type=int, default=512)
    ap.add_argument("--spans", default=str(paths.DATA / "submission" / "scored_spans.json"))
    args = ap.parse_args()

    ep = f"episode_{args.episode:06d}"
    W = os.path.join(os.path.abspath(args.out_root), ep)
    meta = json.load(open(os.path.join(W, "track1_meta.json")))
    n, off, first, last = meta["n_frames"], meta["frame_offset"], meta["scored_first"], meta["scored_last"]
    scored = np.asarray(json.load(open(args.spans))[str(args.episode)]["frames"])
    it = json.load(open(os.path.join(W, "intrinsics_stable.json")))
    K = np.array([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]], [0, 0, 1]])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # body on first..last (video frames) from the one-identity fit
    fit = np.load(os.path.join(W, "body", "mhr_fit.npz"))
    mhr = MHR(args.model, dev, chunk=32, precision="float32")
    vf = np.arange(first, last + 1)
    loc = vf - off
    pose = torch.as_tensor(fit["pose"][vf], dtype=torch.float64, device=dev)
    z = torch.as_tensor(np.concatenate([fit["scales"], fit["shape"]])[None], dtype=torch.float64, device=dev)
    verts = (mhr.run(pose, z)[0] / 1000.0).cpu().numpy()
    hand_ids = hand_vertex_ids(mhr, pose[:1], z)
    hand_ids = np.random.default_rng(0).choice(hand_ids, size=min(args.hand_points, len(hand_ids)), replace=False)

    # 1. MoGe / body depth ratio on the body's pixels, every frame of first..last
    ratio = np.array([moge_over_body(verts[k], K, load_depth(os.path.join(W, "depth", f"{l:06d}.png")))
                      for k, l in enumerate(loc)])
    good = np.isfinite(ratio)
    ratio_f = np.interp(np.arange(len(ratio)), np.flatnonzero(good), ratio[good])
    ratio_s = gaussian_filter1d(median_filter(ratio_f, 15, mode="nearest"), 5, mode="nearest")
    sc_k = scored - first                                            # scored frames as indices into vf
    r_med = float(np.median(ratio_f[sc_k]))
    s_global = 1.0 / r_med
    follow = np.ones(n)                                              # relative trend, 1 at the median
    follow[loc] = r_med / ratio_s
    follow[:loc[0]] = follow[loc[0]]
    follow[loc[-1] + 1:] = follow[loc[-1]]

    # 2. object: smoothed poses (stage 7) and the BUDGETED mesh, exactly as packed
    mv, mf = budget_mesh(os.path.join(W, "mesh_scaled.obj"), 4096, 4096)
    pm = PenetrationMesh(mv, mf)
    tm = trimesh.Trimesh(mv, pm.mesh.faces, process=False)
    tree = cKDTree(trimesh.sample.sample_surface_even(tm, 30000, seed=0)[0])
    R = np.zeros((n, 3, 3)); t = np.zeros((n, 3))
    for f in range(n):
        d = json.load(open(os.path.join(W, "poses_smoothed", f"{f:06d}.json")))
        w_, x, y, z_ = d["rotation"]
        R[f], t[f] = Rotation.from_quat([x, y, z_, w_]).as_matrix(), np.asarray(d["translation"], float)
    s0 = float(np.ravel(d.get("scale", [1.0]))[0])                   # stage-5e size factor of mesh_scaled.obj

    hands = verts[sc_k][:, hand_ids]                                 # [S, P, 3] scored frames
    l_sc = scored - off
    def evaluate(s_pos: np.ndarray, size: float) -> dict:
        """Gap/penetration on scored frames for object position x s_pos[f], size x size."""
        tt = t[l_sc] * s_pos[l_sc][:, None]
        local = np.einsum("tpi,tij->tpj", hands - tt[:, None], R[l_sc]) / (s0 * size)
        flat = local.reshape(-1, 3)
        dep = pm.depth(flat).reshape(local.shape[:2]) * s0 * size
        dist = tree.query(flat)[0].reshape(local.shape[:2]) * s0 * size
        gap = np.where(dep > 0, 0.0, dist).min(1) * 100
        pen = dep.mean(1) * 100
        excess = np.maximum(gap - args.contact_tol_cm, 0.0)
        return {"gap_median_cm": float(np.median(gap)), "contact_frac": float((gap < 2).mean()),
                "pen_cm": float(pen.mean()),
                "score": float(np.median(excess) + args.pen_weight * pen.mean())}

    ks = np.round(np.arange(args.k_min, args.k_max + 1e-9, args.k_step), 4)
    table = []
    for mode in ("fixed", "follow"):
        for k in ks:
            s_pos = k * (follow if mode == "follow" else np.ones(n))
            table.append({"mode": mode, "k": float(k), **evaluate(s_pos, float(k))})
    # every 1 % of k moves the object ~4 cm: among near-ties prefer the smallest change from MoGe's placement
    top = min(r["score"] for r in table)
    near = [r for r in table if r["score"] <= top + args.tie]
    best = min(near, key=lambda r: (abs(np.log(r["k"])), r["mode"] != "fixed", r["score"]))
    runner = min((r for r in table if r is not best), key=lambda r: r["score"])
    legacy = {"none": next(r for r in table if r["mode"] == "fixed" and abs(r["k"] - 1) < 1e-9)}
    legacy["global"] = {"k": s_global, **evaluate(np.full(n, s_global), s_global)}
    legacy["per_frame"] = {"k": s_global, **evaluate(s_global * follow, s_global)}

    s_pos = best["k"] * (follow if best["mode"] == "follow" else np.ones(n))
    out = os.path.join(W, "poses_final")
    os.makedirs(out, exist_ok=True)
    for f in range(n):
        d = json.load(open(os.path.join(W, "poses_smoothed", f"{f:06d}.json")))
        d["translation"] = (np.asarray(d["translation"]) * s_pos[f]).tolist()
        d["scale"] = [s0 * best["k"]] * 3
        json.dump(d, open(os.path.join(out, f"{f:06d}.json"), "w"), indent=1)
    json.dump({"episode": args.episode, "chosen": f"{best['mode']} k={best['k']:.3f}", "best": best,
               "runner_up": runner, "margin": runner["score"] - best["score"], "s_global": s_global,
               "ratio_median_scored": r_med, "legacy_variants": legacy, "table": table,
               "ratio_moge_over_body": ratio.tolist(), "ratio_smoothed": ratio_s.tolist(),
               "video_frames": vf.tolist(), "scored_frames": scored.tolist(), "params": vars(args)},
              open(os.path.join(W, "track", "scale_match.json"), "w"), indent=1)
    n0 = legacy["none"]
    log(f"{ep}: stage 8a OK — chose {best['mode']} k={best['k']:.3f} (body scale would be k={s_global:.3f}); on "
        f"{len(scored)} scored frames: gap {n0['gap_median_cm']:.1f} -> {best['gap_median_cm']:.1f} cm, contact "
        f"{100 * n0['contact_frac']:.0f} -> {100 * best['contact_frac']:.0f}%, PEN {n0['pen_cm']:.3f} -> "
        f"{best['pen_cm']:.3f} cm; runner-up {runner['mode']} k={runner['k']:.3f} (+{runner['score'] - best['score']:.2f})")


if __name__ == "__main__":
    main()
