#!/usr/bin/env python3
"""Stage 8b — remove hand-object penetration (the PEN metric) by moving the object off the hands.

PEN = depth of the MHR hand points inside OUR OWN object mesh, as packed (budget-simplified to
4096 faces / vertices). This step, per frame:
  1. hand points: all MHR hand vertices of the submitted body (mhr_fit.npz), the object posed by
     poses_final/ (stage 8a): point_scene = s R point_mesh + t
  2. inside test exactly like the scorer (winding number on the BUDGETED mesh), depth = distance
     to the surface; for inside points the closest surface point gives the way out
  3. iterate (max --iters): shift the object by the deepest point's exit vector (+ --margin_mm),
     keep the shift with the smallest total depth, cap at --max_shift_cm
  4. shifts are smoothed over time (Gaussian, --sigma frames), applied, and one more unsmoothed
     pass removes what smoothing gave back
Writes out/<ep>/poses_depen/ (same format as poses_final) and track/depen.json (PEN proxy before
/after on 512 fixed hand points, shift statistics, object acceleration before/after).

    /opt/venvs/sam3d_body/bin/python track1/stage8_depen.py --episode 16 --out_root out
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from logutil import log  # noqa: E402

sys.path.insert(0, str(paths.KIT))
sys.path.insert(0, str(paths.KIT_TRACK1))
from pen_fast import PenetrationMesh  # noqa: E402
from v2dlb.mesh_budget import budget_mesh  # noqa: E402
from mesh_to_mhr_params import MHR  # noqa: E402
from qa_episode import hand_vertex_ids  # noqa: E402

MHR_MODEL = paths.DATA / "weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt"


def load_pose(path):
    d = json.load(open(path))
    w, x, y, z = d["rotation"]
    return Rotation.from_quat([x, y, z, w]).as_matrix(), np.asarray(d["translation"], float), d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--margin_mm", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--max_shift_cm", type=float, default=5.0)
    ap.add_argument("--sigma", type=float, default=3.0)
    args = ap.parse_args()

    ep = f"episode_{args.episode:06d}"
    W = Path(args.out_root).resolve() / ep
    meta = json.load(open(W / "track1_meta.json"))
    n, off = meta["n_frames"], meta["frame_offset"]
    first, last = meta["scored_first"] - off, meta["scored_last"] - off
    t0 = time.time()

    # object: budgeted mesh exactly as the packer makes it, per-frame pose from stage 8a
    mv, mf = budget_mesh(str(W / "mesh_scaled.obj"), 4096, 4096)
    tri = mv[mf]
    keep = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) > 0
    mf = mf[keep]
    pm = PenetrationMesh(mv, mf)
    poses = [load_pose(W / "poses_final" / f"{f:06d}.json") for f in range(n)]
    R = np.stack([p[0] for p in poses]); t = np.stack([p[1] for p in poses])
    s = float(np.asarray(poses[0][2]["scale"]).reshape(-1)[0])

    # body: hand vertices of the submitted MHR fit, per local frame
    fit = np.load(W / "body" / "mhr_fit.npz")
    vidx = np.clip(np.arange(n) + off, 0, len(fit["pose"]) - 1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHR(str(MHR_MODEL), dev, 64, "float32")
    pose = torch.as_tensor(fit["pose"][vidx], dtype=torch.float64, device=dev)
    z = torch.as_tensor(np.concatenate([fit["scales"], fit["shape"]])[None], dtype=torch.float64, device=dev)
    hid = hand_vertex_ids(mhr, pose[:1], z)
    hands = (mhr.run(pose, z)[0] / 1000.0).cpu().numpy()[:, hid]          # [n, P, 3] metres
    probe = np.random.default_rng(0).choice(len(hid), size=min(512, len(hid)), replace=False)

    def to_local(f, shift):
        return ((hands[f] - (t[f] + shift)) @ R[f]) / s

    def depth_of(f, shift, pts=None):
        loc = to_local(f, shift) if pts is None else pts
        return pm.depth(loc) * s                                             # metres, scene units

    def pen_proxy(shifts):                                                   # cm, scored frames, 512 points
        vals = []
        for f in range(first, last + 1):
            loc = to_local(f, shifts[f])[probe]
            vals.append(np.maximum(pm.depth(loc) * s, 0).mean() * 100)
        return float(np.mean(vals))

    def solve(f, base):
        """Best extra shift for frame f starting from base shift (scene metres)."""
        loc = to_local(f, base)
        d = pm.depth(loc)
        if not (d > 0).any():
            return np.zeros(3), 0.0
        best, best_cost, extra = np.zeros(3), float(d.sum()), np.zeros(3)
        for _ in range(args.iters):
            loc = to_local(f, base + extra)
            d, closest = pm.depth(loc, return_closest=True)
            inside = np.flatnonzero(d > 0)
            if inside.size == 0:
                return extra, 0.0
            k = inside[np.argmax(d[inside])]
            c = closest[k]
            v_local = (c - loc[k])                                           # point -> surface
            v_local = v_local / (np.linalg.norm(v_local) + 1e-12) * (np.linalg.norm(v_local) + args.margin_mm / 1000 / s)
            extra = extra - s * (R[f] @ v_local)                             # move the object the other way
            if np.linalg.norm(extra) > args.max_shift_cm / 100:
                extra = extra / np.linalg.norm(extra) * args.max_shift_cm / 100
            cost = float(pm.depth(to_local(f, base + extra)).sum())
            if cost < best_cost:
                best, best_cost = extra.copy(), cost
        return best, best_cost

    zero = np.zeros((n, 3))
    pen_before = pen_proxy(zero)
    raw = np.stack([solve(f, zero[f])[0] for f in range(n)])
    smooth = gaussian_filter1d(raw, args.sigma, axis=0, mode="nearest")
    # smoothing spreads each shift over neighbours but lowers its peak: keep at least the raw shift
    # along its own direction on frames that needed one
    for f in range(n):
        if np.linalg.norm(raw[f]) > 0:
            u = raw[f] / np.linalg.norm(raw[f])
            deficit = np.linalg.norm(raw[f]) - smooth[f] @ u
            if deficit > 0:
                smooth[f] = smooth[f] + deficit * u
    final = smooth.copy()
    residual_fix = 0
    for f in range(n):
        extra, _ = solve(f, final[f])
        if np.linalg.norm(extra) > 0:
            final[f] = final[f] + extra
            residual_fix += 1
    pen_after = pen_proxy(final)

    out = W / "poses_depen"
    out.mkdir(exist_ok=True)
    for f in range(n):
        d = dict(poses[f][2])
        d["translation"] = (t[f] + final[f]).tolist()
        json.dump(d, open(out / f"{f:06d}.json", "w"), indent=1)

    def acc(x):
        return float(np.linalg.norm(x[:-2] - 2 * x[1:-1] + x[2:], axis=1).mean() * 100)
    sc = slice(first, last + 1)
    mag = np.linalg.norm(final, axis=1) * 100
    rep = {"episode": args.episode, "pen_proxy_before_cm": pen_before, "pen_proxy_after_cm": pen_after,
           "frames_shifted": int((mag > 0).sum()), "shift_cm_median_of_shifted": float(np.median(mag[mag > 0])) if (mag > 0).any() else 0.0,
           "shift_cm_max": float(mag.max()), "residual_fix_frames": residual_fix,
           "acc_o_before": acc(t[sc]), "acc_o_after": acc((t + final)[sc]), "params": vars(args),
           "seconds": round(time.time() - t0)}
    json.dump(rep, open(W / "track" / "depen.json", "w"), indent=1)
    log(f"{ep}: stage 8b — PEN proxy {pen_before:.4f} -> {pen_after:.4f} cm; {rep['frames_shifted']}/{n} frames "
        f"shifted (median {rep['shift_cm_median_of_shifted']:.2f} cm, max {rep['shift_cm_max']:.2f} cm); object "
        f"accel {rep['acc_o_before']:.3f} -> {rep['acc_o_after']:.3f} cm/frame^2 ({rep['seconds']}s)")


if __name__ == "__main__":
    main()
