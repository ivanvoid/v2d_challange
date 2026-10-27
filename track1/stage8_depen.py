#!/usr/bin/env python3
"""Stage 8b — remove hand-object penetration (the PEN metric) by moving the object off the hands.

PEN = depth of the MHR hand points inside OUR OWN object mesh, as packed (budget-simplified to
4096 faces / vertices). This step, per frame:
  1. hand points: fixed subset of MHR hand vertices of the submitted body (mhr_fit.npz), the object posed by
     poses_final/ (stage 8a): point_scene = s R point_mesh + t
  2. inside test exactly like the scorer (winding number on the BUDGETED mesh), depth = distance
     to the surface; for inside points the closest surface point gives the way out
  3. iterate (max --iters): shift the object by the deepest point's exit vector (+ --margin_mm),
     keep the shift with the smallest total depth, cap the FINAL total shift at --max_shift_cm
  4. shifts are smoothed over time (Gaussian, --sigma frames). --temporal_only omits the original
     unsmoothed peak restoration and residual fix; these can introduce large acceleration spikes.
Writes out/<ep>/poses_depen/ (same format as poses_final) and track/depen.json (PEN proxy before
/after on a fixed hand subset plus an independent holdout, shift and acceleration statistics).
This is an isolated experiment: run_stage.sh does not use poses_depen. Verify the packed result
and contact/acceleration guardrails before assembling a submission candidate.

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


def cap_displacement(shift: np.ndarray, max_shift_m: float) -> np.ndarray:
    """Cap the TOTAL correction, including smoothing and residual repairs."""
    norm = np.linalg.norm(shift, axis=-1, keepdims=True)
    return shift * np.minimum(1.0, max_shift_m / np.maximum(norm, 1e-12))


def scored_local_indices(scored_video_frames: np.ndarray, offset: int, n_frames: int) -> np.ndarray:
    indices = np.asarray(scored_video_frames, dtype=np.int64) - offset
    if indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0) or np.any(indices >= n_frames):
        raise ValueError("scored video frames must map inside the trimmed clip")
    return indices


def solve_translation(pm, hand_points: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
                      scale: float, base_shift: np.ndarray, max_shift_m: float,
                      iters: int = 8, margin_m: float = 0.002) -> tuple[np.ndarray, float]:
    """Find a bounded translation reducing hand-inside-object depth; never exceed total displacement."""
    def local(extra):
        return ((hand_points - (translation + base_shift + extra)) @ rotation) / scale

    depths = pm.depth(local(np.zeros(3)))
    best_cost = float(depths.sum())
    if not (depths > 0).any():
        return np.zeros(3), best_cost
    best = np.zeros(3)
    extra = np.zeros(3)
    for _ in range(iters):
        points = local(extra)
        depths, closest = pm.depth(points, return_closest=True)
        inside = np.flatnonzero(depths > 0)
        if not len(inside):
            return extra, 0.0
        k = inside[np.argmax(depths[inside])]
        direction = closest[k] - points[k]
        direction *= (np.linalg.norm(direction) + margin_m / scale) / (np.linalg.norm(direction) + 1e-12)
        # Object moves opposite the shortest exit direction of the hand point.
        extra = cap_displacement(base_shift + extra - scale * (rotation @ direction), max_shift_m) - base_shift
        cost = float(pm.depth(local(extra)).sum())
        if cost < best_cost:
            best, best_cost = extra.copy(), cost
    return best, best_cost


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--margin_mm", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--max_shift_cm", type=float, default=5.0)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--temporal_only", action="store_true",
                    help="apply smoothed shifts only; do not restore raw peaks or add per-frame residual fixes")
    ap.add_argument("--solver_hand_points", type=int, default=512,
                    help="fixed hand subset for iterative fitting; final verification also uses a disjoint sample")
    ap.add_argument("--spans", type=Path, default=paths.DATA / "submission/scored_spans.json")
    args = ap.parse_args()

    ep = f"episode_{args.episode:06d}"
    W = Path(args.out_root).resolve() / ep
    meta = json.load(open(W / "track1_meta.json"))
    n, off = meta["n_frames"], meta["frame_offset"]
    scored = np.asarray(json.load(open(args.spans))[str(args.episode)]["frames"])
    scored_local = scored_local_indices(scored, off, n)
    max_shift_m = args.max_shift_cm / 100
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
    if args.solver_hand_points < 1:
        raise ValueError("--solver_hand_points must be positive")
    perm = np.random.default_rng(0).permutation(len(hid))
    probe = perm[:min(args.solver_hand_points, len(hid))]
    holdout = perm[len(probe):len(probe) + min(512, len(hid) - len(probe))]

    def to_local(f, shift, sample):
        return ((hands[f, sample] - (t[f] + shift)) @ R[f]) / s

    def pen_proxy(shifts, sample):                                           # cm, exact scored frames
        vals = []
        for f in scored_local:
            loc = to_local(f, shifts[f], sample)
            vals.append(np.maximum(pm.depth(loc) * s, 0).mean() * 100)
        return float(np.mean(vals))

    def solve(f, base):
        """Best extra shift for frame f starting from base shift (scene metres)."""
        return solve_translation(pm, hands[f, probe], R[f], t[f], s, base, max_shift_m,
                                 args.iters, args.margin_mm / 1000)

    zero = np.zeros((n, 3))
    pen_before = pen_proxy(zero, probe)
    raw = np.zeros_like(zero)
    for f in scored_local:
        raw[f] = solve(f, zero[f])[0]
    smooth = gaussian_filter1d(raw, args.sigma, axis=0, mode="nearest")
    # smoothing spreads each shift over neighbours but lowers its peak: keep at least the raw shift
    # along its own direction on frames that needed one
    if not args.temporal_only:
        for f in range(n):
            if np.linalg.norm(raw[f]) > 0:
                u = raw[f] / np.linalg.norm(raw[f])
                deficit = np.linalg.norm(raw[f]) - smooth[f] @ u
                if deficit > 0:
                    smooth[f] = smooth[f] + deficit * u
    final = cap_displacement(smooth, max_shift_m)
    residual_fix = 0
    if not args.temporal_only:
        for f in scored_local:
            extra, _ = solve(f, final[f])
            if np.linalg.norm(extra) > 0:
                final[f] = cap_displacement(final[f] + extra, max_shift_m)
                residual_fix += 1
    pen_after = pen_proxy(final, probe)
    holdout_before = pen_proxy(zero, holdout) if len(holdout) else 0.0
    holdout_after = pen_proxy(final, holdout) if len(holdout) else 0.0
    if pen_after >= pen_before or (len(holdout) and holdout_after > holdout_before):
        raise RuntimeError(f"PEN correction rejected: fit {pen_before:.5f} -> {pen_after:.5f}, "
                           f"holdout {holdout_before:.5f} -> {holdout_after:.5f} cm")

    out = W / "poses_depen"
    out.mkdir(exist_ok=True)
    for f in range(n):
        d = dict(poses[f][2])
        d["translation"] = (t[f] + final[f]).tolist()
        json.dump(d, open(out / f"{f:06d}.json", "w"), indent=1)

    def acc(x, indices):
        consecutive = (indices[2:] - indices[1:-1] == 1) & (indices[1:-1] - indices[:-2] == 1)
        if not consecutive.any():
            return 0.0
        delta = x[:-2] - 2 * x[1:-1] + x[2:]
        return float(np.linalg.norm(delta[consecutive], axis=1).mean() * 100)
    sc = scored_local
    mag = np.linalg.norm(final, axis=1) * 100
    rep = {"episode": args.episode, "pen_proxy_before_cm": pen_before, "pen_proxy_after_cm": pen_after,
           "pen_holdout_before_cm": holdout_before, "pen_holdout_after_cm": holdout_after,
           "frames_shifted": int((mag > 0).sum()), "shift_cm_median_of_shifted": float(np.median(mag[mag > 0])) if (mag > 0).any() else 0.0,
           "shift_cm_max": float(mag.max()), "residual_fix_frames": residual_fix,
            "acc_o_before": acc(t[sc], sc), "acc_o_after": acc((t + final)[sc], sc),
           "scored_frame_count": len(scored_local), "solver_hand_points": len(probe),
           "holdout_hand_points": len(holdout),
           "params": {**vars(args), "spans": str(args.spans)},
           "seconds": round(time.time() - t0)}
    json.dump(rep, open(W / "track" / "depen.json", "w"), indent=1)
    log(f"{ep}: stage 8b — PEN fit {pen_before:.4f} -> {pen_after:.4f} cm, "
        f"holdout {holdout_before:.4f} -> {holdout_after:.4f} cm; {rep['frames_shifted']}/{n} frames "
        f"shifted (median {rep['shift_cm_median_of_shifted']:.2f} cm, max {rep['shift_cm_max']:.2f} cm); object "
        f"accel {rep['acc_o_before']:.3f} -> {rep['acc_o_after']:.3f} cm/frame^2 ({rep['seconds']}s)")


if __name__ == "__main__":
    main()
