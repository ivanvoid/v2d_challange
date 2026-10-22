#!/usr/bin/env python3
"""Stage 7 — smooth the object trajectory over the whole video at once (pose-graph smoothing).

Input: out/episode_XXXXXX/poses/<frame:06d>.json (stage 6) and track/states.json (per-frame fit).
Output: out/episode_XXXXXX/poses_smoothed/<frame:06d>.json, track/smooth.json.

0. hidden    frames stage 6 labelled occluded or lost carry no evidence (the tracker only guessed): they get
             ~zero weight, and the local trend used below is built from the other frames only (hidden
             frames replaced by interpolation), so a long hidden stretch cannot vote for its own garbage.
             Their output is an interpolation between the last and next visible frames: recorded as
             "interpolated" in track/smooth.json. --keep_hidden restores the old behaviour.
1. outliers  a raw pose is an outlier when it disagrees with a robust local trend
             (median over +-7 frames): translation > --out_cm from the median, or rotation
             > --out_deg from the local rotation median. Outliers get weight ~0 (they are
             interpolated by the smoother instead of being followed).
2. weights   w_f = clip((IoU_f - 0.3) / 0.4, 0.05, 1)    (low-fit frames pull less)
3. solve     translation, per axis, minimise over the whole sequence
                 sum_f w_f (x_f - raw_f)^2  +  lam_t * sum_f (x_{f-1} - 2 x_f + x_{f+1})^2
             (second differences = acceleration, what ACC-O scores); one sparse linear solve.
             rotation: the same on the rotation vector relative to a slowly varying reference
             (raw rotations are first made sign-consistent as quaternions, then expressed as
             small rotation vectors relative to a heavily smoothed reference, smoothed with
             lam_r, and composed back) — exact on SO(3) for the small residuals involved.
4. report    per-frame correction (cm, deg), acceleration before/after, outliers.

    /opt/venvs/sam3d_body/bin/python track1/stage7_smooth.py --episode 28 --out_root out
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import median_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logutil import log  # noqa: E402


def second_diff(n: int):
    return diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))


def smooth_series(x: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """argmin sum w (y - x)^2 + lam |D2 y|^2, column-wise."""
    n = len(x)
    D = second_diff(n)
    A = (diags(w) + lam * (D.T @ D)).tocsc()
    return np.stack([spsolve(A, w * x[:, k]) for k in range(x.shape[1])], 1)


def quat_wxyz(R: Rotation) -> np.ndarray:
    q = R.as_quat()                      # xyzw
    return np.c_[q[:, 3], q[:, :3]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--lam_t", type=float, default=50.0, help="translation acceleration weight")
    ap.add_argument("--lam_r", type=float, default=50.0, help="rotation acceleration weight")
    ap.add_argument("--out_cm", type=float, default=8.0)
    ap.add_argument("--out_deg", type=float, default=30.0)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--keep_hidden", action="store_true", help="old behaviour: hidden frames keep their IoU weight")
    args = ap.parse_args()

    ep = f"episode_{args.episode:06d}"
    W = os.path.join(os.path.abspath(args.out_root), ep)
    meta = json.load(open(os.path.join(W, "track1_meta.json")))
    n, off = meta["n_frames"], meta["frame_offset"]
    states = {r["frame"]: r for r in json.load(open(os.path.join(W, "track", "states.json")))["frames"]}

    raw = [json.load(open(os.path.join(W, "poses", f"{f:06d}.json"))) for f in range(n)]
    t = np.array([p["translation"] for p in raw], float)
    q = np.array([p["rotation"] for p in raw], float)          # wxyz
    for i in range(1, n):                                       # sign-consistent quaternions
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    R = Rotation.from_quat(np.c_[q[:, 1:], q[:, :1]])
    iou = np.array([states[f]["iou"] for f in range(n)])
    hidden = np.array([states[f].get("state") in ("occluded", "lost") for f in range(n)])
    if args.keep_hidden or hidden.all() or (~hidden).sum() < 2:
        hidden[:] = False

    # trend inputs: hidden frames replaced by interpolation between the visible ones (never used as data)
    t_fill, R_fill = t.copy(), R
    if hidden.any():
        good = np.flatnonzero(~hidden)
        t_fill = np.stack([np.interp(np.arange(n), good, t[good, k]) for k in range(3)], 1)
        R_fill = Slerp(good, R[good])(np.clip(np.arange(n), good[0], good[-1]))
    q_fill = np.c_[quat_wxyz(R_fill)]
    for i in range(1, n):
        if np.dot(q_fill[i], q_fill[i - 1]) < 0:
            q_fill[i] = -q_fill[i]

    # 1. outliers vs a robust local trend
    t_med = median_filter(t_fill, size=(args.window, 1), mode="nearest")
    q_med = median_filter(q_fill, size=(args.window, 1), mode="nearest")
    R_med = Rotation.from_quat(np.c_[q_med[:, 1:], q_med[:, :1]] /
                               np.linalg.norm(q_med, axis=1, keepdims=True))
    dt_cm = np.linalg.norm(t - t_med, axis=1) * 100
    dr_deg = np.degrees((R_med.inv() * R).magnitude())
    outlier = ((dt_cm > args.out_cm) | (dr_deg > args.out_deg)) & ~hidden

    # 2. weights
    w = np.clip((iou - 0.3) / 0.4, 0.05, 1.0)
    w[outlier] = 1e-4
    w[hidden] = 1e-6

    # 3a. translation
    t_s = smooth_series(t, w, args.lam_t)

    # 3b. rotation: residual rotation vectors w.r.t. a smooth reference, smoothed, composed back
    ref_key = np.linspace(0, n - 1, max(2, n // 10)).round().astype(int)
    ref_key = np.unique(ref_key)
    key_R = Rotation.from_quat(np.c_[q_med[ref_key, 1:], q_med[ref_key, :1]] /
                               np.linalg.norm(q_med[ref_key], axis=1, keepdims=True))
    R_ref = Slerp(ref_key, key_R)(np.arange(n))
    v = (R_ref.inv() * R).as_rotvec()
    v_s = smooth_series(v, w, args.lam_r)
    R_s = R_ref * Rotation.from_rotvec(v_s)

    out_dir = os.path.join(W, "poses_smoothed")
    os.makedirs(out_dir, exist_ok=True)
    qs = quat_wxyz(R_s)
    for f in range(n):
        json.dump({"rotation": qs[f].tolist(), "translation": t_s[f].tolist(), "scale": raw[f].get("scale", [1, 1, 1])},
                  open(os.path.join(out_dir, f"{f:06d}.json"), "w"), indent=1)

    # 4. report
    def acc(x):
        return np.linalg.norm(x[:-2] - 2 * x[1:-1] + x[2:], axis=1) * 100
    def rot_steps(Rs):
        return np.degrees((Rs[:-1].inv() * Rs[1:]).magnitude())
    corr_cm = np.linalg.norm(t_s - t, axis=1) * 100
    corr_deg = np.degrees((R.inv() * R_s).magnitude())
    rs_raw, rs_s = rot_steps(R), rot_steps(R_s)
    rep = {"episode": args.episode, "params": vars(args), "n_outliers": int(outlier.sum()), "n_hidden_interpolated": int(hidden.sum()),
           "outlier_video_frames": [int(f + off) for f in np.flatnonzero(outlier)],
           "acc_raw_cm": float(acc(t).mean()), "acc_smooth_cm": float(acc(t_s).mean()),
           "rot_step_raw_max": float(rs_raw.max()), "rot_step_smooth_max": float(rs_s.max()),
           "corr_cm_median": float(np.median(corr_cm)), "corr_cm_max": float(corr_cm.max()),
           "corr_deg_median": float(np.median(corr_deg)), "corr_deg_max": float(corr_deg.max()),
           "frames": [{"frame": f, "video_frame": f + off, "outlier": bool(outlier[f]), "interpolated": bool(hidden[f] or outlier[f]), "hidden": bool(hidden[f]), "weight": float(w[f]),
                       "corr_cm": float(corr_cm[f]), "corr_deg": float(corr_deg[f])} for f in range(n)]}
    json.dump(rep, open(os.path.join(W, "track", "smooth.json"), "w"), indent=1)
    log(f"{ep}: stage 7 OK — {n} poses; outliers {rep['n_outliers']} {rep['outlier_video_frames'][:10]}"
        f"; hidden frames interpolated {rep['n_hidden_interpolated']}"
        f"{'...' if rep['n_outliers'] > 10 else ''}; acceleration {rep['acc_raw_cm']:.3f} -> "
        f"{rep['acc_smooth_cm']:.3f} cm/frame^2; max rotation step {rep['rot_step_raw_max']:.0f} -> "
        f"{rep['rot_step_smooth_max']:.0f} deg; correction median {rep['corr_cm_median']:.1f} cm / "
        f"{rep['corr_deg_median']:.1f} deg, max {rep['corr_cm_max']:.1f} cm / {rep['corr_deg_max']:.0f} deg")


if __name__ == "__main__":
    main()
