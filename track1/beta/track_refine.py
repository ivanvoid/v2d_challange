#!/usr/bin/env python3
"""Beta stage 7 (part 2) — refine the smoothed object poses with CoTracker point tracks.

stage7_smooth.py only sees the per-frame FoundationPose poses; the point tracks (beta/point_tracks.py) add
direct frame-to-frame evidence of how the object surface moves, which pins down the in-plane motion and
rotation between observed frames, and through occlusions while some points stay visible.

1. inputs   stage 7's poses_smoothed/ is moved to poses_smoothed_s7/ (once per stage-7 run) and read from there;
            mesh_scaled.obj, intrinsics_stable.json, track/smooth.json (weights), track/points.npz.
2. anchor   per window, the tracked points are lifted to the object surface at the query frame q:
            surface samples of the mesh are posed with the stage-7 pose of q, z-buffered, and each query pixel
            takes the nearest front-surface sample (<= --anchor_px). Tracks whose median reprojection error
            under the stage-7 poses exceeds --max_track_px (hand, background, drift) are dropped.
3. solve    per-frame corrections (rotation vector d_r, translation d_t) minimising
                sum_obs huber(|proj(K (Exp(d_r_f) R_f p + t_f + d_t_f)) - uv| / sigma_px)
              + sum_f w_f (|d_t_f|^2 / sigma_t^2 + |d_r_f|^2 / sigma_r^2)        (stay near stage 7)
              + lam_acc * sum_f (|dd d_t_f|^2 / sigma_at^2 + |dd d_r_f|^2 / sigma_ar^2)   (stay smooth)
            with L-BFGS. Frames without tracks just follow the prior and smoothness terms.
4. output   poses_smoothed/ (same JSON as stage 7) and track/refine.json: reprojection error before/after,
            corrections (cm, deg), acceleration before/after. If the refinement does not lower the median
            reprojection error the stage-7 poses are written unchanged (refine.json says so).

    /opt/venvs/sam3d_body/bin/python track1/beta/track_refine.py --episode 6 --out_root out_beta/<tag>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logutil import log  # noqa: E402


def load_poses(d: Path, n: int):
    R, t = np.zeros((n, 3, 3)), np.zeros((n, 3))
    for f in range(n):
        p = json.load(open(d / f"{f:06d}.json"))
        q = np.asarray(p["rotation"], float)
        R[f] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        t[f] = p["translation"]
    return R, t


def save_poses(d: Path, R: np.ndarray, t: np.ndarray) -> None:
    tmp = d.with_name(d.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    q = Rotation.from_matrix(R).as_quat()
    for f in range(len(R)):
        json.dump({"rotation": [float(q[f, 3]), float(q[f, 0]), float(q[f, 1]), float(q[f, 2])],
                   "translation": [float(x) for x in t[f]], "scale": [1.0, 1.0, 1.0]},
                  open(tmp / f"{f:06d}.json", "w"), indent=1)
    shutil.rmtree(d, ignore_errors=True)
    os.replace(tmp, d)


def project(X: np.ndarray, K: dict) -> np.ndarray:
    z = np.maximum(X[..., 2], 1e-6)
    return np.stack([K["fx"] * X[..., 0] / z + K["cx"], K["fy"] * X[..., 1] / z + K["cy"]], -1)


def anchors(P: np.ndarray, Nrm: np.ndarray, R: np.ndarray, t: np.ndarray, K: dict, uvq: np.ndarray,
            shape: tuple, max_px: float) -> np.ndarray:
    """Object-frame surface points under the query pixels uvq (NaN where none is in front)."""
    X = P @ R.T + t
    front = (Nrm @ R.T * X).sum(-1) < 0
    uv = project(X, K)
    H, W = shape
    ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (X[:, 2] > 0)
    zb = np.full((H, W), np.inf)
    np.minimum.at(zb, (vi[ok], ui[ok]), X[ok, 2])
    tol = 0.01 * float(np.ptp(P, 0).max())
    ok[ok] &= X[ok, 2] <= zb[vi[ok], ui[ok]] + tol
    out = np.full((len(uvq), 3), np.nan)
    if ok.sum() == 0:
        return out
    d, i = cKDTree(uv[ok]).query(uvq, distance_upper_bound=max_px)
    hit = np.isfinite(d)
    out[hit] = P[ok][i[hit]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--samples", type=int, default=200000)
    ap.add_argument("--anchor_px", type=float, default=2.0)
    ap.add_argument("--max_track_px", type=float, default=15.0)
    ap.add_argument("--sigma_px", type=float, default=2.0)
    ap.add_argument("--sigma_t", type=float, default=0.02, help="m")
    ap.add_argument("--sigma_r", type=float, default=3.0, help="deg")
    ap.add_argument("--sigma_at", type=float, default=0.002, help="m / frame^2")
    ap.add_argument("--sigma_ar", type=float, default=0.3, help="deg / frame^2")
    ap.add_argument("--lam_acc", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()

    import cv2
    import torch
    import trimesh

    t0 = time.time()
    W = Path(a.out_root).resolve() / f"episode_{a.episode:06d}"
    ps, s7, smooth_p = W / "poses_smoothed", W / "poses_smoothed_s7", W / "track" / "smooth.json"
    if not s7.exists() or s7.stat().st_mtime < smooth_p.stat().st_mtime:
        shutil.rmtree(s7, ignore_errors=True)
        os.replace(ps, s7)                       # fresh stage-7 output becomes the refinement input
    K = json.load(open(W / "intrinsics_stable.json"))
    sm = json.load(open(smooth_p))["frames"]
    n = len(sm)
    R0, t0p = load_poses(s7, n)
    wf = np.array([0.05 if f.get("hidden") or f.get("outlier") else max(f.get("weight", 1.0), 0.05) for f in sm])

    tm = trimesh.load(W / "mesh_scaled.obj", force="mesh", process=False)
    P, fi = trimesh.sample.sample_surface(tm, a.samples, seed=0)
    Nrm = tm.face_normals[fi]
    H, Wd = cv2.imread(str(W / "frames" / f"{0:06d}.png")).shape[:2]

    tr = np.load(W / "track" / "points.npz")
    obs_f, obs_p, obs_uv = [], [], []
    n_tracks = n_kept = 0
    for k, q in enumerate(tr["q"]):
        st, uv, vis = int(tr["start"][k]), tr["uv"][k], tr["vis"][k]
        r_q = q - st
        valid = np.isfinite(uv[r_q, :, 0]) & vis[r_q]
        p = anchors(P, Nrm, R0[q], t0p[q], K, np.nan_to_num(uv[r_q]), (H, Wd), a.anchor_px)
        valid &= np.isfinite(p[:, 0])
        rows = np.arange(len(uv))
        f = st + rows
        rin = (f >= 0) & (f < n)
        for i in np.nonzero(valid)[0]:
            n_tracks += 1
            m = rin & vis[:, i] & np.isfinite(uv[:, i, 0])
            fr = f[m]
            pred = project(np.einsum("fij,j->fi", R0[fr], p[i]) + t0p[fr], K)
            if not len(fr) or np.median(np.linalg.norm(pred - uv[m, i], axis=-1)) > a.max_track_px:
                continue
            n_kept += 1
            obs_f.append(fr)
            obs_p.append(np.repeat(p[i][None], len(fr), 0))
            obs_uv.append(uv[m, i])
    if not obs_f:
        log("  [skip] track refine: no usable point tracks — stage-7 poses kept")
        save_poses(ps, R0, t0p)
        json.dump({"applied": False, "reason": "no usable tracks", "n_tracks": n_tracks},
                  open(W / "track" / "refine.json", "w"), indent=1)
        return
    of, op, ouv = np.concatenate(obs_f), np.concatenate(obs_p), np.concatenate(obs_uv)
    per_frame = np.bincount(of, minlength=n)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = lambda x: torch.as_tensor(x, dtype=torch.float64, device=dev)  # noqa: E731
    R0t, t0t, oft, opt_, ouvt, wft = T(R0), T(t0p), torch.as_tensor(of, device=dev), T(op), T(ouv), T(wf)
    norm = T(1.0 / np.sqrt(np.maximum(per_frame, 1)))[oft]    # each frame's tracks count ~sqrt(n), not n
    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
    sr, sar = np.radians(a.sigma_r), np.radians(a.sigma_ar)

    def exp_so3(w):
        th = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-18)
        k = w / th
        Kx = torch.zeros(w.shape[0], 3, 3, dtype=w.dtype, device=w.device)
        Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
        Kx = Kx - Kx.transpose(1, 2)
        s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
        return torch.eye(3, dtype=w.dtype, device=w.device) + s * Kx + (1 - c) * Kx @ Kx

    def residual_px(dr, dt):
        R = exp_so3(dr) @ R0t
        X = torch.einsum("nij,nj->ni", R[oft], opt_) + (t0t + dt)[oft]
        z = X[:, 2].clamp_min(1e-6)
        uv = torch.stack([fx * X[:, 0] / z + cx, fy * X[:, 1] / z + cy], -1)
        return (uv - ouvt).norm(dim=-1)

    def dd(x):
        return x[2:] - 2 * x[1:-1] + x[:-2]

    dr = torch.zeros(n, 3, dtype=torch.float64, device=dev, requires_grad=True)
    dt = torch.zeros(n, 3, dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([dr, dt], lr=1.0, max_iter=a.iters, line_search_fn="strong_wolfe")

    def loss():
        opt.zero_grad()
        e = residual_px(dr, dt) / a.sigma_px
        hub = torch.where(e < 1, 0.5 * e * e, e - 0.5)
        L = (norm * hub).sum() \
            + 0.5 * (wft * ((dt * dt).sum(-1) / a.sigma_t ** 2 + (dr * dr).sum(-1) / sr ** 2)).sum() \
            + 0.5 * a.lam_acc * ((dd(dt) ** 2).sum() / a.sigma_at ** 2 + (dd(dr) ** 2).sum() / sar ** 2)
        L.backward()
        return L

    with torch.no_grad():
        before = residual_px(torch.zeros_like(dr), torch.zeros_like(dt)).cpu().numpy()
    opt.step(loss)
    with torch.no_grad():
        after = residual_px(dr, dt).cpu().numpy()
        Rn = (exp_so3(dr) @ R0t).cpu().numpy()
        tn = (t0t + dt).cpu().numpy()
    corr_cm = np.linalg.norm(tn - t0p, axis=-1) * 100
    corr_deg = np.degrees(np.linalg.norm(dr.detach().cpu().numpy(), axis=-1))
    acc = lambda x: float(np.median(np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=-1)) * 100)  # noqa: E731
    applied = bool(np.median(after) < np.median(before))
    save_poses(ps, Rn if applied else R0, tn if applied else t0p)
    pf_b = np.bincount(of, weights=before, minlength=n) / np.maximum(per_frame, 1)
    pf_a = np.bincount(of, weights=after, minlength=n) / np.maximum(per_frame, 1)
    rep = {"applied": applied, "params": vars(a), "seconds": round(time.time() - t0, 1),
           "n_windows": int(len(tr["q"])), "n_tracks": n_tracks, "n_tracks_kept": n_kept, "n_obs": int(len(of)),
           "frames_with_tracks": int((per_frame > 0).sum()), "n_frames": n,
           "reproj_px_median_before": float(np.median(before)), "reproj_px_median_after": float(np.median(after)),
           "corr_cm_median": float(np.median(corr_cm)), "corr_cm_max": float(corr_cm.max()),
           "corr_deg_median": float(np.median(corr_deg)), "corr_deg_max": float(corr_deg.max()),
           "acc_cm_median_before": acc(t0p), "acc_cm_median_after": acc(tn),
           "frames": [{"frame": f, "n_obs": int(per_frame[f]),
                       "reproj_px_before": round(float(pf_b[f]), 2) if per_frame[f] else None,
                       "reproj_px_after": round(float(pf_a[f]), 2) if per_frame[f] else None,
                       "corr_cm": round(float(corr_cm[f]), 3), "corr_deg": round(float(corr_deg[f]), 3)}
                      for f in range(n)]}
    json.dump(rep, open(W / "track" / "refine.json", "w"), indent=1)
    log(f"  [done] track refine ({'applied' if applied else 'NOT applied: no gain'}): {n_kept}/{n_tracks} tracks, "
        f"{rep['frames_with_tracks']}/{n} frames; reprojection {rep['reproj_px_median_before']:.2f} -> "
        f"{rep['reproj_px_median_after']:.2f} px; correction median {rep['corr_cm_median']:.2f} cm / "
        f"{rep['corr_deg_median']:.2f} deg (max {rep['corr_cm_max']:.1f} cm / {rep['corr_deg_max']:.1f} deg); "
        f"acc {rep['acc_cm_median_before']:.3f} -> {rep['acc_cm_median_after']:.3f} cm/frame^2")


if __name__ == "__main__":
    main()
