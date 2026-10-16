#!/usr/bin/env python3
"""Trajectory review plot: raw vs smoothed object track, and the submitted body root.

One PNG, four panels, over the scored span (video frame numbers on the x axis):
  1. object translation x/y/z (m): raw FoundationPose (thin) vs smoothed (thick)
  2. object |2nd difference| (cm/frame^2), raw vs smoothed — spikes = jumps / resets
  3. object rotation change per frame (deg), raw vs smoothed — flips and re-registration
  4. body centre (mean of MHR joints) x/y/z (m) and the object's z, so body/object depth agree

Run with the sam3d_body venv:

    /opt/venvs/sam3d_body/bin/python track1/viz_trajectory.py --episode 14 \
        --work_dir out/episode_000014 --sub_dir submission/episodes \
        --spans submission/scored_spans.json \
        --model weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt \
        --out logs/latest/review/episode_000014_trajectory.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
KIT_TRACK1 = paths.KIT_TRACK1
sys.path.insert(0, str(KIT_TRACK1))
from mesh_to_mhr_params import MHR  # noqa: E402


def quat_wxyz_to_rot(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def load_poses(d: str, offset: int, frames: np.ndarray):
    R = np.full((len(frames), 3, 3), np.nan)
    t = np.full((len(frames), 3), np.nan)
    for k, f in enumerate(frames):
        p = os.path.join(d, f"{f - offset:06d}.json")
        if os.path.exists(p):
            j = json.load(open(p))
            R[k] = quat_wxyz_to_rot(j["rotation"])
            t[k] = np.asarray(j["translation"], float).reshape(3)
    return R, t


def rot_step_deg(R: np.ndarray) -> np.ndarray:
    out = np.full(len(R), np.nan)
    for k in range(1, len(R)):
        if np.isfinite(R[k]).all() and np.isfinite(R[k - 1]).all():
            c = (np.trace(R[k - 1].T @ R[k]) - 1) / 2
            out[k] = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return out


def accel_cm(t: np.ndarray) -> np.ndarray:
    a = np.full(len(t), np.nan)
    a[1:-1] = np.linalg.norm(t[:-2] - 2 * t[1:-1] + t[2:], axis=1) * 100
    return a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--sub_dir", required=True)
    ap.add_argument("--spans", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--frame_offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    span = json.load(open(args.spans))[str(args.episode)]
    frames = np.arange(span["first"], span["last"] + 1)
    R_raw, t_raw = load_poses(os.path.join(args.work_dir, "poses"), args.frame_offset, frames)
    R_s, t_s = load_poses(os.path.join(args.work_dir, "poses_smoothed"), args.frame_offset, frames)

    sub = np.load(os.path.join(args.sub_dir, f"episode_{args.episode:06d}.npz"))
    mhr = MHR(args.model, "cuda" if torch.cuda.is_available() else "cpu", chunk=32, precision="float32")
    pose = torch.as_tensor(sub["pose"][frames], dtype=torch.float64, device=mhr.device)
    z = torch.as_tensor(np.concatenate([sub["scales"], sub["shape"]])[None], dtype=torch.float64,
                        device=mhr.device)
    joints = mhr.run(pose, z)[1].cpu().numpy()
    pelvis = joints.mean(1)  # joint 0 is a fixed root; the joint centroid tracks the body

    fig, ax = plt.subplots(4, 1, figsize=(16, 15), sharex=True)
    colors = ("tab:red", "tab:green", "tab:blue")
    for i, c in enumerate("xyz"):
        ax[0].plot(frames, t_raw[:, i], color=colors[i], lw=0.8, alpha=0.6, label=f"raw {c}")
        ax[0].plot(frames, t_s[:, i], color=colors[i], lw=2.2, label=f"smoothed {c}")
    ax[0].set_ylabel("object translation (m)")
    ax[0].legend(ncol=6, fontsize=9)

    a_raw, a_s = accel_cm(t_raw), accel_cm(t_s)
    ax[1].plot(frames, a_raw, color="gray", lw=0.8, label=f"raw (mean {np.nanmean(a_raw):.2f})")
    ax[1].plot(frames, a_s, color="tab:orange", lw=1.8, label=f"smoothed (mean {np.nanmean(a_s):.2f})")
    ax[1].axhline(0.13, color="k", ls="--", lw=1, label="typical GT body accel 0.13")
    ax[1].set_ylabel("object |2nd diff| (cm/frame²)")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-3, max(20.0, float(np.nanmax(a_raw)) if np.isfinite(a_raw).any() else 20.0))
    ax[1].legend(fontsize=9)

    ax[2].plot(frames, rot_step_deg(R_raw), color="gray", lw=0.8, label="raw")
    ax[2].plot(frames, rot_step_deg(R_s), color="tab:purple", lw=1.8, label="smoothed")
    ax[2].set_ylabel("object rotation step (deg/frame)")
    ax[2].legend(fontsize=9)

    for i, c in enumerate("xyz"):
        ax[3].plot(frames, pelvis[:, i], color=colors[i], lw=1.8, label=f"body centre {c}")
    ax[3].plot(frames, t_s[:, 2], color="tab:orange", lw=1.2, ls="--", label="object z (smoothed)")
    ax[3].set_ylabel("metres (camera frame)")
    ax[3].set_xlabel("video frame")
    ax[3].legend(ncol=4, fontsize=9)

    fig.suptitle(f"episode_{args.episode:06d} trajectories over the scored span "
                 f"({span['first']}..{span['last']})", fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=80)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
