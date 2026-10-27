#!/usr/bin/env python3
"""Assemble one Track 1 episode for the submission kit.

Combines
  * body:   <ep_dir>/body/mhr_fit.npz          (track1/fit_mhr.py)
  * object: <ep_dir>/mesh_scaled.obj + <ep_dir>/poses_smoothed/<frame:06d>.json
into the files `v2d_submission_kit/tools/pack_reconstruction.py` reads:

  <out_dir>/episode_XXXXXX.npz          pose [T,136], scales [68], shape [45],
                                        object_rotation [T,3,3], object_translation [T,3],
                                        object_scale ()
  <out_dir>/episode_XXXXXX_object.obj   geometry-only copy of the object mesh

Body and object are both in the OpenCV camera frame in metres (SAM3D-Body
`pred_vertices` and FoundationPose share it), which is also the Track 1 frame,
so no axis change is applied to either.

Run with the submission venv:

    .venv-submit/bin/python track1/build_episode.py --episode 14 \
        --ep_dir out/episode_000014 --out_dir submission/episodes
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import trimesh


def quat_wxyz_to_rot(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def load_object_track(poses_dir: Path, n_frames: int, frame_offset: int):
    """Per-VIDEO-frame R [n,3,3], t [n,3], uniform scale, and which frames had a pose."""
    files = sorted(glob.glob(str(poses_dir / "*.json")))
    if not files:
        raise SystemExit(f"build_episode: no object poses in {poses_dir}")
    R = np.full((n_frames, 3, 3), np.nan)
    t = np.full((n_frames, 3), np.nan)
    scales = []
    for f in files:
        frame = int(Path(f).stem) + frame_offset
        if not 0 <= frame < n_frames:
            continue
        with open(f) as handle:
            d = json.load(handle)
        R[frame] = quat_wxyz_to_rot(d["rotation"])
        t[frame] = np.asarray(d["translation"], float).reshape(3)
        scales.append(np.asarray(d.get("scale", [1.0]), float).reshape(-1))
    have = np.isfinite(t).all(1)
    if not have.any():
        raise SystemExit(f"build_episode: object poses in {poses_dir} cover none of frames 0..{n_frames - 1}")

    s = np.concatenate([np.broadcast_to(v, 3) if v.size == 1 else v for v in scales]).reshape(-1, 3)
    if np.ptp(s) > 1e-6:
        raise SystemExit(f"build_episode: object pose scale is not one uniform value ({s.min()}..{s.max()})")

    good = np.flatnonzero(have)
    nearest = good[np.abs(np.arange(n_frames)[:, None] - good[None]).argmin(1)]
    return R[nearest], t[nearest], float(s[0, 0]), have


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--ep_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--body_fit", type=Path, default=None, help="default <ep_dir>/body/mhr_fit.npz")
    ap.add_argument("--object_mesh", type=Path, default=None, help="default <ep_dir>/mesh_scaled.obj")
    ap.add_argument("--object_poses", type=Path, default=None, help="default <ep_dir>/poses_smoothed")
    ap.add_argument("--frame_offset", type=int, default=0,
                    help="video frame index of pose file 000000 (non-zero for trimmed inputs)")
    args = ap.parse_args()

    body_fit = args.body_fit or args.ep_dir / "body" / "mhr_fit.npz"
    mesh_path = args.object_mesh or args.ep_dir / "mesh_scaled.obj"
    poses_dir = args.object_poses or args.ep_dir / "poses_smoothed"

    body = np.load(body_fit)
    pose = np.asarray(body["pose"], np.float32)
    n_frames = len(pose)

    R, t, scale, have = load_object_track(poses_dir, n_frames, args.frame_offset)
    if not scale > 0:
        raise SystemExit(f"build_episode: non-positive object scale {scale}")

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if len(mesh.faces) == 0:
        raise SystemExit(f"build_episode: {mesh_path} has no faces")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"episode_{args.episode:06d}"
    np.savez(
        f"{stem}.npz",
        pose=pose,
        scales=np.asarray(body["scales"], np.float32),
        shape=np.asarray(body["shape"], np.float32),
        object_rotation=R.astype(np.float32),
        object_translation=t.astype(np.float32),
        object_scale=np.float32(scale),
        object_pose_observed=have,
    )
    plain = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=False)
    plain.export(f"{stem}_object.obj")
    print(f"wrote {stem}.npz (T={n_frames}, object poses observed {int(have.sum())}/{n_frames}, "
          f"scale {scale:g}) and {os.path.basename(str(stem))}_object.obj "
          f"({len(plain.vertices)} v, {len(plain.faces)} f)")


if __name__ == "__main__":
    main()
