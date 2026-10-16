#!/usr/bin/env python3
"""Export the depth-aligned WiLoR hands as a body-placement witness.

WiLoR hands are aligned against the same MoGe depth that FoundationPose uses to
place the object, so they are an independent measurement of where the hands are
in metric camera space. track1/qa_episode.py compares them with the MHR hands.

Reads only REAL detections (`wilor_aligned/`, never the interpolated frames) and
writes `<ep_dir>/body/wilor_witness.npz`:

    frame     [N]        video frame index (before --frame_offset is added)
    is_right  [N] bool
    wrist     [N, 3]     MANO joint 0, OpenCV camera frame, metres
    centroid  [N, 3]     mean of the 778 MANO vertices
    verts     [N, 778, 3]

Run with the hamer venv (manotorch + v2d.hamer):

    /opt/venvs/hamer/bin/python track1/wilor_witness.py \
        --aligned_dir out/episode_000014/wilor_aligned \
        --mano_assets_root weights/wilor/pretrained_models \
        --out out/episode_000014/body/wilor_witness.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
from v2d.hamer.lib.render_hands_aligned_video import _mano_layer, _rec_cam_t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aligned_dir", required=True)
    ap.add_argument("--mano_assets_root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.aligned_dir, "*", "*.json")))
    mano = _mano_layer(args.mano_assets_root)
    frames, is_right, wrists, centroids, verts_all = [], [], [], [], []
    for path in files:
        rec = json.load(open(path))
        if rec.get("interpolated", False):
            continue
        pose_aa = np.concatenate([np.asarray(rec["mano"]["global_orient"], np.float32),
                                  np.asarray(rec["mano"]["hand_pose"], np.float32)])
        betas = np.asarray(rec["mano"]["betas"], np.float32)
        with torch.no_grad():
            out = mano(torch.from_numpy(pose_aa)[None], torch.from_numpy(betas)[None])
        v = out.verts[0].numpy().astype(np.float64)
        j0 = out.joints[0, 0].numpy().astype(np.float64)
        if not rec["is_right"]:
            v[:, 0] *= -1
            j0[0] *= -1
        # Same transform as render_hands_aligned_video._build_hand_mesh.
        scale = float(rec.get("hand_scale", 1.0))
        c = v.mean(0, keepdims=True)
        v = (v - c) * scale + c
        j0 = (j0 - c[0]) * scale + c[0]
        t = _rec_cam_t(rec, False)
        frames.append(int(rec["frame_idx"]))
        is_right.append(bool(rec["is_right"]))
        wrists.append(j0 + t)
        centroids.append(v.mean(0) + t)
        verts_all.append((v + t).astype(np.float32))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        frame=np.asarray(frames, np.int64),
        is_right=np.asarray(is_right, bool),
        wrist=np.asarray(wrists, np.float32).reshape(-1, 3),
        centroid=np.asarray(centroids, np.float32).reshape(-1, 3),
        verts=np.asarray(verts_all, np.float32).reshape(-1, 778, 3),
    )
    print(f"wrote {args.out}: {len(frames)} real hand detections "
          f"({sum(is_right)} right, {len(frames) - sum(is_right)} left)")


if __name__ == "__main__":
    main()
