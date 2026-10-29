#!/usr/bin/env python3
"""Lock an object that stands still for part of a video to ONE pose over that part.

Generalises stationary_stool.py (whole video, episodes 24/26, production poses) to any episode, any
leading still interval, and any assembled episodes dir. The still interval ends where the raw
tracker leaves its starting position by more than --onset_cm for --hold frames (or --until).
Inside it: anchors = clear frames spread over the interval; the constant pose is the anchor pose
that best explains the masks of the OTHER anchors (median IoU); accepted only if the remaining
clear frames (held out) lose <= --max_iou_drop IoU and the lock moves the object <= --max_shift_cm.
After the interval the pose blends back into the existing trajectory over --blend frames.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

import paths
from stationary_stool import RENDER_DOWNSAMPLE, accept_stationary_pose, choose_constant_pose, mask_iou, rendered_mask


def still_until(raw_t: np.ndarray, onset_cm: float, hold: int) -> int:
    """Last local index of the leading still interval (raw tracker translations)."""
    ref = np.median(raw_t[:min(60, len(raw_t))], 0)
    moved = np.linalg.norm(raw_t - ref, axis=1) * 100 > onset_cm
    run = np.convolve(moved.astype(int), np.ones(hold, int), 'valid') >= hold
    first = int(np.flatnonzero(run)[0]) if run.any() else len(raw_t)
    return max(0, first - 1)


def anchors_in(states: list[dict], idx: list[int], n_bins: int = 12) -> tuple[list[int], list[int]]:
    area = np.array([states[i]['area_frac'] for i in idx], float)
    eligible = [i for i in idx if states[i]['state'] == 'observed' and states[i]['occlusion'] <= 0.10
                and states[i]['iou'] >= 0.6 and states[i]['area_frac'] >= np.percentile(area, 50)]
    if len(eligible) < 15:
        raise ValueError(f'not enough clear frames in the still interval ({len(eligible)})')
    bins = np.linspace(idx[0], idx[-1] + 1, n_bins + 1)
    anchors = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        fr = [i for i in eligible if lo <= i < hi]
        if fr:
            anchors.append(max(fr, key=lambda i: (states[i]['quality'], states[i]['iou'])))
    holdout = [i for i in eligible if i not in anchors]
    if len(anchors) < 5 or len(holdout) < 10:
        raise ValueError('too few anchors / held-out frames')
    return anchors, holdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--episode', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--until', type=int, default=None, help='last still VIDEO frame (default: detect)')
    ap.add_argument('--onset_cm', type=float, default=4.0)
    ap.add_argument('--hold', type=int, default=10)
    ap.add_argument('--margin', type=int, default=10, help='frames kept free before the detected onset')
    ap.add_argument('--blend', type=int, default=15)
    ap.add_argument('--max_iou_drop', type=float, default=0.05)
    ap.add_argument('--max_shift_cm', type=float, default=8.0)
    args = ap.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((paths.DATA / 'experiments').resolve()):
        raise SystemExit('--out must be under V2D_DATA/experiments')
    stem = f'episode_{args.episode:06d}'
    w = paths.OUT / stem
    meta = json.loads((w / 'track1_meta.json').read_text())
    off, n = meta['frame_offset'], meta['n_frames']
    states = json.loads((w / 'track/states.json').read_text())['frames']
    raw_t = np.array([json.loads((w / 'poses' / f'{i:06d}.json').read_text())['translation'] for i in range(n)])
    d = dict(np.load(args.src / f'{stem}.npz'))
    n_rows = len(d['object_translation'])
    last = (args.until - off) if args.until is not None else still_until(raw_t, args.onset_cm, args.hold) - args.margin
    last = min(last, n_rows - off - 1)
    idx = list(range(0, last + 1))
    anchors, holdout = anchors_in(states, idx)
    mesh = trimesh.load(args.src / f'{stem}_object.obj', force='mesh', process=False)
    intr = json.loads((w / 'intrinsics_stable.json').read_text())
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]], float)
    hw = (int(intr['height']) // RENDER_DOWNSAMPLE, int(intr['width']) // RENDER_DOWNSAMPLE)
    observed = {}
    for i in anchors + holdout:
        m = cv2.imread(str(w / 'masks/1' / f'{i:06d}.png'), cv2.IMREAD_GRAYSCALE)
        observed[i] = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) > 0
    rot = d['object_rotation'][off:off + last + 1].astype(float)
    tra = d['object_translation'][off:off + last + 1].astype(float)
    s = float(d['object_scale'])
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    choice, scores = choose_constant_pose(rot, tra, anchors, observed, V, F, s, K, hw)
    Rc, tc = rot[choice], tra[choice]
    orig = np.median([mask_iou(rendered_mask(V, F, rot[i], tra[i], s, K, hw), observed[i]) for i in holdout])
    stat = np.median([mask_iou(rendered_mask(V, F, Rc, tc, s, K, hw), observed[i]) for i in holdout])
    shift = float(np.linalg.norm(tra - tc, axis=1).max() * 100)
    ok = accept_stationary_pose(orig, stat, shift, max_iou_drop=args.max_iou_drop, max_allowed_shift_cm=args.max_shift_cm)
    rep = {'episode': args.episode, 'still_video_frames': [off, off + last], 'anchors': len(anchors), 'holdout': len(holdout),
           'chosen_anchor_video_frame': choice + off, 'holdout_iou_original': float(orig), 'holdout_iou_static': float(stat),
           'max_shift_cm': shift, 'accepted': bool(ok)}
    dest = out / 'episodes'
    dest.mkdir(parents=True, exist_ok=True)
    for f in args.src.glob('episode_*'):
        shutil.copy2(f, dest / f.name)
    if ok:
        T, R = d['object_translation'].copy(), d['object_rotation'].copy()
        T[off:off + last + 1] = tc
        R[off:off + last + 1] = Rc
        for k in range(1, args.blend + 1):                    # ease back into the moving trajectory
            r = off + last + k
            if r >= n_rows:
                break
            a = 0.5 - 0.5 * np.cos(np.pi * k / (args.blend + 1))
            T[r] = (1 - a) * tc + a * d['object_translation'][r]
            rv = (Rotation.from_matrix(Rc).inv() * Rotation.from_matrix(d['object_rotation'][r].astype(float))).as_rotvec()
            R[r] = (Rotation.from_matrix(Rc) * Rotation.from_rotvec(a * rv)).as_matrix()
        d['object_translation'], d['object_rotation'] = T.astype(d['object_translation'].dtype), R.astype(d['object_rotation'].dtype)
        np.savez(dest / f'{stem}.npz', **d)
    (out / f'{stem}_static_lock.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(f"{stem}: still video frames {off}..{off + last}; held-out IoU {orig:.3f} -> {stat:.3f}; "
          f"max shift {shift:.1f} cm; {'ACCEPT' if ok else 'REJECT'}")


if __name__ == '__main__':
    main()
