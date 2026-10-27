#!/usr/bin/env python3
"""Isolated stationary-stool pose experiment on episodes 24 and 26.

Select one pose per video from distributed high-visibility anchors, compare it
with the original framewise poses on held-out frames, and write corrected packed
episode NPZs only under --trial. Episode 25 is deliberately left unchanged.
This changes neither the camera nor the body, mesh, or object scale.
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

EPISODES = (24, 26)
RENDER_DOWNSAMPLE = 4


def visible_indices(states: list[dict], *, n_bins: int = 12) -> tuple[list[int], list[int]]:
    """Distribute anchors over the video, reserving other eligible frames for validation."""
    area = np.array([s['area_frac'] for s in states], dtype=float)
    eligible = [i for i, s in enumerate(states) if s['state'] == 'observed'
                and s['occlusion'] <= 0.10 and s['iou'] >= 0.6
                and s['area_frac'] >= np.percentile(area, 60)]
    if len(eligible) < 15:
        raise ValueError(f'not enough clear frames for independent stationary fit ({len(eligible)})')
    bins = np.linspace(0, len(states), n_bins + 1)
    anchors = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        frames = [i for i in eligible if lo <= i < hi]
        if frames:
            anchors.append(max(frames, key=lambda i: (states[i]['quality'], states[i]['iou'])))
    if len(anchors) < 5:
        raise ValueError(f'clear frames cover only {len(anchors)} video intervals')
    holdout = [i for i in eligible if i not in anchors]
    if len(holdout) < 10:
        raise ValueError('not enough unused clear frames for validation')
    return anchors, holdout


def rendered_mask(vertices: np.ndarray, faces: np.ndarray, R: np.ndarray, t: np.ndarray,
                  scale: float, K: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Rasterize the SAME fixed geometry and pose convention used by the submission."""
    posed = vertices @ R.T * scale + t
    uvw = posed @ K.T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:], 1e-6)
    xy = np.rint(uv[faces] / RENDER_DOWNSAMPLE).astype(np.int32)
    valid = (posed[faces][:, :, 2] > 0).all(axis=1)
    hit = np.zeros(hw, np.uint8)
    if valid.any():
        cv2.fillPoly(hit, list(xy[valid]), 1)
    return hit.astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def choose_constant_pose(rotations: np.ndarray, translations: np.ndarray,
                         anchors: list[int], observed_masks: dict[int, np.ndarray],
                         vertices: np.ndarray, faces: np.ndarray, scale: float,
                         K: np.ndarray, hw: tuple[int, int]) -> tuple[int, dict[int, float]]:
    """Pick a recorded pose by median agreement with masks at DIFFERENT anchor times."""
    scores = {}
    for i in anchors:
        pred = rendered_mask(vertices, faces, rotations[i], translations[i], scale, K, hw)
        scores[i] = float(np.median([mask_iou(pred, observed_masks[j]) for j in anchors]))
    return max(anchors, key=lambda i: (scores[i], -abs(i - len(rotations) // 2))), scores


def accept_stationary_pose(original_iou: float, static_iou: float, max_shift_cm: float,
                           *, max_iou_drop: float, max_allowed_shift_cm: float) -> bool:
    """Held-out image gate; self-acceleration alone is not sufficient evidence."""
    return static_iou >= original_iou - max_iou_drop and max_shift_cm <= max_allowed_shift_cm


def load_local_poses(pose_dir: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Use ALL local video frames, including the unscored tail absent from submission NPZs."""
    rows = [json.loads((pose_dir / f'{i:06d}.json').read_text()) for i in range(n)]
    quaternion = np.asarray([p['rotation'] for p in rows], dtype=float)  # stored wxyz
    R = Rotation.from_quat(np.c_[quaternion[:, 1:], quaternion[:, :1]]).as_matrix()
    t = np.asarray([p['translation'] for p in rows], dtype=float)
    return R, t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trial', type=Path, required=True)
    ap.add_argument('--episodes', type=int, nargs='+', default=list(EPISODES))
    ap.add_argument('--max_holdout_iou_drop', type=float, default=0.05)
    ap.add_argument('--max_shift_cm', type=float, default=8.)
    args = ap.parse_args()
    if not args.episodes or not set(args.episodes).issubset(EPISODES):
        raise SystemExit('only episodes 24 and 26 have the confirmed stationary prior; never lock episode 25')
    trial = args.trial.resolve()
    if not trial.is_relative_to(paths.DATA.resolve() / 'experiments'):
        raise SystemExit('--trial must be inside V2D_DATA/experiments')
    trial_episodes = trial / 'episodes'
    trial_episodes.mkdir(parents=True, exist_ok=True)
    summary = []
    for e in args.episodes:
        stem = f'episode_{e:06d}'
        w = paths.DATA / 'out' / stem
        original = paths.DATA / 'submission/episodes' / f'{stem}.npz'
        with np.load(original) as d:
            arrays = {key: d[key] for key in d.files}
        states = json.loads((w / 'track/states.json').read_text())['frames']
        meta = json.loads((w / 'track1_meta.json').read_text())
        offset, n = meta['frame_offset'], meta['n_frames']
        anchors, holdout = visible_indices(states)
        frame_id = lambda i: i + offset
        mesh_file = paths.DATA / 'submission/episodes' / f'{stem}_object.obj'
        mesh = trimesh.load(mesh_file, force='mesh', process=False)
        intr = json.loads((w / 'intrinsics_stable.json').read_text())
        K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]], float)
        hw = (int(intr['height']) // RENDER_DOWNSAMPLE, int(intr['width']) // RENDER_DOWNSAMPLE)
        observed = {}
        for i in anchors + holdout:
            p = w / 'masks/1' / f'{i:06d}.png'
            mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(p)
            observed[i] = cv2.resize(mask, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) > 0
        rotations, translations = load_local_poses(w / 'poses_final', n)
        # Contract: shared VIDEO frames must agree with the already-packed control.
        stop = min(n, len(arrays['object_translation']) - offset)
        np.testing.assert_allclose(rotations[:stop], arrays['object_rotation'][offset:offset+stop], atol=1e-6)
        np.testing.assert_allclose(translations[:stop], arrays['object_translation'][offset:offset+stop], atol=1e-6)
        scale = float(arrays['object_scale'])
        choice, anchor_scores = choose_constant_pose(rotations, translations, anchors, observed,
                                                      np.asarray(mesh.vertices), np.asarray(mesh.faces), scale, K, hw)
        const = rendered_mask(np.asarray(mesh.vertices), np.asarray(mesh.faces),
                              rotations[choice], translations[choice], scale, K, hw)
        original_holdout = [mask_iou(rendered_mask(np.asarray(mesh.vertices), np.asarray(mesh.faces),
                           rotations[i], translations[i], scale, K, hw), observed[i]) for i in holdout]
        fixed_holdout = [mask_iou(const, observed[i]) for i in holdout]
        diff = np.linalg.norm(translations-translations[choice], axis=1)*100
        row = {'episode': e, 'anchor_video_frames': [frame_id(i) for i in anchors],
               'chosen_video_frame': frame_id(choice), 'anchor_median_iou': anchor_scores[choice],
               'holdout_count': len(holdout),
               'holdout_iou_original': float(np.median(original_holdout)),
               'holdout_iou_static': float(np.median(fixed_holdout)),
               'max_translation_correction_cm': float(diff.max()),
               'p95_translation_correction_cm': float(np.percentile(diff, 95)),
               'object_scale_unchanged': scale}
        row['pass'] = accept_stationary_pose(row['holdout_iou_original'], row['holdout_iou_static'],
                                             row['max_translation_correction_cm'],
                                             max_iou_drop=args.max_holdout_iou_drop,
                                             max_allowed_shift_cm=args.max_shift_cm)
        summary.append(row)
        (trial / f'{stem}_report.json').write_text(json.dumps(row, indent=2) + '\n')
        print(f"{stem}: anchor {frame_id(choice)}, holdout IoU {row['holdout_iou_original']:.3f} -> "
              f"{row['holdout_iou_static']:.3f}; max shift {row['max_translation_correction_cm']:.2f} cm; "
              f"{'PASS' if row['pass'] else 'REJECT'}", flush=True)
        if not row['pass']:
            continue
        arrays['object_rotation'] = np.broadcast_to(rotations[choice].astype(arrays['object_rotation'].dtype),
                                                   arrays['object_rotation'].shape).copy()
        arrays['object_translation'] = np.broadcast_to(translations[choice].astype(arrays['object_translation'].dtype),
                                                      arrays['object_translation'].shape).copy()
        np.savez(trial_episodes / f'{stem}.npz', **arrays)
        shutil.copy2(mesh_file, trial_episodes / mesh_file.name)
    (trial / 'report.json').write_text(json.dumps({'episodes': summary,
        'accepted': [r['episode'] for r in summary if r['pass']],
        'rejected': [r['episode'] for r in summary if not r['pass']],
        'validation': 'held-out high-visibility masks; no Kaggle GT'}, indent=2) + '\n')
    if any(not r['pass'] for r in summary):
        raise SystemExit('one or more stationary candidates failed independent held-out checks')


if __name__ == '__main__':
    main()
