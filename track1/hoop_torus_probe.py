#!/usr/bin/env python3
"""Multi-video hoop-geometry probe; never changes production tracking or submissions.

Fit ONE physical torus diameter/tube thickness from clear stage-6 frames in
episodes 0 and 2, score on separate held-out frames, and report episode 1 as a
tracking-limited diagnostic (no high-confidence stage-6 anchors). Existing
per-frame poses, camera intrinsics and body outputs remain fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

import paths
from stationary_stool import mask_iou, rendered_mask

TRAIN_EPISODES = (0, 2)
ALL_EPISODES = (0, 1, 2)


def physical_torus(outer_diameter_m: float, tube_diameter_m: float, object_scale: float) -> trimesh.Trimesh:
    """Mesh units chosen to share physical dimensions AFTER each episode's object scale."""
    if not (0 < tube_diameter_m < outer_diameter_m and object_scale > 0):
        raise ValueError('invalid torus diameters or object scale')
    minor = tube_diameter_m / (2 * object_scale)
    major = outer_diameter_m / (2 * object_scale) - minor
    mesh = trimesh.creation.torus(major_radius=major, minor_radius=minor,
                                    major_sections=96, minor_sections=12)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    return mesh


def clear_frames(rows: list[dict], *, minimum_iou: float = .42) -> list[int]:
    area = np.array([r['area_frac'] for r in rows])
    return [i for i, r in enumerate(rows) if r['state'] == 'observed' and r['occlusion'] < .25
            and r['iou'] > minimum_iou and r['area_frac'] >= np.percentile(area, 35)]


def split_temporal(eligible: list[int], n_frames: int, n_bins: int = 8) -> tuple[list[int], list[int]]:
    anchors = []
    for lo, hi in zip(np.linspace(0, n_frames, n_bins + 1)[:-1],
                      np.linspace(0, n_frames, n_bins + 1)[1:]):
        block = [f for f in eligible if lo <= f < hi]
        if block:
            anchors.append(block[len(block) // 2])
    holdout = [f for f in eligible if f not in anchors]
    return anchors, holdout


def load_episode(e: int, sample: list[int]) -> dict:
    w = paths.DATA / 'out' / f'episode_{e:06d}'
    meta = json.loads((w / 'track1_meta.json').read_text())
    intr = json.loads((w / 'intrinsics_stable.json').read_text())
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]])
    hw = (int(intr['height']) // 4, int(intr['width']) // 4)
    with np.load(paths.DATA / 'submission/episodes' / f'episode_{e:06d}.npz') as packed:
        scale = float(packed['object_scale'])
        off = meta['frame_offset']
        poses = [(packed['object_rotation'][off+i], packed['object_translation'][off+i]) for i in sample]
    masks = []
    for i in sample:
        m = cv2.imread(str(w / 'masks/1' / f'{i:06d}.png'), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(w / 'masks/1' / f'{i:06d}.png')
        masks.append(cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) > 0)
    original = trimesh.load(paths.DATA / 'submission/episodes' / f'episode_{e:06d}_object.obj',
                            force='mesh', process=False)
    return {'K': K, 'hw': hw, 'poses': poses, 'masks': masks, 'scale': scale,
            'original': original, 'sample_frames': sample, 'offset': off}


def silhouette_scores(mesh: trimesh.Trimesh, d: dict) -> list[float]:
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    return [mask_iou(rendered_mask(v, f, R, t, d['scale'], d['K'], d['hw']), mask)
            for (R, t), mask in zip(d['poses'], d['masks'])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trial', type=Path, required=True)
    args = ap.parse_args()
    trial = args.trial.resolve()
    if not trial.is_relative_to(paths.DATA.resolve() / 'experiments'):
        raise SystemExit('trial must be inside V2D_DATA/experiments')
    trial.mkdir(parents=True, exist_ok=True)
    train, holdout, stats = {}, {}, {}
    for e in ALL_EPISODES:
        w = paths.DATA / 'out' / f'episode_{e:06d}'
        meta = json.loads((w / 'track1_meta.json').read_text())
        rows = json.loads((w / 'track/states.json').read_text())['frames']
        valid = [i for i in clear_frames(rows) if i + meta['frame_offset'] < meta['scored_last'] + 1]
        anchors, reserved = split_temporal(valid, meta['n_frames'])
        stats[e] = {'visible_frames': len(valid), 'anchors': [i+meta['frame_offset'] for i in anchors],
                    'holdout': len(reserved)}
        if e in TRAIN_EPISODES:
            if len(anchors) < 2 or len(reserved) < 10:
                raise SystemExit(f'episode {e}: insufficient independent geometry anchors/holdout')
            train[e], holdout[e] = load_episode(e, anchors), load_episode(e, reserved)
        else:
            # Episode 1 has no reliable tracked poses: evaluate only as a pose-limited diagnostic.
            stats[e]['tracking_limited'] = len(anchors) < 2
        print(f"ep {e}: {len(valid)} clear frames, {len(anchors)} distributed train anchors, "
              f"{len(reserved)} holdout; tracking-limited={stats[e].get('tracking_limited', False)}", flush=True)
    attempts = []
    for outer in np.round(np.arange(.66, .901, .04), 3):
        for tube in (.03, .05, .07):
            scores = {e: float(np.median(silhouette_scores(physical_torus(outer, tube, train[e]['scale']),
                                                            train[e]))) for e in TRAIN_EPISODES}
            attempts.append({'outer_diameter_m': float(outer), 'tube_diameter_m': float(tube),
                             'training_iou': scores, 'score': float(np.mean(list(scores.values())))})
    best = max(attempts, key=lambda x: x['score'])
    independent = {}
    for e in TRAIN_EPISODES:
        d = holdout[e]
        original = silhouette_scores(d['original'], d)
        torus = silhouette_scores(physical_torus(best['outer_diameter_m'],
                                  best['tube_diameter_m'], d['scale']), d)
        independent[e] = {'frames': len(original), 'baseline_median_iou': float(np.median(original)),
                          'torus_median_iou': float(np.median(torus))}
        print(f"ep {e}: held-out mask IoU original {np.median(original):.3f} -> "
              f"torus {np.median(torus):.3f}", flush=True)
    accepted = all(r['torus_median_iou'] >= r['baseline_median_iou']-.05 for r in independent.values())
    best['heldout'] = independent
    best['passes_heldout_guard'] = accepted
    for e in ALL_EPISODES:
        with np.load(paths.DATA / 'submission/episodes' / f'episode_{e:06d}.npz') as p:
            scale = float(p['object_scale'])
        mesh = physical_torus(best['outer_diameter_m'], best['tube_diameter_m'], scale)
        mesh.export(trial / f'episode_{e:06d}_torus.obj')
    report = {'best': best, 'all_attempts': attempts, 'episodes': stats,
              'note': 'existing object poses kept fixed; episode 1 has no high-fit anchors; no submission produced'}
    (trial / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f"shared torus diameter {best['outer_diameter_m']:.3f} m, tube {best['tube_diameter_m']:.3f} m; "
          f"held-out guard {'PASS' if accepted else 'REJECT'}", flush=True)


if __name__ == '__main__':
    main()
