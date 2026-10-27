#!/usr/bin/env python3
"""Isolated stage-4 hoop hole-repair pilot; never modifies production masks.

Only episode 0's user-confirmed final floor interval is eligible. SAM2 sometimes
fills the centre of a thin ring as a solid disk. Detect high-confidence filled
masks and remove their eroded interior, retaining only the observed outer band.
Never synthesize a complete ring across held/partially occluded arcs.
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
from stationary_stool import mask_iou, rendered_mask


def fill_fraction(mask: np.ndarray) -> float:
    """Object pixels divided by convex-hull area; robust to a missing/open interior contour."""
    small = cv2.resize(mask.astype(np.uint8),
                       (mask.shape[1] // 2, mask.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
    pts = cv2.findNonZero(small)
    if pts is None or len(pts) < 3:
        return 0.0
    hull_area = cv2.contourArea(cv2.convexHull(pts))
    return float((small > 0).sum() / max(hull_area, 1))


def largest_hole_fraction(mask: np.ndarray) -> float:
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0.0
    hierarchy = hierarchy[0]
    roots = [i for i, row in enumerate(hierarchy) if row[3] < 0]
    root = max(roots, key=lambda i: cv2.contourArea(contours[i]))
    hole, child = 0.0, hierarchy[root][2]
    while child >= 0:
        hole = max(hole, cv2.contourArea(contours[child]))
        child = hierarchy[child][0]
    return float(hole / max(cv2.contourArea(contours[root]), 1))


def remove_filled_interior(mask: np.ndarray, width_px: int) -> np.ndarray:
    if width_px < 1:
        raise ValueError('rim width must be positive')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    interior = cv2.erode(mask.astype(np.uint8), kernel, iterations=width_px) > 0
    return mask & ~interior


def pose_at(w: Path, local_frame: int) -> tuple[np.ndarray, np.ndarray, float]:
    row = json.loads((w / 'poses_final' / f'{local_frame:06d}.json').read_text())
    w_, x, y, z = row['rotation']
    R = Rotation.from_quat([x, y, z, w_]).as_matrix()
    return R, np.asarray(row['translation']), float(np.ravel(row.get('scale', [1]))[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trial', type=Path, required=True)
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--floor_start_local', type=int, default=570)
    ap.add_argument('--min_fill', type=float, default=.72)
    ap.add_argument('--rim_width_px', type=int, default=8)
    args = ap.parse_args()
    if args.episode != 0:
        raise SystemExit('the floor interval and hole-loss prior are verified only for episode 0')
    src = paths.DATA / 'out/episode_000000'
    trial = args.trial.resolve()
    if not trial.is_relative_to(paths.DATA.resolve() / 'experiments'):
        raise SystemExit('trial must be inside V2D_DATA/experiments')
    dst = trial / 'masks/1'
    dst.mkdir(parents=True, exist_ok=True)
    n = json.loads((src / 'track1_meta.json').read_text())['n_frames']
    intr = json.loads((src / 'intrinsics_stable.json').read_text())
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]])
    hw = (int(intr['height']) // 4, int(intr['width']) // 4)
    mesh = trimesh.load(paths.DATA / 'submission/episodes/episode_000000_object.obj',
                        force='mesh', process=False)
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    rows = []
    changed = []
    for local in range(n):
        name = f'{local:06d}.png'
        source = src / 'masks/1' / name
        original = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise FileNotFoundError(source)
        mask = original > 0
        fraction = fill_fraction(mask) if local >= args.floor_start_local else 0.0
        repair = local >= args.floor_start_local and fraction >= args.min_fill
        if repair:
            corrected = remove_filled_interior(mask, args.rim_width_px)
            if not (corrected.sum() > 0 and corrected.sum() < mask.sum()):
                raise ValueError(f'frame {local} could not be repaired safely')
            if not cv2.imwrite(str(dst / name), corrected.astype(np.uint8)*255):
                raise OSError(dst / name)
            changed.append(local)
        else:
            shutil.copy2(source, dst / name)
            corrected = mask
        if repair:
            R, t, scale = pose_at(src, local)
            pred = rendered_mask(v, f, R, t, scale, K, hw)
            down = lambda x: cv2.resize(x.astype(np.uint8), (hw[1], hw[0]),
                                        interpolation=cv2.INTER_NEAREST) > 0
            old_iou, new_iou = mask_iou(pred, down(mask)), mask_iou(pred, down(corrected))
            rows.append({'local_frame': local, 'fill_fraction': fraction,
                         'original_iou_saved_pose': old_iou, 'repaired_iou_saved_pose': new_iou,
                         'hole_area_fraction': largest_hole_fraction(corrected)})
    if not changed:
        raise SystemExit('no confidently filled hoop masks in the confirmed final floor interval')
    before = float(np.median([r['original_iou_saved_pose'] for r in rows]))
    after = float(np.median([r['repaired_iou_saved_pose'] for r in rows]))
    report = {'episode': 0, 'floor_start_local': args.floor_start_local,
              'modified_local_frames': changed, 'modified_count': len(changed),
              'source_masks': str(src / 'masks/1'), 'isolated_masks': str(dst),
              'rim_width_px': args.rim_width_px, 'min_fill': args.min_fill,
              'existing_pose_iou_median_before': before, 'existing_pose_iou_median_after': after,
              'saved_pose_gate_pass': after >= before + .05, 'per_frame': rows,
              'note': 'existing poses are NOT independent ground truth; next gate is multi-frame re-registration'}
    (trial / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    verdict = 'PASS' if report['saved_pose_gate_pass'] else 'REJECT'
    print(f'episode 0: repaired {len(changed)} / {n} masks; IoU of SAVED ring poses '
          f'{before:.3f} -> {after:.3f}; gate {verdict}')
    if not report['saved_pose_gate_pass']:
        raise SystemExit('mask prior did not improve the existing pose/mesh consistency; no tracker rerun')


if __name__ == '__main__':
    main()
