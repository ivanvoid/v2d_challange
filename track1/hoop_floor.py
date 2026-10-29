#!/usr/bin/env python3
"""A ring lying flat on the floor sits at the height of the person's soles (H12, generalised).

Episode 0 ("hula_hoop_inside_ground"): the hoop lay on the floor 22-29 cm above the soles and
50-70 cm in front of the person standing in it, because stage 8 placed it by HAND contact while
the contact was with the FEET. Moving it along the camera rays by one factor (x1.193) fixed it
(Kaggle phys_v7: CD-O -2.87). This tool derives that factor instead of hard-coding it:

  flat frames  scored frames where the ring axis is within --max_tilt_deg of the floor normal
  per frame    k_f = (sole level along the floor normal) / (ring's lowest point along it): scaling
               about the camera keeps the image identical and brings the ring down to the soles
  decision     apply the median k only if >= --min_frames flat frames agree (p10-p90 spread
               <= --max_spread of the median) and |k - 1| >= --min_change; otherwise no change
Episode 2's floor stretch, already at foot level, gives k ~ 1.00 -> unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402


def floor_factor(ring_points: list, ring_axes: np.ndarray, soles: np.ndarray, n: np.ndarray,
                 max_tilt_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: is the ring flat, and the camera-scaling factor that puts its lowest point at the soles.
    n is the floor normal pointing DOWN (camera frame), so larger n.X = lower."""
    tilt = np.degrees(np.arccos(np.clip(np.abs(ring_axes @ n), 0, 1)))
    k = np.array([s / max((P @ n).max(), 1e-6) for P, s in zip(ring_points, soles)])
    return tilt <= max_tilt_deg, k


def decide(k: np.ndarray, flat: np.ndarray, *, min_frames: int, max_spread: float, min_change: float) -> tuple[float, str]:
    kf = k[flat]
    if len(kf) < min_frames:
        return 1.0, f'only {len(kf)} flat frames'
    med = float(np.median(kf))
    spread = float((np.percentile(kf, 90) - np.percentile(kf, 10)) / med)
    if spread > max_spread:
        return 1.0, f'flat frames disagree (p10-p90 spread {spread:.3f})'
    if abs(med - 1) < min_change:
        return 1.0, f'already at foot level (k {med:.3f})'
    return med, f'k {med:.3f} from {len(kf)} flat frames (spread {spread:.3f})'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--episodes', type=int, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max_tilt_deg', type=float, default=15.0)
    ap.add_argument('--min_frames', type=int, default=30)
    ap.add_argument('--max_spread', type=float, default=0.05)
    ap.add_argument('--min_change', type=float, default=0.02)
    args = ap.parse_args()
    import torch
    import trimesh
    from hoop_conic import ring_frame
    from object_depth_scale import scale_about_camera
    sys.path.insert(0, str(paths.KIT_TRACK1))
    from mesh_to_mhr_params import MHR
    out = args.out.resolve()
    if not out.is_relative_to((paths.DATA / 'experiments').resolve()):
        raise SystemExit('--out must be under V2D_DATA/experiments')
    dest = out / 'episodes'
    dest.mkdir(parents=True, exist_ok=True)
    for f in args.src.glob('episode_*'):
        shutil.copy2(f, dest / f.name)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    mhr = MHR(str(paths.DATA / 'weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt'), dev, 64, 'float32')
    spans = json.loads((paths.DATA / 'submission/scored_spans.json').read_text())
    report = []
    for e in args.episodes:
        stem = f'episode_{e:06d}'
        n = np.asarray(json.loads((paths.OUT / stem / 'floor.json').read_text())['normal'], float)
        d = dict(np.load(args.src / f'{stem}.npz'))
        mesh = trimesh.load(args.src / f'{stem}_object.obj', force='mesh', process=False)
        s = float(d['object_scale'])
        V = np.asarray(mesh.vertices) * s
        _, axis, _ = ring_frame(np.asarray(mesh.vertices))
        fr = np.asarray(spans[str(e)]['frames'])
        z = torch.as_tensor(np.concatenate([d['scales'], d['shape']])[None], dtype=torch.float64, device=dev)
        B = (mhr.run(torch.as_tensor(d['pose'][fr], dtype=torch.float64, device=dev), z)[0] / 1000).cpu().numpy()
        soles = (B @ n).max(1)
        pts = [V @ d['object_rotation'][f].T + d['object_translation'][f] for f in fr]
        axes = np.stack([d['object_rotation'][f] @ axis for f in fr])
        flat, k = floor_factor(pts, axes, soles, n, args.max_tilt_deg)
        kk, why = decide(k, flat, min_frames=args.min_frames, max_spread=args.max_spread, min_change=args.min_change)
        if kk != 1.0:
            np.savez(dest / f'{stem}.npz', **scale_about_camera(d, kk))
        report.append({'episode': e, 'flat_frames': int(flat.sum()), 'k': kk, 'decision': why})
        print(f'{stem}: {"SCALE x%.3f" % kk if kk != 1.0 else "unchanged"} — {why}', flush=True)
    (out / 'hoop_floor.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
