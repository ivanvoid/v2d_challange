#!/usr/bin/env python3
"""Extra smoothing of the object translation (ACC-O variants), on top of a finished episodes dir.

ACC-O compares second differences of the object translation with the reference on scored frames.
Kaggle so far: stage-7 lam 150 -> 1000 lowered ACC-O 0.103 -> 0.084. This applies one more
second-difference smoother (min |y - x|^2 + lam |D2 y|^2) over each episode's packed range
(scored_first..scored_last), translation only. Rotation, scale, mesh and body are unchanged, so the
scorer's body alignment is unchanged too. Objects that do not move (translation spread < 1 mm) are left alone.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve

DATA = Path(os.environ.get('V2D_DATA', '/local-data/ivan/v2d_full'))


def smooth(x: np.ndarray, lam: float) -> np.ndarray:
    n = len(x)
    if n < 3:
        return x.copy()
    D = diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))
    A = (identity(n) + lam * (D.T @ D)).tocsc()
    return np.stack([spsolve(A, x[:, k]) for k in range(x.shape[1])], 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', type=Path, required=True, help='episodes dir (NPZ + OBJ)')
    ap.add_argument('--out', type=Path, required=True, help='episodes dir to write')
    ap.add_argument('--lam', type=float, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for npz in sorted(args.src.glob('episode_*.npz')):
        name = npz.stem
        shutil.copy2(args.src / f'{name}_object.obj', args.out / f'{name}_object.obj')
        z = dict(np.load(npz))
        meta = json.loads((DATA / 'out' / name / 'track1_meta.json').read_text())
        a, b = meta['scored_first'], meta['scored_last']
        t = z['object_translation'][a:b + 1].astype(np.float64)
        if np.ptp(t, axis=0).max() < 1e-3:
            report[name] = 'static, unchanged'
        else:
            y = smooth(t, args.lam)
            acc = lambda u: float(np.linalg.norm(u[2:] - 2 * u[1:-1] + u[:-2], axis=1).mean() * 100)
            z['object_translation'] = z['object_translation'].copy()
            z['object_translation'][a:b + 1] = y.astype(z['object_translation'].dtype)
            report[name] = dict(acc_cm_before=acc(t), acc_cm_after=acc(y),
                                move_cm_max=float(np.linalg.norm(y - t, axis=1).max() * 100))
        np.savez(args.out / f'{name}.npz', **z)
    (args.out.parent / f'smooth_object_translation_lam{args.lam:g}.json').write_text(json.dumps(report, indent=1) + '\n')
    moved = [v for v in report.values() if isinstance(v, dict)]
    print(f"lam {args.lam:g}: {len(moved)} moving episodes; mean self-acc {np.mean([v['acc_cm_before'] for v in moved]):.4f} -> "
          f"{np.mean([v['acc_cm_after'] for v in moved]):.4f} cm/frame^2; max move {max(v['move_cm_max'] for v in moved):.1f} cm")


if __name__ == '__main__':
    main()
