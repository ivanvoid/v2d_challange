#!/usr/bin/env python3
"""Assemble an isolated PEN + stronger-smoothing experiment from packed episode NPZs.

No inference stage or production file is written. For episode 16, either apply
the previously validated PEN translation *delta* to the stronger-smoothed pose,
or preserve the PEN pose without stronger smoothing. Every other submitted
array must match the stage67_v2 control exactly.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import paths

PEN_EPISODES = frozenset((15, 16, 17))
SMOOTH_EPISODES = frozenset((8, 16, 25, 28))
CHANGED_EPISODES = PEN_EPISODES | SMOOTH_EPISODES


def compose_episode(base: dict, pen: dict, smooth: dict, *, overlap: str) -> dict:
    """Compose independent translation deltas; reject any unintended body/geometry changes."""
    if overlap not in ('additive', 'pen_only'):
        raise ValueError(f'unknown overlap policy {overlap}')
    if set(base) != set(pen) or set(base) != set(smooth):
        raise ValueError('episode NPZ keys differ across trials')
    for key in base:
        if key != 'object_translation':
            np.testing.assert_array_equal(base[key], pen[key], err_msg=f'PEN trial changed {key}')
            np.testing.assert_array_equal(base[key], smooth[key], err_msg=f'smoothing trial changed {key}')
    x, p, s = (np.asarray(item['object_translation'], dtype=np.float64) for item in (base, pen, smooth))
    if not (x.shape == p.shape == s.shape and x.ndim == 2 and x.shape[1] == 3):
        raise ValueError('expected matching [T,3] object translations')
    correction = p - x
    if np.linalg.norm(correction, axis=1).max() > 0.0301:
        raise ValueError('PEN correction exceeds its 3 cm total-shift contract')
    combined = s + correction if overlap == 'additive' else p
    if not np.isfinite(combined).all():
        raise ValueError('combined object translations contain non-finite values')
    return {**base, 'object_translation': combined.astype(base['object_translation'].dtype)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dest', type=Path, required=True)
    ap.add_argument('--overlap', choices=('additive', 'pen_only'), required=True)
    args = ap.parse_args()
    dest = args.dest.resolve()
    if not dest.is_relative_to(paths.DATA.resolve() / 'experiments'):
        raise SystemExit('output must be an isolated directory under V2D_DATA/experiments')
    source = paths.DATA / 'submission/episodes'
    pen = paths.DATA / 'experiments/penfix_smooth/episodes'
    smooth = paths.DATA / 'experiments/smooth_t150_full/episodes'
    episodes = dest / 'episodes'
    episodes.mkdir(parents=True, exist_ok=True)
    for e in range(30):
        stem = f'episode_{e:06d}'
        for folder in (source, pen, smooth):
            if not (folder / f'{stem}.npz').is_file() or not (folder / f'{stem}_object.obj').is_file():
                raise FileNotFoundError(f'{stem} missing from {folder}')
        mesh = source / f'{stem}_object.obj'
        if e in CHANGED_EPISODES:
            for other in (pen, smooth):
                if mesh.read_bytes() != (other / mesh.name).read_bytes():
                    raise ValueError(f'{stem}: object mesh differs from control in {other}')
        shutil.copy2(mesh, episodes / mesh.name)
        if e == 16:
            def read(folder):
                with np.load(folder / f'{stem}.npz') as data:
                    return {k: data[k] for k in data.files}
            composed = compose_episode(read(source), read(pen), read(smooth), overlap=args.overlap)
            np.savez(episodes / f'{stem}.npz', **composed)
        else:
            origin = pen if e in PEN_EPISODES else smooth if e in SMOOTH_EPISODES else source
            shutil.copy2(origin / f'{stem}.npz', episodes / f'{stem}.npz')
    (dest / 'manifest.json').write_text(json.dumps({
        'name': dest.name, 'overlap': args.overlap,
        'changed_episodes': sorted(CHANGED_EPISODES),
        'sources': {'control': str(source), 'pen': str(pen), 'smooth': str(smooth)},
        'uploadable': False, 'validation': 'requires partial and full packed evaluation before promotion',
    }, indent=2) + '\n')
    print(f'assembled isolated {args.overlap} combination in {episodes}; NOT validated or uploadable')


if __name__ == '__main__':
    main()
