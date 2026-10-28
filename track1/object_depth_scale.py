#!/usr/bin/env python3
"""Move every object along its camera rays by a factor k (translation AND size x k).

Stage 8 places the object where the hands touch it. Its score is flat over a band of k (any k that
keeps a hand on the surface ties) and the tie-break picks the k nearest 1 — the camera-side end of
that band. On the Track 2 local test set (6 episodes, scored against the multi-view reference) this
leaves objects ~4 % too close: CD-O 20.8 -> 16.4 cm when every object is scaled by 1.03-1.05 about
the camera; all episodes agreed (best 1.03-1.05). The same curve predicts the earlier Kaggle result
p1_global (+7-16 % on 8 episodes made CD-O worse by 5.9 cm). Scaling about the camera keeps every
image projection identical, so masks/reprojection are unaffected; only depth and size change.

The submitted frame is the camera frame (MHR's flip [1,-1,-1] is a rotation about the camera
centre), so scaling the stored translation by k is a scaling about the camera.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import paths


def scale_about_camera(d: dict, k: float) -> dict:
    if not 0.8 <= k <= 1.25:
        raise ValueError(f'k={k} outside the plausible range')
    out = dict(d)
    out['object_translation'] = (d['object_translation'] * k).astype(d['object_translation'].dtype)
    out['object_scale'] = np.asarray(d['object_scale'] * k, dtype=d['object_scale'].dtype)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--k', type=float, required=True)
    ap.add_argument('--episodes', type=int, nargs='*', default=list(range(30)))
    args = ap.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((paths.DATA / 'experiments').resolve()):
        raise SystemExit('--out must be under V2D_DATA/experiments')
    dest = out / 'episodes'
    dest.mkdir(parents=True, exist_ok=True)
    for f in args.src.glob('episode_*'):
        shutil.copy2(f, dest / f.name)
    for e in args.episodes:
        stem = f'episode_{e:06d}'
        np.savez(dest / f'{stem}.npz', **scale_about_camera(dict(np.load(args.src / f'{stem}.npz')), args.k))
    (out / 'object_depth_scale.json').write_text(json.dumps({'src': str(args.src), 'k': args.k,
                                                             'episodes': args.episodes}, indent=1) + '\n')
    print(f'scaled {len(args.episodes)} episodes by k={args.k}')


if __name__ == '__main__':
    main()
