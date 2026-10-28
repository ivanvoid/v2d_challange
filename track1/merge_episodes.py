#!/usr/bin/env python3
"""Merge per-episode results of several isolated experiments into one 30-episode directory.

  --base   full episodes dir every other episode is copied from
  --take   DIR:E,E,...  episodes taken from DIR instead (repeatable; an episode may appear once)
  --pose_from DIR       afterwards replace the body `pose` of every episode by DIR's (e.g. a smoothed body)
Every episode keeps its mesh file byte-identical to --base and the same NPZ keys / shapes; the
merged directory and a merge.json with the provenance of each episode are written to --out.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import paths


def parse_take(items: list[str]) -> dict[int, Path]:
    owner: dict[int, Path] = {}
    for item in items:
        folder, _, eps = item.rpartition(':')
        for e in (int(x) for x in eps.split(',') if x):
            if e in owner:
                raise ValueError(f'episode {e} taken twice')
            owner[e] = Path(folder)
    return owner


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--take', nargs='*', default=[])
    ap.add_argument('--pose_from', type=Path, default=None)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((paths.DATA / 'experiments').resolve()):
        raise SystemExit('--out must be under V2D_DATA/experiments')
    owner = parse_take(args.take)
    dest = out / 'episodes'
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in range(30):
        stem = f'episode_{e:06d}'
        src = owner.get(e, args.base)
        base = np.load(args.base / f'{stem}.npz')
        new = np.load(src / f'{stem}.npz')
        if set(base.files) != set(new.files) or any(base[k].shape != new[k].shape for k in base.files):
            raise SystemExit(f'{stem}: NPZ layout differs between {args.base} and {src}')
        if (src / f'{stem}_object.obj').read_bytes() != (args.base / f'{stem}_object.obj').read_bytes():
            raise SystemExit(f'{stem}: object mesh differs')
        if args.pose_from is not None:
            body = np.load(args.pose_from / f'{stem}.npz')
            if body['pose'].shape != new['pose'].shape or not (np.array_equal(body['scales'], new['scales'])
                                                             and np.array_equal(body['shape'], new['shape'])):
                raise SystemExit(f'{stem}: --pose_from body does not match (shape/identity)')
            new = {**{k: new[k] for k in new.files}, 'pose': body['pose']}
            np.savez(dest / f'{stem}.npz', **new)
        else:
            new = {k: new[k] for k in new.files}
            shutil.copy2(src / f'{stem}.npz', dest / f'{stem}.npz')
        changed = [k for k in base.files if not np.array_equal(base[k], new[k])]
        shutil.copy2(args.base / f'{stem}_object.obj', dest / f'{stem}_object.obj')
        rows.append({'episode': e, 'source': str(src), 'changed_vs_base': changed})
    (out / 'merge.json').write_text(json.dumps({'base': str(args.base), 'pose_from': str(args.pose_from),
                                                'episodes': rows}, indent=1) + '\n')
    print('merged; episodes changed vs base:', {r['episode']: r['changed_vs_base'] for r in rows if r['changed_vs_base']})


if __name__ == '__main__':
    main()
