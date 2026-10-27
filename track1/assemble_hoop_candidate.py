#!/usr/bin/env python3
"""Assemble (not upload) a full hoop submission only after isolated tracking passes.

Use the already-scored stool_static_plus_smooth_t150 file as the control so
the only new conceptual change is episode-0 hoop mask/tracking. Body, object
mesh and all other 29 episodes are preserved exactly. The new packed Parquet
still needs local evaluation and user-approved Kaggle scoring.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import paths
from finish_hoop_trial import require_stage6_gate


def validate_episode_delta(original: dict, candidate: dict, *, max_hoop_diameter_m: float = .78,
                           min_hoop_diameter_m: float = .60,
                           mesh_max_extent_m: float = .732) -> None:
    if set(original) != set(candidate):
        raise ValueError('episode data fields changed unexpectedly')
    for key in original:
        if key not in ('object_rotation', 'object_translation', 'object_scale'):
            np.testing.assert_array_equal(original[key], candidate[key],
                                          err_msg=f'forbidden body/mesh-related change: {key}')
    scale = float(candidate['object_scale'])
    if not min_hoop_diameter_m <= mesh_max_extent_m * scale <= max_hoop_diameter_m:
        raise ValueError('hoop size outside the user-supported 67–72 cm soft prior (+slack)')
    if not np.isfinite(candidate['object_translation']).all() or not np.isfinite(candidate['object_rotation']).all():
        raise ValueError('non-finite object pose in hoop trial')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trial', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--episode', type=int, choices=(0, 1), required=True)
    args = ap.parse_args()
    trial, out = args.trial.resolve(), args.out.resolve()
    allowed = paths.DATA.resolve() / 'experiments'
    if not trial.is_relative_to(allowed) or not out.is_relative_to(allowed) or trial == out:
        raise SystemExit('trial and output must be separate experiment directories')
    require_stage6_gate(trial)
    if not (trial / 'eval/proxies.json').is_file() or not (trial / 'compare_production.json').is_file():
        raise SystemExit('packed episode-0 evaluation must finish before assembling a full candidate')
    baseline = paths.DATA / 'experiments/stool_static_plus_smooth_t150/episodes'
    changed = trial / 'episodes'
    episode = args.episode
    stem = f'episode_{episode:06d}'
    mesh_src = baseline / f'{stem}_object.obj'
    if mesh_src.read_bytes() != (changed / mesh_src.name).read_bytes():
        raise SystemExit('hoop mesh changed; this must remain a pose-only experiment')
    def load(folder):
        with np.load(folder / f'{stem}.npz') as data:
            return {k: data[k] for k in data.files}
    old, new = load(baseline), load(changed)
    import trimesh
    mesh_max_extent = float(trimesh.load(mesh_src, force='mesh', process=False).extents.max())
    validate_episode_delta(old, new, mesh_max_extent_m=mesh_max_extent)
    before = json.loads((paths.DATA / 'experiments/stool_static_plus_smooth_t150/eval/proxies.json').read_text())
    after = json.loads((trial / 'eval/proxies.json').read_text())
    old_row = next(r for r in before['episodes'] if r['episode'] == episode)
    new_row = next(r for r in after['episodes'] if r['episode'] == episode)
    if new_row['pen_cm'] > old_row['pen_cm'] + .01 or new_row['acc_o_self'] > old_row['acc_o_self'] + .03:
        raise SystemExit('hoop local PEN/jitter regressed beyond pre-declared budgets')
    episodes = out / 'episodes'
    episodes.mkdir(parents=True, exist_ok=True)
    for e in range(30):
        name = f'episode_{e:06d}'
        for suffix in ('.npz', '_object.obj'):
            source = changed if e == episode else baseline
            path = source / f'{name}{suffix}'
            if not path.is_file():
                raise FileNotFoundError(path)
            shutil.copy2(path, episodes / path.name)
    (out / 'manifest.json').write_text(json.dumps({
        'name': out.name, 'status': 'not packed, not uploaded; isolated hoop episode passed its gate',
        'source': str(baseline), 'hoop_trial': str(trial), 'changed_episodes': [episode],
        'stage6_gate': str(trial / 'stage6_gate.json'),
        'changed_episode_local_baseline': {k: old_row[k] for k in ('pen_cm', 'acc_o_self', 'gap_cm_median')},
        'changed_episode_local_candidate': {k: new_row[k] for k in ('pen_cm', 'acc_o_self', 'gap_cm_median')},
    }, indent=2)+'\n')
    print(f'assembled isolated 30-episode candidate under {out}; pack and validate before any upload')


if __name__ == '__main__':
    main()
