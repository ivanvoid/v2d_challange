#!/usr/bin/env python3
"""Assemble the physical-plausibility candidate (stronger object smoothing + PEN push-out).

Per episode, starting from the last Kaggle-scored control (stool_static_plus_smooth_t150):
  static    episodes whose object is held still for the whole video keep the control arrays
  smooth    stage 7 lam_t=150 + stage 8 (trial/episodes)
  depen     smooth + stage 8b PEN push-out (trial/episodes_depen), only if its report shows lower
            PEN on the fit AND the held-out hand points without more object acceleration
Only object_translation / object_rotation may differ from the control. Body arrays, object scale
and the object mesh file must be byte-identical, otherwise the episode is rejected loudly: this
script never changes depth placement (k) or geometry by accident. Nothing is packed or uploaded.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import paths

POSE_KEYS = ('object_translation', 'object_rotation')


def load_npz(path: Path) -> dict:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def check_only_object_pose_changed(control: dict, candidate: dict) -> None:
    if set(control) != set(candidate):
        raise ValueError(f'NPZ keys differ: {sorted(set(control) ^ set(candidate))}')
    for key in control:
        if key in POSE_KEYS:
            if control[key].shape != candidate[key].shape:
                raise ValueError(f'{key} shape changed')
            if not np.isfinite(candidate[key]).all():
                raise ValueError(f'{key} has non-finite values')
            continue
        if not np.array_equal(control[key], candidate[key]):
            raise ValueError(f'forbidden change in {key}')


def scored_rows(spans: dict, episode: int, n_rows: int) -> np.ndarray:
    """NPZ rows are video frames; the scorer reads exactly the scored ones."""
    frames = np.asarray(spans[str(episode)]['frames'], dtype=int)
    if frames.min() < 0 or frames.max() >= n_rows:
        raise ValueError(f'episode {episode}: scored frames outside the packed rows')
    return frames


def translation_deviation_cm(control: dict, candidate: dict, rows: np.ndarray) -> float:
    d = np.linalg.norm(candidate['object_translation'][rows] - control['object_translation'][rows], axis=1)
    return float(np.percentile(d, 99) * 100)


def depen_accepted(report: dict | None, *, max_acc_rise: float = 0.015, min_gain_cm: float = 1e-3) -> tuple[bool, str]:
    """max_acc_rise is absolute (cm/frame^2 of the episode's own object acceleration): one episode's
    rise of 0.015 moves the 30-episode mean by 0.0005, while its PEN gain is typically 0.05-0.15 cm."""
    if report is None:
        return False, 'no depen report (rejected by stage 8b or not run)'
    fit_gain = report['pen_proxy_before_cm'] - report['pen_proxy_after_cm']
    hold_gain = report['pen_holdout_before_cm'] - report['pen_holdout_after_cm']
    if fit_gain < min_gain_cm or hold_gain < min_gain_cm:
        return False, f'PEN gain too small (fit {fit_gain:.4f}, holdout {hold_gain:.4f} cm)'
    if report['acc_o_after'] > report['acc_o_before'] + max_acc_rise:
        return False, f"object acceleration rises {report['acc_o_before']:.3f} -> {report['acc_o_after']:.3f}"
    return True, f"PEN holdout {report['pen_holdout_before_cm']:.4f} -> {report['pen_holdout_after_cm']:.4f} cm"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--control', type=Path, required=True, help='episodes dir of the scored control')
    ap.add_argument('--trials', type=Path, nargs='+', required=True)
    ap.add_argument('--static', type=int, nargs='*', default=[24, 26])
    ap.add_argument('--no_depen', action='store_true', help='smoothing only (attribution variant)')
    ap.add_argument('--max_dev_cm', type=float, default=6.0, help='p99 scored-frame shift vs control')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    exp = (paths.DATA / 'experiments').resolve()
    out = args.out.resolve()
    if not out.is_relative_to(exp):
        raise SystemExit('--out must be under V2D_DATA/experiments')
    spans = json.loads((paths.DATA / 'submission/scored_spans.json').read_text())
    dest = out / 'episodes'
    dest.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for e in range(30):
        stem = f'episode_{e:06d}'
        control = load_npz(args.control / f'{stem}.npz')
        mesh = args.control / f'{stem}_object.obj'
        choice, reason, source = 'control', 'static object', args.control
        if e not in args.static:
            trial = next((t for t in args.trials if (t / 'episodes' / f'{stem}.npz').exists()), None)
            if trial is None:
                failures.append(f'{stem}: no trial output')
                continue
            report_path = trial / 'out' / stem / 'track/depen.json'
            report = json.loads(report_path.read_text()) if report_path.exists() and \
                (trial / 'episodes_depen' / f'{stem}.npz').exists() else None
            ok, why = depen_accepted(report)
            use_depen = ok and not args.no_depen
            source = trial / ('episodes_depen' if use_depen else 'episodes')
            choice, reason = ('depen' if use_depen else 'smooth'), why
        cand = load_npz(source / f'{stem}.npz')
        try:
            check_only_object_pose_changed(control, cand)
            if (source / f'{stem}_object.obj').read_bytes() != mesh.read_bytes():
                raise ValueError('object mesh changed')
            dev = translation_deviation_cm(control, cand, scored_rows(spans, e, len(control['pose'])))
            if dev > args.max_dev_cm:
                raise ValueError(f'scored-frame translation moved p99 {dev:.1f} cm > {args.max_dev_cm}')
        except ValueError as err:
            failures.append(f'{stem}: {err}')
            continue
        shutil.copy2(source / f'{stem}.npz', dest / f'{stem}.npz')
        shutil.copy2(mesh, dest / mesh.name)
        rows.append({'episode': e, 'choice': choice, 'reason': reason, 'source': str(source),
                     'p99_translation_change_cm': dev})
        print(f'{stem}: {choice:7s} p99 shift {dev:5.2f} cm  {reason}')
    if failures:
        print('\n'.join(failures))
        raise SystemExit(f'{len(failures)} episode(s) failed; no manifest written')
    (out / 'assembly.json').write_text(json.dumps({'control': str(args.control), 'trials': [str(t) for t in args.trials],
                                                   'no_depen': args.no_depen, 'episodes': rows}, indent=2) + '\n')
    print(f'assembled 30 episodes in {dest}')


if __name__ == '__main__':
    main()
