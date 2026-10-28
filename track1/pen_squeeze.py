#!/usr/bin/env python3
"""PEN stage 2: deeper push-out and, where both hands squeeze the object, a slightly smaller object.

phys_v1's push-out moves the object at most 3 cm and only translates it. What is left (mostly the
foam block, episodes 15-17) is (a) single hands 4-6 cm inside and (b) two hands pressing in from
opposite sides, where any translation trades one hand's penetration for the other's: the object is
too big for the grip. Per episode, for each shrink factor c on a grid (the object's packed scale
times c, about the mesh origin), solve per scored frame the bounded translation that pushes the
object off the hands (stage8_depen.solve_translation, exactly the scorer's inside test on the
packed mesh), smooth it over each scored stretch, and measure PEN on a held-out hand subset.
Choose the largest c whose held-out PEN is within --tol of the best one (shrink only on evidence).
Accept the episode only if held-out PEN drops and the object's own acceleration rises <= --max_acc_rise.

Input: an assembled episodes dir (NPZ + mesh, rows = video frames) and its extracted packed NPZs
(eval_local.py extract: budgeted mesh, scored frames). Output: a full copy of the episodes dir where
only object_translation / object_scale of accepted episodes differ, plus pen_squeeze.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

GRID = (1.0, 0.98, 0.96, 0.94, 0.92, 0.90)


def stretches(frames: np.ndarray) -> list[np.ndarray]:
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    return np.split(np.arange(len(frames)), breaks)


def smooth_per_stretch(shift: np.ndarray, frames: np.ndarray, sigma: float) -> np.ndarray:
    out = np.zeros_like(shift)
    for idx in stretches(frames):
        out[idx] = gaussian_filter1d(shift[idx], sigma, axis=0, mode='nearest') if len(idx) > 1 else shift[idx]
    return out


def cap(shift: np.ndarray, max_m: float) -> np.ndarray:
    n = np.linalg.norm(shift, axis=-1, keepdims=True)
    return shift * np.minimum(1.0, max_m / np.maximum(n, 1e-12))


def self_acc(x: np.ndarray, frames: np.ndarray) -> float:
    ok = (np.diff(frames)[1:] == 1) & (np.diff(frames)[:-1] == 1)
    a = x[:-2] - 2 * x[1:-1] + x[2:]
    return float(np.linalg.norm(a[ok], axis=1).mean() * 100) if ok.any() else 0.0


def choose_shrink(rows: list[dict], tol: float) -> dict:
    """Largest c (least shrink) whose held-out PEN is within tol (absolute cm) of the best."""
    best = min(r['pen_holdout_cm'] for r in rows)
    return max((r for r in rows if r['pen_holdout_cm'] <= best + tol), key=lambda r: r['c'])


def interpolate_rows(shift_scored: np.ndarray, frames: np.ndarray, n_rows: int) -> np.ndarray:
    """Per-row shift for the episode NPZ (rows = video frames): scored rows exact, others linear."""
    rows = np.arange(n_rows)
    return np.stack([np.interp(rows, frames, shift_scored[:, k]) for k in range(3)], 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True, help='assembled episodes dir (NPZ + obj)')
    ap.add_argument('--packed', type=Path, required=True, help='eval_local.py extract output of --src')
    ap.add_argument('--episodes', type=int, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max_shift_cm', type=float, default=5.0)
    ap.add_argument('--sigma', type=float, default=4.0)
    ap.add_argument('--tol', type=float, default=0.01, help='held-out PEN tolerance (cm) for less shrink')
    ap.add_argument('--max_acc_rise', type=float, default=0.02)
    ap.add_argument('--grid', type=float, nargs='+', default=list(GRID))
    args = ap.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((paths.DATA / 'experiments').resolve()):
        raise SystemExit('--out must be under V2D_DATA/experiments')
    if any(not 0.8 <= c <= 1.0 for c in args.grid) or 1.0 not in args.grid:
        raise SystemExit('grid must contain 1.0 and stay within [0.8, 1]')
    import torch
    sys.path.insert(0, str(paths.KIT_TRACK1))
    from mesh_to_mhr_params import MHR
    from pen_fast import PenetrationMesh
    from qa_episode import hand_vertex_ids
    from stage8_depen import solve_translation
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    mhr = MHR(str(paths.DATA / 'weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt'), dev, 64, 'float32')
    dest = out / 'episodes'
    dest.mkdir(parents=True, exist_ok=True)
    for f in args.src.glob('episode_*'):
        shutil.copy2(f, dest / f.name)
    report = []
    for e in args.episodes:
        stem = f'episode_{e:06d}'
        p = np.load(args.packed / f'{stem}.npz')
        frames, R, t, s = p['frames'], p['R'], p['t'], float(p['s'])
        pm = PenetrationMesh(p['mesh_v'], p['mesh_f'])
        pose = torch.as_tensor(p['pose'], dtype=torch.float64, device=dev)
        z = torch.as_tensor(np.concatenate([p['scales'], p['shape']])[None], dtype=torch.float64, device=dev)
        hid = hand_vertex_ids(mhr, pose[:1], z)
        hands = (mhr.run(pose, z)[0] / 1000.0).cpu().numpy()[:, hid]
        perm = np.random.default_rng(0).permutation(len(hid))
        probe, hold = perm[:512], perm[512:1024]

        def pen(scale, shift, sample):
            loc = np.einsum('tpi,tij->tpj', hands[:, sample] - (t + shift)[:, None], R) / scale
            return float((pm.depth(loc.reshape(-1, 3)).reshape(loc.shape[:2]) * scale).mean() * 100)

        zero = np.zeros_like(t)
        base = {'pen_fit_cm': pen(s, zero, probe), 'pen_holdout_cm': pen(s, zero, hold), 'acc_self': self_acc(t, frames)}
        rows = []
        for c in args.grid:
            raw = np.stack([solve_translation(pm, hands[i, probe], R[i], t[i], s * c, np.zeros(3),
                                              args.max_shift_cm / 100, 8, 0.002)[0] for i in range(len(frames))])
            sh = cap(smooth_per_stretch(raw, frames, args.sigma), args.max_shift_cm / 100)
            rows.append({'c': c, 'pen_fit_cm': pen(s * c, sh, probe), 'pen_holdout_cm': pen(s * c, sh, hold),
                         'acc_self': self_acc(t + sh, frames), 'shift_cm_mean': float(np.linalg.norm(sh, axis=1).mean() * 100),
                         'shift_cm_p99': float(np.percentile(np.linalg.norm(sh, axis=1), 99) * 100), '_shift': sh})
        pick = choose_shrink(rows, args.tol)
        ok = pick['pen_holdout_cm'] < base['pen_holdout_cm'] - 1e-3 and \
            pick['acc_self'] <= base['acc_self'] + args.max_acc_rise
        if ok:
            d = dict(np.load(args.src / f'{stem}.npz'))
            n_rows = len(d['object_translation'])
            d['object_translation'] = (d['object_translation'] + interpolate_rows(pick['_shift'], frames, n_rows)
                                       ).astype(d['object_translation'].dtype)
            d['object_scale'] = np.asarray(d['object_scale'] * pick['c'], dtype=d['object_scale'].dtype)
            np.savez(dest / f'{stem}.npz', **d)
        table = [{k: v for k, v in r.items() if k != '_shift'} for r in rows]
        report.append({'episode': e, 'accepted': ok, 'before': base, 'chosen_c': pick['c'],
                       'chosen': {k: v for k, v in pick.items() if k != '_shift'}, 'table': table})
        print(f"{stem}: {'ACCEPT' if ok else 'reject'} c={pick['c']:.2f}  PEN holdout {base['pen_holdout_cm']:.4f} -> "
              f"{pick['pen_holdout_cm']:.4f} cm  acc {base['acc_self']:.3f} -> {pick['acc_self']:.3f}  shift mean "
              f"{pick['shift_cm_mean']:.2f} p99 {pick['shift_cm_p99']:.2f} cm  | " +
              ' '.join(f"{r['c']:.2f}:{r['pen_holdout_cm']:.3f}" for r in rows), flush=True)
    (out / 'pen_squeeze.json').write_text(json.dumps({'src': str(args.src), 'params': {k: (str(v) if isinstance(v, Path) else v)
                                                      for k, v in vars(args).items()}, 'episodes': report}, indent=1) + '\n')


if __name__ == '__main__':
    main()
