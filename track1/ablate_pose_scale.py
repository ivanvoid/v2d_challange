#!/usr/bin/env python3
"""Run isolated stage-7/8 or stage-8-only experiments using symlinked read-only inputs.

The requested --trial must be inside V2D_DATA/experiments and may not be production out/.
Only trial/out/<episode>/track and trial/out/<episode>/poses_{smoothed,final} are written.
Outputs are packed as a PARTIAL parquet; not uploadable until separately selected and
assembled against the 30-episode control.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import paths

ROOT = Path(__file__).resolve().parent
PY_REVIEW = Path('/opt/venvs/sam3d_body/bin/python')
PY_SUB = paths.DATA / '.venv-submit/bin/python'
MHR = paths.DATA / 'weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt'


def prepare_episode(src: Path, dest: Path, *, smooth: bool) -> None:
    """Symlink immutable inputs, leaving generated pose/track paths real and private."""
    for required in ('body/mhr_fit.npz', 'mesh_scaled.obj', 'depth', 'intrinsics_stable.json',
                     'poses', 'poses_smoothed', 'track/states.json', 'track1_meta.json'):
        if not (src / required).exists():
            raise FileNotFoundError(src / required)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'track').mkdir(exist_ok=True)
    for name in ('body', 'depth', 'intrinsics_stable.json', 'mesh_scaled.obj', 'track1_meta.json'):
        target = dest / name
        if not target.exists():
            target.symlink_to((src / name).resolve())
    if smooth:
        for name, target in (('poses', dest / 'poses'), ('track/states.json', dest / 'track/states.json')):
            if not target.exists():
                target.symlink_to((src / name).resolve())
        if (dest / 'poses_smoothed').is_symlink():
            raise ValueError(f'{dest}/poses_smoothed must be private output, not a symlink')
    elif not (dest / 'poses_smoothed').exists():
        (dest / 'poses_smoothed').symlink_to((src / 'poses_smoothed').resolve())
    if (dest / 'poses_final').is_symlink() or (dest / 'track/scale_match.json').is_symlink():
        raise ValueError('stage 8 outputs must not link to production')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=('scale', 'smooth'), required=True)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--trial', type=Path, required=True)
    ap.add_argument('--episodes', type=int, nargs='+', required=True)
    ap.add_argument('--gpu', choices=('0', '2'), required=True)
    ap.add_argument('--lam_t', type=float, default=50.0)
    ap.add_argument('--k_min', type=float, default=0.8)
    ap.add_argument('--k_max', type=float, default=1.25)
    ap.add_argument('--k_step', type=float, default=0.025)
    args = ap.parse_args()
    data = paths.DATA.resolve()
    src, trial = args.src.resolve(), args.trial.resolve()
    if not trial.is_relative_to(data / 'experiments') or trial == src or src.is_relative_to(trial):
        raise SystemExit('trial must be a separate directory under V2D_DATA/experiments')
    if args.lam_t < 0 or args.k_min <= 0 or args.k_min > args.k_max or args.k_step <= 0:
        raise SystemExit('invalid smoothing or scale-search parameters')
    if not args.episodes or len(set(args.episodes)) != len(args.episodes):
        raise SystemExit('specify unique episodes')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu, 'V2D_DATA': str(data)}
    out, sub = trial / 'out', trial / 'episodes'
    sub.mkdir(parents=True, exist_ok=True)
    for e in args.episodes:
        stem = f'episode_{e:06d}'
        prepare_episode(src / stem, out / stem, smooth=args.mode == 'smooth')
        if args.mode == 'smooth':
            subprocess.run([str(PY_REVIEW), str(ROOT / 'stage7_smooth.py'), '--episode', str(e),
                            '--out_root', str(out), '--lam_t', str(args.lam_t)], env=env, check=True)
        subprocess.run([str(PY_REVIEW), str(ROOT / 'stage8_scale.py'), '--episode', str(e),
                        '--out_root', str(out), '--model', str(MHR), '--k_min', str(args.k_min),
                        '--k_max', str(args.k_max), '--k_step', str(args.k_step)], env=env, check=True)
        offset = json.loads((src / stem / 'track1_meta.json').read_text())['frame_offset']
        subprocess.run([str(PY_SUB), str(ROOT / 'build_episode.py'), '--episode', str(e),
                        '--ep_dir', str(out / stem), '--object_poses', str(out / stem / 'poses_final'),
                        '--frame_offset', str(offset), '--out_dir', str(sub)], env=env, check=True)
        print(f'{stem}: isolated {args.mode} trial complete', flush=True)
    parquet = trial / 'partial.parquet'
    subprocess.run([str(PY_SUB), str(ROOT / 'pack_submission.py'), '--partial', '--episodes', str(sub),
                    '--out', str(parquet)], env=env, check=True)
    manifest = {"mode": args.mode, "source": str(src), "episodes": args.episodes, "gpu": args.gpu,
                "lam_t": args.lam_t if args.mode == 'smooth' else None,
                "k_min": args.k_min, "k_max": args.k_max, "k_step": args.k_step,
                "partial_parquet": str(parquet), "uploadable": False}
    (trial / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
