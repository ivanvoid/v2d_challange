#!/usr/bin/env python3
"""After a PASSING isolated hoop stage-6 gate, run stages 7/8 and partial validation.

No production path is written. QA body export is intentionally disabled: the
trial/body directory is a read-only symlink to production. Full 30-episode
submission assembly is a separate step, gated on the packed partial result.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import paths

ROOT = Path(__file__).resolve().parent
DATA = paths.DATA
REVIEW = Path('/opt/venvs/sam3d_body/bin/python')
SUBMIT = DATA / '.venv-submit/bin/python'
RENDER = Path('/opt/venvs/wilor/bin/python')
MHR = DATA / 'weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt'


def require_stage6_gate(trial: Path) -> dict:
    p = trial / 'stage6_gate.json'
    if not p.is_file():
        raise ValueError(f'{p} missing; cannot run downstream stages on unvalidated tracking')
    result = json.loads(p.read_text())
    if result.get('pass') is not True:
        raise ValueError('hoop stage-6 gate rejected changed or unaffected frames')
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trial', type=Path, required=True)
    ap.add_argument('--gpu', choices=('0', '2'), default='2')
    args = ap.parse_args()
    trial = args.trial.resolve()
    if not trial.is_relative_to(DATA.resolve() / 'experiments'):
        raise SystemExit('trial must stay under V2D_DATA/experiments')
    require_stage6_gate(trial)
    w = trial / 'out/episode_000000'
    prod = DATA / 'out/episode_000000'
    for path in (w / 'track/states.json', w / 'mesh_scaled.obj', w / 'body/mhr_fit.npz'):
        if not path.exists():
            raise FileNotFoundError(path)
    if (w / 'poses').is_symlink() or (w / 'track').is_symlink():
        raise ValueError('stage 6/7 outputs must be private trial directories')
    if not (w / 'object_track.json').exists():
        (w / 'object_track.json').symlink_to((prod / 'object_track.json').resolve())
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu, 'V2D_DATA': str(DATA)}

    def run(*cmd):
        print('RUN', ' '.join(map(str, cmd)), flush=True)
        subprocess.run(list(map(str, cmd)), env=env, check=True)

    run(REVIEW, ROOT / 'stage7_smooth.py', '--episode', 0, '--out_root', trial / 'out')
    run(REVIEW, ROOT / 'stage8_scale.py', '--episode', 0, '--out_root', trial / 'out', '--model', MHR)
    meta = json.loads((w / 'track1_meta.json').read_text())
    offset = meta['frame_offset']
    sub = trial / 'episodes'
    sub.mkdir(exist_ok=True)
    run(SUBMIT, ROOT / 'build_episode.py', '--episode', 0, '--ep_dir', w,
        '--object_mesh', DATA / 'submission/episodes/episode_000000_object.obj',
        '--object_poses', w / 'poses_final', '--frame_offset', offset, '--out_dir', sub)
    reports = trial / 'reports'
    reports.mkdir(exist_ok=True)
    run(REVIEW, ROOT / 'qa_episode.py', '--episode', 0, '--ep_dir', w,
        '--sub_dir', sub, '--spans', DATA / 'submission/scored_spans.json',
        '--model', MHR, '--frame_offset', offset, '--out', reports / 'episode_000000.json')
    review = trial / 'review'
    review.mkdir(exist_ok=True)
    first, last = meta['scored_first'] - offset, meta['scored_last'] - offset
    frames = ','.join(str(round(first + (last-first)*i/5)) for i in range(6))
    run(RENDER, ROOT.parent / 'render_results.py', '--frames_dir', w / 'frames',
        '--intrinsics', w / 'intrinsics_stable.json',
        '--body_mesh', prod / 'body/submitted_body.pt', '--object_mesh', sub / 'episode_000000_object.obj',
        '--object_poses_dir', w / 'poses_final', '--frames', frames,
        '--title', 'isolated hoop mask + stage6/7/8 pilot', '--output', review / 'episode_000000_result.png')
    run(SUBMIT, ROOT / 'pack_submission.py', '--partial', '--episodes', sub,
        '--out', trial / 'partial.parquet')
    run(SUBMIT, ROOT / 'eval_local.py', 'extract', '--parquet', trial / 'partial.parquet',
        '--out', trial / 'eval')
    run(REVIEW, ROOT / 'eval_local.py', 'proxies', '--out', trial / 'eval')
    run(REVIEW, ROOT / 'eval_pose_ablation.py', '--control',
        DATA / 'logs/eval/beta/prodnow/proxies.json',
        '--candidate', trial / 'eval/proxies.json', '--out', trial / 'compare_production.json')
    print('Hoop episode-0 partial validation finished. No full submission assembled or uploaded.', flush=True)


if __name__ == '__main__':
    main()
