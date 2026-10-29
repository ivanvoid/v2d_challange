#!/usr/bin/env python3
"""Reproducible post-processing: production stage-8 outputs -> packed Track 1 submission.

    run_stage.sh 1..8 (all 30 episodes)  ->  out/, submission/episodes   (production)
    track1/postprocess.py --run NAME     ->  experiments/NAME/<step>/episodes ... experiments/NAME/track1.parquet

Steps and their settings live in track1/postprocess_config.json (each with the reason it exists).
Every step reads the previous step's full episodes directory and writes a full copy; per-episode
results are combined with merge_episodes.py, which refuses changed meshes or NPZ layouts. Two GPUs
(0 and 2) run independent episodes in parallel. Ends with kit validation and local proxies.

    /opt/venvs/sam3d_body/bin/python track1/postprocess.py --run phys_v8 --commit <sha>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

T1 = Path(__file__).resolve().parent
sys.path.insert(0, str(T1))
import paths  # noqa: E402

PY = '/opt/venvs/sam3d_body/bin/python'
PY_FP = '/opt/venvs/foundation_pose/bin/python'
PY_SUB = str(paths.DATA / '.venv-submit/bin/python')
GPUS = ('0', '2')


def log(msg: str, logf: Path) -> None:
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(logf, 'a') as fh:
        fh.write(line + '\n')


def run(cmd: list, logf: Path, gpu: str | None = None, cwd: Path = T1) -> None:
    env = {**os.environ, 'V2D_DATA': str(paths.DATA)}
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = gpu
    with open(logf, 'a') as fh:
        fh.write('$ ' + ' '.join(map(str, cmd)) + '\n')
        fh.flush()
        r = subprocess.run(list(map(str, cmd)), env=env, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f'step failed ({r.returncode}): {" ".join(map(str, cmd))} — see {logf}')


def parallel(jobs: list[tuple[list, str]], logf: Path) -> None:
    """jobs: (cmd, gpu); at most one job per GPU at a time."""
    env = {**os.environ, 'V2D_DATA': str(paths.DATA)}
    pending = list(jobs)
    running: dict[str, subprocess.Popen] = {}
    fh = open(logf, 'a')
    while pending or running:
        for g in GPUS:
            if g not in running and pending:
                idx = next((i for i, (_, gg) in enumerate(pending) if gg == g), None)
                if idx is None:
                    continue
                cmd, _ = pending.pop(idx)
                fh.write(f'$ [gpu {g}] ' + ' '.join(map(str, cmd)) + '\n'); fh.flush()
                running[g] = subprocess.Popen(list(map(str, cmd)), env={**env, 'CUDA_VISIBLE_DEVICES': g}, cwd=T1,
                                              stdout=fh, stderr=subprocess.STDOUT)
        for g, p in list(running.items()):
            if p.poll() is not None:
                if p.returncode:
                    raise SystemExit(f'parallel job failed on gpu {g} — see {logf}')
                del running[g]
        time.sleep(2)
    fh.close()


def split(eps: list[int]) -> list[list[int]]:
    return [eps[0::2], eps[1::2]]


def merge(base: Path, takes: list[tuple[Path, list[int]]], out: Path, logf: Path) -> Path:
    args = [f'{d}:{",".join(map(str, e))}' for d, e in takes if e]
    run([PY, T1 / 'merge_episodes.py', '--base', base, '--take', *args, '--out', out], logf)
    return out / 'episodes'


def pack(episodes: Path, parquet: Path, logf: Path, commit: str | None = None) -> Path:
    cmd = [PY_SUB, T1 / 'pack_submission.py', '--episodes', episodes, '--out', parquet]
    if commit:
        cmd += ['--commit', f'https://github.com/ivanvoid/v2d_challange/commit/{commit}']
    run(cmd, logf)
    ev = parquet.parent / 'eval'
    run([PY_SUB, T1 / 'eval_local.py', 'extract', '--parquet', parquet, '--out', ev], logf)
    return ev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--config', type=Path, default=T1 / 'postprocess_config.json')
    ap.add_argument('--commit', default=None, help='code commit sha for the final pack (placeholder if omitted)')
    ap.add_argument('--enable', nargs='*', default=[], help='enable steps that are off in the config')
    ap.add_argument('--from_step', default=None, help='resume: start at this step, reusing earlier outputs')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    R = paths.DATA / 'experiments' / args.run
    R.mkdir(parents=True, exist_ok=True)
    logf = R / 'postprocess.log'
    (R / 'config.json').write_text(json.dumps(cfg, indent=1) + '\n')
    cur = paths.DATA / 'submission/episodes'
    started = args.from_step is None
    log(f'run {args.run}: start from {cur}', logf)
    for st in cfg['steps']:
        name, tool = st['name'], st['tool']
        enabled = st.get('enabled', True) or name in args.enable
        D = R / name
        if not started:
            if name == args.from_step:
                started = True
            else:
                if enabled and (D / 'episodes').exists():
                    cur = D / 'episodes'
                continue
        if not enabled:
            log(f'{name}: disabled', logf)
            continue
        log(f'{name} ({tool})', logf)
        if tool == 'ablate_pose_scale':
            eps = [e for e in range(30) if e not in st['skip']]
            jobs = [([PY, T1 / 'ablate_pose_scale.py', '--mode', 'smooth', '--src', paths.OUT, '--trial', D / f'trial_g{g}',
                      '--episodes', *part, '--gpu', g, '--lam_t', st['lam_t'], '--pin_source_scale'], g)
                    for g, part in zip(GPUS, split(eps))]
            parallel(jobs, logf)
            cur = merge(cur, [(D / f'trial_g{g}/episodes', part) for g, part in zip(GPUS, split(eps))], D, logf)
        elif tool == 'stationary_stool':
            run([PY, T1 / 'stationary_stool.py', '--trial', D / 'trial', '--episodes', *st['episodes']], logf, GPUS[0])
            cur = merge(cur, [(D / 'trial/episodes', st['episodes'])], D, logf)
        elif tool == 'static_lock':
            takes = []
            for e, until in st['episodes'].items():
                cmd = [PY, T1 / 'static_lock.py', '--src', cur, '--episode', e, '--out', D / f'ep{e}']
                if until:
                    cmd += ['--until', until]
                run(cmd, logf)
                rep = json.loads((D / f'ep{e}' / f'episode_{int(e):06d}_static_lock.json').read_text())
                if rep['accepted']:
                    takes.append((D / f'ep{e}/episodes', [int(e)]))
                log(f'   ep {e}: {"locked" if rep["accepted"] else "REJECTED by the held-out gate"} {rep["still_video_frames"]}', logf)
            cur = merge(cur, takes, D, logf) if takes else cur
        elif tool == 'hoop_conic':
            run([PY, T1 / 'hoop_conic.py', '--src', cur, '--episode', st['episode'], *st['args'], '--out', D], logf, GPUS[0])
            cur = merge(cur, [(D / 'episodes', [st['episode']])], D / 'merged', logf)
        elif tool == 'hoop_floor':
            run([PY, T1 / 'hoop_floor.py', '--src', cur, '--episodes', *st['episodes'], '--out', D], logf, GPUS[0])
            for r in json.loads((D / 'hoop_floor.json').read_text()):
                log(f"   ep {r['episode']}: {r['decision']}", logf)
            cur = D / 'episodes'
        elif tool == 'object_depth_scale':
            run([PY, T1 / 'object_depth_scale.py', '--src', cur, '--out', D, '--k', st['k'], '--episodes', *st['episodes']], logf)
            cur = D / 'episodes'
        elif tool == 'joint_fit':
            eps = [e for e in range(30) if e not in st['skip_episodes']]
            jobs = []
            for g, part in zip(GPUS, split(eps)):
                for e in part:
                    cmd = [PY_FP, T1 / 'joint_fit.py', '--src', cur, '--episode', e, '--out', D / f'ep{e}']
                    if str(e) in st['skip_frames']:
                        cmd += ['--skip_frames', st['skip_frames'][str(e)]]
                    jobs.append((cmd, g))
            parallel(jobs, logf)
            acc = []
            for e in eps:
                rep = json.loads((D / f'ep{e}' / f'joint_fit_episode_{e:06d}.json').read_text())
                if rep['accepted']:
                    acc.append(e)
                log(f'   ep {e}: {"ACCEPT" if rep["accepted"] else "reject"} IoU {rep["iou_mean"][0]:.3f}->{rep["iou_mean"][1]:.3f} '
                    f'contact {rep["contact_share"][0]:.2f}->{rep["contact_share"][1]:.2f} failed {[k for k, v in rep["gates"].items() if not v]}', logf)
            cur = merge(cur, [(D / f'ep{e}/episodes', [e]) for e in acc], D / 'merged', logf) if acc else cur
        elif tool == 'smooth_body':
            run([PY, T1 / 'smooth_body.py', '--src', cur, '--out', D, '--sigma', st['sigma']], logf)
            cur = D / 'episodes'
        elif tool == 'pen_squeeze':
            ev = pack(cur, D / 'input.parquet', logf)
            jobs = [([PY, T1 / 'pen_squeeze.py', '--src', cur, '--packed', ev, '--episodes', *grp['episodes'], *grp['args'],
                      '--out', D / f'g{i}'], GPUS[i % 2]) for i, grp in enumerate(st['groups'])]
            parallel(jobs, logf)
            takes = []
            for i, grp in enumerate(st['groups']):
                rep = json.loads((D / f'g{i}' / 'pen_squeeze.json').read_text())
                takes.append((D / f'g{i}/episodes', [r['episode'] for r in rep['episodes'] if r['accepted']]))
            cur = merge(cur, takes, D / 'merged', logf)
        elif tool == 'hand_depen':
            ev = pack(cur, D / 'input.parquet', logf)
            parts = split(st['episodes'])
            jobs = [([PY, T1 / 'hand_depen.py', '--src', cur, '--packed', ev, '--episodes', *part, *st['args'],
                      '--out', D / f'g{g}'], g) for g, part in zip(GPUS, parts)]
            parallel(jobs, logf)
            cur = merge(cur, [(D / f'g{g}/episodes', part) for g, part in zip(GPUS, parts)], D / 'merged', logf)
        else:
            raise SystemExit(f'unknown tool {tool}')
        log(f'   -> {cur}', logf)
    final = R / 'track1.parquet'
    ev = pack(cur, final, logf, args.commit)
    run([PY, T1 / 'eval_local.py', 'proxies', '--out', ev], logf, GPUS[0])
    run([PY_SUB, '-c', 'import sys,json;from pathlib import Path;sys.path.insert(0,"submit");import v2d_submit as v;'
         f'r=v.validate_submission(Path("{final}"),"track_1",json.load(open("config/tracks.json")));print("kit validate OK",len(r))'],
        logf, cwd=paths.KIT)
    (R / 'final_episodes.txt').write_text(str(cur) + '\n')
    log(f'done: {final} (episodes {cur})', logf)


if __name__ == '__main__':
    main()
