#!/usr/bin/env python3
"""Local test set from Track 2 Tier 1 — EVALUATION ONLY, nothing from it may enter a submission.

Track 2 Tier 1 ships 30 other episodes from the same capture setup (same 1536x1152 30 fps
camera) with the multi-view reference: object pose + scanned mesh and SOMA-X human root. We
run our unchanged Track 1 pipeline on some of them and score the result against that reference.

Two separate trees keep reference data out of the pipeline's reach:
  --root  a V2D_DATA tree for the pipeline: video symlinks, metadata (object name + prompt) and
          submission/scored_spans.json. NO pose, mesh or ground-plane file of the reference.
  --gt    reference arrays (gt/episode_XXXXXX.npz), read only by `eval`.

  setup  (.venv-submit: pandas)  build both trees for --episodes
  eval   (sam3d_body venv: torch + MHR + trimesh)  score packed episode NPZs against the reference

Scored frames (the Track 1 scorer uses hand-contact frames, which Track 2 does not publish):
visible frames from 15 frames before the object first leaves its starting pose to 15 frames
after it reaches its final pose (moved = > 5 cm or > 10 deg from that rest pose for 5 frames in a
row; the reference has ~1-2 cm of capture noise at rest).

Registration (Track 2 has no camera extrinsics): one Sim(3) maps our camera frame onto the
reference world, fitted on the scored frames from our MHR pelvis to the reference root, with a
constant offset in the reference root frame (the two skeletons' roots differ). This is NOT the
scorer's first-frame body alignment: it removes our global body error, so the object numbers
here are about the object relative to a correctly registered camera. Metrics (cm):
  cd_o        scorer formula: per frame sum of the two directed mean nearest distances
  cd_o_centred  the same after moving our object's centroid onto the reference centroid
  centroid    centroid error, split into depth (along the camera ray) and lateral parts
  size_ratio  our object's surface area^0.5 / reference's (after the registration scale)
  acc_o       scorer formula on translations; acc_o_zero = the same for a constant-velocity guess
  pelvis      registered pelvis vs reference root (+offset): mean error and its depth part
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

T2 = Path('/local-data/ivan/v2d/challenge/track_2/tier_1_multiview_caption')
SHARE = ('weights', 'models', 'v2d_submission_kit', '.venv-submit', 'mano_v1_2', 'smpl')


# --------------------------------------------------------------------------- pure helpers (tested)
def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    th = np.linalg.norm(v, axis=-1, keepdims=True)
    k = v / np.maximum(th, 1e-12)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def manipulation_frames(t: np.ndarray, R: np.ndarray, visible: np.ndarray, *, move_m: float = .05,
                        move_deg: float = 10., pad: int = 15, hold: int = 5) -> np.ndarray:
    """Visible frames from just before the object leaves its first pose to just after its last.

    The reference has ~1-2 cm of capture noise at rest, so 'moved' means > move_m or > move_deg
    from the median rest pose of the first/last `hold`*3 visible frames, for `hold` frames in a row."""
    vis = np.flatnonzero(visible)
    if len(vis) < 3 * hold:
        return np.zeros(0, int)

    def moved(ref_idx):
        t0 = np.median(t[ref_idx], axis=0)
        R0 = R[ref_idx[len(ref_idx) // 2]]
        dt = np.linalg.norm(t[vis] - t0, axis=1) > move_m
        c = (np.einsum('nij,ij->n', R[vis], R0) - 1) / 2
        m = dt | (np.degrees(np.arccos(np.clip(c, -1, 1))) > move_deg)
        run = np.convolve(m.astype(int), np.ones(hold, int), 'same') >= hold   # sustained motion only
        return run
    a, b = moved(vis[:3 * hold]), moved(vis[-3 * hold:])
    if not a.any() or not b.any():
        return np.zeros(0, int)
    lo, hi = vis[np.flatnonzero(a)[0]] - pad, vis[np.flatnonzero(b)[-1]] + pad
    return vis[(vis >= lo) & (vis <= hi)]


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimising |s R src + t - dst| (column-vector convention)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / (a ** 2).sum(1).mean())
    return s, R, md - s * R @ ms


def register(ours: np.ndarray, ref: np.ndarray, ref_rot: np.ndarray, iters: int = 2000, tol: float = 1e-9):
    """Sim(3) ours->reference plus a constant offset o in the reference root frame:
    s R ours + t = ref + ref_rot o. Returns s, R, t, o, residual (m, per frame)."""
    o = np.zeros(3)
    A = ref_rot.reshape(-1, 3)
    for _ in range(iters):
        s, R, t = umeyama(ours, ref + ref_rot @ o)
        mapped = (s * (R @ ours.T)).T + t
        o_new = np.linalg.lstsq(A, (mapped - ref).reshape(-1), rcond=None)[0]   # ref_rot o = mapped - ref
        done = np.linalg.norm(o_new - o) < tol
        o = o_new
        if done:
            break
    s, R, t = umeyama(ours, ref + ref_rot @ o)
    mapped = (s * (R @ ours.T)).T + t
    return s, R, t, o, np.linalg.norm(mapped - (ref + ref_rot @ o), axis=1)


FLIP = np.diag([1.0, -1.0, -1.0])


def human_to_opencv(t: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The released human root is in the y-up/z-back frame (x, -y, -z of the OpenCV world the object
    poses and ground plane use; the same flip MHR applies). Checked on the data: without it the
    reference object sits ~6 m from the reference person during manipulation, with it < 1 m."""
    return t @ FLIP, FLIP @ R @ FLIP


def accel(x: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """|second difference| over consecutive scored triplets (the scorer's stretches)."""
    ok = (np.diff(frames)[1:] == 1) & (np.diff(frames)[:-1] == 1)
    a = x[:-2] - 2 * x[1:-1] + x[2:]
    return a[ok]


def reference_leaks(root: Path) -> list[str]:
    """Files under the pipeline root (shared symlinked dirs skipped) that could carry reference
    data: anything resolving into the Track 2 tree except the videos, pose tables, ground planes,
    support stages. Must be empty. (Our own generated meshes in out/ are fine.)"""
    bad = []
    for p in Path(root).rglob('*'):
        if any(part in SHARE for part in p.relative_to(root).parts[:1]):
            continue
        name = p.name.lower()
        into_t2 = T2.resolve() in p.resolve().parents and p.suffix != '.mp4'
        if into_t2 or p.suffix.lower() in ('.parquet', '.usda') or 'ground_plane' in name:
            bad.append(str(p))
    return bad


def chamfer_sum(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial import cKDTree
    return float(cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())


# --------------------------------------------------------------------------- setup
def setup(args) -> None:
    import pandas as pd
    root, gt = args.root.resolve(), args.gt.resolve()
    prod = Path('/local-data/ivan/v2d').resolve()
    if root == prod or prod in root.parents or root in gt.parents or gt == root:
        raise SystemExit('--root must be a fresh tree outside production and must not contain --gt')
    info = json.loads((T2 / 'meta/info.json').read_text())
    prompts = {o['name']: o['prompt'] for o in info['objects']}
    meta = {json.loads(l)['episode_index']: json.loads(l) for l in open(T2 / 'meta/episodes_metadata.jsonl')}
    vdir = root / 'challenge/track_1/videos/chunk-000/observation.images.exo_camera'
    vdir.mkdir(parents=True, exist_ok=True)
    (root / 'challenge/track_1/meta').mkdir(parents=True, exist_ok=True)
    (root / 'submission').mkdir(exist_ok=True)
    gt.mkdir(parents=True, exist_ok=True)
    for name in SHARE:
        if not (root / name).exists() and (prod / name).exists():
            (root / name).symlink_to(prod / name)
    rows, spans = [], {}
    for e in args.episodes:
        m = meta[e]
        stem = f'episode_{e:06d}'
        src = T2 / f'videos/chunk-000/observation.images.exo_camera/{stem}.mp4'
        if not (vdir / f'{stem}.mp4').exists():
            (vdir / f'{stem}.mp4').symlink_to(src)
        df = pd.read_parquet(T2 / f'data/chunk-000/{stem}.parquet')
        pose = np.stack(df['observation.object.pose'].to_numpy()).astype(float)
        vis = df['observation.object.visible'].to_numpy().astype(bool)
        R = np.tile(np.eye(3), (len(pose), 1, 1))
        R[vis] = quat_wxyz_to_matrix(pose[vis, 3:])
        frames = manipulation_frames(pose[:, :3], R, vis)
        if len(frames) < 30:
            raise SystemExit(f'{stem}: only {len(frames)} manipulation frames')
        spans[str(e)] = {'first': int(frames[0]), 'last': int(frames[-1]), 'n_scored': int(len(frames)),
                         'frames': frames.tolist()}
        rows.append({'episode_index': e, 'sequence_id': m['sequence_id'], 'camera': 'exo_camera',
                     'object': m['object'], 'object_prompt': prompts[m['object']],
                     'video_key': 'observation.images.exo_camera'})
        plane = json.loads((T2 / m['ground_plane']).read_text())
        np.savez(gt / f'{stem}.npz', object_t=pose[:, :3], object_R=R, visible=vis, frames=frames,
                 human_t=np.stack(df['observation.human.translation'].to_numpy()).astype(float),
                 human_pose=np.stack(df['observation.human.pose'].to_numpy()).astype(float),
                 ground_plane=np.asarray(plane['plane'], float),
                 mesh=str(T2 / m['mesh']), object=m['object'])
        print(f'{stem}: {m["object"]:20s} {len(df)} frames, scored {len(frames)} ({frames[0]}..{frames[-1]})')
    (root / 'challenge/track_1/meta/episodes_metadata.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (root / 'submission/scored_spans.json').write_text(json.dumps(spans) + '\n')
    leaks = reference_leaks(root)
    if leaks:
        raise SystemExit(f'reference data inside the pipeline root: {leaks[:5]}')
    print(f'pipeline root {root} (no reference data), reference {gt}')


# --------------------------------------------------------------------------- eval
def evaluate_episode(npz: Path, mesh_obj: Path, gt_npz: Path, floor: Path | None, n_points: int = 3000) -> dict:
    import sys
    import torch
    import trimesh
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paths
    sys.path.insert(0, str(paths.KIT_TRACK1))
    from mesh_to_mhr_params import MHR
    g = np.load(gt_npz, allow_pickle=False)
    frames = g['frames']
    d = np.load(npz)
    if d['pose'].shape[0] <= frames.max():
        raise ValueError(f'{npz}: packed rows end at {d["pose"].shape[0]} < last scored frame {frames.max()}')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    mhr = MHR(str(paths.DATA / 'weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt'), dev, 64, 'float32')
    pose = torch.as_tensor(d['pose'][frames], dtype=torch.float64, device=dev)
    z = torch.as_tensor(np.concatenate([d['scales'], d['shape']])[None], dtype=torch.float64, device=dev)
    _, j = mhr.run(pose, z)
    pelvis = j.cpu().numpy()[:, 1]
    ref_root, ref_rot = human_to_opencv(g['human_t'][frames], rotvec_to_matrix(g['human_pose'][frames, :3]))
    s, R, t, o, res = register(pelvis, ref_root, ref_rot)
    to_world = lambda x: (s * (R @ x.T)).T + t                                   # noqa: E731
    cam = t                                                                       # our camera centre
    ours_mesh = trimesh.load(mesh_obj, force='mesh', process=False)
    gt_scene = trimesh.load(str(g['mesh']))
    gt_mesh = gt_scene.dump(concatenate=True) if hasattr(gt_scene, 'dump') else gt_scene
    P = trimesh.sample.sample_surface(ours_mesh, n_points, seed=0)[0]
    G = trimesh.sample.sample_surface(gt_mesh, n_points, seed=0)[0]
    so = float(d['object_scale'])
    oR, ot = d['object_rotation'][frames], d['object_translation'][frames]
    gR, gtt = g['object_R'][frames], g['object_t'][frames]
    cd, cdc, cerr, cdepth, clat = [], [], [], [], []
    for i in range(len(frames)):
        a = to_world(so * P @ oR[i].T + ot[i])
        b = G @ gR[i].T + gtt[i]
        cd.append(chamfer_sum(a, b))
        cdc.append(chamfer_sum(a - a.mean(0) + b.mean(0), b))
        e = a.mean(0) - b.mean(0)
        ray = b.mean(0) - cam
        ray /= np.linalg.norm(ray)
        cerr.append(np.linalg.norm(e)); cdepth.append(e @ ray); clat.append(np.linalg.norm(e - (e @ ray) * ray))
    ours_t = to_world(ot)
    acc_err = np.linalg.norm(accel(ours_t, frames) - accel(gtt, frames), axis=1)
    acc_zero = np.linalg.norm(accel(gtt, frames), axis=1)
    pel_w = to_world(pelvis)
    pel_ref = ref_root + np.einsum('nij,j->ni', ref_rot, o)
    pe = pel_w - pel_ref
    rays = pel_ref - cam
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    gap = float(np.median(np.linalg.norm(gtt - ref_root, axis=1)))
    if gap > 2.0:
        raise ValueError(f'reference object is {gap:.1f} m from the reference person: frame convention broken')
    out = {'frames': int(len(frames)), 'ref_object_to_root_m': gap, 'registration_scale': s, 'registration_residual_cm': float(res.mean() * 100),
           'root_offset_cm': (o * 100).round(1).tolist(),
           'cd_o': float(np.mean(cd) * 100), 'cd_o_centred': float(np.mean(cdc) * 100),
           'centroid_cm': float(np.mean(cerr) * 100), 'centroid_depth_cm': float(np.mean(cdepth) * 100),
           'centroid_depth_abs_cm': float(np.mean(np.abs(cdepth)) * 100), 'centroid_lateral_cm': float(np.mean(clat) * 100),
           'size_ratio': float(np.sqrt(ours_mesh.area) * so * s / np.sqrt(gt_mesh.area)),
           'acc_o': float(acc_err.mean() * 100), 'acc_o_zero': float(acc_zero.mean() * 100),
           'acc_o_self': float(np.linalg.norm(accel(ours_t, frames), axis=1).mean() * 100),
           'pelvis_cm': float(np.linalg.norm(pe, axis=1).mean() * 100),
           'pelvis_depth_cm': float(np.mean(np.einsum('ni,ni->n', pe, rays)) * 100),
           'per_frame': {'frames': frames.tolist(), 'cd_o': (np.asarray(cd) * 100).round(2).tolist(),
                         'centroid_depth_cm': (np.asarray(cdepth) * 100).round(2).tolist()}}
    if floor is not None and floor.exists() and 'ground_plane' in g.files and g['ground_plane'].size == 4:
        n_cam = np.asarray(json.loads(floor.read_text())['normal'], float)
        n_ref = g['ground_plane'][:3] / np.linalg.norm(g['ground_plane'][:3])
        c = abs(float((R @ n_cam) @ n_ref))
        out['floor_normal_angle_deg'] = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return out


def evaluate(args) -> None:
    rows = []
    for e in args.episodes:
        stem = f'episode_{e:06d}'
        floor = args.root / 'out' / stem / 'floor.json'
        r = {'episode': e, **evaluate_episode(args.episodes_dir / f'{stem}.npz',
                                              args.episodes_dir / f'{stem}_object.obj',
                                              args.gt / f'{stem}.npz', floor)}
        rows.append(r)
        print(f"{stem}: CD-O {r['cd_o']:6.1f} (centred {r['cd_o_centred']:5.1f})  centroid {r['centroid_cm']:5.1f} "
              f"(depth {r['centroid_depth_cm']:+6.1f}, lateral {r['centroid_lateral_cm']:5.1f})  size x{r['size_ratio']:.2f}  "
              f"ACC-O {r['acc_o']:.3f} (zero-acc {r['acc_o_zero']:.3f}, self {r['acc_o_self']:.3f})  "
              f"pelvis {r['pelvis_cm']:5.1f} (depth {r['pelvis_depth_cm']:+5.1f})  reg s {r['registration_scale']:.3f} "
              f"res {r['registration_residual_cm']:.1f} floor {r.get('floor_normal_angle_deg', float('nan')):.1f} deg", flush=True)
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    print('MEAN ' + '  '.join(f'{k} {v:.3f}' for k, v in mean.items()))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({'episodes_dir': str(args.episodes_dir), 'mean': mean, 'episodes': rows}, indent=1) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('setup')
    a.add_argument('--root', type=Path, required=True)
    a.add_argument('--gt', type=Path, required=True)
    a.add_argument('--episodes', type=int, nargs='+', required=True)
    b = sub.add_parser('eval')
    b.add_argument('--root', type=Path, required=True)
    b.add_argument('--gt', type=Path, required=True)
    b.add_argument('--episodes_dir', type=Path, required=True)
    b.add_argument('--episodes', type=int, nargs='+', required=True)
    b.add_argument('--out', type=Path)
    args = ap.parse_args()
    setup(args) if args.cmd == 'setup' else evaluate(args)


if __name__ == '__main__':
    main()
