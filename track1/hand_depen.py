#!/usr/bin/env python3
"""H4: remove hand-object penetration by moving the HANDS (wrist + finger MHR parameters), not the object.

The object pushes in pen_squeeze.py cost CD-O (phys_v5: PEN -45 %, CD-O +0.11). Here the object
arrays stay byte-identical; only the 60 hand parameters (fingers 68-121, wrist rotations 37-39 and
47-49, which move the hand but not the wrist position) change on scored frames.

Per episode, on the packed scored frames:
  sdf      signed distance of the BUDGETED object mesh (as packed), on a grid in object coordinates:
           sign from the scorer's winding-number inside test, distance from dense surface samples;
           sampled trilinearly (differentiable)
  losses   pen      every hand vertex outside the surface by >= --margin_mm
           contact  frames that touched before (min hand distance < 1.5 cm) keep their closest 5 %
                    of hand vertices within 5 mm of the surface (RHINO / IPMAN), so hands do not
                    simply open away from the object
           reg      parameter change^2 (stay near the fit), temp: second differences over
                    consecutive scored frames (ACC-H)
Writes <out>/episodes (a full copy of --src with changed `pose` rows), hand_depen.json and plots.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

WRIST_COLS = [37, 38, 39, 47, 48, 49]
ARM_COLS = list(range(30, 37)) + list(range(40, 47))        # shoulder/upper arm + elbow/forearm, both sides


def hand_param_cols(arm: bool = False) -> list[int]:
    return (ARM_COLS if arm else []) + WRIST_COLS + list(range(68, 122))


def consecutive_triplets(frames: np.ndarray) -> np.ndarray:
    """Indices i with frames[i-1], frames[i], frames[i+1] consecutive (centre of each triplet)."""
    f = np.asarray(frames)
    ok = (f[1:-1] - f[:-2] == 1) & (f[2:] - f[1:-1] == 1)
    return np.flatnonzero(ok) + 1


class SdfGrid:
    """Signed distance (negative inside) of a closed-ish mesh on a regular grid, trilinear lookup."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, res: int = 72, margin: float = 0.04,
                 inside_fn=None):
        import trimesh
        from scipy.spatial import cKDTree
        lo, hi = vertices.min(0) - margin, vertices.max(0) + margin
        n = np.maximum(8, np.round(res * (hi - lo) / (hi - lo).max()).astype(int))
        axes = [np.linspace(lo[k], hi[k], n[k]) for k in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        surf = trimesh.sample.sample_surface_even(mesh, 200_000, seed=0)[0]
        dist = cKDTree(surf).query(grid)[0]
        inside = inside_fn(grid) if inside_fn is not None else mesh.contains(grid)
        self.lo, self.hi, self.n = lo, hi, n
        self.values = np.where(inside, -dist, dist).reshape(n).astype(np.float32)

    def torch(self, device):
        import torch
        self.t_values = torch.as_tensor(self.values, device=device)[None, None]   # [1,1,X,Y,Z]
        self.t_lo = torch.as_tensor(self.lo, dtype=torch.float32, device=device)
        self.t_hi = torch.as_tensor(self.hi, dtype=torch.float32, device=device)
        return self

    def __call__(self, pts):
        """pts [N,3] torch (object coordinates) -> sdf [N]; outside the grid: clamped (far = outside)."""
        import torch
        import torch.nn.functional as F
        g = (pts - self.t_lo) / (self.t_hi - self.t_lo) * 2 - 1                    # [-1, 1]
        # grid_sample wants (x, y, z) = (W, H, D) = our (Z, Y, X) index order
        s = F.grid_sample(self.t_values, g.flip(-1).view(1, -1, 1, 1, 3).float(), align_corners=True,
                          padding_mode='border')
        return s.view(-1).to(pts.dtype)


def mhr_forward(mhr, pose, z):
    """Differentiable MHR: pose [N,136], z [1,113] -> vertices [N,V,3] in metres (Track 1 frame)."""
    import torch
    from mesh_to_mhr_params import NS
    zz = z.expand(len(pose), -1).to(mhr.mdtype)
    params = torch.cat([pose.to(mhr.mdtype), zz[:, :NS]], 1)
    expr = torch.zeros(len(pose), 72, dtype=mhr.mdtype, device=pose.device)
    v, _ = mhr.model(zz[:, NS:].contiguous(), params, expr, True)
    return v.double() * mhr.flip * 0.01


def run_episode(e, args, mhr, dev):
    import torch
    from pen_fast import PenetrationMesh
    from qa_episode import hand_vertex_ids
    stem = f'episode_{e:06d}'
    p = np.load(args.packed / f'{stem}.npz')
    frames, R, t, s = p['frames'], p['R'], p['t'], float(p['s'])
    pm = PenetrationMesh(p['mesh_v'], p['mesh_f'])
    sdf = SdfGrid(p['mesh_v'], p['mesh_f'], res=args.res, inside_fn=lambda q: np.abs(pm.winding(q)) > 0.5).torch(dev)
    z = torch.as_tensor(np.concatenate([p['scales'], p['shape']])[None], dtype=torch.float64, device=dev)
    pose0 = torch.as_tensor(p['pose'], dtype=torch.float64, device=dev)
    hid = torch.as_tensor(hand_vertex_ids(mhr, pose0[:1], z), device=dev)
    col_list = hand_param_cols(args.arm)
    cols = torch.as_tensor(col_list, device=dev)
    reg_w = torch.as_tensor([args.w_reg_arm if c in ARM_COLS else args.w_reg for c in col_list],
                            dtype=torch.float64, device=dev)
    Rt = torch.as_tensor(R, dtype=torch.float64, device=dev)
    tt = torch.as_tensor(t, dtype=torch.float64, device=dev)
    T = len(frames)
    Kd = json.loads((paths.OUT / stem / 'intrinsics_stable.json').read_text())

    def project(x):                                                              # [..., 3] -> pixels
        return torch.stack([Kd['fx'] * x[..., 0] / x[..., 2] + Kd['cx'], Kd['fy'] * x[..., 1] / x[..., 2] + Kd['cy']], -1)

    def hand_sdf(delta, idx):
        pose = pose0[idx].clone()
        pose[:, cols] = pose[:, cols] + delta[idx]
        hv = mhr_forward(mhr, pose, z)[:, hid]                                   # [n, H, 3]
        local = torch.einsum('nhi,nij->nhj', hv - tt[idx, None], Rt[idx]) / s
        return sdf(local.reshape(-1, 3)).view(len(idx), -1) * s, hv            # metres, scene scale

    with torch.no_grad():
        sd0, hv0 = [], []
        for a in range(0, T, args.chunk):
            idx = torch.arange(a, min(T, a + args.chunk), device=dev)
            d_, h_ = hand_sdf(torch.zeros(T, len(cols), dtype=torch.float64, device=dev), idx)
            sd0.append(d_); hv0.append(h_)
        sd0, hv0 = torch.cat(sd0), torch.cat(hv0)
    touching = (sd0.min(1).values < 0.015)
    uv0 = project(hv0)
    delta = torch.zeros(T, len(cols), dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=args.lr)
    trip = torch.as_tensor(consecutive_triplets(frames), device=dev)
    margin = args.margin_mm / 1000
    k = max(1, int(0.05 * len(hid)))
    for step in range(args.steps):
        opt.zero_grad()
        for a in range(0, T, args.chunk):
            idx = torch.arange(a, min(T, a + args.chunk), device=dev)
            sd, hv = hand_sdf(delta, idx)
            pen = torch.relu(margin - sd).pow(2).sum(1)
            px = (project(hv) - uv0[idx]).norm(dim=-1)                          # image motion of each hand vertex
            img = torch.relu(px - args.px_tol).pow(2).mean(1) * args.w_2d
            close = torch.topk(sd, k, dim=1, largest=False).values
            cont = (torch.relu(close - 0.005).pow(2).mean(1)) * touching[idx]
            loss = (pen.sum() * args.w_pen + cont.sum() * args.w_contact + img.sum()) / T
            loss.backward()
        reg = (reg_w * delta.pow(2)).sum(1).mean()
        acc = delta[trip - 1] - 2 * delta[trip] + delta[trip + 1]
        temp = args.w_temp * acc.pow(2).sum(1).mean() if len(trip) else 0 * reg
        (reg + temp).backward()
        opt.step()
    with torch.no_grad():
        sd1, hv1 = [], []
        for a in range(0, T, args.chunk):
            idx = torch.arange(a, min(T, a + args.chunk), device=dev)
            d_, h_ = hand_sdf(delta, idx)
            sd1.append(d_); hv1.append(h_)
        sd1, hv1 = torch.cat(sd1), torch.cat(hv1)
    move = (hv1 - hv0).norm(dim=-1)                                              # [T, H] metres
    px_move = (project(hv1) - uv0).norm(dim=-1)                                  # [T, H] pixels
    # exact scorer-style PEN on a fixed 512-vertex subset (as eval_local.py), before / after
    sub = np.random.default_rng(0).choice(len(hid), size=min(512, len(hid)), replace=False)

    def exact_pen(hv):
        h = hv[:, sub].cpu().numpy()
        loc = np.einsum('tpi,tij->tpj', h - t[:, None], R) / s
        return float((pm.depth(loc.reshape(-1, 3)).reshape(loc.shape[:2]) * s).mean() * 100), \
            (pm.depth(loc.reshape(-1, 3)).reshape(loc.shape[:2]) * s).mean(1) * 100
    pen0, pf0 = exact_pen(hv0)
    pen1, pf1 = exact_pen(hv1)
    gap0 = sd0.clamp(min=0).min(1).values.cpu().numpy() * 100
    gap1 = sd1.clamp(min=0).min(1).values.cpu().numpy() * 100
    top = np.argsort(-move.mean(1).cpu().numpy())[:3]
    viz = {'frames': frames[top], 'before': hv0[top].cpu().numpy(), 'after': hv1[top].cpu().numpy(),
           'object': [((s * p['mesh_v']) @ R[i].T + t[i]) for i in top]}
    new_pose = p['pose'].copy()
    new_pose[:, col_list] += delta.detach().cpu().numpy().astype(new_pose.dtype)
    dd = delta.detach().abs().cpu().numpy()
    is_arm = np.array([c in ARM_COLS for c in col_list])
    return {'episode': e, 'frames': frames, 'new_pose_scored': new_pose, 'viz': viz,
            'stats': {'pen_before_cm': pen0, 'pen_after_cm': pen1,
                      'hand_move_mean_cm': float(move.mean() * 100), 'hand_move_p99_cm': float(torch.quantile(move.flatten()[::7], .99) * 100),
                      'touching_frames': int(touching.sum()), 'frames': int(T),
                      'image_move_mean_px': float(px_move.mean()), 'image_move_p99_px': float(torch.quantile(px_move.flatten()[::7], .99)),
                      'contact_before': float((gap0 < 2).mean()), 'contact_after': float((gap1 < 2).mean()),
                      'param_change_max_rad': float(dd.max()),
                      'finger_wrist_change_p99_rad': float(np.percentile(dd[:, ~is_arm], 99)),
                      'arm_change_p99_rad': float(np.percentile(dd[:, is_arm], 99)) if is_arm.any() else 0.0},
            'series': {'pen_before': pf0, 'pen_after': pf1, 'move_mean': move.mean(1).cpu().numpy() * 100,
                       'gap_before': gap0, 'gap_after': gap1}}


def overlay(viz, stem, path) -> None:
    """Hands before (red) / after (green) and the object (orange) on the frames that moved most,
    cropped around the hands."""
    import matplotlib.pyplot as plt
    from PIL import Image
    W = paths.OUT / stem
    K = json.loads((W / 'intrinsics_stable.json').read_text())
    off = json.loads((W / 'track1_meta.json').read_text())['frame_offset']
    fig, axes = plt.subplots(2, len(viz['frames']), figsize=(5 * len(viz['frames']), 10), squeeze=False)

    def uv(x):
        return K['fx'] * x[:, 0] / x[:, 2] + K['cx'], K['fy'] * x[:, 1] / x[:, 2] + K['cy']
    for ax, f, b, a, o in zip(axes[0], viz['frames'], viz['before'], viz['after'], viz['object']):
        img = Image.open(W / 'frames' / f'{int(f) - off:06d}.png')
        ax.imshow(img)
        ax.scatter(*uv(o[::3]), s=.3, c='orange', alpha=.4)
        ax.scatter(*uv(b[::4]), s=.3, c='red', alpha=.5, label='hand before')
        ax.scatter(*uv(a[::4]), s=.3, c='lime', alpha=.5, label='hand after')
        u, v = uv(np.r_[a, b])
        ax.set_xlim(u.min() - 60, u.max() + 60); ax.set_ylim(v.max() + 60, v.min() - 60)
        ax.set_title(f'video frame {int(f)}', fontsize=9); ax.axis('off')
    for ax, b, a, o in zip(axes[1], viz['before'], viz['after'], viz['object']):
        ax.scatter(o[::3, 0], o[::3, 2], s=.3, c='orange', alpha=.4)
        ax.scatter(b[::4, 0], b[::4, 2], s=.3, c='red', alpha=.5)
        ax.scatter(a[::4, 0], a[::4, 2], s=.3, c='lime', alpha=.5)
        c = np.r_[a, b].mean(0)
        ax.set_xlim(c[0] - .25, c[0] + .25); ax.set_ylim(c[2] + .25, c[2] - .25)
        ax.set_xlabel('x (m)'); ax.set_ylabel('depth z (m), camera at top'); ax.set_aspect('equal')
        ax.set_title('top-down view (camera frame)', fontsize=9)
    axes[0, 0].legend(markerscale=20, fontsize=7)
    fig.suptitle(f'{stem}: hands before (red) / after (green), object (orange)', fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--packed', type=Path, required=True)
    ap.add_argument('--episodes', type=int, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=150)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--chunk', type=int, default=96)
    ap.add_argument('--res', type=int, default=72)
    ap.add_argument('--margin_mm', type=float, default=1.0)
    ap.add_argument('--w_2d', type=float, default=0.0, help='keep hand vertices where they project (px^2)')
    ap.add_argument('--px_tol', type=float, default=3.0, help='free image motion per hand vertex (px)')
    ap.add_argument('--w_pen', type=float, default=1e4)
    ap.add_argument('--w_contact', type=float, default=1e3)
    ap.add_argument('--w_reg', type=float, default=1.0, help='finger/wrist parameter change weight')
    ap.add_argument('--arm', action='store_true', help='also move shoulder/elbow parameters (hand translates)')
    ap.add_argument('--w_reg_arm', type=float, default=1.0)
    ap.add_argument('--w_temp', type=float, default=50.0)
    args = ap.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((paths.DATA / 'experiments').resolve()):
        raise SystemExit('--out must be under V2D_DATA/experiments')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    sys.path.insert(0, str(paths.KIT_TRACK1))
    from mesh_to_mhr_params import MHR
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    mhr = MHR(str(paths.DATA / 'weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt'), dev, 64, 'float32')
    dest = out / 'episodes'
    (out / 'plots').mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    for f in args.src.glob('episode_*'):
        shutil.copy2(f, dest / f.name)
    report = []
    for e in args.episodes:
        r = run_episode(e, args, mhr, dev)
        stem = f'episode_{e:06d}'
        d = dict(np.load(args.src / f'{stem}.npz'))
        d['pose'] = d['pose'].copy()
        d['pose'][r['frames']] = r['new_pose_scored'].astype(d['pose'].dtype)
        np.savez(dest / f'{stem}.npz', **d)
        st = r['stats']
        report.append({'episode': e, **st})
        sr = r['series']
        fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        ax[0].plot(r['frames'], sr['pen_before'], label='before'); ax[0].plot(r['frames'], sr['pen_after'], label='after')
        ax[0].set_ylabel('PEN cm (512 hand pts)'); ax[0].legend(fontsize=8)
        ax[1].plot(r['frames'], sr['move_mean']); ax[1].set_ylabel('mean hand move cm')
        ax[2].plot(r['frames'], sr['gap_before'], label='before'); ax[2].plot(r['frames'], sr['gap_after'], label='after')
        ax[2].set_ylabel('hand-object gap cm'); ax[2].set_xlabel('video frame'); ax[2].legend(fontsize=8)
        fig.suptitle(f'{stem}: hand-only depenetration — PEN {st["pen_before_cm"]:.4f} -> {st["pen_after_cm"]:.4f} cm, '
                     f'hand move mean {st["hand_move_mean_cm"]:.2f} cm', fontsize=10)
        fig.tight_layout(); fig.savefig(out / 'plots' / f'{stem}.png', dpi=80); plt.close(fig)
        overlay(r['viz'], stem, out / 'plots' / f'{stem}_hands.png')
        print(f"{stem}: PEN {st['pen_before_cm']:.4f} -> {st['pen_after_cm']:.4f} cm | hand move mean "
              f"{st['hand_move_mean_cm']:.2f} p99 {st['hand_move_p99_cm']:.2f} cm | contact {st['contact_before']:.2f} -> "
              f"{st['contact_after']:.2f} | image move mean {st['image_move_mean_px']:.1f} p99 {st['image_move_p99_px']:.1f} px | p99 change finger/wrist {st['finger_wrist_change_p99_rad']:.2f} arm "
              f"{st['arm_change_p99_rad']:.2f} rad", flush=True)
    (out / 'hand_depen.json').write_text(json.dumps({'src': str(args.src), 'params': {k: (str(v) if isinstance(v, Path) else v)
                                                      for k, v in vars(args).items()}, 'episodes': report}, indent=1) + '\n')


if __name__ == '__main__':
    main()
