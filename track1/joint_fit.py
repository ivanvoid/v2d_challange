#!/usr/bin/env python3
"""Joint whole-video object fit (pre-registered as J in docs/EXPERIMENTS.md).

The pipeline places the object in separate steps (FoundationPose -> pose-graph smoothing -> one depth
factor per episode -> push-outs), and no step can correct an earlier one. Here all frames of one
episode are refined TOGETHER against the image and the (fixed) body:

  variables  per frame: a small rotation about the object's centre (axis-angle) and a translation
  sil        rendered object silhouette (nvdiffrast, antialiased => differentiable) vs the SAM2 mask,
             ignoring pixels covered by the person but not by the object (occlusion)
  contact    frames where the hand overlaps the object mask in the image and was already < 8 cm away:
             the closest 5 % of hand points are pulled to the surface (object SDF)
  pen        every hand point outside the object (SDF >= 1 mm)
  smooth     second differences of the final translation / rotation (ACC-O)
  prior      stay near the current pose: depth along the camera ray much stiffer than lateral (the
             depth lessons of phys_v3 / p1_global), rotations ~10 deg scale
Scale and mesh are fixed. Static (locked) intervals are left untouched by --skip_frames.

Gates per episode (on scored frames), all required to accept: silhouette IoU median not lower and
mean IoU gain >= --min_iou_gain, hand contact share not lower by > 0.02, PEN proxy not higher,
object self-acceleration not higher by > 0.01, max translation change <= --max_shift_cm.
Writes <out>/episodes (full copy of --src, the episode replaced only if accepted), joint_fit_<ep>.json,
and a review sheet (silhouette before red / after green / mask outline on 8 frames + time series).
Runs in /opt/venvs/foundation_pose (nvdiffrast + torch; MHR loads there too).
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

DOWN = 4                                   # render / compare at 1/4 resolution


def axis_angle_to_matrix(v):
    """Rodrigues for a batch [N,3] (torch, differentiable, safe at 0)."""
    import torch
    th = v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    k = v / th
    K = torch.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype, device=v.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    th = th[..., None]
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand_as(K)
    return eye + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


def second_diff_index(frames: np.ndarray) -> np.ndarray:
    f = np.asarray(frames)
    ok = (f[1:-1] - f[:-2] == 1) & (f[2:] - f[1:-1] == 1)
    return np.flatnonzero(ok) + 1


def ray_split(delta, centers):
    """Split translation changes into the component along each camera ray and the rest."""
    ray = centers / centers.norm(dim=-1, keepdim=True)
    along = (delta * ray).sum(-1, keepdim=True)
    return along.squeeze(-1), delta - along * ray


class Renderer:
    """Antialiased (differentiable) silhouettes of camera-frame meshes with OpenCV intrinsics."""

    def __init__(self, K: dict, H: int, W: int, faces, device='cuda', down: int = 1):
        import nvdiffrast.torch as dr
        import torch
        self.dr, self.torch, self.H, self.W, self.faces = dr, torch, H, W, faces
        self.ctx = dr.RasterizeCudaContext()
        fx, fy, cx, cy = K['fx'] / down, K['fy'] / down, K['cx'] / down, K['cy'] / down
        n, f = 0.05, 20.0
        # pixel u = fx x/z + cx  <->  NDC x = 2u/W - 1; nvdiffrast's NDC y points UP, image rows DOWN
        self.P = torch.tensor([[2 * fx / W, 0, 2 * cx / W - 1, 0], [0, -2 * fy / H, 1 - 2 * cy / H, 0],
                               [0, 0, (f + n) / (f - n), -2 * f * n / (f - n)], [0, 0, 1, 0]],
                              dtype=torch.float32, device=device)

    def __call__(self, X):
        """X [B, V, 3] camera frame -> silhouettes [B, H, W] in image row order."""
        torch = self.torch
        clip = torch.cat([X, torch.ones_like(X[..., :1])], -1) @ self.P.T
        clip = clip.contiguous()
        rast, _ = self.dr.rasterize(self.ctx, clip, self.faces, resolution=[self.H, self.W])
        sil = self.dr.antialias((rast[..., 3:] > 0).float(), rast, clip, self.faces)[..., 0]
        return sil.flip(1)          # nvdiffrast row 0 is the bottom (NDC y = -1)


def iou(a, b, valid):
    a, b = a[valid], b[valid]
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--episode', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--chunk', type=int, default=48)
    ap.add_argument('--w_sil', type=float, default=1.0)
    ap.add_argument('--w_contact', type=float, default=200.0)
    ap.add_argument('--w_pen', type=float, default=2000.0)
    ap.add_argument('--lam_t', type=float, default=1000.0)
    ap.add_argument('--lam_r', type=float, default=300.0)
    ap.add_argument('--sigma_depth_cm', type=float, default=4.0)
    ap.add_argument('--sigma_lat_cm', type=float, default=10.0)
    ap.add_argument('--sigma_rot_deg', type=float, default=10.0)
    ap.add_argument('--skip_frames', type=str, default='', help='video frame ranges to leave untouched, e.g. 80-364')
    ap.add_argument('--min_iou_gain', type=float, default=0.005)
    ap.add_argument('--max_shift_cm', type=float, default=15.0)
    args = ap.parse_args()
    import cv2
    import torch
    import trimesh
    from hand_depen import SdfGrid
    sys.path.insert(0, str(paths.KIT_TRACK1))
    from mesh_to_mhr_params import MHR
    from qa_episode import hand_vertex_ids
    out = args.out.resolve()
    if not out.is_relative_to((paths.DATA / 'experiments').resolve()):
        raise SystemExit('--out must be under V2D_DATA/experiments')
    dev = 'cuda'
    stem = f'episode_{args.episode:06d}'
    W = paths.OUT / stem
    meta = json.loads((W / 'track1_meta.json').read_text())
    off = meta['frame_offset']
    K = json.loads((W / 'intrinsics_stable.json').read_text())
    H, Wd = int(K['height']) // DOWN, int(K['width']) // DOWN
    d = dict(np.load(args.src / f'{stem}.npz'))
    n_rows = len(d['object_translation'])
    vid = np.arange(off, min(off + meta['n_frames'], n_rows))
    skip = np.zeros(len(vid), bool)
    for part in filter(None, args.skip_frames.split(',')):
        a, b = map(int, part.split('-'))
        skip |= (vid >= a) & (vid <= b)
    keep = ~skip
    spans = json.loads((paths.DATA / 'submission/scored_spans.json').read_text())[str(args.episode)]['frames']
    scored = np.isin(vid, spans)
    # --- data: masks, valid pixels (not hidden by the person), hands
    M, V = [], []
    for f in vid:
        m = cv2.imread(str(W / 'masks/1' / f'{f - off:06d}.png'), cv2.IMREAD_GRAYSCALE)
        pm = cv2.imread(str(W / 'body/person_masks' / f'{f - off:06d}.png'), cv2.IMREAD_GRAYSCALE)
        m = (cv2.resize(m, (Wd, H), interpolation=cv2.INTER_AREA) > 127) if m is not None else np.zeros((H, Wd), bool)
        pm = (cv2.resize(pm, (Wd, H), interpolation=cv2.INTER_AREA) > 127) if pm is not None else np.zeros((H, Wd), bool)
        M.append(m); V.append(~(pm & ~m))
    M = torch.as_tensor(np.stack(M), dtype=torch.float32, device=dev)
    Vv = torch.as_tensor(np.stack(V), dtype=torch.float32, device=dev)
    mhr = MHR(str(paths.DATA / 'weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt'), dev, 64, 'float32')
    z = torch.as_tensor(np.concatenate([d['scales'], d['shape']])[None], dtype=torch.float64, device=dev)
    pose = torch.as_tensor(d['pose'][vid], dtype=torch.float64, device=dev)
    hid = hand_vertex_ids(mhr, pose[:1], z)
    hands = (mhr.run(pose, z)[0] / 1000)[:, hid[::3]].float()                 # [T, Hh, 3]
    # --- object
    mesh = trimesh.load(args.src / f'{stem}_object.obj', force='mesh', process=False)
    s = float(d['object_scale'])
    mv = np.asarray(mesh.vertices, np.float64)
    mf = np.asarray(mesh.faces, np.int32)
    if len(mf) > 6000:                                      # keep rendering cheap
        sim = mesh.simplify_quadric_decimation(face_count=6000)
        mv, mf = np.asarray(sim.vertices, np.float64), np.asarray(sim.faces, np.int32)
    c_m = mv.mean(0)
    from pen_fast import PenetrationMesh
    pm = PenetrationMesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    sdf = SdfGrid(np.asarray(mesh.vertices), np.asarray(mesh.faces), res=64,
                  inside_fn=lambda q: np.abs(pm.winding(q)) > 0.5).torch(dev)   # scorer's inside test
    verts = torch.as_tensor((mv - c_m) * s, dtype=torch.float32, device=dev)  # centred, scaled
    faces = torch.as_tensor(mf, device=dev)
    R0 = torch.as_tensor(d['object_rotation'][vid], dtype=torch.float32, device=dev)
    t0 = torch.as_tensor(d['object_translation'][vid], dtype=torch.float32, device=dev)
    c0 = torch.einsum('tij,j->ti', R0, torch.as_tensor(c_m * s, dtype=torch.float32, device=dev)) + t0   # centres
    T = len(vid)
    render = Renderer(K, H, Wd, faces, dev, DOWN)
    fx, fy, cx, cy = K['fx'] / DOWN, K['fy'] / DOWN, K['cx'] / DOWN, K['cy'] / DOWN

    def posed(dr_, dt_, idx):
        Rf = axis_angle_to_matrix(dr_[idx]) @ R0[idx]
        cen = c0[idx] + dt_[idx]
        return torch.einsum('tij,vj->tvi', Rf, verts) + cen[:, None], Rf, cen

    def hand_terms(X_R, X_c, idx):
        Rf, cen = X_R, X_c
        local = torch.einsum('thi,tij->thj', hands[idx] - cen[:, None], Rf) / s + torch.as_tensor(c_m, dtype=torch.float32, device=dev)
        return sdf(local.reshape(-1, 3)).view(len(idx), -1) * s

    # contact frames: hand projects onto the object mask AND was already close in 3D
    with torch.no_grad():
        sd0 = torch.cat([hand_terms(*posed(torch.zeros(T, 3, device=dev), torch.zeros(T, 3, device=dev), torch.arange(a, min(T, a + args.chunk), device=dev))[1:],
                                    torch.arange(a, min(T, a + args.chunk), device=dev)) for a in range(0, T, args.chunk)])
        uv = torch.stack([hands[..., 0] / hands[..., 2] * fx + cx, hands[..., 1] / hands[..., 2] * fy + cy], -1).round().long()
        uv[..., 0].clamp_(0, Wd - 1); uv[..., 1].clamp_(0, H - 1)
        on_mask = M[torch.arange(T, device=dev)[:, None], uv[..., 1], uv[..., 0]].mean(1) > 0.05
        contact = on_mask & (sd0.clamp(min=0).min(1).values < 0.08) & torch.as_tensor(keep, device=dev)
    k5 = max(1, int(0.05 * hands.shape[1]))
    drot = torch.zeros(T, 3, device=dev, requires_grad=True)
    dtr = torch.zeros(T, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([drot, dtr], lr=args.lr)
    keep_t = torch.as_tensor(keep, dtype=torch.float32, device=dev)[:, None]
    trip = torch.as_tensor(second_diff_index(vid), device=dev)
    sd_, sl_, sr_ = args.sigma_depth_cm / 100, args.sigma_lat_cm / 100, np.radians(args.sigma_rot_deg)
    for step in range(args.steps):
        opt.zero_grad()
        for a in range(0, T, args.chunk):
            idx = torch.arange(a, min(T, a + args.chunk), device=dev)
            dr_m, dt_m = drot * keep_t, dtr * keep_t                 # skipped frames stay exactly as they are
            X, Rf, cen = posed(dr_m, dt_m, idx)
            sil = render(X)
            l_sil = (((sil - M[idx]) ** 2) * Vv[idx]).mean((1, 2)).sum() * args.w_sil * 100
            sd = hand_terms(Rf, cen, idx)
            close = torch.topk(sd, k5, dim=1, largest=False).values
            l_con = (close.clamp(min=0).mean(1) ** 2 * contact[idx]).sum() * args.w_contact * 100
            l_pen = torch.relu(0.001 - sd).pow(2).sum(1).sum() * args.w_pen
            ((l_sil + l_con + l_pen) / T).backward()
        dr_m, dt_m = drot * keep_t, dtr * keep_t
        along, lat = ray_split(dt_m, c0)
        prior = ((along / sd_) ** 2 + (lat ** 2).sum(-1) / sl_ ** 2 + (dr_m ** 2).sum(-1) / sr_ ** 2).mean() * 0.01
        cen = c0 + dt_m
        rv = dr_m
        sm = (args.lam_t * ((cen[trip - 1] - 2 * cen[trip] + cen[trip + 1]) ** 2).sum(-1).mean() +
              args.lam_r * ((rv[trip - 1] - 2 * rv[trip] + rv[trip + 1]) ** 2).sum(-1).mean()) if len(trip) else 0 * prior
        (prior + sm).backward()
        opt.step()
    # --- evaluate before / after on scored frames
    with torch.no_grad():
        dr_m, dt_m = drot * keep_t, dtr * keep_t
        res = {}
        for name, (a_r, a_t) in (('before', (torch.zeros_like(dr_m), torch.zeros_like(dt_m))), ('after', (dr_m, dt_m))):
            ious, sds, sils, Rs, Cs = [], [], [], [], []
            for a in range(0, T, args.chunk):
                idx = torch.arange(a, min(T, a + args.chunk), device=dev)
                X, Rf, cen = posed(a_r, a_t, idx)
                sil = render(X) > 0.5
                sils.append(sil.cpu()); Rs.append(Rf.cpu()); Cs.append(cen.cpu())
                for j, i in enumerate(idx.tolist()):
                    ious.append(iou(sil[j].cpu().numpy(), M[i].bool().cpu().numpy(), Vv[i].bool().cpu().numpy()))
                sds.append(hand_terms(Rf, cen, idx).cpu())
            sds = torch.cat(sds)
            C = torch.cat(Cs).numpy()
            acc = np.linalg.norm(C[second_diff_index(vid) - 1] - 2 * C[second_diff_index(vid)] + C[second_diff_index(vid) + 1], axis=1)
            res[name] = {'iou': np.array(ious), 'gap': sds.clamp(min=0).min(1).values.numpy() * 100,
                         'pen': torch.relu(-sds).mean(1).numpy() * 100, 'acc': float(acc.mean() * 100) if len(acc) else 0.0,
                         'R': torch.cat(Rs).numpy(), 'c': C, 'sil': torch.cat(sils)}
    b, a_ = res['before'], res['after']
    sc = scored & keep
    shift = np.linalg.norm(a_['c'] - b['c'], axis=1) * 100
    along_cm = np.abs(ray_split(torch.as_tensor(a_['c'] - b['c']), torch.as_tensor(b['c']))[0].numpy()) * 100
    stats = {'episode': args.episode, 'frames': int(T), 'scored_fitted': int(sc.sum()), 'contact_frames': int(contact.sum()),
             'iou_median': [float(np.nanmedian(b['iou'][sc])), float(np.nanmedian(a_['iou'][sc]))],
             'iou_mean': [float(np.nanmean(b['iou'][sc])), float(np.nanmean(a_['iou'][sc]))],
             'contact_share': [float((b['gap'][sc] < 2).mean()), float((a_['gap'][sc] < 2).mean())],
             'pen_cm': [float(b['pen'][sc].mean()), float(a_['pen'][sc].mean())],
             'self_acc': [b['acc'], a_['acc']],
             'shift_cm_p99': float(np.percentile(shift[sc], 99)) if sc.any() else 0.0,
             'depth_change_cm_p99': float(np.percentile(along_cm[sc], 99)) if sc.any() else 0.0}
    gates = {'iou_median_not_lower': stats['iou_median'][1] >= stats['iou_median'][0] - 1e-4,
             'iou_mean_gain': stats['iou_mean'][1] - stats['iou_mean'][0] >= args.min_iou_gain,
             'contact_not_lower': stats['contact_share'][1] >= stats['contact_share'][0] - 0.02,
             'pen_not_higher': stats['pen_cm'][1] <= stats['pen_cm'][0] + 1e-4,
             'acc_not_higher': stats['self_acc'][1] <= stats['self_acc'][0] + 0.01,
             'shift_bounded': stats['shift_cm_p99'] <= args.max_shift_cm}
    stats['gates'] = gates
    stats['accepted'] = bool(all(gates.values()))
    dest = out / 'episodes'
    dest.mkdir(parents=True, exist_ok=True)
    for f in args.src.glob('episode_*'):
        if not (dest / f.name).exists():
            shutil.copy2(f, dest / f.name)
    if stats['accepted']:
        Rn = a_['R'].astype(np.float64)
        tn = a_['c'] - np.einsum('tij,j->ti', Rn, c_m * s)
        Rn[skip] = d['object_rotation'][vid][skip]
        tn[skip] = d['object_translation'][vid][skip]
        d['object_rotation'] = d['object_rotation'].copy(); d['object_translation'] = d['object_translation'].copy()
        d['object_rotation'][vid] = Rn.astype(d['object_rotation'].dtype)
        d['object_translation'][vid] = tn.astype(d['object_translation'].dtype)
        np.savez(dest / f'{stem}.npz', **d)
    (out / f'joint_fit_{stem}.json').write_text(json.dumps({'params': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                                                            **stats}, indent=1) + '\n')
    print(f"{stem}: {'ACCEPT' if stats['accepted'] else 'REJECT'} | IoU median {stats['iou_median'][0]:.3f} -> {stats['iou_median'][1]:.3f}, "
          f"mean {stats['iou_mean'][0]:.3f} -> {stats['iou_mean'][1]:.3f} | contact {stats['contact_share'][0]:.2f} -> "
          f"{stats['contact_share'][1]:.2f} | PEN {stats['pen_cm'][0]:.4f} -> {stats['pen_cm'][1]:.4f} | self-acc "
          f"{stats['self_acc'][0]:.3f} -> {stats['self_acc'][1]:.3f} | shift p99 {stats['shift_cm_p99']:.1f} cm "
          f"(depth p99 {stats['depth_change_cm_p99']:.1f}) | contact frames {stats['contact_frames']} | "
          f"failed gates: {[k for k, v in gates.items() if not v]}", flush=True)
    # --- review sheet
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sidx = np.flatnonzero(sc)
    pick = sidx[np.linspace(0, len(sidx) - 1, min(8, len(sidx))).round().astype(int)] if len(sidx) else []
    fig = plt.figure(figsize=(22, 13))
    for j, k in enumerate(pick):
        ax = fig.add_subplot(3, 4, j + 1)
        img = cv2.cvtColor(cv2.imread(str(W / 'frames' / f'{vid[k] - off:06d}.png')), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (Wd, H))
        ov = img.copy()
        ov[b['sil'][k].numpy()] = (0.5 * ov[b['sil'][k].numpy()] + [127, 0, 0]).astype(np.uint8)
        ov[a_['sil'][k].numpy()] = (0.5 * ov[a_['sil'][k].numpy()] + [0, 127, 0]).astype(np.uint8)
        cnt, _ = cv2.findContours(M[k].cpu().numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cnt, -1, (255, 255, 0), 1)
        ys, xs = np.nonzero(M[k].cpu().numpy() | a_['sil'][k].numpy() | b['sil'][k].numpy())
        ax.imshow(ov)
        if len(xs):
            ax.set_xlim(max(0, xs.min() - 40), min(Wd, xs.max() + 40)); ax.set_ylim(min(H, ys.max() + 40), max(0, ys.min() - 40))
        ax.set_title(f'frame {vid[k]}: IoU {b["iou"][k]:.2f} -> {a_["iou"][k]:.2f}', fontsize=9); ax.axis('off')
    ax = fig.add_subplot(3, 2, 5)
    ax.plot(vid, b['iou'], '.', ms=2, label='IoU before'); ax.plot(vid, a_['iou'], '.', ms=2, label='IoU after'); ax.legend(fontsize=8)
    ax = fig.add_subplot(3, 2, 6)
    ax.plot(vid, shift, label='centre shift cm'); ax.plot(vid, along_cm, label='of which along the ray'); ax.plot(vid, a_['gap'], label='hand gap after cm', alpha=.6)
    ax.legend(fontsize=8)
    fig.suptitle(f'{stem}: silhouette before (red) / after (green), SAM2 mask outline (yellow) — {"ACCEPT" if stats["accepted"] else "REJECT"}')
    fig.tight_layout(); fig.savefig(out / f'joint_fit_{stem}.png', dpi=60)


if __name__ == '__main__':
    main()
