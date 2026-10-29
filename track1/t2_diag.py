#!/usr/bin/env python3
"""Test-set diagnostics for the pre-registered hypotheses H1 and H-depth (docs/EXPERIMENTS.md).

H1  Does the "objects 4 % too close" result survive a scorer-like alignment? The scorer fits ONE
    similarity on the body of the first scored frame. We have no reference body surface, so we
    emulate it with the root pose: a rigid transform from our body frame (pelvis + hips + spine
    joints) to the reference root frame at the first scored frame. The fixed convention rotation
    between the two body frames is estimated over all frames with the trajectory registration
    (`t2_testset.register`), then the first-frame alignment uses only frame 0. Two scales are
    reported: 1 (our body is metric, as the scorer's body-size scale would be) and the trajectory
    scale. Output: CD-O vs object depth factor under each alignment.
H-depth  Is MoGe's object depth wrong? Per frame, median MoGe depth over our object mask vs the
    reference object's front-surface depth (z-buffer of reference surface points) over the same
    pixels, in our camera frame (trajectory registration). Also our final placement over the same
    pixels. Output: per-episode median |error|, per-frame plots.

    /opt/venvs/sam3d_body/bin/python track1/t2_diag.py --root /local-data/ivan/v2d_t2 \
        --gt /local-data/ivan/v2d_t2eval/gt --episodes_dir <episodes> --episodes 9 10 17 21 27 \
        --out /local-data/ivan/v2d_t2eval/diag
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t2_testset import chamfer_sum, human_to_opencv, register, rotvec_to_matrix  # noqa: E402

PELVIS, L_HIP, R_HIP, SPINE3 = 1, 2, 18, 37          # MHR skeleton (scorer's 22-joint list order)


def body_frames(J: np.ndarray) -> np.ndarray:
    """[T,3,3] orthonormal body frames (columns: left, up, forward) from pelvis/hips/spine joints."""
    x = J[:, L_HIP] - J[:, R_HIP]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    up = J[:, SPINE3] - J[:, PELVIS]
    up -= (up * x).sum(1, keepdims=True) * x
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    z = np.cross(x, up)
    return np.stack([x, up, z], axis=2)


def mean_rotation(Rs: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(Rs.sum(0))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def first_frame_alignment(B_ours: np.ndarray, p_ours: np.ndarray, B_ref: np.ndarray, p_ref: np.ndarray,
                          R_traj: np.ndarray, scale: float = 1.0):
    """Scorer-like alignment from frame 0 only. Convention C maps our body frame to the reference
    root frame: B_ref ~ R_traj B_ours C (estimated over all frames). Returns (s, R, t) with
    world = s R ours + t, exact for frame 0's root pose."""
    C = mean_rotation(np.einsum('tji,tjk->tik', R_traj @ B_ours, B_ref))   # (R_traj B)^T B_ref
    R0 = B_ref[0] @ C.T @ B_ours[0].T
    t0 = p_ref[0] - scale * R0 @ p_ours[0]
    return scale, R0, t0, C


def zbuffer_median_depth(points_cam: np.ndarray, K: dict, mask: np.ndarray, splat: int = 1) -> float | None:
    """Median front-surface depth of a point cloud over the pixels of `mask` (None if < 20 px)."""
    p = points_cam[points_cam[:, 2] > 0.05]
    u = np.round(K['fx'] * p[:, 0] / p[:, 2] + K['cx']).astype(int)
    v = np.round(K['fy'] * p[:, 1] / p[:, 2] + K['cy']).astype(int)
    h, w = mask.shape
    zb = np.full((h, w), np.inf)
    for du in range(-splat, splat + 1):
        for dv in range(-splat, splat + 1):
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            np.minimum.at(zb, (vv[ok], uu[ok]), p[ok, 2])
    vals = zb[mask & np.isfinite(zb)]
    return float(np.median(vals)) if len(vals) >= 20 else None


def load_moge(path: Path) -> np.ndarray:
    import cv2
    px = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    return 1.0 / np.maximum(px / 65535.0, 1e-6) - 1.0


def run(args) -> None:
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    import trimesh
    import paths
    sys.path.insert(0, str(paths.KIT_TRACK1))
    from mesh_to_mhr_params import MHR
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    mhr = MHR(str(paths.DATA / 'weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt'), dev, 64, 'float32')
    args.out.mkdir(parents=True, exist_ok=True)
    ks = np.round(np.arange(0.94, 1.081, 0.01), 3)
    report = {'h1': {}, 'hdepth': {}}
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 4.5))
    fig2, axes2 = plt.subplots(len(args.episodes), 1, figsize=(10, 2.6 * len(args.episodes)), squeeze=False)
    for row, e in enumerate(args.episodes):
        stem = f'episode_{e:06d}'
        W = args.root / 'out' / stem
        g = np.load(args.gt / f'{stem}.npz')
        d = np.load(args.episodes_dir / f'{stem}.npz')
        K = json.loads((W / 'intrinsics_stable.json').read_text())
        off = json.loads((W / 'track1_meta.json').read_text())['frame_offset']
        fr = g['frames'][g['frames'] < len(d['pose'])]
        z = torch.as_tensor(np.concatenate([d['scales'], d['shape']])[None], dtype=torch.float64, device=dev)
        J = mhr.run(torch.as_tensor(d['pose'][fr], dtype=torch.float64, device=dev), z)[1].cpu().numpy()
        ref_root, ref_rot = human_to_opencv(g['human_t'][fr], rotvec_to_matrix(g['human_pose'][fr, :3]))
        s_tr, R_tr, t_tr, o, res = register(J[:, PELVIS], ref_root, ref_rot)
        B = body_frames(J)
        p_ref = ref_root + np.einsum('nij,j->ni', ref_rot, o)
        aligns = {'trajectory': (s_tr, R_tr, t_tr)}
        for name, sc in (('first_frame_s1', 1.0), ('first_frame_strajectory', s_tr)):
            s0, R0, t0, C = first_frame_alignment(B, J[:, PELVIS], ref_rot, p_ref, R_tr, sc)
            aligns[name] = (s0, R0, t0)
        conv_spread = float(np.degrees(np.mean([np.arccos(np.clip((np.trace(
            ((R_tr @ B[i]).T @ ref_rot[i]) @ C.T) - 1) / 2, -1, 1)) for i in range(len(fr))])))
        ours = trimesh.load(args.episodes_dir / f'{stem}_object.obj', force='mesh', process=False)
        gts = trimesh.load(str(g['mesh']))
        gtm = gts.dump(concatenate=True) if hasattr(gts, 'dump') else gts
        P = trimesh.sample.sample_surface(ours, 1500, seed=0)[0]
        G = trimesh.sample.sample_surface(gtm, 3000, seed=0)[0]
        so = float(d['object_scale'])
        sub = np.arange(0, len(fr), 4)
        h1 = {}
        for name, (s, R, t) in aligns.items():
            curve = []
            for k in ks:
                cds = [chamfer_sum((s * (R @ (k * (so * P @ d['object_rotation'][fr[i]].T + d['object_translation'][fr[i]])).T)).T + t,
                                   G @ g['object_R'][fr[i]].T + g['object_t'][fr[i]]) * 100 for i in sub]
                curve.append(float(np.mean(cds)))
            h1[name] = {'k': ks.tolist(), 'cd_o': curve, 'best_k': float(ks[int(np.argmin(curve))])}
        h1['convention_spread_deg'] = conv_spread
        h1['registration_residual_cm'] = float(res.mean() * 100)
        report['h1'][e] = h1
        ax1.plot(ks, h1['first_frame_s1']['cd_o'], '-o', ms=3, label=f'ep {e} first-frame')
        ax1.plot(ks, h1['trajectory']['cd_o'], '--', alpha=.5, label=f'ep {e} trajectory')
        # H-depth: MoGe / ours vs reference surface depth over our object mask, in our camera frame
        s, R, t = aligns['trajectory']
        moge_err, ours_err, fidx = [], [], []
        for i in range(0, len(fr), 3):
            f = int(fr[i])
            mp = W / 'masks' / '1' / f'{f - off:06d}.png'
            dp = W / 'depth' / f'{f - off:06d}.png'
            if not (mp.exists() and dp.exists()):
                continue
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 0
            if mask.sum() < 200:
                continue
            ref_cam = (((G @ g['object_R'][f].T + g['object_t'][f]) - t) @ R) / s
            z_ref = zbuffer_median_depth(ref_cam, K, mask)
            z_ours = zbuffer_median_depth(so * P @ d['object_rotation'][f].T + d['object_translation'][f], K, mask)
            if z_ref is None:
                continue
            depth = load_moge(dp)
            z_moge = float(np.median(depth[mask]))
            moge_err.append((z_moge - z_ref) * 100)
            ours_err.append(((z_ours - z_ref) * 100) if z_ours is not None else np.nan)
            fidx.append(f)
        moge_err, ours_err = np.array(moge_err), np.array(ours_err)
        report['hdepth'][e] = {'frames': len(fidx), 'moge_median_abs_cm': float(np.median(np.abs(moge_err))),
                               'moge_median_signed_cm': float(np.median(moge_err)),
                               'ours_median_abs_cm': float(np.nanmedian(np.abs(ours_err))),
                               'ours_median_signed_cm': float(np.nanmedian(ours_err))}
        ax = axes2[row, 0]
        ax.plot(fidx, moge_err, '.', ms=3, label='MoGe depth over our mask − reference')
        ax.plot(fidx, ours_err, '.', ms=3, label='our final object − reference')
        ax.axhline(0, c='k', lw=.5)
        ax.set_ylabel('cm'); ax.set_title(f'ep {e} ({g["object"]})', fontsize=9)
        ax.legend(fontsize=7)
        r = report['hdepth'][e]
        print(f"ep {e:2d} H1 best k: trajectory {h1['trajectory']['best_k']:.2f}  first-frame(s=1) "
              f"{h1['first_frame_s1']['best_k']:.2f}  first-frame(s=traj) {h1['first_frame_strajectory']['best_k']:.2f}  "
              f"[conv spread {conv_spread:.0f} deg] | H-depth MoGe |err| {r['moge_median_abs_cm']:.1f} "
              f"(signed {r['moge_median_signed_cm']:+.1f}) ours |err| {r['ours_median_abs_cm']:.1f} "
              f"(signed {r['ours_median_signed_cm']:+.1f}) cm over {r['frames']} frames", flush=True)
    ax1.set_xlabel('object depth factor k (about our camera)'); ax1.set_ylabel('CD-O (cm)')
    ax1.set_title('H1: CD-O vs depth factor, first-frame (solid) vs trajectory (dashed) alignment', fontsize=9)
    ax1.legend(fontsize=6, ncol=2)
    fig1.tight_layout(); fig1.savefig(args.out / 'h1_depth_factor.png', dpi=90)
    fig2.tight_layout(); fig2.savefig(args.out / 'hdepth_per_frame.png', dpi=80)
    mean_curve = {n: np.mean([report['h1'][e][n]['cd_o'] for e in args.episodes], 0) for n in
                  ('trajectory', 'first_frame_s1', 'first_frame_strajectory')}
    report['h1_mean'] = {n: {'cd_o': c.tolist(), 'best_k': float(ks[int(np.argmin(c))])} for n, c in mean_curve.items()}
    report['hdepth_mean'] = {k: float(np.mean([report['hdepth'][e][k] for e in args.episodes]))
                             for k in ('moge_median_abs_cm', 'moge_median_signed_cm', 'ours_median_abs_cm', 'ours_median_signed_cm')}
    (args.out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print('MEAN best k:', {n: v['best_k'] for n, v in report['h1_mean'].items()})
    print('MEAN H-depth:', {k: round(v, 2) for k, v in report['hdepth_mean'].items()})
    print('plots:', args.out / 'h1_depth_factor.png', args.out / 'hdepth_per_frame.png')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--gt', type=Path, required=True)
    ap.add_argument('--episodes_dir', type=Path, required=True)
    ap.add_argument('--episodes', type=int, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
