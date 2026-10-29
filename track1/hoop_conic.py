#!/usr/bin/env python3
"""Hoop pose from its image ellipse: a circle of known radius seen by a calibrated camera.

FoundationPose loses the thin hoop (episode 1: 57 of 174 scored frames lost, object ~60 cm from
the hand that holds it), but SAM2 segments it cleanly and an ellipse fits the mask almost exactly.
A circle of radius R projects to an ellipse; its size fixes the depth, its shape the tilt, up to
the classic two-fold ambiguity (tilted towards / away from the camera). The hand holding the rim
picks the candidate (and temporal continuity breaks ties).

Per frame (local trimmed index):
  1. ellipse  cv2.fitEllipse on the mask pixels; quality = share of mask pixels within 6 px of it
  2. init     depth Z = fx R / a (a = semi-major, px), centre on the ray through the ellipse centre,
              tilt acos(b/a) about the major axis, two signs
  3. refine   least squares on (centre, normal): projected rim samples <-> fitted ellipse samples,
              both directions (perspective-exact)
  4. choose   the candidate whose rim passes closest to a hand vertex; continuity with the previous
              chosen normal when the hands are > --hand_max_cm from both
Then a pose for OUR mesh: its ring axis (smallest principal axis) onto the normal with the minimal
rotation from the existing pose (the ring's spin about its axis is unobservable and irrelevant to the
metric), translation so the ring centre lands on the circle centre. Replaced only where the existing
pose is poor (stage-6 lost, or IoU < --iou_bad) and the ellipse fit is good; smoothed, with a
blend at segment ends. Writes <out>/episodes (full copy of --src) + hoop_conic.json + review sheets.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402


def circle_points(c, n, R, k=96):
    n = n / np.linalg.norm(n)
    a = np.cross(n, [1, 0, 0] if abs(n[0]) < .9 else [0, 1, 0]); a /= np.linalg.norm(a)
    b = np.cross(n, a)
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return c + R * (np.cos(t)[:, None] * a + np.sin(t)[:, None] * b)


def project(x, K):
    return np.c_[K['fx'] * x[:, 0] / x[:, 2] + K['cx'], K['fy'] * x[:, 1] / x[:, 2] + K['cy']]


def ellipse_points(ell, k=180):
    (cx, cy), (w, h), ang = ell
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    th = np.radians(ang)
    x, y = w / 2 * np.cos(t), h / 2 * np.sin(t)
    return np.c_[cx + x * np.cos(th) - y * np.sin(th), cy + x * np.sin(th) + y * np.cos(th)]


def normal_from_angles(p):
    th, ph = p
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


def angles_from_normal(n):
    n = n / np.linalg.norm(n)
    return np.array([np.arccos(np.clip(n[2], -1, 1)), np.arctan2(n[1], n[0])])


def circle_candidates(ell, K, R):
    """Two refined (centre, normal, residual_px) candidates for a circle of radius R."""
    (cx, cy), (w, h), ang = ell
    a, b = max(w, h) / 2, min(w, h) / 2
    Z = K['fx'] * R / a
    ray = np.array([(cx - K['cx']) / K['fx'], (cy - K['cy']) / K['fy'], 1.0])
    c0 = Z * ray
    th = np.radians(ang) + (np.pi / 2 if h > w else 0)                 # major axis direction in the image
    major = np.array([np.cos(th), np.sin(th), 0.0])
    view = -ray / np.linalg.norm(ray)
    minor = np.cross(view, major); minor /= np.linalg.norm(minor)
    tilt = np.arccos(np.clip(b / a, 0, 1))
    target = ellipse_points(ell)
    tree_t = cKDTree(target)
    out = []
    for sgn in (1, -1):
        n0 = np.cos(tilt) * view + sgn * np.sin(tilt) * minor

        def res(p):
            c, n = p[:3], normal_from_angles(p[3:])
            uv = project(circle_points(c, n, R), K)
            return np.r_[tree_t.query(uv)[0], cKDTree(uv).query(target)[0]]
        sol = least_squares(res, np.r_[c0, angles_from_normal(n0)], loss='soft_l1', f_scale=3.0)
        c, n = sol.x[:3], normal_from_angles(sol.x[3:])
        out.append((c, n / np.linalg.norm(n), float(np.sqrt(np.mean(sol.fun ** 2)))))
    return out


def rim_distance(points, c, n, R):
    """Distance of 3D points to a circle (centre c, unit normal n, radius R)."""
    q = points - c
    h = q @ n
    r = np.linalg.norm(q - h[:, None] * n, axis=1)
    return np.sqrt((r - R) ** 2 + h ** 2)


def min_rotation(a, b):
    """Smallest rotation taking unit vector a to unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else Rotation.from_rotvec(np.pi * np.cross(a, [1, 0, 0] if abs(a[0]) < .9 else [0, 1, 0]) /
                                                           np.linalg.norm(np.cross(a, [1, 0, 0] if abs(a[0]) < .9 else [0, 1, 0]))).as_matrix()
    return Rotation.from_rotvec(v / np.linalg.norm(v) * np.arctan2(np.linalg.norm(v), c)).as_matrix()


def smooth_second_difference(x: np.ndarray, lam: float) -> np.ndarray:
    """argmin sum |x - raw|^2 + lam sum |x[i-1] - 2 x[i] + x[i+1]|^2, per column (sparse solve)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    n = len(x)
    if n < 3:
        return x.copy()
    D = sp.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))
    A = (sp.eye(n) + lam * (D.T @ D)).tocsc()
    return np.stack([spsolve(A, x[:, k]) for k in range(x.shape[1])], 1)


def ring_frame(mesh_v):
    """Ring centre, axis (smallest principal direction) and median radius in mesh units."""
    c = mesh_v.mean(0)
    axis = np.linalg.svd(mesh_v - c)[2][2]
    q = mesh_v - c
    r = np.linalg.norm(q - (q @ axis)[:, None] * axis, axis=1)
    return c, axis, float(np.median(r))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True, help='episodes dir (NPZ + obj)')
    ap.add_argument('--episode', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--iou_bad', type=float, default=0.3)
    ap.add_argument('--fit_min', type=float, default=0.85, help='share of mask px within 6 px of the ellipse')
    ap.add_argument('--hand_max_cm', type=float, default=15.0)
    ap.add_argument('--sigma', type=float, default=3.0, help='temporal smoothing of the replaced poses (frames)')
    ap.add_argument('--blend', type=int, default=8, help='frames to blend into the existing pose at segment ends')
    ap.add_argument('--replace_all', action='store_true', help='use the ellipse pose wherever its fit is good')
    ap.add_argument('--radius', type=float, default=0.0,
                    help='ring radius prior in metres (sets object_scale; e.g. 0.35 for a 70 cm hoop)')
    ap.add_argument('--lam_t', type=float, default=0.0, help='final second-difference smoothing of translation')
    ap.add_argument('--lam_r', type=float, default=0.0, help='... of rotation (rotation vectors)')
    args = ap.parse_args()
    import cv2
    import torch
    import trimesh
    from scipy.ndimage import gaussian_filter1d
    sys.path.insert(0, str(paths.KIT_TRACK1))
    from mesh_to_mhr_params import MHR
    from qa_episode import hand_vertex_ids
    out = args.out.resolve()
    if not out.is_relative_to((paths.DATA / 'experiments').resolve()):
        raise SystemExit('--out must be under V2D_DATA/experiments')
    stem = f'episode_{args.episode:06d}'
    W = paths.OUT / stem
    K = json.loads((W / 'intrinsics_stable.json').read_text())
    meta = json.loads((W / 'track1_meta.json').read_text())
    off = meta['frame_offset']
    states = json.loads((W / 'track/states.json').read_text())['frames']
    d = dict(np.load(args.src / f'{stem}.npz'))
    s = s_old = float(d['object_scale'])
    mesh = trimesh.load(args.src / f'{stem}_object.obj', force='mesh', process=False)
    c_m, axis_m, R_m = ring_frame(np.asarray(mesh.vertices))
    if args.radius:                         # physical prior: rescale the object so the ring has this radius
        d['object_scale'] = np.asarray(args.radius / R_m, dtype=d['object_scale'].dtype)
        s = float(d['object_scale'])
    R = R_m * s
    n_rows = len(d['object_translation'])
    vid = np.arange(off, min(off + len(states), n_rows))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    mhr = MHR(str(paths.DATA / 'weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt'), dev, 64, 'float32')
    z = torch.as_tensor(np.concatenate([d['scales'], d['shape']])[None], dtype=torch.float64, device=dev)
    pose = torch.as_tensor(d['pose'][vid], dtype=torch.float64, device=dev)
    hid = hand_vertex_ids(mhr, pose[:1], z)
    hands = (mhr.run(pose, z)[0] / 1000).cpu().numpy()[:, hid[::4]]
    rows, prev_n = [], None
    for k, f in enumerate(vid):
        st = states[f - off]
        rec = {'frame': int(f), 'state': st['state'], 'iou': st['iou'], 'use': False}
        mp = W / 'masks' / '1' / f'{f - off:06d}.png'
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 0 if mp.exists() else None
        if m is not None and m.sum() > 500:
            pts = np.c_[np.nonzero(m)[1], np.nonzero(m)[0]].astype(np.float32)
            ell = cv2.fitEllipse(pts)
            fit = float((cKDTree(ellipse_points(ell, 720)).query(pts[::5])[0] < 6).mean())
            cands = circle_candidates(ell, K, R)
            dist = [float(rim_distance(hands[k], c, n, R).min() * 100) for c, n, _ in cands]
            pick = int(np.argmin(dist))
            if min(dist) > args.hand_max_cm and prev_n is not None:              # no hand: continuity
                pick = int(np.argmax([abs(n @ prev_n) for _, n, _ in cands]))
            c, n, rp = cands[pick]
            if prev_n is not None and n @ prev_n < 0:
                n = -n
            rec.update(fit=fit, res_px=rp, hand_cm=dist[pick], hand_cm_other=dist[1 - pick], c=c.tolist(), n=n.tolist())
            poor = st['state'] == 'lost' or st['iou'] < args.iou_bad
            rec['use'] = bool((poor or args.replace_all) and fit >= args.fit_min and rp < 8)
            prev_n = n if rec['use'] or prev_n is None else prev_n
        rows.append(rec)
    use = np.array([r['use'] for r in rows])
    Rold, told = d['object_rotation'][vid].astype(np.float64), d['object_translation'][vid].astype(np.float64)
    Rnew, tnew = Rold.copy(), told.copy()
    for k in np.flatnonzero(use):
        n = np.asarray(rows[k]['n'])
        cur_axis = Rold[k] @ axis_m
        if cur_axis @ n < 0:
            n = -n
        Rk = min_rotation(cur_axis, n) @ Rold[k]
        Rnew[k] = Rk
        tnew[k] = np.asarray(rows[k]['c']) - s * Rk @ c_m
    # smooth replaced segments (translation; rotation via rotvec relative to the first) and blend ends
    w = use.astype(float)
    for part in np.split(np.arange(len(vid)), np.flatnonzero(np.diff(use.astype(int)) != 0) + 1):
        if use[part[0]] and len(part) > 2:
            tnew[part] = gaussian_filter1d(tnew[part], args.sigma, axis=0, mode='nearest')
            ref = Rotation.from_matrix(Rnew[part[0]])
            rv = (ref.inv() * Rotation.from_matrix(Rnew[part])).as_rotvec()
            Rnew[part] = (ref * Rotation.from_rotvec(gaussian_filter1d(rv, args.sigma, axis=0, mode='nearest'))).as_matrix()
    ramp = np.clip(gaussian_filter1d(w, args.blend / 2), 0, 1) if use.any() else w
    ramp = np.maximum(ramp, w)
    t_out = ramp[:, None] * tnew + (1 - ramp[:, None]) * told
    R_out = np.stack([(Rotation.from_matrix(Rold[k]) * Rotation.from_rotvec(ramp[k] * (Rotation.from_matrix(Rold[k]).inv()
                       * Rotation.from_matrix(Rnew[k])).as_rotvec())).as_matrix() for k in range(len(vid))])
    if args.lam_t > 0:                     # one acceleration-penalised solve over the clip (as stage 7)
        t_out = smooth_second_difference(t_out, args.lam_t)
        ref = Rotation.from_matrix(R_out[len(R_out) // 2])
        rv = (ref.inv() * Rotation.from_matrix(R_out)).as_rotvec()
        R_out = (ref * Rotation.from_rotvec(smooth_second_difference(rv, args.lam_r))).as_matrix()
    dest = out / 'episodes'
    dest.mkdir(parents=True, exist_ok=True)
    for f in args.src.glob('episode_*'):
        shutil.copy2(f, dest / f.name)
    d['object_translation'] = d['object_translation'].copy()
    d['object_rotation'] = d['object_rotation'].copy()
    d['object_translation'][vid] = t_out.astype(d['object_translation'].dtype)
    d['object_rotation'][vid] = R_out.astype(d['object_rotation'].dtype)
    np.savez(dest / f'{stem}.npz', **d)
    # report on scored frames
    spans = json.loads((paths.DATA / 'submission/scored_spans.json').read_text())[str(args.episode)]['frames']
    sc = np.array([k for k, f in enumerate(vid) if f in set(spans)])
    P0 = trimesh.sample.sample_surface(mesh, 1500, seed=0)[0]
    P = P0 * s

    def hand_gap(Rs, ts, scale):
        return np.array([cKDTree(P0 * scale @ Rs[k].T + ts[k]).query(hands[k])[0].min() * 100 for k in sc])
    g0, g1 = hand_gap(Rold, told, s_old), hand_gap(R_out, t_out, s)
    rep = {'episode': args.episode, 'ring_radius_m': R, 'object_scale': [s_old, s], 'replaced_frames': int(use.sum()),
           'replaced_scored': int(use[sc].sum()), 'scored': len(sc),
           'hand_gap_scored_median_cm': [float(np.median(g0)), float(np.median(g1))],
           'hand_gap_scored_p90_cm': [float(np.percentile(g0, 90)), float(np.percentile(g1, 90))],
           'frames': rows}
    (out / 'hoop_conic.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(f"{stem}: ring radius {R:.3f} m; replaced {use.sum()} frames ({use[sc].sum()}/{len(sc)} scored); "
          f"hand-object gap on scored frames median {np.median(g0):.1f} -> {np.median(g1):.1f} cm, "
          f"p90 {np.percentile(g0, 90):.1f} -> {np.percentile(g1, 90):.1f} cm")
    # review sheet: 8 replaced scored frames, old (red) vs new (green) rim on the video
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pick = [k for k in sc if use[k]]
    pick = [pick[i] for i in np.linspace(0, len(pick) - 1, min(8, len(pick))).round().astype(int)] if pick else []
    if pick:
        fig, axes = plt.subplots(2, 4, figsize=(20, 9))
        for ax in axes.ravel():
            ax.axis('off')
        for ax, k in zip(axes.ravel(), pick):
            img = cv2.cvtColor(cv2.imread(str(W / 'frames' / f'{vid[k] - off:06d}.png')), cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            for Rs, ts, col in ((Rold, told, 'red'), (R_out, t_out, 'lime')):
                uv = project(P[::2] @ Rs[k].T + ts[k], K)
                ax.scatter(uv[:, 0], uv[:, 1], s=.6, c=col, alpha=.6)
            uvh = project(hands[k], K)
            ax.scatter(uvh[:, 0], uvh[:, 1], s=.6, c='deepskyblue')
            ax.set_xlim(max(0, uvh[:, 0].min() - 350), uvh[:, 0].max() + 250)
            ax.set_ylim(uvh[:, 1].max() + 300, uvh[:, 1].min() - 300)
            r = rows[k]
            ax.set_title(f"frame {vid[k]} ({r['state']}, IoU {r['iou']:.2f}) hand-rim {r['hand_cm']:.0f} cm "
                         f"(other tilt {r['hand_cm_other']:.0f})", fontsize=8)
        fig.suptitle(f'{stem}: hoop old (red) vs image-ellipse pose (green), hands blue', fontsize=11)
        fig.tight_layout(); fig.savefig(out / f'{stem}_hoop.png', dpi=70)


if __name__ == '__main__':
    main()
