#!/usr/bin/env python3
"""Automatic object fixes on top of the motion_fix episodes (new outputs only; direction tests E5, E2, E7).

1. Snap gate (E5). A symmetry-switch repair made by motion_fix is kept only if the repaired output
   does not jump at the switch frame: visible surface motion <= max(1.5 cm, 4x the median of the
   surrounding frames). Otherwise the window goes back to the full_v2 rotations (objects without a
   continuous symmetry only, where that is exactly the pre-repair state).
2. Recovery after tracking loss (E2), compact objects only (not the ring objects of step 3). In every
   span of >= 15 frames the tracker calls lost/occluded, a mask frame is usable when its area is
   0.3-3x the rendered object and its centroid lies within 80 px (640x480) of it. Only runs of >= 5
   usable frames count (confirmed); there the object is shifted parallel to the image plane at its
   current depth (<= 35 cm) onto the mask centroid; shifts are interpolated, ramped in/out over 10
   frames and smoothed with zero shift pinned at the span ends. Kept only if the mask IoU over the
   span improves by >= 0.05 and the nearest-hand gap does not grow by more than 5 cm.
3. Rigid ring (E7), hoops, only with --ring (experimental: per-frame fits still jump up to ~17 cm between
   frames and are not yet smoothed over time, so the submitted candidate does not use it). Per frame an ellipse is fitted to the SAM2 mask and the two circles of the
   ring radius that project onto it are solved (hoop_conic.circle_candidates). A candidate whose
   lowest rim point would be > 8 cm below the person's soles (MHR body, floor normal of stage 1) is
   impossible; of the rest, the one whose rim passes within 15 cm of a hand wins, else the one
   continuing the previous chosen normal, else the one closest to the current ring axis.
   Ring size: on frames where the current hoop lies flat at the soles, the mask pixels are cast onto
   the plane at sole height and a circle is fitted there; if >= 20 such frames agree (p10-p90 spread
   <= 10 %) and their median radius differs from the mesh ring by >= 3 %, the object scale is changed
   to it (the hoop keeps its centre on frames that are not replaced).
   A frame is replaced when the ellipse fit is good (>= 75 % of mask pixels within 6 px, circle rms
   < 5 px), a candidate is possible, and the circle explains the ellipse better than the current
   ring by >= 2 px rms. Replaced stretches are smoothed (Gaussian, sigma 3) and blended over 8 frames.

Reads $V2D_DATA (default /local-data/ivan/v2d_full) read-only; writes --out/episodes and --out/report.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hoop_conic as hc  # noqa: E402
import motion_fix as mf  # noqa: E402

DATA = mf.DATA
FULLV2 = mf.SRC
RING_OBJECTS = {'hula_hoop'}
W, H = 640, 480


def episode_info(e):
    name = f'episode_{e:06d}'
    d = DATA / 'out' / name
    meta = json.loads((d / 'track1_meta.json').read_text())
    return dict(name=name, dir=d, meta=meta, off=meta['frame_offset'], first=meta['scored_first'], last=meta['scored_last'],
                cam=json.loads((d / 'intrinsics_stable.json').read_text()),
                states={s['video_frame']: s for s in json.loads((d / 'track/states.json').read_text())['frames']},
                obj=json.loads((d / 'object_track.json').read_text())['object'],
                scored=set(json.loads((DATA / 'submission/scored_spans.json').read_text())[str(e)]['frames']))


def read_mask(info, f, size=(W, H)):
    m = cv2.imread(str(info['dir'] / 'masks/1' / f'{f - info["off"]:06d}.png'), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (cv2.resize(m, size, interpolation=cv2.INTER_NEAREST) > 127) if size else (m > 127)


def centroid(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else None


def surface_motion(mesh, s, R, t, n=2000):
    pts = trimesh.sample.sample_surface_even(mesh, n, seed=0)[0]
    P = [s * pts @ R[k].T + t[k] for k in range(len(R))]
    out = [0.0]
    for a, b in zip(P[:-1], P[1:]):
        out.append(0.5 * (cKDTree(a).query(b)[0].mean() + cKDTree(b).query(a)[0].mean()) * 100)
    return np.asarray(out)


def iou(info, f, mesh, s, R, t):
    m = read_mask(info, f)
    if m is None or m.sum() < 20:
        return np.nan
    sil = mf.silhouette(np.asarray(mesh.vertices) * s @ R.T + t, np.asarray(mesh.faces), info['cam'])
    return float((sil & m).sum() / max(1, (sil | m).sum()))


# ---------- 1. snap gate ----------

def gate_switch_repairs(z, e, info, mesh, mf_report, sym):
    notes = []
    if any(x['kind'] == 'continuous' for x in sym['symmetries']):
        return z, notes
    reps = [n for n in mf_report.get(str(e), {}).get('switches', []) if n['action'] == 'repaired']
    if not reps:
        return z, notes
    R = z['object_rotation'].copy()
    t, s = z['object_translation'], float(z['object_scale'])
    frames = np.arange(info['first'], info['last'] + 1)
    motion = surface_motion(mesh, s, R[frames].astype(float), t[frames].astype(float))
    full = np.load(FULLV2 / f'{info["name"]}.npz')
    for n in reps:
        j = int(np.searchsorted(frames, n['frame']))
        ctx = np.r_[motion[max(1, j - 8):j], motion[j + 1:j + 9]]
        jump, base = float(motion[j]), float(np.median(ctx))
        note = dict(frame=n['frame'], window=n['window'], surface_jump_cm=jump, context_cm=base)
        if jump > max(1.5, 4 * base):
            a, b = n['window']
            R[a:b + 1] = full['object_rotation'][a:b + 1]
            note['action'] = 'reverted to full_v2 (repair made a visible snap)'
        else:
            note['action'] = 'kept'
        notes.append(note)
    return dict(z, object_rotation=R), notes


# ---------- 2. recovery after tracking loss ----------

def smooth_pinned(x, lam=50.0):
    n = len(x)
    w = np.ones(n)
    w[[0, -1]] = 1e8
    D = diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))
    A = (diags(w) + lam * (D.T @ D)).tocsc()
    return np.stack([spsolve(A, w * x[:, k]) for k in range(x.shape[1])], 1)


def recover_lost(z, e, info, mesh, body, surf):
    notes = []
    if info['obj'] in RING_OBJECTS:
        return z, notes
    frames = np.arange(info['first'], info['last'] + 1)
    hid = np.array([info['states'][int(f)]['state'] in ('occluded', 'lost') for f in frames])
    spans, start = [], None
    for i, h in enumerate(np.r_[hid, False]):
        if h and start is None:
            start = i
        elif not h and start is not None:
            if i - start >= 15:
                spans.append((start, i - 1))
            start = None
    if not spans:
        return z, notes
    R, t = z['object_rotation'].astype(float), z['object_translation'].copy()
    s, v, faces, cam = float(z['object_scale']), np.asarray(mesh.vertices), np.asarray(mesh.faces), info['cam']
    fx, fy = cam['fx'] * W / cam['width'], cam['fy'] * H / cam['height']
    for a, b in spans:
        fr = frames[a:b + 1]
        note = dict(span=[int(fr[0]), int(fr[-1])])
        meas, usable = {}, np.zeros(len(fr), bool)
        for i, f in enumerate(fr):
            m = read_mask(info, f)
            sil = mf.silhouette(v * s @ R[f].T + t[f], faces, cam)
            if m is None or m.sum() < 30 or sil.sum() == 0 or not 0.3 <= m.sum() / sil.sum() <= 3:
                continue
            d = centroid(sil) - centroid(m)
            if np.linalg.norm(d) >= 80:
                continue
            usable[i] = True
            sh = np.array([d[0] * t[f][2] / fx, d[1] * t[f][2] / fy, 0.0])
            meas[i] = sh * min(1.0, 0.35 / max(np.linalg.norm(sh), 1e-9))
        weight, run = np.zeros(len(fr)), 0
        for i in range(len(fr)):
            run = run + 1 if usable[i] else 0
            if run >= 5:
                weight[i - run + 1:i + 1] = 1
        note['usable_mask_frames'] = int(usable.sum())
        note['confirmed_frames'] = int(weight.sum())
        if weight.sum() == 0:
            note['action'] = 'none (no confirmed mask run)'
            notes.append(note)
            continue
        known = np.array(sorted(meas))
        S = np.stack([np.interp(np.arange(len(fr)), known, [meas[k][j] for k in known]) for j in range(3)], 1)
        weight = np.clip(np.convolve(weight, np.ones(10) / 10, 'same'), 0, 1)
        S = smooth_pinned(np.r_[np.zeros((1, 3)), S * weight[:, None], np.zeros((1, 3))])[1:-1]
        t_new = t[fr] - S
        i0 = np.nanmean([iou(info, f, mesh, s, R[f], t[f]) for f in fr])
        i1 = np.nanmean([iou(info, f, mesh, s, R[f], t_new[k]) for k, f in enumerate(fr)])
        V = body.verts(fr)
        g = lambda tt: float(np.median(np.minimum(*mf.hand_gaps(V, body.hands, surf, R[fr], tt, s).values())))
        g0, g1 = g(t[fr]), g(t_new)
        note.update(mask_iou_before=float(i0), mask_iou_after=float(i1), hand_gap_cm_before=g0, hand_gap_cm_after=g1,
                    shift_cm_median=float(np.median(np.linalg.norm(S, axis=1)) * 100),
                    shift_cm_max=float(np.linalg.norm(S, axis=1).max() * 100))
        if i1 >= i0 + 0.05 and g1 <= g0 + 5:
            t[fr] = t_new.astype(t.dtype)
            note['action'] = 're-centred on the returning mask'
        else:
            note['action'] = 'rejected (mask IoU or grip not better)'
        notes.append(note)
    return dict(z, object_translation=t), notes


# ---------- 3. rigid ring ----------

def _ellipse(job):
    mask_path, = job
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    ys, xs = np.nonzero(m > 127)
    if len(xs) < 500:
        return None
    pts = np.c_[xs, ys].astype(np.float32)
    if len(pts) > 6000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 6000, replace=False)]
    ell = cv2.fitEllipse(pts)
    if min(ell[1]) < 2:
        return None
    q = float((cKDTree(hc.ellipse_points(ell, 360)).query(pts)[0] < 6).mean())
    return ell, q, pts


def _candidates(job):
    ell, cam, radius = job
    return hc.circle_candidates(ell, cam, radius)


def plane_circle(pts, cam, n, h):
    """Circle fitted to mask pixels cast onto the plane n.X = h (camera frame): radius, median radial error."""
    rays = np.c_[(pts[:, 0] - cam['cx']) / cam['fx'], (pts[:, 1] - cam['cy']) / cam['fy'], np.ones(len(pts))]
    X = rays * (h / (rays @ n))[:, None]
    U = np.linalg.svd(n[None])[2][1:].T
    P = X @ U
    A = np.c_[2 * P, np.ones(len(P))]
    p0 = np.linalg.lstsq(A, (P * P).sum(1), rcond=None)[0]
    r0 = np.sqrt(max(p0[2] + p0[:2] @ p0[:2], 1e-8))
    sol = least_squares(lambda p: np.linalg.norm(P - p[:2], axis=1) - p[2], np.r_[p0[:2], r0], loss='soft_l1', f_scale=.01)
    return float(sol.x[2]), float(np.median(np.abs(sol.fun)))


def ring_fix(z, e, info, mesh, body, workers):
    cam, off = info['cam'], info['off']
    frames = np.arange(info['first'], info['last'] + 1)
    v = np.asarray(mesh.vertices)
    c_m, axis_m, r_m = hc.ring_frame(v)
    s_old = float(z['object_scale'])
    tube = s_old * float(np.percentile(np.abs((v - c_m) @ axis_m), 90))
    nf = np.asarray(json.loads((info['dir'] / 'floor.json').read_text())['normal'], float)   # points down
    R0, t0 = z['object_rotation'][frames].astype(float), z['object_translation'][frames].astype(float)
    centre0 = t0 + s_old * R0 @ c_m
    axis0 = R0 @ axis_m
    V = body.verts(frames)
    soles = (V @ nf).max(1)
    hands = V[:, np.r_[body.hands['left'], body.hands['right']][::4]]
    del V
    with ProcessPoolExecutor(workers) as ex:
        ells = list(ex.map(_ellipse, [(str(info['dir'] / 'masks/1' / f'{f - off:06d}.png'),) for f in frames], chunksize=4))
    # ring size from flat frames at the soles
    lowest0 = np.array([((v * s_old) @ R0[k].T + t0[k]) @ nf for k in range(len(frames))]).max(1)
    tilt0 = np.degrees(np.arccos(np.clip(np.abs(axis0 @ nf), 0, 1)))
    flat = (tilt0 <= 15) & (np.abs(lowest0 - soles) <= 0.10)
    radii = []
    for k in np.flatnonzero(flat):
        if ells[k] is None or ells[k][1] < 0.75:
            continue
        r, err = plane_circle(ells[k][2], cam, nf, soles[k] - tube)
        if err < 0.03:
            radii.append(r)
    r_old = r_m * s_old
    size = dict(radius_m_before=r_old, flat_frames=int(flat.sum()), fitted_frames=len(radii))
    s_new = s_old
    if len(radii) >= 20:
        med = float(np.median(radii))
        spread = float((np.percentile(radii, 90) - np.percentile(radii, 10)) / med)
        size.update(radius_m_floor_fit=med, spread=spread)
        if spread <= 0.10 and abs(med / r_old - 1) >= 0.03:
            s_new = s_old * med / r_old
    radius = r_m * s_new
    size['radius_m_after'] = radius
    # baseline at the new size: same ring centre and orientation
    t_base = centre0 - s_new * R0 @ c_m
    good = [k for k, x in enumerate(ells) if x is not None and x[1] >= 0.75]
    with ProcessPoolExecutor(workers) as ex:
        cands = dict(zip(good, ex.map(_candidates, [(ells[k][0], cam, radius) for k in good], chunksize=2)))
    use = np.zeros(len(frames), bool)
    Rn, tn = R0.copy(), t_base.copy()
    prev_n, rows = None, []
    for k, f in enumerate(frames):
        row = dict(frame=int(f))
        if k not in cands:
            rows.append(row)
            continue
        ok = []
        for c, n, rms in cands[k]:
            low = c @ nf + radius * np.sqrt(max(0.0, 1 - (n @ nf) ** 2))
            if low <= soles[k] + 0.08:
                ok.append((c, n, rms))
        row['possible_candidates'] = len(ok)
        if not ok:
            rows.append(row)
            continue
        dist = [float(hc.rim_distance(hands[k], c, n, radius).min() * 100) for c, n, _ in ok]
        if min(dist) <= 15:
            pick = int(np.argmin(dist))
            row['chosen_by'] = 'hand'
        elif prev_n is not None:
            pick = int(np.argmax([abs(n @ prev_n) for _, n, _ in ok]))
            row['chosen_by'] = 'continuity'
        else:
            pick = int(np.argmax([abs(n @ axis0[k]) for _, n, _ in ok]))
            row['chosen_by'] = 'current axis'
        if len(ok) == 1 and len(cands[k]) == 2:
            row['chosen_by'] = 'floor'
        c, n, rms = ok[pick]
        if n @ axis0[k] < 0:
            n = -n
        tgt = hc.ellipse_points(ells[k][0])
        uv = hc.project(hc.circle_points(centre0[k], axis0[k], radius), cam)
        cur = float(np.sqrt(np.mean(np.r_[cKDTree(tgt).query(uv)[0], cKDTree(uv).query(tgt)[0]] ** 2)))
        row.update(rms_px=rms, current_rms_px=cur, quality=ells[k][1])
        if rms < 5 and rms + 2 < cur:
            use[k] = True
            prev_n = n
            Rk = hc.min_rotation(axis0[k], n) @ R0[k]
            Rn[k], tn[k] = Rk, c - s_new * Rk @ c_m
        row['replaced'] = bool(use[k])
        rows.append(row)
    for part in np.split(np.arange(len(frames)), np.flatnonzero(np.diff(use.astype(int)) != 0) + 1):
        if use[part[0]] and len(part) > 2:
            tn[part] = gaussian_filter1d(tn[part], 3.0, axis=0, mode='nearest')
            ref = Rotation.from_matrix(Rn[part[0]])
            rv = (ref.inv() * Rotation.from_matrix(Rn[part])).as_rotvec()
            Rn[part] = (ref * Rotation.from_rotvec(gaussian_filter1d(rv, 3.0, axis=0, mode='nearest'))).as_matrix()
    ramp = np.maximum(np.clip(gaussian_filter1d(use.astype(float), 4.0), 0, 1), use.astype(float)) if use.any() else use.astype(float)
    t_out = ramp[:, None] * tn + (1 - ramp[:, None]) * t_base
    R_out = np.stack([(Rotation.from_matrix(R0[k]) * Rotation.from_rotvec(ramp[k] * (Rotation.from_matrix(R0[k]).inv()
                       * Rotation.from_matrix(Rn[k])).as_rotvec())).as_matrix() for k in range(len(frames))])
    R, t = z['object_rotation'].copy(), z['object_translation'].copy()
    R[frames], t[frames] = R_out.astype(R.dtype), t_out.astype(t.dtype)
    sc = np.array([int(f) in info['scored'] for f in frames])
    depth = (t_out + s_new * R_out @ c_m)[:, 2] - centre0[:, 2]
    note = dict(size=size, replaced_frames=int(use.sum()), replaced_scored=int((use & sc).sum()), scored=int(sc.sum()),
                chosen_by={k: int(sum(r.get('chosen_by') == k and r.get('replaced', False) for r in rows))
                           for k in ('floor', 'hand', 'continuity', 'current axis')},
                ring_depth_change_cm_scored_median=float(np.median(depth[sc]) * 100),
                ring_depth_change_cm_scored_p90abs=float(np.percentile(np.abs(depth[sc]), 90) * 100), frames=rows)
    z = dict(z, object_rotation=R, object_translation=t,
             object_scale=np.asarray(s_new, dtype=z['object_scale'].dtype))
    return z, note


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True, help='motion_fix episodes folder (NPZ + OBJ)')
    ap.add_argument('--motion_report', type=Path, required=True, help='motion_fix report.json (switch repairs)')
    ap.add_argument('--symmetry', type=Path, required=True, help='symmetry.json (motion_symmetry.py)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--episodes', type=int, nargs='*', default=list(range(30)))
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--ring', action='store_true', help='also run the experimental rigid-ring step on the hoops')
    args = ap.parse_args()
    dest = args.out / 'episodes'
    dest.mkdir(parents=True, exist_ok=True)
    mf_report = json.loads(args.motion_report.read_text())
    sym_all = json.loads(args.symmetry.read_text())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    report_path = args.out / 'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for e in args.episodes:
        info = episode_info(e)
        name = info['name']
        shutil.copy2(args.src / f'{name}_object.obj', dest / f'{name}_object.obj')
        src = np.load(args.src / f'{name}.npz')
        z = {k: src[k] for k in src.files}
        mesh = trimesh.load(args.src / f'{name}_object.obj', force='mesh', process=False)
        rep = dict(object=info['obj'])
        z, rep['snap_gate'] = gate_switch_repairs(z, e, info, mesh, mf_report, sym_all[str(e)])
        body = None
        if info['obj'] not in RING_OBJECTS:
            if any(info['states'][int(f)]['state'] in ('occluded', 'lost') for f in range(info['first'], info['last'] + 1)):
                body = mf.Body(z, device)
                surf = trimesh.sample.sample_surface_even(mesh, 4000, seed=0)[0]
                z, rep['recovery'] = recover_lost(z, e, info, mesh, body, surf)
        elif args.ring:
            body = mf.Body(z, device)
            z, rep['ring'] = ring_fix(z, e, info, mesh, body, args.workers)
        for k in src.files:
            assert z[k].shape == src[k].shape and z[k].dtype == src[k].dtype, k
        np.savez(dest / f'{name}.npz', **z)
        rep['changed_arrays'] = [k for k in src.files if not np.array_equal(z[k], src[k])]
        report[str(e)] = rep
        msg = f"{e:2d} {info['obj']:17s} changed {rep['changed_arrays'] or '-'}"
        msg += ''.join(f" | switch {n['frame']}: {n['action']} (jump {n['surface_jump_cm']:.2f} cm)" for n in rep['snap_gate'])
        msg += ''.join(f" | loss {n['span']}: {n['action']}" + (f" IoU {n['mask_iou_before']:.3f}->{n['mask_iou_after']:.3f}, "
                       f"gap {n['hand_gap_cm_before']:.1f}->{n['hand_gap_cm_after']:.1f} cm" if 'mask_iou_after' in n else '')
                       for n in rep.get('recovery', []))
        if 'ring' in rep:
            r = rep['ring']
            msg += (f" | ring radius {r['size']['radius_m_before']:.3f}->{r['size']['radius_m_after']:.3f} m, replaced "
                    f"{r['replaced_frames']} ({r['replaced_scored']}/{r['scored']} scored), by {r['chosen_by']}, scored depth "
                    f"change median {r['ring_depth_change_cm_scored_median']:+.1f} cm")
        print(msg, flush=True)
        del body
        json.dump(report, open(report_path, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
