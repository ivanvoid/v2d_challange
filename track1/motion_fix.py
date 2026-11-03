#!/usr/bin/env python3
"""Automatic motion fixes on top of the full_v2 final episodes (new outputs only).

1. Symmetry spin: for objects whose mesh is rotationally symmetric (symmetry.json), the image
   cannot show rotation about the symmetry axis. Remove it by parallel transport: each frame keeps
   its visible orientation (where the axis points) and takes the smallest rotation from the previous
   frame. 180-degree flips that map the object onto itself are unwrapped the same way. Translation is
   left unchanged (ACC-O scores translation only); the surface moves by at most ~0.6 cm.
2. Lost track: where the tracker has ~zero mask overlap for >= 15 frames, and the saved object
   sits far from both hands, replace the pose between the last and the first well-observed frames.
   The pre-gap pose rides on the hand that held it before, the post-gap pose on the hand that held
   it after, and the two are blended; with no hand near either end it moves smoothly between them.
3. Symmetry-switch tumbles: a single-frame raw-track jump whose relative rotation maps the mesh
   (almost) onto itself is a representation switch, but smoothing turns it into a multi-frame tumble.
   Around it the raw track is re-expressed continuously, smoothed with pinned ends and switched back
   at the jump frame. Kept only if the rendered silhouette fits the SAM2 mask better in that window
   and the object does not move away from the hand holding it (nearly-symmetric objects: roller handle).
   Hoops are skipped (their orientation comes from the hoop geometry steps, not the tracker).

Order: 2, 3, 1. Reads $V2D_DATA (default /local-data/ivan/v2d_full) read-only; writes --out and
report.json next to it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get('V2D_DATA', '/local-data/ivan/v2d_full'))
SRC = DATA / 'experiments/full_v2/pen_hands/merged/episodes'
T1 = HERE.parent if (HERE.parent / 'qa_episode.py').exists() else Path('/local-data/ivan/v2d/v2d_challange/track1')
LOST_IOU, GOOD_IOU, MIN_SPAN, ANCHOR_SEARCH = 0.05, 0.5, 15, 30
FAR_CM, HOLD_CM = 15.0, 30.0  # body sits 16-53 cm behind the scene depth, so held objects show 10-30 cm gaps


# ---------- 1. symmetry ----------

def twist_about(Q: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Rotation part of Q about `axis` (swing-twist decomposition), as a matrix."""
    x, y, z, w = Rotation.from_matrix(Q).as_quat()
    p = (np.array([x, y, z]) @ axis) * axis
    q = np.r_[p, w]
    n = np.linalg.norm(q)
    return np.eye(3) if n < 1e-9 else Rotation.from_quat(q / n).as_matrix()


def angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(R).magnitude()))


def remove_spin(R: np.ndarray, axis: np.ndarray, flips: list[np.ndarray]) -> np.ndarray:
    """Per frame R_t * g_t with g_t in the symmetry group, chosen to minimise the step from the previous output."""
    out = np.empty_like(R)
    out[0] = R[0]
    for i in range(1, len(R)):
        best = None
        for F in [np.eye(3), *flips]:
            cand = R[i] @ F
            cand = cand @ twist_about(out[i - 1].T @ cand, axis).T
            step = angle_deg(out[i - 1].T @ cand)
            if best is None or step < best[0]:
                best = (step, cand)
        out[i] = best[1]
    return out


def fix_symmetry(z: dict, sym: dict) -> tuple[dict, dict]:
    cont = [s for s in sym['symmetries'] if s['kind'] == 'continuous']
    if not cont:
        return z, {}
    axis = np.asarray(cont[0]['axis'])
    flips = [Rotation.from_rotvec(np.pi * np.asarray(s['axis'])).as_matrix()
             for s in sym['symmetries'] if s['kind'] == 'discrete' and s['order'] == 2]
    c, s = np.asarray(sym['centre']), float(z['object_scale'])
    R, t = z['object_rotation'], z['object_translation']
    R2 = remove_spin(R, axis, flips)
    # Translation is kept: ACC-O scores translation only, and the per-frame surface-preserving correction
    # s (R - R2) c (<= 0.6 cm, the mesh centroid is a few mm off its origin) adds jitter to it.
    shift = s * np.einsum('tij,j->ti', R - R2, c)
    z = dict(z, object_rotation=R2.astype(R.dtype))
    visible = np.degrees(np.arccos(np.clip(np.einsum('ti,ti->t', R @ axis, R2 @ axis) * 1, -1, 1)))
    return z, dict(symmetry_axis=axis.tolist(), flips=len(flips),
                   visible_axis_change_deg_max=float(np.minimum(visible, 180 - visible).max()),
                   surface_offset_cm_max=float(np.linalg.norm(shift, axis=1).max() * 100))


# ---------- 2. lost track ----------

def lost_spans(states, off, first, last):
    iou = {s['video_frame']: s['iou'] for s in states}
    bad = [f for f in range(first, last + 1) if iou[f] < LOST_IOU]
    spans, run = [], []
    for f in bad:
        if run and f != run[-1] + 1:
            spans.append(run); run = []
        run.append(f)
    if run:
        spans.append(run)
    out = []
    for r in spans:
        if len(r) < MIN_SPAN:
            continue
        a, b = r[0], r[-1]
        pre = next((f for f in range(a - 1, max(first, a - ANCHOR_SEARCH) - 1, -1) if iou[f] >= GOOD_IOU), a - 1)
        post = next((f for f in range(b + 1, min(last, b + ANCHOR_SEARCH) + 1) if iou[f] >= GOOD_IOU), b + 1)
        out.append(dict(lost=(a, b), pre=max(pre, first), post=min(post, last)))
    return out


def kabsch(P: np.ndarray, Q: np.ndarray):
    """R, t with Q ~ P @ R.T + t."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, qc - pc @ R.T


class Body:
    def __init__(self, z, device):
        sys.path.insert(0, str(DATA / 'v2d_submission_kit/tools/track1'))
        sys.path.insert(0, str(T1))
        from mesh_to_mhr_params import MHR
        from qa_episode import hand_vertex_ids, LEFT_WRIST, RIGHT_WRIST
        self.mhr = MHR(str(DATA / 'weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt'), device, 64, 'float32')
        self.dev = device
        self.z = torch.as_tensor(np.r_[z['scales'], z['shape']][None], dtype=torch.float64, device=device)
        self.pose = z['pose']
        p0 = torch.as_tensor(self.pose[:1], dtype=torch.float64, device=device)
        hid = hand_vertex_ids(self.mhr, p0, self.z)
        v, j = self.mhr.run(p0, self.z)
        v, j = v[0].cpu().numpy() / 1000, j[0].cpu().numpy()
        left = np.linalg.norm(v[hid] - j[LEFT_WRIST], axis=1) < np.linalg.norm(v[hid] - j[RIGHT_WRIST], axis=1)
        self.hands = {'left': hid[left], 'right': hid[~left]}

    def verts(self, frames):
        out = []
        with torch.no_grad():
            for i in range(0, len(frames), 64):
                q = torch.as_tensor(self.pose[frames[i:i + 64]], dtype=torch.float64, device=self.dev)
                out.append(self.mhr.run(q, self.z)[0].cpu().numpy() / 1000)
        return np.concatenate(out)


def hand_gaps(verts, hands, surf_local, R, t, s):
    """Per frame and hand: smallest distance (cm) from a hand vertex to the posed object surface samples."""
    gaps = {}
    tree = cKDTree(surf_local)
    for name, ids in hands.items():
        g = []
        for k in range(len(verts)):
            local = (verts[k, ids] - t[k]) @ R[k] / s
            g.append(tree.query(local)[0].min() * s * 100)
        gaps[name] = np.asarray(g)
    return gaps


def smooth_window(R, t, a, b, lam=200.0, pin=2):
    """Stage-7-style smoother (min sum w|y-x|^2 + lam |D2 y|^2) on frames a..b. The `pin` frames at each
    end are held fixed, so position and velocity join the untouched frames without a kink."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    n = b - a + 1
    w = np.ones(n); w[:pin] = w[-pin:] = 1e8
    D = diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))
    A = (diags(w) + lam * (D.T @ D)).tocsc()
    t[a:b + 1] = np.stack([spsolve(A, w * t[a:b + 1, k]) for k in range(3)], 1)
    q = Rotation.from_matrix(R[a:b + 1]).as_quat()
    for i in range(1, n):
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    q = np.stack([spsolve(A, w * q[:, k]) for k in range(4)], 1)
    R[a:b + 1] = Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix()


def smoothstep(x):
    return x * x * (3 - 2 * x)


def fix_lost(z, e, body: Body, surf_local):
    ep = DATA / 'out' / f'episode_{e:06d}'
    meta = json.loads((ep / 'track1_meta.json').read_text())
    states = json.loads((ep / 'track/states.json').read_text())['frames']
    R, t, s = z['object_rotation'].copy(), z['object_translation'].copy(), float(z['object_scale'])
    notes = []
    for sp in lost_spans(states, meta['frame_offset'], meta['scored_first'], meta['scored_last']):
        p, q = sp['pre'], sp['post']
        frames = np.arange(p, q + 1)
        V = body.verts(frames)
        gaps = hand_gaps(V, body.hands, surf_local, R[frames], t[frames], s)
        nearest = np.minimum(gaps['left'], gaps['right'])
        inner = nearest[1:-1]
        end_gap = {h: (float(gaps[h][0]), float(gaps[h][-1])) for h in gaps}
        hand_p = min(gaps, key=lambda h: gaps[h][0])
        hand_q = min(gaps, key=lambda h: gaps[h][-1])
        note = dict(lost=sp['lost'], pre=p, post=q, span_gap_cm_median=float(np.median(inner)),
                    end_gaps_cm=end_gap)
        if np.median(inner) < FAR_CM or np.median(inner) < max(nearest[0], nearest[-1]) + 10:
            note['action'] = 'none (object stays near a hand)'
            notes.append(note)
            continue
        Op, Oq = (R[p], t[p]), (R[q], t[q])
        w = smoothstep((frames - p) / (q - p))
        slerp = lambda Ra, Rb, ww: Rotation.from_matrix(Ra if ww <= 0 else Rb if ww >= 1 else
                                                       Slerp([0, 1], Rotation.from_matrix([Ra, Rb]))([ww]).as_matrix()[0]).as_matrix()
        if gaps[hand_p][0] < HOLD_CM and gaps[hand_q][-1] < HOLD_CM:
            ip, iq = body.hands[hand_p], body.hands[hand_q]
            for k, f in enumerate(frames):
                Ra_, ta_ = kabsch(V[0, ip], V[k, ip])       # pre-holding hand motion p -> f
                Rb_, tb_ = kabsch(V[-1, iq], V[k, iq])      # post-holding hand motion q -> f
                A = (Ra_ @ Op[0], Op[1] @ Ra_.T + ta_)      # pre pose carried by that hand
                B = (Rb_ @ Oq[0], Oq[1] @ Rb_.T + tb_)      # post pose carried by that hand
                R[f] = slerp(A[0], B[0], w[k])
                t[f] = (1 - w[k]) * A[1] + w[k] * B[1]
            note['action'] = f'follow {hand_p} hand' + ('' if hand_p == hand_q else f' -> {hand_q} hand')
        else:
            for k, f in enumerate(frames):
                R[f] = slerp(Op[0], Oq[0], w[k])
                t[f] = (1 - w[k]) * Op[1] + w[k] * Oq[1]
            note['action'] = 'smooth move between observed poses'
        smooth_window(R, t, max(meta['scored_first'], p - 20), min(meta['scored_last'], q + 20))
        after = hand_gaps(V, body.hands, surf_local, R[frames], t[frames], s)
        note['span_gap_cm_median_after'] = float(np.median(np.minimum(after['left'], after['right'])[1:-1]))
        note['max_move_cm'] = float(np.linalg.norm(t[frames] - z['object_translation'][frames], axis=1).max() * 100)
        notes.append(note)
    return dict(z, object_rotation=R, object_translation=t), notes


# ---------- 3. symmetry-switch tumbles ----------

JUMP_DEG, JUMP_APPEARANCE, JUMP_IOU, WINDOW_MAX, WINDOW_DEV_DEG, IOU_GAIN = 45.0, 0.05, 0.3, 40, 8.0, 0.01
GRIP_WORSE_CM = 1.0   # a repair may not move the object away from the hand holding it


def raw_rotations(e, frames, off):
    q = []
    for f in frames:
        d = json.loads((DATA / 'out' / f'episode_{e:06d}' / 'poses' / f'{f - off:06d}.json').read_text())
        w, x, y, zz = d['rotation']
        q.append([x, y, zz, w])
    return Rotation.from_quat(q).as_matrix()


def appearance_error(mesh, G, n=3000):
    """Mean distance from the mesh rotated by G about its surface centroid to itself, / object size."""
    ref = trimesh.sample.sample_surface_even(mesh, 20000, seed=1)[0]
    pts = trimesh.sample.sample_surface_even(mesh, n, seed=2)[0]
    c = ref.mean(0)
    tree = cKDTree(ref)
    return float((tree.query((pts - c) @ G.T + c)[0].mean() - tree.query(pts)[0].mean()) / np.max(mesh.extents))


def silhouette(v, faces, camera, width=640, height=480):
    """Filled projected mesh at width x height (same projection as the pipeline_videos renderer)."""
    import cv2
    z = np.maximum(v[:, 2], 1e-4)
    uv = np.c_[camera['fx'] * v[:, 0] / z + camera['cx'], camera['fy'] * v[:, 1] / z + camera['cy']]
    uv *= np.array([width / camera['width'], height / camera['height']])
    tri = np.clip(np.rint(uv[faces[(v[faces, 2] > 0.01).all(1)]]), -10000, 10000).astype(np.int32)
    tri = tri[(tri[:, :, 0].min(1) < width) & (tri[:, :, 0].max(1) >= 0) & (tri[:, :, 1].min(1) < height) & (tri[:, :, 1].max(1) >= 0)]
    canvas = np.zeros((height, width), np.uint8)
    for p in tri:
        cv2.fillConvexPoly(canvas, p, 255)
    return canvas > 0


def mask_iou(e, frames, off, R, t, s, mesh, camera):
    """Mean IoU of the rendered object silhouette with the SAM2 object mask (640x480), frames with a mask."""
    import cv2
    v, f_ = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    vals = []
    for k, f in enumerate(frames):
        m = cv2.imread(str(DATA / 'out' / f'episode_{e:06d}' / 'masks/1' / f'{f - off:06d}.png'), cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (640, 480), interpolation=cv2.INTER_NEAREST) > 127
        if m.sum() < 20:
            continue
        sil = silhouette(v * s @ R[k].T + t[k], f_, camera)
        vals.append((sil & m).sum() / max(1, (sil | m).sum()))
    return float(np.mean(vals)) if vals else float('nan')


def repair_window(Rr, Rf, j, G):
    """Window [a, b] around a raw switch at index j (G = Rr[j-1]^T Rr[j]) and its repaired rotations, or None.

    The window runs from where the final output last agrees with the raw track before j to where it
    agrees again after j. Inside, the raw track is re-expressed without the switch, smoothed with two
    pinned frames at each end, and switched back from j on, so the look is continuous and the
    representation changes in a single frame.
    """
    dev = lambda A, B: np.degrees(Rotation.from_matrix(A.T @ B).magnitude())
    a = next((k for k in range(j - 3, max(-1, j - WINDOW_MAX), -1) if dev(Rf[k], Rr[k]) < WINDOW_DEV_DEG), None)
    b = next((k for k in range(j + 2, min(len(Rr), j + WINDOW_MAX)) if dev(Rf[k], Rr[k]) < WINDOW_DEV_DEG), None)
    if a is None or b is None:
        return None
    a, b = a - 3, b + 3          # pin well outside the tumble, so its speed doesn't carry into the window
    if a < 0 or b > len(Rr) - 1:
        return None
    U = np.concatenate([Rr[a:j], Rr[j:b + 1] @ G.T])
    U[:2], U[-2:] = Rf[a:a + 2], Rf[b - 1:b + 1] @ G.T
    smooth_window(U, np.zeros((len(U), 3)), 0, len(U) - 1)
    return a, b, np.concatenate([U[:j - a], U[j - a:] @ G])


def fix_switch_tumbles(z, e, mesh, obj, body, surf_local):
    """Undo multi-frame tumbles that smoothing made out of single-frame (near-)symmetry switches in the raw track."""
    if obj == 'hula_hoop':        # hoop orientation comes from the dedicated hoop geometry steps, not the tracker
        return z, []
    ep = DATA / 'out' / f'episode_{e:06d}'
    meta = json.loads((ep / 'track1_meta.json').read_text())
    camera = json.loads((ep / 'intrinsics_stable.json').read_text())
    off, f0, f1 = meta['frame_offset'], meta['scored_first'], meta['scored_last']
    frames = np.arange(f0, f1 + 1)
    iou = {st['video_frame']: st['iou'] for st in json.loads((ep / 'track/states.json').read_text())['frames']}
    Rr = raw_rotations(e, frames, off)
    R, t, s = z['object_rotation'].copy(), z['object_translation'], float(z['object_scale'])
    step = np.degrees(Rotation.from_matrix(np.einsum('tji,tjk->tik', Rr[:-1], Rr[1:])).magnitude())
    notes = []
    for i in np.flatnonzero(step > JUMP_DEG):
        j = i + 1
        fj = int(frames[j])
        G = Rr[j - 1].T @ Rr[j]
        note = dict(frame=fj, jump_deg=float(step[i]), appearance_err=appearance_error(mesh, G))
        notes.append(note)
        if note['appearance_err'] > JUMP_APPEARANCE or min(iou[fj - 1], iou[fj]) < JUMP_IOU:
            note['action'] = 'skip (not a symmetry switch, or poorly tracked)'
            continue
        Rf = R[frames].astype(np.float64)
        rep = repair_window(Rr, Rf, j, G)
        if rep is None:
            note['action'] = 'skip (final never returns to the raw track nearby)'
            continue
        a, b, new = rep
        win = frames[a:b + 1]
        before = mask_iou(e, win, off, Rf[a:b + 1], t[win], s, mesh, camera)
        after = mask_iou(e, win, off, new, t[win], s, mesh, camera)
        V = body.verts(win)
        grip = lambda RR: float(np.median(np.minimum(*hand_gaps(V, body.hands, surf_local, RR, t[win], s).values())))
        g0, g1 = grip(Rf[a:b + 1]), grip(new)
        note.update(window=[int(win[0]), int(win[-1])], mask_iou_before=before, mask_iou_after=after,
                    hand_gap_cm_before=g0, hand_gap_cm_after=g1)
        if g1 > g0 + GRIP_WORSE_CM:
            note['action'] = 'rejected (object would move away from the holding hand)'
        elif after >= before + IOU_GAIN:
            R[win] = new.astype(R.dtype)
            note['action'] = 'repaired'
        else:
            note['action'] = 'rejected (silhouette does not fit the mask better)'
    return dict(z, object_rotation=R), notes


def rotation_stats(R, first, last):
    r = Rotation.from_matrix(R[first:last + 1])
    d = np.degrees((r[1:] * r[:-1].inv()).magnitude())
    return dict(median=float(np.median(d)), p99=float(np.percentile(d, 99)), over5=int((d > 5).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--episodes', type=int, nargs='*', default=list(range(30)))
    ap.add_argument('--out', type=Path, required=True, help='episodes folder to write (report.json goes next to it)')
    ap.add_argument('--symmetry', type=Path, required=True, help='written by motion_symmetry.py (experiments copy: symmetry.py)')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sym_all = json.loads(args.symmetry.read_text())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    report_path = args.out.parent / 'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for e in args.episodes:
        name = f'episode_{e:06d}'
        shutil.copy2(SRC / f'{name}_object.obj', args.out / f'{name}_object.obj')
        src = np.load(SRC / f'{name}.npz')
        z = {k: src[k] for k in src.files}
        meta = json.loads((DATA / 'out' / name / 'track1_meta.json').read_text())
        mesh = trimesh.load(SRC / f'{name}_object.obj', force='mesh', process=False)
        surf = trimesh.sample.sample_surface_even(mesh, 4000, seed=0)[0]
        body = Body(z, device)
        z, lost = fix_lost(z, e, body, surf)
        obj = json.loads((DATA / 'out' / name / 'object_track.json').read_text())['object']
        z, switches = fix_switch_tumbles(z, e, mesh, obj, body, surf)
        z, spin = fix_symmetry(z, sym_all[str(e)])
        for k in src.files:
            assert z[k].shape == src[k].shape and z[k].dtype == src[k].dtype, k
        np.savez(args.out / f'{name}.npz', **z)
        changed = [k for k in src.files if not np.array_equal(z[k], src[k])]
        f0, f1 = meta['scored_first'], meta['scored_last']
        report[str(e)] = dict(changed_arrays=changed, lost_track=lost, switches=switches, symmetry=spin,
                              rot_before=rotation_stats(src['object_rotation'], f0, f1),
                              rot_after=rotation_stats(z['object_rotation'], f0, f1))
        rb, ra = report[str(e)]['rot_before'], report[str(e)]['rot_after']
        print(f"{e:2d} changed {changed or '-'} | rot/frame median {rb['median']:.2f}->{ra['median']:.2f} deg, "
              f">5deg {rb['over5']}->{ra['over5']}" + ''.join(f" | lost {n['lost']}: {n['action']}, gap "
              f"{n['span_gap_cm_median']:.0f}->{n.get('span_gap_cm_median_after', n['span_gap_cm_median']):.0f} cm"
              for n in lost) + ''.join(f" | switch {n['frame']}: {n['action']}" + (
                  f" (mask IoU {n['mask_iou_before']:.3f}->{n['mask_iou_after']:.3f}, hand gap "
                  f"{n['hand_gap_cm_before']:.1f}->{n['hand_gap_cm_after']:.1f} cm)" if 'mask_iou_after' in n else '')
                  for n in switches), flush=True)
    json.dump(report, open(report_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
