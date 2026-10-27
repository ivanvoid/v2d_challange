#!/usr/bin/env python3
"""Ground-truth-free QA for one packed Track 1 episode.

Reads the episode exactly as it will be submitted (submission/episodes/*.npz +
*_object.obj), re-poses the body with the MHR model the host uses, and measures
what can be checked without the reference:

  acc_h_self_cm      mean |2nd difference| of the MHR joints   (GT is ~0.12-0.13)
  acc_o_self_cm      mean |2nd difference| of object translation
  pen_cm             mean depth of MHR hand vertices inside the object mesh
                     (the PEN metric up to the host's alignment scale)
  contact_gap_cm     per frame, closest MHR hand vertex to the object surface;
                     the scored span IS the hand-object contact span, so this
                     should be ~0 if body and object share one metric frame
  wrist_gap_cm       MHR wrist vs depth-aligned WiLoR wrist (real detections only)
  wrist_dz_cm        same, depth component only (MHR minus WiLoR)
  obj_iou            object silhouette vs the SAM2 object mask
  body_in_bbox       fraction of projected body vertices inside the person bbox

Writes <out>.json and prints one summary line plus any failed gates.

Run with the sam3d_body venv:

    /opt/venvs/sam3d_body/bin/python track1/qa_episode.py --episode 14 \
        --ep_dir out/episode_000014 --sub_dir submission/episodes \
        --spans submission/scored_spans.json \
        --model weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt \
        --out submission/reports/episode_000014.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
KIT_TRACK1 = paths.KIT_TRACK1
sys.path.insert(0, str(KIT_TRACK1))
from mesh_to_mhr_params import HAND_COLS, NS, MHR  # noqa: E402

RIGHT_WRIST, LEFT_WRIST = 41, 77  # MHR skeleton joints (sam_3d_body wrist_twist_joint_idxs)

GATES = {  # metric: (threshold, direction, flag)
    "acc_h_self_cm": (0.5, ">", "body_jitter"),
    "acc_o_self_cm": (0.5, ">", "object_jitter"),
    # Local QA is approximate (sampled surface normals), but one centimetre
    # was far too permissive for a leaderboard where PEN is reported in cm.
    # This is a review WARN, not an automatic submission rejection.
    "pen_cm": (0.05, ">", "penetration"),
    "contact_gap_median_cm": (10.0, ">", "no_contact"),
    "abs_body_minus_object_dz_cm": (10.0, ">", "body_object_depth_mismatch"),
    "obj_iou_mean": (0.5, "<", "object_reprojection"),
    "body_in_bbox": (0.8, "<", "body_reprojection"),
}
# Reported, not gated: on ~30 px exo hands the WiLoR depth alignment is unreliable.
INFO_ONLY = ("wrist_gap_median_cm", "wrist_dz_median_cm")


def accel_cm(x: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    a = x[:-2] - 2 * x[1:-1] + x[2:]
    return float(np.linalg.norm(a, axis=-1).mean() * 100)


def hand_vertex_ids(mhr: MHR, pose: torch.Tensor, z: torch.Tensor) -> np.ndarray:
    """Vertices that move with the finger parameters and lie distal to a wrist: the hands.

    Pose correctives also nudge the forearm, so movement alone over-selects; keep
    only vertices within 20 cm of a wrist on the finger side of it.
    """
    base_mm, joints = mhr.run(pose[:1], z)
    bumped = pose[:1].clone()
    bumped[:, HAND_COLS] += 0.3
    moved = (mhr.run(bumped, z)[0] - base_mm).norm(dim=-1)[0].cpu().numpy() > 2.0  # mm
    v = base_mm[0].cpu().numpy() / 1000.0
    j = joints[0].cpu().numpy()
    keep = np.zeros(len(v), bool)
    for w in (RIGHT_WRIST, LEFT_WRIST):
        near = moved & (np.linalg.norm(v - j[w], axis=1) < 0.20)
        if not near.any():
            continue
        axis = v[near].mean(0) - j[w]
        axis /= np.linalg.norm(axis) + 1e-9
        keep |= near & ((v - j[w]) @ axis > 0.0)
    return np.flatnonzero(keep)


def surface_query(mesh: trimesh.Trimesh, n: int = 40000):
    pts, fid = trimesh.sample.sample_surface_even(mesh, n, seed=0)
    if len(pts) < n // 4:
        pts, fid = trimesh.sample.sample_surface(mesh, n, seed=0)
    return cKDTree(pts), pts, mesh.face_normals[fid]


def signed_dist(tree, pts, normals, q: np.ndarray) -> np.ndarray:
    """Approximate signed distance (negative inside) via nearest surface sample + its normal."""
    d, i = tree.query(q)
    sign = np.sign(np.einsum("ij,ij->i", q - pts[i], normals[i]))
    sign[sign == 0] = 1
    return d * sign


def raster(uv: np.ndarray, faces: np.ndarray, hw) -> np.ndarray:
    m = np.zeros(hw, np.uint8)
    tri = np.round(uv[faces]).astype(np.int32)
    cv2.fillPoly(m, list(tri), 1)
    return m.astype(bool)


def load_moge_depth(path: Path) -> np.ndarray:
    """uint16 PNG with px = 65535/(d+1) -> metres (v2d.hamer.lib.align_hands._load_depth)."""
    px = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    return 1.0 / np.maximum(px / 65535.0, 1e-6) - 1.0


def depth_bias_cm(points: np.ndarray, uv: np.ndarray, depth: np.ndarray) -> float | None:
    """Median (model z - MoGe z) over the front-most model point per pixel, in cm."""
    h, w = depth.shape
    px = np.round(uv).astype(np.int64)
    ok = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h) & (points[:, 2] > 0)
    if ok.sum() < 20:
        return None
    lin = px[ok, 1] * w + px[ok, 0]
    z = points[ok, 2]
    order = np.lexsort((z, lin))
    lin, z = lin[order], z[order]
    first = np.r_[True, lin[1:] != lin[:-1]]
    lin, z = lin[first], z[first]
    diff = z - depth.reshape(-1)[lin]
    return float(np.median(diff) * 100)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--ep_dir", type=Path, required=True)
    ap.add_argument("--sub_dir", type=Path, required=True)
    ap.add_argument("--spans", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frame_offset", type=int, default=0, help="video frame of local frame 000000")
    ap.add_argument("--reproj_every", type=int, default=5)
    ap.add_argument("--export_body", type=Path, default=None,
                    help="write the submitted body as {pred_vertices [n_local,V,3], faces} for render_results.py")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    span = json.loads(args.spans.read_text())[str(args.episode)]
    first, last = span["first"], span["last"]
    stem = args.sub_dir / f"episode_{args.episode:06d}"
    sub = np.load(f"{stem}.npz")
    frames = np.arange(first, last + 1)

    # Body, re-posed exactly as the host will.
    mhr = MHR(str(args.model), args.device, chunk=16, precision="float32")
    pose = torch.as_tensor(sub["pose"][frames], dtype=torch.float64, device=args.device)
    z = torch.as_tensor(np.concatenate([sub["scales"], sub["shape"]])[None], dtype=torch.float64,
                        device=args.device)
    verts_mm, joints = mhr.run(pose, z)
    verts = (verts_mm / 1000.0).cpu().numpy()
    joints = joints.cpu().numpy()
    faces = mhr.model.faces.cpu().numpy() if hasattr(mhr.model, "faces") else None
    hand_ids = hand_vertex_ids(mhr, pose, z)

    # Object, posed exactly as the host will: p = s R p_mesh + t.
    mesh = trimesh.load(f"{stem}_object.obj", force="mesh", process=False)
    R = sub["object_rotation"][frames].astype(np.float64)
    t = sub["object_translation"][frames].astype(np.float64)
    s = float(sub["object_scale"])
    observed = sub["object_pose_observed"][frames] if "object_pose_observed" in sub.files else None
    tree, spts, snrm = surface_query(mesh)

    # Hand-object relation: move hand points into object-mesh coordinates per frame.
    pen, gap = [], []
    for k in range(len(frames)):
        hp = verts[k, hand_ids]
        local = ((hp - t[k]) @ R[k]) / s  # inverse of s R p + t (R orthonormal)
        sd = signed_dist(tree, spts, snrm, local) * s
        pen.append(np.maximum(-sd, 0).mean() * 100)
        gap.append(np.abs(sd).min() * 100)
    pen, gap = np.asarray(pen), np.asarray(gap)

    rep: dict = {
        "episode": args.episode,
        "scored_first": first, "scored_last": last,
        "acc_h_self_cm": accel_cm(joints),
        "acc_o_self_cm": accel_cm(t),
        "pen_cm": float(pen.mean()),
        "contact_gap_median_cm": float(np.median(gap)),
        "contact_frac_under_2cm": float((gap < 2).mean()),
        "object_scale": s,
        "object_extent_m": float(np.ptp(mesh.vertices, axis=0).max() * s),
        "object_pose_observed_frac": float(observed.mean()) if observed is not None else None,
        "body_depth_median_m": float(np.median(joints[:, :, 2])),
        "object_depth_median_m": float(np.median(t[:, 2])),
        "n_hand_vertices": int(len(hand_ids)),
    }

    # WiLoR witness (real, depth-aligned detections only).
    wpath = args.ep_dir / "body" / "wilor_witness.npz"
    if wpath.exists():
        w = np.load(wpath)
        wf = w["frame"] + args.frame_offset
        keep = (wf >= first) & (wf <= last)
        if keep.any():
            k = wf[keep] - first
            mhr_w = np.stack([joints[k, RIGHT_WRIST], joints[k, LEFT_WRIST]], 1)  # [n,2,3]
            ww = w["wrist"][keep][:, None]
            d = np.linalg.norm(mhr_w - ww, axis=-1)
            same = np.where(w["is_right"][keep], 0, 1)
            pick = d.argmin(1)  # handedness-agnostic: nearest MHR wrist
            rep["wrist_gap_median_cm"] = float(np.median(d[np.arange(len(d)), pick]) * 100)
            rep["wrist_dz_median_cm"] = float(np.median(
                (mhr_w[np.arange(len(d)), pick, 2] - ww[:, 0, 2])) * 100)
            rep["wrist_handedness_agree"] = float((pick == same).mean())
            rep["wilor_frames_in_span"] = int(keep.sum())
        else:
            rep["wilor_frames_in_span"] = 0
    else:
        rep["wilor_frames_in_span"] = None

    # Reprojection on a subset of frames.
    intr = json.load(open(args.ep_dir / "intrinsics_stable.json"))
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
    hw = (int(intr["height"]), int(intr["width"]))

    def project(p):
        uvw = p @ K.T
        return uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)

    obj_id = None
    ot = args.ep_dir / "object_track.json"
    if ot.exists():
        obj_id = json.load(open(ot))["object_id"]
    bbox_track = None
    bt = args.ep_dir / "body" / "bbox_track.pt"
    if bt.exists():
        bbox_track = np.asarray(torch.load(bt, map_location="cpu", weights_only=False)["bbox_track"])

    ious, inside, body_dz, obj_dz = [], [], [], []
    obj_v = np.asarray(mesh.vertices) * s
    obj_f = np.asarray(mesh.faces)
    for k in range(0, len(frames), args.reproj_every):
        local = frames[k] - args.frame_offset
        dpath = args.ep_dir / "depth" / f"{local:06d}.png"
        depth = load_moge_depth(dpath) if dpath.exists() else None
        pv = obj_v @ R[k].T + t[k]
        if depth is not None:
            b = depth_bias_cm(verts[k], project(verts[k]), depth)
            o = depth_bias_cm(pv, project(pv), depth)
            if b is not None:
                body_dz.append(b)
            if o is not None:
                obj_dz.append(o)
        if obj_id is not None:
            mpath = args.ep_dir / "masks" / str(obj_id) / f"{local:06d}.png"
            if mpath.exists():
                gt = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE) > 0
                if (pv[:, 2] > 0).all():
                    pr = raster(project(pv), obj_f, hw)
                    union = (pr | gt).sum()
                    if union:
                        ious.append((pr & gt).sum() / union)
        if bbox_track is not None and 0 <= local < len(bbox_track):
            x0, y0, x1, y1 = bbox_track[local][:4]
            uv = project(verts[k])
            pad = 0.1 * max(x1 - x0, y1 - y0)
            inside.append(((uv[:, 0] >= x0 - pad) & (uv[:, 0] <= x1 + pad) &
                           (uv[:, 1] >= y0 - pad) & (uv[:, 1] <= y1 + pad)).mean())
    rep["obj_iou_mean"] = float(np.mean(ious)) if ious else None
    rep["obj_iou_min"] = float(np.min(ious)) if ious else None
    rep["body_in_bbox"] = float(np.mean(inside)) if inside else None
    rep["body_vs_moge_dz_cm"] = float(np.median(body_dz)) if body_dz else None
    rep["object_vs_moge_dz_cm"] = float(np.median(obj_dz)) if obj_dz else None
    if body_dz and obj_dz:
        rep["body_minus_object_dz_cm"] = rep["body_vs_moge_dz_cm"] - rep["object_vs_moge_dz_cm"]
        rep["abs_body_minus_object_dz_cm"] = abs(rep["body_minus_object_dz_cm"])

    fit = args.ep_dir / "body" / "mhr_fit.npz"
    if fit.exists():
        rep["mhr_fit"] = json.loads(str(np.load(fit)["report"])).get("vertex_error_mm")

    flags = []
    for key, (thr, op, flag) in GATES.items():
        v = rep.get(key)
        if v is None or not np.isfinite(v):
            continue
        if (op == ">" and v > thr) or (op == "<" and v < thr):
            flags.append(flag)
    rep["flags"] = flags

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=1))

    if args.export_body is not None:
        n_local = len(list((args.ep_dir / "frames").glob("*.png")))
        faces_src = torch.load(args.ep_dir / "body" / "mhr_mesh.pt", map_location="cpu",
                               weights_only=False)["faces"]
        out_v = np.full((n_local, verts.shape[1], 3), np.nan, np.float32)
        local = frames - args.frame_offset
        ok = (local >= 0) & (local < n_local)
        out_v[local[ok]] = verts[ok]
        args.export_body.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"pred_vertices": torch.from_numpy(out_v), "faces": faces_src}, args.export_body)

    def f(key, fmt="{:.2f}"):
        v = rep.get(key)
        return "-" if v is None else fmt.format(v)
    print(f"[qa {args.episode:06d}] accH={f('acc_h_self_cm')} accO={f('acc_o_self_cm')} "
          f"pen={f('pen_cm')} gap={f('contact_gap_median_cm')} "
          f"bodyDz={f('body_vs_moge_dz_cm')} objDz={f('object_vs_moge_dz_cm')} "
          f"wilorWrist={f('wrist_gap_median_cm')} objIoU={f('obj_iou_mean')} "
          f"bodyInBox={f('body_in_bbox')} flags={','.join(flags) or 'none'}")


if __name__ == "__main__":
    main()
