#!/usr/bin/env python3
"""Local evaluation of a packed Track 1 submission — what the scorer sees, without the reference.

  extract  (.venv-submit: pandas + pyarrow)   parquet -> one npz per episode with exactly the
           packed arrays: scored frames, MHR pose/scales/shape, object rotation/translation/scale,
           and the BUDGETED object mesh (zero-area padding faces dropped).
  pen      (sam3d_body venv: torch + MHR)      contact penetration as the PEN scorer computes it
           (v2dlb.mhr_submission._penetration_depth: winding-number inside test + distance to the
           surface, in the object's mesh frame, x object scale), mean over hand points per frame,
           mean over frames per episode, mean over episodes. Two approximations, both reported:
             * hand points: the scorer uses 512 fixed MHR hand vertices from a private asset; we use
               512 vertices evenly sampled (fixed seed) from the MHR hand vertices
             * the alignment scale (reference / ours) is unknown and taken as 1
           Writes <out>/pen.json and prints per-episode values, worst first.

    .venv-submit/bin/python track1/eval_local.py extract --parquet submission/track1.parquet --out logs/eval/baseline
    /opt/venvs/sam3d_body/bin/python track1/eval_local.py pen --out logs/eval/baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

sys.path.insert(0, str(paths.KIT))
from v2dlb import mhr_submission as ms  # noqa: E402

EPISODE_FRAME = 999999


def extract(parquet: str, out: str) -> None:
    import pandas as pd
    t0 = time.time()
    df = pd.read_parquet(parquet, columns=["row_id", "x", "y", "z"])
    p = df["row_id"].str.split("_", expand=True)
    df = df.assign(ep=p[1].astype(int), fr=p[2].astype(int), arr=p[3].astype(int), pt=p[4].astype(int))
    names = ms.array_order()
    os.makedirs(out, exist_ok=True)
    for ep, g in df.groupby("ep"):
        g = g.sort_values(["arr", "fr", "pt"])
        per_ep = {names[a]: gg[["x", "y", "z"]].to_numpy() for a, gg in g[g.fr == EPISODE_FRAME].groupby("arr")}
        frames = np.sort(g.loc[g.fr != EPISODE_FRAME, "fr"].unique())
        per_fr = {}
        for a, gg in g[g.fr != EPISODE_FRAME].groupby("arr"):
            vals = gg[["x", "y", "z"]].to_numpy()
            per_fr[names[a]] = vals.reshape(len(frames), -1, 3)
        pose = per_fr["mhr_pose"].reshape(len(frames), -1)[:, :136]
        scales = per_ep["mhr_scales"].reshape(-1)[:68]
        shape = per_ep["mhr_shape"].reshape(-1)[:45]
        R = per_fr["object_rotation"]
        t = per_fr["object_translation"][:, 0, :]
        s = float(per_ep["object_scale"][0, 0])
        v = per_ep["object_mesh_vertices"]
        f = per_ep["object_mesh_faces"].round().astype(np.int64)
        tri = v[f]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        f = f[area > 0]
        np.savez(os.path.join(out, f"episode_{ep:06d}.npz"), frames=frames, pose=pose, scales=scales, shape=shape,
                 R=R, t=t, s=s, mesh_v=v, mesh_f=f)
    print(f"extracted {df.ep.nunique()} episodes to {out} in {time.time() - t0:.0f}s")


def pen(out: str, n_hand: int = 512) -> None:
    import torch
    sys.path.insert(0, str(paths.KIT_TRACK1))
    from mesh_to_mhr_params import MHR
    from qa_episode import hand_vertex_ids
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHR(str(paths.DATA / "weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt"), dev, 64, "float32")
    rows = []
    t0 = time.time()
    for path in sorted(Path(out).glob("episode_*.npz")):
        d = np.load(path)
        e = int(path.stem.split("_")[1])
        pose = torch.as_tensor(d["pose"], dtype=torch.float64, device=dev)
        z = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device=dev)
        verts = (mhr.run(pose, z)[0] / 1000.0).cpu().numpy()                     # [T, V, 3] metres
        hid = hand_vertex_ids(mhr, pose[:1], z)
        hid = np.random.default_rng(0).choice(hid, size=min(n_hand, len(hid)), replace=False)
        hands = verts[:, hid]                                                   # [T, P, 3]
        local = np.einsum("tpi,tij->tpj", hands - d["t"][:, None], d["R"]) / d["s"]
        depth = ms._penetration_depth(local.reshape(-1, 3), d["mesh_v"], d["mesh_f"]).reshape(hands.shape[:2])
        per_frame = np.maximum(depth * d["s"], 0).mean(1) * 100                 # cm
        rows.append({"episode": e, "pen_cm": float(per_frame.mean()), "max_frame_cm": float(per_frame.max()),
                     "frames_penetrating": int((per_frame > 0).sum()), "frames": int(len(per_frame)),
                     "per_frame_cm": per_frame.round(5).tolist()})
    mean = float(np.mean([r["pen_cm"] for r in rows]))
    json.dump({"pen_mean_over_episodes_cm": mean, "hand_points": n_hand, "episodes": rows},
              open(os.path.join(out, "pen.json"), "w"), indent=1)
    for r in sorted(rows, key=lambda r: -r["pen_cm"]):
        if r["pen_cm"] > 0:
            print(f"  episode {r['episode']:2d}: PEN {r['pen_cm']:.4f} cm  (worst frame {r['max_frame_cm']:.3f} cm, "
                  f"{r['frames_penetrating']}/{r['frames']} frames penetrating)")
    print(f"PEN mean over {len(rows)} episodes: {mean:.4f} cm  ({time.time() - t0:.0f}s)")


def stretches(frames: np.ndarray) -> list[np.ndarray]:
    """Index runs of consecutive frame numbers (the scorer computes accelerations per run)."""
    cut = np.flatnonzero(np.diff(frames) != 1) + 1
    return [r for r in np.split(np.arange(len(frames)), cut)]


def accel_cm(x: np.ndarray, runs: list[np.ndarray]) -> np.ndarray:
    """|second difference| per frame inside each run (cm/frame^2); x [T, ..., 3] in metres."""
    out = []
    for r in runs:
        if len(r) >= 3:
            a = x[r][:-2] - 2 * x[r][1:-1] + x[r][2:]
            out.append(np.linalg.norm(a, axis=-1).reshape(len(r) - 2, -1).mean(1) * 100)
    return np.concatenate(out) if out else np.zeros(0)


def proxies(out: str, n_hand: int = 512, contact_cm: float = 2.0) -> None:
    """Per-episode proxies on exactly the packed scored frames (see the module docstring)."""
    import torch
    from scipy.spatial import cKDTree
    import trimesh
    sys.path.insert(0, str(paths.KIT_TRACK1))
    from mesh_to_mhr_params import MHR
    from qa_episode import hand_vertex_ids, LEFT_WRIST, RIGHT_WRIST
    from pen_fast import PenetrationMesh
    joints22 = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]  # scorer's MHR_TABLE3_BODY_JOINT_INDICES
    meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open(paths.META)}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHR(str(paths.DATA / "weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt"), dev, 64, "float32")
    rows, t0 = [], time.time()
    for path in sorted(Path(out).glob("episode_*.npz")):
        d = np.load(path)
        e = int(path.stem.split("_")[1])
        frames = d["frames"]
        runs = stretches(frames)
        pose = torch.as_tensor(d["pose"], dtype=torch.float64, device=dev)
        z = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device=dev)
        v_mm, j = mhr.run(pose, z)
        verts, joints = (v_mm / 1000.0).cpu().numpy(), j.cpu().numpy()
        hid = hand_vertex_ids(mhr, pose[:1], z)
        rest = verts[0]
        left = np.linalg.norm(rest[hid] - joints[0, LEFT_WRIST], axis=1) < np.linalg.norm(rest[hid] - joints[0, RIGHT_WRIST], axis=1)
        pick = np.random.default_rng(0).choice(len(hid), size=min(n_hand, len(hid)), replace=False)
        hid, left = hid[pick], left[pick]
        R, t, s = d["R"], d["t"], float(d["s"])
        local = np.einsum("tpi,tij->tpj", verts[:, hid] - t[:, None], R) / s       # hands in mesh frame
        pm = PenetrationMesh(d["mesh_v"], d["mesh_f"])
        tm = trimesh.Trimesh(d["mesh_v"], d["mesh_f"], process=False)
        tree = cKDTree(trimesh.sample.sample_surface_even(tm, 30000, seed=0)[0])
        depth = pm.depth(local.reshape(-1, 3)).reshape(local.shape[:2]) * s           # m, inside only
        dist = tree.query(local.reshape(-1, 3))[0].reshape(local.shape[:2]) * s       # m, unsigned (sampled)
        gap = np.where(depth > 0, 0.0, dist)
        gap_any = gap.min(1) * 100
        gap_l, gap_r = gap[:, left].min(1) * 100, gap[:, ~left].min(1) * 100
        pen = depth.mean(1) * 100
        acc_h = accel_cm(joints[:, joints22], runs)
        acc_o = accel_cm(t, runs)
        steps = np.concatenate([np.linalg.norm(np.diff(t[r], axis=0), axis=1) for r in runs if len(r) > 1]) * 100
        extent = (d["mesh_v"].max(0) - d["mesh_v"].min(0)) * s
        body_z = joints[:, 1, 2]                                                    # joint 1 (pelvis) depth
        rows.append({
            "episode": e, "object": meta[e]["object"], "frames": int(len(frames)), "stretches": len(runs),
            "pen_cm": float(pen.mean()),
            "gap_cm_median": float(np.median(gap_any)), "gap_cm_p90": float(np.percentile(gap_any, 90)),
            "contact_frac": float((gap_any < contact_cm).mean()),
            "gap_left_cm_median": float(np.median(gap_l)), "gap_right_cm_median": float(np.median(gap_r)),
            "acc_h_self": float(acc_h.mean()) if len(acc_h) else 0.0,
            "acc_o_self": float(acc_o.mean()) if len(acc_o) else 0.0,
            "obj_step_cm_p99": float(np.percentile(steps, 99)) if len(steps) else 0.0,
            "obj_steps_over_10cm": int((steps > 10).sum()),
            "object_extent_m": [round(float(x), 3) for x in extent], "object_size_m": float(extent.max()),
            "object_z_m": float(np.median(t[:, 2])), "body_z_m": float(np.median(body_z)),
            "body_minus_object_z_cm": float(np.median(body_z - t[:, 2]) * 100),
            "per_frame": {"frames": frames.tolist(), "gap_cm": gap_any.round(2).tolist(),
                          "pen_cm": pen.round(4).tolist()},
        })
        r = rows[-1]
        print(f"  ep {e:2d} {r['object'][:12]:12s} gap {r['gap_cm_median']:6.1f} cm (contact {100 * r['contact_frac']:3.0f}%)"
              f"  PEN {r['pen_cm']:.4f}  accH {r['acc_h_self']:.3f}  accO {r['acc_o_self']:.3f}  "
              f"size {r['object_size_m']:.2f} m  body-obj z {r['body_minus_object_z_cm']:+.0f} cm", flush=True)
    # size consistency of each object across its videos
    by_obj = {}
    for r in rows:
        by_obj.setdefault(r["object"], []).append(r["object_size_m"])
    for r in rows:
        sizes = by_obj[r["object"]]
        r["size_vs_object_median"] = float(r["object_size_m"] / np.median(sizes))
    keys = ["pen_cm", "gap_cm_median", "contact_frac", "acc_h_self", "acc_o_self", "obj_steps_over_10cm"]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary["size_spread_max"] = float(max(max(v) / min(v) for v in by_obj.values()))
    json.dump({"summary": summary, "episodes": rows, "hand_points": n_hand, "contact_cm": contact_cm},
              open(os.path.join(out, "proxies.json"), "w"), indent=1)
    print("mean over episodes: " + ", ".join(f"{k} {v:.4f}" for k, v in summary.items()) + f"  ({time.time() - t0:.0f}s)")


def compare(a: str, b: str) -> None:
    """Per-episode proxy deltas between two evaluated runs (b - a)."""
    A = {r["episode"]: r for r in json.load(open(os.path.join(a, "proxies.json")))["episodes"]}
    B = {r["episode"]: r for r in json.load(open(os.path.join(b, "proxies.json")))["episodes"]}
    keys = [("gap_cm_median", "gap", "{:7.1f}"), ("contact_frac", "contact", "{:7.2f}"), ("pen_cm", "PEN", "{:7.4f}"),
            ("acc_h_self", "accH", "{:7.3f}"), ("acc_o_self", "accO", "{:7.3f}"), ("object_size_m", "size", "{:7.2f}"),
            ("body_minus_object_z_cm", "bodyobj_z", "{:7.0f}")]
    print("ep  object        " + "".join(f"{n:>17s}" for _, n, _ in keys))
    for e in sorted(set(A) & set(B)):
        cells = []
        for k, _, f in keys:
            x, y = A[e][k], B[e][k]
            cells.append((f.format(x).strip() + "->" + f.format(y).strip()).rjust(17) if abs(y - x) > 1e-9 else "=".rjust(17))
        print(f"{e:2d}  {A[e]['object'][:12]:12s} " + "".join(cells))
    sa, sb = (json.load(open(os.path.join(p, "proxies.json")))["summary"] for p in (a, b))
    print("mean: " + ", ".join(f"{k} {sa[k]:.4f}->{sb[k]:.4f}" for k in sa))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("extract"); a.add_argument("--parquet", required=True); a.add_argument("--out", required=True)
    b = sub.add_parser("pen"); b.add_argument("--out", required=True); b.add_argument("--hand_points", type=int, default=512)
    c = sub.add_parser("proxies"); c.add_argument("--out", required=True)
    c.add_argument("--hand_points", type=int, default=512); c.add_argument("--contact_cm", type=float, default=2.0)
    k = sub.add_parser("compare"); k.add_argument("a"); k.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.parquet, args.out)
    elif args.cmd == "pen":
        pen(args.out, args.hand_points)
    elif args.cmd == "proxies":
        proxies(args.out, args.hand_points, args.contact_cm)
    else:
        compare(args.a, args.b)


if __name__ == "__main__":
    main()
