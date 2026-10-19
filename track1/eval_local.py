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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("extract"); a.add_argument("--parquet", required=True); a.add_argument("--out", required=True)
    b = sub.add_parser("pen"); b.add_argument("--out", required=True); b.add_argument("--hand_points", type=int, default=512)
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.parquet, args.out)
    else:
        pen(args.out, args.hand_points)


if __name__ == "__main__":
    main()
