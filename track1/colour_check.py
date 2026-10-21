#!/usr/bin/env python3
"""Prototype: appearance agreement of candidate meshes (rendered texture vs the photo), for 5e near-ties.

For each given candidate + per-video relative size, on the 5e test frames of every video of the object:
register, keep the best distinct orientation by silhouette IoU (as 5e), then on pixels inside both the mask and
the model's silhouette compare the rendered texture with the photo after matching the mean brightness of the two
(per-frame lighting / exposure must not count). Reports median IoU and median colour error per video.

    /opt/venvs/foundation_pose/bin/python colour_check.py --object iron --cands 6:cand_000643 7:cand_000517
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
from logutil import to_debug  # noqa: E402
from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask  # noqa: E402
from v2d.common.datatypes import Image as V2dImage  # noqa: E402
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker, nvdiffrast_render  # noqa: E402
from v2d.mesh.lib.mesh import Mesh  # noqa: E402


def rot_deg(a, b):
    c = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def colour_err(rgb: np.ndarray, col: np.ndarray, both: np.ndarray) -> float:
    a = rgb[both].astype(float) / 255.0
    b = col[both].astype(float)
    if b.max() > 1.0:
        b = b / 255.0
    b = b * (a.mean() / max(b.mean(), 1e-6))          # match overall brightness only (not hue)
    return float(np.abs(a - b).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--cands", nargs="+", required=True, help="episode:cand_XXXXXX")
    ap.add_argument("--out_root", default="out")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--frames_per_video", type=int, default=5)
    args = ap.parse_args()
    rep = json.load(open(os.path.join("logs/stages/05e_object", f"{args.object}.json")))
    rels = {(c["source_episode"], c["candidate"]): {int(k): p["rel"] for k, p in c["per_video"].items()} for c in rep["candidates"]}
    eps = sorted(json.loads(l)["episode_index"] for l in open(paths.META) if json.loads(l)["object"] == args.object)
    videos = {}
    for e in eps:
        W = os.path.join(os.path.abspath(args.out_root), f"episode_{e:06d}")
        c = json.load(open(os.path.join(W, "mesh", "choice.json")))
        chk = json.load(open(os.path.join(W, "mesh", f"cand_{c['chosen']:06d}", "check.json")))
        fr = [p["frame"] for p in chk["frames"] if not p["own"]] or [p["frame"] for p in chk["frames"]]
        fr = [fr[int(i)] for i in np.linspace(0, len(fr) - 1, min(args.frames_per_video, len(fr))).round()]
        intr = CameraIntrinsics.load(os.path.join(W, "intrinsics_stable.json"))
        data = [{"rgb": np.asarray(Image.open(os.path.join(W, "frames", f"{f:06d}.png")).convert("RGB")),
                 "depth": DepthImage.load(os.path.join(W, "depth", f"{f:06d}.png")),
                 "mask": Mask.load(os.path.join(W, "masks", "1", f"{f:06d}.png"))} for f in fr]
        videos[e] = {"intr": intr, "data": [d for d in data if d["mask"].mask.astype(bool).any()]}
    for spec in args.cands:
        se, name = spec.split(":")
        se = int(se)
        path = os.path.join(args.out_root, f"episode_{se:06d}", "mesh", name, "mesh_scaled.obj")
        with to_debug("/tmp/opencode/colour_check.debug.log", "load"):
            tr = FoundationPoseTracker(Mesh.load(path), os.path.join(os.path.abspath(args.weights), "foundation_pose"))
        out = []
        for e, v in videos.items():
            rel = rels.get((se, name), {}).get(e, 1.0)
            with to_debug("/tmp/opencode/colour_check.debug.log", "rescale"):
                tr.rescale_to(rel)
            ious, cerrs = [], []
            for d in v["data"]:
                with to_debug("/tmp/opencode/colour_check.debug.log", "register"):
                    tr.register(V2dImage(data=d["rgb"]), d["depth"], d["mask"], v["intr"], iteration=5)
                distinct = []
                for h in tr._est.poses[:40].cpu().numpy():
                    if all(rot_deg(h, g) >= 45 for g in distinct):
                        distinct.append(h)
                    if len(distinct) == 5:
                        break
                K = v["intr"].to_matrix()
                with torch.no_grad():
                    col, dep, _ = nvdiffrast_render(K, v["intr"].height, v["intr"].width,
                                                    torch.as_tensor(np.stack(distinct), device="cuda", dtype=torch.float),
                                                    glctx=tr._glctx, mesh_tensors=tr._est.mesh_tensors, get_normal=False)
                dep = dep.cpu().numpy() > 0.001
                m = d["mask"].mask.astype(bool)
                iou = [(dep[i] & m).sum() / max((dep[i] | m).sum(), 1) for i in range(len(distinct))]
                b = int(np.argmax(iou))
                ious.append(float(iou[b]))
                # among poses that fit the outline about as well (a cube fits in many orientations), the one whose
                # texture matches the photo best: "can this texture match the photo at an outline-fitting pose"
                ce = [colour_err(d["rgb"], col[i].cpu().numpy(), dep[i] & m) if (dep[i] & m).sum() > 50 else np.nan
                      for i in range(len(distinct)) if iou[i] >= iou[b] - 0.03]
                cerrs.append(float(np.nanmin(ce)) if np.isfinite(ce).any() else np.nan)
            out.append(f"ep{e} x{rel:.2f} IoU {np.median(ious):.2f} colour err {np.nanmedian(cerrs):.3f}")
        print(f"{name} of ep {se}: " + " | ".join(out) + f"  -> mean colour err "
              f"{np.mean([float(s.split('colour err ')[1]) for s in out]):.3f}", flush=True)
        del tr
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
