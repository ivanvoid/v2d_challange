#!/usr/bin/env python3
"""Stage 5b' — metric scale of each candidate mesh (FoundationPose grid search, corrected scoring).

Same coarse-to-fine search as the library's estimate_scale_grid_search (log grid, n_samples per level,
range halved around the winner each level, FoundationPose register() per candidate scale — a size already
tried on an earlier level is reused, not refitted), but every
candidate is scored by rendering the CENTRED mesh with pose @ inv(tf_to_center). The library renders
the centred mesh with the original mesh's pose, i.e. shifted by the mesh's centre offset (up to ~8 cm
and 15-19 % of the object size on our meshes), which corrupts both terms of its score.

Score = mask IoU + exp(-median |MoGe / rendered depth - 1|)   (the library's default weights 1 / 1)

Per candidate in mesh/cand_<frame>/ (search range from scale_init.json, made by stage5_mesh.py):
  scale.json          {"scale", "score", "iou", "depth_mare", "lo", "hi"}
  scale_search.json   every evaluated scale with its IoU, depth error and score, per level
  mesh_scaled.obj     mesh_pretransformed.obj x scale (texture kept)
  ref_pose.json       FoundationPose pose at the candidate frame for the chosen scale
Skips a candidate whose mesh_scaled.obj exists.

    /opt/venvs/foundation_pose/bin/python track1/stage5_scale.py --episode 28 --out_root out \
        --weights weights --debug_log logs/stages/05_mesh/episode_000028.debug.log
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logutil import log, to_debug  # noqa: E402

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask  # noqa: E402
from v2d.common.datatypes import Image as V2dImage  # noqa: E402
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker, nvdiffrast_render  # noqa: E402
from v2d.mesh.lib.mesh import Mesh  # noqa: E402
from PIL import Image  # noqa: E402


def score_pose(tracker, pose, depth: DepthImage, mask: Mask, intr: CameraIntrinsics) -> dict:
    """IoU and depth agreement of the mesh rendered at `pose` (pose of the ORIGINAL mesh)."""
    K = intr.to_matrix()
    Mc = pose.to_matrix() @ np.linalg.inv(tracker._est.get_tf_to_centered_mesh().cpu().numpy())
    with torch.no_grad():
        _, rd, _ = nvdiffrast_render(K, intr.height, intr.width, torch.as_tensor(Mc[None], device="cuda", dtype=torch.float),
                                     glctx=tracker._glctx, mesh_tensors=tracker._est.mesh_tensors, get_normal=False)
    rd = rd[0].cpu().numpy()
    sil, obs = rd > 0.001, mask.mask.astype(bool)
    iou = float((sil & obs).sum() / max((sil | obs).sum(), 1))
    valid = sil & obs & (depth.depth > 0.001)
    mare = float(np.median(np.abs(depth.depth[valid] / rd[valid] - 1.0))) if valid.sum() >= 10 else None
    return {"iou": iou, "depth_mare": mare, "score": iou + (float(np.exp(-mare)) if mare is not None else 0.0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--debug_log", required=True)
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--n_levels", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5, help="register() iterations per candidate scale")
    args = ap.parse_args()

    ep = f"episode_{args.episode:06d}"
    W = os.path.join(os.path.abspath(args.out_root), ep)
    off = json.load(open(os.path.join(W, "track1_meta.json")))["frame_offset"]
    intr = CameraIntrinsics.load(os.path.join(W, "intrinsics_stable.json"))
    frames = json.load(open(os.path.join(W, "mesh", "candidates.json")))["chosen"]
    weights = os.path.join(os.path.abspath(args.weights), "foundation_pose")

    for f in frames:
        C = os.path.join(W, "mesh", f"cand_{f:06d}")
        tag = f"video frame {f + off}"
        if os.path.exists(os.path.join(C, "mesh_scaled.obj")):
            log(f"  {tag}: [skip] scale search")
            continue
        si = json.load(open(os.path.join(C, "scale_init.json")))
        rgb = V2dImage(data=np.asarray(Image.open(os.path.join(W, "frames", f"{f:06d}.png")).convert("RGB")))
        depth = DepthImage.load(os.path.join(W, "depth", f"{f:06d}.png"))
        mask = Mask.load(os.path.join(W, "masks", "1", f"{f:06d}.png"))
        log(f"  {tag}: [run ] scale search {si['lo']:.2f}-{si['hi']:.2f} (start {si['init_scale']:.2f})")
        t0 = time.time()
        search = []
        with to_debug(args.debug_log, f"{tag} scale search"):
            tracker = FoundationPoseTracker(Mesh.load(os.path.join(C, "mesh_pretransformed.obj")), weights)
            log_lo, log_hi = np.log(si["lo"]), np.log(si["hi"])
            best = {"score": -np.inf, "scale": float(np.sqrt(si["lo"] * si["hi"]))}
            seen = {}                                   # the levels overlap: a size already tried is not refitted
            for level in range(args.n_levels):
                for s in np.exp(np.linspace(log_lo, log_hi, args.n_samples)):
                    key = round(float(np.log(s)), 6)
                    if key in seen:
                        r = {**seen[key], "level": level, "cached": True}
                    else:
                        tracker.rescale_to(float(s))
                        pose = tracker.register(rgb, depth, mask, intr, iteration=args.iters)
                        r = {"level": level, "scale": float(s), **score_pose(tracker, pose, depth, mask, intr)}
                        seen[key] = r
                    search.append(r)
                    if r["score"] > best["score"]:
                        best = r
                radius = (log_hi - log_lo) / 4
                log_lo, log_hi = np.log(best["scale"]) - radius, np.log(best["scale"]) + radius
            tracker.rescale_to(best["scale"])
            pose = tracker.register(rgb, depth, mask, intr)
            final = score_pose(tracker, pose, depth, mask, intr)
            tracker._mesh.save(os.path.join(C, "mesh_scaled.obj"))
            pose.save(os.path.join(C, "ref_pose.json"))
        json.dump({"scale": best["scale"], "score": final["score"], "iou": final["iou"], "depth_mare": final["depth_mare"],
                   "lo": si["lo"], "hi": si["hi"], "scoring": "centred render (pose @ inv(tf_to_center))"},
                  open(os.path.join(C, "scale.json"), "w"), indent=1)
        json.dump(search, open(os.path.join(C, "scale_search.json"), "w"), indent=1)
        edge = best["scale"] <= si["lo"] * 1.03 or best["scale"] >= si["hi"] / 1.03
        mare = f"{100 * final['depth_mare']:.1f}%" if final["depth_mare"] is not None else "n/a"
        log(f"  {tag}: [done] scale {best['scale']:.3f} (start {si['init_scale']:.2f}) in {time.time() - t0:.0f}s; "
            f"IoU {final['iou']:.2f}, depth error {mare}{'  WARNING: at the search edge' if edge else ''}")


if __name__ == "__main__":
    main()
