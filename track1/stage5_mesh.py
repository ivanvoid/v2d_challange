#!/usr/bin/env python3
"""Stage 5b — build one object mesh per candidate frame (SAM3D) and fix its metric scale.

For each frame in out/episode_XXXXXX/mesh/candidates.json, in mesh/cand_<frame>/:
  textured_mesh.obj + material_0.png        SAM3D from the frame, its mask and MoGe depth (seed 0),
                                              WITH texture baking (FoundationPose compares rendered
                                              colour to the image; a flat grey mesh tracks worse)
  mesh_transform.json, mesh_pretransformed.obj  SAM3D rotation + scale applied (camera-aligned)
  scale_init.json                            initial scale from the mask: mask width/height in metres
                                              (pixels x MoGe depth / f) vs the camera-aligned mesh's
                                              x/y extent; geometric mean of the two ratios
  mesh_scaled.obj, scale.json, ref_pose.json FoundationPose scale search in [init/1.6, init*1.6]
                                              (log grid, 5 samples x 3 levels) + the pose at that frame
Steps skip when their output exists. Model output goes to --debug_log.

Run with the pipeline interpreter and V2D_BACKEND=venv:
    python track1/stage5_mesh.py --episode 28 --out_root out --weights weights --debug_log ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logutil import log, to_debug  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale  # noqa: E402
from v2d.pipelines.run_ego_wilor import _apply_sam3d_transform  # noqa: E402
from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--debug_log", required=True)
    args = ap.parse_args()

    ep = f"episode_{args.episode:06d}"
    W = os.path.join(os.path.abspath(args.out_root), ep)
    Wt = os.path.abspath(args.weights)
    off = json.load(open(os.path.join(W, "track1_meta.json")))["frame_offset"]
    intr = os.path.join(W, "intrinsics_stable.json")
    frames = json.load(open(os.path.join(W, "mesh", "candidates.json")))["chosen"]

    K = json.load(open(intr))

    def init_scale(C: str, f: int) -> dict:
        """Mask size in metres vs the camera-aligned mesh's x/y size -> starting scale."""
        path = os.path.join(C, "scale_init.json")
        if os.path.exists(path):
            return json.load(open(path))
        v = np.asarray(trimesh.load(os.path.join(C, "mesh_pretransformed.obj"), force="mesh", process=False).vertices)
        m = cv2.imread(os.path.join(W, "masks", "1", f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
        px = cv2.imread(os.path.join(W, "depth", f"{f:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        d = 1.0 / np.maximum(px / 65535.0, 1e-6) - 1.0
        ys, xs = np.nonzero(m)
        z = float(np.median(d[m]))
        mw, mh = float(np.ptp(xs)) * z / K["fx"], float(np.ptp(ys)) * z / K["fy"]
        ew, eh = float(np.ptp(v[:, 0])), float(np.ptp(v[:, 1]))
        s0 = float(np.sqrt((mw / ew) * (mh / eh)))
        r = {"init_scale": s0, "lo": s0 / 1.6, "hi": s0 * 1.6, "mask_wh_m": [mw, mh], "mesh_wh_m": [ew, eh],
             "mask_depth_m": z}
        json.dump(r, open(path, "w"), indent=1)
        return r

    for f in frames:
        C = os.path.join(W, "mesh", f"cand_{f:06d}")
        os.makedirs(C, exist_ok=True)
        rgb, depth = os.path.join(W, "frames", f"{f:06d}.png"), os.path.join(W, "depth", f"{f:06d}.png")
        mask = os.path.join(W, "masks", "1", f"{f:06d}.png")
        tag = f"video frame {f + off}"
        steps = [
            ("SAM3D mesh", os.path.join(C, "textured_mesh.obj"),
             lambda: run_image_to_mesh(image_path=rgb, mask_path=mask, mesh_path=os.path.join(C, "textured_mesh.obj"),
                                       transform_path=os.path.join(C, "mesh_transform.json"),
                                       intrinsics_path=os.path.join(C, "mesh_intrinsics.json"),
                                       weights_dir=f"{Wt}/sam3d", seed=0, with_mesh_postprocess=True,
                                       with_texture_baking=True, depth_path=depth,
                                       depth_intrinsics_path=intr, depth_mask_path=mask, dev=False)),
            ("apply SAM3D transform", os.path.join(C, "mesh_pretransformed.obj"),
             lambda: _apply_sam3d_transform(os.path.join(C, "textured_mesh.obj"),
                                            os.path.join(C, "mesh_transform.json"),
                                            os.path.join(C, "mesh_pretransformed.obj"))),
            ("FoundationPose scale search", os.path.join(C, "mesh_scaled.obj"),
             lambda: run_estimate_mesh_scale(mesh_path=os.path.join(C, "mesh_pretransformed.obj"), rgb_path=rgb,
                                             depth_path=depth, mask_path=mask, intrinsics_path=intr,
                                             weights_dir=f"{Wt}/foundation_pose",
                                             scale_path=os.path.join(C, "scale.json"),
                                             rescaled_mesh_path=os.path.join(C, "mesh_scaled.obj"),
                                             pose_path=os.path.join(C, "ref_pose.json"),
                                             lo=init_scale(C, f)["lo"], hi=init_scale(C, f)["hi"],
                                             n_samples=5, n_levels=3, dev=False)),
        ]
        for name, out, fn in steps:
            if name == "FoundationPose scale search" and not os.path.exists(out):
                si = init_scale(C, f)
                log(f"  {tag}: mask {si['mask_wh_m'][0]:.2f} x {si['mask_wh_m'][1]:.2f} m vs mesh "
                    f"{si['mesh_wh_m'][0]:.2f} x {si['mesh_wh_m'][1]:.2f} m -> start scale {si['init_scale']:.2f}, "
                    f"search {si['lo']:.2f}-{si['hi']:.2f}")
            if os.path.exists(out):
                log(f"  {tag}: [skip] {name}")
                continue
            log(f"  {tag}: [run ] {name}")
            t0 = time.time()
            with to_debug(args.debug_log, f"{tag} {name}"):
                fn()
            if not os.path.exists(out):
                raise SystemExit(f"{ep}: {name} for {tag} produced no {os.path.basename(out)}")
            log(f"  {tag}: [done] {name} in {time.time() - t0:.0f}s")
        sc, si = json.load(open(os.path.join(C, "scale.json")))["scale"], init_scale(C, f)
        edge = sc <= si["lo"] * 1.03 or sc >= si["hi"] / 1.03
        tex = os.path.exists(os.path.join(C, "material_0.png"))
        log(f"  {tag}: scale {sc:.3f} (start {si['init_scale']:.2f}){'  WARNING: at the search edge' if edge else ''}"
            f", texture {'yes' if tex else 'NO'}")


if __name__ == "__main__":
    main()
