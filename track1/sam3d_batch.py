#!/usr/bin/env python3
"""Build several SAM3D object meshes in ONE process (the model is loaded once and reused).

The library wrapper (v2d.sam3d.docker.run_image_to_mesh) starts a new process per mesh, so the model is
loaded again for every candidate (~1 min each). This runs v2d.sam3d.lib.image_to_mesh, whose pipeline is a
process-wide singleton, for every job in a JSON list, with the same settings stage 5 used before.
Jobs whose mesh already exists are skipped.

jobs.json: [{"image_path", "mask_path", "mesh_path", "transform_path", "intrinsics_path",
             "depth_path", "depth_intrinsics_path", "depth_mask_path"}, ...]

    /opt/venvs/sam3d/bin/python track1/sam3d_batch.py --jobs jobs.json --weights_dir weights/sam3d
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--weights_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    weights = os.path.abspath(args.weights_dir)
    os.environ.setdefault("TORCH_HOME", os.path.join(weights, "torch_home"))   # as the library wrapper sets them
    os.environ.setdefault("HF_HOME", os.path.join(weights, "hf_home"))
    from v2d.sam3d.lib.image_to_mesh import image_to_mesh

    for i, job in enumerate(json.load(open(args.jobs))):
        if os.path.exists(job["mesh_path"]):
            print(f"[{i}] {job['mesh_path']}: exists, skipped", flush=True)
            continue
        t0 = time.time()
        image_to_mesh(weights_dir=weights, seed=args.seed, with_mesh_postprocess=True, with_texture_baking=True, **job)
        print(f"[{i}] {job['mesh_path']}: built in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
