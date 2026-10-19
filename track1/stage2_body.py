#!/usr/bin/env python3
"""Stage 2b — SAM3D-Body per frame with the actor's box AND person mask.

The library runner (v2d.sam3d_body ... run_estimate_mhr_params) always runs the body decoder only
and, when given masks, crops to the MASK's bounding box — which drops body parts hidden behind the
object (legs behind a desk). This runner keeps the library's model, output format and conventions:
  crop        the repaired actor box (body/bbox_track.pt, stage2_track.py)
  prompt      the actor's person mask (body/person_masks/NNNNNN.png) where one exists (mask-conditioned
              inference), otherwise the box only; if a masked frame returns nothing it is re-run box-only
  decoder     --mode body (default): body decoder; full: + hand decoder (finger poses; only PEN uses hand
              points, and it made SAM3D-Body ~2x and the MHR fit far slower, so it is off)
  intrinsics  intrinsics_stable.json (real focal length)
Writes body/mhr_params.pt and body/mhr_mesh.pt exactly like the library (export_mhr_outputs), and
body/mhr_params.pt.meta.json with the mode and the prompt used per frame.

    /opt/venvs/sam3d_body/bin/python track1/stage2_body.py --episode 11 --out_root out --weights weights
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logutil import log  # noqa: E402

from v2d.common.datatypes import CameraIntrinsics  # noqa: E402
from v2d.sam3d_body.lib.estimate_mhr_params import (_normalize_mask, coalesce_mhr_outputs_dict,  # noqa: E402
                                                     export_mhr_outputs)
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--mode", choices=["full", "body"], default="body")
    ap.add_argument("--no_mask", action="store_true", help="box prompt only")
    args = ap.parse_args()

    ep = f"episode_{args.episode:06d}"
    W = Path(args.out_root).resolve() / ep
    B = W / "body"
    n = json.load(open(W / "track1_meta.json"))["n_frames"]
    K = CameraIntrinsics.load(str(W / "intrinsics_stable.json")).to_matrix()
    cam_int = torch.from_numpy(np.asarray(K, dtype=np.float64))[None]
    boxes = np.asarray(torch.load(B / "bbox_track.pt", map_location="cpu", weights_only=False)["bbox_track"], np.float32)
    assert len(boxes) == n, f"{len(boxes)} boxes for {n} frames"

    wd = Path(args.weights).resolve() / "sam3d_body" / "sam-3d-body-dinov3"
    model, cfg = load_sam_3d_body(checkpoint_path=str(wd / "model.ckpt"), device=torch.device("cuda"),
                                  mhr_path=str(wd / "assets" / "mhr_model.pt"))
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    torch.cuda.empty_cache = lambda: None                 # as the library: no per-frame CUDA sync

    outs, prompts, t0 = [], [], time.time()
    for f in range(n):
        img = cv2.cvtColor(cv2.imread(str(W / "frames" / f"{f:06d}.png")), cv2.COLOR_BGR2RGB)
        mpath = B / "person_masks" / f"{f:06d}.png"
        mask = None
        if not args.no_mask and mpath.exists():
            m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            if (m > 0).any():
                mask = _normalize_mask(m, img.shape[:2])
        res = est.process_one_image(img=img, bboxes=boxes[f][None], masks=mask, cam_int=cam_int,
                                    inference_type=args.mode)
        prompt = "box+mask" if mask is not None else "box"
        if not res and mask is not None:
            res = est.process_one_image(img=img, bboxes=boxes[f][None], masks=None, cam_int=cam_int,
                                        inference_type=args.mode)
            prompt = "box (mask run returned nothing)"
        if not res:
            raise SystemExit(f"{ep}: SAM3D-Body returned no person at local frame {f}")
        outs.append(res[0]); prompts.append(prompt)
        if f % 100 == 0:
            print(f"  frame {f}/{n} ({(time.time() - t0) / (f + 1):.2f} s/frame)", flush=True)

    params, mesh = export_mhr_outputs(coalesce_mhr_outputs_dict(outs), faces=est.model.head_pose.faces)
    torch.save(params, B / "mhr_params.pt")
    torch.save(mesh, B / "mhr_mesh.pt")
    counts = {p: prompts.count(p) for p in sorted(set(prompts))}
    json.dump({"runner": "track1/stage2_body.py", "mode": args.mode, "mask_prompt": not args.no_mask,
               "prompt_counts": counts, "per_frame_prompt": prompts, "n_frames": n,
               "seconds": round(time.time() - t0)}, open(B / "mhr_params.pt.meta.json", "w"), indent=1)
    log(f"{ep}: SAM3D-Body ({args.mode} decoder) on {n} frames in {time.time() - t0:.0f}s; prompts {counts}")


if __name__ == "__main__":
    main()
