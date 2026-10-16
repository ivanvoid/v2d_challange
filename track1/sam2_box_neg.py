#!/usr/bin/env python3
"""SAM2 video propagation with BOX + NEGATIVE POINT prompts (the v2d wrapper allows only one prompt type).

Run in the sam2 venv:
    /opt/venvs/sam2/bin/python track1/sam2_box_neg.py --frames_dir out/ep/frames \
        --prompts out/ep/mask_prompts.json --masks_dir out/ep/masks --weights_dir weights/sam2
mask_prompts.json: {"prompts": [{"frame": int, "box": {x0,y0,x1,y1}, "neg": [[x, y], ...]}, ...]}
Writes <masks_dir>/1/<frame:06d>.png for every frame (propagated backward and forward).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from v2d.sam2.lib.video_to_masks import _get_predictor

OBJECT_ID = 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--weights_dir", required=True)
    args = ap.parse_args()

    prompts = json.load(open(args.prompts))["prompts"]
    predictor = _get_predictor(os.path.abspath(args.weights_dir))
    frames = {}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = predictor.init_state(os.path.abspath(args.frames_dir))
        for p in prompts:
            b = p["box"]
            neg = p.get("neg") or []
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=int(p["frame"]), obj_id=OBJECT_ID,
                points=np.asarray(neg, np.float32) if neg else None,
                labels=np.zeros(len(neg), np.int32) if neg else None,
                box=np.asarray([b["x0"], b["y0"], b["x1"], b["y1"]], np.float32))
        for reverse in (True, False):
            for fidx, obj_ids, masks in predictor.propagate_in_video(state, reverse=reverse):
                for i, oid in enumerate(obj_ids):
                    if oid == OBJECT_ID:
                        frames[fidx] = (masks[i, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
    out = os.path.join(args.masks_dir, str(OBJECT_ID))
    os.makedirs(out, exist_ok=True)
    for fidx, m in frames.items():
        Image.fromarray(m).save(os.path.join(out, f"{fidx:06d}.png"))
    print(f"wrote {len(frames)} masks to {out}")


if __name__ == "__main__":
    main()
