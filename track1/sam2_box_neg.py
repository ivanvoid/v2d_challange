#!/usr/bin/env python3
"""SAM2 video propagation with BOX + NEGATIVE POINT prompts (the v2d wrapper allows only one prompt type).
Run in the sam2 venv:
    /opt/venvs/sam2/bin/python track1/sam2_box_neg.py --frames_dir out/ep/frames \
        --prompts out/ep/mask_prompts.json --masks_dir out/ep/masks --weights_dir weights/sam2
mask_prompts.json: {"prompts": [{"frame": int, "box": {x0,y0,x1,y1}, "neg": [[x, y], ...]}, ...]}

SAM2 propagates BACKWARD (from the last prompt to frame 0) and FORWARD (from the first prompt to the
end). Frames between the first and the last prompt get both passes; the final mask is the AVERAGE of the
two probability maps, thresholded at 0.5 (before, the forward pass simply overwrote the backward one).
Writes, for every frame:
  <masks_dir>/1/NNNNNN.png              final binary mask (0/255)
  <masks_dir>_prob/1/NNNNNN.png         final probability x 255 (soft mask)
  <masks_dir>_pass/{bwd,fwd}/NNNNNN.png each pass's own binary mask (where that pass reached)
  <masks_dir>_conf.json                 per frame: area, prob (mean probability inside the mask),
                                        uncertain (pixels with probability 0.3-0.7 / mask area),
                                        fb_iou (IoU of the two passes' masks, null if only one reached),
                                        passes
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
    md = args.masks_dir.rstrip("/")
    out_bin, out_prob = os.path.join(md, str(OBJECT_ID)), os.path.join(md + "_prob", str(OBJECT_ID))
    out_pass = {k: os.path.join(md + "_pass", k) for k in ("bwd", "fwd")}
    for d in (out_bin, out_prob, *out_pass.values()):
        os.makedirs(d, exist_ok=True)

    bwd = {}                                       # frame -> uint8 probability x 255 of the backward pass
    conf = {}

    def save(fidx: int, p: np.ndarray, fb_iou, passes: int) -> None:
        m = p >= 128
        Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(out_bin, f"{fidx:06d}.png"))
        Image.fromarray(p).save(os.path.join(out_prob, f"{fidx:06d}.png"))
        area = int(m.sum())
        unc = int(((p >= 77) & (p <= 178)).sum())
        conf[fidx] = {"area": area, "prob": float(p[m].mean() / 255) if area else None,
                      "uncertain": float(unc / max(area, 1)) if area else None, "fb_iou": fb_iou, "passes": passes}

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
        for fidx, obj_ids, masks in predictor.propagate_in_video(state, reverse=True):
            i = list(obj_ids).index(OBJECT_ID)
            p = (torch.sigmoid(masks[i, 0].float()) * 255).round().to(torch.uint8).cpu().numpy()
            bwd[fidx] = p
            Image.fromarray(((p >= 128) * 255).astype(np.uint8)).save(os.path.join(out_pass["bwd"], f"{fidx:06d}.png"))
        for fidx, obj_ids, masks in predictor.propagate_in_video(state, reverse=False):
            i = list(obj_ids).index(OBJECT_ID)
            pf = (torch.sigmoid(masks[i, 0].float()) * 255).round().to(torch.uint8).cpu().numpy()
            Image.fromarray(((pf >= 128) * 255).astype(np.uint8)).save(os.path.join(out_pass["fwd"], f"{fidx:06d}.png"))
            if fidx in bwd:
                pb = bwd.pop(fidx)
                mb, mf = pb >= 128, pf >= 128
                union = int((mb | mf).sum())
                fb = float((mb & mf).sum() / union) if union else 1.0
                save(fidx, ((pb.astype(np.uint16) + pf) // 2).astype(np.uint8), fb, 2)
            else:
                save(fidx, pf, None, 1)
    for fidx, pb in bwd.items():                   # frames before the first prompt: backward pass only
        save(fidx, pb, None, 1)
    json.dump({"frames": {str(k): v for k, v in sorted(conf.items())},
               "note": "final mask = mean of the backward and forward SAM2 probabilities >= 0.5"},
              open(md + "_conf.json", "w"), indent=1)
    print(f"wrote {len(conf)} masks (+ probabilities, per-pass masks, confidence) to {md}")


if __name__ == "__main__":
    main()
