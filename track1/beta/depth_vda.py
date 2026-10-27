#!/usr/bin/env python3
"""Beta stage 1 depth: Video Depth Anything (metric, temporally consistent) instead of per-frame MoGe.

Reads the undistorted frames (out/<ep>/frames/, the same images MoGe saw) and writes depth_raw/<frame>.png
in the pipeline's encoding (uint16 inverse depth, px = 65535 / (depth_m + 1), 0 = no depth), so
stage1_prep.py then runs its usual static-background stabilisation, floor fit, etc. on it unchanged.
MoGe's intrinsics (intrinsics/, intrinsics_stable.json) are kept: VDA does not estimate a camera.

VDA runs on the video at --max_res (long side), its own --input_size inside; the depth is resized back to
the frame size. Writes depth_source.json (model, checkpoint, sizes, per-frame median depth). Skipped when
depth_raw/ already holds this model's output for every frame.

    /opt/venvs/vda/bin/python track1/beta/depth_vda.py --episode 6 --out_root out_beta/<tag> --weights weights
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
from logutil import log, to_debug  # noqa: E402

ENCODERS = {"vda": "vitl", "vda-small": "vits"}
CONFIGS = {"vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
           "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", choices=list(ENCODERS), default="vda")
    ap.add_argument("--repo", default=str(paths.DATA / "models" / "Video-Depth-Anything"))
    ap.add_argument("--input_size", type=int, default=518)
    ap.add_argument("--max_res", type=int, default=1280)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--debug_log", default=None)
    a = ap.parse_args()

    ep = f"episode_{a.episode:06d}"
    W = Path(a.out_root).resolve() / ep
    frames_dir, raw = W / "frames", W / "depth_raw"
    names = sorted(p.name for p in frames_dir.glob("*.png"))
    if not names:
        raise SystemExit(f"{ep}: no frames in {frames_dir}")
    enc = ENCODERS[a.model]
    ckpt = Path(a.weights) / "video_depth_anything" / "checkpoints" / f"metric_video_depth_anything_{enc}.pth"
    src_path = W / "depth_source.json"
    want = {"model": a.model, "checkpoint": str(ckpt), "input_size": a.input_size, "max_res": a.max_res}
    if src_path.exists() and raw.is_dir() and not raw.is_symlink() \
            and len(list(raw.glob("*.png"))) == len(names) \
            and {k: json.load(open(src_path)).get(k) for k in want} == want:
        log(f"  [skip] {a.model} depth ({len(names)} frames, already complete)")
        return

    log(f"  [run ] {a.model} metric depth ({enc}) on {len(names)} frames")
    t0 = time.time()
    H, Wd = cv2.imread(str(frames_dir / names[0])).shape[:2]
    s = min(1.0, a.max_res / max(H, Wd))
    h, w = (int(round(H * s)) // 2 * 2, int(round(Wd * s)) // 2 * 2) if s < 1 else (H, Wd)
    video = np.stack([cv2.cvtColor(cv2.resize(cv2.imread(str(frames_dir / n)), (w, h), interpolation=cv2.INTER_AREA)
                                   if s < 1 else cv2.imread(str(frames_dir / n)), cv2.COLOR_BGR2RGB) for n in names])

    with to_debug(a.debug_log, f"{ep} {a.model} depth"):
        import torch
        sys.path.insert(0, a.repo)
        from video_depth_anything.video_depth import VideoDepthAnything
        model = VideoDepthAnything(**CONFIGS[enc], metric=True)
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"), strict=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(dev).eval()
        depth, _ = model.infer_video_depth(video, 30, input_size=a.input_size, device=dev, fp32=a.fp32)
        del model
    if len(depth) != len(names):
        raise SystemExit(f"{ep}: VDA returned {len(depth)} maps for {len(names)} frames")

    tmp = W / "depth_raw.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    med = []
    for n, d in zip(names, depth):
        d = cv2.resize(d.astype(np.float32), (Wd, H), interpolation=cv2.INTER_LINEAR) if d.shape != (H, Wd) else d
        ok = np.isfinite(d) & (d > 1e-3)
        px = np.where(ok, np.clip(np.rint(65535.0 / (np.where(ok, d, 0) + 1.0)), 1, 65535), 0).astype(np.uint16)
        cv2.imwrite(str(tmp / n), px)
        med.append(round(float(np.median(d[ok])), 4) if ok.any() else None)
    if raw.is_symlink():                     # a seeded link to production depth: drop the link, never its target
        raw.unlink()
    elif raw.exists():
        shutil.rmtree(raw)
    os.replace(tmp, raw)
    json.dump({**want, "frame_size": [Wd, H], "processed_size": [w, h], "seconds": round(time.time() - t0, 1),
               "median_depth_m": med}, open(src_path, "w"), indent=1)
    m = np.array([x for x in med if x is not None])
    log(f"  [done] {a.model} depth in {time.time() - t0:.0f}s — median depth {np.median(m):.2f} m "
        f"(frames {m.min():.2f}..{m.max():.2f})")


if __name__ == "__main__":
    main()
