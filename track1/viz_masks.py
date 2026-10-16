#!/usr/bin/env python3
"""Object-mask review sheet: the SAM2 object mask over a few frames of the scored span.

Green tint + outline = mask, with its pixel area. Frames with an empty mask are
labelled "NO MASK" in red, so tracking losses are visible at a glance.

    python track1/viz_masks.py --frames_dir out/episode_000014/frames \
        --masks_dir out/episode_000014/masks/1 --first 90 --last 387 \
        --title "episode 14 object mask" --out logs/review/episode_000014_masks.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_object_candidates import font, tile  # noqa: E402


def sample_frames(first: int, last: int, n: int) -> list[int]:
    return sorted({int(round(first + (last - first) * i / max(n - 1, 1))) for i in range(n)})


def render_masks(frames_dir: str, masks_dir: str, frames: list[int], out: str, title: str = "",
                 scale: float = 0.5) -> str:
    tiles = []
    for f in frames:
        fp = os.path.join(frames_dir, f"{f:06d}.png")
        if not os.path.exists(fp):
            continue
        img = Image.open(fp).convert("RGB")
        mp = os.path.join(masks_dir, f"{f:06d}.png")
        m = (np.asarray(Image.open(mp).convert("L")) > 0) if os.path.exists(mp) else None
        arr = np.asarray(img).copy()
        if m is not None and m.any():
            arr[m] = (0.45 * arr[m] + 0.55 * np.array([40, 220, 60])).astype(np.uint8)
            edge = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES)
            arr[np.asarray(edge.filter(ImageFilter.MaxFilter(5))) > 0] = (40, 255, 60)
            label, color = f"frame {f}  mask {int(m.sum())} px", (255, 255, 255)
        else:
            label, color = f"frame {f}  NO MASK", (255, 60, 60)
        img = Image.fromarray(arr)
        ImageDraw.Draw(img).text((14, 10), label, fill=color, font=font(34))
        tiles.append(img.resize((int(img.width * scale), int(img.height * scale))))
    if not tiles:
        raise SystemExit(f"no frames for {frames} in {frames_dir}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tile(tiles, title, cols=3).save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--first", type=int, required=True, help="LOCAL frame index")
    ap.add_argument("--last", type=int, required=True, help="LOCAL frame index")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print("wrote", render_masks(args.frames_dir, args.masks_dir, sample_frames(args.first, args.last, args.n),
                                args.out, args.title))


if __name__ == "__main__":
    main()
