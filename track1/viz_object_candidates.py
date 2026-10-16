#!/usr/bin/env python3
"""Draw every Grounding DINO candidate on the frames it was found in, for human review.

For each candidate frame: the actor (blue), every object detection over all prompt
rungs and thresholds (red = rejected, orange = accepted but not chosen), the chosen
detection (thick green) and, when given, the legacy object mask box (magenta).
Each box is labelled with id, confidence, image gap to the actor (person heights)
and depth difference to the actor (m). Writes one tiled PNG.

Usually called from select_object.py; standalone:

    python track1/viz_object_candidates.py --frames_dir out/episode_000009/track1_select/near_frames \
        --cand_dir out/episode_000009/track1_select/dino_candidates/near \
        --person_dir out/episode_000009/track1_select/dino_candidates/person \
        --depth_dir out/episode_000009/depth --out logs/review/episode_000009_object_candidates.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_object import box_depth, rect_gap  # noqa: E402

BLUE, RED, ORANGE, GREEN, MAGENTA, CYAN = ((60, 140, 255), (235, 50, 50), (255, 165, 0), (40, 220, 60),
                                          (255, 0, 255), (0, 230, 230))


def font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def same_box(a: dict, b: dict, tol: float = 4.0) -> bool:
    return all(abs(a[k] - b[k]) <= tol for k in ("x0", "y0", "x1", "y1"))


def tile(tiles: list[Image.Image], title: str, cols: int = 2) -> Image.Image:
    tw = max(t.width for t in tiles)
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    heights = [max(t.height for t in r) for r in rows]
    sheet = Image.new("RGB", (tw * cols, sum(heights) + 50), (0, 0, 0))
    ImageDraw.Draw(sheet).text((10, 10), title, fill=(255, 255, 255), font=font(28))
    y = 50
    for r, rh in zip(rows, heights):
        for c, t in enumerate(r):
            sheet.paste(t, (c * tw, y))
        y += rh
    return sheet


def render_candidates(frames_dir: str, cand_dir: str, out: str, person_dir: str | None = None,
                      depth_dir: str | None = None, chosen_frame: int | None = None,
                      chosen_box: dict | None = None, legacy_box: dict | None = None, title: str = "",
                      max_gap_rel: float = 0.1, max_depth_gap_m: float = 0.6, scale: float = 0.6,
                      frame_notes: dict | None = None, protected_boxes: dict | None = None) -> str:
    by_frame: dict[int, list[tuple[str, dict]]] = {}
    person: dict[int, dict] = {}
    for path in sorted(glob.glob(os.path.join(person_dir or cand_dir, "person_*.json"))):
        frame = int(re.search(r"(\d{6})\.json$", path).group(1))
        dets = json.load(open(path))
        if dets:
            person[frame] = max(dets, key=lambda d: (d["box"]["x1"] - d["box"]["x0"]) *
                                (d["box"]["y1"] - d["box"]["y0"]))["box"]
    for path in sorted(glob.glob(os.path.join(cand_dir, "r*_t*_*.json"))):
        m = re.fullmatch(r"(r\d+)_t(\d+)_(\d{6})\.json", os.path.basename(path))
        if not m:
            continue
        tag, frame = m.group(1), int(m.group(3))
        for d in json.load(open(path)):
            lst = by_frame.setdefault(frame, [])
            if not any(same_box(d["box"], o["box"]) for _, o in lst):
                lst.append((tag, d))

    tiles = []
    for frame in sorted(set(by_frame) | set(person)):
        fpath = os.path.join(frames_dir, f"{frame:06d}.png")
        if not os.path.exists(fpath):
            continue
        img = Image.open(fpath).convert("RGB")
        draw = ImageDraw.Draw(img)
        f_big, f_small = font(26), font(22)
        p = person.get(frame)
        pz = box_depth(depth_dir, frame, p) if p else None
        if p:
            draw.rectangle([p["x0"], p["y0"], p["x1"], p["y1"]], outline=BLUE, width=4)
            draw.text((p["x0"], p["y0"] - 30), f"actor z={pz:.2f}m" if pz else "actor", fill=BLUE, font=f_big)
        pb = (protected_boxes or {}).get(frame)
        if pb:
            for k in range(0, int(pb["x1"] - pb["x0"]), 24):  # dashed: never-blanked actor region
                draw.line([pb["x0"] + k, pb["y0"], min(pb["x0"] + k + 12, pb["x1"]), pb["y0"]], fill=CYAN, width=3)
                draw.line([pb["x0"] + k, pb["y1"], min(pb["x0"] + k + 12, pb["x1"]), pb["y1"]], fill=CYAN, width=3)
            for k in range(0, int(pb["y1"] - pb["y0"]), 24):
                draw.line([pb["x0"], pb["y0"] + k, pb["x0"], min(pb["y0"] + k + 12, pb["y1"])], fill=CYAN, width=3)
                draw.line([pb["x1"], pb["y0"] + k, pb["x1"], min(pb["y0"] + k + 12, pb["y1"])], fill=CYAN, width=3)
        if legacy_box is not None and frame == chosen_frame:
            b = legacy_box
            draw.rectangle([b["x0"], b["y0"], b["x1"], b["y1"]], outline=MAGENTA, width=5)
            draw.text((b["x0"] + 4, b["y1"] - 30), "old mask", fill=MAGENTA, font=f_big)
        legend = []
        for i, (tag, d) in enumerate(by_frame.get(frame, [])):
            b = d["box"]
            gap = rect_gap(b, p) / max(p["y1"] - p["y0"], 1) if p else None
            oz = box_depth(depth_dir, frame, b)
            dz = (oz - pz) if (oz is not None and pz is not None) else None
            ok = (gap is None or gap <= max_gap_rel) and (dz is None or abs(dz) <= max_depth_gap_m)
            is_chosen = chosen_box is not None and frame == chosen_frame and same_box(b, chosen_box)
            color = GREEN if is_chosen else (ORANGE if ok else RED)
            draw.rectangle([b["x0"], b["y0"], b["x1"], b["y1"]], outline=color, width=8 if is_chosen else 3)
            draw.text((b["x0"] + 4, b["y0"] + 2), f"#{i}", fill=color, font=f_big)
            legend.append((color, f"#{i} {tag} '{d.get('label', '')}' conf={d.get('confidence', 0):.2f} "
                                  f"gap={'-' if gap is None else f'{gap:.2f}'} "
                                  f"dz={'-' if dz is None else f'{dz:+.2f}m'}"
                                  f"{'  <== CHOSEN' if is_chosen else ('' if ok else '  rejected')}"))
        note = (frame_notes or {}).get(frame)
        if note:
            legend.insert(0, (CYAN, note))
        w, h = img.size
        band = 34 * (len(legend) + 1) + 10
        canvas = Image.new("RGB", (w, h + band), (20, 20, 20))
        canvas.paste(img, (0, 0))
        cd = ImageDraw.Draw(canvas)
        cd.text((10, h + 6), f"frame {frame}", fill=(255, 255, 255), font=f_big)
        for j, (color, text) in enumerate(legend):
            cd.text((10, h + 40 + 34 * j), text, fill=color, font=f_small)
        tiles.append(canvas.resize((int(w * scale), int((h + band) * scale))))

    if not tiles:
        raise SystemExit(f"no candidates in {cand_dir}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tile(tiles, title).save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--cand_dir", required=True)
    ap.add_argument("--person_dir", default=None)
    ap.add_argument("--depth_dir", default=None)
    ap.add_argument("--chosen_frame", type=int, default=None)
    ap.add_argument("--chosen_box", default=None, help="x0,y0,x1,y1")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    chosen = None
    if args.chosen_box:
        x0, y0, x1, y1 = (float(v) for v in args.chosen_box.split(","))
        chosen = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
    print("wrote", render_candidates(args.frames_dir, args.cand_dir, args.out, args.person_dir,
                                     args.depth_dir, args.chosen_frame, chosen, title=args.title))


if __name__ == "__main__":
    main()
