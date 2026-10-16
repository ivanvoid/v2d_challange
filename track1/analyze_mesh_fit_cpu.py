#!/usr/bin/env python3
"""CPU version of analyze_mesh_fit.py: uses the poses stage 5 already saved (check.json).

For each check frame: render the textured mesh (painter's algorithm, per-face colour and depth)
at the saved pose and at 180-degree flips about the mesh's own x / y / z axes through its centre;
measure IoU, OVER (rendered outside the mask = too big / misplaced), UNDER (mask not covered),
area ratio, depth error vs MoGe and colour error vs the image. Writes a sheet + JSON.

    /opt/venvs/sam3d_body/bin/python track1/analyze_mesh_fit_cpu.py --episode 6 --cand 643 --out_root out \
        --out logs/stages/05_mesh/review/episode_000006_fit_analysis.png
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

FLIPS = {"flip x": np.diag([1., -1, -1]), "flip y": np.diag([-1., 1, -1]), "flip z": np.diag([-1., -1, 1])}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_depth(p):
    px = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return 1.0 / np.maximum(px / 65535.0, 1e-6) - 1.0


def render(v, f, fcol, M, K, hw):
    pc = v @ M[:3, :3].T + M[:3, 3]
    uvw = pc @ K.T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
    z = pc[f][:, :, 2].mean(1)
    col = np.zeros((*hw, 3), np.uint8)
    dep = np.zeros(hw, np.float32)
    pts = np.round(uv[f]).astype(np.int32)
    for i in np.argsort(-z):                      # far first; nearer faces overwrite
        cv2.fillConvexPoly(col, pts[i], tuple(int(c) for c in fcol[i]))
        cv2.fillConvexPoly(dep, pts[i], float(z[i]))
    return col, dep


def measure(rgb, d, m, col, dep):
    sil = dep > 0
    both = sil & m
    return {"iou": float((sil & m).sum() / max((sil | m).sum(), 1)),
            "over": float((sil & ~m).sum() / max(sil.sum(), 1)),
            "under": float((m & ~sil).sum() / max(m.sum(), 1)),
            "area_ratio": float(sil.sum() / max(m.sum(), 1)),
            "depth_cm": float(np.median(np.abs(dep[both] - d[both])) * 100) if both.any() else None,
            "color_err": float(np.abs(col[both].astype(float) - rgb[both]).mean()) if both.any() else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--cand", type=int, required=True, help="candidate LOCAL frame")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    W = os.path.join(args.out_root, f"episode_{args.episode:06d}")
    C = os.path.join(W, "mesh", f"cand_{args.cand:06d}")
    off = json.load(open(os.path.join(W, "track1_meta.json")))["frame_offset"]
    it = json.load(open(os.path.join(W, "intrinsics_stable.json")))
    K = np.array([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]], [0, 0, 1]])
    hw = (it["height"], it["width"])
    tm = trimesh.load(os.path.join(C, "mesh_scaled.obj"), force="mesh", process=False)
    v, f = np.asarray(tm.vertices), np.asarray(tm.faces)
    vis = tm.visual.to_color() if type(tm.visual).__name__ == "TextureVisuals" else tm.visual
    fcol = np.asarray(vis.vertex_colors)[:, :3].astype(float)[f].mean(1)
    centre = (v.min(0) + v.max(0)) / 2
    T = np.eye(4); T[:3, 3] = centre
    Ti = np.eye(4); Ti[:3, 3] = -centre

    rows, tiles = [], []
    for p in json.load(open(os.path.join(C, "check.json")))["frames"]:
        fr = p["frame"]
        M = np.asarray(p["pose"])
        rgb = cv2.cvtColor(cv2.imread(os.path.join(W, "frames", f"{fr:06d}.png")), cv2.COLOR_BGR2RGB)
        m = cv2.imread(os.path.join(W, "masks", "1", f"{fr:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
        d = load_depth(os.path.join(W, "depth", f"{fr:06d}.png"))
        poses = {"chosen": M} | {k: M @ T @ np.pad(R, ((0, 1), (0, 1))) + 0 for k, R in FLIPS.items()}
        for k, R in FLIPS.items():                 # flip about the mesh centre, not its origin
            R4 = np.eye(4); R4[:3, :3] = R
            poses[k] = M @ T @ R4 @ Ti
        rend = {k: render(v, f, fcol, P, K, hw) for k, P in poses.items()}
        res = {k: measure(rgb, d, m, *rend[k]) for k in poses}
        best_flip = max(FLIPS, key=lambda k: res[k]["iou"] - (res[k]["color_err"] or 99) / 255)
        rows.append({"frame": fr, "video_frame": fr + off, "own": p["own"], **res, "best_flip": best_flip})
        r = res["chosen"]
        print(f"video frame {fr + off:4d}{' (own)' if p['own'] else '      '}: chosen IoU {r['iou']:.2f} over "
              f"{r['over']:.2f} under {r['under']:.2f} area x{r['area_ratio']:.2f} depth {r['depth_cm']:.1f}cm "
              f"colour {r['color_err']:.0f} || " + " | ".join(
                  f"{k}: IoU {res[k]['iou']:.2f} col {res[k]['color_err'] or 0:.0f}" for k in FLIPS))

        ys, xs = np.nonzero(m)
        pad = int(0.8 * max(np.ptp(xs), np.ptp(ys))) + 20
        y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad
        panels = []
        for name in ("image", "chosen", best_flip):
            img = rgb.copy()
            if name != "image":
                c, dp = rend[name]
                img[dp > 0] = c[dp > 0]
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
            crop = Image.fromarray(img[y0:y1, x0:x1])
            crop = crop.resize((320, max(1, int(320 * crop.height / crop.width))))
            t = res.get(name)
            label = "image + SAM2 mask" if t is None else \
                f"{name}: IoU {t['iou']:.2f} over {t['over']:.2f} col {t['color_err']:.0f}"
            cv_ = Image.new("RGB", (330, crop.height + 30), (20, 20, 20))
            cv_.paste(crop, (5, 30))
            ImageDraw.Draw(cv_).text((5, 6), label, fill=(255, 255, 255), font=ImageFont.truetype(FONT, 15))
            panels.append(cv_)
        h = max(pn.height for pn in panels)
        tile = Image.new("RGB", (1010, h + 38), (0, 0, 0))
        ImageDraw.Draw(tile).text((8, 6), f"video frame {fr + off}{' (OWN frame)' if p['own'] else ''}",
                                  fill=(255, 220, 0), font=ImageFont.truetype(FONT, 22))
        for j, pn in enumerate(panels):
            tile.paste(pn, (j * 335, 38))
        tiles.append(tile)
    tw, th = max(t.width for t in tiles), max(t.height for t in tiles)
    sheet = Image.new("RGB", (tw * 2, th * ((len(tiles) + 1) // 2) + 50), (0, 0, 0))
    ImageDraw.Draw(sheet).text((10, 10), f"episode {args.episode}, mesh from video frame {args.cand + off}: "
                               "fitted pose vs best 180-degree flip (textured render, green = SAM2 mask)",
                               fill=(255, 255, 255), font=ImageFont.truetype(FONT, 24))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 2) * tw, 50 + (i // 2) * th))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sheet.save(args.out)
    json.dump(rows, open(os.path.splitext(args.out)[0] + ".json", "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
