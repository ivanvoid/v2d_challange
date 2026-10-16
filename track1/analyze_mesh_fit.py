#!/usr/bin/env python3
"""Debug tool: how well does an object mesh fit, and is its orientation right?

For one stage-5 candidate mesh, on each of its check frames (mesh/cand_X/check.json):
  * re-registers the mesh with FoundationPose (independent, like the stage-5 check)
  * measures the fitted pose: IoU, OVER = share of the rendered silhouette outside the mask
    (mesh too big / misplaced), UNDER = share of the mask not covered (mesh too small),
    depth error vs MoGe, colour error (rendered texture vs image, inside the overlap)
  * measures 180-degree flips of that pose about the mesh's own x, y, z axes (through its
    centre) the same way — if a flip fits better, the registration picked the wrong way round
  * reports how many of FoundationPose's top-10 hypotheses are near the chosen rotation (<30 deg)
    vs near a flip — a near tie means the orientation is ambiguous for this object/view
Writes a sheet: per frame, image | chosen pose (textured render) | best flip, with the numbers.

    /opt/venvs/foundation_pose/bin/python track1/analyze_mesh_fit.py --episode 6 --cand 643 \
        --out_root out --weights weights --out logs/stages/05_mesh/review/episode_000006_fit_analysis.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logutil import to_debug  # noqa: E402

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask  # noqa: E402
from v2d.common.datatypes import Image as V2dImage  # noqa: E402
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker, nvdiffrast_render  # noqa: E402
from v2d.mesh.lib.mesh import Mesh  # noqa: E402

FLIPS = {"flip x": np.diag([1, -1, -1]), "flip y": np.diag([-1, 1, -1]), "flip z": np.diag([-1, -1, 1])}


def rot_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--cand", type=int, required=True, help="candidate LOCAL frame (mesh/cand_XXXXXX)")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug_log", default="/tmp/opencode/analyze_mesh_fit.debug.log")
    args = ap.parse_args()

    W = os.path.join(os.path.abspath(args.out_root), f"episode_{args.episode:06d}")
    C = os.path.join(W, "mesh", f"cand_{args.cand:06d}")
    off = json.load(open(os.path.join(W, "track1_meta.json")))["frame_offset"]
    intr = CameraIntrinsics.load(os.path.join(W, "intrinsics_stable.json"))
    K = intr.to_matrix()
    chk = json.load(open(os.path.join(C, "check.json")))
    with to_debug(args.debug_log, "load"):
        tr = FoundationPoseTracker(Mesh.load(os.path.join(C, "mesh_scaled.obj")),
                                   os.path.join(os.path.abspath(args.weights), "foundation_pose"))
    est = tr._est
    tfc = est.get_tf_to_centered_mesh().cpu().numpy()

    def render(Mc_list):
        with torch.no_grad():
            col, dep, _ = nvdiffrast_render(K, intr.height, intr.width,
                                            torch.as_tensor(np.stack(Mc_list), device="cuda", dtype=torch.float),
                                            glctx=tr._glctx, mesh_tensors=est.mesh_tensors, get_normal=False)
        return col.cpu().numpy(), dep.cpu().numpy()

    def measure(rgb, d, m, col, dep):
        sil = dep > 0.001
        inter, union = (sil & m).sum(), (sil | m).sum()
        both = sil & m & (d > 0)
        return {"iou": float(inter / max(union, 1)),
                "over": float((sil & ~m).sum() / max(sil.sum(), 1)),
                "under": float((m & ~sil).sum() / max(m.sum(), 1)),
                "area_ratio": float(sil.sum() / max(m.sum(), 1)),
                "depth_cm": float(np.median(np.abs(dep[both] - d[both])) * 100) if both.any() else None,
                "color_err": float(np.abs(col[both] * 255 - rgb[both]).mean()) if both.any() else None}

    rows, tiles = [], []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    for p in chk["frames"]:
        f = p["frame"]
        rgb = np.asarray(Image.open(os.path.join(W, "frames", f"{f:06d}.png")).convert("RGB"))
        mask = Mask.load(os.path.join(W, "masks", "1", f"{f:06d}.png"))
        m = mask.mask.astype(bool)
        depth = DepthImage.load(os.path.join(W, "depth", f"{f:06d}.png"))
        with to_debug(args.debug_log, f"register {f}"):
            M = tr.register(V2dImage(data=rgb), depth, mask, intr, iteration=10).to_matrix()
        Mc = M @ np.linalg.inv(tfc)                      # pose of the CENTRED mesh (what is rendered)
        cands = {"chosen": Mc} | {k: Mc @ np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
                                  for k, R in FLIPS.items()}
        cols, deps = render(list(cands.values()))
        res = {k: measure(rgb, depth.depth, m, cols[i], deps[i]) for i, k in enumerate(cands)}
        # FoundationPose's own top-10 hypotheses: near the chosen rotation, or near a flip?
        hyp = est.poses[:10].cpu().numpy()
        near_chosen = sum(rot_deg(h, Mc) < 30 for h in hyp)
        near_flip = sum(min(rot_deg(h, Mc @ np.pad(R, ((0, 1), (0, 1)))) for R in FLIPS.values()) < 30 for h in hyp)
        sc = est.scores[:10].cpu().numpy().tolist()
        best_flip = max(FLIPS, key=lambda k: res[k]["iou"] - (res[k]["color_err"] or 99) / 255)
        rows.append({"frame": f, "video_frame": f + off, "own": p["own"], **{k: v for k, v in res.items()},
                     "top10_near_chosen": near_chosen, "top10_near_flip": near_flip, "top10_scores": sc,
                     "best_flip": best_flip})
        r = res["chosen"]
        print(f"video frame {f + off:4d}{' (own)' if p['own'] else '      '}: chosen IoU {r['iou']:.2f} "
              f"over {r['over']:.2f} under {r['under']:.2f} area x{r['area_ratio']:.2f} colour err "
              f"{r['color_err']:.0f} | " + " | ".join(
                  f"{k}: IoU {res[k]['iou']:.2f} col {res[k]['color_err'] or 0:.0f}" for k in FLIPS)
              + f" | top-10 near chosen {near_chosen}, near a flip {near_flip}")

        # sheet tile: image crop | chosen render over image | best flip render over image
        ys, xs = np.nonzero(m)
        pad = int(0.8 * max(np.ptp(xs), np.ptp(ys))) + 20
        y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad
        panels = []
        for name, i in (("image + mask", None), ("chosen", 0), (best_flip, list(cands).index(best_flip))):
            img = rgb.copy()
            if i is not None:
                sil = deps[i] > 0.001
                img[sil] = (cols[i][sil] * 255).astype(np.uint8)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
            crop = Image.fromarray(img[y0:y1, x0:x1]).resize((320, int(320 * (y1 - y0) / (x1 - x0))))
            t = res["chosen" if i == 0 else name] if i is not None else None
            label = name if t is None else f"{name}: IoU {t['iou']:.2f}, over {t['over']:.2f}, col {t['color_err']:.0f}"
            canvas = Image.new("RGB", (330, crop.height + 34), (20, 20, 20))
            canvas.paste(crop, (5, 34))
            ImageDraw.Draw(canvas).text((5, 5), label, fill=(255, 255, 255), font=ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15))
            panels.append(canvas)
        h = max(pn.height for pn in panels)
        tile = Image.new("RGB", (330 * 3 + 20, h + 40), (0, 0, 0))
        ImageDraw.Draw(tile).text((8, 8), f"video frame {f + off}{' (OWN)' if p['own'] else ''}   top-10 FP "
                                   f"hypotheses: {near_chosen} near chosen, {near_flip} near a flip",
                                   fill=(255, 220, 0), font=font)
        for j, pn in enumerate(panels):
            tile.paste(pn, (j * 335, 40))
        tiles.append(tile)

    cols_n = 2
    tw, th = max(t.width for t in tiles), max(t.height for t in tiles)
    sheet = Image.new("RGB", (tw * cols_n, th * ((len(tiles) + 1) // cols_n) + 50), (0, 0, 0))
    ImageDraw.Draw(sheet).text((10, 10), f"episode {args.episode} mesh cand {args.cand + off}: fitted pose vs 180-degree "
                               "flips (rendered with texture; green = SAM2 mask)", fill=(255, 255, 255), font=font)
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols_n) * tw, 50 + (i // cols_n) * th))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sheet.save(args.out)
    json.dump(rows, open(os.path.splitext(args.out)[0] + ".json", "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
