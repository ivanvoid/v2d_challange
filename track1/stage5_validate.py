#!/usr/bin/env python3
"""Stage 5f — validate the FINAL object model of one video (the exact mesh_scaled.obj tracking will use).

After stage 5e chose one mesh per object and wrote it (scaled for this video) to out/<ep>/mesh_scaled.obj,
this fits that exact file on HELD-OUT clear frames: stage-5a eligible frames (stage-4 status 'ok', not
occluded by the body, not at the border) that were not used to build or choose the mesh (not a candidate
frame, not a stage-5c check frame), --n_frames of them spread over the contact span. Per frame, FoundationPose
registers the mesh (10 iterations); of its top hypotheses, up to 5 distinct orientations (>= 45 deg apart) are
rendered and the one with the best IoU is kept:
  iou        rendered silhouette vs the SAM2 mask
  coverage   share of the mask covered by the rendered mesh (a missing part leaves it uncovered)
  depth_cm   median |rendered depth - MoGe depth| where both exist
Accepted if median IoU >= --min_iou and median coverage >= --min_cov. Otherwise exit code 3: the stage is not
stamped and tracking does not run on a model that does not fit — the least-bad mesh is never passed on silently.

Writes out/<ep>/mesh/validate.json and the review sheet --sheet.

    /opt/venvs/foundation_pose/bin/python track1/stage5_validate.py --episode 6 --out_root out --weights weights \
        --debug_log logs/stages/05e_object/episode_000006.debug.log --sheet logs/stages/05e_object/review/episode_000006_validate.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logutil import log, to_debug  # noqa: E402
from viz import contrast_bg  # noqa: E402

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask  # noqa: E402
from v2d.common.datatypes import Image as V2dImage  # noqa: E402
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker, nvdiffrast_render  # noqa: E402
from v2d.mesh.lib.mesh import Mesh  # noqa: E402


def rot_deg(a: np.ndarray, b: np.ndarray) -> float:
    r = a[:3, :3].T @ b[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--debug_log", required=True)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--min_iou", type=float, default=0.5)
    ap.add_argument("--min_cov", type=float, default=0.6)
    ap.add_argument("--thin_tol_px", type=int, default=3,
                    help='objects with "thin": true (hula hoop): outlines compared with this pixel tolerance')
    args = ap.parse_args()
    import cv2
    import paths
    obj = next(json.loads(l) for l in open(paths.META) if json.loads(l)["episode_index"] == args.episode)["object"]
    thin = bool(json.load(open(paths.PRIORS)).get(obj, {}).get("thin"))
    # a hoop rim is ~4 px wide: a 1-2 px offset halves the plain IoU of a perfect fit, so a thin object is judged
    # with both outlines widened by the tolerance (plain IoU is still reported)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.thin_tol_px + 1,) * 2)
    def widen(x):
        return cv2.dilate(x.astype(np.uint8), ker) > 0

    ep = f"episode_{args.episode:06d}"
    W = os.path.join(os.path.abspath(args.out_root), ep)
    off = json.load(open(os.path.join(W, "track1_meta.json")))["frame_offset"]
    intr = CameraIntrinsics.load(os.path.join(W, "intrinsics_stable.json"))
    K = intr.to_matrix()
    cj = json.load(open(os.path.join(W, "mesh", "candidates.json")))
    ch = json.load(open(os.path.join(W, "mesh", "choice.json")))
    used = set(cj["chosen"]) | set(ch.get("check_frames", []))
    pool = sorted(set(cj.get("eligible_frames", [])) - used)
    if len(pool) < 3:                                  # few clear frames: fall back to any non-empty mask frame
        pool = sorted({r["frame"] for r in cj["rows"]} - used)
    frames = [pool[int(round(i))] for i in np.linspace(0, len(pool) - 1, min(args.n_frames, len(pool)))]
    mesh_path = os.path.join(W, "mesh_scaled.obj")
    oc = json.load(open(os.path.join(W, "mesh", "object_choice.json")))

    per = []
    with to_debug(args.debug_log, f"{ep} validate"):
        tr = FoundationPoseTracker(Mesh.load(mesh_path), os.path.join(os.path.abspath(args.weights), "foundation_pose"))
        for f in frames:
            rgb = np.asarray(Image.open(os.path.join(W, "frames", f"{f:06d}.png")).convert("RGB"))
            depth = DepthImage.load(os.path.join(W, "depth", f"{f:06d}.png"))
            mask = Mask.load(os.path.join(W, "masks", "1", f"{f:06d}.png"))
            obs = mask.mask.astype(bool)
            if not obs.any():
                continue
            tr.register(V2dImage(data=rgb), depth, mask, intr, iteration=10)
            distinct = []
            for h in tr._est.poses[:40].cpu().numpy():
                if all(rot_deg(h, g) >= 45 for g in distinct):
                    distinct.append(h)
                if len(distinct) == 5:
                    break
            with torch.no_grad():
                col, dep, _ = nvdiffrast_render(K, intr.height, intr.width,
                                                torch.as_tensor(np.stack(distinct), device="cuda", dtype=torch.float),
                                                glctx=tr._glctx, mesh_tensors=tr._est.mesh_tensors, get_normal=False)
            dep = dep.cpu().numpy()
            ious = [((dep[i] > 0.001) & obs).sum() / max(((dep[i] > 0.001) | obs).sum(), 1) for i in range(len(distinct))]
            b = int(np.argmax(ious))
            sil = dep[b] > 0.001
            both = sil & obs & (depth.depth > 0)
            plain_iou, plain_cov = float(ious[b]), float((sil & obs).sum() / max(obs.sum(), 1))
            if thin:
                sw, ow = widen(sil), widen(obs)
                t_iou = float((sw & ow).sum() / max((sw | ow).sum(), 1))
                t_cov = float((obs & sw).sum() / max(obs.sum(), 1))
            per.append({"frame": f, "video_frame": f + off, "iou": t_iou if thin else plain_iou,
                        "coverage": t_cov if thin else plain_cov, "plain_iou": plain_iou, "plain_coverage": plain_cov,
                        "depth_cm": float(np.median(np.abs(dep[b][both] - depth.depth[both])) * 100) if both.any() else None,
                        "_render": col[b].cpu().numpy(), "_sil": sil, "_obs": obs, "_rgb": rgb})
    if not per:
        raise SystemExit(f"{ep}: no held-out frame to validate on")
    med_iou = float(np.median([p["iou"] for p in per]))
    med_cov = float(np.median([p["coverage"] for p in per]))
    dcs = [p["depth_cm"] for p in per if p["depth_cm"] is not None]
    med_d = float(np.median(dcs)) if dcs else None
    accepted = med_iou >= args.min_iou and med_cov >= args.min_cov

    # sheet: per held-out frame, image with the mask (green) and the rendered mesh (blue) outlines + the render
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    tiles = []
    for p in per:
        img = p["_rgb"].copy()
        for m, colr in ((p["_obs"], (40, 230, 60)), (p["_sil"], (60, 120, 255))):
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, colr, 3)
        ys, xs = np.nonzero(p["_obs"] | p["_sil"])
        pad = int(0.3 * max(np.ptp(xs), np.ptp(ys))) + 20
        y0, y1, x0, x1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad), max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
        crop = Image.fromarray(img[y0:y1, x0:x1]).resize((360, 360))
        rc = np.clip(p["_render"] * (255 if p["_render"].max() <= 1.0 else 1), 0, 255).astype(np.uint8)
        rc[~p["_sil"]] = contrast_bg(rc[p["_sil"]])           # a dark object must not vanish on a dark background
        rend = Image.fromarray(rc[y0:y1, x0:x1]).resize((360, 360))
        t = Image.new("RGB", (720, 410), (20, 20, 20))
        t.paste(crop, (0, 50)); t.paste(rend, (360, 50))
        bad = p["iou"] < args.min_iou or p["coverage"] < args.min_cov
        plain = " (plain %.2f)" % p["plain_iou"] if thin else ""
        dtxt = "n/a" if p["depth_cm"] is None else f"{p['depth_cm']:.1f} cm"
        ImageDraw.Draw(t).text((8, 8), f"video frame {p['video_frame']}: IoU {p['iou']:.2f}{plain}  coverage {p['coverage']:.2f}  "
                                         f"depth {dtxt}", fill=(255, 80, 80) if bad else (230, 230, 230), font=small)
        tiles.append(t)
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (720 * cols, 410 * rows + 80), (0, 0, 0))
    tol_txt = " (thin object, %d px tolerance)" % args.thin_tol_px if thin else ""
    ImageDraw.Draw(sheet).text((10, 8), f"{ep} - FINAL model validation on held-out clear frames - "
                                        f"{'ACCEPTED' if accepted else 'REJECTED'}: median IoU{tol_txt} {med_iou:.2f} "
                                        f"(>= {args.min_iou}), coverage {med_cov:.2f} (>= {args.min_cov}), depth "
                                        f"{'n/a' if med_d is None else f'{med_d:.1f} cm'}",
                               fill=(120, 255, 120) if accepted else (255, 90, 90), font=font)
    ImageDraw.Draw(sheet).text((10, 44), f"mesh from episode {oc['source_episode']} ({oc['source_candidate']}), size "
                                         f"x{oc['rel_scale']:.2f}. Green = SAM2 mask, blue = fitted model; right = "
                                         f"textured render", fill=(200, 200, 200), font=small)
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 720, 80 + (i // cols) * 410))
    os.makedirs(os.path.dirname(os.path.abspath(args.sheet)), exist_ok=True)
    sheet.save(args.sheet)

    # the final acceptance summary of THIS exported mesh (sha1 binds it to the file stage 6 will track)
    import hashlib
    json.dump({"episode": args.episode, "mesh": mesh_path, "thin_tolerance_px": args.thin_tol_px if thin else 0,
               "mesh_sha1": hashlib.sha1(open(mesh_path, "rb").read()).hexdigest(),
               "extent_m": oc.get("extent_m"), "rel_scale": oc.get("rel_scale"), "mesh_source": oc, "accepted": accepted,
               "median_iou": med_iou, "median_coverage": med_cov, "median_depth_cm": med_d,
               "thresholds": {"min_iou": args.min_iou, "min_cov": args.min_cov},
               "frames": [{k: v for k, v in p.items() if not k.startswith("_")} for p in per],
               "sheet": os.path.basename(args.sheet)}, open(os.path.join(W, "mesh", "validate.json"), "w"), indent=1)
    log(f"{ep}: final model on {len(per)} held-out clear frames — median IoU{tol_txt} {med_iou:.2f}, coverage {med_cov:.2f}, "
        f"depth {'n/a' if med_d is None else f'{med_d:.1f} cm'} -> {'ACCEPTED' if accepted else 'REJECTED'}")
    if not accepted:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
