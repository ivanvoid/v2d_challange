#!/usr/bin/env python3
"""Stage 3 — pick the manipulated object for one episode, and record why.

Inputs (from stages 1 and 2, all in out/episode_XXXXXX/):
  frames/, depth/, intrinsics_stable.json, track1_meta.json
  body/bbox_track.pt     the actor's box on every frame (detectron2 track)
  body/mhr_params.pt     SAM3D-Body 2D wrists (keypoints 41 = right, 62 = left)

Method, on 5 frames spread over the scored span:
  1. V-wedge cover: seen from above, two vertical planes meet at an apex --v_back_m behind the
     actor and open toward the camera (half-width --v_reach_m at the actor's depth). Pixels
     outside the V (behind the actor, or off to the sides) are greyed -> images used for
     detection ONLY; topdown_XXXXXX.png shows the V over the depth points
  2. Grounding DINO over a prompt ladder: metadata prompt, shorter phrases, aliases from
     track1/object_priors.json; one model load for all frames and prompts
  3. every detection is measured and gated:
       gap      image gap to the actor box, in actor heights           reject if > --max_gap
       dz       object depth - actor depth (MoGe, metres)              reject if |dz| > --max_dz
       size     largest box side in metres (box px * depth / fx)       reject if outside the prior
       person   the box is the actor himself (IoU with actor box)       reject if > 0.6
     and scored  conf * exp(-wrist_gap / 0.25)   (wrist_gap = nearest wrist to the box, actor heights)
  4. the first prompt / threshold with accepted detections on >= 2 frames wins; best score is the seed
  5. if nothing passes on the near-only images, repeat on the full images (same gates)
  A manual seed in seeds/episode_XXXXXX.json  {"video_frame": 300, "box": [x0, y0, x1, y1]}
  (undistorted image pixels, as on the review sheets) overrides everything.

Writes out/episode_XXXXXX/select/:
  candidates.json    every detection with its measurements and verdict
  object_track.json  the decision (seed frame, box, prompt, mode, ...)
  near_frames/       the depth-covered images
  dino/              raw Grounding DINO outputs (cache)
and out/episode_XXXXXX/dino_detections.json + object_track.json for the next stage.

Run with the sam3d_body venv (Grounding DINO runs in its own venv as a subprocess):

    /opt/venvs/sam3d_body/bin/python track1/stage3_select.py --episode 6 --out_root out --weights weights
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_object import (box_iou, cache_name, candidate_frames, gdino_batch, load_depth,  # noqa: E402
                           load_dets, prompt_ladder, rect_gap)

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import paths  # noqa: E402
PRIORS, SEEDS, META = str(paths.PRIORS), str(paths.SEEDS), str(paths.META)
THRESHOLDS = (0.35, 0.25, 0.2)
R_WRIST, L_WRIST = 41, 62


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def box_dict(b) -> dict:
    return {"x0": float(b[0]), "y0": float(b[1]), "x1": float(b[2]), "y1": float(b[3])}


def region_depth(d: np.ndarray, box: dict, pct: float, min_valid: int = 20) -> float | None:
    """Percentile of the VALID depth inside the box (NaN = no depth or masked out); None if too few."""
    h, w = d.shape
    x0, x1 = int(max(0, box["x0"])), int(min(w, box["x1"]))
    y0, y1 = int(max(0, box["y0"])), int(min(h, box["y1"]))
    if x1 <= x0 or y1 <= y0:
        return None
    v = d[y0:y1, x0:x1]
    v = v[np.isfinite(v)]
    return float(np.percentile(v, pct)) if v.size >= min_valid else None


def valid_depth(depth_dir: str, f: int) -> np.ndarray:
    """Depth in metres with pixels that have no depth (encoded 0 -> ~1e6 m) set to NaN."""
    d = load_depth(depth_dir, f).astype(np.float32)
    d[~(d < 100.0)] = np.nan
    return d


def point_box_gap(p, box: dict) -> float:
    dx = max(box["x0"] - p[0], p[0] - box["x1"], 0.0)
    dy = max(box["y0"] - p[1], p[1] - box["y1"], 0.0)
    return float(math.hypot(dx, dy))


def topdown(xs: np.ndarray, d: np.ndarray, far: np.ndarray, xp: float, pz: float, apex_z: float,
            half_tan: float, out: str, size: int = 420, step: int = 6) -> None:
    """Bird's-eye view of the frame's depth points: kept (green) vs greyed (grey), the V, the actor."""
    from PIL import ImageDraw
    x, z, f = xs[::step, ::step].ravel(), d[::step, ::step].ravel(), far[::step, ::step].ravel()
    xr, zr = (xp - 3.5, xp + 3.5), (0.0, apex_z + 3.0)
    to = lambda X, Z: (int((X - xr[0]) / (xr[1] - xr[0]) * (size - 1)),
                       int((1 - (Z - zr[0]) / (zr[1] - zr[0])) * (size - 1)))
    img = Image.new("RGB", (size, size), (15, 15, 15))
    dr = ImageDraw.Draw(img)
    ok = (x > xr[0]) & (x < xr[1]) & (z > zr[0]) & (z < zr[1])
    for X, Z, F in zip(x[ok], z[ok], f[ok]):
        dr.point(to(X, Z), fill=(90, 90, 90) if F else (60, 220, 90))
    for sgn in (-1, 1):  # the two planes, from the apex toward the camera (z = 0)
        dr.line([to(xp, apex_z), to(xp + sgn * apex_z * half_tan, 0.0)], fill=(255, 60, 60), width=2)
    ax, az = to(xp, pz)
    dr.ellipse([ax - 6, az - 6, ax + 6, az + 6], fill=(255, 220, 0))
    cx_, cz_ = to(xp, 0.05)
    dr.polygon([(cx_, cz_ - 10), (cx_ - 8, cz_ + 4), (cx_ + 8, cz_ + 4)], fill=(255, 255, 255))
    dr.text((6, 4), "top view: green kept, grey hidden", fill=(230, 230, 230))
    dr.text((6, 18), "yellow = actor, red = V, white = camera", fill=(230, 230, 230))
    img.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n_frames", type=int, default=9, help="frames sampled across the contact span")
    ap.add_argument("--xvideo", type=float, default=1.6,
                    help="an option must be within this factor of the object's size in at least one of its other videos")
    ap.add_argument("--consistency", type=float, default=1.5,
                    help="a supporting box must be within this factor of the object's typical size")
    ap.add_argument("--v_back_m", type=float, default=0.7,
                    help="V apex distance behind the actor = this + the object's max size (prior), metres")
    ap.add_argument("--v_reach_m", type=float, default=0.9,
                    help="V half-width at the actor's depth = arm reach (this) + the object's max size, metres")
    ap.add_argument("--max_gap", type=float, default=0.1, help="max image gap to the actor, actor heights")
    ap.add_argument("--max_dz", type=float, default=0.6, help="max |object - actor| depth, metres")
    ap.add_argument("--size_slack", type=float, default=1.6, help="size prior [lo/slack, hi*slack]")
    args = ap.parse_args()

    ep = f"episode_{args.episode:06d}"
    W = os.path.join(os.path.abspath(args.out_root), ep)
    S = os.path.join(W, "select")
    os.makedirs(S, exist_ok=True)
    meta = json.load(open(os.path.join(W, "track1_meta.json")))
    off, first, last = meta["frame_offset"], meta["scored_first"], meta["scored_last"]
    intr = json.load(open(os.path.join(W, "intrinsics_stable.json")))
    fx = intr["fx"]
    md = next(json.loads(l) for l in open(META) if json.loads(l)["episode_index"] == args.episode)
    priors = json.load(open(PRIORS))
    prior = priors.get(md["object"], {})
    lo_m, hi_m = prior.get("size_m", [0.05, 3.0])
    size_lo, size_hi = lo_m / args.size_slack, hi_m * args.size_slack
    ladder = prompt_ladder(md["object_prompt"], md["object"])
    if prior.get("prompt") and prior["prompt"].lower() not in {p.lower() for p in ladder}:
        ladder.insert(0, prior["prompt"])               # cleaned prompt (metadata has typos)
    for a in prior.get("aliases", []):
        if a.lower() not in {p.lower() for p in ladder}:
            ladder.append(a)
    log(f"{ep}: object '{md['object']}', prompt {md['object_prompt']!r}, size prior {lo_m}-{hi_m} m "
        f"(accept {size_lo:.2f}-{size_hi:.2f} m)")
    log(f"  prompt ladder: {ladder}")

    boxes = np.asarray(torch.load(os.path.join(W, "body", "bbox_track.pt"), map_location="cpu",
                                  weights_only=False)["bbox_track"], float)
    kp = torch.load(os.path.join(W, "body", "mhr_params.pt"), map_location="cpu",
                    weights_only=False)["pred_keypoints_2d"].numpy()

    frames = candidate_frames(first - off, last - off, args.n_frames)
    pt_path = os.path.join(W, "body", "person_track.json")
    labels = [r["label"] for r in json.load(open(pt_path))["per_frame"]] if os.path.exists(pt_path) else None
    if labels:                                   # snap each sample to the nearest directly detected actor frame
        snapped = []
        for f in frames:
            near = [g for g in range(max(0, f - 10), min(len(labels), f + 11)) if labels[g] == "detected"]
            snapped.append(min(near, key=lambda g: abs(g - f)) if near else f)
        frames = sorted(set(snapped))
    person_mask = {}
    def actor_mask(f: int):
        if f not in person_mask:
            mp = os.path.join(W, "body", "person_masks", f"{f:06d}.png")
            person_mask[f] = np.asarray(Image.open(mp)) > 0 if os.path.exists(mp) else None
        return person_mask[f]
    frames_dir, depth_dir = os.path.join(W, "frames"), os.path.join(W, "depth")
    # V = arm reach + object size: an object held at arm's length (hoop to the side, roller pushed along,
    # desk or cart pushed ahead) must stay fully inside, otherwise the detector sees a cut object.
    v_back = args.v_back_m + hi_m
    v_reach = args.v_reach_m + hi_m
    cx, half_tan = intr["cx"], v_reach / v_back            # tan of the V half-angle
    log(f"  V wedge: apex {v_back:.2f} m behind the actor, half-width {v_reach:.2f} m at the actor's depth")
    actor, near_info = {}, {}
    near_dir = os.path.join(S, "near_frames")
    os.makedirs(near_dir, exist_ok=True)
    for f in frames:
        pb = box_dict(boxes[f])
        d = valid_depth(depth_dir, f)
        ph = pb["y1"] - pb["y0"]
        core = {"x0": pb["x0"] + 0.25 * (pb["x1"] - pb["x0"]), "x1": pb["x1"] - 0.25 * (pb["x1"] - pb["x0"]),
                "y0": pb["y0"] + 0.15 * ph, "y1": pb["y1"] - 0.15 * ph}
        am = actor_mask(f)
        pz = float(np.nanmedian(d[am])) if am is not None and np.isfinite(d[am]).sum() >= 50 else region_depth(d, core, 50)
        actor[f] = {"box": pb, "height_px": ph, "depth_m": pz,
                    "wrist_r": kp[f, R_WRIST].tolist(), "wrist_l": kp[f, L_WRIST].tolist()}
        # V wedge seen from above: two vertical planes meeting at the apex, v_back_m behind the actor,
        # opening toward the camera (half-width v_reach_m at the actor's depth). Keep inside, grey outside.
        h, w = d.shape
        xs = (np.arange(w)[None, :] - cx) * d / fx                       # lateral metres, per pixel
        xp = ((pb["x0"] + pb["x1"]) / 2 - cx) * pz / fx                  # actor's lateral position
        apex_z = pz + v_back
        far = ~(np.abs(xs - xp) <= (apex_z - d) * half_tan)              # outside the V, behind the apex, or no depth
        # Thin objects (a hula hoop rim) take the depth of the wall behind them, so the per-pixel depth test would
        # grey the object the actor is holding: for them, never grey the zone the actor can reach (the actor's box
        # widened by arm reach + object size, in pixels at the actor's depth, above the head too).
        if prior.get("thin"):                    # object_priors.json "thin": true (hula hoop)
            rp = (args.v_reach_m + hi_m) * fx / pz
            zy0, zy1 = int(max(0, pb["y0"] - rp)), int(min(h, pb["y1"] + 0.25 * rp))
            zx0, zx1 = int(max(0, pb["x0"] - rp)), int(min(w, pb["x1"] + rp))
            far[zy0:zy1, zx0:zx1] = False
        # How much of the actor's own body (core of its box, at actor depth) the V would hide.
        y0, y1, x0, x1 = int(core["y0"]), int(core["y1"]), int(core["x0"]), int(core["x1"])
        body = np.abs(np.nan_to_num(d[y0:y1, x0:x1], nan=1e6) - pz) < 0.4
        actor_hidden = float(far[y0:y1, x0:x1][body].mean()) if body.any() else 0.0
        near_info[f] = {"greyed_fraction": float(far.mean()), "actor_hidden_fraction": actor_hidden,
                        "apex": [xp, apex_z], "actor_xz": [xp, pz], "half_angle_deg": math.degrees(math.atan(half_tan))}
        dst = os.path.join(near_dir, f"{f:06d}.png")
        img = np.asarray(Image.open(os.path.join(frames_dir, f"{f:06d}.png")).convert("RGB")).copy()
        img[far] = 128
        Image.fromarray(img).save(dst)
        topdown(xs, d, far, xp, pz, apex_z, half_tan, os.path.join(S, f"topdown_{f:06d}.png"))
        log(f"  frame {f + off}: actor at {pz:.2f} m, {100 * near_info[f]['greyed_fraction']:.0f}% greyed by the V "
            f"(apex {apex_z:.2f} m), actor pixels hidden {100 * actor_hidden:.1f}%")

    small = hi_m <= 0.6
    gd_weights = os.path.join(os.path.abspath(args.weights), "grounding_dino")

    def measure_on(b: dict, f: int, d: np.ndarray) -> dict:
        a = actor[f]
        oz = min((v for v in (region_depth(d, b, 20),
                              region_depth(d, {"x0": b["x0"] + 0.25 * (b["x1"] - b["x0"]),
                                               "x1": b["x1"] - 0.25 * (b["x1"] - b["x0"]),
                                               "y0": b["y0"] + 0.25 * (b["y1"] - b["y0"]),
                                               "y1": b["y1"] - 0.25 * (b["y1"] - b["y0"])}, 50))
                  if v is not None), default=None)
        if oz is None:
            return {"depth_m": None, "dz_m": None, "size_m": None, "gap": None, "wrist_gap": None, "actor_iou": None,
                    "reasons": ["no valid depth in the box"]}
        size = max(b["x1"] - b["x0"], b["y1"] - b["y0"]) * oz / fx
        gap = rect_gap(b, a["box"]) / a["height_px"]
        wgap = min(point_box_gap(a["wrist_r"], b), point_box_gap(a["wrist_l"], b)) / a["height_px"]
        dz = oz - a["depth_m"]
        iou_p = box_iou(b, a["box"])
        reasons = []
        if gap > args.max_gap:
            reasons.append(f"far from actor ({gap:.2f} heights)")
        if abs(dz) > args.max_dz:
            reasons.append(f"depth {dz:+.2f} m from actor")
        if not size_lo <= size <= size_hi:
            reasons.append(f"size {size:.2f} m not in {size_lo:.2f}-{size_hi:.2f}")
        if iou_p > 0.6:
            reasons.append(f"is the actor (IoU {iou_p:.2f})")
        return {"depth_m": oz, "dz_m": dz, "size_m": size, "gap": gap, "wrist_gap": wgap, "actor_iou": iou_p,
                "reasons": reasons}

    def measure(b: dict, f: int, d: np.ndarray) -> dict:
        """Gates for one box. The object's depth is measured on valid depth OUTSIDE the actor's person mask
        (hands and arms holding the object would pull its depth and size). A hollow object around the actor
        (hula hoop) shows mostly background through it once the actor is removed: if that measurement fails
        only on depth while the one including the actor's pixels passes, the latter is used and labelled."""
        am = actor_mask(f)
        if am is None:
            return {**measure_on(b, f, d), "depth_source": "all pixels (no person mask)"}
        dm = d.copy(); dm[am] = np.nan
        m = measure_on(b, f, dm)
        if not m["reasons"]:
            return {**m, "depth_source": "object pixels (actor excluded)"}
        if m["dz_m"] is None or all(r.startswith("depth") for r in m["reasons"]):
            m_all = measure_on(b, f, d)
            if not m_all["reasons"]:
                return {**m_all, "depth_source": "with actor pixels (object overlaps the actor, e.g. a hoop around it)"}
        return {**m, "depth_source": "object pixels (actor excluded)"}

    def views(images_dir: str, tag: str, f: int) -> list[dict]:
        """Test-time augmentation: the image, its mirror, and (small objects) a zoom on the hands.
        Each view: image path + a function mapping a box in the view back to frame pixels."""
        src = os.path.join(images_dir, f"{f:06d}.png")
        vdir = os.path.join(S, "views", tag)
        os.makedirs(vdir, exist_ok=True)
        img = Image.open(src).convert("RGB")
        W_, H_ = img.size
        out = [{"name": "image", "path": src, "back": lambda b: b}]
        fp = os.path.join(vdir, f"flip_{f:06d}.png")
        if not os.path.exists(fp):
            img.transpose(Image.FLIP_LEFT_RIGHT).save(fp)
        out.append({"name": "flip", "path": fp,
                    "back": lambda b: {"x0": W_ - b["x1"], "x1": W_ - b["x0"], "y0": b["y0"], "y1": b["y1"]}})
        if small:
            a = actor[f]
            wr, wl = np.array(a["wrist_r"]), np.array(a["wrist_l"])
            c = (wr + wl) / 2 if np.linalg.norm(wr - wl) < 0.5 * a["height_px"] else None
            centres = [c] if c is not None else [wr, wl]
            for i, cc in enumerate(centres):
                half = 0.32 * a["height_px"]
                x0, y0 = int(max(0, cc[0] - half)), int(max(0, cc[1] - half))
                x1, y1 = int(min(W_, cc[0] + half)), int(min(H_, cc[1] + half))
                if x1 - x0 < 20 or y1 - y0 < 20:
                    continue
                zs = 800.0 / max(x1 - x0, y1 - y0)
                hp = os.path.join(vdir, f"hand{i}_{f:06d}.png")
                if not os.path.exists(hp):
                    img.crop((x0, y0, x1, y1)).resize((int((x1 - x0) * zs), int((y1 - y0) * zs)),
                                                      Image.BICUBIC).save(hp)
                out.append({"name": f"hand{i}", "path": hp,
                            "back": (lambda b, x0=x0, y0=y0, zs=zs: {"x0": x0 + b["x0"] / zs, "x1": x0 + b["x1"] / zs,
                                                                     "y0": y0 + b["y0"] / zs, "y1": y0 + b["y1"] / zs})})
        return out

    def merged_dets(views_f: list[dict], cand_dir: str, rung: int, f: int) -> list[dict]:
        """Detections from all views mapped to frame pixels; same object (IoU >= 0.6) merged, max conf kept."""
        dets = []
        for v in views_f:
            for d in load_dets(cache_name(os.path.join(cand_dir, v["name"]), rung, f), 0.2):
                dets.append({**d, "box": v["back"](d["box"]), "views": [v["name"]]})
        dets.sort(key=lambda d: -float(d.get("confidence", 0)))
        out = []
        for d in dets:
            hit = next((o for o in out if box_iou(o["box"], d["box"]) >= 0.6), None)
            if hit is None:
                out.append(d)
            elif d["views"][0] not in hit["views"]:
                hit["views"].append(d["views"][0])
        return out

    def evaluate(images_dir: str, tag: str):
        cand_dir = os.path.join(S, "dino", tag)
        t0 = time.time()
        vmap = {f: views(images_dir, tag, f) for f in frames}
        jobs = [{"image": v["path"], "out": cache_name(os.path.join(cand_dir, v["name"]), r, f), "prompt": p,
                 "thr": 0.2} for r, p in enumerate(ladder) for f in frames for v in vmap[f]]
        gdino_batch(jobs, gd_weights)
        log(f"  Grounding DINO on {tag} images: {len(jobs)} runs ({len(ladder)} prompts x {len(frames)} frames x "
            f"views {sorted({v['name'] for vs in vmap.values() for v in vs})}) in {time.time() - t0:.0f}s")
        cands, options = [], []
        for rung, prompt in enumerate(ladder):
            for f in frames:
                d = valid_depth(depth_dir, f)
                for k, det in enumerate(merged_dets(vmap[f], cand_dir, rung, f)):
                    m = measure(det["box"], f, d)
                    conf = float(det.get("confidence", 0)) * (1.0 + 0.1 * (len(det["views"]) - 1))
                    cands.append({"tag": tag, "rung": rung, "prompt": prompt, "frame": f, "video_frame": f + off,
                                  "k": k, "label": det.get("label", ""), "conf": conf, "views": det["views"],
                                  "box": det["box"], **{x: m[x] for x in ("depth_m", "dz_m", "size_m", "gap",
                                                                         "wrist_gap", "actor_iou", "depth_source")},
                                  "score": conf * math.exp(-(m["wrist_gap"] or 9) / 0.25),
                                  "accepted": not m["reasons"], "reasons": m["reasons"]})
        for rung, prompt in enumerate(ladder):
            # Size consensus: the same rigid object should have about the same metric size on
            # every frame; a loose box (object + background) is an outlier against the median.
            acc = [c["size_m"] for c in cands if c["rung"] == rung and c["accepted"]]
            ref_size = float(np.median(acc)) if acc else None
            for c in cands:
                if c["rung"] == rung:
                    c["ref_size_m"] = ref_size
                    ok_size = c["size_m"] is not None and ref_size
                    c["size_factor"] = (math.exp(-abs(math.log(c["size_m"] / ref_size)) / 0.3) if ok_size else 1.0)
                    c["consistent"] = bool(not ok_size or max(c["size_m"] / ref_size, ref_size / c["size_m"])
                                           <= args.consistency)
                    c["score"] = c["conf"] * math.exp(-(c["wrist_gap"] or 9) / 0.25) * c["size_factor"]
            for thr in THRESHOLDS:
                best = {}
                for c in cands:
                    if c["rung"] == rung and c["accepted"] and c["consistent"] and c["conf"] >= thr:
                        if c["frame"] not in best or c["score"] > best[c["frame"]]["score"]:
                            best[c["frame"]] = c
                if len(best) >= min(2, len(frames)):
                    top = max(best.values(), key=lambda c: c["score"])
                    sizes = [c["size_m"] for c in best.values()]
                    inconsistent = [{"video_frame": c["video_frame"], "size_m": c["size_m"], "conf": c["conf"],
                                     "box": c["box"]} for c in cands
                                    if c["rung"] == rung and c["accepted"] and not c["consistent"] and c["conf"] >= thr]
                    options.append({"rung": rung, "prompt": prompt, "threshold": thr, "chosen": top,
                                    "frames_with_object": sorted(best), "per_frame_best": best,
                                    "size_spread": max(sizes) / min(sizes), "ref_size_m": ref_size,
                                    "rejected_inconsistent": inconsistent})
        if not options:
            return cands, None
        # Most frames with the object wins; then the higher threshold (stricter), then the earlier prompt.
        if sib_sizes:
            # Cross-video consistency: the same physical object was already found in the object's other videos.
            # An option whose typical size is far from theirs is another object (episode 19: the generic prompt
            # 'roller.' found the white TABLE top on all 9 frames, 0.44-0.62 m, vs 0.20-0.21 m in episodes 18/20).
            # must agree with AT LEAST ONE other video (one wrong sibling must not move the reference for the rest)
            ok = [o for o in options if o["ref_size_m"] and any(max(o["ref_size_m"] / x, x / o["ref_size_m"]) <= args.xvideo
                                                                for x in sib_sizes)]
            for o in options:
                if o not in ok:
                    log(f"  option '{o['prompt']}' (thr {o['threshold']}) rejected: typical size {o['ref_size_m']:.2f} m agrees "
                        f"with none of the object's other videos {[round(x, 2) for x in sib_sizes]} m")
            if ok:
                options = ok
            else:
                log(f"  no option agrees with the other videos' sizes {[round(x, 2) for x in sib_sizes]} m; keeping all")
        return cands, max(options, key=lambda o: (len(o["frames_with_object"]), o["threshold"], -o["rung"]))

    # the same object's size in its other videos (their stage-3 results), if already known
    sib = [json.loads(l)["episode_index"] for l in open(META)
           if json.loads(l)["object"] == md["object"] and json.loads(l)["episode_index"] != args.episode]
    sib_sizes = []
    for e2 in sib:
        p2 = os.path.join(os.path.abspath(args.out_root), f"episode_{e2:06d}", "object_track.json")
        if os.path.exists(p2):
            r2 = json.load(open(p2))
            if r2.get("status") == "ok" and (r2.get("ref_size_m") or r2.get("size_m")):
                sib_sizes.append(r2.get("ref_size_m") or r2.get("size_m"))
    if sib_sizes:
        log(f"  the object's typical size in its other videos {sib}: {[round(x, 2) for x in sib_sizes]} m (an option must "
            f"agree with at least one within x{args.xvideo})")

    seed_path = os.path.join(SEEDS, f"{ep}.json")
    all_cands, decision, mode = [], None, None
    if os.path.exists(seed_path):
        seed = json.load(open(seed_path))
        f = int(seed["video_frame"]) - off
        b = box_dict(seed["box"])
        H_, W_ = np.asarray(Image.open(os.path.join(frames_dir, "000000.png"))).shape[:2]
        n_clip = meta["n_frames"]
        problems = []
        if not 0 <= f < n_clip:
            problems.append(f"video frame {f + off} is outside the clip {off}..{off + n_clip - 1}")
        if not (0 <= b["x0"] < b["x1"] <= W_ and 0 <= b["y0"] < b["y1"] <= H_):
            problems.append(f"box {seed['box']} is not inside the {W_}x{H_} image")
        if not problems:
            if f not in actor:                                 # gates need the actor at that frame
                pb_ = box_dict(boxes[f]); d_ = valid_depth(depth_dir, f)
                am_ = actor_mask(f)
                pz_ = float(np.nanmedian(d_[am_])) if am_ is not None else region_depth(d_, pb_, 50)
                actor[f] = {"box": pb_, "height_px": pb_["y1"] - pb_["y0"], "depth_m": pz_,
                            "wrist_r": kp[f, R_WRIST].tolist(), "wrist_l": kp[f, L_WRIST].tolist()}
            ms = measure(b, f, valid_depth(depth_dir, f))
            if ms["size_m"] is None or not size_lo <= ms["size_m"] <= size_hi:
                problems.append(f"seed size {ms['size_m']} m is outside the prior {size_lo:.2f}-{size_hi:.2f} m")
        if problems:
            log(f"{ep}: stage 3 FAILED — manual seed {seed_path} rejected: " + "; ".join(problems))
            raise SystemExit(3)
        mode = "manual"
        decision = {"chosen": {"frame": f, "video_frame": f + off, "box": b, "conf": 1.0, "prompt": "manual seed",
                               "label": "manual", "size_m": ms["size_m"], "dz_m": ms["dz_m"], "gap": ms["gap"],
                               "wrist_gap": ms["wrist_gap"]},
                    "prompt": "manual seed", "threshold": None, "frames_with_object": [f], "size_spread": 1.0}
        log(f"  MANUAL seed from {seed_path}: video frame {f + off}, box {seed['box']}")
    else:
        all_cands, decision = evaluate(near_dir, "near")
        mode = "near" if decision else None
        if decision is None:
            log("  nothing passed on the depth-covered images; trying the full images")
            full_cands, decision = evaluate(frames_dir, "full")
            all_cands += full_cands
            mode = "full" if decision else None

    if decision and mode == "near":
        # The covered images can cut a large object (part of it lies outside the V). Re-detect with the
        # winning prompt on the FULL frames (+ mirror) and grow each box to the largest gated detection
        # that contains it (>= 80 % of the covered box, <= 4x its area).
        full_dir = os.path.join(S, "dino", "full_refine")
        fr_frames = sorted(decision["per_frame_best"])
        vmap = {f: [v for v in views(frames_dir, "full", f) if not v["name"].startswith("hand")] for f in fr_frames}
        gdino_batch([{"image": v["path"], "out": cache_name(os.path.join(full_dir, v["name"]), 0, f),
                      "prompt": decision["prompt"], "thr": 0.2} for f in fr_frames for v in vmap[f]], gd_weights)
        area = lambda b: max(0.0, b["x1"] - b["x0"]) * max(0.0, b["y1"] - b["y0"])
        def covered(inner, outer):
            ix = max(0.0, min(inner["x1"], outer["x1"]) - max(inner["x0"], outer["x0"]))
            iy = max(0.0, min(inner["y1"], outer["y1"]) - max(inner["y0"], outer["y0"]))
            return ix * iy / max(area(inner), 1.0)
        grown = 0
        for f, c in decision["per_frame_best"].items():
            d = valid_depth(depth_dir, f)
            pool = []
            for det in merged_dets(vmap[f], full_dir, 0, f):
                b = det["box"]
                if covered(c["box"], b) >= 0.8 and area(b) <= 4 * area(c["box"]) and not measure(b, f, d)["reasons"]:
                    pool.append(b)
            if pool:
                b = max(pool, key=area)
                if area(b) > 1.15 * area(c["box"]):
                    m = measure(b, f, d)                      # every measurement of the grown box, re-gated
                    if not m["reasons"]:
                        c["box_near"], c["box"] = c["box"], b
                        c.update({x: m[x] for x in ("depth_m", "dz_m", "size_m", "gap", "wrist_gap", "actor_iou")})
                        grown += 1
        log(f"  whole-object refinement on full frames: {grown}/{len(fr_frames)} boxes grown (re-measured, re-gated)")

    if decision and mode in ("near", "full"):
        # Final consistency over the supports as they are now (after growth): one rigid object has about the
        # same metric size on every frame; a box far from the typical size is another object (ep 13: the table).
        best = decision["per_frame_best"]
        dropped = {}
        while len(best) >= 2:                       # repeat until stable: the typical size moves when one drops
            typical = float(np.median([c["size_m"] for c in best.values()]))
            ratio = {f: max(c["size_m"] / typical, typical / c["size_m"]) for f, c in best.items()}
            worst = max(ratio, key=ratio.get)
            if ratio[worst] <= args.consistency:
                break
            dropped[worst] = best.pop(worst)
            log(f"  support at video frame {worst + off} dropped: size {dropped[worst]['size_m']:.2f} m vs typical "
                f"{typical:.2f} m (x{ratio[worst]:.2f})")
        decision["rejected_inconsistent"] = decision.get("rejected_inconsistent", []) + [
            {"video_frame": int(f + off), "size_m": c["size_m"], "conf": c["conf"], "box": c["box"], "stage": "final"}
            for f, c in dropped.items()]
        decision["frames_with_object"] = sorted(best)
        if best:
            sizes = [c["size_m"] for c in best.values()]
            decision["size_spread"] = max(sizes) / min(sizes)
            decision["ref_size_m"] = float(np.median(sizes))
            if decision["chosen"]["frame"] not in best:
                decision["chosen"] = max(best.values(), key=lambda c: c["score"])
        if len(best) < 2:
            log(f"{ep}: stage 3 FAILED — only {len(best)} frame(s) support one consistent object "
                f"(need 2 of {len(frames)}); add a manual seed at {seed_path}")
            decision = None

    n_acc = sum(c["accepted"] for c in all_cands)
    json.dump({"episode": args.episode, "object": md["object"], "ladder": ladder,
               "size_accept_m": [size_lo, size_hi], "frames": frames, "frame_offset": off,
               "actor": {str(k): v for k, v in actor.items()}, "near": {str(k): v for k, v in near_info.items()},
               "candidates": all_cands}, open(os.path.join(S, "candidates.json"), "w"), indent=1)
    record = {"episode": args.episode, "status": "ok" if decision else "not_found", "mode": mode,
              "object": md["object"], "original_prompt": md["object_prompt"], "object_id": 1,
              "n_candidates": len(all_cands), "n_accepted": n_acc}
    if decision:
        c = decision["chosen"]
        record.update({"reference_frame": c["frame"], "reference_video_frame": c["video_frame"],
                       "box": c["box"], "prompt": decision["prompt"], "label": c["label"],
                       "confidence": c["conf"], "box_threshold": decision["threshold"],
                       "frames_with_object": decision["frames_with_object"],
                       "size_m": c["size_m"], "size_spread": decision["size_spread"],
                       "ref_size_m": decision.get("ref_size_m"),
                       "per_frame_boxes": {str(k): {"box": v["box"], "conf": v["conf"], "size_m": v["size_m"]}
                                           for k, v in decision.get("per_frame_best", {}).items()},
                       "dz_m": c["dz_m"], "gap": c["gap"], "wrist_gap": c["wrist_gap"],
                       "frames_sampled": [int(x) for x in frames],
                       "rejected_inconsistent": decision.get("rejected_inconsistent", []),
                       "consistency_factor": args.consistency})
        json.dump([{"label": c["label"], "confidence": c["conf"], "box": c["box"]}],
                  open(os.path.join(W, "dino_detections.json"), "w"), indent=2)
        json.dump(record, open(os.path.join(W, "object_track.json"), "w"), indent=2)
    json.dump(record, open(os.path.join(S, "object_track.json"), "w"), indent=2)

    if decision:
        c = decision["chosen"]
        size_txt = "-" if c["size_m"] is None else f"{c['size_m']:.2f} m"
        log(f"{ep}: stage 3 OK — {mode}: prompt {decision['prompt']!r}, video frame {c['video_frame']}, "
            f"conf {c['conf']:.2f}, size {size_txt}, object on {len(decision['frames_with_object'])}/"
            f"{len(frames)} frames ({n_acc}/{len(all_cands)} detections accepted)")
    else:
        log(f"{ep}: stage 3 FAILED — no consistent object passed the gates ({len(all_cands)} detections); "
            f"add a manual seed at {seed_path}")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
