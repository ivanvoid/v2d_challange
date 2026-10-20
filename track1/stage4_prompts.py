#!/usr/bin/env python3
"""Stage 4 helpers: build SAM2 prompts, fill gaps after a SAM2 pass, repair transient leaks, assess.

  --mode init    prompts from stage 3: the (whole-object) box on each stage-3 frame whose metric size
                 agrees with the others (within x1.5 of their median), plus NEGATIVE points on the
                 actor: points spread over the actor's person mask (stage 2) and the SAM3D-Body 2D
                 keypoints (nose, shoulders, elbows, hips, knees, ankles, mid upper arm, mid forearm),
                 all outside the box (+margin) — so SAM2 does not grow into the person holding the object.
                 -> out/<ep>/mask_prompts.json
  --mode fill    after SAM2: every frame of the contact span gets a status (mask_quality.py: ok /
                 low_conf / suspicious / leak / hidden / lost). Suspicious and lost frames are grouped into
                 stretches (>= --min_stretch frames); on frames in each stretch the object is re-detected
                 (Grounding DINO on the image, its mirror and zooms on the hands) and a box is added as a
                 prompt only if it passes the stage-3 gates, is within x1.6 of the typical size, is within
                 0.15 actor heights of a wrist (contact span: the hand touches the object), and is not the
                 actor (< 50 % of the box on the person mask). Exit code 10 = prompts changed (rerun SAM2).
  --mode repair  transient leaks: a frame whose mask area spikes > 1.6 x the local median (+-7
                 frames) is cut to the dilated union of its good neighbours; a small mask far from
                 where the object is (e.g. a shoe) is REJECTED (emptied, with the reason recorded).
                 Raw SAM2 masks kept in masks_raw/1. -> out/<ep>/masks_repair.json
  --mode assess  final status of every contact-span frame after recovery and repair ->
                 out/<ep>/masks_quality.json (per frame: status, reasons, measurements; unresolved
                 lost/rejected stretches on scored frames, longest one). Exit code 3 (acceptance failed)
                 if more than --max_lost of the scored frames are lost or rejected.

Run with the sam3d_body venv.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logutil import log  # noqa: E402
from select_object import box_iou, cache_name, gdino_batch, load_dets, rect_gap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import paths  # noqa: E402
import mask_quality as mq  # noqa: E402
PRIORS = json.load(open(paths.PRIORS))
META = str(paths.META)
NEG_KP = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)           # mhr70: nose, shoulders, elbows, hips, knees, ankles
R_WRIST, L_WRIST, L_SH, R_SH, L_EL, R_EL = 41, 62, 5, 6, 7, 8
SLACK = 1.6


def depth_map(W, f):
    px = cv2.imread(os.path.join(W, "depth", f"{f:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    return 1.0 / np.maximum(px / 65535.0, 1e-6) - 1.0


def mask(W, f, sub="masks"):
    m = cv2.imread(os.path.join(W, sub, "1", f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
    return (m > 0) if m is not None else None


def area_b(b):
    return max(0.0, b["x1"] - b["x0"]) * max(0.0, b["y1"] - b["y0"])


def covered(inner, outer):
    ix = max(0.0, min(inner["x1"], outer["x1"]) - max(inner["x0"], outer["x0"]))
    iy = max(0.0, min(inner["y1"], outer["y1"]) - max(inner["y0"], outer["y0"]))
    return ix * iy / max(area_b(inner), 1.0)


class Ctx:
    def __init__(self, W):
        self.W = W
        self.meta = json.load(open(os.path.join(W, "track1_meta.json")))
        self.off, self.n = self.meta["frame_offset"], self.meta["n_frames"]
        self.first, self.last = self.meta["scored_first"] - self.off, self.meta["scored_last"] - self.off
        it = json.load(open(os.path.join(W, "intrinsics_stable.json")))
        self.fx, self.hw = it["fx"], (it["height"], it["width"])
        self.track = json.load(open(os.path.join(W, "object_track.json")))
        md = next(json.loads(l) for l in open(META) if json.loads(l)["episode_index"] == self.meta["episode"])
        self.prior = PRIORS.get(md["object"], {})
        lo, hi = self.prior.get("size_m", [0.05, 3.0])
        self.size_lo, self.size_hi = lo / SLACK, hi * SLACK
        self.small = hi <= 0.6
        self.boxes = np.asarray(torch.load(os.path.join(W, "body", "bbox_track.pt"), map_location="cpu",
                                           weights_only=False)["bbox_track"], float)
        self.kp = torch.load(os.path.join(W, "body", "mhr_params.pt"), map_location="cpu",
                             weights_only=False)["pred_keypoints_2d"].numpy()
        sp = json.load(open(paths.DATA / "submission" / "scored_spans.json"))[str(self.meta["episode"])]
        self.scored = set(int(f) - self.off for f in sp["frames"])
        self._pm = {}

    def person(self, f):
        """The actor's person mask (stage 2), or None."""
        if f not in self._pm:
            m = cv2.imread(os.path.join(self.W, "body", "person_masks", f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
            self._pm[f] = (m > 0) if m is not None else None
        return self._pm[f]

    def ph(self, f):
        b = self.boxes[f]
        return max(b[3] - b[1], 1.0)

    def actor_depth(self, f, d):
        b = self.boxes[f]
        h, w = b[3] - b[1], b[2] - b[0]
        reg = d[int(b[1] + 0.15 * h):int(b[3] - 0.15 * h), int(b[0] + 0.25 * w):int(b[2] - 0.25 * w)]
        return float(np.median(reg)) if reg.size else float("nan")

    def negatives(self, f, box):
        k = self.kp[f]
        pts = [k[i] for i in NEG_KP]
        pts += [(k[L_SH] + k[L_EL]) / 2, (k[R_SH] + k[R_EL]) / 2,           # mid upper arms
                (k[L_EL] + k[L_WRIST]) / 2, (k[R_EL] + k[R_WRIST]) / 2]      # mid forearms
        m = max(0.15 * max(box["x1"] - box["x0"], box["y1"] - box["y0"]), 0.06 * self.ph(f))
        H, Wd = self.hw
        pm = self.person(f)
        if pm is not None:                       # points spread over the actor's body (farthest-point sampling)
            core = cv2.erode(pm.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            ys, xs = np.nonzero(core[::4, ::4])
            cand = np.stack([xs * 4, ys * 4], 1).astype(float)
            keep = ~((cand[:, 0] >= box["x0"] - m) & (cand[:, 0] <= box["x1"] + m)
                     & (cand[:, 1] >= box["y0"] - m) & (cand[:, 1] <= box["y1"] + m))
            cand = cand[keep]
            if len(cand):
                chosen = [cand[len(cand) // 2]]
                d = np.linalg.norm(cand - chosen[0], axis=1)
                for _ in range(11):
                    i = int(np.argmax(d))
                    if d[i] < 0.05 * self.ph(f):
                        break
                    chosen.append(cand[i]); d = np.minimum(d, np.linalg.norm(cand - cand[i], axis=1))
                pts = pts + [np.asarray(c) for c in chosen]
        out = []
        for p in pts:
            x, y = float(p[0]), float(p[1])
            if not (0 <= x < Wd and 0 <= y < H):
                continue
            if box["x0"] - m <= x <= box["x1"] + m and box["y0"] - m <= y <= box["y1"] + m:
                continue
            out.append([x, y])
        return out

    def measure(self, b, f, d):
        """Same gates as stage 3 (depth vs actor, metric size, gap to actor)."""
        cx0, cx1 = b["x0"] + 0.25 * (b["x1"] - b["x0"]), b["x1"] - 0.25 * (b["x1"] - b["x0"])
        cy0, cy1 = b["y0"] + 0.25 * (b["y1"] - b["y0"]), b["y1"] - 0.25 * (b["y1"] - b["y0"])
        reg = d[int(max(0, cy0)):int(cy1) + 1, int(max(0, cx0)):int(cx1) + 1]
        oz = float(np.median(reg)) if reg.size else float("nan")
        size = max(b["x1"] - b["x0"], b["y1"] - b["y0"]) * oz / self.fx
        pb = dict(zip(("x0", "y0", "x1", "y1"), self.boxes[f]))
        gap = rect_gap(b, pb) / self.ph(f)
        k = self.kp[f]
        wgap = min(math.hypot(max(b["x0"] - p[0], p[0] - b["x1"], 0), max(b["y0"] - p[1], p[1] - b["y1"], 0))
                   for p in (k[R_WRIST], k[L_WRIST])) / self.ph(f)
        return {"size_m": size, "dz_m": oz - self.actor_depth(f, d), "gap": gap, "wrist_gap": wgap,
                "actor_iou": box_iou(b, pb)}


def init_prompts(c: Ctx, W: str) -> list[dict]:
    t = c.track
    per = {int(k): v for k, v in t.get("per_frame_boxes", {}).items()}
    if not per:
        per = {int(t["reference_frame"]): {"box": t["box"], "size_m": t.get("size_m")}}
    sizes = [v["size_m"] for v in per.values() if v.get("size_m")]
    med = float(np.median(sizes)) if sizes else None
    out = []
    for f, v in sorted(per.items()):
        if med and v.get("size_m") and not (med / 1.5 <= v["size_m"] <= med * 1.5):
            log(f"  stage-3 box at video frame {f + c.off} dropped: size {v['size_m']:.2f} m vs median {med:.2f} m")
            continue
        out.append({"frame": f, "box": v["box"], "neg": c.negatives(f, v["box"]), "source": "stage3"})
    return out


def colour_lab(rgb, m):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return np.median(lab[m], axis=0) if m.sum() >= 20 else None


def frame_stats(c: Ctx, W: str, prompts: list[dict], sub="masks"):
    """Per contact-span frame: area, size, dz, colour distance to the prompt-frame colour, distance from the
    nearest wrist (cm at the actor's depth), share of the mask on the actor's person mask, SAM2 confidence."""
    conf_path = os.path.join(W, "masks_conf.json")
    conf = json.load(open(conf_path))["frames"] if os.path.exists(conf_path) else {}
    ref_cols = []
    for p in prompts:
        m = mask(W, p["frame"], sub)
        if m is not None and m.any():
            rgb = np.asarray(Image.open(os.path.join(W, "frames", f"{p['frame']:06d}.png")).convert("RGB"))
            col = colour_lab(rgb, m)
            if col is not None:
                ref_cols.append(col)
    ref_col = np.median(ref_cols, axis=0) if ref_cols else None
    rows = {}
    for f in range(c.first, c.last + 1):
        m = mask(W, f, sub)
        r = {"area": int(m.sum()) if m is not None else 0}
        cf = conf.get(str(f), {})
        r.update({k: cf.get(k) for k in ("prob", "uncertain", "fb_iou")})
        if r["area"] > 0:
            ys, xs = np.nonzero(m)
            d = depth_map(W, f)
            oz = float(np.percentile(d[m], 20))      # nearest 20 %: wall depth bleeds into thin / edge-on objects
            az = c.actor_depth(f, d)
            r["size_m"] = max(np.ptp(xs), np.ptp(ys)) * oz / c.fx
            r["dz_m"] = oz - az
            r["centroid"] = [float(xs.mean()), float(ys.mean())]
            dt = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
            H, Wd = m.shape
            px = [dt[int(min(H - 1, max(0, k[1]))), int(min(Wd - 1, max(0, k[0])))]
                  for k in (c.kp[f][R_WRIST], c.kp[f][L_WRIST])]
            r["wrist_cm"] = float(min(px) * az / c.fx * 100)
            pm = c.person(f)
            r["person_frac"] = float((m & pm).sum() / r["area"]) if pm is not None else None
            r["body_cover"] = float((m & pm).sum() / max(pm.sum(), 1)) if pm is not None else None
            if ref_col is not None:
                rgb = np.asarray(Image.open(os.path.join(W, "frames", f"{f:06d}.png")).convert("RGB"))
                col = colour_lab(rgb, m)
                r["de"] = float(np.linalg.norm(col - ref_col)) if col is not None else None
        rows[f] = r
    return rows


def assign_status(c: Ctx, rows: dict, rejected: dict | None = None) -> dict:
    """Status of every contact-span frame (mask_quality.classify). An empty mask is 'hidden' when the
    actor's (dilated) person mask covers where the object should be: its centroid interpolated between the
    nearest frames that still have a mask; otherwise 'lost'."""
    frames = sorted(rows)
    areas = np.array([rows[f]["area"] for f in frames], float)
    sizes = [rows[f]["size_m"] for f in frames if rows[f].get("size_m")]
    typical = float(np.median(sizes)) if sizes else c.track.get("ref_size_m")   # the episode's own mask size
    have = [f for f in frames if rows[f]["area"] > 0]
    for i, f in enumerate(frames):
        r = rows[f]
        if rejected and f in rejected:
            r["rejected"] = rejected[f]
        if r["area"] == 0 and have:
            a = [g for g in have if g < f]; b = [g for g in have if g > f]
            if a and b:
                ca, cb = np.array(rows[a[-1]]["centroid"]), np.array(rows[b[0]]["centroid"])
                w = (f - a[-1]) / (b[0] - a[-1]); cen = (1 - w) * ca + w * cb
            else:
                cen = np.array(rows[(a or b)[-1 if a else 0]]["centroid"])
            pm = c.person(f)
            if pm is not None:
                pm = cv2.dilate(pm.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
                y, x = int(min(pm.shape[0] - 1, max(0, cen[1]))), int(min(pm.shape[1] - 1, max(0, cen[0])))
                r["hidden_hint"] = bool(pm[y, x])
        r["status"], r["reasons"] = mq.classify(r, typical, mq.local_median_area(areas, i), True)
    return rows


def detect_on(c: Ctx, W: str, frames: list[int], prompts_txt: list[str], gd: str) -> dict:
    """TTA detection (image, mirror, hand zooms) -> best gated box per frame."""
    vdir = os.path.join(W, "masks_fill", "views")
    os.makedirs(vdir, exist_ok=True)
    jobs, views = [], {}
    for f in frames:
        img = Image.open(os.path.join(W, "frames", f"{f:06d}.png")).convert("RGB")
        Wd, H = img.size
        vs = [("image", os.path.join(W, "frames", f"{f:06d}.png"), lambda b: b)]
        fp = os.path.join(vdir, f"flip_{f:06d}.png")
        if not os.path.exists(fp):
            img.transpose(Image.FLIP_LEFT_RIGHT).save(fp)
        vs.append(("flip", fp, lambda b, Wd=Wd: {"x0": Wd - b["x1"], "x1": Wd - b["x0"], "y0": b["y0"], "y1": b["y1"]}))
        k = c.kp[f]
        for i, cc in enumerate((k[R_WRIST], k[L_WRIST])):
            half = 0.32 * c.ph(f)
            x0, y0 = int(max(0, cc[0] - half)), int(max(0, cc[1] - half))
            x1, y1 = int(min(Wd, cc[0] + half)), int(min(H, cc[1] + half))
            if x1 - x0 < 20 or y1 - y0 < 20:
                continue
            zs = 800.0 / max(x1 - x0, y1 - y0)
            hp = os.path.join(vdir, f"hand{i}_{f:06d}.png")
            if not os.path.exists(hp):
                img.crop((x0, y0, x1, y1)).resize((int((x1 - x0) * zs), int((y1 - y0) * zs)), Image.BICUBIC).save(hp)
            vs.append((f"hand{i}", hp, lambda b, x0=x0, y0=y0, zs=zs: {"x0": x0 + b["x0"] / zs, "x1": x0 + b["x1"] / zs,
                                                                       "y0": y0 + b["y0"] / zs, "y1": y0 + b["y1"] / zs}))
        views[f] = vs
        for r, ptxt in enumerate(prompts_txt):
            for name, path, _ in vs:
                jobs.append({"image": path, "out": cache_name(os.path.join(W, "masks_fill", "dino", name), r, f),
                             "prompt": ptxt, "thr": 0.25})
    gdino_batch(jobs, gd)
    ref = c.track.get("ref_size_m") or c.track.get("size_m")
    best = {}
    for f in frames:
        d = depth_map(W, f)
        for r in range(len(prompts_txt)):
            for name, _, back in views[f]:
                for det in load_dets(cache_name(os.path.join(W, "masks_fill", "dino", name), r, f), 0.25):
                    b = back(det["box"])
                    m = c.measure(b, f, d)
                    pm = c.person(f)
                    on_person = 0.0
                    if pm is not None:
                        x0, y0, x1, y1 = (int(max(0, b["x0"])), int(max(0, b["y0"])), int(b["x1"]), int(b["y1"]))
                        on_person = float(pm[y0:y1, x0:x1].mean()) if y1 > y0 and x1 > x0 else 1.0
                    ok = (m["gap"] <= 0.1 and abs(m["dz_m"]) <= 0.6 and c.size_lo <= m["size_m"] <= c.size_hi
                          and m["actor_iou"] <= 0.6 and (ref is None or ref / 1.6 <= m["size_m"] <= ref * 1.6)
                          and m["wrist_gap"] <= 0.15 and on_person < 0.5)    # contact span: the hand touches it
                    if not ok:
                        continue
                    score = float(det["confidence"]) * math.exp(-m["wrist_gap"] / 0.25)
                    if f not in best or score > best[f]["score"]:
                        best[f] = {"box": b, "score": score, "conf": float(det["confidence"]), "view": name,
                                   "prompt": prompts_txt[r], **m}
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--mode", choices=("init", "fill", "repair", "assess"), required=True)
    ap.add_argument("--max_de", type=float, default=30.0, help="max colour distance (Lab) to the prompt frames")
    ap.add_argument("--max_new", type=int, default=8)
    ap.add_argument("--min_stretch", type=int, default=3, help="re-detect in stretches of >= this many bad frames")
    ap.add_argument("--max_lost", type=float, default=0.25,
                    help="assess: fail if more than this share of scored frames is lost or rejected")
    args = ap.parse_args()
    ep = f"episode_{args.episode:06d}"
    W = os.path.join(os.path.abspath(args.out_root), ep)
    c = Ctx(W)
    ppath = os.path.join(W, "mask_prompts.json")

    if args.mode == "init":
        prompts = init_prompts(c, W)
        old = json.load(open(ppath)) if os.path.exists(ppath) else None
        # keep gap-fill prompts from an earlier run if the stage-3 part is unchanged
        if old and [p for p in old["prompts"] if p["source"] == "stage3"] == prompts:
            log(f"{ep}: prompts unchanged ({len(old['prompts'])}, incl. "
                f"{sum(p['source'] != 'stage3' for p in old['prompts'])} gap-fill)")
            return
        json.dump({"prompts": prompts}, open(ppath, "w"), indent=1)
        log(f"{ep}: {len(prompts)} box prompts from stage 3 at video frames {[p['frame'] + c.off for p in prompts]}, "
            f"negative body points per prompt {[len(p['neg']) for p in prompts]}")
        return

    prompts = json.load(open(ppath))["prompts"]
    if args.mode == "assess":
        assess(c, W, ep, prompts, args.max_lost)
        return
    if args.mode == "fill":
        rows = assign_status(c, frame_stats(c, W, prompts))
        bad = [(f, r["status"]) for f, r in sorted(rows.items()) if r["status"] in ("suspicious", "lost")]
        stretches, cur = [], []
        for f, why in bad:
            if cur and f - cur[-1][0] > 3:
                stretches.append(cur); cur = []
            cur.append((f, why))
        if cur:
            stretches.append(cur)
        stretches = [s for s in stretches if len(s) >= args.min_stretch]
        have = {p["frame"] for p in prompts}
        targets = []
        for s in stretches:
            f0, f1 = s[0][0], s[-1][0]
            k = max(1, (f1 - f0) // 30 + 1)
            targets += [int(round(f0 + (f1 - f0) * (i + 0.5) / k)) for i in range(k)]
        targets = [f for f in targets if f not in have]
        log(f"{ep}: {len(bad)} suspicious/lost frames in {len(stretches)} stretches >= {args.min_stretch} frames; "
            f"re-detecting on video frames {[f + c.off for f in targets]}")
        if not targets:
            return
        txt = [c.track["prompt"]] + ([c.prior["prompt"]] if c.prior.get("prompt") and
                                     c.prior["prompt"] != c.track["prompt"] else [])
        found = detect_on(c, W, targets, txt, os.path.join(os.path.abspath(args.weights), "grounding_dino"))
        added = []
        for f in sorted(found, key=lambda f: -found[f]["score"])[:args.max_new]:
            b = found[f]["box"]
            prompts.append({"frame": f, "box": b, "neg": c.negatives(f, b), "source": f"fill:{found[f]['view']}"})
            added.append(f)
            log(f"  + prompt at video frame {f + c.off}: conf {found[f]['conf']:.2f} ({found[f]['view']}), "
                f"size {found[f]['size_m']:.2f} m, wrist gap {found[f]['wrist_gap']:.2f}")
        for f in targets:
            if f not in found:
                log(f"  - video frame {f + c.off}: no gated detection")
        if added:
            prompts.sort(key=lambda p: p["frame"])
            json.dump({"prompts": prompts}, open(ppath, "w"), indent=1)
            raise SystemExit(10)
        return

    # ---- repair
    raw_dir = os.path.join(W, "masks_raw", "1")
    rep_path = os.path.join(W, "masks_repair.json")
    seed_t = os.path.getmtime(os.path.join(W, "masks_seed.json"))
    if not os.path.isdir(raw_dir) or not os.path.exists(rep_path) or os.path.getmtime(rep_path) < seed_t:
        shutil.rmtree(os.path.join(W, "masks_raw"), ignore_errors=True)
        shutil.copytree(os.path.join(W, "masks"), os.path.join(W, "masks_raw"))
    n = c.n
    ms = [mask(W, f, "masks_raw") for f in range(n)]
    area = np.array([m.sum() if m is not None else 0 for m in ms], float)
    cen = np.array([np.argwhere(m).mean(0)[::-1] if m is not None and m.any() else [np.nan, np.nan] for m in ms])
    fixed, cleared, reasons = [], [], {}
    for f in range(n):
        lo, hi = max(0, f - 7), min(n, f + 8)
        nb = [g for g in range(lo, hi) if g != f and area[g] > 0]
        if not nb or area[f] == 0:
            continue
        med_a = float(np.median(area[nb]))
        med_c = np.nanmedian(cen[nb], axis=0)
        if area[f] > 1.6 * med_a:                               # leak: cut to the good neighbours
            good = [g for g in (f - 2, f - 1, f + 1, f + 2) if 0 <= g < n and 0 < area[g] <= 1.3 * med_a]
            if good:
                ref = np.zeros_like(ms[f])
                for g in good:
                    ref |= ms[g]
                r = int(max(5, 0.25 * math.sqrt(med_a)))
                ref = cv2.dilate(ref.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
                new = ms[f] & ref
                if new.sum() < area[f]:
                    ms[f] = new; fixed.append(f)
        elif area[f] < 0.6 * med_a and np.linalg.norm(cen[f] - med_c) > 1.5 * math.sqrt(med_a):
            ms[f] = np.zeros_like(ms[f]); cleared.append(f)    # small blob far from the object: rejected
            reasons[f] = (f"stray blob: {area[f] / med_a:.2f} x the local area, "
                          f"{np.linalg.norm(cen[f] - med_c) / math.sqrt(med_a):.1f} object-sizes from it")
    for f in fixed + cleared:
        cv2.imwrite(os.path.join(W, "masks", "1", f"{f:06d}.png"), (ms[f] * 255).astype(np.uint8))
    for f in range(n):                                          # restore untouched frames from raw
        if f not in fixed and f not in cleared:
            src = os.path.join(raw_dir, f"{f:06d}.png")
            dst = os.path.join(W, "masks", "1", f"{f:06d}.png")
            if os.path.exists(src):
                shutil.copyfile(src, dst)
    json.dump({"leak_cut_frames": [f + c.off for f in fixed], "cleared_frames": [f + c.off for f in cleared],
               "rejected": {str(f + c.off): reasons[f] for f in cleared}},
              open(rep_path, "w"), indent=1)
    log(f"{ep}: repair — {len(fixed)} leak frames cut {[f + c.off for f in fixed][:12]}, "
        f"{len(cleared)} stray blobs rejected {[f + c.off for f in cleared][:12]}")
    return


def assess(c: Ctx, W: str, ep: str, prompts: list[dict], max_lost: float) -> None:
    rep_path = os.path.join(W, "masks_repair.json")
    rejected = {int(k) - c.off: v for k, v in json.load(open(rep_path)).get("rejected", {}).items()} \
        if os.path.exists(rep_path) else {}
    rows = assign_status(c, frame_stats(c, W, prompts), rejected)
    sc = sorted(f for f in rows if f in c.scored)
    count = {k: sum(rows[f]["status"] == k for f in sc) for k in mq.ORDER}
    gaps = []
    for a, b in mq.runs([rows[f]["status"] in ("lost", "rejected") for f in sc]):
        gaps.append({"video_first": sc[a] + c.off, "video_last": sc[b] + c.off, "scored_frames": b - a + 1})
    lost_share = (count["lost"] + count["rejected"]) / max(len(sc), 1)
    accepted = lost_share <= max_lost
    json.dump({"episode": c.meta["episode"], "scored_frames": len(sc), "status_counts_scored": count,
               "unresolved_gaps_scored": gaps, "longest_gap": max([g["scored_frames"] for g in gaps] or [0]),
               "lost_share": lost_share, "accepted": accepted, "max_lost": max_lost,
               "thresholds": {k: getattr(mq, k) for k in ("SIZE", "DZ_M", "DE", "WRIST_CM", "AT_HAND_CM", "COLOUR_OK", "LEAK_BODY", "LEAK_AREA",
                                                         "PROB_MIN", "FB_MIN", "UNC_MAX")},
               "frames": {str(f + c.off): {k: v for k, v in r.items()} for f, r in sorted(rows.items())}},
              open(os.path.join(W, "masks_quality.json"), "w"), indent=1)
    log(f"{ep}: masks on {len(sc)} scored frames — " + ", ".join(f"{k} {v}" for k, v in count.items() if v)
        + f"; unresolved gaps {len(gaps)} (longest {max([g['scored_frames'] for g in gaps] or [0])} frames)")
    if not accepted:
        log(f"{ep}: ACCEPTANCE FAILED — {100 * lost_share:.0f}% of scored frames lost/rejected (> {100 * max_lost:.0f}%)")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
