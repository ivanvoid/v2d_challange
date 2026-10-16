#!/usr/bin/env python3
"""Stage 4 helpers: build SAM2 prompts, fill gaps after a first SAM2 pass, repair transient leaks.

  --mode init    prompts from stage 3: the (whole-object) box on each stage-3 frame whose metric size
                 agrees with the others (within x1.5 of their median), plus NEGATIVE points on the
                 actor's body (SAM3D-Body 2D keypoints: nose, shoulders, elbows, hips, knees, ankles,
                 mid upper arm, mid forearm) that lie outside the box (+margin) — so SAM2 does not
                 grow into the arm / legs / torso of the person holding the object.
                 -> out/<ep>/mask_prompts.json
  --mode fill    after SAM2: per scored frame measure mask area, metric size, depth vs the actor,
                 and colour (median Lab inside the mask vs the colour on the prompt frames). Bad
                 frames (empty, < 0.4 x median area, size outside the prior, > 0.6 m from the actor,
                 colour Delta-E > --max_de) are grouped into stretches; on frames in each stretch the
                 object is re-detected (Grounding DINO on the image, its mirror and zooms on the hands;
                 same gates as stage 3 + size within x1.6 of the stage-3 size) and good boxes are added
                 as prompts. Exit code 10 = prompts changed (rerun SAM2), 0 = nothing to add.
  --mode repair  transient leaks: a frame whose mask area spikes > 1.6 x the local median (+-7
                 frames) is cut to the dilated union of its good neighbours; a small mask far from
                 where the object is (e.g. a shoe) is cleared. Raw SAM2 masks kept in masks_raw/1.
                 -> out/<ep>/masks_repair.json

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
    """Per scored frame: area, size, dz, colour distance to the prompt-frame colour."""
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
        if r["area"] > 0:
            ys, xs = np.nonzero(m)
            d = depth_map(W, f)
            oz = float(np.median(d[m]))
            r["size_m"] = max(np.ptp(xs), np.ptp(ys)) * oz / c.fx
            r["dz_m"] = oz - c.actor_depth(f, d)
            r["centroid"] = [float(xs.mean()), float(ys.mean())]
            if ref_col is not None:
                rgb = np.asarray(Image.open(os.path.join(W, "frames", f"{f:06d}.png")).convert("RGB"))
                col = colour_lab(rgb, m)
                r["de"] = float(np.linalg.norm(col - ref_col)) if col is not None else None
        rows[f] = r
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
                    ok = (m["gap"] <= 0.1 and abs(m["dz_m"]) <= 0.6 and c.size_lo <= m["size_m"] <= c.size_hi
                          and m["actor_iou"] <= 0.6 and (ref is None or ref / 1.6 <= m["size_m"] <= ref * 1.6))
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
    ap.add_argument("--mode", choices=("init", "fill", "repair"), required=True)
    ap.add_argument("--max_de", type=float, default=30.0, help="max colour distance (Lab) to the prompt frames")
    ap.add_argument("--max_new", type=int, default=8)
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
    if args.mode == "fill":
        rows = frame_stats(c, W, prompts)
        areas = np.array([r["area"] for r in rows.values()], float)
        med = float(np.median(areas[areas > 0])) if (areas > 0).any() else 1.0
        bad = []
        for f, r in rows.items():
            why = None
            if r["area"] == 0:
                why = "empty"
            elif r["area"] < 0.4 * med:
                why = "small"
            elif not c.size_lo <= r["size_m"] <= c.size_hi:
                why = "size"
            elif abs(r["dz_m"]) > 0.6:
                why = "depth"
            elif r.get("de") is not None and r["de"] > args.max_de:
                why = "colour"
            if why:
                bad.append((f, why))
        stretches, cur = [], []
        for f, why in bad:
            if cur and f - cur[-1][0] > 3:
                stretches.append(cur); cur = []
            cur.append((f, why))
        if cur:
            stretches.append(cur)
        stretches = [s for s in stretches if len(s) >= 5]
        have = {p["frame"] for p in prompts}
        targets = []
        for s in stretches:
            f0, f1 = s[0][0], s[-1][0]
            k = max(1, (f1 - f0) // 30 + 1)
            targets += [int(round(f0 + (f1 - f0) * (i + 0.5) / k)) for i in range(k)]
        targets = [f for f in targets if f not in have]
        log(f"{ep}: {len(bad)} bad frames ({', '.join(sorted({w for _, w in bad})) or '-'}) in "
            f"{len(stretches)} stretches >= 5 frames; re-detecting on video frames {[f + c.off for f in targets]}")
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
    fixed, cleared = [], []
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
            ms[f] = np.zeros_like(ms[f]); cleared.append(f)    # small blob far from the object: drop
    for f in fixed + cleared:
        cv2.imwrite(os.path.join(W, "masks", "1", f"{f:06d}.png"), (ms[f] * 255).astype(np.uint8))
    for f in range(n):                                          # restore untouched frames from raw
        if f not in fixed and f not in cleared:
            src = os.path.join(raw_dir, f"{f:06d}.png")
            dst = os.path.join(W, "masks", "1", f"{f:06d}.png")
            if os.path.exists(src):
                shutil.copyfile(src, dst)
    json.dump({"leak_cut_frames": [f + c.off for f in fixed], "cleared_frames": [f + c.off for f in cleared]},
              open(rep_path, "w"), indent=1)
    log(f"{ep}: repair — {len(fixed)} leak frames cut {[f + c.off for f in fixed][:12]}, "
        f"{len(cleared)} stray blobs cleared {[f + c.off for f in cleared][:12]}")


if __name__ == "__main__":
    main()
