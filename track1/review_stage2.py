#!/usr/bin/env python3
"""Stage 2 review (person + body): one sheet per episode + checks + index.html.

Per episode (review/episode_XXXXXX_body.png):
  top      full frame with the tracked person box (is it the actor, not a bystander?)
  middle   6 crops across the scored span: person box (yellow), FITTED body silhouette
           (blue, what will be submitted), its outline, and SAM3D-Body wrists (R red, L green)
  bottom   (a) body depth vs MoGe depth on the body's own pixels, per frame
           (b) joint |2nd difference| raw SAM3D-Body vs fitted (cm/frame^2)
           (c) fit error per frame (mm) and detector score (interpolated frames = -1)

Checks (episode_XXXXXX.checks.json):
  person detected (not interpolated) on >= 90 % of frames          WARN
  largest frame-to-frame box-centre jump <= 0.5 box heights        WARN  (identity switch)
  >= 90 % of projected body vertices inside the padded person box  FAIL
  standing height of the fitted identity 1.45 .. 2.05 m            WARN
  |body depth - MoGe depth| median <= 15 cm                        WARN  (shared metric scale)
  fitted joint |2nd difference| <= 0.3 cm/frame^2                  WARN
  mean fit error <= 3 mm                                           WARN

Run with the sam3d_body venv:

    /opt/venvs/sam3d_body/bin/python track1/review_stage2.py --episodes 6 8 17 26 28 \
        --out_root out --stage_dir logs/stages/02_body \
        --model weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
KIT_TRACK1 = paths.KIT_TRACK1
sys.path.insert(0, str(KIT_TRACK1))
from mesh_to_mhr_params import MHR  # noqa: E402

R_WRIST_2D, L_WRIST_2D = 41, 62  # mhr70 keypoints
REVIEW_VERSION = 6          # bump when checks change: older checks.json files are recomputed


def load_depth(path: str) -> np.ndarray:
    px = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return 1.0 / np.maximum(px / 65535.0, 1e-6) - 1.0


def project(v: np.ndarray, K: np.ndarray) -> np.ndarray:
    uvw = v @ K.T
    return uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)


def silhouette(uv: np.ndarray, faces: np.ndarray, hw) -> np.ndarray:
    m = np.zeros(hw, np.uint8)
    cv2.fillPoly(m, list(np.round(uv[faces]).astype(np.int32)), 1)
    return m.astype(bool)


def depth_bias(v: np.ndarray, uv: np.ndarray, depth: np.ndarray) -> tuple[float, float] | None:
    """Median (body z - MoGe z) in cm and median MoGe/body ratio over the front-most vertex per pixel."""
    h, w = depth.shape
    px = np.round(uv).astype(np.int64)
    ok = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h) & (v[:, 2] > 0)
    if ok.sum() < 50:
        return None
    lin, z = px[ok, 1] * w + px[ok, 0], v[ok, 2]
    o = np.lexsort((z, lin))
    lin, z = lin[o], z[o]
    first = np.r_[True, lin[1:] != lin[:-1]]
    lin, z = lin[first], z[first]
    dz = depth.reshape(-1)[lin]
    return float(np.median(z - dz) * 100), float(np.median(dz / z))


def accel_cm(x: np.ndarray) -> np.ndarray:
    a = np.full(len(x), np.nan)
    a[1:-1] = np.linalg.norm(x[:-2] - 2 * x[1:-1] + x[2:], axis=-1).mean(-1) * 100
    return a


JOINTS22 = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]  # scorer's ACC-H joints


def runs_of(frames: np.ndarray) -> list[np.ndarray]:
    cut = np.flatnonzero(np.diff(frames) != 1) + 1
    return np.split(np.arange(len(frames)), cut)


def concerns_sheet(ep_dir: str, meta: dict, review_dir: str, K: np.ndarray, hw, fit_v: np.ndarray,
                   faces: np.ndarray, vf: np.ndarray) -> tuple[list, str | None, dict]:
    """Visual checks of the concerns in review/stage2.md (third sheet, review/<ep>_concerns.png).

    (a) actor track: per-frame label (detected / redetected / interpolated / held, body/person_track.json)
        + thumbnails of the library track's largest gaps (library box red, repaired box green, mask)
    (b) image agreement: IoU of the fitted body silhouette with the actor's person mask, worst frames
    (c) body - MoGe depth on ALL body pixels vs on pixels where the person mask says the body is visible
    (d) feet: lowest body point above the stage-1 floor plane (body put in MoGe scale by the per-frame
        MoGe/body ratio) and its horizontal speed while within 3 cm of the floor
    """
    ep = f"episode_{meta['episode']:06d}"
    off = meta["frame_offset"]
    B = os.path.join(ep_dir, "body")
    pt_path = os.path.join(B, "person_track.json")
    if not os.path.exists(pt_path):
        return [{"check": "actor track", "status": "WARN", "detail": "no person_track.json (old stage 2 output)"}], None, {}
    pt = json.load(open(pt_path))
    lab = np.array([r["label"] for r in pt["per_frame"]])
    spans = json.load(open(paths.DATA / "submission" / "scored_spans.json"))[str(meta["episode"])]
    scored = np.asarray(spans["frames"])
    runs = runs_of(scored)
    k_sc = scored - vf[0]
    colors = {"detected": (0.2, 0.7, 0.2), "redetected": (0.1, 0.6, 0.9), "interpolated": (1.0, 0.6, 0.0),
              "held": (0.9, 0.1, 0.1)}
    fl_path = os.path.join(ep_dir, "floor.json")
    floor = json.load(open(fl_path)) if os.path.exists(fl_path) else None

    def mask_of(l):
        p = os.path.join(B, "person_masks", f"{l:06d}.png")
        return cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 0 if os.path.exists(p) else None

    rows = []                                     # every 2nd scored frame
    for k in k_sc[::2]:
        l = vf[k] - off
        uv = project(fit_v[k], K)
        sil = silhouette(uv, faces, hw)
        m = mask_of(l)
        d = load_depth(os.path.join(ep_dir, "depth", f"{l:06d}.png"))
        allb = depth_bias(fit_v[k], uv, d)
        vis = None
        if m is not None:
            dm = d.copy(); dm[~m] = np.nan
            px = np.round(uv).astype(int)
            ok = (px[:, 0] >= 0) & (px[:, 0] < hw[1]) & (px[:, 1] >= 0) & (px[:, 1] < hw[0])
            on = np.zeros(len(uv), bool); on[ok] = m[px[ok, 1], px[ok, 0]]
            vis = depth_bias(fit_v[k][on], uv[on], dm) if on.sum() > 50 else None
        iou_ = float((sil & m).sum() / max((sil | m).sum(), 1)) if m is not None else np.nan
        cover = float((sil & m).sum() / max(m.sum(), 1)) if m is not None else np.nan      # visible person covered
        prec = float((sil & m).sum() / max(sil.sum(), 1)) if m is not None else np.nan     # body on visible person
        rows.append({"video_frame": int(vf[k]), "iou": iou_, "cover": cover, "precision": prec,
                     "dz_all_cm": allb[0] if allb else np.nan,
                     "dz_visible_cm": vis[0] if vis and np.isfinite(vis[0]) else np.nan,
                     "ratio": allb[1] if allb else np.nan})
    iou_arr = np.array([r["iou"] for r in rows], float)
    cov_arr = np.array([r["cover"] for r in rows], float)
    pre_arr = np.array([r["precision"] for r in rows], float)
    stats = {"track_counts_scored": pt["counts_scored"], "track_accepted": pt["accepted"],
             "ambiguous": len(pt["ambiguous_frames"]), "gaps": len(pt["gaps"]),
             "mask_iou_median": float(np.nanmedian(iou_arr)) if np.isfinite(iou_arr).any() else float("nan"),
             "mask_iou_p10": float(np.nanpercentile(iou_arr, 10)) if np.isfinite(iou_arr).any() else float("nan"),
             "mask_cover_median": float(np.nanmedian(cov_arr)) if np.isfinite(cov_arr).any() else float("nan"),
             "mask_cover_p10": float(np.nanpercentile(cov_arr, 10)) if np.isfinite(cov_arr).any() else float("nan"),
             "body_on_mask_median": float(np.nanmedian(pre_arr)) if np.isfinite(pre_arr).any() else float("nan"),
             "dz_all_cm_median": float(np.nanmedian([r["dz_all_cm"] for r in rows])),
             "dz_visible_cm_median": float(np.nanmedian([r["dz_visible_cm"] for r in rows]))}

    # feet vs floor
    foot = None
    if floor is not None:
        n_ = np.asarray(floor["normal"]); d_ = floor["d"]
        rf = np.array([r["video_frame"] for r in rows]); rr = np.array([r["ratio"] for r in rows], float)
        good = np.isfinite(rr)
        ratio_k = np.interp(vf[k_sc], rf[good], rr[good]) if good.sum() > 2 else np.ones(len(k_sc))
        heights, lowest = [], []
        for i, k in enumerate(k_sc):
            P = fit_v[k] * ratio_k[i]
            h = d_ - P @ n_
            j = int(np.argmin(h))
            heights.append(h[j] * 100); lowest.append(P[j])
        heights, lowest = np.array(heights), np.array(lowest)
        tang = lowest - (lowest @ n_)[:, None] * n_[None]
        speed = np.full(len(k_sc), np.nan)
        for r in runs:
            if len(r) > 1:
                speed[r[1:]] = np.linalg.norm(np.diff(tang[r], axis=0), axis=1) * 100
        base = float(np.percentile(heights, 10))                     # body-floor offset of this episode
        contact = heights < base + 3
        foot = {"height_cm": heights, "speed_cm": speed, "base": base}
        stats.update({"lowest_point_height_cm_median": float(np.median(heights)), "floor_offset_cm": base,
                      "below_floor_5cm_frac": float((heights < -5).mean()),
                      "above_floor_10cm_frac": float((heights > 10).mean()),
                      "slide_cm_per_frame_in_contact_median": float(np.nanmedian(speed[contact])) if contact.any() else float("nan")})

    checks = []
    def add(name, ok_, detail, level="WARN"):
        checks.append({"check": name, "status": "PASS" if ok_ else level, "detail": detail})
    cs = pt["counts_scored"]; nsc = max(sum(cs.values()), 1)
    add("actor track", pt["accepted"] and (cs["redetected"] + cs["interpolated"]) / nsc <= 0.05,
        f"scored frames: {cs['detected']} detected, {cs['redetected']} re-detected, {cs['interpolated']} interpolated, "
        f"{cs['held']} held; {len(pt['gaps'])} gaps in the actor track; {stats['ambiguous']} ambiguous frames",
        "WARN" if pt["accepted"] else "FAIL")
    add("body vs person mask", stats["mask_cover_median"] >= 0.65 and stats["mask_cover_p10"] >= stats["mask_cover_median"] - 0.15,
        f"visible person covered by the body: median {100 * stats['mask_cover_median']:.0f}%, 10th percentile "
        f"{100 * stats['mask_cover_p10']:.0f}%; body pixels on the visible person {100 * stats['body_on_mask_median']:.0f}% "
        f"(lower when objects hide body parts); IoU {stats['mask_iou_median']:.2f}. Typical ~70-80 % (loose clothes, hair, "
        f"mask edges); WARN below 65 % or when the 10th percentile is > 15 points below the median (frames that drop)")
    add("depth on visible pixels", True,
        f"body - MoGe median {stats['dz_all_cm_median']:+.1f} cm on all body pixels, {stats['dz_visible_cm_median']:+.1f} cm "
        f"where the person mask says the body is visible")
    if foot is not None:
        add("feet vs floor", stats["below_floor_5cm_frac"] <= 0.1,
            f"lowest body point {stats['lowest_point_height_cm_median']:+.1f} cm above the floor (median; offset "
            f"{stats['floor_offset_cm']:+.1f} cm = 10th percentile); "
            f"{100 * stats['below_floor_5cm_frac']:.0f}% of frames > 5 cm below, {100 * stats['above_floor_10cm_frac']:.0f}% "
            f"> 10 cm above; sliding {stats['slide_cm_per_frame_in_contact_median']:.2f} cm/frame while within 3 cm of "
            f"the offset")

    # ------------------------------------------------------------------ sheet
    gaps = sorted(pt["gaps"], key=lambda g: -g["frames"])[:3]
    worst = [r for r in sorted(rows, key=lambda r: (np.nan_to_num(r["cover"], nan=9))) if np.isfinite(r["cover"])][:4]
    n_rows = 4 + (1 if gaps else 0)
    fig = plt.figure(figsize=(24, 5.5 * n_rows))
    gs = fig.add_gridspec(n_rows, 4)
    ax = fig.add_subplot(gs[0, :])
    vfr = np.array([r["video_frame"] for r in pt["per_frame"]])
    for name, c in colors.items():
        sel = lab == name
        ax.vlines(vfr[sel], 0, 1, colors=[c], lw=1.2, label=f"{name} ({int(sel.sum())})")
    for r in runs:
        ax.axvspan(scored[r[0]], scored[r[-1]], ymin=0.92, ymax=1.0, color="k")
    amb = pt["ambiguous_frames"]
    if amb:
        ax.plot(amb, np.full(len(amb), 0.5), "kx", ms=6, label=f"ambiguous ({len(amb)})")
    ax.set_yticks([]); ax.set_ylim(0, 1); ax.legend(loc="lower right", ncol=5, fontsize=10); ax.set_xlabel("video frame")
    ax.set_title("actor track per frame (black band at the top = scored stretches)", fontsize=13)
    row = 1
    if gaps:
        tiles = []
        for g in gaps:
            for f in sorted({g["video_first"], (g["video_first"] + g["video_last"]) // 2, g["video_last"]}):
                tiles.append((g, f))
        for j, (g, f) in enumerate(tiles[:4]):
            l = f - off
            img = cv2.cvtColor(cv2.imread(os.path.join(ep_dir, "frames", f"{l:06d}.png")), cv2.COLOR_BGR2RGB)
            m = mask_of(l)
            if m is not None:
                cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnt, -1, (0, 255, 0), 3)
            x0, y0, x1, y1 = np.asarray(pt["per_frame"][l]["box"]).astype(int); cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0), 3)
            ax = fig.add_subplot(gs[row, j]); ax.imshow(img); ax.axis("off")
            ax.set_title(f"gap {g['video_first']}-{g['video_last']} ({g['frames']} fr, {g['scored_frames']} scored): frame {f}, "
                         f"{pt['per_frame'][l]['label']}\ngreen = actor box + person mask", fontsize=10)
        row += 1
    for j, r in enumerate(worst):
        l = r["video_frame"] - off; k = r["video_frame"] - vf[0]
        img = cv2.cvtColor(cv2.imread(os.path.join(ep_dir, "frames", f"{l:06d}.png")), cv2.COLOR_BGR2RGB)
        m = mask_of(l); sil = silhouette(project(fit_v[k], K), faces, hw)
        over = img.copy()
        over[m & ~sil] = (0.4 * over[m & ~sil] + 0.6 * np.array([40, 220, 40])).astype(np.uint8)
        over[sil & ~m] = (0.4 * over[sil & ~m] + 0.6 * np.array([60, 120, 255])).astype(np.uint8)
        over[sil & m] = (0.5 * over[sil & m] + 0.5 * np.array([230, 230, 230])).astype(np.uint8)
        ys, xs = np.nonzero(m | sil)
        y0, y1, x0, x1 = max(0, ys.min() - 40), min(hw[0], ys.max() + 40), max(0, xs.min() - 40), min(hw[1], xs.max() + 40)
        ax = fig.add_subplot(gs[row, j]); ax.imshow(over[y0:y1, x0:x1]); ax.axis("off")
        ax.set_title(f"frame {r['video_frame']}: person covered {100 * r['cover']:.0f}% (IoU {r['iou']:.2f})\n"
                     f"green = person not covered, blue = body not on person, white = both", fontsize=10)
    row += 1
    ax = fig.add_subplot(gs[row, :2])
    fr_ = [r["video_frame"] for r in rows]
    ax.plot(fr_, cov_arr, color="tab:green", lw=1.6, label="visible person covered by the body")
    ax.plot(fr_, pre_arr, color="tab:blue", lw=1.0, label="body pixels on the visible person (occlusion lowers it)")
    ax.plot(fr_, iou_arr, color="tab:purple", lw=0.8, label="IoU")
    ax.set_ylim(0, 1); ax.set_xlabel("video frame"); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower left")
    ax.set_title(f"fitted body vs person mask: covered median {100 * stats['mask_cover_median']:.0f}%, "
                 f"p10 {100 * stats['mask_cover_p10']:.0f}%")
    ax = fig.add_subplot(gs[row, 2:])
    ax.plot(fr_, [r["dz_all_cm"] for r in rows], color="tab:gray", lw=1, label="all body pixels")
    ax.plot(fr_, [r["dz_visible_cm"] for r in rows], color="tab:blue", lw=1.6, label="visible (person mask) only")
    ax.axhline(0, color="k", lw=0.6); ax.legend(fontsize=9); ax.set_xlabel("video frame")
    ax.set_title("body − MoGe depth (cm)")
    row += 1
    if foot is not None:
        ax = fig.add_subplot(gs[row, :2])
        ax.plot(scored, foot["height_cm"], color="tab:brown", lw=1.2)
        ax.axhline(0, color="k", lw=0.8); ax.axhline(foot["base"], color="tab:green", lw=1, ls="--")
        ax.axhspan(foot["base"] - 3, foot["base"] + 3, color="tab:green", alpha=0.08)
        ax.set_xlabel("video frame"); ax.grid(alpha=0.3)
        ax.set_title(f"lowest body point above the floor (cm): median {stats['lowest_point_height_cm_median']:+.1f}; "
                     f"dashed = offset (10th pct), band = touching")
        ax = fig.add_subplot(gs[row, 2:])
        ax.plot(scored, foot["speed_cm"], color="tab:gray", lw=0.8, label="all")
        tc = foot["height_cm"] < foot["base"] + 3
        ax.plot(scored[tc], foot["speed_cm"][tc], ".", color="tab:red", ms=3,
                label="while touching the floor")
        ax.set_yscale("log"); ax.legend(fontsize=9); ax.set_xlabel("video frame")
        ax.set_title("horizontal speed of the lowest point (cm/frame) — ~0 while standing still; spikes when the "
                     "lowest point switches feet (walking, stepping)")
    fig.suptitle(f"{ep} — stage 2 concerns: actor track, image agreement, visible-pixel depth, feet", fontsize=16)
    fig.tight_layout()
    sheet = os.path.join(review_dir, f"{ep}_concerns.png")
    fig.savefig(sheet, dpi=45)
    plt.close(fig)
    return checks, os.path.basename(sheet), stats


def time_checks(ep_dir: str, meta: dict, review_dir: str, K: np.ndarray, hw, boxes: np.ndarray,
                scores: np.ndarray, raw_v: np.ndarray, faces: np.ndarray, fit_v: np.ndarray,
                fit_j: np.ndarray, err: np.ndarray, err_lo: int, vf: np.ndarray, mhr=None) -> tuple[list, str, dict]:
    """Scored-frame checks of the body over time (second sheet, review/<ep>_body_time.png).

    fit_v / fit_j cover video frames vf = first..last; scored frames come from scored_spans.json
    ("frames", with gaps). Body centre = mean of the scorer's 22 ACC-H joints (fitted) or of all
    vertices (raw SAM3D-Body).
    """
    ep = f"episode_{meta['episode']:06d}"
    off = meta["frame_offset"]
    spans = json.load(open(paths.DATA / "submission" / "scored_spans.json"))[str(meta["episode"])]
    scored = np.asarray(spans.get("frames", list(range(spans["first"], spans["last"] + 1))))
    k_sc = scored - vf[0]                                         # index into fit arrays
    runs = runs_of(scored)

    # jitter of the 22 scored joints, per stretch (like the scorer)
    j22 = fit_j[k_sc][:, JOINTS22]
    acc22 = np.concatenate([np.linalg.norm(j22[r][:-2] - 2 * j22[r][1:-1] + j22[r][2:], axis=-1).mean(1) * 100
                            for r in runs if len(r) >= 3] or [np.zeros(0)])

    # body centre depth: raw SAM3D-Body vs fitted, every frame of first..last
    loc = vf - off
    raw_c = raw_v[loc].mean(1)                                    # [T, 3]
    fit_c = fit_j[:, JOINTS22].mean(1)
    raw_c_fit, trend = None, None                  # the fit before stage2_depth.py moved it along the ray
    raw_fit_path = os.path.join(ep_dir, "body", "mhr_fit_raw.npz")
    if mhr is not None and os.path.exists(raw_fit_path):
        rf = np.load(raw_fit_path)
        zz = torch.as_tensor(np.concatenate([rf["scales"], rf["shape"]])[None], dtype=torch.float64, device=mhr.device)
        raw_c_fit = mhr.run(torch.as_tensor(rf["pose"][vf], dtype=torch.float64, device=mhr.device), zz)[1].cpu().numpy()[:, JOINTS22].mean(1)
    tp = os.path.join(ep_dir, "body", "depth_trend.json")
    if os.path.exists(tp):
        trend = json.load(open(tp))
    bh = (boxes[loc, 3] - boxes[loc, 1]).astype(float)
    rel2 = np.full(len(vf), np.nan)
    rel2[1:-1] = np.abs(bh[:-2] - 2 * bh[1:-1] + bh[2:]) / bh[1:-1] * 100
    dz_raw = np.full(len(vf), np.nan)
    dz_raw[1:-1] = np.abs(raw_c[:-2, 2] - 2 * raw_c[1:-1, 2] + raw_c[2:, 2]) * 100
    ok = np.isfinite(rel2) & np.isfinite(dz_raw)
    corr_box_z = float(np.corrcoef(rel2[ok], dz_raw[ok])[0, 1]) if ok.sum() > 10 else float("nan")

    # MoGe / body depth ratio on the body's own pixels, every 2nd scored frame
    rows = []
    for k in k_sc[::2]:
        uv = project(fit_v[k], K)
        b = depth_bias(fit_v[k], uv, load_depth(os.path.join(ep_dir, "depth", f"{vf[k] - off:06d}.png")))
        if b:
            rows.append((vf[k], fit_c[k, 2], b[1]))
    rows = np.array(rows)
    ratio = rows[:, 2]
    from scipy.ndimage import median_filter
    slow = median_filter(ratio, size=min(9, len(ratio) | 1), mode="nearest")    # ~18 frames
    drift_pct = float((np.percentile(slow, 95) - np.percentile(slow, 5)) / np.median(slow) * 100)
    drift_cm = float(drift_pct / 100 * np.median(rows[:, 1]) * 100)
    moge_body_z = rows[:, 1] * rows[:, 2] / np.median(rows[:, 2])            # MoGe depth in body scale
    ratio_depth_corr = float(np.corrcoef(rows[:, 1], ratio)[0, 1]) if len(rows) > 10 and np.ptp(rows[:, 1]) > 0.05 else float("nan")
    ratio_per_m = float(np.polyfit(rows[:, 1], ratio / np.median(ratio), 1)[0] * 100) if len(rows) > 10 else float("nan")

    # first scored frame card
    f0 = int(scored[0]); k0 = f0 - vf[0]; l0 = f0 - off
    nb = np.arange(k0, min(len(vf), k0 + 16))

    def quad_residual(c: np.ndarray) -> float:
        """Distance (cm) of frame k0 from a quadratic through the next 15 frames (motion-aware)."""
        t = nb[1:] - k0
        coef = np.polyfit(t, c[nb[1:]], 2)                                   # [3, 3]
        pred = coef[-1]                                                      # value at t = 0
        return float(np.linalg.norm(c[k0] - pred) * 100)
    root_dev = quad_residual(fit_c) if len(nb) >= 6 else float("nan")
    raw_dev = quad_residual(raw_c) if len(nb) >= 6 else float("nan")
    rel = fit_j[:, JOINTS22] - fit_c[:, None]
    win = slice(max(0, k0 - 7), min(len(vf), k0 + 8))
    pose_dev = float(np.linalg.norm(rel[k0] - np.median(rel[win], axis=0), axis=-1).mean() * 100)
    b0 = depth_bias(fit_v[k0], project(fit_v[k0], K), load_depth(os.path.join(ep_dir, "depth", f"{l0:06d}.png")))
    ratio_dev = float((b0[1] / np.median(ratio) - 1) * 100) if b0 else float("nan")
    e_idx = f0 - err_lo
    err0 = float(err[e_idx]) if 0 <= e_idx < len(err) else float("nan")
    margin = int(l0)                                              # processed frames before the first scored one
    x0, y0, x1, y1 = boxes[l0]
    border = bool(x0 <= 2 or y0 <= 2 or x1 >= hw[1] - 3 or y1 >= hw[0] - 3)
    card = {"frame": f0, "detected": bool(scores[l0] > 0), "detector_score": float(scores[l0]),
            "frames_before": margin, "box_touches_border": border, "fit_error_mm": err0,
            "fit_error_median_mm": float(np.median(err)), "fitted_centre_off_curve_cm": root_dev,
            "raw_centre_off_curve_cm": raw_dev, "pose_vs_neighbours_cm": pose_dev,
            "moge_ratio_vs_episode_pct": ratio_dev}

    stats = {"scored_frames": int(len(scored)), "stretches": len(runs),
             "acc22_mean": float(acc22.mean()) if len(acc22) else 0.0,
             "acc22_p95": float(np.percentile(acc22, 95)) if len(acc22) else 0.0,
             "box_height_jitter_pct_median": float(np.nanmedian(rel2)),
             "raw_depth_accel_cm_median": float(np.nanmedian(dz_raw)),
             "box_jitter_vs_raw_depth_jitter_corr": corr_box_z,
             "body_vs_moge_drift_pct": drift_pct, "body_vs_moge_drift_cm": drift_cm,
             "ratio_vs_depth_corr": ratio_depth_corr, "ratio_change_pct_per_m": ratio_per_m,
             "body_z_range_cm": float(np.ptp(rows[:, 1]) * 100), "moge_z_range_cm": float(np.ptp(moge_body_z) * 100),
             "first_frame": card}
    checks = []
    def add(name, ok_, detail, level="WARN"):
        checks.append({"check": name, "status": "PASS" if ok_ else level, "detail": detail})
    add("first scored frame", card["detected"] and margin >= 5 and root_dev <= 5 and pose_dev <= 5
        and abs(ratio_dev) <= 3 and not border,
        f"video frame {f0}: {'detected' if card['detected'] else 'box INTERPOLATED'}, {margin} frames before it, "
        f"body centre {root_dev:.1f} cm off the curve through the next 15 frames (raw {raw_dev:.1f}), pose {pose_dev:.1f} cm from "
        f"neighbours, MoGe ratio {ratio_dev:+.1f}% vs episode, fit error {err0:.2f} mm"
        f"{', box touches the border' if border else ''}")
    add("body depth drift vs MoGe", drift_pct <= 4,
        f"slow MoGe/body ratio range (5-95%) {drift_pct:.1f}% = {drift_cm:.0f} cm at the body; body depth range "
        f"{stats['body_z_range_cm']:.0f} cm vs MoGe {stats['moge_z_range_cm']:.0f} cm over scored frames; ratio "
        f"changes {ratio_per_m:+.1f}%/m with depth (corr {ratio_depth_corr:.2f})")
    if trend is not None:
        add("body depth follows MoGe", True,
            f"depth factor {trend['factor_scored'][0]:.3f}..{trend['factor_scored'][1]:.3f} on scored frames; body moved "
            f"median {trend['shift_cm_scored']['median']:.1f} / max {trend['shift_cm_scored']['max']:.1f} cm along the "
            f"camera ray; depth range {trend['depth_range_cm_before']:.0f} -> {trend['depth_range_cm_after']:.0f} cm")
    add("scored-joint jitter", stats["acc22_mean"] <= 0.2,
        f"22 scored joints, per stretch: mean {stats['acc22_mean']:.3f}, p95 {stats['acc22_p95']:.3f} cm/frame²")
    add("person box jitter", stats["box_height_jitter_pct_median"] <= 1.0,
        f"box height |2nd diff| median {stats['box_height_jitter_pct_median']:.2f}%; raw body depth |2nd diff| "
        f"median {stats['raw_depth_accel_cm_median']:.2f} cm/frame²; correlation {corr_box_z:.2f}")

    # sheet: first frame crop + time series
    fig = plt.figure(figsize=(22, 15))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 1])
    img = cv2.cvtColor(cv2.imread(os.path.join(ep_dir, "frames", f"{l0:06d}.png")), cv2.COLOR_BGR2RGB)
    sil = silhouette(project(fit_v[k0], K), faces, hw)
    over = img.copy()
    over[sil] = (0.45 * over[sil] + 0.55 * np.array([60, 140, 255])).astype(np.uint8)
    h_ = y1 - y0
    cx0, cy0, cx1, cy1 = int(max(0, x0 - 0.4 * h_)), int(max(0, y0 - 0.15 * h_)), int(min(hw[1], x1 + 0.4 * h_)), int(min(hw[0], y1 + 0.15 * h_))
    cv2.rectangle(over, (int(x0), int(y0)), (int(x1), int(y1)), (255, 230, 0), 2)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(over[cy0:cy1, cx0:cx1]); ax.axis("off")
    ax.set_title(f"FIRST SCORED FRAME {f0} (the scorer aligns everything here)", fontsize=12)
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.text(0, 1, "\n".join(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in card.items())
            + "\n\n(centre = distance from a smooth curve\n through the next 15 frames)",
            va="top", family="monospace", fontsize=12)
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(rows[:, 0], (ratio / np.median(ratio) - 1) * 100, color="tab:gray", lw=0.8, label="per frame")
    ax.plot(rows[:, 0], (slow / np.median(ratio) - 1) * 100, color="tab:red", lw=2, label="slow")
    ax.axhline(0, color="k", lw=0.6); ax.legend(fontsize=9); ax.set_xlabel("video frame")
    ax.set_title(f"MoGe/body depth ratio on body pixels (% vs median): drift {drift_pct:.1f}% = {drift_cm:.0f} cm")

    def shade(ax):
        for r in runs:
            ax.axvspan(scored[r[0]], scored[r[-1]], color="tab:green", alpha=0.07)
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(vf, raw_c[:, 2], color="tab:gray", lw=0.8, label="raw SAM3D-Body (per frame)")
    raw_fit = os.path.join(ep_dir, "body", "mhr_fit_raw.npz")
    if os.path.exists(raw_fit) and raw_c_fit is not None:
        ax.plot(vf, raw_c_fit[:, 2], color="tab:blue", lw=1, ls="--", label="fitted before following MoGe")
    ax.plot(vf, fit_c[:, 2], color="tab:blue", lw=2, label="fitted (submitted)")
    ax.plot(rows[:, 0], moge_body_z, ".", color="tab:red", ms=4, label="MoGe on body pixels, in body scale")
    ax.axvline(f0, color="k", ls="--", lw=1); shade(ax); ax.legend(fontsize=9); ax.set_xlabel("video frame")
    ax.set_title("body centre depth (m) — green = scored stretches, dashed = first scored frame")
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(vf, rel2, color="tab:orange", lw=0.8, label="box height |2nd diff| (%)")
    ax.plot(vf, dz_raw, color="tab:gray", lw=0.8, label="raw depth |2nd diff| (cm/frame²)")
    ax.set_yscale("log"); ax.legend(fontsize=9); shade(ax); ax.set_xlabel("video frame")
    ax.set_title(f"person box jitter drives raw depth jitter? corr {corr_box_z:.2f}")
    ax = fig.add_subplot(gs[2, :])
    xs = np.concatenate([scored[r][1:-1] for r in runs if len(r) >= 3] or [np.zeros(0)])
    ax.plot(xs, acc22, ".", color="tab:purple", ms=3)
    shade(ax); ax.set_xlabel("video frame")
    ax.set_title(f"22 scored joints |2nd diff| per scored frame (cm/frame²): mean {stats['acc22_mean']:.3f} — "
                 f"green = scored stretches; gaps between them are not scored (invalid ground truth)")
    fig.suptitle(f"{ep} — stage 2 body over time (scored frames: {len(scored)} in {len(runs)} stretches)", fontsize=16)
    fig.tight_layout()
    sheet = os.path.join(review_dir, f"{ep}_body_time.png")
    fig.savefig(sheet, dpi=50)
    plt.close(fig)
    return checks, os.path.basename(sheet), stats


def review_episode(ep_dir: str, review_dir: str, mhr: MHR) -> dict:
    meta = json.load(open(os.path.join(ep_dir, "track1_meta.json")))
    ep = f"episode_{meta['episode']:06d}"
    off, first, last = meta["frame_offset"], meta["scored_first"], meta["scored_last"]
    body = os.path.join(ep_dir, "body")
    intr = json.load(open(os.path.join(ep_dir, "intrinsics_stable.json")))
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
    hw = (intr["height"], intr["width"])

    bt = torch.load(os.path.join(body, "bbox_track.pt"), map_location="cpu", weights_only=False)
    boxes, scores = np.asarray(bt["bbox_track"], float), np.asarray(bt["scores"], float)
    params = torch.load(os.path.join(body, "mhr_params.pt"), map_location="cpu", weights_only=False)
    kp2d = params["pred_keypoints_2d"].numpy()
    mesh = torch.load(os.path.join(body, "mhr_mesh.pt"), map_location="cpu", weights_only=False)
    raw_v = mesh["pred_vertices"].numpy()
    faces = mesh["faces"].numpy()
    fit = np.load(os.path.join(body, "mhr_fit.npz"))
    report = json.loads(str(fit["report"]))

    # Fitted body on the scored span (video frames) -> local frames.
    vf = np.arange(first, last + 1)
    loc = vf - off
    pose = torch.as_tensor(fit["pose"][vf], dtype=torch.float64, device=mhr.device)
    z = torch.as_tensor(np.concatenate([fit["scales"], fit["shape"]])[None], dtype=torch.float64,
                        device=mhr.device)
    v_mm, joints = mhr.run(pose, z)
    fit_v = (v_mm / 1000.0).cpu().numpy()
    fit_j = joints.cpu().numpy()
    raw_j = np.stack([raw_v[l] for l in loc])  # raw vertices, for accel comparison (use verts mean per region)
    rest_v = (mhr.run(torch.zeros(1, 136, dtype=torch.float64, device=mhr.device), z)[0] / 1000.0)[0].cpu().numpy()
    height = float(np.ptp(rest_v, axis=0).max())

    # Per-frame measures on every 3rd scored frame.
    depth_rows, inside = [], []
    for k in range(0, len(vf), 3):
        l = loc[k]
        uv = project(fit_v[k], K)
        x0, y0, x1, y1 = boxes[l]
        pad = 0.1 * (y1 - y0)
        inside.append(float(((uv[:, 0] >= x0 - pad) & (uv[:, 0] <= x1 + pad) &
                             (uv[:, 1] >= y0 - pad) & (uv[:, 1] <= y1 + pad)).mean()))
        b = depth_bias(fit_v[k], uv, load_depth(os.path.join(ep_dir, "depth", f"{l:06d}.png")))
        if b:
            depth_rows.append((vf[k], np.median(fit_v[k][:, 2]), *b))
    depth_rows = np.array(depth_rows)
    dz_med = float(np.median(depth_rows[:, 2])) if len(depth_rows) else float("nan")
    ratio_med = float(np.median(depth_rows[:, 3])) if len(depth_rows) else float("nan")

    det_frac = float((scores > 0).mean())
    c = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
    jumps = np.linalg.norm(np.diff(c, axis=0), axis=1) / np.maximum(boxes[1:, 3] - boxes[1:, 1], 1)
    max_jump = float(jumps.max()) if len(jumps) else 0.0
    a_raw, a_fit = accel_cm(raw_j), accel_cm(fit_v)  # both: mean over all body vertices
    err = np.asarray(fit["per_frame_vertex_error_mm"], float)

    checks = []
    def add(name, ok, detail, level="WARN"):
        checks.append({"check": name, "status": "PASS" if ok else level, "detail": detail})
    add("person detected", det_frac >= 0.9, f"{100 * det_frac:.0f}% of frames detected, rest interpolated")
    add("person box continuity", max_jump <= 0.5, f"largest box-centre jump {max_jump:.2f} box heights")
    add("body inside person box", np.mean(inside) >= 0.9, f"{100 * np.mean(inside):.0f}% of body vertices", "FAIL")
    add("body height", 1.45 <= height <= 2.05, f"{height:.2f} m (fitted identity, rest pose)")
    add("body vs MoGe depth", abs(dz_med) <= 15,
        f"body is {dz_med:+.1f} cm vs MoGe on its own pixels (MoGe/body depth ratio {ratio_med:.3f})")
    add("body jitter", np.nanmean(a_fit) <= 0.3,
        f"fitted {np.nanmean(a_fit):.3f} vs raw {np.nanmean(a_raw):.3f} cm/frame² (mean over vertices)")
    add("fit error", report["vertex_error_mm"]["mean"] <= 3,
        f"mean {report['vertex_error_mm']['mean']:.2f} mm, max {report['vertex_error_mm']['max']:.1f} mm")
    os.makedirs(review_dir, exist_ok=True)
    try:
        tc, time_sheet, time_stats = time_checks(ep_dir, meta, review_dir, K, hw, boxes, scores, raw_v, faces,
                                                 fit_v, fit_j, err, int(fit["fit_video_lo"]), vf, mhr)
        checks += tc
    except Exception as exc:                                    # a broken check must not hide the rest
        import traceback; traceback.print_exc()
        checks.append({"check": "time checks", "status": "WARN", "detail": f"could not run: {exc!r}"})
        time_sheet, time_stats = None, None
    try:
        cc, conc_sheet, conc_stats = concerns_sheet(ep_dir, meta, review_dir, K, hw, fit_v, faces, vf)
        checks += cc
    except Exception as exc:
        import traceback; traceback.print_exc()
        checks.append({"check": "concerns sheet", "status": "WARN", "detail": f"could not run: {exc!r}"})
        conc_sheet, conc_stats = None, None

    # ------------------------------------------------------------------ sheet
    sample = [int(round(i)) for i in np.linspace(0, len(vf) - 1, 6)]
    fig = plt.figure(figsize=(22, 24))
    gs = fig.add_gridspec(4, 6, height_ratios=[1.25, 1.1, 1.1, 0.9])
    l_mid = loc[len(loc) // 2]
    full = cv2.cvtColor(cv2.imread(os.path.join(ep_dir, "frames", f"{l_mid:06d}.png")), cv2.COLOR_BGR2RGB)
    x0, y0, x1, y1 = boxes[l_mid].astype(int)
    cv2.rectangle(full, (x0, y0), (x1, y1), (255, 230, 0), 5)
    ax = fig.add_subplot(gs[0, 1:5]); ax.imshow(full); ax.axis("off")
    ax.set_title(f"tracked person (yellow) at video frame {l_mid + off} — is this the actor?", fontsize=14)

    for j, k in enumerate(sample):
        l = loc[k]
        img = cv2.cvtColor(cv2.imread(os.path.join(ep_dir, "frames", f"{l:06d}.png")), cv2.COLOR_BGR2RGB)
        uv = project(fit_v[k], K)
        sil = silhouette(uv, faces, hw)
        over = img.copy()
        over[sil] = (0.45 * over[sil] + 0.55 * np.array([60, 140, 255])).astype(np.uint8)
        cnts, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cnts, -1, (30, 90, 255), 2)
        bx0, by0, bx1, by1 = boxes[l]
        cv2.rectangle(over, (int(bx0), int(by0)), (int(bx1), int(by1)), (255, 230, 0), 2)
        for idx, col in ((R_WRIST_2D, (255, 40, 40)), (L_WRIST_2D, (40, 220, 40))):
            cv2.circle(over, tuple(np.round(kp2d[l, idx]).astype(int)), 6, col, -1)
        bh = by1 - by0
        cx0, cy0 = int(max(0, bx0 - 0.4 * bh)), int(max(0, by0 - 0.15 * bh))
        cx1, cy1 = int(min(hw[1], bx1 + 0.4 * bh)), int(min(hw[0], by1 + 0.15 * bh))
        ax = fig.add_subplot(gs[1 + j // 3, (j % 3) * 2:(j % 3) * 2 + 2])
        ax.imshow(over[cy0:cy1, cx0:cx1]); ax.axis("off")
        tag = "" if scores[l] > 0 else "  (box interpolated)"
        ax.set_title(f"video frame {l + off}{tag}", fontsize=12)

    ax = fig.add_subplot(gs[3, 0:2])
    sp = json.load(open(paths.DATA / "submission" / "scored_spans.json"))[str(meta["episode"])]
    for r in runs_of(np.asarray(sp["frames"])):
        ax.axvspan(sp["frames"][r[0]], sp["frames"][r[-1]], color="tab:green", alpha=0.08)
    if len(depth_rows):
        ax.plot(depth_rows[:, 0], (depth_rows[:, 3] / ratio_med - 1) * 100, color="tab:blue", lw=1.8)
        ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"MoGe/body depth ratio vs its median (%), median ratio {ratio_med:.3f}\n"
                 f"flat = consistent; green = scored stretches")
    ax.set_xlabel("video frame")
    ax = fig.add_subplot(gs[3, 2:4])
    ax.plot(vf, a_raw, color="gray", lw=0.8, label=f"raw SAM3D-Body (mean {np.nanmean(a_raw):.2f})")
    ax.plot(vf, a_fit, color="tab:orange", lw=1.6, label=f"fitted (mean {np.nanmean(a_fit):.2f})")
    ax.set_yscale("log"); ax.legend(fontsize=9)
    ax.set_title("body |2nd difference| (cm/frame²)"); ax.set_xlabel("video frame")
    ax = fig.add_subplot(gs[3, 4:6])
    lo_v = int(fit["fit_video_lo"])
    ax.plot(np.arange(lo_v, lo_v + len(err)), err, color="tab:purple", lw=1.2, label="fit error (mm)")
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(scores)) + off, scores, color="tab:green", lw=0.8, alpha=0.7, label="person score")
    ax2.set_ylim(-1.1, 1.1)
    ax.set_title(f"fit error (mm, purple) / detector score (green, −1 = interpolated)")
    ax.set_xlabel("video frame")

    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else \
             "WARN" if any(c["status"] == "WARN" for c in checks) else "PASS"
    fig.suptitle(f"{ep} — stage 2 person + body — {status} — height {height:.2f} m, "
                 f"body−MoGe {dz_med:+.1f} cm", fontsize=17)
    fig.tight_layout()
    os.makedirs(review_dir, exist_ok=True)
    sheet = os.path.join(review_dir, f"{ep}_body.png")
    fig.savefig(sheet, dpi=50)
    plt.close(fig)
    return {"episode": meta["episode"], "status": status, "checks": checks, "sheet": os.path.basename(sheet),
            "sheet_time": time_sheet, "time": time_stats, "sheet_concerns": conc_sheet, "concerns": conc_stats,
            "version": REVIEW_VERSION,
            "height_m": height, "body_minus_moge_cm": dz_med, "moge_over_body_ratio": ratio_med,
            "seconds": meta.get("stage2_seconds")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--stage_dir", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    mhr = MHR(args.model, "cuda" if torch.cuda.is_available() else "cpu", chunk=32, precision="float32")
    review = os.path.join(args.stage_dir, "review")
    results = []
    for e in args.episodes:
        ep_dir = os.path.join(args.out_root, f"episode_{e:06d}")
        if not os.path.exists(os.path.join(ep_dir, "body", "mhr_fit.npz")):
            for old in glob.glob(os.path.join(review, f"episode_{e:06d}_*.png")):   # stale sheet removed
                if not any(t in os.path.basename(old) for t in ("_before_", "_v2_", "_v3_", "_old", "_refine")):
                    os.remove(old)
            results.append({"episode": e, "status": "FAIL", "checks": [{"check": "stage 2 output",
                            "status": "FAIL", "detail": "missing (see the episode log)"}]})
            continue
        chk_path = os.path.join(args.stage_dir, f"episode_{e:06d}.checks.json")
        key_out = os.path.join(ep_dir, "body", "mhr_fit.npz")
        newest = max(os.path.getmtime(p) for p in (key_out, os.path.join(ep_dir, "depth_scale.json"))
                     if os.path.exists(p))
        reuse_ok = (os.path.exists(chk_path) and os.path.exists(key_out)
                    and os.path.getmtime(chk_path) > newest
                    and json.load(open(chk_path)).get("version") == REVIEW_VERSION
                    and os.path.exists(os.path.join(review, json.load(open(chk_path)).get("sheet", "-"))))
        if reuse_ok:                      # sheet already up to date: reuse it
            results.append(json.load(open(chk_path)))
            continue
        r = review_episode(ep_dir, review, mhr)
        json.dump(r, open(os.path.join(args.stage_dir, f"episode_{e:06d}.checks.json"), "w"), indent=1)
        results.append(r)
        print(f"episode {e:3d}: {r['status']:4s}  " + "; ".join(
            f"{c['check']}: {c['detail']}" for c in r["checks"] if c["status"] != "PASS"))

    color = {"PASS": "#4c4", "WARN": "#fb3", "FAIL": "#f55"}
    out = ["<html><head><meta charset='utf-8'><title>Stage 2 body</title><style>body{font-family:sans-serif;"
           "background:#111;color:#eee} td,th{border:1px solid #444;padding:3px 7px} table{border-collapse:"
           "collapse} img{max-width:100%}</style></head><body><h1>Stage 2 — person + body</h1><p>Check: the "
           "yellow box is the actor; the blue body covers the person in every crop (arms, legs, posture); "
           "wrists (red = right, green = left) sit on the hands.</p><table><tr><th>episode</th><th>status</th>"
           "<th>checks</th><th>seconds per step</th></tr>"]
    for r in results:
        det = "<br>".join(f"<span style='color:{color[c['status']]}'>{c['status']}</span> "
                          f"{html.escape(c['check'])}: {html.escape(c['detail'])}" for c in r["checks"])
        out.append(f"<tr><td><a href='#e{r['episode']}'>{r['episode']}</a></td><td style='color:"
                   f"{color[r['status']]}'>{r['status']}</td><td>{det}</td><td>{html.escape(str(r.get('seconds')))}"
                   f"</td></tr>")
    out.append("</table>")
    for r in results:
        out.append(f"<h2 id='e{r['episode']}'>episode {r['episode']} — {r['status']}</h2>"
                   f"<p>logs: <a href='episode_{r['episode']:06d}.log'>progress</a> · "
                   f"<a href='episode_{r['episode']:06d}.debug.log'>debug</a></p>")
        if r.get("sheet"):
            made = __import__("time").strftime("%Y-%m-%d %H:%M:%S", __import__("time").localtime(
                os.path.getmtime(os.path.join(args.stage_dir, "review", r["sheet"]))))
            out.append(f"<p style='color:#999'>sheet made at {made}</p>")
            out.append(f"<a href='review/{r['sheet']}'><img src='review/{r['sheet']}'></a>")
            if r.get("sheet_time"):
                out.append(f"<a href='review/{r['sheet_time']}'><img src='review/{r['sheet_time']}'></a>")
            if r.get("sheet_concerns"):
                out.append(f"<a href='review/{r['sheet_concerns']}'><img src='review/{r['sheet_concerns']}'></a>")
    out.append("</body></html>")
    open(os.path.join(args.stage_dir, "index.html"), "w", encoding="utf-8").write("\n".join(out))
    print(f"wrote {os.path.join(args.stage_dir, 'index.html')}")


if __name__ == "__main__":
    main()
