#!/usr/bin/env python3
"""Stage 4 review (object mask): one sheet per episode + checks + index.html.

Measured on EVERY frame of the scored span:
  area       mask pixels (0 = object lost)
  size_m     largest side of the mask box in metres (px * median mask depth / fx)
  dz_m       median MoGe depth inside the mask minus the actor's depth
  wrist_gap  nearest SAM3D-Body wrist to the mask box, in actor heights (contact indicator)
  det_iou    on the 5 stage-3 frames: IoU between the mask box and the accepted detection box

Sheet (review/episode_XXXXXX_mask.png):
  6 evenly spaced frames + the 2 most suspicious frames (smallest area, largest size), cropped
  around actor + object: mask in green with outline, actor box yellow, wrists R red / L green
  plots: area, size vs the prior band, depth vs actor, wrist gap — the scored span shaded

Checks:
  masks present on every local frame                                           FAIL
  empty masks <= 5 % of scored frames                                         WARN
  size inside the prior (with slack) on >= 90 % of non-empty scored frames    WARN  (leak / shrink)
  |dz| <= 0.6 m on >= 90 % of non-empty scored frames                         WARN  (leak to background)
  abrupt area changes (x2 between consecutive frames) <= 2 % of frames        WARN
  mask box agrees with the stage-3 detections (median IoU >= 0.5)             WARN

    /opt/venvs/sam3d_body/bin/python track1/review_stage4.py --episodes 6 8 17 26 28 \
        --out_root out --stage_dir logs/stages/04_mask
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import paths  # noqa: E402
from select_object import box_iou  # noqa: E402

PRIORS = json.load(open(paths.PRIORS))
META = str(paths.META)
R_WRIST, L_WRIST = 41, 62
SLACK = 1.6


def load_depth(path: str) -> np.ndarray:
    px = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return 1.0 / np.maximum(px / 65535.0, 1e-6) - 1.0


def point_box_gap(p, b) -> float:
    dx = max(b[0] - p[0], p[0] - b[2], 0.0)
    dy = max(b[1] - p[1], p[1] - b[3], 0.0)
    return float(np.hypot(dx, dy))


def review_episode(W: str, review_dir: str) -> dict:
    meta = json.load(open(os.path.join(W, "track1_meta.json")))
    e = meta["episode"]
    ep = f"episode_{e:06d}"
    off, first, last, n = meta["frame_offset"], meta["scored_first"], meta["scored_last"], meta["n_frames"]
    intr = json.load(open(os.path.join(W, "intrinsics_stable.json")))
    fx = intr["fx"]
    track = json.load(open(os.path.join(W, "object_track.json")))
    obj = next(json.loads(l) for l in open(META) if json.loads(l)["episode_index"] == e)["object"]
    lo_m, hi_m = PRIORS.get(obj, {}).get("size_m", [0.05, 3.0])
    lo_m, hi_m = lo_m / SLACK, hi_m * SLACK
    boxes = np.asarray(torch.load(os.path.join(W, "body", "bbox_track.pt"), map_location="cpu",
                                  weights_only=False)["bbox_track"], float)
    kp = torch.load(os.path.join(W, "body", "mhr_params.pt"), map_location="cpu",
                    weights_only=False)["pred_keypoints_2d"].numpy()
    mdir = os.path.join(W, "masks", "1")
    n_masks = len([f for f in os.listdir(mdir) if f.endswith(".png")]) if os.path.isdir(mdir) else 0

    loc = np.arange(first - off, last - off + 1)
    rows = []
    for l in loc:
        m = cv2.imread(os.path.join(mdir, f"{l:06d}.png"), cv2.IMREAD_GRAYSCALE)
        m = m > 0 if m is not None else None
        pb = boxes[l]
        ph = max(pb[3] - pb[1], 1.0)
        d = load_depth(os.path.join(W, "depth", f"{l:06d}.png"))
        core = d[int(pb[1] + 0.15 * ph):int(pb[3] - 0.15 * ph),
                 int(pb[0] + 0.25 * (pb[2] - pb[0])):int(pb[2] - 0.25 * (pb[2] - pb[0]))]
        az = float(np.median(core)) if core.size else np.nan
        if m is None or not m.any():
            rows.append((l + off, 0, np.nan, np.nan, np.nan, None))
            continue
        ys, xs = np.nonzero(m)
        mb = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        oz = float(np.median(d[m]))
        size = max(mb[2] - mb[0], mb[3] - mb[1]) * oz / fx
        wg = min(point_box_gap(kp[l, R_WRIST], mb), point_box_gap(kp[l, L_WRIST], mb)) / ph
        rows.append((l + off, int(m.sum()), size, oz - az, wg, mb))
    vf = np.array([r[0] for r in rows])
    area = np.array([r[1] for r in rows], float)
    size = np.array([r[2] for r in rows], float)
    dz = np.array([r[3] for r in rows], float)
    wgap = np.array([r[4] for r in rows], float)
    ok = area > 0

    det_ious = []
    for k, v in track.get("per_frame_boxes", {}).items():
        r = rows[int(k) - (first - off)] if first - off <= int(k) <= last - off else None
        if r and r[5]:
            mb = r[5]
            det_ious.append(box_iou({"x0": mb[0], "y0": mb[1], "x1": mb[2], "y1": mb[3]}, v["box"]))
    # Sudden lasting drop: the smoothed area falls by >35 % within 5 frames and stays down for 15.
    from scipy.ndimage import median_filter
    sm = median_filter(area, size=9, mode="nearest")
    drop_at = None
    for i in range(len(sm) - 20):
        if sm[i] > 0 and sm[i + 5] < 0.65 * sm[i] and sm[i + 5:i + 20].mean() < 0.7 * sm[i]:
            drop_at = (int(vf[i]), float(sm[i]), float(sm[i + 5:i + 20].mean()))
            break
    seed = json.load(open(os.path.join(W, "masks_seed.json"))) if os.path.exists(os.path.join(W, "masks_seed.json")) else {}
    prompt_vf = [p["frame"] + off for p in seed.get("prompts", [])] or [track["reference_frame"] + off]
    ratio = area[1:] / np.maximum(area[:-1], 1)
    abrupt = float(np.mean((ok[1:] & ok[:-1]) & ((ratio > 2) | (ratio < 0.5)))) if len(ratio) else 0.0

    checks = []
    def add(name, good, detail, level="WARN"):
        checks.append({"check": name, "status": "PASS" if good else level, "detail": detail})
    add("masks present", n_masks == n, f"{n_masks} mask files, expected {n}", "FAIL")
    add("object visible", (~ok).mean() <= 0.05, f"{100 * (~ok).mean():.1f}% of scored frames have an empty mask")
    in_prior = (size[ok] >= lo_m) & (size[ok] <= hi_m)
    add("size plausible", in_prior.mean() >= 0.9,
        f"{100 * in_prior.mean():.0f}% of frames inside {lo_m:.2f}-{hi_m:.2f} m (median {np.nanmedian(size):.2f} m, "
        f"max {np.nanmax(size):.2f} m)")
    near = np.abs(dz[ok]) <= 0.6
    add("at actor depth", near.mean() >= 0.9,
        f"{100 * near.mean():.0f}% of frames within 0.6 m (median {np.nanmedian(dz):+.2f} m)")
    add("stable area", abrupt <= 0.02, f"{100 * abrupt:.1f}% of frame pairs change area by >2x")
    add("no lasting area drop", drop_at is None,
        "none" if drop_at is None else f"area drops from {drop_at[1]:.0f} to ~{drop_at[2]:.0f} px at video frame "
        f"{drop_at[0]} and stays down (part of the object lost, or the person covers it — look at that frame)")
    if det_ious:
        add("agrees with detections", np.median(det_ious) >= 0.5,
            f"median IoU {np.median(det_ious):.2f} with the stage-3 boxes on {len(det_ious)} frames")
    rp = os.path.join(W, "masks_repair.json")
    if os.path.exists(rp):
        rj = json.load(open(rp))
        checks.append({"check": "leak repair", "status": "PASS",
                       "detail": f"cut {len(rj['leak_cut_frames'])} leak frames {rj['leak_cut_frames'][:8]}, cleared "
                                 f"{len(rj['cleared_frames'])} stray blobs {rj['cleared_frames'][:8]}"})
    mpj = os.path.join(W, "mask_prompts.json")
    if os.path.exists(mpj):
        pl = json.load(open(mpj))["prompts"]
        checks.append({"check": "SAM2 prompts", "status": "PASS",
                       "detail": f"{len(pl)} boxes ({sum(p['source'] != 'stage3' for p in pl)} gap-fill) with "
                                 f"{sum(len(p.get('neg', [])) for p in pl)} negative body points"})
    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else \
             "WARN" if any(c["status"] == "WARN" for c in checks) else "PASS"

    # ---------------------------------------------------------------- sheet
    show = [int(round(i)) for i in np.linspace(0, len(rows) - 1, 6)]
    tags = {k: "" for k in show}
    if ok.any():
        k_small = int(np.argmin(np.where(ok, area, np.inf)))
        k_big = int(np.nanargmax(np.where(ok, size, -np.inf)))
        for k, t in ((k_small, "smallest mask"), (k_big, "largest size")):
            tags[k] = (tags.get(k, "") + " " + t).strip()
    if (~ok).any():
        tags[int(np.argmax(~ok))] = "EMPTY"
    if drop_at is not None:
        k = int(np.searchsorted(vf, drop_at[0] + 8))
        tags[min(k, len(rows) - 1)] = (tags.get(min(k, len(rows) - 1), "") + " after area drop").strip()
    show = sorted(tags)[:9]
    fig = plt.figure(figsize=(24, 11 + 5.2 * ((len(show) + 2) // 3)))
    nrow = (len(show) + 2) // 3
    gs = fig.add_gridspec(nrow + 2, 3, height_ratios=[1.0] * nrow + [0.55, 0.55])
    for j, k in enumerate(show):
        l = loc[k]
        img = cv2.cvtColor(cv2.imread(os.path.join(W, "frames", f"{l:06d}.png")), cv2.COLOR_BGR2RGB)
        m = cv2.imread(os.path.join(mdir, f"{l:06d}.png"), cv2.IMREAD_GRAYSCALE)
        m = (m > 0) if m is not None else np.zeros(img.shape[:2], bool)
        img[m] = (0.45 * img[m] + 0.55 * np.array([40, 230, 60])).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
        pb = boxes[l].astype(int)
        cv2.rectangle(img, (pb[0], pb[1]), (pb[2], pb[3]), (255, 220, 0), 2)
        for idx, col in ((R_WRIST, (255, 40, 40)), (L_WRIST, (40, 220, 40))):
            cv2.circle(img, tuple(np.round(kp[l, idx]).astype(int)), 6, col, -1)
        xs = [pb[0], pb[2]] + ([rows[k][5][0], rows[k][5][2]] if rows[k][5] else [])
        ys = [pb[1], pb[3]] + ([rows[k][5][1], rows[k][5][3]] if rows[k][5] else [])
        ph = pb[3] - pb[1]
        x0, x1 = int(max(0, min(xs) - 0.35 * ph)), int(min(img.shape[1], max(xs) + 0.35 * ph))
        y0, y1 = int(max(0, min(ys) - 0.15 * ph)), int(min(img.shape[0], max(ys) + 0.15 * ph))
        ax = fig.add_subplot(gs[j // 3, j % 3])
        ax.imshow(img[y0:y1, x0:x1]); ax.axis("off")
        r = rows[k]
        sz = "-" if np.isnan(r[2]) else f"{r[2]:.2f} m"
        title = f"video frame {r[0]}  area {r[1]} px  size {sz}"
        if tags.get(k):
            title += f"  [{tags[k]}]"
        ax.set_title(title, fontsize=13, color="red" if tags.get(k) else "black")
    ax = fig.add_subplot(gs[nrow, 0:2])
    ax.plot(vf, area, color="tab:green", lw=1.5)
    for pv in prompt_vf:
        ax.axvline(pv, color="tab:orange", ls="--", lw=1.2)
    if drop_at is not None:
        ax.axvline(drop_at[0], color="red", lw=2)
    ax.set_title("mask area (px) — orange dashed = SAM2 prompt frames, red = lasting drop")
    ax.set_xlabel("video frame")
    ax = fig.add_subplot(gs[nrow, 2])
    ax.plot(vf, size, color="tab:blue", lw=1.5)
    ax.axhspan(lo_m, hi_m, color="tab:green", alpha=0.12, label=f"plausible {lo_m:.2f}-{hi_m:.2f} m")
    ax.set_title("mask size (m)"); ax.legend(fontsize=9); ax.set_xlabel("video frame")
    ax = fig.add_subplot(gs[nrow + 1, 0:2])
    ax.plot(vf, dz, color="tab:purple", lw=1.5)
    ax.axhspan(-0.6, 0.6, color="tab:green", alpha=0.12)
    ax.set_title("mask depth − actor depth (m) — far outside the band = mask leaked to the background")
    ax.set_xlabel("video frame")
    ax = fig.add_subplot(gs[nrow + 1, 2])
    ax.plot(vf, wgap, color="tab:red", lw=1.5)
    ax.set_title("nearest wrist to mask (actor heights) — 0 = touching"); ax.set_xlabel("video frame")
    fig.suptitle(f"{ep} — stage 4 object mask — {status} — SAM2 prompts at video frames "
                 f"{', '.join(map(str, prompt_vf))}", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(review_dir, exist_ok=True)
    sheet = os.path.join(review_dir, f"{ep}_mask.png")
    fig.savefig(sheet, dpi=50)
    plt.close(fig)
    return {"episode": e, "status": status, "checks": checks, "sheet": os.path.basename(sheet)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--stage_dir", required=True)
    args = ap.parse_args()
    review = os.path.join(args.stage_dir, "review")
    results = []
    for e in args.episodes:
        W = os.path.join(args.out_root, f"episode_{e:06d}")
        if not os.path.isdir(os.path.join(W, "masks", "1")):
            for old in glob.glob(os.path.join(review, f"episode_{e:06d}_*.png")):   # stale sheet removed
                if not any(t in os.path.basename(old) for t in ("_before_", "_v2_", "_v3_", "_old", "_refine")):
                    os.remove(old)
            results.append({"episode": e, "status": "FAIL", "checks": [{"check": "stage 4 output", "status": "FAIL",
                            "detail": "missing (see the episode log)"}]})
            continue
        chk_path = os.path.join(args.stage_dir, f"episode_{e:06d}.checks.json")
        key_out = os.path.join(W, "masks_seed.json")
        reuse_ok = (os.path.exists(chk_path) and os.path.exists(key_out)
                    and os.path.getmtime(chk_path) > os.path.getmtime(key_out)
                    and os.path.exists(os.path.join(review, json.load(open(chk_path)).get("sheet", "-"))))
        if reuse_ok:                      # sheet already up to date: reuse it
            results.append(json.load(open(chk_path)))
            continue
        r = review_episode(W, review)
        json.dump(r, open(os.path.join(args.stage_dir, f"episode_{e:06d}.checks.json"), "w"), indent=1)
        results.append(r)
        print(f"episode {e:3d}: {r['status']:4s}  " + "; ".join(
            f"{c['check']}: {c['detail']}" for c in r["checks"] if c["status"] != "PASS"))
    color = {"PASS": "#4c4", "WARN": "#fb3", "FAIL": "#f55"}
    out = ["<html><head><meta charset='utf-8'><title>Stage 4 object mask</title><style>body{font-family:"
           "sans-serif;background:#111;color:#eee} td,th{border:1px solid #444;padding:3px 7px} table{border-"
           "collapse:collapse} img{max-width:100%}</style></head><body><h1>Stage 4 — object mask (SAM2)</h1>"
           "<p>Check: green covers the whole object and nothing else, on every shown frame (red titles = the most "
           "suspicious frames). Plots: area should not drop to 0 while the object is visible; size stays in the "
           "band; depth stays near the actor.</p><table><tr><th>episode</th><th>status</th><th>checks</th></tr>"]
    for r in results:
        det = "<br>".join(f"<span style='color:{color[c['status']]}'>{c['status']}</span> "
                          f"{html.escape(c['check'])}: {html.escape(c['detail'])}" for c in r["checks"])
        out.append(f"<tr><td><a href='#e{r['episode']}'>{r['episode']}</a></td><td style='color:"
                   f"{color[r['status']]}'>{r['status']}</td><td>{det}</td></tr>")
    out.append("</table>")
    for r in results:
        out.append(f"<h2 id='e{r['episode']}'>episode {r['episode']} — {r['status']}</h2><p>logs: "
                   f"<a href='episode_{r['episode']:06d}.log'>progress</a> · "
                   f"<a href='episode_{r['episode']:06d}.debug.log'>debug</a></p>")
        if r.get("sheet"):
            made = __import__("time").strftime("%Y-%m-%d %H:%M:%S", __import__("time").localtime(
                os.path.getmtime(os.path.join(args.stage_dir, "review", r["sheet"]))))
            out.append(f"<p style='color:#999'>sheet made at {made}</p>")
            out.append(f"<a href='review/{r['sheet']}'><img src='review/{r['sheet']}'></a>")
    out.append("</body></html>")
    open(os.path.join(args.stage_dir, "index.html"), "w", encoding="utf-8").write("\n".join(out))
    print(f"wrote {os.path.join(args.stage_dir, 'index.html')}")


if __name__ == "__main__":
    main()
