#!/usr/bin/env python3
"""Stage 1 review: one sheet per episode + checks + an index.html for all episodes.

Per episode (review/episode_XXXXXX_prep.png):
  rows    first scored, middle and last scored frame
  columns original video frame | undistorted frame | MoGe depth (colour, metres)
  bottom  per-frame depth percentiles over the whole kept window: the camera is static,
          so the background depth should be flat; drift = MoGe scale changing over time

Checks (episode_XXXXXX.checks.json, shown in index.html):
  frames / depth maps == expected count                         FAIL if not
  focal length fx / width in [0.45, 0.80] (wide exo cameras)    WARN if not
  principal point at the image centre                           WARN if not
  MoGe fx vs AnyCalib undistorted fx within 10%                 WARN if not
  median scene depth 0.5 .. 15 m on every frame                 FAIL if not
  scene-depth drift: robust std / median <= 3 %, max frame-to-frame jump <= 5 %   WARN if not
  (background_checks, second sheet review/episode_XXXXXX_background.png — thresholds provisional)
  depth local warping on static background: median <= 1.5 %, worst scored frame <= 4 %   WARN if not
  floor plane: RMS <= 2 cm, near/far bend <= 5 deg, >= 30 % of lower background on it   WARN if not
  undistortion: long straight edges in the outer ring not shorter after undistortion (>= x0.9)   WARN if not

Run with the sam3d_body venv (cv2, matplotlib):

    /opt/venvs/sam3d_body/bin/python track1/review_stage1.py --episodes 6 8 17 26 28 \
        --out_root out --stage_dir logs/stages/01_prep
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
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import paths  # noqa: E402

REVIEW_VERSION = 2          # bump when checks change: older checks.json files are recomputed


def load_depth(path: str) -> np.ndarray:
    px = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    d = 1.0 / np.maximum(px / 65535.0, 1e-6) - 1.0
    d[px == 0] = np.nan
    return d


def source_frame(video: str, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else None


def long_segments(rgb: np.ndarray, min_len: float) -> np.ndarray:
    """Line segments (x0, y0, x1, y1) at least min_len pixels long (OpenCV LSD)."""
    seg = cv2.createLineSegmentDetector().detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))[0]
    if seg is None:
        return np.zeros((0, 4))
    seg = seg.reshape(-1, 4)
    return seg[np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) >= min_len]


def outer_length(seg: np.ndarray, w: int, h: int, frac: float = 0.55) -> float:
    """Total length of the segment parts whose midpoint lies in the outer image ring (where distortion bends lines)."""
    mid = np.stack([(seg[:, 0] + seg[:, 2]) / 2 - w / 2, (seg[:, 1] + seg[:, 3]) / 2 - h / 2], 1)
    outer = np.hypot(mid[:, 0], mid[:, 1]) > frac * np.hypot(w / 2, h / 2)
    return float(np.hypot(seg[outer, 2] - seg[outer, 0], seg[outer, 3] - seg[outer, 1]).sum())


def background_checks(ep_dir: str, meta: dict, intr: dict, review_dir: str, ds: int = 4) -> tuple[list, str, dict]:
    """Static-scene checks of the depth maps and the undistortion (stage-1 outputs only).

    Moving pixels (person, object) = colour differs from the episode's median background image.
    * depth stability: per pixel, log(depth / background depth) on static pixels; per frame the
      median is the global scale change, the median absolute rest is LOCAL warping
    * floor: plane fitted (RANSAC, 3 cm) to the background depth in the lower image; flatness (RMS),
      bend (angle between planes fitted to the near and the far half), camera height and pitch
    * undistortion: long straight segments (LSD, >= 120 px) in the outer image ring, original vs
      undistorted frame: removing lens bending should make long straight edges longer, not shorter
    """
    ep = f"episode_{meta['episode']:06d}"
    off, n = meta["frame_offset"], meta["n_frames"]
    H, W = intr["height"] // ds, intr["width"] // ds
    rgb = np.zeros((n, H, W, 3), np.uint8)
    logd = np.full((n, H, W), np.nan, np.float32)
    for i in range(n):
        im = cv2.imread(os.path.join(ep_dir, "frames", f"{i:06d}.png"))
        rgb[i] = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        d = load_depth(os.path.join(ep_dir, "depth", f"{i:06d}.png"))
        logd[i] = np.log(cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST))
    bg = np.median(rgb[::max(1, n // 150)], axis=0).astype(np.int16)
    moving = np.abs(rgb.astype(np.int16) - bg[None]).max(-1) > 20
    kernel = np.ones((7, 7), np.uint8)
    static = np.stack([~cv2.dilate(m.astype(np.uint8), kernel).astype(bool) for m in moving]) & np.isfinite(logd)
    ref = np.nanmedian(np.where(static, logd, np.nan), axis=0)                 # background log-depth
    static_frac = static.mean(0)
    lr = np.where(static, logd - ref[None], np.nan)
    scale = np.nanmedian(lr.reshape(n, -1), axis=1)                            # global, per frame
    rest = np.abs(lr - scale[:, None, None])
    local = np.nanmedian(rest.reshape(n, -1), axis=1)                          # local warping, per frame
    pix_instab = np.nanmedian(rest, axis=0)
    pix_instab[static_frac < 0.3] = np.nan

    # floor: background depth -> points; plane on the lower image
    D = np.exp(ref)
    fx, fy, cx, cy = intr["fx"] / ds, intr["fy"] / ds, intr["cx"] / ds, intr["cy"] / ds
    v, u = np.mgrid[0:H, 0:W]
    P = np.stack([(u - cx) / fx * D, (v - cy) / fy * D, D], -1)
    cand = (v > 0.55 * H) & np.isfinite(D) & (static_frac >= 0.3)
    pts = P[cand]
    rng = np.random.default_rng(0)
    best = (0, None)
    for _ in range(400):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        if np.linalg.norm(nrm) < 1e-9:
            continue
        nrm /= np.linalg.norm(nrm)
        k = int((np.abs((pts - a) @ nrm) < 0.03).sum())
        if k > best[0]:
            best = (k, (a, nrm))
    a, nrm = best[1]
    inl = np.abs((pts - a) @ nrm) < 0.03

    def fit(q):
        c0 = q.mean(0)
        nn = np.linalg.svd(q - c0)[2][-1]
        return c0, nn * np.sign(nn[1] if nn[1] != 0 else 1)
    c0, nrm = fit(pts[inl])
    resid = (pts[inl] - c0) @ nrm
    zmed = np.median(pts[inl][:, 2])
    near, far = pts[inl][pts[inl][:, 2] < zmed], pts[inl][pts[inl][:, 2] >= zmed]
    bend = float(np.degrees(np.arccos(np.clip(abs(fit(near)[1] @ fit(far)[1]), -1, 1))))
    cam_h = float(abs(c0 @ nrm))
    pitch = float(np.degrees(np.arccos(abs(nrm[1]))))
    floor_frac = float(inl.mean())
    resid_map = np.full((H, W), np.nan)
    resid_map[cand] = (pts - c0) @ nrm
    resid_map[np.abs(resid_map) > 0.15] = np.nan

    # undistortion: long straight segments in the outer ring, original vs undistorted
    i0 = meta["scored_first"] - off
    orig = source_frame(meta["source_video"], i0 + off)
    und = cv2.cvtColor(cv2.imread(os.path.join(ep_dir, "frames", f"{i0:06d}.png")), cv2.COLOR_BGR2RGB)
    so, su = long_segments(orig, 120), long_segments(und, 120)
    lo_, lu_ = outer_length(so, intr["width"], intr["height"]), outer_length(su, intr["width"], intr["height"])
    line_ratio = lu_ / max(lo_, 1.0)

    sc = slice(meta["scored_first"] - off, meta["scored_last"] - off + 1)
    stats = {"local_warp_median_pct": float(100 * np.nanmedian(local[sc])),
             "local_warp_worst_frame_pct": float(100 * np.nanmax(local[sc])),
             "worst_frame_video": int(np.nanargmax(local[sc]) + meta["scored_first"]),
             "floor_rms_cm": float(100 * np.sqrt(np.mean(resid ** 2))), "floor_bend_deg": bend,
             "floor_inlier_frac": floor_frac, "camera_height_m": cam_h, "camera_pitch_deg": pitch,
             "outer_lines_px_original": lo_, "outer_lines_px_undistorted": lu_, "outer_line_ratio": line_ratio}
    checks = []
    def add(name, ok, detail, level="WARN"):
        checks.append({"check": name, "status": "PASS" if ok else level, "detail": detail})
    add("depth local warping", stats["local_warp_median_pct"] <= 1.5 and stats["local_warp_worst_frame_pct"] <= 4.0,
        f"static background after removing the global scale: median {stats['local_warp_median_pct']:.2f}%, "
        f"worst frame {stats['local_warp_worst_frame_pct']:.2f}% (video frame {stats['worst_frame_video']})")
    add("floor plane", stats["floor_rms_cm"] <= 2.0 and bend <= 5.0 and floor_frac >= 0.3,
        f"RMS {stats['floor_rms_cm']:.1f} cm, near/far bend {bend:.1f} deg, {100 * floor_frac:.0f}% of lower-image "
        f"background on it; camera {cam_h:.2f} m above, pitched {pitch:.1f} deg")
    add("undistortion lines", line_ratio >= 0.9,
        f"long straight edges in the outer ring: {lo_:.0f} px original -> {lu_:.0f} px undistorted "
        f"(x{line_ratio:.2f})")

    # sheet
    fig, axs = plt.subplots(2, 3, figsize=(21, 11))
    for ax, img, seg, title in ((axs[0, 0], orig, so, f"original video frame {i0 + off}: long segments"),
                                (axs[0, 1], und, su, "undistorted: long segments")):
        ax.imshow(img)
        for x0, y0, x1, y1 in seg:
            ax.plot([x0, x1], [y0, y1], color="yellow", lw=1.2)
        ax.add_patch(matplotlib.patches.Ellipse((intr["width"] / 2, intr["height"] / 2),
                     2 * 0.55 * np.hypot(intr["width"] / 2, intr["height"] / 2),
                     2 * 0.55 * np.hypot(intr["width"] / 2, intr["height"] / 2), fill=False, color="cyan", ls="--"))
        ax.set_title(title + "  (outer ring outside cyan)", fontsize=11); ax.axis("off")
    im = axs[0, 2].imshow(D, cmap="turbo"); fig.colorbar(im, ax=axs[0, 2], fraction=0.035)
    axs[0, 2].contour(np.isfinite(resid_map), [0.5], colors="white", linewidths=0.8)
    axs[0, 2].set_title("background depth (m); white = floor region used", fontsize=11); axs[0, 2].axis("off")
    im = axs[1, 0].imshow(100 * pix_instab, cmap="magma", vmin=0, vmax=3)
    fig.colorbar(im, ax=axs[1, 0], fraction=0.035)
    axs[1, 0].set_title("depth instability per pixel (%, after global scale)", fontsize=11); axs[1, 0].axis("off")
    im = axs[1, 1].imshow(100 * resid_map, cmap="coolwarm", vmin=-5, vmax=5)
    fig.colorbar(im, ax=axs[1, 1], fraction=0.035)
    axs[1, 1].set_title(f"floor: distance to the fitted plane (cm), RMS {stats['floor_rms_cm']:.1f}, bend {bend:.1f} deg",
                        fontsize=11); axs[1, 1].axis("off")
    x = np.arange(n) + off
    axs[1, 2].plot(x, 100 * (np.exp(scale) - 1), label="global scale change (%)")
    axs[1, 2].plot(x, 100 * local, label="local warping (%)")
    axs[1, 2].axvspan(meta["scored_first"], meta["scored_last"], color="tab:green", alpha=0.08, label="scored span")
    axs[1, 2].set_xlabel("video frame"); axs[1, 2].legend(); axs[1, 2].grid(alpha=0.3)
    axs[1, 2].set_title("static background depth over time", fontsize=11)
    fig.suptitle(f"{ep} — stage 1 background checks (camera {meta.get('camera', '?')})", fontsize=15)
    fig.tight_layout()
    sheet = os.path.join(review_dir, f"{ep}_background.png")
    fig.savefig(sheet, dpi=55)
    plt.close(fig)
    return checks, os.path.basename(sheet), stats


def review_episode(ep_dir: str, review_dir: str) -> dict:
    meta = json.load(open(os.path.join(ep_dir, "track1_meta.json")))
    ep = f"episode_{meta['episode']:06d}"
    off, n = meta["frame_offset"], meta["n_frames"]
    frames_dir, depth_dir = os.path.join(ep_dir, "frames"), os.path.join(ep_dir, "depth")
    n_frames = len([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    n_depth = len([f for f in os.listdir(depth_dir) if f.endswith(".png")])
    intr = json.load(open(os.path.join(ep_dir, "intrinsics_stable.json")))
    ac_path = os.path.join(ep_dir, "anycalib", "undistorted_intrinsics.json")
    ac = json.load(open(ac_path)) if os.path.exists(ac_path) else None
    dist_path = os.path.join(ep_dir, "anycalib", "distortion.json")
    dist = json.load(open(dist_path)) if os.path.exists(dist_path) else None

    # Depth statistics over every kept frame.
    pct = np.full((n_depth, 3), np.nan)
    for i in range(n_depth):
        d = load_depth(os.path.join(depth_dir, f"{i:06d}.png"))
        pct[i] = np.nanpercentile(d, [10, 50, 90])
    med = pct[:, 1]
    rel_std = float(1.4826 * np.nanmedian(np.abs(med - np.nanmedian(med))) / np.nanmedian(med))
    jumps = np.abs(np.diff(med)) / med[:-1]
    max_jump = float(np.nanmax(jumps)) if len(jumps) else 0.0

    checks = []
    def add(name, ok, detail, level="FAIL"):
        checks.append({"check": name, "status": "PASS" if ok else level, "detail": detail})
    add("frame count", n_frames == n, f"{n_frames} frames, expected {n}")
    add("depth count", n_depth == n, f"{n_depth} depth maps, expected {n}")
    fw = intr["fx"] / intr["width"]
    add("focal length", 0.45 <= fw <= 0.80, f"fx={intr['fx']:.1f} px, fx/width={fw:.3f}", "WARN")
    add("principal point", abs(intr["cx"] - intr["width"] / 2) < 2 and abs(intr["cy"] - intr["height"] / 2) < 2,
        f"cx={intr['cx']:.1f}, cy={intr['cy']:.1f} for {intr['width']}x{intr['height']}", "WARN")
    if ac:
        r = intr["fx"] / ac["fx"]
        add("MoGe vs AnyCalib fx", abs(r - 1) <= 0.10, f"MoGe {intr['fx']:.1f} / AnyCalib {ac['fx']:.1f} = {r:.3f}", "WARN")
    add("depth range", bool(np.all((med > 0.5) & (med < 15))),
        f"median scene depth {np.nanmin(med):.2f}..{np.nanmax(med):.2f} m")
    add("depth scale drift", rel_std <= 0.03 and max_jump <= 0.05,
        f"robust std {100 * rel_std:.1f}% of median, max frame-to-frame jump {100 * max_jump:.1f}%", "WARN")
    sw_path = os.path.join(ep_dir, "depth_scale.json")
    if os.path.exists(sw_path):
        sw = json.load(open(sw_path))
        add("depth stabilised", sw["frames_interpolated"] <= 0.1 * n,
            f"MoGe scale swing removed from depth_raw: {sw['scored_swing_pct']['min']:+.1f}..."
            f"{sw['scored_swing_pct']['max']:+.1f}% (jitter {sw['scored_jitter_pct_median_abs_diff']:.2f}%/frame), "
            f"{sw['frames_interpolated']} frames with too little background interpolated", "WARN")
    else:
        add("depth stabilised", False, "no depth_scale.json: depth/ is raw MoGe output", "WARN")

    # Sheet.
    try:
        cam = {json.loads(l)["episode_index"]: json.loads(l)["camera"] for l in open(paths.META)}.get(meta["episode"], "?")
        os.makedirs(review_dir, exist_ok=True)
        bg_checks, bg_sheet, bg_stats = background_checks(ep_dir, dict(meta, camera=cam), intr, review_dir)
        checks += bg_checks
    except Exception as exc:                                    # a broken check must not hide the rest
        checks.append({"check": "background checks", "status": "WARN", "detail": f"could not run: {exc!r}"})
        bg_sheet, bg_stats, cam = None, None, "?"
    rows = sorted({meta["scored_first"] - off, (meta["scored_first"] + meta["scored_last"]) // 2 - off,
                   meta["scored_last"] - off})
    fig = plt.figure(figsize=(21, 5.2 * len(rows) + 4.5))
    gs = fig.add_gridspec(len(rows) + 1, 3, height_ratios=[1] * len(rows) + [0.75])
    vmin, vmax = np.nanpercentile(pct[:, 0], 5), np.nanpercentile(pct[:, 2], 95)
    for r, i in enumerate(rows):
        orig = source_frame(meta["source_video"], i + off)
        und = cv2.cvtColor(cv2.imread(os.path.join(frames_dir, f"{i:06d}.png")), cv2.COLOR_BGR2RGB)
        d = load_depth(os.path.join(depth_dir, f"{i:06d}.png"))
        for c, (img, title) in enumerate(((orig, f"original video frame {i + off}"),
                                          (und, f"undistorted (local {i})"), (None, "MoGe depth (m)"))):
            ax = fig.add_subplot(gs[r, c])
            if img is not None:
                ax.imshow(img)
            else:
                im = ax.imshow(d, cmap="turbo", vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax, fraction=0.035)
                title += f"  median {np.nanmedian(d):.2f} m"
            ax.set_title(title, fontsize=12)
            ax.axis("off")
    ax = fig.add_subplot(gs[len(rows), :])
    x = np.arange(n_depth) + off
    ax.fill_between(x, pct[:, 0], pct[:, 2], color="tab:blue", alpha=0.2, label="10–90th percentile")
    ax.plot(x, med, color="tab:blue", lw=2, label="median scene depth")
    ax.axvspan(meta["scored_first"], meta["scored_last"], color="tab:green", alpha=0.08, label="scored span")
    ax.set_xlabel("video frame"); ax.set_ylabel("depth (m)"); ax.legend(loc="upper right")
    ax.set_title(f"scene depth over time: drift {100 * rel_std:.1f}%, max jump {100 * max_jump:.1f}%")
    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else \
             "WARN" if any(c["status"] == "WARN" for c in checks) else "PASS"
    fig.suptitle(f"{ep} — stage 1 prep — {status}   (fx={intr['fx']:.1f}, {n} frames from video "
                 f"{off}..{off + n - 1})", fontsize=16)
    fig.tight_layout()
    os.makedirs(review_dir, exist_ok=True)
    sheet = os.path.join(review_dir, f"{ep}_prep.png")
    fig.savefig(sheet, dpi=55)
    plt.close(fig)

    result = {"episode": meta["episode"], "status": status, "checks": checks, "sheet": os.path.basename(sheet),
              "sheet_background": bg_sheet, "background": bg_stats, "camera": cam, "version": REVIEW_VERSION,
              "intrinsics": intr, "anycalib_undistorted": ac, "anycalib_distortion": dist,
              "seconds": meta.get("stage1_seconds"), "frames": n, "frame_offset": off}
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--stage_dir", required=True)
    args = ap.parse_args()

    review = os.path.join(args.stage_dir, "review")
    results = []
    for e in args.episodes:
        ep_dir = os.path.join(args.out_root, f"episode_{e:06d}")
        if not os.path.exists(os.path.join(ep_dir, "intrinsics_stable.json")):
            for old in glob.glob(os.path.join(review, f"episode_{e:06d}_*.png")):   # stale sheet removed
                if not any(t in os.path.basename(old) for t in ("_before_", "_v2_", "_v3_", "_old", "_refine")):
                    os.remove(old)
            results.append({"episode": e, "status": "FAIL", "checks": [{"check": "stage 1 output",
                            "status": "FAIL", "detail": "missing (see the episode log)"}]})
            continue
        chk_path = os.path.join(args.stage_dir, f"episode_{e:06d}.checks.json")
        key_out = os.path.join(ep_dir, "intrinsics_stable.json")
        newest = max(os.path.getmtime(p) for p in (key_out, os.path.join(ep_dir, "depth_scale.json"))
                     if os.path.exists(p))
        reuse_ok = (os.path.exists(chk_path) and os.path.exists(key_out)
                    and os.path.getmtime(chk_path) > newest
                    and json.load(open(chk_path)).get("version") == REVIEW_VERSION
                    and os.path.exists(os.path.join(review, json.load(open(chk_path)).get("sheet", "-"))))
        if reuse_ok:                      # sheet already up to date: reuse it
            results.append(json.load(open(chk_path)))
            continue
        r = review_episode(ep_dir, review)
        json.dump(r, open(os.path.join(args.stage_dir, f"episode_{e:06d}.checks.json"), "w"), indent=1)
        results.append(r)
        print(f"episode {e:3d}: {r['status']:4s}  " + "; ".join(
            f"{c['check']}: {c['status']}" for c in r["checks"] if c["status"] != "PASS"))

    color = {"PASS": "#4c4", "WARN": "#fb3", "FAIL": "#f55"}
    out = ["<html><head><meta charset='utf-8'><title>Stage 1 prep</title><style>body{font-family:sans-serif;"
           "background:#111;color:#eee} td,th{border:1px solid #444;padding:3px 7px} table{border-collapse:"
           "collapse} img{max-width:100%}</style></head><body><h1>Stage 1 — prep (trim, undistort, frames, "
           "depth, intrinsics)</h1><p>Check: undistorted frames have straight walls/ceiling lines; depth "
           "is smooth and ordered (near = blue); the scene-depth curve is flat (static camera). Second sheet: "
           "yellow long straight segments should get longer after undistortion; the per-pixel instability map "
           "should be dark on walls/floor; the floor residual map should be flat (white), not bowl- or "
           "ramp-shaped.</p><table>"
           "<tr><th>episode</th><th>status</th><th>checks</th><th>seconds per step</th></tr>"]
    for r in results:
        det = "<br>".join(f"<span style='color:{color[c['status']]}'>{c['status']}</span> {html.escape(c['check'])}: "
                          f"{html.escape(c['detail'])}" for c in r["checks"])
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
            if r.get("sheet_background"):
                out.append(f"<a href='review/{r['sheet_background']}'><img src='review/{r['sheet_background']}'></a>")
    out.append("</body></html>")
    open(os.path.join(args.stage_dir, "index.html"), "w", encoding="utf-8").write("\n".join(out))
    print(f"wrote {os.path.join(args.stage_dir, 'index.html')}")


if __name__ == "__main__":
    main()
