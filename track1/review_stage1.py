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

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


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

    # Sheet.
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
        reuse_ok = (os.path.exists(chk_path) and os.path.exists(key_out)
                    and os.path.getmtime(chk_path) > os.path.getmtime(key_out)
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
           "is smooth and ordered (near = blue); the scene-depth curve is flat (static camera).</p><table>"
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
    out.append("</body></html>")
    open(os.path.join(args.stage_dir, "index.html"), "w", encoding="utf-8").write("\n".join(out))
    print(f"wrote {os.path.join(args.stage_dir, 'index.html')}")


if __name__ == "__main__":
    main()
