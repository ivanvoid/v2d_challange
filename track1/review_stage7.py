#!/usr/bin/env python3
"""Stage 7 review (smoothing): one sheet per episode + checks + index.html.

Sheet (review/episode_XXXXXX_smooth.png):
  6 frames incl. the frames with the largest correction: image | raw pose | smoothed pose
  (textured mesh, SAM2 mask outline green), IoU of each in the title
  plots: translation x/y/z raw (thin) vs smoothed (thick) with outliers marked; object
  |2nd difference| raw vs smoothed (log); rotation step per frame raw vs smoothed;
  correction per frame (cm, deg)

Checks:
  poses_smoothed for every frame                                   FAIL
  max rotation step after smoothing <= 30 deg                      WARN (a flip survived)
  smoothed IoU median >= raw IoU median - 0.03                     WARN (smoothing pulled off the image)
  median correction <= 2 cm and <= 5 deg                           WARN (over-smoothing)
  acceleration reduced                                             WARN

    /opt/venvs/sam3d_body/bin/python track1/review_stage7.py --episodes 6 8 17 26 28 \
        --out_root out --stage_dir logs/stages/07_smooth
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


def load_pose(path):
    d = json.load(open(path))
    w, x, y, z = d["rotation"]
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    M[:3, 3] = d["translation"]
    return M


def render(v, f, fcol, M, K, hw):
    pc = v @ M[:3, :3].T + M[:3, 3]
    uvw = pc @ K.T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
    z = pc[f][:, :, 2].mean(1)
    col = np.zeros((*hw, 3), np.uint8)
    hit = np.zeros(hw, np.uint8)
    pts = np.round(uv[f]).astype(np.int32)
    for i in np.argsort(-z):
        cv2.fillConvexPoly(col, pts[i], tuple(int(c) for c in fcol[i]))
        cv2.fillConvexPoly(hit, pts[i], 1)
    return col, hit > 0


def review_episode(W, review_dir):
    meta = json.load(open(os.path.join(W, "track1_meta.json")))
    e, off, first, last, n = (meta[k] for k in ("episode", "frame_offset", "scored_first", "scored_last", "n_frames"))
    ep = f"episode_{e:06d}"
    rep = json.load(open(os.path.join(W, "track", "smooth.json")))
    it = json.load(open(os.path.join(W, "intrinsics_stable.json")))
    K = np.array([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]], [0, 0, 1]])
    hw = (it["height"], it["width"])
    tm = trimesh.load(os.path.join(W, "mesh_scaled.obj"), force="mesh", process=False)
    v, fa = np.asarray(tm.vertices), np.asarray(tm.faces)
    vis = tm.visual.to_color() if type(tm.visual).__name__ == "TextureVisuals" else tm.visual
    fcol = np.asarray(vis.vertex_colors)[:, :3].astype(float)[fa].mean(1)

    loc = np.arange(first - off, last - off + 1)
    raw = np.stack([load_pose(os.path.join(W, "poses", f"{f:06d}.json")) for f in loc])
    sm = np.stack([load_pose(os.path.join(W, "poses_smoothed", f"{f:06d}.json")) for f in loc])
    fr = {r["frame"]: r for r in rep["frames"]}
    out = np.array([fr[f]["outlier"] for f in loc])
    cc = np.array([fr[f]["corr_cm"] for f in loc])
    cd = np.array([fr[f]["corr_deg"] for f in loc])
    vf = loc + off

    def iou_at(M, f):
        m = cv2.imread(os.path.join(W, "masks", "1", f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
        _, hit = render(v, fa, fcol, M, K, hw)
        return float((hit & m).sum() / max((hit | m).sum(), 1))
    samp = sorted(set(np.linspace(0, len(loc) - 1, 12).round().astype(int).tolist()))
    iou_raw = np.array([iou_at(raw[k], loc[k]) for k in samp])
    iou_sm = np.array([iou_at(sm[k], loc[k]) for k in samp])

    def acc(T):
        a = np.full(len(T), np.nan)
        a[1:-1] = np.linalg.norm(T[:-2] - 2 * T[1:-1] + T[2:], axis=1) * 100
        return a
    def steps(Ms):
        s = np.zeros(len(Ms))
        for i in range(1, len(Ms)):
            c = (np.trace(Ms[i - 1][:3, :3].T @ Ms[i][:3, :3]) - 1) / 2
            s[i] = np.degrees(np.arccos(np.clip(c, -1, 1)))
        return s
    a_raw, a_sm = acc(raw[:, :3, 3]), acc(sm[:, :3, 3])
    s_raw, s_sm = steps(raw), steps(sm)

    checks = []
    def add(name, ok, detail, level="WARN"):
        checks.append({"check": name, "status": "PASS" if ok else level, "detail": detail})
    nps = len(glob.glob(os.path.join(W, "poses_smoothed", "*.json")))
    add("smoothed poses present", nps == n, f"{nps} files, expected {n}", "FAIL")
    add("no flip left", s_sm.max() <= 30, f"max rotation step {s_raw.max():.0f} -> {s_sm.max():.0f} deg")
    add("still on the image", np.median(iou_sm) >= np.median(iou_raw) - 0.03,
        f"IoU median raw {np.median(iou_raw):.2f} -> smoothed {np.median(iou_sm):.2f} (12 sampled frames)")
    add("not over-smoothed", np.median(cc) <= 2 and np.median(cd) <= 5,
        f"median correction {np.median(cc):.1f} cm / {np.median(cd):.1f} deg (max {cc.max():.1f} cm / {cd.max():.0f} deg)")
    add("acceleration reduced", np.nanmean(a_sm) < np.nanmean(a_raw),
        f"{np.nanmean(a_raw):.3f} -> {np.nanmean(a_sm):.3f} cm/frame^2; outliers {int(out.sum())}")
    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else \
             "WARN" if any(c["status"] == "WARN" for c in checks) else "PASS"

    show = sorted(set(np.linspace(0, len(loc) - 1, 4).round().astype(int).tolist()
                      + [int(np.argmax(cc)), int(np.argmax(cd))]))
    nrow = (len(show) + 1) // 2
    fig = plt.figure(figsize=(24, 5.5 * nrow + 15))
    gs = fig.add_gridspec(nrow + 4, 2, height_ratios=[1.0] * nrow + [0.5, 0.45, 0.45, 0.45])
    for j, k in enumerate(show):
        f = loc[k]
        img = cv2.cvtColor(cv2.imread(os.path.join(W, "frames", f"{f:06d}.png")), cv2.COLOR_BGR2RGB)
        m = cv2.imread(os.path.join(W, "masks", "1", f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
        panels = [img.copy()]
        ious = []
        for M in (raw[k], sm[k]):
            col, hit = render(v, fa, fcol, M, K, hw)
            o = img.copy()
            o[hit] = (0.25 * o[hit] + 0.75 * col[hit]).astype(np.uint8)
            ious.append((hit & m).sum() / max((hit | m).sum(), 1))
            panels.append(o)
        for p in panels:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(p, cnts, -1, (0, 255, 0), 2)
        ys, xs = np.nonzero(m)
        if len(ys):
            pad = int(0.9 * max(np.ptp(xs), np.ptp(ys))) + 30
            y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad
            panels = [p[y0:y1, x0:x1] for p in panels]
        ax = fig.add_subplot(gs[j // 2, j % 2])
        ax.imshow(np.concatenate(panels, 1)); ax.axis("off")
        ax.set_title(f"video frame {f + off}: image | raw IoU {ious[0]:.2f} | smoothed IoU {ious[1]:.2f}   "
                     f"(correction {cc[k]:.1f} cm, {cd[k]:.0f} deg{', OUTLIER' if out[k] else ''})", fontsize=12,
                     color="red" if out[k] else "black")
    ax = fig.add_subplot(gs[nrow, :])
    for i, (c, lab) in enumerate(zip(("tab:red", "tab:green", "tab:blue"), "xyz")):
        ax.plot(vf, raw[:, i, 3], color=c, lw=0.8, alpha=0.6)
        ax.plot(vf, sm[:, i, 3], color=c, lw=2.2, label=f"{lab} smoothed (thin = raw)")
    if out.any():
        ax.scatter(vf[out], raw[out, 2, 3], color="k", marker="x", s=40, label="outlier (ignored)", zorder=5)
    ax.legend(ncol=4, fontsize=9); ax.set_title("translation (m)")
    ax = fig.add_subplot(gs[nrow + 1, :])
    ax.plot(vf, a_raw, color="gray", lw=0.8, label=f"raw (mean {np.nanmean(a_raw):.3f})")
    ax.plot(vf, a_sm, color="tab:orange", lw=1.8, label=f"smoothed (mean {np.nanmean(a_sm):.3f})")
    ax.set_yscale("log"); ax.legend(fontsize=9); ax.set_title("object |2nd difference| (cm/frame^2) — what ACC-O scores")
    ax = fig.add_subplot(gs[nrow + 2, :])
    ax.plot(vf, s_raw, color="gray", lw=0.8, label="raw")
    ax.plot(vf, s_sm, color="tab:purple", lw=1.8, label="smoothed")
    ax.legend(fontsize=9); ax.set_title("rotation step per frame (deg)")
    ax = fig.add_subplot(gs[nrow + 3, :])
    ax.plot(vf, cc, color="tab:blue", lw=1.2, label="correction (cm)")
    ax2 = ax.twinx(); ax2.plot(vf, cd, color="tab:red", lw=1.0, label="correction (deg)")
    ax.set_title("how far smoothing moved each frame (blue cm, red deg)"); ax.set_xlabel("video frame")
    fig.suptitle(f"{ep} — stage 7 smoothing — {status} — raw (stage 6) vs smoothed; textured mesh, green = SAM2 mask",
                 fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(review_dir, exist_ok=True)
    sheet = os.path.join(review_dir, f"{ep}_smooth.png")
    fig.savefig(sheet, dpi=50)
    plt.close(fig)
    return {"episode": e, "status": status, "checks": checks, "sheet": os.path.basename(sheet)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--stage_dir", required=True)
    args = ap.parse_args()
    review = os.path.join(args.stage_dir, "review")
    results = []
    for e in args.episodes:
        W = os.path.join(args.out_root, f"episode_{e:06d}")
        if not os.path.exists(os.path.join(W, "track", "smooth.json")):
            for old in glob.glob(os.path.join(review, f"episode_{e:06d}_smooth.png")):
                os.remove(old)
            results.append({"episode": e, "status": "FAIL", "checks": [{"check": "stage 7 output", "status": "FAIL",
                            "detail": "missing (see the episode log)"}]})
            continue
        chk_path = os.path.join(args.stage_dir, f"episode_{e:06d}.checks.json")
        key_out = os.path.join(W, "track", "smooth.json")
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
    out = ["<html><head><meta charset='utf-8'><title>Stage 7 smoothing</title><style>body{font-family:sans-serif;"
           "background:#111;color:#eee} td,th{border:1px solid #444;padding:3px 7px} table{border-collapse:collapse}"
           " img{max-width:100%}</style></head><body><h1>Stage 7 — smoothing</h1><p>Check: smoothed (right) sits on "
           "the object as well as raw (middle); no flip left in the rotation plot; real motion kept (smoothed "
           "translation follows raw, only noise removed).</p><table><tr><th>episode</th><th>status</th><th>checks"
           "</th></tr>"]
    for r in results:
        det = "<br>".join(f"<span style='color:{color[c['status']]}'>{c['status']}</span> "
                          f"{html.escape(c['check'])}: {html.escape(c['detail'])}" for c in r["checks"])
        out.append(f"<tr><td><a href='#e{r['episode']}'>{r['episode']}</a></td><td style='color:"
                   f"{color[r['status']]}'>{r['status']}</td><td>{det}</td></tr>")
    out.append("</table>")
    for r in results:
        out.append(f"<h2 id='e{r['episode']}'>episode {r['episode']} — {r['status']}</h2>")
        if r.get("sheet"):
            made = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(os.path.join(review, r["sheet"]))))
            out.append(f"<p style='color:#999'>sheet made at {made}</p>"
                       f"<a href='review/{r['sheet']}'><img src='review/{r['sheet']}'></a>")
    out.append("</body></html>")
    open(os.path.join(args.stage_dir, "index.html"), "w", encoding="utf-8").write("\n".join(out))
    print(f"wrote {os.path.join(args.stage_dir, 'index.html')}")


if __name__ == "__main__":
    main()
