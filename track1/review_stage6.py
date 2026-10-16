#!/usr/bin/env python3
"""Stage 6 review (tracking): one sheet per episode + checks + index.html.

Sheet (review/episode_XXXXXX_track.png):
  8 frames across the scored span + the worst-fit frame: image | textured mesh at the tracked
  pose (CPU render), SAM2 mask outline in green, state + IoU in the title
  plots over the scored span: IoU and colour error, rotation step per frame (deg), translation
  (x/y/z, m), and the per-frame state as a coloured band (observed / recovered / uncertain /
  jump / occluded)

Checks:
  poses for every frame                                   FAIL
  median IoU over the scored span >= 0.6                  WARN
  frames with IoU < 0.4 <= 10 %                           WARN
  rotation steps > 45 deg in the output <= 0              WARN  (unexplained flips)
  recoveries accepted <= 5 % of frames                    WARN
  uncertain + jump frames <= 15 %                         WARN

    /opt/venvs/sam3d_body/bin/python track1/review_stage6.py --episodes 6 8 17 26 28 \
        --out_root out --stage_dir logs/stages/06_track
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
import trimesh  # noqa: E402

STATE_COL = {"observed": "#2ca02c", "recovered": "#1f77b4", "uncertain": "#ff7f0e", "jump": "#d62728",
             "occluded": "#7f7f7f"}


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def load_pose(path):
    d = json.load(open(path))
    M = np.eye(4)
    M[:3, :3] = quat_to_R(d["rotation"])
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


def review_episode(W: str, review_dir: str) -> dict:
    meta = json.load(open(os.path.join(W, "track1_meta.json")))
    e, off, first, last, n = (meta[k] for k in ("episode", "frame_offset", "scored_first", "scored_last", "n_frames"))
    ep = f"episode_{e:06d}"
    st = json.load(open(os.path.join(W, "track", "states.json")))
    rows = {r["frame"]: r for r in st["frames"]}
    params = st["params"]
    it = json.load(open(os.path.join(W, "intrinsics_stable.json")))
    K = np.array([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]], [0, 0, 1]])
    hw = (it["height"], it["width"])
    tm = trimesh.load(os.path.join(W, "mesh_scaled.obj"), force="mesh", process=False)
    v, fa = np.asarray(tm.vertices), np.asarray(tm.faces)
    vis = tm.visual.to_color() if type(tm.visual).__name__ == "TextureVisuals" else tm.visual
    fcol = np.asarray(vis.vertex_colors)[:, :3].astype(float)[fa].mean(1)

    loc = np.arange(first - off, last - off + 1)
    sel = [rows[l] for l in loc if l in rows]
    iou = np.array([r["iou"] for r in sel])
    ce = np.array([r["colour_err"] for r in sel])
    rs = np.array([r["rot_step"] for r in sel])
    states = [r["state"] for r in sel]
    vf = np.array([r["video_frame"] for r in sel])
    T = np.array([load_pose(os.path.join(W, "poses", f"{r['frame']:06d}.json"))[:3, 3] for r in sel])
    n_poses = len([x for x in os.listdir(os.path.join(W, "poses")) if x.endswith(".json")])
    bad = sum(s in ("uncertain", "jump") for s in states)
    anchors_vf = [a["video_frame"] for a in st.get("anchors", [])]
    overr = sum(1 for r in sel if r.get("best_alt_fit") is not None and r["best_alt_fit"] > r["fit"] + 0.05)

    checks = []
    def add(name, ok, detail, level="WARN"):
        checks.append({"check": name, "status": "PASS" if ok else level, "detail": detail})
    add("poses present", n_poses == n, f"{n_poses} pose files, expected {n}", "FAIL")
    add("silhouette fit", np.median(iou) >= 0.6, f"median IoU {np.median(iou):.2f} over the scored span")
    add("few bad frames", (iou < 0.4).mean() <= 0.10, f"{100 * (iou < 0.4).mean():.0f}% of frames IoU < 0.4")
    add("no flips", (rs > params["max_rot_step"]).sum() == 0,
        f"{(rs > params['max_rot_step']).sum()} rotation steps > {params['max_rot_step']:.0f} deg; max {rs.max():.0f} deg")
    add("anchors", len(anchors_vf) >= 2, f"{len(anchors_vf)} anchors at video frames {anchors_vf}")
    add("continuity overrides", overr <= 0.2 * len(sel),
        f"{overr} frames kept a worse-fitting candidate for continuity (a better-fitting one jumped)")
    add("tracking confident", bad <= 0.15 * len(sel), f"{100 * bad / len(sel):.0f}% uncertain or jump frames")
    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else \
             "WARN" if any(c["status"] == "WARN" for c in checks) else "PASS"

    show = [int(round(i)) for i in np.linspace(0, len(sel) - 1, 8)]
    worst = int(np.argmin(iou))
    if worst not in show:
        show.append(worst)
    show = sorted(set(show))
    ncol = 3
    nrow = (len(show) + ncol - 1) // ncol
    fig = plt.figure(figsize=(24, 5.6 * nrow + 11))
    gs = fig.add_gridspec(nrow + 3, ncol, height_ratios=[1.0] * nrow + [0.45, 0.45, 0.45])
    for j, k in enumerate(show):
        r = sel[k]
        f = r["frame"]
        img = cv2.cvtColor(cv2.imread(os.path.join(W, "frames", f"{f:06d}.png")), cv2.COLOR_BGR2RGB)
        m = cv2.imread(os.path.join(W, "masks", "1", f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
        col, hit = render(v, fa, fcol, load_pose(os.path.join(W, "poses", f"{f:06d}.json")), K, hw)
        over = img.copy()
        over[hit] = (0.25 * over[hit] + 0.75 * col[hit]).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cnts, -1, (0, 255, 0), 2)
        ys, xs = np.nonzero(m | hit)
        if len(ys):
            pad = int(0.9 * max(np.ptp(xs), np.ptp(ys))) + 30
            y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad
            both = np.concatenate([img[y0:y1, x0:x1], over[y0:y1, x0:x1]], axis=1)
        else:
            both = np.concatenate([img, over], axis=1)
        ax = fig.add_subplot(gs[j // ncol, j % ncol])
        ax.imshow(both); ax.axis("off")
        ax.set_title(f"video frame {r['video_frame']}  {r['state']}  IoU {r['iou']:.2f}  [{r.get('src', '')}]"
                     + ("  [worst]" if k == worst else ""), fontsize=13,
                     color=STATE_COL.get(r["state"], "black") if r["state"] != "observed" else "black")
    ax = fig.add_subplot(gs[nrow, :])
    ax.plot(vf, iou, color="tab:green", lw=1.5, label="IoU")
    ax.plot(vf, ce / 255, color="tab:purple", lw=1, label="colour error / 255")
    for s, c in STATE_COL.items():
        xs_ = vf[[x == s for x in states]]
        ax.scatter(xs_, np.full(len(xs_), -0.05), color=c, s=10, label=s, marker="s")
    best_alt = np.array([r.get("best_alt_fit", r["fit"]) for r in sel])
    if "quality" in sel[0]:
        ax.plot(vf, [r["quality"] for r in sel], color="tab:blue", lw=0.8, alpha=0.6, label="anchor quality")
        ax.plot(vf, [min(1.0, 5 * r["mask_change"]) for r in sel], color="gray", lw=0.6, alpha=0.6,
                label="mask change x5")
    ax.plot(vf, np.clip(best_alt + ce / 255, 0, 1), color="tab:green", lw=0.6, ls=":", label="best candidate IoU")
    for a in anchors_vf:
        ax.axvline(a, color="tab:blue", ls="--", lw=1)
    ax.set_ylim(-0.1, 1.0); ax.legend(ncol=8, fontsize=9)
    ax.set_title("fit per frame (blue dashed = anchors, bottom band = state)")
    ax = fig.add_subplot(gs[nrow + 1, :])
    ax.plot(vf, rs, color="tab:red", lw=1.2)
    ax.axhline(params["max_rot_step"], color="k", ls="--", lw=1)
    for a in anchors_vf:
        ax.axvline(a, color="tab:blue", ls="--", lw=1)
    ax.set_title("rotation change per frame (deg); dashed = jump limit")
    ax = fig.add_subplot(gs[nrow + 2, :])
    for i, (c, lab) in enumerate(zip(("tab:red", "tab:green", "tab:blue"), "xyz")):
        ax.plot(vf, T[:, i], color=c, lw=1.5, label=lab)
    ax.legend(); ax.set_title("translation (m, camera frame)"); ax.set_xlabel("video frame")
    fig.suptitle(f"{ep} — stage 6 tracking (raw, before smoothing) — {status} — {len(anchors_vf)} anchors; "
                 f"left = image, right = textured mesh at the chosen pose, green = SAM2 mask", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(review_dir, exist_ok=True)
    sheet = os.path.join(review_dir, f"{ep}_track.png")
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
        if not os.path.exists(os.path.join(W, "track", "states.json")):
            for old in glob.glob(os.path.join(review, f"episode_{e:06d}_*.png")):   # stale sheet removed
                if not any(t in os.path.basename(old) for t in ("_before_", "_v2_", "_v3_", "_old", "_refine")):
                    os.remove(old)
            results.append({"episode": e, "status": "FAIL", "checks": [{"check": "stage 6 output", "status": "FAIL",
                            "detail": "missing (see the episode log)"}]})
            continue
        chk_path = os.path.join(args.stage_dir, f"episode_{e:06d}.checks.json")
        key_out = os.path.join(W, "track", "states.json")
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
    out = ["<html><head><meta charset='utf-8'><title>Stage 6 tracking</title><style>body{font-family:sans-serif;"
           "background:#111;color:#eee} td,th{border:1px solid #444;padding:3px 7px} table{border-collapse:collapse}"
           " img{max-width:100%}</style></head><body><h1>Stage 6 — object tracking</h1><p>Check: the textured mesh "
           "sits on the object with the right orientation (handle, front/back, up/down) on every frame; no sudden "
           "flips in the rotation plot; translation is smooth.</p><table><tr><th>episode</th><th>status</th>"
           "<th>checks</th></tr>"]
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
