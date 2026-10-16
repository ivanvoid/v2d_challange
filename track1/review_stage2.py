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
    if len(depth_rows):
        ax.plot(depth_rows[:, 0], depth_rows[:, 2], color="tab:blue", lw=1.8)
        ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"body − MoGe depth on body pixels (cm): median {dz_med:+.1f}")
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
        reuse_ok = (os.path.exists(chk_path) and os.path.exists(key_out)
                    and os.path.getmtime(chk_path) > os.path.getmtime(key_out)
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
    out.append("</body></html>")
    open(os.path.join(args.stage_dir, "index.html"), "w", encoding="utf-8").write("\n".join(out))
    print(f"wrote {os.path.join(args.stage_dir, 'index.html')}")


if __name__ == "__main__":
    main()
