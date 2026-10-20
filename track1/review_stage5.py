#!/usr/bin/env python3
"""Stage 5 review (object mesh): one sheet per episode + checks + index.html.

Sheet (review/episode_XXXXXX_mesh.png):
  top        frame-choice plot: score, mask area and body occlusion over the scored span
  per mesh   the frame it was built from (mask outline), the mesh from the camera, the side
             and the top (flat shading, vertex colours), size / scale / cross-frame scores;
             CHOSEN marked
  per mesh   the mesh registered independently on each check frame: silhouette (blue) vs
             SAM2 mask outline (green), IoU and depth error in the title

Checks:
  a mesh was built                                               FAIL
  mesh size inside the object prior (x1.6 slack)                 WARN
  mesh size vs the mask's metric size on its frame in [0.8, 2.0] WARN  (too small / too big)
  cross-frame silhouette IoU median >= 0.6                       WARN  (wrong shape)
  cross-frame depth error median <= 5 cm                         WARN  (wrong size / placement)
  scale not at the edge of the search range                      WARN
  the two candidates agree in size within 30 %                   WARN

    /opt/venvs/sam3d_body/bin/python track1/review_stage5.py --episodes 6 8 17 26 28 \
        --out_root out --stage_dir logs/stages/05_mesh
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
import trimesh  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import paths  # noqa: E402
from viz import contrast_bg  # noqa: E402
PRIORS = json.load(open(paths.PRIORS))
META = str(paths.META)
SLACK = 1.6


def load_depth(path: str) -> np.ndarray:
    px = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return 1.0 / np.maximum(px / 65535.0, 1e-6) - 1.0


def render_view(v: np.ndarray, f: np.ndarray, col: np.ndarray, view: str, size: int = 300) -> np.ndarray:
    """Flat-shaded painter's render. Camera frame: x right, y down, z forward."""
    if view == "camera":
        p, depth = v[:, [0, 1]], v[:, 2]
    elif view == "side":                       # looking from the right, along -x
        p, depth = v[:, [2, 1]] * [-1, 1], -v[:, 0]
    else:                                      # top: looking down, along +y
        p, depth = v[:, [0, 2]] * [1, -1], v[:, 1]
    p = p - p.min(0)
    s = (size - 20) / max(p.max(), 1e-9)
    uv = p * s + 10
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = {"camera": [0, 0, -1], "side": [1, 0, 0], "top": [0, -1, 0]}[view]
    shade = 0.35 + 0.65 * np.abs(n @ np.array(light, float))
    fc = col[f].mean(1) * shade[:, None]
    order = np.argsort(-depth[f].mean(1))       # far first
    img = np.full((size, size, 3), contrast_bg(col), np.uint8)    # a dark object stays visible
    pts = np.round(uv[f]).astype(np.int32)
    for i in order:
        cv2.fillConvexPoly(img, pts[i], tuple(int(c) for c in fc[i]))
    return img


def project(v, M, K):
    pc = v @ M[:3, :3].T + M[:3, 3]
    uvw = pc @ K.T
    return uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)


def review_episode(W: str, review_dir: str) -> dict:
    meta = json.load(open(os.path.join(W, "track1_meta.json")))
    e, off = meta["episode"], meta["frame_offset"]
    ep = f"episode_{e:06d}"
    intr = json.load(open(os.path.join(W, "intrinsics_stable.json")))
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
    obj = next(json.loads(l) for l in open(META) if json.loads(l)["episode_index"] == e)["object"]
    plo, phi = PRIORS.get(obj, {}).get("size_m", [0.05, 3.0])
    plo, phi = plo / SLACK, phi * SLACK
    cand = json.load(open(os.path.join(W, "mesh", "candidates.json")))
    choice = json.load(open(os.path.join(W, "mesh", "choice.json")))
    checks_by = {c: json.load(open(os.path.join(W, "mesh", f"cand_{c:06d}", "check.json")))
                 for c in cand["chosen"] if os.path.exists(os.path.join(W, "mesh", f"cand_{c:06d}", "check.json"))}
    best = checks_by[choice["chosen"]]

    # metric size of the mask on the chosen frame (projected size, a lower bound of the 3D extent)
    f0 = choice["chosen"]
    m0 = cv2.imread(os.path.join(W, "masks", "1", f"{f0:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
    ys, xs = np.nonzero(m0)
    d0 = load_depth(os.path.join(W, "depth", f"{f0:06d}.png"))
    mask_size = max(np.ptp(xs), np.ptp(ys)) * float(np.median(d0[m0])) / intr["fx"]

    checks = []
    def add(name, ok, detail, level="WARN"):
        checks.append({"check": name, "status": "PASS" if ok else level, "detail": detail})
    add("mesh built", bool(checks_by), f"{len(checks_by)} candidate mesh(es)", "FAIL")
    add("size in prior", plo <= best["extent_m"] <= phi, f"{best['extent_m']:.2f} m (accept {plo:.2f}-{phi:.2f})")
    r = best["extent_m"] / mask_size
    add("size vs mask", 0.8 <= r <= 2.0, f"mesh {best['extent_m']:.2f} m vs mask {mask_size:.2f} m on its frame "
        f"(ratio {r:.2f})")
    add("fits other frames", best["median_iou"] >= 0.6,
        f"silhouette IoU median {best['median_iou']:.2f}, min {best['min_iou']:.2f} over {len(best['frames']) - 1} frames")
    add("depth agrees", best["median_depth_cm"] is not None and best["median_depth_cm"] <= 5,
        f"median depth error {best['median_depth_cm']:.1f} cm")
    si_path = os.path.join(W, "mesh", f"cand_{choice['chosen']:06d}", "scale_init.json")
    si = json.load(open(si_path)) if os.path.exists(si_path) else {"init_scale": None, "lo": 0.5, "hi": 2.0}
    add("scale not at search edge", si["lo"] * 1.03 < best["scale"] < si["hi"] / 1.03,
        f"scale {best['scale']:.3f}, started at {si['init_scale'] or 1:.2f}, searched {si['lo']:.2f}-{si['hi']:.2f}")
    add("textured", bool(best.get("textured")), "texture baked" if best.get("textured")
        else "flat colour: FoundationPose has no appearance cues to track orientation")
    if len(checks_by) > 1:
        ex = [c["extent_m"] for c in checks_by.values()]
        add("candidates agree", max(ex) / min(ex) <= 1.3, "extents " + ", ".join(f"{x:.2f} m" for x in ex))
    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else \
             "WARN" if any(c["status"] == "WARN" for c in checks) else "PASS"

    # ---------------------------------------------------------------- sheet
    ncand = len(checks_by)
    nchk = max(len(c["frames"]) for c in checks_by.values())
    fig = plt.figure(figsize=(26, 4.5 + ncand * 9.6))
    gs = fig.add_gridspec(1 + 2 * ncand, max(5, nchk), height_ratios=[0.8] + [1.0, 0.9] * ncand)
    ax = fig.add_subplot(gs[0, :])
    rows = cand["rows"]
    vfr = [r["video_frame"] for r in rows]
    amax = max(r["area"] for r in rows)
    ax.plot(vfr, [r["score"] for r in rows], "k-", lw=2, label="frame score")
    ax.plot(vfr, [r["area"] / amax for r in rows], color="tab:green", lw=1, label="mask area (relative)")
    ax.plot(vfr, [r["occlusion"] for r in rows], color="tab:red", lw=1, label="body in front of object")
    for c in cand["chosen"]:
        ax.axvline(c + off, color="tab:blue", ls="--", lw=2)
    ax.set_title("which frame to build the mesh from (blue dashed = candidates)"); ax.legend(fontsize=9)
    ax.set_xlabel("video frame")

    for k, (c, chk) in enumerate(sorted(checks_by.items())):
        tm = trimesh.load(os.path.join(W, "mesh", f"cand_{c:06d}", "mesh_scaled.obj"), force="mesh", process=False)
        v, f = np.asarray(tm.vertices), np.asarray(tm.faces)
        try:  # texture -> per-vertex colour for the preview
            vis = tm.visual.to_color() if type(tm.visual).__name__ == "TextureVisuals" else tm.visual
            col = np.asarray(vis.vertex_colors)[:, :3].astype(float)
        except Exception:
            col = np.full((len(v), 3), 200.0)
        if col.shape[0] != len(v):
            col = np.full((len(v), 3), 200.0)
        r0 = 1 + 2 * k
        img = cv2.cvtColor(cv2.imread(os.path.join(W, "frames", f"{c:06d}.png")), cv2.COLOR_BGR2RGB)
        m = cv2.imread(os.path.join(W, "masks", "1", f"{c:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
        ys, xs = np.nonzero(m)
        pad = int(0.6 * max(np.ptp(xs), np.ptp(ys))) + 20
        crop = img[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]
        chosen = c == choice["chosen"]
        ax = fig.add_subplot(gs[r0, 0]); ax.imshow(crop); ax.axis("off")
        ax.set_title(f"built from video frame {c + off}" + ("  — CHOSEN" if chosen else ""), fontsize=14,
                     color="green" if chosen else "black")
        for j, view in enumerate(("camera", "side", "top")):
            ax = fig.add_subplot(gs[r0, 1 + j]); ax.imshow(render_view(v, f, col, view)); ax.axis("off")
            ax.set_title(f"mesh — {view} view", fontsize=12)
        ax = fig.add_subplot(gs[r0, 4]); ax.axis("off")
        ax.text(0, 0.95, "\n".join([
            f"extent {chk['extent_m']:.2f} m (prior {plo * SLACK:.2f}-{phi / SLACK:.2f})",
            f"scale factor {chk['scale']:.3f}",
            f"vertices {len(v)}, faces {len(f)}",
            f"own-frame IoU {chk['own_iou']:.2f}",
            f"other frames IoU median {chk['median_iou']:.2f}",
            f"other frames IoU min {chk['min_iou']:.2f}",
            f"depth error median {chk['median_depth_cm']:.1f} cm"]), va="top", fontsize=13, family="monospace")
        for j, p in enumerate(chk["frames"]):
            img = cv2.cvtColor(cv2.imread(os.path.join(W, "frames", f"{p['frame']:06d}.png")), cv2.COLOR_BGR2RGB)
            m = cv2.imread(os.path.join(W, "masks", "1", f"{p['frame']:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
            uv = project(v, np.asarray(p["pose"]), K)
            sil = np.zeros(m.shape, np.uint8)
            cv2.fillPoly(sil, list(np.round(uv[f]).astype(np.int32)), 1)
            img[sil > 0] = (0.5 * img[sil > 0] + 0.5 * np.array([60, 120, 255])).astype(np.uint8)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
            both = (sil > 0) | m
            ys, xs = np.nonzero(both)
            pad = int(0.5 * max(np.ptp(xs), np.ptp(ys))) + 20
            crop = img[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]
            ax = fig.add_subplot(gs[r0 + 1, j]); ax.imshow(crop); ax.axis("off")
            dtxt = "-" if p["depth_cm"] is None else f"{p['depth_cm']:.1f}cm"
            ax.set_title(f"{'OWN ' if p['own'] else ''}frame {p['video_frame']}\nIoU {p['iou']:.2f}  depth {dtxt}",
                         fontsize=11, color="red" if p["iou"] < 0.5 else "black")
    fig.suptitle(f"{ep} — stage 5 object mesh — {status} — {obj}, chosen mesh from video frame "
                 f"{choice['chosen_video_frame']} (blue = mesh fitted on that frame, green = SAM2 mask)", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(review_dir, exist_ok=True)
    sheet = os.path.join(review_dir, f"{ep}_mesh.png")
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
        if not os.path.exists(os.path.join(W, "mesh", "choice.json")):
            for old in glob.glob(os.path.join(review, f"episode_{e:06d}_*.png")):   # stale sheet removed
                if not any(t in os.path.basename(old) for t in ("_before_", "_v2_", "_v3_", "_old")):
                    os.remove(old)
            results.append({"episode": e, "status": "FAIL", "checks": [{"check": "stage 5 output", "status": "FAIL",
                            "detail": "missing (see the episode log)"}]})
            continue
        chk_path = os.path.join(args.stage_dir, f"episode_{e:06d}.checks.json")
        key_out = os.path.join(W, "mesh", "choice.json")
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
    out = ["<html><head><meta charset='utf-8'><title>Stage 5 object mesh</title><style>body{font-family:"
           "sans-serif;background:#111;color:#eee} td,th{border:1px solid #444;padding:3px 7px} table{border-"
           "collapse:collapse} img{max-width:100%}</style></head><body><h1>Stage 5c — per-video candidate meshes "
           "(SAM3D + scale)</h1><p>These are each video's OWN candidates. The final mesh is chosen across the "
           "object's videos and validated in <a href='../05e_object/index.html'>05e_object</a>.</p><p>Check: the mesh views look like the object (shape, proportions, parts like handles and wheels); the "
           "blue silhouette matches the green mask outline on frames the mesh was NOT built from.</p><table><tr>"
           "<th>episode</th><th>status</th><th>checks</th></tr>"]
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
