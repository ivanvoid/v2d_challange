#!/usr/bin/env python3
"""Stage 6 review video: the tracked object drawn over the video, at reduced resolution (CPU only).

Per frame: the textured mesh at the chosen pose blended over the photo, the SAM2 mask outline (green), a header
(video frame, state, IoU, pose source) and a timeline strip (state colours, anchors, a cursor).
With --compare poses the raw track is drawn as a cyan outline over the smoothed mesh.
The frame border has the state colour: green observed, orange tracked, red lost, grey occluded; magenta =
interpolated by stage 7 (hidden or outlier frame). Mesh parts behind the body (MHR mesh depth) are drawn faintly.

Needs no GPU, so it can run next to a GPU job. Reads poses/ and track/states.json (raw stage 6) or, with
--poses poses_smoothed / poses_final, the later stages (states are still those of stage 6).

    /opt/venvs/sam3d_body/bin/python track1/render_stage6_video.py --episodes 4 7 --out_root out
    ... --scale 0.35 --crop      # smaller and faster; --crop follows the object (zoomed)
Writes out/<ep>/track/overlay.mp4 (or --out).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import cv2
import numpy as np
import trimesh

from review_stage6 import STATE_COL, load_pose

BGR = {k: tuple(int(c[i:i + 2], 16) for i in (5, 3, 1)) for k, c in STATE_COL.items()}


def draw_mesh(pts, z, fcol, hw, depth=False):
    col = np.zeros((*hw, 3), np.uint8)
    hit = np.zeros(hw, np.uint8)
    dep = np.full(hw, np.inf, np.float32) if depth else None
    for i in np.argsort(-z):
        cv2.fillConvexPoly(col, pts[i], fcol[i])
        cv2.fillConvexPoly(hit, pts[i], 1)
        if depth:
            cv2.fillConvexPoly(dep, pts[i], float(z[i]))
    dep = None if dep is None else np.where(hit > 0, dep, np.inf)
    return (col, hit > 0, dep) if depth else (col, hit > 0)


def body_depth(bv, bf, K, hw):
    """Depth map of the body mesh (nearest surface wins; inf where there is no body)."""
    pc = bv
    uvw = pc @ K.T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
    pts = np.round(uv[bf]).astype(np.int32)
    z = pc[bf][:, :, 2].mean(1)
    dep = np.full(hw, np.inf, np.float32)
    for i in np.argsort(-z):
        cv2.fillConvexPoly(dep, pts[i], float(z[i]))
    return dep


def render_episode(W, a):
    ep = os.path.basename(W)
    st = json.load(open(os.path.join(W, "track", "states.json")))
    rows = {r["frame"]: r for r in st["frames"]}
    meta = json.load(open(os.path.join(W, "track1_meta.json")))
    n = meta["n_frames"]
    it = json.load(open(os.path.join(W, "intrinsics_stable.json")))
    s = a.scale
    K = np.array([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]], [0, 0, 1]]) * s
    K[2, 2] = 1
    H, Wd = int(round(it["height"] * s)) // 2 * 2, int(round(it["width"] * s)) // 2 * 2
    tm = trimesh.load(os.path.join(W, "mesh_scaled.obj"), force="mesh", process=False)
    v, fa = np.asarray(tm.vertices), np.asarray(tm.faces)
    vis = tm.visual.to_color() if type(tm.visual).__name__ == "TextureVisuals" else tm.visual
    fcol = [tuple(int(c) for c in x) for x in np.asarray(vis.vertex_colors)[:, :3][fa].mean(1)[:, ::-1]]
    anchors = [x["frame"] for x in st.get("anchors", [])]
    bv = bf = None
    if a.occlusion:
        import torch
        body = torch.load(os.path.join(W, "body", "mhr_mesh.pt"), map_location="cpu", weights_only=False)
        bv, bf = body["pred_vertices"].numpy(), body["faces"].numpy()
    interp = set()
    sp = os.path.join(W, "track", "smooth.json")
    if a.poses == "poses_smoothed" and os.path.exists(sp):
        interp = {r["frame"] for r in json.load(open(sp))["frames"] if r.get("interpolated")}
    states = [rows[f]["state"] if f in rows else "occluded" for f in range(n)]
    pose_dir = os.path.join(W, a.poses)
    pose_dir2 = os.path.join(W, a.compare) if a.compare else None
    out = a.out or os.path.join(W, "track", "overlay.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bar_h, strip_h = 26, 14
    if a.crop:
        side = 480
        size = (side, side + bar_h + strip_h)
    else:
        size = (Wd, H + bar_h + strip_h)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                           f"{size[0]}x{size[1]}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                           "-crf", str(a.crf), "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    strip = np.zeros((strip_h, size[0], 3), np.uint8)
    for f in range(n):
        strip[:, min(size[0] - 1, f * size[0] // n)] = BGR.get(states[f], (80, 80, 80))
    for f in anchors:
        strip[:4, min(size[0] - 1, f * size[0] // n)] = (255, 120, 0)
    c_prev = None
    for f in range(n):
        r = rows.get(f, {})
        img = cv2.resize(cv2.imread(os.path.join(W, "frames", f"{f:06d}.png")), (Wd, H), interpolation=cv2.INTER_AREA)
        m = cv2.resize(cv2.imread(os.path.join(W, "masks", "1", f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE), (Wd, H),
                       interpolation=cv2.INTER_NEAREST) > 0
        pp = os.path.join(pose_dir, f"{f:06d}.json")
        hit = np.zeros((H, Wd), bool)
        if os.path.exists(pp):
            M = load_pose(pp)
            pc = v @ M[:3, :3].T + M[:3, 3]
            uvw = pc @ K.T
            uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
            pts = np.round(uv[fa]).astype(np.int32)
            zf = pc[fa][:, :, 2].mean(1)
            if bv is not None:
                col, hit, dep = draw_mesh(pts, zf, fcol, (H, Wd), depth=True)
                behind = hit & (dep > body_depth(bv[f], bf, K, (H, Wd)) + 0.02)     # body in front of the object
                front = hit & ~behind
                img[front] = (0.3 * img[front] + 0.7 * col[front]).astype(np.uint8)
                img[behind] = (0.8 * img[behind] + 0.2 * col[behind]).astype(np.uint8)
            else:
                col, hit = draw_mesh(pts, zf, fcol, (H, Wd))
                img[hit] = (0.3 * img[hit] + 0.7 * col[hit]).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 255, 0), 1)
        if pose_dir2 and os.path.exists(os.path.join(pose_dir2, f"{f:06d}.json")):
            M2 = load_pose(os.path.join(pose_dir2, f"{f:06d}.json"))
            pc2 = v @ M2[:3, :3].T + M2[:3, 3]
            uvw2 = pc2 @ K.T
            uv2 = uvw2[:, :2] / np.clip(uvw2[:, 2:3], 1e-6, None)
            _, hit2 = draw_mesh(np.round(uv2[fa]).astype(np.int32), pc2[fa][:, :, 2].mean(1), fcol, (H, Wd))
            c2, _ = cv2.findContours(hit2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, c2, -1, (255, 200, 0), 1)
        if a.crop:
            ys, xs = np.nonzero(m | hit)
            if len(ys):
                c = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2])
                half = 1.2 * max(np.ptp(xs), np.ptp(ys)) + 30
                c_prev = c if c_prev is None else 0.8 * c_prev + 0.2 * c
                half = max(half, 60)
            elif c_prev is None:
                c_prev, half = np.array([Wd / 2, H / 2]), H / 2
            x0, y0 = int(c_prev[0] - half), int(c_prev[1] - half)
            pad = int(half) + 2
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
            img = cv2.resize(img[y0 + pad:y0 + pad + int(2 * half), x0 + pad:x0 + pad + int(2 * half)], (side, side))
        state = states[f]
        shown = "interpolated" if f in interp else state
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1),
                      (255, 0, 255) if f in interp else BGR.get(state, (80, 80, 80)), 3)
        bar = np.zeros((bar_h, size[0], 3), np.uint8)
        txt = (f"[{a.poses}{' vs ' + a.compare if a.compare else ''}] " if a.label else "") + f"{meta['frame_offset'] + f}  {shown}  IoU {r.get('iou', 0):.2f}" + ("" if a.crop else f"  {r.get('src', '')}")
        cv2.putText(bar, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        st_img = strip.copy()
        st_img[:, min(size[0] - 1, f * size[0] // n)] = (255, 255, 255)
        ff.stdin.write(np.vstack([bar, img, st_img]).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{ep}: {n} frames -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--poses", default="poses")
    ap.add_argument("--compare", default=None, help="second pose folder drawn as a cyan outline (e.g. poses = raw)")
    ap.add_argument("--no_occlusion", dest="occlusion", action="store_false",
                    help="do not hide the mesh behind the body (default: parts behind the body are drawn faintly)")
    ap.add_argument("--label", action="store_true", help="show the pose folder names in the header")
    ap.add_argument("--scale", type=float, default=0.4, help="image scale (1536x1152 x 0.4 = 614x460)")
    ap.add_argument("--crop", action="store_true", help="zoomed square view that follows the object")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=28)
    ap.add_argument("--out", default=None, help="only with one episode")
    a = ap.parse_args()
    for e in a.episodes:
        render_episode(os.path.join(os.path.abspath(a.out_root), f"episode_{e:06d}"), a)


if __name__ == "__main__":
    main()
