#!/usr/bin/env python3
"""Render a fused body-mesh + object overlay video (monocular, portable).

Composites the SAM3D body mesh (mhr_mesh*.pt: pred_vertices + faces) and the
tracked object (mesh + per-frame poses_smoothed/*.json) onto the source frames
under one pinhole camera, producing a single overlay mp4.

Both the body `pred_vertices` and the object vertices live in the SAME OpenCV
camera frame (+X right, +Y down, +Z forward). Rendering flips verts by
[1,-1,-1] (CV->GL) under an identity-posed IntrinsicsCamera -- the exact
convention used by v2d's Renderer and the hamer hands renderer.

Deps (all present in /opt/venvs/wilor): numpy, torch, trimesh, pyrender,
pillow, tqdm, and the `ffmpeg` binary.

Example (run inside the wilor venv on the target machine):

/opt/venvs/wilor/bin/python render_body_object_overlay.py \
    --frames_dir       $OUT/episode_000000/frames \
    --intrinsics       $OUT/episode_000000/intrinsics_stable.json \
    --body_mesh        $OUT/episode_000000/sam3d_body/mhr_mesh_mv.pt \
    --object_mesh      $OUT/episode_000000/mesh_scaled.obj \
    --object_poses_dir $OUT/episode_000000/poses_smoothed \
    --output           $OUT/episode_000000/body_object_overlay.mp4 \
    --fps 30
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import tempfile

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import torch
import trimesh
from PIL import Image
from tqdm import tqdm

_CV_TO_GL_VEC = np.array([1.0, -1.0, -1.0], dtype=np.float32)
_BODY_COLOR = (0.65, 0.74, 0.86)   # light blue
_OBJECT_COLOR = (0.90, 0.60, 0.25) # orange
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
        [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
    ], dtype=np.float32)


def _create_raymond_lights() -> list[pyrender.Node]:
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    nodes = []
    for phi, theta in zip(phis, thetas):
        z = np.array([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        x = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(x) == 0 else x / np.linalg.norm(x)
        y = np.cross(z, x)
        m = np.eye(4)
        m[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.5),
            matrix=m,
        ))
    return nodes


def _material(color) -> pyrender.MetallicRoughnessMaterial:
    return pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1, roughnessFactor=0.7, alphaMode="OPAQUE",
        baseColorFactor=(color[0], color[1], color[2], 1.0),
    )


def _load_intrinsics(path: str):
    with open(path) as f:
        d = json.load(f)
    return (float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]),
            int(d["width"]), int(d["height"]))


def _load_body(path: str):
    mesh = torch.load(path, map_location="cpu", weights_only=False)
    verts = mesh["pred_vertices"]
    faces = mesh["faces"]
    verts = verts.numpy() if torch.is_tensor(verts) else np.asarray(verts)
    faces = faces.numpy() if torch.is_tensor(faces) else np.asarray(faces)
    return verts.astype(np.float32), faces.astype(np.int64)


def _object_verts_cam(verts_base: np.ndarray, pose_path: str) -> np.ndarray:
    with open(pose_path) as f:
        pd = json.load(f)
    R = _quat_to_rot(*[float(v) for v in pd["rotation"]])
    t = np.asarray(pd["translation"], dtype=np.float32).reshape(3)
    s = np.asarray(pd["scale"], dtype=np.float32).reshape(-1)
    s = s if s.size == 3 else np.full(3, float(s[0]), dtype=np.float32)
    return (verts_base * s) @ R.T + t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames_dir", required=True, help="Dir of source frames (sorted).")
    ap.add_argument("--intrinsics", required=True, help="intrinsics_stable JSON.")
    ap.add_argument("--output", required=True, help="Output mp4 path.")
    ap.add_argument("--body_mesh", default=None, help="mhr_mesh*.pt (pred_vertices + faces).")
    ap.add_argument("--object_mesh", default=None, help="Object mesh (e.g. mesh_scaled.obj).")
    ap.add_argument("--object_poses_dir", default=None,
                    help="Dir with per-frame {idx:06d}.json object poses.")
    ap.add_argument("--body_convention", choices=["cv", "gl"], default="cv",
                    help="Frame of body pred_vertices. 'cv' (default) flips Y/Z to GL "
                         "like the object; use 'gl' if the body comes out mirrored/behind.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.9, help="Mesh opacity over frame.")
    ap.add_argument("--frame_start", type=int, default=0, help="First frame index to render.")
    ap.add_argument("--frame_end", type=int, default=None, help="Last frame index to render (inclusive).")
    ap.add_argument("--frames", default=None,
                    help="Comma-separated frame indices to render instead of a range. With an --output "
                         "ending in .png, writes one tiled review sheet instead of a video.")
    ap.add_argument("--title", default="", help="Title for a .png review sheet.")
    args = ap.parse_args()

    if not args.body_mesh and not args.object_mesh:
        ap.error("Provide at least one of --body_mesh / --object_mesh.")

    fx, fy, cx, cy, W, H = _load_intrinsics(args.intrinsics)

    frame_files = sorted(
        p for p in glob.glob(os.path.join(args.frames_dir, "*"))
        if os.path.splitext(p)[1].lower() in _IMG_EXTS
    )
    if not frame_files:
        ap.error(f"No frames found in {args.frames_dir}")

    body_verts = body_faces = None
    if args.body_mesh:
        body_verts, body_faces = _load_body(args.body_mesh)
        if len(body_verts) != len(frame_files):
            print(f"[warn] body frames ({len(body_verts)}) != image frames "
                  f"({len(frame_files)}); body drawn where indices overlap.")

    obj_base = obj_faces = None
    if args.object_mesh:
        if not args.object_poses_dir:
            ap.error("--object_mesh requires --object_poses_dir.")
        om = trimesh.load(args.object_mesh, force="mesh", process=False)
        obj_base = np.asarray(om.vertices, dtype=np.float32)
        obj_faces = np.asarray(om.faces, dtype=np.int64)

    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)
    lights = _create_raymond_lights()

    with tempfile.TemporaryDirectory() as tmp:
        if args.frames:
            selected = [int(v) for v in args.frames.split(",") if v.strip()]
            selected = [i for i in selected if 0 <= i < len(frame_files)]
        else:
            end = len(frame_files) - 1 if args.frame_end is None else min(args.frame_end, len(frame_files) - 1)
            selected = list(range(max(0, args.frame_start), end + 1))
        for out_idx, i in enumerate(tqdm(selected, desc="render", ncols=80)):
            fpath = frame_files[i]
            bg = np.asarray(Image.open(fpath).convert("RGB").resize((W, H)),
                            dtype=np.float32) / 255.0

            scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])
            scene.add(camera, pose=np.eye(4))
            for ln in lights:
                scene.add_node(ln)

            drew = False
            if body_verts is not None and i < len(body_verts):
                v = body_verts[i]
                if np.isfinite(v).all():
                    v_gl = v * _CV_TO_GL_VEC if args.body_convention == "cv" else v
                    tm = trimesh.Trimesh(v_gl, body_faces, process=False)
                    scene.add(pyrender.Mesh.from_trimesh(tm, material=_material(_BODY_COLOR),
                                                         smooth=True))
                    drew = True

            if obj_base is not None:
                pose_path = os.path.join(args.object_poses_dir, f"{i:06d}.json")
                if os.path.exists(pose_path):
                    v_gl = _object_verts_cam(obj_base, pose_path) * _CV_TO_GL_VEC
                    tm = trimesh.Trimesh(v_gl, obj_faces, process=False)
                    scene.add(pyrender.Mesh.from_trimesh(tm, material=_material(_OBJECT_COLOR),
                                                         smooth=True))
                    drew = True

            if drew:
                color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                color = color.astype(np.float32) / 255.0
                m = color[:, :, 3:4] * args.alpha
                out = color[:, :, :3] * m + bg * (1.0 - m)
            else:
                out = bg

            Image.fromarray((out * 255.0).clip(0, 255).astype(np.uint8)).save(
                os.path.join(tmp, f"{out_idx:06d}.png"))

        renderer.delete()
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        if args.output.lower().endswith(".png"):
            _write_sheet(tmp, selected, args.output, args.title)
        else:
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-r", str(args.fps),
                "-i", os.path.join(tmp, "%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                args.output,
            ], check=True)
    print(f"Saved → {args.output}")


def _write_sheet(tmp: str, frames: list[int], output: str, title: str, cols: int = 3,
                 scale: float = 0.5) -> None:
    """Tile the rendered frames into one labelled PNG for review."""
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    tiles = []
    for k, f in enumerate(frames):
        im = Image.open(os.path.join(tmp, f"{k:06d}.png")).convert("RGB")
        ImageDraw.Draw(im).text((14, 10), f"frame {f}", fill=(255, 255, 255), font=font)
        tiles.append(im.resize((int(im.width * scale), int(im.height * scale))))
    tw, th = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (tw * cols, th * rows + 50), (0, 0, 0))
    ImageDraw.Draw(sheet).text((10, 8), title, fill=(255, 255, 255), font=font)
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * tw, 50 + (k // cols) * th))
    sheet.save(output)


if __name__ == "__main__":
    main()