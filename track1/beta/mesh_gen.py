#!/usr/bin/env python3
"""Beta stage 5b — candidate object meshes from TRELLIS.2 or Hunyuan3D-2 instead of SAM3D.

Drop-in for stage5_mesh.py: for each frame in mesh/candidates.json (stage5_pick.py), in mesh/cand_<frame>/:
  input_rgba.png                     the frame cropped around the stage-4 object mask, mask as alpha
                                     (the models' own background removal is not used)
  textured_mesh.obj + material files the generated, textured mesh (canonical orientation, unit size)
  mesh_pretransformed.obj            the same mesh: there is no camera alignment to apply — FoundationPose
                                     finds the orientation in stage5_scale.py, which also centres the mesh
  scale_init.json                    starting scale, rotation-free (the mesh orientation is unknown):
                                     sqrt(mask area in m^2 / mean projected area of the mesh); the mean
                                     projected area of a convex body is its surface area / 4 (Cauchy), taken on
                                     the convex hull. Search range [init/2, init*2] (wider than SAM3D's 1.6:
                                     the hand hides part of the mask)
stage5_scale.py, 5c, 5e and everything after run unchanged. Skips a candidate whose textured_mesh.obj exists.

  trellis2   TRELLIS.2-4B (weights/trellis2), DINOv3 ViT-L from weights/trellis2/dinov3-vitl16-pretrain-lvd1689m,
             --pipeline_type 1024_cascade (needs ~24 GB), GLB export decimated to --faces, texture --tex_size
  hunyuan    Hunyuan3D-2 DiT v2.0 shape (+ floater / degenerate-face removal, reduced to --faces) and
             Hunyuan3D-Paint v2.0 texture (weights/hunyuan3d/Hunyuan3D-2)

    /opt/venvs/trellis2/bin/python track1/beta/mesh_gen.py --model trellis2 --episode 6 \
        --out_root out_beta/<tag> --weights weights --debug_log ...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logutil import log, to_debug  # noqa: E402
import paths  # noqa: E402


def rgba_crop(frame: Path, mask: Path, margin: float = 0.15):
    from PIL import Image
    img = cv2.cvtColor(cv2.imread(str(frame)), cv2.COLOR_BGR2RGB)
    m = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE) > 0
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        raise SystemExit(f"mask {mask} is (nearly) empty")
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(np.ptp(xs), np.ptp(ys)) * (0.5 + margin)
    x0, y0, x1, y1 = int(cx - half), int(cy - half), int(cx + half) + 1, int(cy + half) + 1
    H, W = m.shape
    out = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0, :3] = img[sy0:sy1, sx0:sx1]
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0, 3] = m[sy0:sy1, sx0:sx1] * 255
    return Image.fromarray(out, "RGBA")


def to_simple_material(mesh):
    """PBR (GLB) material -> plain diffuse texture, which the OBJ export and FoundationPose use."""
    import trimesh
    mat = getattr(mesh.visual, "material", None)
    if isinstance(mat, trimesh.visual.material.PBRMaterial):
        tex = mat.baseColorTexture
        img = tex.convert("RGB") if tex is not None else None
        mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv,
                                                    material=trimesh.visual.material.SimpleMaterial(image=img))
    return mesh


class Trellis2:
    def __init__(self, weights: Path, repo: Path, pipeline_type: str, faces: int, tex_size: int):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        sys.path.insert(0, str(repo))
        from trellis2.modules import image_feature_extractor as ife
        from trellis2.pipelines import rembg, Trellis2ImageTo3DPipeline
        dino = weights / "trellis2" / "dinov3-vitl16-pretrain-lvd1689m"
        base = ife.DinoV3FeatureExtractor

        class LocalDino(base):              # the config names the gated HF repo; use the local copy
            def __init__(self, model_name, **kw):
                super().__init__(str(dino), **kw)
        ife.DinoV3FeatureExtractor = LocalDino
        rembg.BiRefNet = lambda **kw: None  # never used: the input has the mask as alpha
        self.pipe = Trellis2ImageTo3DPipeline.from_pretrained(str(weights / "trellis2" / "TRELLIS.2-4B"))
        self.pipe.cuda()
        self.pipeline_type, self.faces, self.tex_size = pipeline_type, faces, tex_size

    def __call__(self, image, seed: int):
        import o_voxel
        mesh = self.pipe.run(image, seed=seed, pipeline_type=self.pipeline_type)[0]
        mesh.simplify(16777216)
        return o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
            attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=self.faces, texture_size=self.tex_size, remesh=True, remesh_band=1,
            remesh_project=0, verbose=True)


class Hunyuan:
    def __init__(self, weights: Path, repo: Path, faces: int):
        sys.path.insert(0, str(repo))
        from hy3dgen.shapegen import (DegenerateFaceRemover, FaceReducer, FloaterRemover,
                                      Hunyuan3DDiTFlowMatchingPipeline)
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        root = str(weights / "hunyuan3d" / "Hunyuan3D-2")    # absolute: no hub lookup
        self.shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(root)
        self.paint = Hunyuan3DPaintPipeline.from_pretrained(root)
        self.post = (FloaterRemover(), DegenerateFaceRemover(), FaceReducer())
        self.faces = faces

    def __call__(self, image, seed: int):
        import torch
        mesh = self.shape(image=image, generator=torch.manual_seed(seed))[0]
        mesh = self.post[1](self.post[0](mesh))
        mesh = self.post[2](mesh, max_facenum=self.faces)
        return self.paint(mesh, image=image)


def init_scale(C: Path, W: Path, f: int, K: dict) -> dict:
    import trimesh
    tm = trimesh.load(C / "mesh_pretransformed.obj", force="mesh", process=False)
    m = cv2.imread(str(W / "masks" / "1" / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
    px = cv2.imread(str(W / "depth" / f"{f:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    d = 65535.0 / np.maximum(px, 1e-6) - 1.0
    ok = m & (px > 0)
    z = float(np.median(d[ok]))
    area_m2 = float(m.sum()) * z * z / (K["fx"] * K["fy"])
    proj = float(tm.convex_hull.area) / 4.0
    s0 = float(np.sqrt(area_m2 / proj))
    ys, xs = np.nonzero(m)
    ext = tm.extents * s0
    r = {"init_scale": s0, "lo": s0 / 2.0, "hi": s0 * 2.0, "method": "sqrt(mask area / (hull area / 4))",
         "mask_area_m2": area_m2, "mesh_mean_projected_area": proj, "mask_depth_m": z,
         "mask_wh_m": [float(np.ptp(xs)) * z / K["fx"], float(np.ptp(ys)) * z / K["fy"]],
         "mesh_wh_m": [float(ext[0]), float(ext[1])], "mesh_extents_at_init_m": [float(x) for x in ext]}
    json.dump(r, open(C / "scale_init.json", "w"), indent=1)
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["trellis2", "hunyuan"], required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--debug_log", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--faces", type=int, default=100000)
    ap.add_argument("--tex_size", type=int, default=2048)
    ap.add_argument("--pipeline_type", default="1024_cascade")
    a = ap.parse_args()

    ep = f"episode_{a.episode:06d}"
    W = Path(a.out_root).resolve() / ep
    Wt = Path(a.weights).resolve()
    off = json.load(open(W / "track1_meta.json"))["frame_offset"]
    K = json.load(open(W / "intrinsics_stable.json"))
    frames = json.load(open(W / "mesh" / "candidates.json"))["chosen"]
    todo = [f for f in frames if not (W / "mesh" / f"cand_{f:06d}" / "textured_mesh.obj").exists()]

    if todo:
        log(f"  [run ] {a.model} for {len(todo)} candidate(s) (video frames {[f + off for f in todo]})")
        t0 = time.time()
        with to_debug(a.debug_log, f"{ep} {a.model} load"):
            gen = (Trellis2(Wt, paths.DATA / "models" / "TRELLIS.2", a.pipeline_type, a.faces, a.tex_size)
                   if a.model == "trellis2" else Hunyuan(Wt, paths.DATA / "models" / "Hunyuan3D-2", a.faces))
        log(f"  model loaded in {time.time() - t0:.0f}s")
        for f in todo:
            C = W / "mesh" / f"cand_{f:06d}"
            C.mkdir(parents=True, exist_ok=True)
            t1 = time.time()
            img = rgba_crop(W / "frames" / f"{f:06d}.png", W / "masks" / "1" / f"{f:06d}.png")
            img.save(C / "input_rgba.png")
            with to_debug(a.debug_log, f"{ep} {a.model} video frame {f + off}"):
                mesh = to_simple_material(gen(img, a.seed))
                tmp = C / "gen_tmp"
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir()
                mesh.export(tmp / "textured_mesh.obj")
                for p in tmp.iterdir():
                    os.replace(p, C / p.name)
                tmp.rmdir()
            log(f"  video frame {f + off}: {a.model} mesh {len(mesh.faces)} faces in {time.time() - t1:.0f}s")
        del gen
    for f in frames:
        C = W / "mesh" / f"cand_{f:06d}"
        tag = f"video frame {f + off}"
        if not (C / "textured_mesh.obj").exists():
            raise SystemExit(f"{ep}: {a.model} produced no mesh for {tag}")
        if not (C / "mesh_pretransformed.obj").exists():
            shutil.copy2(C / "textured_mesh.obj", C / "mesh_pretransformed.obj")
        si = json.load(open(C / "scale_init.json")) if (C / "scale_init.json").exists() else init_scale(C, W, f, K)
        tex = any(p.suffix.lower() in (".png", ".jpg") and p.name != "input_rgba.png" for p in C.iterdir())
        log(f"  {tag}: mask {si['mask_area_m2'] * 1e4:.0f} cm^2 at {si['mask_depth_m']:.2f} m -> start scale "
            f"{si['init_scale']:.3f}, search {si['lo']:.3f}-{si['hi']:.3f}; texture {'yes' if tex else 'NO'}")


if __name__ == "__main__":
    main()
