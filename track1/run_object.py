#!/usr/bin/env python3
"""Object-only Track 1 branch: trimmed video -> object mesh + per-frame pose.

Uses the same v2d stages as run_ego_wilor's object branch, without any of its
hand steps (which abort the whole run when WiLoR sees no hands, as it does on
these wide exo shots) and without its debug overlay videos.

  0. trim the video to the scored span (+ margin)      -> video_trim.mp4
  1. AnyCalib undistortion                              -> anycalib/undistorted.mp4
  2. frames                                             -> frames/
  3. MoGe depth + stable intrinsics                     -> depth/, intrinsics_stable.json
  4. actor-centred object selection (select_object.py)  -> object_track.json, dino_detections.json
  5. SAM2 object-only propagation (from the frames dir; the mp4 path segfaults)
                                                        -> masks/1/
  6. SAM3D mesh, SAM3D transform, FoundationPose scale  -> mesh_scaled.obj
  7. FoundationPose tracking + EKF smoothing            -> poses/, poses_smoothed/

Review sheets (--review_dir): <ep>_object_candidates.png after step 4 and
<ep>_masks.png after step 5.

Every stage is skipped when its output exists. Local frame i is video frame
i + frame_offset; frame_offset is recorded in track1_meta.json.

Run with the pipeline interpreter and V2D_BACKEND=venv:

    python track1/run_object.py --video .../episode_000012.mp4 --out_dir out/episode_000012/track1 \
        --prompt "a black frying pan." --object_name black_pan --first 69 --last 365 \
        --weights weights --review_dir logs/<run>/review --episode 12
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v2d.anycalib.docker.run_video_to_calibration import (  # noqa: E402
    run_video_to_calibration as run_anycalib_video_to_calibration,
)
from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts  # noqa: E402
from v2d.common.utils import extract_images  # noqa: E402
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics  # noqa: E402
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing  # noqa: E402
from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale  # noqa: E402
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses  # noqa: E402
from v2d.moge.docker.run_video_to_depth import run_video_to_depth as run_moge_depth  # noqa: E402
from v2d.pipelines.run_ego_wilor import _apply_sam3d_transform  # noqa: E402
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks  # noqa: E402
from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh  # noqa: E402

from select_object import OBJECT_ID, select_object  # noqa: E402
from viz_masks import render_masks, sample_frames  # noqa: E402


def step(label: str, done: bool) -> bool:
    print(f"  [{'skip' if done else 'run '}] {label}", flush=True)
    return done


def has_files(d: str) -> bool:
    return os.path.isdir(d) and any(os.scandir(d))


def n_files(d: str, ext: str = ".png") -> int:
    return len([f for f in os.listdir(d) if f.endswith(ext)]) if os.path.isdir(d) else 0


def count_frames(video: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", video],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


def trim_video(src: str, dst: str, lo: int, hi: int) -> None:
    tmp = dst + ".tmp.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-vf", f"select='between(n\\,{lo}\\,{hi})'", "-vsync", "0",
         "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", tmp],
        check=True)
    n = count_frames(tmp)
    if n != hi - lo + 1:
        raise RuntimeError(f"trim produced {n} frames, expected {hi - lo + 1}")
    os.replace(tmp, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--object_name", default=None, help="metadata 'object' field, e.g. black_pan")
    ap.add_argument("--first", type=int, required=True, help="first scored video frame")
    ap.add_argument("--last", type=int, required=True, help="last scored video frame")
    ap.add_argument("--margin", type=int, default=10)
    ap.add_argument("--no_trim", action="store_true")
    ap.add_argument("--no_undistort", action="store_true")
    ap.add_argument("--n_candidates", type=int, default=5, help="frames tried for object selection")
    ap.add_argument("--weights", required=True, help="weights root (anycalib, moge, ...)")
    ap.add_argument("--review_dir", default=None, help="write review sheets here")
    ap.add_argument("--episode", type=int, default=None, help="episode index, for sheet names")
    ap.add_argument("--ready_marker", default=None,
                    help="touch this file once frames + depth + intrinsics are ready (lets the body lane start)")
    ap.add_argument("--reuse_from", default=None,
                    help="earlier run dir whose frames/depth/intrinsics to reuse (no trim, offset 0)")
    args = ap.parse_args()

    out = os.path.abspath(args.out_dir)
    os.makedirs(out, exist_ok=True)
    W = os.path.abspath(args.weights)
    tag = f"episode_{args.episode:06d}" if args.episode is not None else os.path.basename(out)
    meta_path = os.path.join(out, "track1_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    anycalib_dir = os.path.join(out, "anycalib")
    undistorted = os.path.join(anycalib_dir, "undistorted.mp4")
    undistorted_intr = os.path.join(anycalib_dir, "undistorted_intrinsics.json")
    if args.reuse_from:
        # Reuse an earlier run's undistorted frames, depth and intrinsics (full video, offset 0);
        # only the object stages are redone.
        src = os.path.abspath(args.reuse_from)
        for name in ("frames", "depth", "intrinsics", "intrinsics_stable.json"):
            link = os.path.join(out, name)
            if not os.path.lexists(link):
                os.symlink(os.path.join(src, name), link)
        lo, n_expected = 0, n_files(os.path.join(src, "frames"))
        hi = n_expected - 1
        src_video = os.path.join(src, "anycalib", "undistorted.mp4")
        video = src_video if os.path.exists(src_video) else args.video
        print(f"  [reuse] frames/depth/intrinsics from {src} ({n_expected} frames)", flush=True)
        meta.update({"source_video": os.path.abspath(args.video), "reuse_from": src, "frame_offset": 0,
                     "n_frames": n_expected, "scored_first": args.first, "scored_last": args.last})
        json.dump(meta, open(meta_path, "w"), indent=1)
    else:
        # 0. Trim.
        n_video = count_frames(args.video)
        lo = 0 if args.no_trim else max(0, args.first - args.margin)
        hi = n_video - 1 if args.no_trim else min(n_video - 1, args.last + args.margin)
        if meta.get("frame_offset", lo) != lo or meta.get("n_frames", hi - lo + 1) != hi - lo + 1:
            raise SystemExit(f"{out} was built with a different trim window; remove it to rebuild")
        trimmed = os.path.join(out, "video_trim.mp4")
        if not step(f"Trim video to frames {lo}..{hi} of {n_video}", os.path.exists(trimmed)):
            trim_video(args.video, trimmed, lo, hi)
        meta.update({"source_video": os.path.abspath(args.video), "frame_offset": lo,
                     "n_frames": hi - lo + 1, "scored_first": args.first, "scored_last": args.last})
        json.dump(meta, open(meta_path, "w"), indent=1)
        video = trimmed
        n_expected = hi - lo + 1

    # 1. Undistort.
    if not args.no_undistort and not args.reuse_from:
        os.makedirs(anycalib_dir, exist_ok=True)
        if not step("AnyCalib undistortion", os.path.exists(undistorted)):
            run_anycalib_video_to_calibration(
                video_path=video,
                intrinsics_path=os.path.join(anycalib_dir, "intrinsics.json"),
                distortion_path=os.path.join(anycalib_dir, "distortion.json"),
                weights_path=f"{W}/anycalib",
                undistorted_video_path=undistorted,
                undistorted_intrinsics_path=undistorted_intr,
                dev=False)
        video = undistorted

    # 2-3. Frames, depth, intrinsics — with frame-count checks (a truncated stage must not pass).
    frames_dir = os.path.join(out, "frames")
    depth_dir = os.path.join(out, "depth")
    intr_dir = os.path.join(out, "intrinsics")
    intr_stable = os.path.join(out, "intrinsics_stable.json")
    if not step("Extract frames", n_files(frames_dir) == n_expected):
        extract_images(video, frames_dir)
    if n_files(frames_dir) != n_expected:
        raise RuntimeError(f"frames: {n_files(frames_dir)} extracted, expected {n_expected}")
    if not step("MoGe depth + intrinsics", n_files(depth_dir) == n_expected):
        run_moge_depth(video_path=video, depth_folder=depth_dir, intrinsics_folder=intr_dir,
                       weights_path=f"{W}/moge",
                       input_intrinsics_path=(None if (args.no_undistort or args.reuse_from)
                                              else undistorted_intr),
                       dev=False)
    if n_files(depth_dir) != n_expected:
        raise RuntimeError(f"depth: {n_files(depth_dir)} maps, expected {n_expected} (truncated MoGe run?)")
    if not step("Stabilise intrinsics", os.path.exists(intr_stable)):
        stabilize_intrinsics(intr_dir, intr_stable)
    if args.ready_marker:
        os.makedirs(os.path.dirname(os.path.abspath(args.ready_marker)), exist_ok=True)
        json.dump({"frames_dir": frames_dir, "intrinsics": intr_stable, "frame_offset": lo,
                   "n_frames": n_expected}, open(args.ready_marker, "w"))

    # 4. Actor-centred object selection.
    track_path = os.path.join(out, "object_track.json")
    dino_path = os.path.join(out, "dino_detections.json")
    review_cand = os.path.join(args.review_dir, f"{tag}_object_candidates.png") if args.review_dir else None
    print("  [run ] Object selection (actor-centred, near-only; cached detections reused)", flush=True)
    rec = select_object(frames_dir, depth_dir, out, max(0, args.first - lo),
                        min(n_expected - 1, args.last - lo), args.prompt, args.object_name, W,
                        args.n_candidates, review_png=review_cand)
    print(f"  selection: {rec['status']} mode={rec.get('mode')} prompt={rec.get('prompt')!r} "
          f"frame={rec.get('reference_frame')} conf={rec.get('confidence')} "
          f"hits={rec.get('candidate_hits')} dz={rec.get('depth_gap_m')}", flush=True)
    if rec["status"] != "ok":
        raise RuntimeError(f"object selection found nothing near the actor; see {review_cand}")
    track = json.load(open(track_path))
    ref = int(track["reference_frame"])
    ref_rgb = os.path.join(frames_dir, f"{ref:06d}.png")
    ref_depth = os.path.join(depth_dir, f"{ref:06d}.png")

    # 5. SAM2, object only, from the frames directory.
    masks_dir = os.path.join(out, "masks")
    obj_masks = os.path.join(masks_dir, str(OBJECT_ID))
    prompts_path = os.path.join(out, "sam2_prompts.json")
    seed = {"frame": ref, "box": json.load(open(dino_path))[0]["box"]}
    seed_path = os.path.join(out, "object_seed.json")
    if os.path.exists(seed_path) and json.load(open(seed_path)) != seed:
        # The object choice changed: every artefact derived from the old seed is stale.
        import shutil
        print(f"  [inval] object seed changed; removing masks, mesh and poses", flush=True)
        for d in (masks_dir, os.path.join(out, "mesh"), os.path.join(out, "poses"),
                  os.path.join(out, "poses_smoothed")):
            shutil.rmtree(d, ignore_errors=True)
        for f in ("mesh_pretransformed.obj", "mesh_scaled.obj", "scale.json", prompts_path):
            if os.path.exists(os.path.join(out, f)):
                os.remove(os.path.join(out, f))
    json.dump(seed, open(seed_path, "w"), indent=1)
    if not step("SAM2 object propagation", n_files(obj_masks) == n_expected):
        box = BoundingBox.from_dict(json.load(open(dino_path))[0]["box"])
        Sam2Prompts(prompts=[Sam2Prompt(frame_index=ref, object_id=OBJECT_ID, box=box)]).save(prompts_path)
        run_video_to_masks(video_path=frames_dir, prompts_path=prompts_path, masks_dir=masks_dir,
                           weights_dir=f"{W}/sam2", dev=False)
    if n_files(obj_masks) != n_expected:
        raise RuntimeError(f"masks: {n_files(obj_masks)} written, expected {n_expected}")
    if args.review_dir:
        render_masks(frames_dir, obj_masks,
                     sample_frames(max(0, args.first - lo), min(n_expected - 1, args.last - lo), 6),
                     os.path.join(args.review_dir, f"{tag}_masks.png"),
                     title=f"{tag} object mask (SAM2) | prompt: {track.get('prompt')} | seed frame {ref}")
    ref_mask = os.path.join(obj_masks, f"{ref:06d}.png")

    # 6. Mesh + metric scale.
    mesh_dir = os.path.join(out, "mesh")
    mesh_path = os.path.join(mesh_dir, "textured_mesh.obj")
    mesh_transform = os.path.join(mesh_dir, "mesh_transform.json")
    mesh_pre = os.path.join(out, "mesh_pretransformed.obj")
    mesh_scaled = os.path.join(out, "mesh_scaled.obj")
    os.makedirs(mesh_dir, exist_ok=True)
    if not step("SAM3D mesh generation", os.path.exists(mesh_path)):
        run_image_to_mesh(image_path=ref_rgb, mask_path=ref_mask, mesh_path=mesh_path,
                          transform_path=mesh_transform,
                          intrinsics_path=os.path.join(mesh_dir, "mesh_intrinsics.json"),
                          weights_dir=f"{W}/sam3d", with_texture_baking=True,
                          with_mesh_postprocess=True, depth_path=ref_depth,
                          depth_intrinsics_path=intr_stable, depth_mask_path=ref_mask, dev=False)
    if not step("Apply SAM3D transform", os.path.exists(mesh_pre)):
        _apply_sam3d_transform(mesh_path, mesh_transform, mesh_pre)
    if not step("FoundationPose scale estimation", os.path.exists(mesh_scaled)):
        run_estimate_mesh_scale(mesh_path=mesh_pre, rgb_path=ref_rgb, depth_path=ref_depth,
                                mask_path=ref_mask, intrinsics_path=intr_stable,
                                weights_dir=f"{W}/foundation_pose",
                                scale_path=os.path.join(out, "scale.json"),
                                rescaled_mesh_path=mesh_scaled, lo=0.5, hi=2.0, n_samples=9,
                                n_levels=4, iou_weight=1.0, depth_weight=1.0,
                                registration_iterations=5, dev=False)

    # 7. Pose tracking + smoothing.
    poses_dir = os.path.join(out, "poses")
    smooth_dir = os.path.join(out, "poses_smoothed")
    if not step("FoundationPose tracking", n_files(poses_dir, ".json") == n_expected):
        run_video_to_poses(video_path=video, depth_folder=depth_dir, masks_folder=obj_masks,
                           camera_intrinsics_path=intr_stable, mesh_path=mesh_scaled,
                           poses_dir=poses_dir, weights_dir=f"{W}/foundation_pose",
                           reference_frame=ref, mask_depth=True, reregister_iou_thresh=0.3,
                           dev=False)
    if n_files(poses_dir, ".json") != n_expected:
        raise RuntimeError(f"poses: {n_files(poses_dir, '.json')} written, expected {n_expected}")
    if not step("EKF pose smoothing", n_files(smooth_dir, ".json") == n_expected):
        run_ekf_smoothing(poses_dir=poses_dir, mesh_path=mesh_scaled, intrinsics_path=intr_stable,
                          weights_dir=f"{W}/foundation_pose", output_dir=smooth_dir,
                          masks_folder=obj_masks, process_noise_xy=0.01, process_noise_z=0.01,
                          process_noise_r=0.02, measurement_noise_xy=0.01, measurement_noise_z=0.04,
                          measurement_noise_r=0.02, dev=False)
    print(f"object branch done: {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
