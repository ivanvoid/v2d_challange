#!/usr/bin/env python3
"""Stage 1 (prep) for one episode: trim -> undistort -> frames -> MoGe depth -> depth scale stabilisation -> intrinsics.

Writes into out/episode_XXXXXX/:
  video_trim.mp4            source frames [first-margin, last+margin] (lossless-ish H.264, crf 12)
  anycalib/                 AnyCalib intrinsics, distortion, undistorted.mp4
  frames/000000.png ...     undistorted frames; local frame i == video frame i + frame_offset
  depth_raw/, intrinsics/   MoGe per-frame depth (uint16 inverse) and intrinsics
  depth/                    depth_raw with MoGe's per-frame global scale swing removed (static
                            background held at its median depth); what all later stages read
  depth_scale.json          the per-frame scale factors and the swing that was removed
  background.npz            static background depth (1/4 resolution) and per-pixel static fraction
  floor.json                floor plane from the background (normal, offset, flatness, camera height/tilt)
  intrinsics_stable.json    temporal median of the per-frame intrinsics
  track1_meta.json          frame_offset, n_frames, scored span, source video

stdout gets one line per step (the progress log); everything the models print goes to
--debug_log. Each step is skipped when its output is complete (exact frame counts).

Run with the pipeline interpreter and V2D_BACKEND=venv:

    python track1/stage1_prep.py --episode 28 --out_root out --weights weights \
        --spans submission/scored_spans.json --debug_log logs/stages/01_prep/episode_000028.debug.log
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from v2d.anycalib.docker.run_video_to_calibration import (
    run_video_to_calibration as run_anycalib_video_to_calibration,
)
from v2d.common.utils import extract_images
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.moge.docker.run_video_to_depth import run_video_to_depth as run_moge_depth

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import paths  # noqa: E402
VIDEO_DIR = str(paths.VIDEO_DIR)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@contextlib.contextmanager
def to_debug(path: str):
    """Send everything written to fd 1/2 (this process and its subprocesses) to the debug log."""
    sys.stdout.flush(); sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(path, "a") as fh:
        fh.write(f"\n===== {time.strftime('%H:%M:%S')} =====\n"); fh.flush()
        os.dup2(fh.fileno(), 1); os.dup2(fh.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(saved[0], 1); os.dup2(saved[1], 2)
            os.close(saved[0]); os.close(saved[1])


def n_files(d: str, ext: str) -> int:
    return len([f for f in os.listdir(d) if f.endswith(ext)]) if os.path.isdir(d) else 0


def count_frames(video: str) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", video],
                         capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


def step(name: str, done: bool, fn, debug_log: str) -> float:
    if done:
        log(f"  [skip] {name} (already complete)")
        return 0.0
    log(f"  [run ] {name}")
    t0 = time.time()
    with to_debug(debug_log):
        fn()
    dt = time.time() - t0
    log(f"  [done] {name} in {dt:.0f}s")
    return dt


def read_depth_px(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def depth_from_px(px: np.ndarray) -> np.ndarray:
    """uint16 inverse depth (v2d convention: px = 65535 / (depth_m + 1)) -> metres; 0 -> NaN (infinity)."""
    d = 65535.0 / np.maximum(px.astype(np.float32), 1e-6) - 1.0
    d[px == 0] = np.nan
    return d


def stabilize_depth(frames: str, raw: str, out: str, report: str, scored: tuple[int, int],
                    ds: int = 4, min_static: float = 0.2) -> None:
    """Remove MoGe's per-frame global depth scale swing using the static background.

    The camera and the background are static, so the background must have the same depth in every
    frame. Static pixels = colour within 20 of the episode's median background image (moving
    pixels dilated by 7 px at 1/ds resolution) with valid depth. Per frame, the scale is the median
    of log(depth / background depth) over static pixels (background re-estimated once after the
    first scale estimate). Frames with less than min_static static pixels get the scale
    interpolated from their neighbours. Each raw map is divided by its scale and written to `out`
    in the same encoding; zero (infinite-depth) pixels stay zero.
    """
    names = sorted(f for f in os.listdir(raw) if f.endswith(".png"))
    n = len(names)
    im0 = read_depth_px(os.path.join(raw, names[0]))
    H, W = im0.shape[0] // ds, im0.shape[1] // ds
    rgb = np.zeros((n, H, W, 3), np.uint8)
    logd = np.full((n, H, W), np.nan, np.float32)
    for i, name in enumerate(names):
        rgb[i] = cv2.resize(cv2.imread(os.path.join(frames, name)), (W, H), interpolation=cv2.INTER_AREA)
        d = cv2.resize(depth_from_px(read_depth_px(os.path.join(raw, name))), (W, H), interpolation=cv2.INTER_NEAREST)
        logd[i] = np.log(d)
    bg = np.median(rgb[::max(1, n // 150)], axis=0).astype(np.int16)
    kernel = np.ones((7, 7), np.uint8)
    static = np.stack([~cv2.dilate((np.abs(f.astype(np.int16) - bg).max(-1) > 20).astype(np.uint8), kernel).astype(bool)
                       for f in rgb]) & np.isfinite(logd)
    frac = static.reshape(n, -1).mean(1)
    scale = np.zeros(n)
    for _ in range(2):
        ref = np.nanmedian(np.where(static, logd - scale[:, None, None], np.nan), axis=0)
        scale = np.nanmedian(np.where(static, logd - ref[None], np.nan).reshape(n, -1), axis=1)
    ok = (frac >= min_static) & np.isfinite(scale)
    if not ok.any():
        raise SystemExit("depth stabilisation: no frame has enough static background")
    scale = np.interp(np.arange(n), np.flatnonzero(ok), scale[ok])
    factor = np.exp(scale)
    np.savez_compressed(os.path.join(os.path.dirname(out), "background.npz"),
                        depth=np.exp(ref).astype(np.float32), static_fraction=static.mean(0).astype(np.float32),
                        downsample=ds)

    tmp = out + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    for i, name in enumerate(names):
        px = read_depth_px(os.path.join(raw, name))
        d = depth_from_px(px) / factor[i]
        new = np.where(px == 0, 0, np.clip(np.rint(65535.0 / (d + 1.0)), 1, 65535)).astype(np.uint16)
        cv2.imwrite(os.path.join(tmp, name), new)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.replace(tmp, out)

    s = slice(*scored)
    pct = 100 * (factor[s] - 1)
    rep = {"method": "per-frame median log ratio to the median static background", "downsample": ds,
           "min_static_fraction": min_static,
           "frames_interpolated": int((~ok).sum()),
           "scored_swing_pct": {"p5": float(np.percentile(pct, 5)), "p95": float(np.percentile(pct, 95)),
                                "min": float(pct.min()), "max": float(pct.max())},
           "scored_jitter_pct_median_abs_diff": float(np.median(np.abs(np.diff(pct)))) if len(pct) > 1 else 0.0,
           "factor": [round(float(f), 6) for f in factor], "static_fraction": [round(float(f), 3) for f in frac]}
    json.dump(rep, open(report, "w"), indent=1)
    print(f"depth stabilisation: scored swing {pct.min():+.1f}..{pct.max():+.1f}%, jitter "
          f"{rep['scored_jitter_pct_median_abs_diff']:.2f}%/frame, {rep['frames_interpolated']} frames interpolated")


def fit_floor(background: str, intr_path: str, out: str, thr: float = 0.03, iters: int = 400) -> None:
    """Floor plane from the static background depth (background.npz, stabilised scale).

    Candidates: pixels in the lower 45 % of the image that are static in >= 30 % of frames. RANSAC
    (inlier distance `thr`), then a least-squares refit on the inliers. Plane: n . X = d with n the unit
    normal pointing down (+y in the OpenCV camera frame), X in metres. Also reported: RMS of the inliers,
    the angle between planes fitted to the near and the far half (bend), the fraction of candidates on
    the plane, the camera height above it and the camera's downward tilt.
    """
    b = np.load(background)
    D, frac, ds = b["depth"].astype(np.float64), b["static_fraction"], int(b["downsample"])
    intr = json.load(open(intr_path))
    fx, fy, cx, cy = intr["fx"] / ds, intr["fy"] / ds, intr["cx"] / ds, intr["cy"] / ds
    H, W = D.shape
    v, u = np.mgrid[0:H, 0:W]
    P = np.stack([(u - cx) / fx * D, (v - cy) / fy * D, D], -1)
    cand = (v > 0.55 * H) & np.isfinite(D) & (frac >= 0.3)
    pts = P[cand]
    if len(pts) < 100:
        raise SystemExit(f"floor: only {len(pts)} static lower-image pixels")
    rng = np.random.default_rng(0)
    best_k, best = 0, None
    for _ in range(iters):
        a, b_, c = pts[rng.choice(len(pts), 3, replace=False)]
        nrm = np.cross(b_ - a, c - a)
        if np.linalg.norm(nrm) < 1e-9:
            continue
        nrm /= np.linalg.norm(nrm)
        k = int((np.abs((pts - a) @ nrm) < thr).sum())
        if k > best_k:
            best_k, best = k, (a, nrm)

    def plane(q):
        c0 = q.mean(0)
        nn = np.linalg.svd(q - c0, full_matrices=False)[2][-1]
        return c0, nn * (1 if nn[1] >= 0 else -1)
    a, nrm = best
    inl = np.abs((pts - a) @ nrm) < thr
    c0, nrm = plane(pts[inl])
    inl = np.abs((pts - c0) @ nrm) < thr
    c0, nrm = plane(pts[inl])
    q = pts[inl]
    res = (q - c0) @ nrm
    zm = np.median(q[:, 2])
    bend = float(np.degrees(np.arccos(min(1.0, abs(plane(q[q[:, 2] < zm])[1] @ plane(q[q[:, 2] >= zm])[1])))))
    rep = {"normal": nrm.tolist(), "d": float(nrm @ c0), "rms_cm": float(np.sqrt(np.mean(res ** 2)) * 100),
           "bend_deg": bend, "inlier_fraction": float(inl.mean()), "n_inliers": int(inl.sum()),
           "camera_height_m": float(abs(nrm @ c0)), "camera_tilt_down_deg": float(np.degrees(np.arccos(abs(nrm[1])))),
           "depth_range_m": [float(q[:, 2].min()), float(q[:, 2].max())], "inlier_threshold_m": thr,
           "note": "plane n.X = d, n points down (+y), OpenCV camera frame, stabilised MoGe scale"}
    json.dump(rep, open(out, "w"), indent=1)
    print(f"floor: camera {rep['camera_height_m']:.2f} m above, tilted {rep['camera_tilt_down_deg']:.1f} deg down; "
          f"RMS {rep['rms_cm']:.1f} cm, bend {bend:.1f} deg, {100 * rep['inlier_fraction']:.0f}% of candidates")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--spans", required=True)
    ap.add_argument("--debug_log", required=True)
    ap.add_argument("--margin", type=int, default=10)
    args = ap.parse_args()

    ep = f"episode_{args.episode:06d}"
    out = os.path.abspath(os.path.join(args.out_root, ep))
    W = os.path.abspath(args.weights)
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.debug_log)), exist_ok=True)
    video = os.path.join(VIDEO_DIR, f"{ep}.mp4")
    span = json.load(open(args.spans))[str(args.episode)]

    n_video = count_frames(video)
    lo = max(0, span["first"] - args.margin)
    hi = min(n_video - 1, span["last"] + args.margin)
    n = hi - lo + 1
    log(f"{ep}: video {n_video} frames, scored {span['first']}..{span['last']} "
        f"({span['n_scored']} scored), keeping {lo}..{hi} ({n} frames)")
    meta = {"episode": args.episode, "source_video": video, "n_video_frames": n_video,
            "frame_offset": lo, "n_frames": n, "scored_first": span["first"],
            "scored_last": span["last"], "n_scored": span["n_scored"], "margin": args.margin}
    meta_path = os.path.join(out, "track1_meta.json")
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        if (old.get("frame_offset"), old.get("n_frames")) != (lo, n):
            raise SystemExit(f"{out} was built with another frame window; delete it to rebuild")
    json.dump(meta, open(meta_path, "w"), indent=1)

    trimmed = os.path.join(out, "video_trim.mp4")
    anycalib = os.path.join(out, "anycalib")
    undist = os.path.join(anycalib, "undistorted.mp4")
    undist_intr = os.path.join(anycalib, "undistorted_intrinsics.json")
    frames, depth, intr = (os.path.join(out, d) for d in ("frames", "depth", "intrinsics"))
    depth_raw = os.path.join(out, "depth_raw")
    depth_report = os.path.join(out, "depth_scale.json")
    intr_stable = os.path.join(out, "intrinsics_stable.json")
    timings = {}
    # episodes made before depth stabilisation: their depth/ is the raw MoGe output
    if not os.path.isdir(depth_raw) and not os.path.exists(depth_report) and n_files(depth, ".png") == n:
        os.rename(depth, depth_raw)
        log("  moved the existing MoGe depth/ to depth_raw/ (made before depth stabilisation)")

    def trim():
        tmp = trimmed + ".tmp.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
                        "-vf", f"select='between(n\\,{lo}\\,{hi})'", "-vsync", "0",
                        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", tmp], check=True)
        os.replace(tmp, trimmed)

    def undistort():
        os.makedirs(anycalib, exist_ok=True)
        run_anycalib_video_to_calibration(
            video_path=trimmed, intrinsics_path=os.path.join(anycalib, "intrinsics.json"),
            distortion_path=os.path.join(anycalib, "distortion.json"), weights_path=f"{W}/anycalib",
            undistorted_video_path=undist, undistorted_intrinsics_path=undist_intr, dev=False)

    timings["trim"] = step("trim video", os.path.exists(trimmed) and count_frames(trimmed) == n, trim,
                           args.debug_log)
    if count_frames(trimmed) != n:
        raise SystemExit(f"trim produced {count_frames(trimmed)} frames, expected {n}")
    timings["undistort"] = step("AnyCalib undistortion", os.path.exists(undist), undistort, args.debug_log)
    timings["frames"] = step("extract frames", n_files(frames, ".png") == n,
                             lambda: extract_images(undist, frames), args.debug_log)
    if n_files(frames, ".png") != n:
        raise SystemExit(f"frames: {n_files(frames, '.png')} extracted, expected {n}")
    timings["depth"] = step("MoGe depth + intrinsics", n_files(depth_raw, ".png") == n,
                            lambda: run_moge_depth(video_path=undist, depth_folder=depth_raw,
                                                   intrinsics_folder=intr, weights_path=f"{W}/moge",
                                                   input_intrinsics_path=undist_intr, dev=False),
                            args.debug_log)
    if n_files(depth_raw, ".png") != n:
        raise SystemExit(f"depth_raw: {n_files(depth_raw, '.png')} maps, expected {n} (truncated MoGe run?)")
    scored = (span["first"] - lo, span["last"] - lo + 1)
    timings["stabilize"] = step("stabilise depth scale (static background)",
                                os.path.exists(depth_report) and n_files(depth, ".png") == n
                                and os.path.getmtime(depth_report) > os.path.getmtime(depth_raw),
                                lambda: stabilize_depth(frames, depth_raw, depth, depth_report, scored),
                                args.debug_log)
    if n_files(depth, ".png") != n:
        raise SystemExit(f"depth: {n_files(depth, '.png')} stabilised maps, expected {n}")
    sw = json.load(open(depth_report))
    log(f"  depth scale swing removed: {sw['scored_swing_pct']['min']:+.1f}..{sw['scored_swing_pct']['max']:+.1f}% "
        f"(jitter {sw['scored_jitter_pct_median_abs_diff']:.2f}%/frame, {sw['frames_interpolated']} frames interpolated)")
    floor = os.path.join(out, "floor.json")
    background = os.path.join(out, "background.npz")
    if not os.path.exists(background):          # made by older stabilisation runs without it: redo that step
        stabilize_depth(frames, depth_raw, depth, depth_report, scored)
    timings["intrinsics"] = step("stabilise intrinsics", os.path.exists(intr_stable),
                                 lambda: stabilize_intrinsics(intr, intr_stable), args.debug_log)
    timings["floor"] = step("floor plane (static background)",
                            os.path.exists(floor) and os.path.getmtime(floor) > os.path.getmtime(background),
                            lambda: fit_floor(background, intr_stable, floor), args.debug_log)
    fl = json.load(open(floor))
    log(f"  floor: camera {fl['camera_height_m']:.2f} m above, tilted {fl['camera_tilt_down_deg']:.1f} deg; "
        f"RMS {fl['rms_cm']:.1f} cm, bend {fl['bend_deg']:.1f} deg")

    meta["stage1_seconds"] = {k: round(v, 1) for k, v in timings.items()}
    json.dump(meta, open(meta_path, "w"), indent=1)
    log(f"{ep}: stage 1 OK — {n} frames, {n_files(depth, '.png')} depth maps, "
        f"intrinsics {json.load(open(intr_stable))}")


if __name__ == "__main__":
    main()
