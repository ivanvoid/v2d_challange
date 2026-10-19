#!/usr/bin/env python3
"""Fit Track 1 MHR parameters (one identity per episode) to SAM3D-Body meshes.

SAM3D-Body predicts a full MHR parameter vector per frame, so the body identity
(scales + shape) drifts from frame to frame. The submission needs exactly one
identity per episode. This script:

  1. loads the per-frame MHR vertices (`mhr_mesh.pt`, `pred_vertices`, which are
     already in the Track 1 frame: OpenCV camera coordinates, metres),
  2. keeps only the frames around the scored span,
  3. optionally smooths them in time (SAM3D-Body is per-frame and jittery,
     which is what ACC-H punishes),
  4. runs the kit's `mesh_to_mhr_params.convert` to get one shared identity plus a
     per-frame pose,
  5. writes `pose [last+1, 136]` indexed by VIDEO frame, plus `scales`, `shape`.

Run with the sam3d_body venv (torch + the MHR TorchScript model):

    /opt/venvs/sam3d_body/bin/python track1/fit_mhr.py \
        --mesh out/episode_000014/body/mhr_mesh.pt \
        --model weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt \
        --first 90 --last 387 --out out/episode_000014/body/mhr_fit.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
KIT_TRACK1 = paths.KIT_TRACK1
sys.path.insert(0, str(KIT_TRACK1))
from mesh_to_mhr_params import convert  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", type=Path, required=True, help="SAM3D-Body mhr_mesh.pt")
    ap.add_argument("--model", type=Path, required=True, help="MHR TorchScript model (lod1)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first", type=int, required=True, help="first scored VIDEO frame")
    ap.add_argument("--last", type=int, required=True, help="last scored VIDEO frame")
    ap.add_argument("--frame_offset", type=int, default=0,
                    help="video frame index of mesh frame 0 (non-zero for trimmed inputs)")
    ap.add_argument("--margin", type=int, default=10, help="extra frames fitted on each side of the span")
    ap.add_argument("--smooth_sigma", type=float, default=2.0,
                    help="temporal Gaussian sigma (frames) applied to vertices before fitting; 0 = off")
    ap.add_argument("--precision", default="float32", choices=["float32", "float64"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--profile", action="store_true",
                    help="time each fit phase and record every outlier re-solve -> <out>.profile.json (fit_profile.py)")
    ap.add_argument("--outlier_floor_mm", type=float, default=10.0,
                    help="a frame is re-solved as an outlier only if its mean vertex error is ALSO above this "
                         "(the kit's rule alone, 2 x median + 0.02 mm, flags hard-but-fine poses: on episode 1 it "
                         "re-solved 71 frames at ~3 mm error for 0.003-0.2 mm gain and 61%% of the fit time); "
                         "0 = the kit's rule unchanged")
    args = ap.parse_args()

    mesh = torch.load(args.mesh, map_location="cpu", weights_only=False)
    verts = mesh["pred_vertices"]
    verts = verts.numpy() if torch.is_tensor(verts) else np.asarray(verts)
    n_mesh = len(verts)

    # Mesh-local window covering the scored span (+ margin), clipped to what exists.
    lo = max(0, args.first - args.margin - args.frame_offset)
    hi = min(n_mesh - 1, args.last + args.margin - args.frame_offset)
    if args.last - args.frame_offset > n_mesh - 1 or args.first - args.frame_offset < 0:
        raise SystemExit(f"fit_mhr: mesh covers video frames {args.frame_offset}.."
                         f"{args.frame_offset + n_mesh - 1}, scored span is {args.first}..{args.last}")
    window = verts[lo:hi + 1].astype(np.float64)

    finite = np.isfinite(window).all(axis=(1, 2))
    if not finite.all():
        # Nearest-valid fill so smoothing does not smear NaNs; the converter would
        # otherwise reuse the nearest pose for these frames anyway.
        good = np.flatnonzero(finite)
        if len(good) == 0:
            raise SystemExit("fit_mhr: no finite body frames in the scored span")
        nearest = good[np.abs(np.arange(len(window))[:, None] - good[None]).argmin(1)]
        window = window[nearest]

    if args.smooth_sigma > 0:
        # odd reflection at both ends (2 x_0 - x_k): a body moving at constant speed is not pulled towards
        # standing still at the clip edges (mode="nearest" did that; matters when the first scored frame
        # is the first video frame, e.g. episode 11)
        p = min(len(window) - 1, int(4 * args.smooth_sigma + 0.5))
        padded = np.concatenate([2 * window[:1] - window[p:0:-1], window, 2 * window[-1:] - window[-2:-p - 2:-1]])
        window = gaussian_filter1d(padded, args.smooth_sigma, axis=0, mode="nearest")[p:p + len(window)]

    import mesh_to_mhr_params as kit_fit
    if args.outlier_floor_mm > 0:
        floor = args.outlier_floor_mm
        kit_fit.is_outlier = lambda err, med: err > max(2 * med + 0.02, floor)
    if args.profile:
        import fit_profile
        fit_profile.install(kit_fit)
    out = convert(window, str(args.model), device=args.device, precision=args.precision,
                  log=lambda *x, **k: print(*x, **k, file=sys.stderr, flush=True))

    # Re-index to VIDEO frames 0..last; frames outside the fitted window copy the edge.
    pose_win = np.asarray(out["pose"], dtype=np.float32)
    v_lo = lo + args.frame_offset
    n_video = args.last + 1
    idx = np.clip(np.arange(n_video) - v_lo, 0, len(pose_win) - 1)
    pose = pose_win[idx]

    report = dict(out["report"])
    report.update({
        "fit_video_frames": [int(v_lo), int(v_lo + len(pose_win) - 1)],
        "smooth_sigma": args.smooth_sigma,
        "finite_input_frames": int(finite.sum()),
        "window_frames": int(len(finite)),
        "filled_nonfinite_video_frames": [int(v_lo + i) for i in np.flatnonzero(~finite)],
    })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.profile:
        prof = fit_profile.report()
        for r in prof["flagged"]:
            r["video_frame"] = int(v_lo + r["frame"])
        prof["seconds_total"] = report.get("seconds")
        json.dump(prof, open(args.out.with_suffix(".profile.json"), "w"), indent=1)
    np.savez(
        args.out,
        pose=pose,
        scales=np.asarray(out["scales"], dtype=np.float32),
        shape=np.asarray(out["shape"], dtype=np.float32),
        per_frame_vertex_error_mm=np.asarray(out["per_frame_vertex_error_mm"], dtype=np.float32),
        fit_video_lo=np.int64(v_lo),
        report=json.dumps(report),
    )
    err = report.get("vertex_error_mm", {})
    print(f"wrote {args.out}: frames {v_lo}..{v_lo + len(pose_win) - 1}, "
          f"vertex error mean {err.get('mean', float('nan')):.3f} mm, "
          f"max {err.get('max', float('nan')):.3f} mm")


if __name__ == "__main__":
    main()
