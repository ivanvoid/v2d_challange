#!/usr/bin/env python3
"""Stage 2d — the body's distance to the camera follows the stabilised MoGe depth trend.

SAM3D-Body sets the body's distance from its apparent size, frame by frame; on the body's own visible
pixels the (static-background-stabilised) MoGe depth disagrees with it slowly over time, mostly growing
with distance (stage 2 review: about -2..-8 % per metre). Here the body is moved along the camera ray
through its centre so that its distance follows MoGe's SLOW trend while keeping the body's own overall
scale:
  ratio_f  = median(MoGe depth / body depth) over the body's front-most pixels that the actor's person
             mask marks as visible (body/person_masks; all body pixels where no mask exists)
  c_f      = ratio_f / median over scored frames, median-filtered (--median frames) then Gaussian (--sigma)
  body_f  -> body_f + (c_f - 1) * centre_f   (centre = mean of the 22 scored joints; only the root
                                              translation pose[:, 0:3] changes: 1 unit = 10 cm along x, -y, -z)
Frames outside the fitted window keep the edge correction.

Writes body/mhr_fit.npz (corrected) after saving the uncorrected fit as body/mhr_fit_raw.npz, and
body/depth_trend.json (ratio per frame, correction per frame, statistics).

    /opt/venvs/sam3d_body/bin/python track1/stage2_depth.py --episode 11 --out_root out \
        --model weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d, median_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from logutil import log  # noqa: E402

sys.path.insert(0, str(paths.KIT_TRACK1))
from mesh_to_mhr_params import MHR  # noqa: E402

JOINTS22 = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]
ROOT_TO_M = np.array([0.1, -0.1, -0.1])        # metres per unit of pose[0:3], camera frame


def load_depth(path: str) -> np.ndarray:
    px = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return 1.0 / np.maximum(px / 65535.0, 1e-6) - 1.0


def visible_ratio(v: np.ndarray, K: np.ndarray, depth: np.ndarray, mask: np.ndarray | None) -> float:
    """Median MoGe / body depth over the front-most body vertex per pixel, visible pixels only."""
    uvw = v @ K.T
    uv = np.round(uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)).astype(np.int64)
    h, w = depth.shape
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h) & (v[:, 2] > 0)
    lin, z = uv[ok, 1] * w + uv[ok, 0], v[ok, 2]
    o = np.lexsort((z, lin))
    lin, z = lin[o], z[o]
    first = np.r_[True, lin[1:] != lin[:-1]]
    lin, z = lin[first], z[first]
    if mask is not None:
        keep = mask.reshape(-1)[lin]
        lin, z = lin[keep], z[keep]
    if len(lin) < 50:
        return np.nan
    return float(np.median(depth.reshape(-1)[lin] / z))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--median", type=int, default=15)
    ap.add_argument("--sigma", type=float, default=10.0)
    ap.add_argument("--step", type=int, default=2, help="measure every N frames, interpolate between")
    args = ap.parse_args()

    ep = f"episode_{args.episode:06d}"
    W = Path(args.out_root).resolve() / ep
    B = W / "body"
    meta = json.load(open(W / "track1_meta.json"))
    off = meta["frame_offset"]
    it = json.load(open(W / "intrinsics_stable.json"))
    K = np.array([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]], [0, 0, 1]])
    scored = np.asarray(json.load(open(paths.DATA / "submission" / "scored_spans.json"))[str(args.episode)]["frames"])
    raw = B / "mhr_fit_raw.npz"
    if not raw.exists():
        shutil.copy2(B / "mhr_fit.npz", raw)
    fit = dict(np.load(raw))
    pose = fit["pose"].astype(np.float64)
    report = json.loads(str(fit["report"]))
    lo_v, hi_v = report["fit_video_frames"]
    vf = np.arange(lo_v, min(hi_v, len(pose) - 1) + 1)    # the pose array ends at the last scored frame

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHR(args.model, dev, 64, "float32")
    z = torch.as_tensor(np.concatenate([fit["scales"], fit["shape"]])[None], dtype=torch.float64, device=dev)
    v_mm, j = mhr.run(torch.as_tensor(pose[vf], dtype=torch.float64, device=dev), z)
    verts, centre = (v_mm / 1000).cpu().numpy(), j.cpu().numpy()[:, JOINTS22].mean(1)

    meas_f, meas_r, n_mask = [], [], 0
    for k in range(0, len(vf), args.step):
        l = vf[k] - off
        mp = B / "person_masks" / f"{l:06d}.png"
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 0 if mp.exists() else None
        n_mask += m is not None
        r = visible_ratio(verts[k], K, load_depth(str(W / "depth" / f"{l:06d}.png")), m)
        if np.isfinite(r):
            meas_f.append(vf[k]); meas_r.append(r)
    meas_f, meas_r = np.asarray(meas_f), np.asarray(meas_r)
    if len(meas_f) < 5:
        raise SystemExit(f"{ep}: too few frames with a MoGe/body ratio ({len(meas_f)})")
    ratio = np.interp(vf, meas_f, meas_r)
    ref = float(np.median(np.interp(scored, meas_f, meas_r)))
    c = gaussian_filter1d(median_filter(ratio / ref, size=args.median, mode="nearest"), args.sigma, mode="nearest")
    delta = (c - 1)[:, None] * centre
    full = np.zeros((len(pose), 3))
    full[vf] = delta
    full[:vf[0]] = delta[0]
    full[vf[-1] + 1:] = delta[-1]
    pose[:, 0:3] += full / ROOT_TO_M
    check = mhr.run(torch.as_tensor(pose[vf], dtype=torch.float64, device=dev), z)[1].cpu().numpy()[:, JOINTS22].mean(1)
    err_mm = float(np.abs(check - centre * c[:, None]).max() * 1000)
    if err_mm > 1.0:
        raise SystemExit(f"{ep}: root-translation check failed ({err_mm:.2f} mm)")
    fit["pose"] = pose.astype(np.float32)
    report["depth_trend"] = {"applied": True, "ratio_reference": ref, "median": args.median, "sigma": args.sigma}
    fit["report"] = np.asarray(json.dumps(report))
    np.savez(B / "mhr_fit.npz", **fit)

    sc = np.isin(vf, scored)
    shift = np.linalg.norm(delta, axis=1) * 100
    rep = {"episode": args.episode, "ratio_reference": ref, "frames_with_mask": int(n_mask),
           "frames_measured": int(len(meas_f)), "factor_scored": [float(c[sc].min()), float(c[sc].max())],
           "shift_cm_scored": {"median": float(np.median(shift[sc])), "max": float(shift[sc].max())},
           "depth_range_cm_before": float(np.ptp(centre[sc, 2]) * 100),
           "depth_range_cm_after": float(np.ptp((centre * c[:, None])[sc, 2]) * 100),
           "params": vars(args), "check_error_mm": err_mm,
           "per_frame": {"video_frame": vf.tolist(), "ratio": [round(float(x), 5) for x in ratio],
                         "factor": [round(float(x), 5) for x in c]}}
    json.dump(rep, open(B / "depth_trend.json", "w"), indent=1)
    log(f"{ep}: body depth follows MoGe trend — factor {rep['factor_scored'][0]:.3f}..{rep['factor_scored'][1]:.3f} "
        f"on scored frames, body moved median {rep['shift_cm_scored']['median']:.1f} / max "
        f"{rep['shift_cm_scored']['max']:.1f} cm (ratio on visible pixels: {n_mask} of {len(meas_f)} measured frames masked)")


if __name__ == "__main__":
    main()
