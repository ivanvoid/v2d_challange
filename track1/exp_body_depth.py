#!/usr/bin/env python3
"""Experiment: body distance to the camera follows the (stabilised) MoGe depth trend.

SAM3D-Body sets the body's distance from its apparent size, frame by frame; the static-background-
stabilised MoGe depth on the body's own pixels disagrees with it slowly over time (1–10 % within a
video, mostly growing with distance). Here the body is moved along the camera ray through its centre
so that its distance follows MoGe's SLOW trend, at the body's own overall scale:

  ratio_f  = median(MoGe depth / body depth) over the body's front-most pixels, frame f
  c_f      = ratio_f / median over scored frames, median-filtered (15 frames) then Gaussian (sigma 10)
  body_f  -> body_f + (c_f - 1) * centre_f      (centre = mean of the 22 scored joints; root
                                                translation only, so pose and image are unchanged)
Frames outside first..last keep the edge correction. Only `pose[:, 0:3]` (root translation; 1 unit =
10 cm along x, -y, -z of the camera frame) changes; the object is untouched.

    /opt/venvs/sam3d_body/bin/python track1/exp_body_depth.py --base submission/episodes \
        --out experiments/p2_body_moge/episodes
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d, median_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from review_stage2 import JOINTS22, depth_bias, load_depth, project  # noqa: E402

sys.path.insert(0, str(paths.KIT_TRACK1))
from mesh_to_mhr_params import MHR  # noqa: E402

ROOT_TO_M = np.array([0.1, -0.1, -0.1])          # metres per unit of pose[0:3] in the camera frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", default="0-29")
    ap.add_argument("--median", type=int, default=15)
    ap.add_argument("--sigma", type=float, default=10.0)
    ap.add_argument("--step", type=int, default=2, help="measure the ratio every N frames, interpolate between")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.episodes.split("-")) if "-" in a.episodes else (int(a.episodes),) * 2
    eps = list(range(lo, hi + 1))

    a.out.mkdir(parents=True, exist_ok=True)
    for f in sorted(a.base.iterdir()):
        shutil.copy2(f, a.out / f.name)
    spans = json.load(open(paths.DATA / "submission" / "scored_spans.json"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHR(str(paths.DATA / "weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt"), dev, 64, "float32")
    log = []
    for e in eps:
        W = paths.DATA / "out" / f"episode_{e:06d}"
        meta = json.load(open(W / "track1_meta.json"))
        intr = json.load(open(W / "intrinsics_stable.json"))
        K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
        off, first, last = meta["frame_offset"], meta["scored_first"], meta["scored_last"]
        scored = np.asarray(spans[str(e)]["frames"])
        path = a.out / f"episode_{e:06d}.npz"
        d = dict(np.load(path))
        pose = d["pose"].astype(np.float64)
        z = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device=dev)
        vf = np.arange(first, last + 1)
        v_mm, j = mhr.run(torch.as_tensor(pose[vf], dtype=torch.float64, device=dev), z)
        verts, centre = (v_mm / 1000).cpu().numpy(), j.cpu().numpy()[:, JOINTS22].mean(1)
        meas_f, meas_r = [], []
        for k in range(0, len(vf), a.step):
            b = depth_bias(verts[k], project(verts[k], K), load_depth(str(W / "depth" / f"{vf[k] - off:06d}.png")))
            if b:
                meas_f.append(vf[k]); meas_r.append(b[1])
        meas_f, meas_r = np.asarray(meas_f), np.asarray(meas_r)
        ratio = np.interp(vf, meas_f, meas_r)
        c = ratio / np.median(np.interp(scored, meas_f, meas_r))
        c = gaussian_filter1d(median_filter(c, size=a.median, mode="nearest"), a.sigma, mode="nearest")
        delta = (c - 1)[:, None] * centre                                   # metres, camera frame
        full = np.zeros((len(pose), 3))
        full[vf] = delta
        full[:first] = delta[0]
        full[last + 1:] = delta[-1]
        pose[:, 0:3] += full / ROOT_TO_M
        # verify: the new centre is c x the old one
        j2 = mhr.run(torch.as_tensor(pose[vf], dtype=torch.float64, device=dev), z)[1].cpu().numpy()[:, JOINTS22].mean(1)
        err = float(np.abs(j2 - centre * c[:, None]).max() * 1000)
        d["pose"] = pose.astype(np.float32)
        np.savez(path, **d)
        sc = np.isin(vf, scored)
        rec = {"episode": e, "c_min": float(c[sc].min()), "c_max": float(c[sc].max()),
               "shift_cm_max": float(np.linalg.norm(delta[sc], axis=1).max() * 100),
               "shift_cm_median": float(np.median(np.linalg.norm(delta[sc], axis=1)) * 100),
               "check_error_mm": err}
        log.append(rec)
        print(f"episode {e:2d}: depth factor {rec['c_min']:.3f}..{rec['c_max']:.3f} on scored frames, body moved "
              f"median {rec['shift_cm_median']:.1f} cm / max {rec['shift_cm_max']:.1f} cm (check {err:.2f} mm)", flush=True)
    json.dump({"experiment": "body depth follows the slow MoGe trend", "params": {k: str(v) for k, v in vars(a).items()},
               "episodes": log}, open(a.out.parent / "experiment.json", "w"), indent=1)


if __name__ == "__main__":
    main()
