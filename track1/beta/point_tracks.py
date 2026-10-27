#!/usr/bin/env python3
"""Beta stage 7 (part 1) — 2D point tracks on the object with CoTracker3 (offline, bidirectional).

Query frames are picked from track/states.json: one per --every frames, the best 'observed' frame
(IoU x fit) with IoU >= --min_iou. On each, up to --points points are sampled inside the stage-4 object mask
(eroded, minus the dilated person mask; corners first, then random fill), and tracked forward and backward
over [q - --half, q + --half] on the undistorted frames (downscaled to --res long side; tracks are returned
in full-resolution pixels).

Writes track/points.npz:
  q (K,)              local query frame per window
  start (K,)          local frame of window row 0
  uv (K, L, N, 2)     pixel tracks, NaN where the window runs past the clip or a point slot is unused
  vis (K, L, N)       CoTracker visibility
consumed by beta/track_refine.py. Skipped when points.npz is newer than states.json.

    /opt/venvs/cotracker/bin/python track1/beta/point_tracks.py --episode 6 --out_root out_beta/<tag> --weights weights
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logutil import log, to_debug  # noqa: E402
import paths  # noqa: E402


def pick_queries(frames: list, every: int, min_iou: float) -> list[int]:
    out = []
    for b in range(0, len(frames), every):
        ok = [((f.get("iou") or 0) * (f.get("fit") or 0), f["frame"]) for f in frames[b:b + every]
              if f.get("state") == "observed" and (f.get("iou") or 0) >= min_iou]
        if ok:
            out.append(max(ok)[1])
    return out


def sample_points(W: Path, q: int, n: int, rng) -> np.ndarray:
    m = cv2.imread(str(W / "masks" / "1" / f"{q:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
    m = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    pm = W / "body" / "person_masks" / f"{q:06d}.png"
    if pm.exists():
        m &= ~(cv2.dilate((cv2.imread(str(pm), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8),
                          np.ones((9, 9), np.uint8)) > 0)
    if m.sum() < 30:
        return np.zeros((0, 2), np.float32)
    g = cv2.cvtColor(cv2.imread(str(W / "frames" / f"{q:06d}.png")), cv2.COLOR_BGR2GRAY)
    c = cv2.goodFeaturesToTrack(g, n // 2, 0.01, 4, mask=m.astype(np.uint8))
    c = c.reshape(-1, 2) if c is not None else np.zeros((0, 2), np.float32)
    ys, xs = np.nonzero(m)
    k = min(n - len(c), len(xs))
    i = rng.choice(len(xs), k, replace=False)
    return np.concatenate([c, np.c_[xs[i], ys[i]].astype(np.float32)]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", choices=["cotracker"], default="cotracker")
    ap.add_argument("--repo", default=str(paths.DATA / "models" / "co-tracker"))
    ap.add_argument("--every", type=int, default=40)
    ap.add_argument("--half", type=int, default=60)
    ap.add_argument("--points", type=int, default=200)
    ap.add_argument("--min_iou", type=float, default=0.7)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--debug_log", default=None)
    a = ap.parse_args()

    ep = f"episode_{a.episode:06d}"
    W = Path(a.out_root).resolve() / ep
    out = W / "track" / "points.npz"
    states = W / "track" / "states.json"
    if out.exists() and out.stat().st_mtime > states.stat().st_mtime:
        log(f"  [skip] point tracks (points.npz newer than states.json)")
        return
    st = json.load(open(states))["frames"]
    n = len(st)
    qs = pick_queries(st, a.every, a.min_iou)
    if not qs:
        raise SystemExit(f"{ep}: no observed frame with IoU >= {a.min_iou} to start point tracks from")
    log(f"  [run ] CoTracker3 from {len(qs)} query frames, {a.points} points each, +-{a.half} frames")
    t0 = time.time()

    H, Wd = cv2.imread(str(W / "frames" / f"{0:06d}.png")).shape[:2]
    s = min(1.0, a.res / max(H, Wd))
    h, w = int(round(H * s)), int(round(Wd * s))
    cache: dict[int, np.ndarray] = {}

    def frame(i: int) -> np.ndarray:
        if i not in cache:
            im = cv2.cvtColor(cv2.imread(str(W / "frames" / f"{i:06d}.png")), cv2.COLOR_BGR2RGB)
            cache[i] = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) if s < 1 else im
        return cache[i]

    L = 2 * a.half + 1
    K = len(qs)
    uv = np.full((K, L, a.points, 2), np.nan, np.float32)
    vis = np.zeros((K, L, a.points), bool)
    starts = np.zeros(K, int)
    rng = np.random.default_rng(0)
    with to_debug(a.debug_log, f"{ep} CoTracker3"):
        import torch
        sys.path.insert(0, a.repo)
        from cotracker.predictor import CoTrackerPredictor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = CoTrackerPredictor(checkpoint=str(Path(a.weights) / "cotracker3" / "scaled_offline.pth"),
                                   offline=True, window_len=60).to(dev)
        for k, q in enumerate(qs):
            a0, a1 = max(0, q - a.half), min(n, q + a.half + 1)
            starts[k] = q - a.half
            for i in [i for i in cache if i < a0]:
                del cache[i]
            p = sample_points(W, q, a.points, rng)
            if not len(p):
                continue
            video = torch.from_numpy(np.stack([frame(i) for i in range(a0, a1)])).permute(0, 3, 1, 2)[None]
            qq = torch.from_numpy(np.c_[np.full(len(p), q - a0), p * s].astype(np.float32))[None]
            with torch.no_grad():
                tr, vi = model(video.float().to(dev), queries=qq.to(dev), backward_tracking=True)
            r0 = a0 - starts[k]
            uv[k, r0:r0 + (a1 - a0), :len(p)] = tr[0].cpu().numpy() / s
            vis[k, r0:r0 + (a1 - a0), :len(p)] = vi[0].cpu().numpy() > 0.5
            print(f"window {k + 1}/{K} q={q} frames {a0}-{a1 - 1} points {len(p)}", flush=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, q=np.array(qs), start=starts, uv=uv, vis=vis)
    os.replace(tmp, out)
    v = vis.sum(-1)
    log(f"  [done] point tracks in {time.time() - t0:.0f}s: {K} windows, median "
        f"{np.median(v[v > 0]):.0f} visible points per frame")


if __name__ == "__main__":
    main()
