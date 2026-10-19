#!/usr/bin/env python3
"""Experiment: switch chosen episodes of an existing submission to the stage-8 "global" scale variant.

Stage 8 places the object either in MoGe's depth scale ("none") or in the body's scale ("global":
object translation and size x s_global, s_global = 1 / median(MoGe depth / body depth) on the
body's pixels). This rewrites ONLY those two arrays of the chosen episodes, starting from the
baseline submission files, so the result differs from the baseline in exactly one thing. The
factor is the s_global stored in out/<ep>/track/scale_match.json when the baseline was built.
Every other file is copied unchanged.

    python track1/exp_force_global.py --base submission/episodes --out experiments/p1_global/episodes \
        --episodes 0 1 8 14 25 27 28 29
    .venv-submit/bin/python track1/pack_submission.py --episodes experiments/p1_global/episodes \
        --out experiments/p1_global/track1.parquet --commit <url>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--out_root", type=Path, default=paths.DATA / "out")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for f in sorted(args.base.iterdir()):
        shutil.copy2(f, args.out / f.name)
    log = []
    for e in args.episodes:
        sm = json.load(open(args.out_root / f"episode_{e:06d}" / "track" / "scale_match.json"))
        if sm["chosen"] != "none":
            raise SystemExit(f"episode {e}: baseline variant is {sm['chosen']!r}, expected 'none'")
        s = float(sm["s_global"])
        path = args.out / f"episode_{e:06d}.npz"
        d = dict(np.load(path))
        d["object_translation"] = (d["object_translation"].astype(np.float64) * s).astype(np.float32)
        d["object_scale"] = np.float32(float(d["object_scale"]) * s)
        np.savez(path, **d)
        log.append({"episode": e, "s_global": s, "object_scale": float(d["object_scale"])})
        print(f"episode {e:2d}: none -> global, translation and size x{s:.4f}")
    json.dump({"experiment": "force stage-8 'global' scale variant", "base": str(args.base),
               "changed": log}, open(args.out.parent / "experiment.json", "w"), indent=1)


if __name__ == "__main__":
    main()
