#!/usr/bin/env python3
"""Assemble an isolated Track-1 episode directory from validated collision-correction trials.

This never writes to the production out/ or submission/ directories. Run the existing
packer on the returned `episodes/` directory, then evaluate the packed parquet.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trial", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True, help="episodes with corrected poses")
    args = ap.parse_args()
    changed = set(args.episodes)
    if not changed or not changed.issubset(set(range(30))):
        raise SystemExit("expected valid episode indices from 0 to 29")

    src = paths.DATA / "submission/episodes"
    dst = args.trial.resolve() / "episodes"
    dst.mkdir(parents=True, exist_ok=True)
    # Copy, don't hardlink: writing to a hardlink could silently corrupt production.
    for e in range(30):
        stem = f"episode_{e:06d}"
        for suffix in (".npz", "_object.obj"):
            name = stem + suffix
            if not (src / name).is_file():
                raise SystemExit(f"missing production input {src / name}")
            if e not in changed:
                shutil.copy2(src / name, dst / name)

    for e in sorted(changed):
        stem = f"episode_{e:06d}"
        trial_episode = args.trial.resolve() / "out" / stem
        report_path = trial_episode / "track/depen.json"
        poses = trial_episode / "poses_depen"
        if not report_path.is_file() or not poses.is_dir():
            raise SystemExit(f"trial episode {stem} has no correction report and poses")
        rep = json.loads(report_path.read_text())
        if rep["pen_proxy_after_cm"] >= rep["pen_proxy_before_cm"]:
            raise SystemExit(f"{stem}: local PEN did not improve: {rep}")
        original_episode = paths.DATA / "out" / stem
        off = json.loads((original_episode / "track1_meta.json").read_text())["frame_offset"]
        subprocess.run([sys.executable, str(Path(__file__).with_name("build_episode.py")),
                        "--episode", str(e), "--ep_dir", str(original_episode),
                        "--object_poses", str(poses), "--out_dir", str(dst),
                        "--frame_offset", str(off)], check=True)
    print(f"assembled {dst}: 30 episodes, corrected {sorted(changed)} (not yet validated or uploaded)")


if __name__ == "__main__":
    main()
