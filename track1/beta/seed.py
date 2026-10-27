#!/usr/bin/env python3
"""Seed a beta output root from the production results, so only the stages a beta switch changes rerun.

For each episode, out_beta/<tag>/episode_XXXXXX/ gets
  symlinks  frames/, anycalib/, video_trim.mp4, intrinsics/ (never written after stage 1) and, unless the
            beta changes stage 1 (depth), depth_raw/ and depth/
  copies    everything else of the production episode (body/, masks, mesh, ...; not _archive/)
  stamps    of the stages BEFORE the first changed one (and stage 2, which does not depend on depth), so
            run_stage.sh skips them; the copies of the changed stages' outputs are deleted
With --first 1 (BETA_DEPTH) the stage-1 depth products (depth_raw/, depth/, depth_scale.json,
background.npz, floor.json) are not taken: stage 1 rebuilds them from the new depth.
An episode that is already seeded (.beta_seed.json) is left alone.

    python3 track1/beta/seed.py --src out --dst out_beta/vda-trellis2-cotracker --first 1 --episodes 6 8
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stamps  # noqa: E402

ORDER = ["1", "2", "3", "4", "5", "5c", "5e", "6", "7", "8"]
LINK = ["frames", "anycalib", "video_trim.mp4", "intrinsics"]
DEPTH = ["depth_raw", "depth"]
DEPTH_PRODUCTS = ["depth_scale.json", "background.npz", "floor.json"]


def seed(src: Path, dst: Path, first: str) -> str:
    if not (src / "track1_meta.json").exists():
        return f"{src.name}: production episode missing (no track1_meta.json) — run stage 1 there first"
    marker = dst / ".beta_seed.json"
    if marker.exists():
        return f"{src.name}: already seeded ({json.load(open(marker))['first']})"
    dst.mkdir(parents=True, exist_ok=True)
    link = LINK + (DEPTH if first != "1" else [])
    skip = set(link) | {"_archive", "stamps"} | (set(DEPTH + DEPTH_PRODUCTS) if first == "1" else set())
    for name in link:
        if (src / name).exists() and not (dst / name).exists():
            os.symlink((src / name).resolve(), dst / name)
    for p in sorted(src.iterdir()):
        if p.name in skip or (dst / p.name).exists():
            continue
        if p.is_dir():
            shutil.copytree(p, dst / p.name, symlinks=True)
        else:
            shutil.copy2(p, dst / p.name)
    changed = ORDER[ORDER.index(first):]
    for s in changed:
        if s == "2":
            continue
        for o in stamps.STAGES[s]["outputs"]:
            q = dst / o
            if o == "SUB_EPISODE" or not (q.exists() or q.is_symlink()):
                continue
            shutil.rmtree(q) if q.is_dir() and not q.is_symlink() else q.unlink()
    kept = []
    (dst / "stamps").mkdir(exist_ok=True)
    for s in ORDER:
        if (s in changed and s != "2") or not (src / "stamps" / f"{s}.json").exists():
            continue
        shutil.copy2(src / "stamps" / f"{s}.json", dst / "stamps" / f"{s}.json")
        kept.append(s)
    json.dump({"src": str(src), "first": first, "stamps_kept": kept, "time": time.strftime("%F %T")},
              open(marker, "w"), indent=1)
    return f"{src.name}: seeded (stages {', '.join(kept) or 'none'} kept; rerun from {first})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="production out/")
    ap.add_argument("--dst", type=Path, required=True, help="beta out root")
    ap.add_argument("--first", choices=ORDER, required=True, help="first stage a beta switch changes")
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    a = ap.parse_args()
    if a.src.resolve() == a.dst.resolve():
        raise SystemExit("--dst must not be the production out/")
    for e in a.episodes:
        ep = f"episode_{e:06d}"
        print(seed(a.src.resolve() / ep, a.dst.resolve() / ep, a.first), flush=True)


if __name__ == "__main__":
    main()
