#!/usr/bin/env python3
"""Pick a reference frame for run_ego_wilor from a wilor_raw/ directory.

Scans wilor_raw/<frame:06d>.json, keeps frames that contain >=1 hand
detection, and prints the index of the *median* such frame to stdout.

Exit codes:
  0  a frame was chosen (index printed to stdout)
  1  no frame has any hands (nothing printed to stdout; note on stderr)
  2  usage / directory error
"""
from __future__ import annotations

import glob
import json
import os
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: pick_reference_frame.py <wilor_raw_dir>", file=sys.stderr)
        return 2

    raw_dir = sys.argv[1]
    files = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
    if not files:
        print(f"no json files under {raw_dir}", file=sys.stderr)
        return 2

    hits: list[int] = []
    for f in files:
        try:
            data = json.load(open(f))
        except Exception:
            continue
        if data:  # non-empty list/dict == at least one hand
            hits.append(int(os.path.basename(f)[:6]))

    if not hits:
        print("no frames with hands", file=sys.stderr)
        return 1

    ref = hits[len(hits) // 2]
    print(f"{len(hits)} frames with hands; picked median {ref}", file=sys.stderr)
    print(ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
