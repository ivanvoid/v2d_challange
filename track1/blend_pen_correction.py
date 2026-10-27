#!/usr/bin/env python3
"""Scale a validated object translation correction without changing geometry, rotation or object size."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def blend(base: dict, corrected: dict, fraction: float) -> dict:
    if not 0 <= fraction <= 1:
        raise ValueError("correction fraction must lie in [0, 1]")
    if base["rotation"] != corrected["rotation"] or base["scale"] != corrected["scale"]:
        raise ValueError("only translation may change in a PEN trial")
    out = dict(base)
    t0 = np.asarray(base["translation"], dtype=np.float64)
    t1 = np.asarray(corrected["translation"], dtype=np.float64)
    out["translation"] = (t0 + fraction * (t1 - t0)).tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--corrected", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fraction", type=float, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.base.glob("*.json"))
    if not files:
        raise SystemExit("no input poses")
    for path in files:
        other = args.corrected / path.name
        if not other.is_file():
            raise SystemExit(f"missing correction for {path.name}")
        result = blend(json.loads(path.read_text()), json.loads(other.read_text()), args.fraction)
        (args.out / path.name).write_text(json.dumps(result) + "\n")
    print(f"wrote {len(files)} blended poses to {args.out} (fraction {args.fraction})")


if __name__ == "__main__":
    main()
