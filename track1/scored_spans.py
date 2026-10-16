#!/usr/bin/env python3
"""Extract the scored frame span of every Track 1 episode from the sample submission.

Writes {"<episode>": {"first": int, "last": int, "n_scored": int}} as JSON.
Run with the submission venv (needs pandas + pyarrow):

    .venv-submit/bin/python track1/scored_spans.py \
        --sample v2d_submission_kit/data/track_1_sample_submission.parquet \
        --out submission/scored_spans.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EPISODE_FRAME = 999999  # v2dlb.mhr_submission.EPISODE_FRAME: rows carried once per episode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ids = pd.read_parquet(args.sample, columns=["row_id"])["row_id"]
    parts = ids.str.split("_", expand=True)
    keys = pd.DataFrame({"episode": parts[1].astype(int), "frame": parts[2].astype(int)})
    keys = keys[keys["frame"] != EPISODE_FRAME].drop_duplicates()

    spans = {}
    for episode, group in keys.groupby("episode"):
        frames = group["frame"]
        spans[str(int(episode))] = {
            "first": int(frames.min()),
            "last": int(frames.max()),
            "n_scored": int(len(frames)),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spans, indent=1))
    print(f"wrote {args.out}: {len(spans)} episodes")


if __name__ == "__main__":
    main()
