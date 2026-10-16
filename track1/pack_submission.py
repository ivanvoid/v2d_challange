#!/usr/bin/env python3
"""Pack submission/episodes into a Track 1 Parquet with the kit's own packer.

Full run (every scored episode must be present; this is the uploadable file):

    .venv-submit/bin/python track1/pack_submission.py --commit "$COMMIT_URL"

Partial run for local validation (packs only the episodes that exist, against a
filtered copy of the sample; the result is NOT uploadable):

    .venv-submit/bin/python track1/pack_submission.py --partial
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
ROOT = paths.DATA
KIT = paths.KIT
PLACEHOLDER_COMMIT = "https://github.com/placeholder/placeholder/commit/" + "0" * 40


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=Path, default=ROOT / "submission" / "episodes")
    ap.add_argument("--sample", type=Path, default=KIT / "data" / "track_1_sample_submission.parquet")
    ap.add_argument("--commit", default=None, help="https://github.com/<owner>/<repo>/commit/<sha>")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--partial", action="store_true", help="pack only the episodes present (validation only)")
    args = ap.parse_args()

    commit = args.commit or PLACEHOLDER_COMMIT
    if commit == PLACEHOLDER_COMMIT:
        print("NOTE: using a placeholder commit URL; the result is for local validation only.")
    out = args.out or ROOT / "submission" / ("track1_partial.parquet" if args.partial else "track1.parquet")

    sample = args.sample
    with tempfile.TemporaryDirectory() as tmp:
        if args.partial:
            have = sorted(int(m.group(1)) for p in args.episodes.glob("episode_*.npz")
                          if (m := re.fullmatch(r"episode_(\d+)\.npz", p.name)))
            if not have:
                raise SystemExit(f"no episode_*.npz in {args.episodes}")
            df = pd.read_parquet(sample)
            ep = df["row_id"].str.split("_", expand=True)[1].astype(int)
            sample = Path(tmp) / "sample_partial.parquet"
            df[ep.isin(have)].to_parquet(sample, index=False)
            print(f"partial pack: episodes {have}")

        cmd = [sys.executable, "eval_reconstruction.py",
               "--episodes", str(args.episodes.resolve()),
               "--sample", str(Path(sample).resolve()),
               "--commit", commit,
               "--out", str(out.resolve())]
        raise SystemExit(subprocess.call(cmd, cwd=KIT))


if __name__ == "__main__":
    main()
