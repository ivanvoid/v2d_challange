"""Where things live.

CODE  this repository (track1/, run_*.sh, render_results.py) — resolved from this file.
DATA  the working directory with the inputs and outputs, NOT in the repository:
        challenge/            the released dataset (git clone of nvidia/video_to_data_challenge)
        weights/              model weights
        v2d_submission_kit/   the official submission kit
        out/                  per-episode results;  logs/  review pages;  submission/  packed files
        seeds/                optional manual object seeds (stage 3)
      Set with the V2D_DATA environment variable (default /local-data/ivan/v2d).
OUT   per-episode results, default DATA/out; V2D_OUT moves them (run_beta.sh: out_beta/<tag>)
SUB   submission files, default DATA/submission; V2D_SUB moves them (scored_spans.json stays in DATA/submission)
"""
from __future__ import annotations

import os
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
TRACK1 = CODE / "track1"
DATA = Path(os.environ.get("V2D_DATA", "/local-data/ivan/v2d")).resolve()
OUT = Path(os.environ.get("V2D_OUT") or DATA / "out").resolve()
SUB = Path(os.environ.get("V2D_SUB") or DATA / "submission").resolve()
KIT = DATA / "v2d_submission_kit"
KIT_TRACK1 = KIT / "tools" / "track1"
META = DATA / "challenge" / "track_1" / "meta" / "episodes_metadata.jsonl"
VIDEO_DIR = DATA / "challenge" / "track_1" / "videos" / "chunk-000" / "observation.images.exo_camera"
PRIORS = TRACK1 / "object_priors.json"
SEEDS = DATA / "seeds"
