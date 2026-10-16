#!/usr/bin/env python3
"""Run many Grounding DINO detections with one model load.

Each job: {"image": path, "out": json path, "prompt": str, "thr": float}. Jobs whose
output already exists are skipped. Run inside the grounding_dino venv:

    /opt/venvs/grounding_dino/bin/python track1/gdino_batch.py jobs.json weights/grounding_dino
"""
from __future__ import annotations

import json
import os
import sys

from v2d.grounding_dino.lib.image_to_object_bboxes import image_to_bboxes


def main() -> None:
    jobs = json.load(open(sys.argv[1]))
    model_dir = sys.argv[2]
    done = 0
    for job in jobs:
        if os.path.exists(job["out"]):
            continue
        os.makedirs(os.path.dirname(job["out"]), exist_ok=True)
        image_to_bboxes(job["image"], job["out"], job["prompt"], model_dir,
                        box_threshold=job["thr"], text_threshold=0.25)
        done += 1
    print(f"gdino_batch: {done} new, {len(jobs) - done} cached")


if __name__ == "__main__":
    main()
