#!/usr/bin/env python3
"""Stage 4 — object mask: SAM2 propagates the stage-3 seed box through every frame.

Input:  out/episode_XXXXXX/object_track.json (seed frame + box), frames/
Output: out/episode_XXXXXX/masks/1/<frame:06d>.png (one per local frame), sam2_prompts.json, and from
        sam2_box_neg.py: masks_prob/ (soft masks), masks_pass/{bwd,fwd}/ (each pass), masks_conf.json

SAM2 gets one box prompt on EACH stage-3 frame where the object was found: the largest accepted
detection that contains that frame's best box (whole object, not a part of it).

SAM2 is run on the frames directory (the mp4 path segfaulted in earlier runs) and
propagates backward and forward from the seed frame. Model output goes to --debug_log.

    python track1/stage4_mask.py --episode 28 --out_root out --weights weights \
        --debug_log logs/stages/04_mask/episode_000028.debug.log
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time

from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks

OBJECT_ID = 1


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@contextlib.contextmanager
def to_debug(path: str):
    sys.stdout.flush(); sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(path, "a") as fh:
        fh.write(f"\n===== {time.strftime('%H:%M:%S')} =====\n"); fh.flush()
        os.dup2(fh.fileno(), 1); os.dup2(fh.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(saved[0], 1); os.dup2(saved[1], 2)
            os.close(saved[0]); os.close(saved[1])


def whole_object_prompts(W: str, track: dict) -> list[tuple[int, dict]]:
    """One box per stage-3 frame: the largest accepted detection that contains that frame's best box.

    The best-scoring box on a frame can cover only part of the object (e.g. the tray of a cart);
    another accepted detection on the same frame often covers all of it. Prompting SAM2 on every
    frame with the whole-object box keeps the mask from shrinking to the part.
    """
    cand_path = os.path.join(W, "select", "candidates.json")
    best = {int(k): v["box"] for k, v in track.get("per_frame_boxes", {}).items()}
    if not best or not os.path.exists(cand_path):
        return [(int(track["reference_frame"]), track["box"])]
    cands = json.load(open(cand_path))["candidates"]
    tag = track.get("mode") or "near"
    area = lambda b: max(0.0, b["x1"] - b["x0"]) * max(0.0, b["y1"] - b["y0"])
    def covered(inner, outer):  # fraction of `inner` inside `outer`
        ix = max(0.0, min(inner["x1"], outer["x1"]) - max(inner["x0"], outer["x0"]))
        iy = max(0.0, min(inner["y1"], outer["y1"]) - max(inner["y0"], outer["y0"]))
        return ix * iy / max(area(inner), 1.0)
    out = []
    for f, b in sorted(best.items()):
        pool = [c["box"] for c in cands if c["frame"] == f and c["accepted"] and c["tag"] == tag
                and covered(b, c["box"]) >= 0.8 and area(c["box"]) <= 4.0 * area(b)]
        out.append((f, max(pool + [b], key=area)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--debug_log", required=True)
    args = ap.parse_args()

    ep = f"episode_{args.episode:06d}"
    W = os.path.join(os.path.abspath(args.out_root), ep)
    meta = json.load(open(os.path.join(W, "track1_meta.json")))
    n = meta["n_frames"]
    track = json.load(open(os.path.join(W, "object_track.json")))
    mp = os.path.join(W, "mask_prompts.json")
    if os.path.exists(mp):        # built by stage4_prompts.py: boxes + negative body points (+ gap-fill boxes)
        plist = json.load(open(mp))["prompts"]
    else:
        plist = [{"frame": f, "box": b, "neg": [], "source": "stage3"} for f, b in whole_object_prompts(W, track)]
    masks_dir = os.path.join(W, "masks")
    obj_dir = os.path.join(masks_dir, str(OBJECT_ID))
    seed = {"prompts": plist}
    seed_path = os.path.join(W, "masks_seed.json")

    have = len([f for f in os.listdir(obj_dir) if f.endswith(".png")]) if os.path.isdir(obj_dir) else 0
    old_seed = json.load(open(seed_path)) if os.path.exists(seed_path) else None
    if have == n and old_seed == seed:
        log(f"{ep}: [skip] SAM2 masks complete ({n} frames, same seed)")
        return
    if os.path.isdir(masks_dir):
        log(f"{ep}: removing old masks ({have} frames, seed {'changed' if old_seed != seed else 'same'})")
        shutil.rmtree(masks_dir)
    for extra in (masks_dir + "_prob", masks_dir + "_pass"):      # soft masks / per-pass masks of an older run
        shutil.rmtree(extra, ignore_errors=True)
    if os.path.exists(masks_dir + "_conf.json"):
        os.remove(masks_dir + "_conf.json")

    log(f"{ep}: SAM2 with {len(plist)} box prompts, {n} frames, backward + forward")
    for p in plist:
        b = p["box"]
        log(f"  prompt video frame {p['frame'] + meta['frame_offset']} ({p['source']}): box [{b['x0']:.0f}, "
            f"{b['y0']:.0f}, {b['x1']:.0f}, {b['y1']:.0f}] + {len(p.get('neg', []))} negative body points")
    json.dump({"prompts": plist}, open(os.path.join(W, "sam2_prompts.json"), "w"), indent=1)
    t0 = time.time()
    import subprocess
    with to_debug(args.debug_log):
        subprocess.run([os.path.join(os.environ.get("V2D_VENV_ROOT", "/opt/venvs"), "sam2", "bin", "python"),
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2_box_neg.py"),
                        "--frames_dir", os.path.join(W, "frames"), "--prompts", os.path.join(W, "sam2_prompts.json"),
                        "--masks_dir", masks_dir,
                        "--weights_dir", os.path.join(os.path.abspath(args.weights), "sam2")], check=True)
    have = len([f for f in os.listdir(obj_dir) if f.endswith(".png")]) if os.path.isdir(obj_dir) else 0
    if have != n:
        raise SystemExit(f"{ep}: SAM2 wrote {have} masks, expected {n}")
    json.dump(seed, open(seed_path, "w"))
    log(f"{ep}: stage 4 OK — {have} masks in {time.time() - t0:.0f}s ({have / (time.time() - t0):.1f} frames/s)")


if __name__ == "__main__":
    main()
