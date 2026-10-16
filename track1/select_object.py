#!/usr/bin/env python3
"""Pick the manipulated object for an episode and leave a visual record of why.

The text detector (Grounding DINO) alone picks the most "desk-like" box in the
scene, which in these wide exo shots is often a background desk (episode 9).
Selection here is actor-centred:

  1. find the actor on a few candidate frames inside the scored span
     (largest "person." box; bystanders are smaller / farther),
  2. blank every pixel deeper than actor depth + --near_margin_m (MoGe depth), so
     background furniture disappears before detection ("near-only" frames),
  3. run the prompt ladder on the near-only frames; accept a detection only if it
     touches the actor in the image and sits at the actor's depth,
  4. if nothing is accepted, repeat on the full frames with the same gates,
  5. require acceptance on >= 2 candidate frames, take the best-scoring one.

Writes into --out_dir:
  object_track.json, dino_detections.json, near_frames/, dino_candidates/{near,full}/
and, with --review_png, a sheet showing every candidate (see viz_object_candidates.py).

With --legacy_masks_dir it also compares the pick with an existing object mask track
(run_ego_wilor output) and records "legacy_match" in object_track.json, so the batch
can reuse a legacy object branch only when it segmented the same object.

Run with the pipeline interpreter and V2D_BACKEND=venv:

    python track1/select_object.py --frames_dir out/episode_000009/frames \
        --depth_dir out/episode_000009/depth --out_dir out/episode_000009/track1_select \
        --prompt "a medium sized white desk." --object_name white_desk \
        --first 90 --last 414 --weights weights \
        --legacy_masks_dir out/episode_000009/masks/1 \
        --review_png logs/review/episode_000009_object_candidates.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import numpy as np
from PIL import Image


OBJECT_ID = 1


# ----------------------------------------------------------------------------- geometry helpers
def rect_gap(a: dict, b: dict) -> float:
    """Pixel gap between two {x0,y0,x1,y1} boxes; 0 when they overlap."""
    dx = max(a["x0"] - b["x1"], b["x0"] - a["x1"], 0.0)
    dy = max(a["y0"] - b["y1"], b["y0"] - a["y1"], 0.0)
    return float((dx * dx + dy * dy) ** 0.5)


def box_iou(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]))
    iy = max(0.0, min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]))
    inter = ix * iy
    area = lambda r: max(0.0, r["x1"] - r["x0"]) * max(0.0, r["y1"] - r["y0"])
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def load_depth(depth_dir: str | None, frame: int) -> np.ndarray | None:
    """MoGe uint16 PNG (px = 65535/(d+1)) -> metres."""
    if depth_dir is None:
        return None
    path = os.path.join(depth_dir, f"{frame:06d}.png")
    if not os.path.exists(path):
        return None
    px = np.asarray(Image.open(path)).astype(np.float32)
    return 1.0 / np.maximum(px / 65535.0, 1e-6) - 1.0


def box_depth(depth_dir: str | None, frame: int, box: dict) -> float | None:
    """Median MoGe depth (m) over the central half of a box."""
    d = load_depth(depth_dir, frame)
    if d is None:
        return None
    h, w = d.shape
    cx, cy = (box["x0"] + box["x1"]) / 2, (box["y0"] + box["y1"]) / 2
    hw, hh = (box["x1"] - box["x0"]) / 4, (box["y1"] - box["y0"]) / 4
    x0, x1 = int(max(0, cx - hw)), int(min(w, cx + hw + 1))
    y0, y1 = int(max(0, cy - hh)), int(min(h, cy + hh + 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return float(np.median(d[y0:y1, x0:x1]))


def mask_box(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    m = np.asarray(Image.open(path).convert("L")) > 0
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return {"x0": float(xs.min()), "y0": float(ys.min()), "x1": float(xs.max() + 1), "y1": float(ys.max() + 1)}


# ----------------------------------------------------------------------------- prompts + detection
def prompt_ladder(prompt: str, object_name: str | None) -> list[str]:
    """Full caption first, then shorter noun phrases GDINO handles better."""
    out = [prompt.strip()]
    base = re.sub(r"[.]+$", "", prompt.strip())
    head = re.split(r"\b(?:with|of|that|which|in|on)\b|,", base, maxsplit=1)[0].strip()
    head_words = re.sub(r"^(a|an|the)\s+", "", head, flags=re.I).split()
    if head and head != base and len(head_words) >= 2:
        out.append(head + ".")
    if len(head_words) > 2:
        out.append(" ".join(head_words[-2:]) + ".")
    if object_name:
        name = object_name.replace("_", " ").strip()
        out.append(name + ".")
        out.append(name.split()[-1] + ".")
    seen, ladder = set(), []
    for p in out:
        if p and p.lower() not in seen:
            seen.add(p.lower())
            ladder.append(p)
    return ladder


BASE_THR = 0.2          # one detection pass; higher thresholds are filters on its output
THRESHOLDS = (0.35, 0.25, 0.2)


def gdino_batch(jobs: list[dict], weights: str) -> None:
    """Run all missing detections in ONE grounding_dino process (one model load)."""
    import subprocess
    import tempfile
    todo = [j for j in jobs if not os.path.exists(j["out"])]
    if not todo:
        return
    venv = os.environ.get("V2D_VENV_ROOT", "/opt/venvs")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(todo, fh)
        path = fh.name
    try:
        subprocess.run([os.path.join(venv, "grounding_dino", "bin", "python"),
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "gdino_batch.py"),
                        path, weights], check=True, stdout=sys.stderr)  # model chatter -> debug log
    finally:
        os.remove(path)


def load_dets(cache_path: str, thr: float) -> list[dict]:
    if not os.path.exists(cache_path):
        return []
    return [d for d in json.load(open(cache_path)) if float(d.get("confidence", 0.0)) >= thr]


def cache_name(cand_dir: str, rung: int, frame: int) -> str:
    return os.path.join(cand_dir, f"r{rung}_t{int(BASE_THR * 100)}_{frame:06d}.json")


def prefetch(images_dir: str, cand_dir: str, frames: list[int], ladder: list[str], weights: str) -> None:
    jobs = [{"image": os.path.join(images_dir, f"{f:06d}.png"), "out": cache_name(cand_dir, r, f),
             "prompt": p, "thr": BASE_THR}
            for r, p in enumerate(ladder) for f in frames
            if os.path.exists(os.path.join(images_dir, f"{f:06d}.png"))]
    gdino_batch(jobs, weights)


def candidate_frames(first: int, last: int, n: int) -> list[int]:
    n = max(1, n)
    return sorted({int(round(first + (last - first) * (i + 0.5) / n)) for i in range(n)})


def write_near_frames(frames_dir: str, depth_dir: str, out_dir: str, frames: list[int], people: dict,
                      person_z: dict, margin_m: float, pad_rel: float = 0.15) -> dict:
    """Grey out pixels deeper than actor depth + margin, never inside the actor's padded box.

    Returns {frame: {"blanked": fraction of image greyed,
                     "actor_box_beyond_cut": fraction of the padded actor box that is deeper
                     than the cut (kept anyway; mostly background seen around the actor)}}.
    """
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    for f in frames:
        dst = os.path.join(out_dir, f"{f:06d}.png")
        d = load_depth(depth_dir, f)
        p = people.get(f)
        if p is None or person_z.get(f) is None or d is None:
            continue
        h, w = d.shape
        pad = pad_rel * (p["y1"] - p["y0"])
        x0, x1 = int(max(0, p["x0"] - pad)), int(min(w, p["x1"] + pad))
        y0, y1 = int(max(0, p["y0"] - pad)), int(min(h, p["y1"] + pad))
        far = d > person_z[f] + margin_m
        beyond = float(far[y0:y1, x0:x1].mean()) if (y1 > y0 and x1 > x0) else 0.0
        far[y0:y1, x0:x1] = False
        stats[f] = {"blanked": float(far.mean()), "actor_box_beyond_cut": beyond,
                    "protected_box": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}}
        if not os.path.exists(dst):
            img = np.asarray(Image.open(os.path.join(frames_dir, f"{f:06d}.png")).convert("RGB")).copy()
            img[far] = 128
            Image.fromarray(img).save(dst)
    return stats


def pick(images_dir: str, cand_dir: str, frames: list[int], ladder: list[str], people: dict,
         person_z: dict, depth_dir: str | None, weights: str, max_gap_rel: float,
         max_depth_gap_m: float) -> tuple[dict | None, list[dict]]:
    """First ladder rung / threshold with gated detections on >= 2 frames; best score wins."""
    rejected = []
    prefetch(images_dir, cand_dir, frames, ladder, weights)
    for rung, prompt in enumerate(ladder):
        for thr in THRESHOLDS:
            hits = []
            for f in frames:
                if not os.path.exists(os.path.join(images_dir, f"{f:06d}.png")):
                    continue
                person, best = people.get(f), None
                for d in load_dets(cache_name(cand_dir, rung, f), thr):
                    conf = float(d.get("confidence", 0.0))
                    rel = dz = None
                    if person is not None:
                        rel = rect_gap(d["box"], person) / max(person["y1"] - person["y0"], 1.0)
                        oz = box_depth(depth_dir, f, d["box"])
                        if oz is not None and person_z.get(f) is not None:
                            dz = oz - person_z[f]
                        if rel > max_gap_rel or (dz is not None and abs(dz) > max_depth_gap_m):
                            rejected.append({"prompt": prompt, "thr": thr, "frame": f,
                                             "conf": round(conf, 3), "gap": round(rel, 2),
                                             "dz_m": None if dz is None else round(dz, 2)})
                            continue
                    score = conf * (math.exp(-rel / 0.1) if rel is not None else 1.0)
                    if best is None or score > best["score"]:
                        best = {"score": score, "frame": f, "det": d, "gap": rel, "dz": dz}
                if best is not None:
                    hits.append(best)
            if len(hits) >= min(2, len(frames)):
                top = max(hits, key=lambda h: h["score"])
                top.update({"prompt": prompt, "box_threshold": thr, "hits": len(hits),
                            "n_frames": len(frames), "rung": rung})
                return top, rejected
    return None, rejected


def select_object(frames_dir: str, depth_dir: str, out_dir: str, first: int, last: int,
                  prompt: str, object_name: str | None, weights: str, n_candidates: int = 5,
                  near_margin_m: float = 0.7, max_gap_rel: float = 0.1, max_depth_gap_m: float = 0.6,
                  legacy_masks_dir: str | None = None, review_png: str | None = None) -> dict:
    """Select the object, write object_track.json + dino_detections.json, return the record."""
    gd = os.path.join(weights, "grounding_dino")
    os.makedirs(out_dir, exist_ok=True)
    frames = candidate_frames(first, last, n_candidates)
    cand_root = os.path.join(out_dir, "dino_candidates")

    people, person_z = {}, {}
    person_path = lambda f: os.path.join(cand_root, "person", f"person_t35_{f:06d}.json")
    gdino_batch([{"image": os.path.join(frames_dir, f"{f:06d}.png"), "out": person_path(f),
                  "prompt": "person.", "thr": 0.35} for f in frames], gd)
    for f in frames:
        dets = load_dets(person_path(f), 0.35)
        if dets:
            p = max(dets, key=lambda d: (d["box"]["x1"] - d["box"]["x0"]) * (d["box"]["y1"] - d["box"]["y0"]))
            people[f] = p["box"]
            person_z[f] = box_depth(depth_dir, f, p["box"])
    ladder = prompt_ladder(prompt, object_name)

    near_dir = os.path.join(out_dir, "near_frames")
    near_stats = write_near_frames(frames_dir, depth_dir, near_dir, frames, people, person_z, near_margin_m)
    mode = "near"
    found, rejected = pick(near_dir, os.path.join(cand_root, "near"), frames, ladder, people, person_z,
                           depth_dir, gd, max_gap_rel, max_depth_gap_m)
    if found is None:
        mode = "full"
        found, rej_full = pick(frames_dir, os.path.join(cand_root, "full"), frames, ladder, people,
                               person_z, depth_dir, gd, max_gap_rel, max_depth_gap_m)
        rejected += rej_full

    record = {
        "status": "ok" if found else "not_found",
        "object_id": OBJECT_ID,
        "original_prompt": prompt,
        "ladder": ladder,
        "candidate_frames": frames,
        "people": {str(k): v for k, v in people.items()},
        "person_depth_m": {str(k): v for k, v in person_z.items()},
        "near_margin_m": near_margin_m,
        "near_frames": {str(k): v for k, v in near_stats.items()},
        "mode": mode if found else None,
        "rejected": rejected[:40],
    }
    if found:
        record.update({
            "reference_frame": int(found["frame"]),
            "prompt": found["prompt"],
            "label": found["det"].get("label"),
            "confidence": float(found["det"].get("confidence", 0.0)),
            "box": found["det"]["box"],
            "score": found["score"],
            "box_threshold": found["box_threshold"],
            "candidate_hits": f"{found['hits']}/{found['n_frames']}",
            "person_box": people.get(found["frame"]),
            "gap_person_heights": found["gap"],
            "depth_gap_m": found["dz"],
        })
        json.dump([found["det"]], open(os.path.join(out_dir, "dino_detections.json"), "w"), indent=2)

    if found and legacy_masks_dir:
        # Same object? Compare the pick with the legacy mask on every candidate frame where
        # this prompt produced an accepted box, using the frame of the pick primarily.
        mb = mask_box(os.path.join(legacy_masks_dir, f"{found['frame']:06d}.png"))
        iou = box_iou(found["det"]["box"], mb) if mb else 0.0
        record["legacy_mask_box"] = mb
        record["legacy_box_iou"] = round(iou, 3)
        record["legacy_match"] = bool(iou >= 0.4)

    json.dump(record, open(os.path.join(out_dir, "object_track.json"), "w"), indent=2)

    if review_png:
        from viz_object_candidates import render_candidates
        cand_dir = os.path.join(cand_root, mode if found else "near")
        render_candidates(
            frames_dir=near_dir if mode == "near" else frames_dir, cand_dir=cand_dir,
            person_dir=os.path.join(cand_root, "person"), depth_dir=depth_dir,
            chosen_frame=record.get("reference_frame"), chosen_box=record.get("box"),
            legacy_box=record.get("legacy_mask_box"),
            frame_notes={k: (f"near-only: {100 * v['blanked']:.0f}% of image greyed (depth > actor + "
                             f"{near_margin_m:.1f} m); actor box + 15% pad protected "
                             f"({100 * v['actor_box_beyond_cut']:.0f}% of it was beyond the cut)")
                         for k, v in near_stats.items()} if mode == "near" else None,
            protected_boxes={k: v["protected_box"] for k, v in near_stats.items()} if mode == "near" else None,
            title=(f"{os.path.basename(review_png).split('_object')[0]} | prompt: {record.get('prompt', prompt)} | "
                   f"mode: {record['mode']} | status: {record['status']}"
                   + (f" | legacy IoU {record['legacy_box_iou']}" if 'legacy_box_iou' in record else "")),
            out=review_png, max_gap_rel=max_gap_rel, max_depth_gap_m=max_depth_gap_m)
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--depth_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--object_name", default=None)
    ap.add_argument("--first", type=int, required=True, help="first scored frame, LOCAL index")
    ap.add_argument("--last", type=int, required=True, help="last scored frame, LOCAL index")
    ap.add_argument("--weights", required=True, help="weights root")
    ap.add_argument("--n_candidates", type=int, default=5)
    ap.add_argument("--near_margin_m", type=float, default=0.7)
    ap.add_argument("--legacy_masks_dir", default=None)
    ap.add_argument("--review_png", default=None)
    args = ap.parse_args()
    rec = select_object(args.frames_dir, args.depth_dir, args.out_dir, args.first, args.last, args.prompt,
                        args.object_name, os.path.abspath(args.weights), args.n_candidates,
                        args.near_margin_m, legacy_masks_dir=args.legacy_masks_dir,
                        review_png=args.review_png)
    summary = {k: rec.get(k) for k in ("status", "mode", "prompt", "reference_frame", "confidence",
                                       "candidate_hits", "gap_person_heights", "depth_gap_m",
                                       "legacy_box_iou", "legacy_match")}
    print("SELECT " + json.dumps(summary))
    if rec["status"] != "ok":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
