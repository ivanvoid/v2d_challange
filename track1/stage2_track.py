#!/usr/bin/env python3
"""Stage 2a — the actor's person track and person masks, from ONE detector pass.

Detector: Cascade Mask R-CNN ViTDet-B (the library's person detector), 1024 px, score >= --thr, no
full-image fallback; every frame, boxes + masks. Then:
  1. tracks      greedy frame-to-frame association (box IoU >= 0.3, a track may be lost <= 30 frames)
  2. fragments   a track that ends and another that starts within 60 frames are joined when the box
                 continues (centre within 0.5 box heights, height ratio 0.6-1.6) — an abrupt box change
                 (actor picks up a desk, the box loses the desk legs) otherwise splits the actor's track
  3. actor       the joined track with the largest (frames inside the contact span x median box
                 height): the person handling the object for the whole span; bystanders are smaller,
                 further away or present for less of it. The runner-up's share is logged.
  4. per frame   detected     on the actor track
                 redetected   not on it, but a detection continues from the nearest assigned frame
                              (same continuity rule), propagated forward and backward; several
                              candidates -> nearest, flagged ambiguous
                 interpolated internal frames still without one: box interpolated, no mask
                 held         before the first / after the last assigned frame: edge box, no mask

Writes into out/episode_XXXXXX/body/:
  bbox_track.pt          the actor's boxes, the library's format (scores: detection score, or -1 for
                         interpolated / held) — what SAM3D-Body crops to
  person_track.json      per-frame label, score, box, ambiguity; tracks and actor choice; gaps (frames
                         not on the actor track) with scored overlap; counts
  person_masks/NNNNNN.png  actor mask (0/255) for detected / redetected frames
Exit code 3 (acceptance failed) if any SCORED frame is held, or an interpolated run inside the scored
frames is longer than --max_interp frames: the stage is then not stamped and later stages do not run.

    /opt/venvs/detectron2/bin/python track1/stage2_track.py --episode 11 --out_root out --weights weights
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])
    return inter / max(area(a) + area(b) - inter, 1e-9)


def detect_all(frames_dir: str, n: int, weights: str, batch: int, thr: float):
    """Person detections (boxes, scores, masks) on every frame, 1024 px short edge."""
    import detectron2.data.transforms as T
    from v2d.detectron2.lib.build_detector import Detector
    det = Detector(name="vitdet", device=torch.device("cuda"), model_size="b",
                   path=os.path.join(weights, "detectron2", "cascade_mask_rcnn_vitdet_b"), test_score_thresh=0.25)
    tf = T.ResizeShortestEdge(short_edge_length=1024, max_size=1024)
    out = []
    for s in range(0, n, batch):
        inputs = []
        for f in range(s, min(n, s + batch)):
            img = cv2.cvtColor(cv2.imread(os.path.join(frames_dir, f"{f:06d}.png")), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            im = tf(T.AugInput(img)).apply_image(img)
            inputs.append({"image": torch.as_tensor(im.astype("float32").transpose(2, 0, 1)), "height": h, "width": w})
        with torch.no_grad():
            res = det.detector(inputs)
        for r in res:
            ins = r["instances"]
            keep = (ins.pred_classes == 0) & (ins.scores >= thr)
            out.append((ins.pred_boxes.tensor[keep].cpu().numpy(), ins.scores[keep].cpu().numpy(),
                        ins.pred_masks[keep].cpu().numpy()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max_interp", type=int, default=10)
    args = ap.parse_args()

    ep = f"episode_{args.episode:06d}"
    W = os.path.join(os.path.abspath(args.out_root), ep)
    B = os.path.join(W, "body")
    meta = json.load(open(os.path.join(W, "track1_meta.json")))
    n, off = meta["n_frames"], meta["frame_offset"]
    spans = json.load(open(paths.DATA / "submission" / "scored_spans.json"))[str(args.episode)]
    scored = set(np.asarray(spans["frames"]) - off)
    first, last = spans["first"] - off, spans["last"] - off

    t0 = time.time()
    dets = detect_all(os.path.join(W, "frames"), n, os.path.abspath(args.weights), args.batch, args.thr)
    t_det = time.time() - t0

    def continues(pb: np.ndarray, x: np.ndarray) -> float | None:
        """Normalised centre distance if box x continues box pb, else None."""
        h = max(pb[3] - pb[1], 1.0)
        d = np.hypot((x[0] + x[2] - pb[0] - pb[2]) / 2, (x[1] + x[3] - pb[1] - pb[3]) / 2) / h
        r = (x[3] - x[1]) / h
        return d if d <= 0.5 and 0.6 <= r <= 1.6 else None

    # 1. greedy IoU tracking over all detections
    tracks, active = [], []                                            # track = {frame: det index}
    for f in range(n):
        b = dets[f][0]
        pairs = sorted(((iou(tracks[t]["box"], b[k]), t, k) for t in active for k in range(len(b))), reverse=True)
        used_t, used_k = set(), set()
        for v, t, k in pairs:
            if v < 0.3 or t in used_t or k in used_k:
                continue
            tracks[t]["fr"][f] = k; tracks[t]["box"] = b[k]; tracks[t]["last"] = f
            used_t.add(t); used_k.add(k)
        for k in range(len(b)):
            if k not in used_k:
                tracks.append({"fr": {f: k}, "box": b[k], "last": f}); active.append(len(tracks) - 1)
        active = [t for t in active if f - tracks[t]["last"] <= 30]
    # 2. join fragments whose boxes continue across a short gap
    parent = list(range(len(tracks)))
    def root(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]; t = parent[t]
        return t
    order = sorted(range(len(tracks)), key=lambda t: min(tracks[t]["fr"]))
    for a in order:
        fa = max(tracks[a]["fr"]); ba = dets[fa][0][tracks[a]["fr"][fa]]
        best = None
        for c in order:
            fc = min(tracks[c]["fr"])
            if c == a or not (fa < fc <= fa + 60) or root(c) == root(a):
                continue
            d = continues(ba, dets[fc][0][tracks[c]["fr"][fc]])
            if d is not None and (best is None or (fc - fa, d) < best[0]):
                best = ((fc - fa, d), c)
        if best is not None:
            parent[root(best[1])] = root(a)
    groups = {}
    for t in range(len(tracks)):
        for f, k in tracks[t]["fr"].items():
            groups.setdefault(root(t), {})
            if f not in groups[root(t)] or dets[f][1][k] > dets[f][1][groups[root(t)][f]]:
                groups[root(t)][f] = k
    # 3. the actor: most of the contact span, at the largest size
    def g_score(g):
        fr = [f for f in g if first <= f <= last]
        return len(fr) * float(np.median([dets[f][0][g[f]][3] - dets[f][0][g[f]][1] for f in fr])) if fr else 0.0
    ranked = sorted(groups.values(), key=g_score, reverse=True)
    actor = ranked[0]
    share = g_score(ranked[1]) / max(g_score(actor), 1e-9) if len(ranked) > 1 else 0.0

    label = np.array(["none"] * n, dtype=object)
    box = np.zeros((n, 4)); score = np.full(n, -1.0); pick = np.full(n, -1); ambiguous = np.zeros(n, bool)
    for f, k in actor.items():
        label[f], box[f], score[f], pick[f] = "detected", dets[f][0][k], dets[f][1][k], k
    on_track = label == "detected"

    def continue_from(f_prev: int, f: int) -> bool:                    # 4. continuity re-detection
        b, s_, _ = dets[f]
        cand = sorted((d, k) for k, x in enumerate(b) if (d := continues(box[f_prev], x)) is not None)
        if not cand:
            return False
        cand.sort()
        k = cand[0][1]
        label[f], box[f], score[f], pick[f] = "redetected", b[k], s_[k], k
        ambiguous[f] = len(cand) > 1 and cand[1][0] < 1.5 * cand[0][0] + 0.05
        return True
    for _ in range(3):
        changed = False
        for f in range(1, n):                                          # forward
            if label[f] == "none" and label[f - 1] != "none" and continue_from(f - 1, f):
                changed = True
        for f in range(n - 2, -1, -1):                                 # backward
            if label[f] == "none" and label[f + 1] != "none" and continue_from(f + 1, f):
                changed = True
        if not changed:
            break

    have = np.flatnonzero(label != "none")                             # 5. interpolate / hold the rest
    if not len(have):
        raise SystemExit(f"{ep}: no actor detection at all")
    for f in np.flatnonzero(label == "none"):
        if f < have[0] or f > have[-1]:
            g = have[0] if f < have[0] else have[-1]
            label[f], box[f] = "held", box[g]
        else:
            a, b_ = have[have < f][-1], have[have > f][0]
            w = (f - a) / (b_ - a)
            label[f], box[f] = "interpolated", (1 - w) * box[a] + w * box[b_]

    os.makedirs(os.path.join(B, "person_masks"), exist_ok=True)
    for f in range(n):
        if pick[f] >= 0:
            cv2.imwrite(os.path.join(B, "person_masks", f"{f:06d}.png"), dets[f][2][pick[f]].astype(np.uint8) * 255)

    # gaps (frames not on the actor track) and acceptance
    def runs(mask):
        out, i = [], 0
        while i < n:
            if mask[i]:
                j = i
                while j < n and mask[j]:
                    j += 1
                out.append((i, j - 1)); i = j
            else:
                i += 1
        return out
    gaps = []
    for a, b_ in runs(~on_track):
        labs = {k: int((label[a:b_ + 1] == k).sum()) for k in ("redetected", "interpolated", "held")}
        gaps.append({"video_first": int(a + off), "video_last": int(b_ + off), "frames": int(b_ - a + 1),
                     "scored_frames": int(sum(1 for f in range(a, b_ + 1) if f in scored)), "after_repair": labs})
    sc = np.array(sorted(scored))
    held_scored = int(sum(label[f] == "held" for f in sc))
    interp_runs = [(a, b_) for a, b_ in runs(label == "interpolated")]
    longest_interp_scored = max([sum(1 for f in range(a, b_ + 1) if f in scored) for a, b_ in interp_runs] or [0])
    counts = {k: int((label == k).sum()) for k in ("detected", "redetected", "interpolated", "held")}
    counts_scored = {k: int(sum(label[f] == k for f in sc)) for k in ("detected", "redetected", "interpolated", "held")}
    accepted = held_scored == 0 and longest_interp_scored <= args.max_interp
    rep = {"episode": args.episode, "frames": n, "counts": counts, "counts_scored": counts_scored,
           "tracking": {"tracks": len(tracks), "joined_tracks": len(groups), "actor_frames": len(actor),
                        "runner_up_share": round(share, 3)},
           "ambiguous_frames": [int(f + off) for f in np.flatnonzero(ambiguous)],
           "gaps": gaps, "held_scored": held_scored, "longest_interpolated_scored_run": longest_interp_scored,
           "accepted": accepted, "params": vars(args), "detector_seconds": round(t_det),
           "per_frame": [{"video_frame": int(f + off), "label": str(label[f]), "score": float(score[f]),
                          "box": [round(float(x), 1) for x in box[f]], "ambiguous": bool(ambiguous[f])} for f in range(n)]}
    json.dump(rep, open(os.path.join(B, "person_track.json"), "w"), indent=1)
    torch.save({"det_cat_id": 0, "scores": score, "bbox_track": box.astype(np.float32)}, os.path.join(B, "bbox_track.pt"))
    msg = (f"{ep}: actor track — {counts['detected']} detected, {counts['redetected']} re-detected, "
           f"{counts['interpolated']} interpolated, {counts['held']} held (scored: {counts_scored}); "
           f"{len(gaps)} gaps in the actor track; {len(rep['ambiguous_frames'])} ambiguous; {len(groups)} people tracked, "
           f"runner-up {100 * share:.0f}% of the actor; detector {t_det:.0f}s")
    print(msg, flush=True)
    if not accepted:
        print(f"{ep}: ACCEPTANCE FAILED — {held_scored} scored frames held, longest interpolated scored run "
              f"{longest_interp_scored} (> {args.max_interp})", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
