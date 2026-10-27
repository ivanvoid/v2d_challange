#!/usr/bin/env python3
"""Matched-episode proxy comparison; avoids the misleading unmatched `eval_local.py compare` mean."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ("pen_cm", "gap_cm_median", "contact_frac", "acc_h_self", "acc_o_self",
        "object_size_m", "body_minus_object_z_cm")


def matched_summary(control: dict, candidate: dict) -> dict:
    old = {r["episode"]: r for r in control["episodes"]}
    new = {r["episode"]: r for r in candidate["episodes"]}
    episodes = sorted(new)
    if not episodes or any(e not in old for e in episodes):
        raise ValueError("candidate has no episodes or the control lacks a candidate episode")
    means = {label: {k: float(np.mean([rows[e][k] for e in episodes])) for k in KEYS}
             for label, rows in (("control", old), ("candidate", new))}
    per_episode = [{"episode": e, "object": new[e]["object"],
                    "control": {k: old[e][k] for k in KEYS},
                    "candidate": {k: new[e][k] for k in KEYS}} for e in episodes]
    return {"episodes": episodes, "matched_mean": means, "per_episode": per_episode}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = matched_summary(json.loads(args.control.read_text()), json.loads(args.candidate.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("episodes:", result["episodes"])
    for k in KEYS:
        c, v = (result["matched_mean"][label][k] for label in ("control", "candidate"))
        print(f"{k:24s} {c:8.4f} -> {v:8.4f} (delta {v-c:+.4f})")
    for r in result["per_episode"]:
        a, b = r["control"], r["candidate"]
        print(f"episode {r['episode']:2d} {r['object']:19s}: gap {a['gap_cm_median']:.2f} -> "
              f"{b['gap_cm_median']:.2f} cm; contact {a['contact_frac']:.2f} -> {b['contact_frac']:.2f}; "
              f"PEN {a['pen_cm']:.4f} -> {b['pen_cm']:.4f}; accO {a['acc_o_self']:.3f} -> {b['acc_o_self']:.3f}")


if __name__ == "__main__":
    main()
