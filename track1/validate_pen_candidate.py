#!/usr/bin/env python3
"""Gate a PEN trial against matching control episodes before packing a full submission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(control: dict, candidate: dict, *, max_acc_increase: float = 0.03,
            max_gap_increase: float = 2.0, max_contact_drop: float = 0.10,
            min_pen_fraction: float = 0.20, episodes: set[int] | None = None) -> dict:
    before = {r["episode"]: r for r in control["episodes"]}
    rows = []
    for after in candidate["episodes"]:
        if episodes is not None and after["episode"] not in episodes:
            continue
        old = before[after["episode"]]
        pen = old["pen_cm"] - after["pen_cm"]
        ok = {
            "pen_drop": pen >= max(0.001, old["pen_cm"] * min_pen_fraction),
            "object_acc": after["acc_o_self"] <= old["acc_o_self"] + max_acc_increase,
            "contact_gap": after["gap_cm_median"] <= old["gap_cm_median"] + max_gap_increase,
            "contact_fraction": after["contact_frac"] >= old["contact_frac"] - max_contact_drop,
        }
        rows.append({"episode": after["episode"], "pass": all(ok.values()), "checks": ok,
                     "before": {k: old[k] for k in ("pen_cm", "acc_o_self", "gap_cm_median", "contact_frac")},
                     "after": {k: after[k] for k in ("pen_cm", "acc_o_self", "gap_cm_median", "contact_frac")}})
    found = {r["episode"] for r in rows}
    return {"pass": bool(rows) and all(r["pass"] for r in rows) and
            (episodes is None or found == episodes), "episodes": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", help="only changed episodes; other episodes must be unchanged by assembly")
    args = ap.parse_args()
    result = compare(json.loads(args.control.read_text()), json.loads(args.candidate.read_text()),
                     episodes=set(args.episodes) if args.episodes else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["episodes"]:
        print(f"episode {row['episode']}: {'PASS' if row['pass'] else 'REJECT'} {row['checks']} "
              f"PEN {row['before']['pen_cm']:.4f} -> {row['after']['pen_cm']:.4f} cm")
    if not result["pass"]:
        raise SystemExit("candidate rejected by guardrails; not ready for submission")


if __name__ == "__main__":
    main()
