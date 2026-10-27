#!/usr/bin/env python3
"""Read-only cross-stage risk audit on exactly the released scored frames.

This detects disagreement between stage-4 'accepted' and stage-6 actual fit,
stage-8 scale boundary choices, and high local PEN without modifying outputs.
It is a triage report, not a held-out leaderboard evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import paths


def episode_risks(mask_accepted: bool, median_iou: float, lost_fraction: float,
                  scale_at_edge: bool, local_pen_cm: float) -> list[str]:
    flags = []
    if mask_accepted and (median_iou < .5 or lost_fraction > .05):
        flags.append('mask_accepted_but_pose_bad')
    if lost_fraction > .05:
        flags.append('tracking_lost')
    if median_iou < .5:
        flags.append('poor_object_reprojection')
    if scale_at_edge:
        flags.append('scale_search_boundary')
    if local_pen_cm >= .1:
        flags.append('hand_object_penetration')
    return flags


def audit(data: Path) -> dict:
    meta = {r['episode_index']: r['object'] for r in
            (json.loads(s) for s in (data / 'challenge/track_1/meta/episodes_metadata.jsonl').read_text().splitlines())}
    spans = json.loads((data / 'submission/scored_spans.json').read_text())
    proxy_path = data / 'logs/eval/beta/prodnow/proxies.json'
    proxies = {r['episode']: r for r in json.loads(proxy_path.read_text())['episodes']}
    rows = []
    for e in range(30):
        w = data / 'out' / f'episode_{e:06d}'
        m = json.loads((w / 'track1_meta.json').read_text())
        states = json.loads((w / 'track/states.json').read_text())['frames']
        local = [int(f - m['frame_offset']) for f in spans[str(e)]['frames']]
        if not local or min(local) < 0 or max(local) >= len(states):
            raise ValueError(f'episode {e}: scorer frame IDs do not index the trimmed clip')
        scored = [states[i] for i in local]
        iou = float(np.median([r['iou'] for r in scored]))
        lost = float(np.mean([r['state'] == 'lost' for r in scored]))
        quality = json.loads((w / 'masks_quality.json').read_text())
        sm = json.loads((w / 'track/scale_match.json').read_text())
        k = sm['best']['k']
        params = sm['params']
        edge = k <= params['k_min'] + .001 or k >= params['k_max'] - .001
        pen = float(proxies[e]['pen_cm'])
        flags = episode_risks(quality['accepted'], iou, lost, edge, pen)
        rows.append({'episode': e, 'object': meta[e], 'scored_frames': len(local),
                     'stage4_accepted': bool(quality['accepted']), 'stage6_median_scored_iou': iou,
                     'stage6_lost_scored_fraction': lost, 'stage8_choice': sm['chosen'],
                     'stage8_boundary': edge, 'local_pen_cm': pen,
                     'flags': flags})
    counts = {flag: sum(flag in r['flags'] for r in rows)
              for flag in sorted({f for r in rows for f in r['flags']})}
    return {'scored_frame_basis': 'submission/scored_spans.json exact frame IDs',
            'proxy_basis': str(proxy_path), 'flag_counts': counts, 'episodes': rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    rep = audit(paths.DATA)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2)+'\n')
    print('flags:', rep['flag_counts'])
    for row in rep['episodes']:
        if row['flags']:
            print(f"ep {row['episode']:2d} {row['object']:20s} IoU {row['stage6_median_scored_iou']:.2f} "
                  f"lost {row['stage6_lost_scored_fraction']:.1%} PEN {row['local_pen_cm']:.3f}: "
                  f"{', '.join(row['flags'])}")


if __name__ == '__main__':
    main()
