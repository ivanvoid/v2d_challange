#!/usr/bin/env python3
"""Gate hoop stage-6 retracking on changed and UNCHANGED scored frames separately.

A repaired mask can trivially raise its own IoU; unaffected frames must not
regress. This remains a consistency check, NOT hidden CD-O/ACC-O scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import paths


def compare_states(before: list[dict], after: list[dict], local_scored: list[int],
                   edited: set[int], *, min_edited_gain: float = .10,
                   max_unedited_drop: float = .03,
                   max_lost_increase: float = .02) -> dict:
    if len(before) != len(after) or not local_scored or not edited:
        raise ValueError('frame counts, scored IDs and edited set must be nonempty and aligned')
    if min(local_scored) < 0 or max(local_scored) >= len(before):
        raise ValueError('scored frames must index the local trimmed clip')
    groups = {'edited': [i for i in local_scored if i in edited],
              'unedited': [i for i in local_scored if i not in edited]}
    if not groups['edited'] or not groups['unedited']:
        raise ValueError('both edited and untouched scored frames are required for an independent gate')
    metrics = {}
    for label, indices in groups.items():
        b, a = (float(np.median([rows[i]['iou'] for i in indices])) for rows in (before, after))
        lb, la = (float(np.mean([rows[i]['state'] == 'lost' for i in indices])) for rows in (before, after))
        metrics[label] = {'frames': len(indices), 'iou_before': b, 'iou_after': a,
                          'lost_before': lb, 'lost_after': la}
    checks = {
        'edited_median_iou_gain': (metrics['edited']['iou_after']
                                   >= metrics['edited']['iou_before'] + min_edited_gain),
        'unchanged_median_iou_preserved': (metrics['unedited']['iou_after']
                                          >= metrics['unedited']['iou_before'] - max_unedited_drop),
        'unchanged_lost_frames_not_increased': (metrics['unedited']['lost_after']
                                               <= metrics['unedited']['lost_before'] + max_lost_increase),
    }
    return {'pass': all(checks.values()), 'checks': checks, 'groups': metrics}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trial', type=Path, required=True)
    args = ap.parse_args()
    base = paths.DATA / 'out/episode_000000'
    trial = args.trial.resolve()
    if not trial.is_relative_to(paths.DATA.resolve() / 'experiments'):
        raise SystemExit('trial must be inside V2D_DATA/experiments')
    other = trial / 'out/episode_000000'
    before = json.loads((base / 'track/states.json').read_text())['frames']
    after = json.loads((other / 'track/states.json').read_text())['frames']
    offset = json.loads((base / 'track1_meta.json').read_text())['frame_offset']
    scored = [int(f - offset) for f in json.loads((paths.DATA / 'submission/scored_spans.json').read_text())['0']['frames']]
    changed = set(json.loads((trial / 'report.json').read_text())['modified_local_frames'])
    result = compare_states(before, after, scored, changed)
    (trial / 'stage6_gate.json').write_text(json.dumps(result, indent=2)+'\n')
    for name, row in result['groups'].items():
        print(f"{name} scored {row['frames']} frames: IoU {row['iou_before']:.3f} -> "
              f"{row['iou_after']:.3f}; lost {row['lost_before']:.1%} -> {row['lost_after']:.1%}")
    print('checks:', result['checks'], 'PASS' if result['pass'] else 'REJECT')
    if not result['pass']:
        raise SystemExit('isolated hoop re-track failed the unaffected-frame/pose gate')


if __name__ == '__main__':
    main()
