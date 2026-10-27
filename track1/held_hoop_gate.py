#!/usr/bin/env python3
"""Gate episode-1 anchor changes using the SAME unmodified scored-frame masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import paths


def compare_held_hoop(before: list[dict], after: list[dict], frames: list[int]) -> dict:
    if not frames or len(before) != len(after) or min(frames) < 0 or max(frames) >= len(before):
        raise ValueError('invalid scored frame mapping or tracking state length')
    def summary(rows):
        scored = [rows[f] for f in frames]
        return {'median_iou': float(np.median([r['iou'] for r in scored])),
                'lost_fraction': float(np.mean([r['state'] == 'lost' for r in scored])),
                'large_rotation_steps': sum(r.get('rot_step', 0) > 45 for r in scored),
                'p99_translation_step_cm': float(np.percentile([r.get('trans_step_cm', 0) for r in scored], 99))}
    a,b=summary(before),summary(after)
    checks={'image_fit': b['median_iou'] >= a['median_iou'] - .02,
            'lost_stretches': b['lost_fraction'] <= a['lost_fraction'] - .10,
            'rotation_continuity': b['large_rotation_steps'] <= a['large_rotation_steps'] + 2,
            'translation_continuity': b['p99_translation_step_cm'] <= a['p99_translation_step_cm'] + 5.}
    return {'pass':all(checks.values()),'scored_frames':len(frames),'before':a,'after':b,'checks':checks}


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trial',type=Path,required=True)
    args=ap.parse_args()
    trial=args.trial.resolve()
    if not trial.is_relative_to(paths.DATA.resolve()/'experiments'):
        raise SystemExit('trial must stay under V2D_DATA/experiments')
    base=paths.DATA/'out/episode_000001';pilot=trial/'out/episode_000001'
    meta=json.loads((base/'track1_meta.json').read_text());offset=meta['frame_offset']
    frames=[int(f-offset) for f in json.loads((paths.DATA/'submission/scored_spans.json').read_text())['1']['frames']]
    old=json.loads((base/'track/states.json').read_text())['frames']
    new=json.loads((pilot/'track/states.json').read_text())['frames']
    result=compare_held_hoop(old,new,frames)
    (trial/'stage6_gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print('held-hoop stage6 scored frames',len(frames),'mask unchanged; before:',result['before'],
          'after:',result['after'],'checks:',result['checks'],'PASS' if result['pass'] else 'REJECT')
    if not result['pass']:
        raise SystemExit('held-hoop stage 6 did not pass image/lost/continuity gates')


if __name__=='__main__':
    main()
