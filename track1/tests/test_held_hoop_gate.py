"""An episode-1 held-ring pilot must improve coverage without breaking pose/image fit."""
import sys
import unittest
from pathlib import Path
import json

T1=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(T1))
from held_hoop_gate import compare_held_hoop  # noqa: E402
import paths  # noqa: E402


class HeldHoopGate(unittest.TestCase):
    def test_gain_requires_lost_reduction_and_continuity(self):
        before=[{'iou':.45,'state':'lost' if i<6 else 'observed','rot_step':3,
                 'trans_step_cm':.1} for i in range(20)]
        after=[{**r,'iou':.52,'state':'observed'} for r in before]
        self.assertTrue(compare_held_hoop(before,after,list(range(20)))['pass'])
        for i in range(15):after[i]['iou']=.20
        self.assertFalse(compare_held_hoop(before,after,list(range(20)))['pass'])
        for r in after:r['iou']=.52
        for i in range(10):after[i]['rot_step']=70
        self.assertFalse(compare_held_hoop(before,after,list(range(20)))['pass'])
        with self.assertRaises(ValueError):
            compare_held_hoop(before,after,[30])

    def test_completed_visibility_pilot_did_not_recover_scored_coverage(self):
        path = paths.DATA / 'experiments/hoop_vis_v1/stage6_gate.json'
        if not path.exists():
            self.skipTest('episode-1 held-hoop pilot not yet evaluated')
        r = json.loads(path.read_text())
        self.assertFalse(r['pass'])
        self.assertGreater(r['after']['lost_fraction'], r['before']['lost_fraction'])
        self.assertFalse(r['checks']['lost_stretches'])

    def test_quality_ranked_pilot_rejected_pose_jumps_and_lost_coverage(self):
        path = paths.DATA / 'experiments/hoop_quality_relaxed_v1/stage6_gate.json'
        if not path.exists():
            self.skipTest('quality-ranked hoop pilot not yet evaluated')
        r = json.loads(path.read_text())
        self.assertFalse(r['pass'])
        self.assertGreater(r['after']['median_iou'], r['before']['median_iou'])
        self.assertGreater(r['after']['lost_fraction'], r['before']['lost_fraction'])
        self.assertFalse(r['checks']['lost_stretches'])
        self.assertFalse(r['checks']['translation_continuity'])


if __name__=='__main__':
    unittest.main()
