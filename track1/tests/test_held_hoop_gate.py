"""An episode-1 held-ring pilot must improve coverage without breaking pose/image fit."""
import sys
import unittest
from pathlib import Path

T1=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(T1))
from held_hoop_gate import compare_held_hoop  # noqa: E402


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


if __name__=='__main__':
    unittest.main()
