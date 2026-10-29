"""postprocess: the config must only name tools that exist, with their reasons, in a sane order."""
import json
import sys
import unittest
from pathlib import Path

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
from postprocess import split  # noqa: E402

TOOLS = {'ablate_pose_scale', 'stationary_stool', 'static_lock', 'hoop_conic', 'object_depth_scale',
         'joint_fit', 'smooth_body', 'pen_squeeze', 'hand_depen'}


class Postprocess(unittest.TestCase):
    def test_config(self):
        cfg = json.loads((T1 / 'postprocess_config.json').read_text())
        names = [s['name'] for s in cfg['steps']]
        self.assertEqual(len(names), len(set(names)))
        for st in cfg['steps']:
            self.assertIn(st['tool'], TOOLS)
            self.assertTrue((T1 / f"{st['tool']}.py").exists(), st['tool'])
            self.assertIn('_why', st, st['name'])
        order = [s['tool'] for s in cfg['steps']]
        # objects first, then the body, then penetration fixes (they depend on both)
        self.assertLess(order.index('smooth_body'), order.index('pen_squeeze'))
        self.assertLess(order.index('pen_squeeze'), order.index('hand_depen'))
        self.assertLess(order.index('joint_fit'), order.index('smooth_body'))

    def test_split_covers_every_episode_once(self):
        a, b = split(list(range(11)))
        self.assertEqual(sorted(a + b), list(range(11)))
        self.assertFalse(set(a) & set(b))


if __name__ == '__main__':
    unittest.main()
