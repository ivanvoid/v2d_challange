"""Scored-frame stage audit must not equate an accepted mask with good tracking."""
import sys
import unittest
from pathlib import Path

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
import paths  # noqa: E402
from stage_audit import audit, episode_risks  # noqa: E402


class StageAudit(unittest.TestCase):
    def test_accepted_mask_with_lost_pose_is_flagged(self):
        flags = episode_risks(True, .2, .5, False, 0.)
        self.assertIn('mask_accepted_but_pose_bad', flags)
        self.assertIn('tracking_lost', flags)
        self.assertIn('poor_object_reprojection', flags)
        self.assertNotIn('hand_object_penetration', flags)
        self.assertEqual(episode_risks(True, .9, 0., False, .001), [])

    def test_scored_frames_audit_finds_hoop_and_foam_failure_modes(self):
        p = paths.DATA / 'logs/eval/beta/prodnow/proxies.json'
        if not p.exists():
            self.skipTest('30-episode control proxies unavailable')
        r = audit(paths.DATA)
        self.assertEqual(len(r['episodes']), 30)
        hoop = r['episodes'][0]
        self.assertTrue(hoop['stage4_accepted'])
        self.assertIn('tracking_lost', hoop['flags'])
        self.assertIn('mask_accepted_but_pose_bad', hoop['flags'])
        self.assertIn('hand_object_penetration', r['episodes'][16]['flags'])


if __name__ == '__main__':
    unittest.main()
