"""The hoop stage-6 gate cannot pass only by scoring its own repaired masks."""
import sys
import unittest
from pathlib import Path

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
from hoop_trial_gate import compare_states  # noqa: E402
from finish_hoop_trial import require_stage6_gate  # noqa: E402
import tempfile
import json


class HoopTrialGate(unittest.TestCase):
    def test_stronger_edited_fit_requires_unchanged_frame_stability(self):
        baseline = [{'iou': .40, 'state': 'observed'} for _ in range(20)]
        scored = list(range(20))
        edited = set(range(10, 20))
        trial = [dict(x) for x in baseline]
        for i in edited:
            trial[i]['iou'] = .70
        self.assertTrue(compare_states(baseline, trial, scored, edited)['pass'])
        for i in range(10):
            trial[i]['iou'] = .30
        self.assertFalse(compare_states(baseline, trial, scored, edited)['pass'])
        for i in range(10):
            trial[i]['iou'] = .40
        trial[0]['state'] = 'lost'
        self.assertFalse(compare_states(baseline, trial, scored, edited)['pass'])
        with self.assertRaises(ValueError):
            compare_states(baseline, trial, list(edited), edited)

    def test_downstream_stages_refuse_rejected_or_missing_tracking_gate(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as folder:
            p = Path(folder)
            with self.assertRaises(ValueError):
                require_stage6_gate(p)
            (p / 'stage6_gate.json').write_text(json.dumps({'pass': False}))
            with self.assertRaises(ValueError):
                require_stage6_gate(p)
            (p / 'stage6_gate.json').write_text(json.dumps({'pass': True}))
            self.assertTrue(require_stage6_gate(p)['pass'])


if __name__ == '__main__':
    unittest.main()
