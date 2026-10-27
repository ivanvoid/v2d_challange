"""Regression tests for the active stage-8 scale_match.json review format."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
import paths  # noqa: E402
from review_stage8 import review_episode, scale_summary  # noqa: E402


class ReviewStage8(unittest.TestCase):
    def test_current_scale_report_uses_best_and_none_not_obsolete_variants(self):
        p = paths.DATA / 'out_beta/ripple_stool/episode_000025/track/scale_match.json'
        sm = json.loads(p.read_text())
        chosen, none = scale_summary(sm)
        self.assertEqual(chosen, sm['best'])
        self.assertEqual(none, sm['legacy_variants']['none'])
        self.assertGreater(chosen['score'], sm['legacy_variants']['global']['score'])

    def test_legacy_scale_report_remains_readable(self):
        old = {'chosen': 'global', 'variants': {'none': {'gap_median_cm': 3},
                                               'global': {'gap_median_cm': 1}}}
        chosen, none = scale_summary(old)
        self.assertEqual(chosen['gap_median_cm'], 1)
        self.assertEqual(none['gap_median_cm'], 3)

    def test_generate_review_sheet_from_real_scale_report_without_model_rerun(self):
        w = paths.DATA / 'out_beta/ripple_stool/episode_000025'
        sub = paths.DATA / 'submission_beta/ripple_stool'
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as tmp:
            r = review_episode(str(w), str(sub), tmp)
            self.assertEqual(r['episode'], 25)
            self.assertTrue((Path(tmp) / r['sheet']).is_file())
            names = [c['check'] for c in r['checks']]
            self.assertIn('scale match helps', names)
            self.assertIn('hands touch the object', names)


if __name__ == '__main__':
    unittest.main()
