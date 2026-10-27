"""Failure-propagation and scorer-input contracts for the active stage runner."""
import json
import sys
import unittest
from pathlib import Path

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
import paths  # noqa: E402


class StageContracts(unittest.TestCase):
    def test_stage8_qa_and_render_failures_are_not_reported_as_success(self):
        runner = (T1.parent / 'run_stage.sh').read_text(encoding='utf-8')
        stage8 = runner.split('stage8_final() {', 1)[1].split('MHR_MODEL=', 1)[0]
        self.assertIn('qa_episode.py', stage8)
        qa_end = stage8.split('qa_episode.py', 1)[1].split('FR=', 1)[0]
        self.assertIn('PIPESTATUS[0]', qa_end)
        self.assertIn('QA FAILED', qa_end)
        renderer = stage8.split('render_results.py', 1)[1]
        self.assertIn('render FAILED', renderer)
        self.assertIn('render produced no sheet', renderer)

    def test_stage4_accepted_the_hula_hoop_despite_known_hole_loss(self):
        quality = paths.DATA / 'out/episode_000000/masks_quality.json'
        if not quality.exists():
            self.skipTest('original stage-4 report missing')
        rep = json.loads(quality.read_text())
        self.assertTrue(rep['accepted'])
        self.assertEqual(rep['status_counts_scored'].get('lost'), 0)
        # Combined with test_hoop_mask_repair's direct mask hole checks, this
        # demonstrates that generic accepted!=correct topology for thin rings.


if __name__ == '__main__':
    unittest.main()
