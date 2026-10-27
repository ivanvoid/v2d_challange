"""Hoop topology and split contracts, without a GPU or a full tracker run."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
import paths  # noqa: E402
from hoop_torus_probe import clear_frames, physical_torus, silhouette_scores, split_temporal  # noqa: E402
from stationary_stool import rendered_mask  # noqa: E402


class HoopTorusProbe(unittest.TestCase):
    def test_one_shared_physical_size_across_episode_object_scales(self):
        for scale in (.925, 1.15, 1.075):
            m = physical_torus(.76, .05, scale)
            self.assertTrue(m.is_watertight)
            self.assertEqual(m.euler_number, 0)  # genus-one closed torus
            self.assertEqual(len(m.split(only_watertight=False)), 1)
            np.testing.assert_allclose(m.extents[[0, 2]] * scale, [.76, .76], atol=.002)
            self.assertAlmostEqual(m.extents[1] * scale, .05, places=3)

    def test_holdout_is_disjoint_and_spread_across_video(self):
        valid = list(range(10, 25)) + list(range(65, 80)) + list(range(110, 125))
        anchors, holdout = split_temporal(valid, 150, n_bins=6)
        self.assertTrue(set(anchors).isdisjoint(holdout))
        self.assertLess(min(anchors), 25)
        self.assertGreater(max(anchors), 100)
        self.assertGreater(len(holdout), len(anchors))

    def test_silhouette_score_distinguishes_wrong_diameter(self):
        K = np.array([[800., 0, 320.], [0, 800., 240.], [0, 0, 1.]])
        scale = 1.
        pose = (np.eye(3), np.array([0., 0., 3.]))
        target = physical_torus(.76, .05, scale)
        observed = rendered_mask(target.vertices, target.faces, *pose, scale, K, (120, 160))
        data = {'K': K, 'hw': (120, 160), 'scale': scale, 'poses': [pose], 'masks': [observed]}
        perfect = silhouette_scores(target, data)[0]
        wrong = silhouette_scores(physical_torus(.90, .05, scale), data)[0]
        self.assertAlmostEqual(perfect, 1.)
        self.assertGreater(perfect, wrong + .15)

    def test_episode_one_has_no_high_fit_anchors_in_production(self):
        w = paths.DATA / 'out/episode_000001/track/states.json'
        if not w.exists():
            self.skipTest('production hoop tracking artifacts not found')
        rows = json.loads(w.read_text())['frames']
        self.assertEqual(clear_frames(rows), [])
        for e in (0, 2):
            src = paths.DATA / 'out' / f'episode_{e:06d}/track/states.json'
            self.assertGreaterEqual(len(clear_frames(json.loads(src.read_text())['frames'])), 15)


if __name__ == '__main__':
    unittest.main()
