"""Guard the episode-16 overlap before assembling a full PEN + smoothing candidate."""
import sys
import unittest
from pathlib import Path

import numpy as np

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
import paths  # noqa: E402
from combine_ablation import compose_episode  # noqa: E402


class CombineAblation(unittest.TestCase):
    def test_only_translation_is_composed(self):
        base = {'pose': np.zeros((2, 136), np.float32), 'object_scale': np.array(1., np.float32),
                'object_rotation': np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
                'object_translation': np.array([[0, 0, 3], [0, 0, 3]], np.float32)}
        pen = {**base, 'object_translation': base['object_translation'] + [0.01, 0, 0]}
        smooth = {**base, 'object_translation': base['object_translation'] + [0, 0, 0.02]}
        combined = compose_episode(base, pen, smooth, overlap='additive')
        np.testing.assert_allclose(combined['object_translation'],
                                   base['object_translation'] + [0.01, 0, 0.02], atol=1e-6)
        np.testing.assert_array_equal(combined['pose'], base['pose'])
        np.testing.assert_allclose(compose_episode(base, pen, smooth, overlap='pen_only')
                                   ['object_translation'], pen['object_translation'], atol=1e-8)

    def test_mismatched_body_or_excessive_pen_shift_rejected(self):
        base = {'pose': np.zeros((1, 136), np.float32),
                'object_translation': np.zeros((1, 3), np.float32)}
        with self.assertRaises(AssertionError):
            compose_episode(base, {**base, 'pose': np.ones((1, 136), np.float32)}, base,
                            overlap='additive')
        with self.assertRaises(ValueError):
            compose_episode(base, {**base, 'object_translation': np.array([[0.04, 0, 0]], np.float32)},
                            base, overlap='additive')

    def test_actual_episode_16_inputs_share_every_other_field(self):
        root = paths.DATA
        folders = (root / 'submission/episodes', root / 'experiments/penfix_smooth/episodes',
                   root / 'experiments/smooth_t150_full/episodes')
        if not all((p / 'episode_000016.npz').exists() for p in folders):
            self.skipTest('episode-16 experiment inputs unavailable')
        def read(path):
            with np.load(path / 'episode_000016.npz') as data:
                return {k: data[k] for k in data.files}
        base, pen, smooth = map(read, folders)
        out = compose_episode(base, pen, smooth, overlap='additive')
        self.assertEqual(set(out), set(base))
        for key in base:
            if key != 'object_translation':
                np.testing.assert_array_equal(out[key], base[key])
        self.assertLess(float(np.linalg.norm(out['object_translation']-base['object_translation'],
                                              axis=1).max()), 0.05)


if __name__ == '__main__':
    unittest.main()
