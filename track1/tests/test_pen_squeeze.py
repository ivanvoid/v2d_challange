"""pen_squeeze: shrink only on evidence, smoothing never crosses a scored-frame gap."""
import sys
import unittest
from pathlib import Path

import numpy as np

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
from pen_squeeze import cap, choose_shrink, interpolate_rows, self_acc, smooth_per_stretch  # noqa: E402


class PenSqueeze(unittest.TestCase):
    def test_least_shrink_within_tolerance_of_the_best(self):
        rows = [{'c': 1.0, 'pen_holdout_cm': .30}, {'c': .96, 'pen_holdout_cm': .055},
                {'c': .92, 'pen_holdout_cm': .05}, {'c': .90, 'pen_holdout_cm': .049}]
        self.assertEqual(choose_shrink(rows, .01)['c'], .96)
        self.assertEqual(choose_shrink(rows, .0)['c'], .90)
        self.assertEqual(choose_shrink([{'c': 1.0, 'pen_holdout_cm': .01}, {'c': .9, 'pen_holdout_cm': .012}], .01)['c'], 1.0)

    def test_smoothing_stays_inside_each_scored_stretch(self):
        frames = np.array([10, 11, 12, 13, 40, 41, 42])
        shift = np.zeros((7, 3)); shift[4:, 0] = .05
        sm = smooth_per_stretch(shift, frames, 3.0)
        np.testing.assert_allclose(sm[:4], 0)             # the later stretch does not bleed back
        np.testing.assert_allclose(sm[4:, 0], .05)

    def test_rows_exact_on_scored_frames_and_cap(self):
        frames = np.array([2, 3, 6])
        sh = np.array([[.01, 0, 0], [.02, 0, 0], [.05, 0, 0]])
        r = interpolate_rows(sh, frames, 8)
        np.testing.assert_allclose(r[frames], sh)
        np.testing.assert_allclose(r[4:6, 0], [.03, .04])
        self.assertAlmostEqual(np.linalg.norm(cap(np.array([[.3, .4, 0]]), .05)), .05)
        self.assertEqual(self_acc(np.zeros((3, 3)), np.array([1, 5, 9])), 0.0)


if __name__ == '__main__':
    unittest.main()
