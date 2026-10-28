"""Body smoothing must never average across an Euler-angle representation switch."""
import sys
import unittest
from pathlib import Path

import numpy as np

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
from smooth_body import smooth_pose  # noqa: E402


class SmoothBody(unittest.TestCase):
    def test_2pi_wrap_of_global_rotation_is_unwrapped_not_averaged(self):
        n = 60
        P = np.zeros((n, 136), np.float32)
        P[:, 4] = np.linspace(0, .5, n)
        P[30:, 4] += 2 * np.pi                                   # same rotation, other branch
        Q, cuts = smooth_pose(P, 3.0)
        self.assertEqual(cuts, 0)
        self.assertLess(np.abs(np.diff(Q[:, 4])).max(), .05)     # continuous after unwrap

    def test_alternative_euler_switch_cuts_the_sequence(self):
        n = 60
        P = np.zeros((n, 136), np.float32)
        P[:, 10] = .3
        P[30:, 10] = .3 + np.pi                                  # e.g. (a+pi, pi-b, c+pi) branch switch
        P[:, 0] = np.linspace(0, 1, n)
        P[:, 20] = np.sin(np.linspace(0, 3, n))                  # a joint that never switches
        Q, cuts = smooth_pose(P, 3.0)
        self.assertEqual(cuts, 1)
        np.testing.assert_allclose(Q[:30, 10], .3, atol=1e-6)    # no blending across the switch
        np.testing.assert_allclose(Q[30:, 10], .3 + np.pi, atol=1e-5)
        np.testing.assert_allclose(Q[29:31, 20], P[29:31, 20], atol=1e-6)   # all rotations raw at the switch
        from smooth_body import taper
        w = taper(slice(30, 60), 60, 3.0)
        self.assertEqual(w[0], 0.0)                               # raw fit right at the switch
        self.assertEqual(w[-1], 1.0)                              # sequence end is not a switch
        self.assertLess(np.abs(np.diff(Q[25:35, 0], 2)).max(), 1e-6)  # root translation: continuous

    def test_noise_is_reduced_and_dtype_kept(self):
        rng = np.random.default_rng(0)
        P = (rng.normal(scale=.02, size=(200, 136))).astype(np.float32)
        Q, _ = smooth_pose(P, 4.0)
        self.assertEqual(Q.dtype, np.float32)
        acc = lambda X: np.abs(X[:-2] - 2 * X[1:-1] + X[2:]).mean()   # noqa: E731
        self.assertLess(acc(Q), acc(P) / 10)


if __name__ == '__main__':
    unittest.main()
