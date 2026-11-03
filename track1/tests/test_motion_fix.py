"""motion_fix: symmetry spin removal, flip unwrap, symmetry-switch tumble repair, pinned smoothing, Kabsch."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
import motion_fix as m  # noqa: E402


def steps_deg(R):
    return np.degrees((Rotation.from_matrix(R[1:]) * Rotation.from_matrix(R[:-1]).inv()).magnitude())


class MotionFix(unittest.TestCase):
    def test_spin_removal_keeps_visible_axis_and_removes_spin(self):
        rng = np.random.default_rng(0)
        axis = np.array([0.0, 1.0, 0.0])
        tilt = Rotation.from_rotvec(np.c_[np.linspace(0, .5, 100), np.zeros(100), np.linspace(0, .3, 100)])
        spin = Rotation.from_rotvec(np.outer(np.cumsum(rng.uniform(-.3, .3, 100)), axis))
        R = (tilt * spin).as_matrix()
        out = m.remove_spin(R, axis, [])
        np.testing.assert_allclose(np.einsum('tij,j->ti', out, axis), np.einsum('tij,j->ti', R, axis), atol=1e-9)
        self.assertLess(steps_deg(out).max(), steps_deg(tilt.as_matrix()).max() + 1e-6)
        self.assertGreater(steps_deg(R).mean(), 5 * steps_deg(out).mean())

    def test_flip_unwrap(self):
        F = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
        R = np.stack([np.eye(3)] * 10)
        R[5:] = R[5:] @ F                       # same symmetric object, flipped representation
        np.testing.assert_allclose(m.remove_spin(R, np.array([0, 0, 1.]), [F]), np.stack([np.eye(3)] * 10), atol=1e-9)

    def test_kabsch_recovers_motion(self):
        P = np.random.default_rng(1).normal(size=(50, 3))
        Rt = Rotation.from_rotvec([.2, -.4, .1]).as_matrix()
        R, t = m.kabsch(P, P @ Rt.T + [1, 2, 3])
        np.testing.assert_allclose(R, Rt, atol=1e-9)
        np.testing.assert_allclose(t, [1, 2, 3], atol=1e-9)

    def test_smooth_window_pins_both_ends(self):
        t = np.cumsum(np.random.default_rng(2).normal(size=(80, 3)), 0)
        R = np.stack([np.eye(3)] * 80)
        before = t.copy()
        m.smooth_window(R, t, 10, 60)
        np.testing.assert_allclose(t[:12], before[:12], atol=1e-6)
        np.testing.assert_allclose(t[59:], before[59:], atol=1e-6)
        self.assertFalse(np.allclose(t[30], before[30]))

    def test_switch_repair_replaces_tumble_with_single_frame_switch(self):
        n, j = 80, 40
        base = Rotation.from_rotvec(np.c_[np.linspace(0, .4, n), np.linspace(0, .2, n), np.zeros(n)]).as_matrix()
        G = Rotation.from_rotvec([0, 0, np.pi]).as_matrix()          # 180-degree representation switch
        Rr = base.copy()
        Rr[j:] = Rr[j:] @ G
        Rf = Rr.copy()                                               # a smoother's tumble over +-12 frames
        for k in range(j - 12, j + 12):
            Rf[k] = base[k] @ Rotation.from_rotvec([0, 0, np.pi * (k - (j - 12)) / 24]).as_matrix()
        a, b, new = m.repair_window(Rr, Rf, j, G)
        self.assertLessEqual(a, j - 12)
        self.assertGreaterEqual(b, j + 11)
        np.testing.assert_allclose(new[:2], Rf[a:a + 2], atol=1e-6)
        np.testing.assert_allclose(new[-2:], Rf[b - 1:b + 1], atol=1e-6)
        look = np.concatenate([new[:j - a], new[j - a:] @ G.T])     # appearance follows the raw track
        dev = np.degrees((Rotation.from_matrix(look) * Rotation.from_matrix(base[a:b + 1]).inv()).magnitude())
        self.assertLess(dev.max(), 3.0)
        s = steps_deg(new)
        self.assertGreater(s[j - a - 1], 170)                        # the switch itself: one frame
        self.assertLess(np.delete(s, j - a - 1).max(), 3.0)


if __name__ == '__main__':
    unittest.main()
