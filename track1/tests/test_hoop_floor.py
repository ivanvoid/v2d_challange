"""hoop_floor: derive the floor factor from flat frames; refuse when the evidence is weak."""
import sys
import unittest
from pathlib import Path

import numpy as np

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
from hoop_floor import decide, floor_factor  # noqa: E402


class HoopFloor(unittest.TestCase):
    def test_factor_brings_the_ring_to_the_soles(self):
        n = np.array([0, 1., 0])                                  # floor normal pointing down (+y)
        ring = np.c_[np.cos(np.linspace(0, 6.28, 50)) * .35, np.full(50, 1.20), 3.0 + np.sin(np.linspace(0, 6.28, 50)) * .35]
        flat, k = floor_factor([ring], np.array([[0, 1., 0]]), np.array([1.44]), n, 15)
        self.assertTrue(flat[0])
        self.assertAlmostEqual(k[0], 1.2)                          # 1.20 * 1.2 = 1.44: ring lowest point at the soles
        flat, _ = floor_factor([ring], np.array([[1., 0, 0]]), np.array([1.44]), n, 15)
        self.assertFalse(flat[0])                                  # an upright ring is not lying on the floor

    def test_decision_rules(self):
        flat = np.ones(40, bool)
        self.assertAlmostEqual(decide(np.full(40, 1.19), flat, min_frames=30, max_spread=.05, min_change=.02)[0], 1.19)
        self.assertEqual(decide(np.full(40, 1.005), flat, min_frames=30, max_spread=.05, min_change=.02)[0], 1.0)
        self.assertEqual(decide(np.full(10, 1.2), flat[:10], min_frames=30, max_spread=.05, min_change=.02)[0], 1.0)
        noisy = np.linspace(.9, 1.4, 40)
        self.assertEqual(decide(noisy, flat, min_frames=30, max_spread=.05, min_change=.02)[0], 1.0)


if __name__ == '__main__':
    unittest.main()
