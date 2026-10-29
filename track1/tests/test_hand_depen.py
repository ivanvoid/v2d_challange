"""hand_depen: the SDF, the triplet selection and the parameter set must be right before trusting it."""
import sys
import unittest
from pathlib import Path

import numpy as np

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
from hand_depen import SdfGrid, consecutive_triplets, hand_param_cols  # noqa: E402


class HandDepen(unittest.TestCase):
    def test_hand_parameters_are_fingers_and_wrist_rotations_only(self):
        cols = hand_param_cols()
        self.assertEqual(len(cols), 60)
        self.assertTrue(set(range(68, 122)) <= set(cols))
        self.assertFalse({0, 1, 2, 3, 4, 5} & set(cols))           # never the root
        arm = hand_param_cols(arm=True)
        self.assertEqual(len(arm), 74)
        self.assertFalse(set(range(0, 30)) & set(arm))             # no root/legs/spine

    def test_triplets_never_span_a_scored_gap(self):
        np.testing.assert_array_equal(consecutive_triplets(np.array([3, 4, 5, 6, 20, 21, 22])), [1, 2, 5])

    def test_sdf_sign_and_distance_on_a_cube(self):
        import torch
        import trimesh
        box = trimesh.creation.box(extents=(.2, .2, .2))
        g = SdfGrid(np.asarray(box.vertices), np.asarray(box.faces), res=48).torch('cpu')
        pts = torch.tensor([[0., 0, 0], [0.05, 0, 0], [0.125, 0, 0], [0, 0.13, 0], [0.5, 0, 0]])
        v = g(pts).numpy()
        self.assertAlmostEqual(v[0], -0.10, delta=.006)              # centre: 10 cm inside
        self.assertAlmostEqual(v[1], -0.05, delta=.006)
        self.assertAlmostEqual(v[2], 0.025, delta=.006)             # 2.5 cm outside
        self.assertAlmostEqual(v[3], 0.03, delta=.006)
        self.assertGreater(v[4], 0.03)                              # beyond the grid: clamped, still outside
        pts.requires_grad_(True)
        g(pts).sum().backward()                                     # differentiable
        self.assertGreater(float(pts.grad[2, 0]), 0.5)              # moving outwards increases sdf


if __name__ == '__main__':
    unittest.main()
