"""CARI4D collision-loss feasibility; NOT a pretrained-model or official-PEN test."""
import sys
import unittest
from pathlib import Path

import torch

SOURCE = Path('/workspace/v2d_cari4d/lib/cari4d')
sys.path.insert(0, str(SOURCE))


@unittest.skipUnless(SOURCE.is_dir(), 'native-MHR CARI4D source not mounted')
class Cari4dFeasibility(unittest.TestCase):
    def test_reverse_direction_hand_inside_object_has_gradient(self):
        from lib_mhr.sdf import object_sdf_penetration_loss

        def cube_sdf(points):
            q = points.abs() - 0.02
            return torch.clamp(q, min=0).norm(dim=-1) + torch.clamp(q.max(dim=-1).values, max=0)

        hand = torch.tensor([[[0.01, 0.0, 0.0], [0.04, 0.0, 0.0]]], requires_grad=True)
        loss = object_sdf_penetration_loss(hand, cube_sdf)
        self.assertAlmostEqual(float(loss), 0.005, places=5)
        loss.backward()
        self.assertLess(float(hand.grad[0, 0, 0]), 0)
        self.assertEqual(float(hand.grad[0, 1, 0]), 0.0)

    def test_default_cari4d_penetration_direction_is_not_kaggle_pen(self):
        text = (SOURCE / 'learning/training/mhr_opt_refineout.py').read_text()
        parity = text.split('class MHRParityPostOptConfig:', 1)[1]
        self.assertIn('return object_inside_human_penetration_loss(', parity)
        self.assertIn('hand_surface_spec_path:', parity)
        self.assertIn('contact_activation_distance_m:', parity)


if __name__ == '__main__':
    unittest.main()
