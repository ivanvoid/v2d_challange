"""The H1 / H-depth diagnostics must be correct on synthetic data before their numbers count."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
from t2_diag import (L_HIP, PELVIS, R_HIP, SPINE3, body_frames, first_frame_alignment,  # noqa: E402
                     mean_rotation, zbuffer_median_depth)


def skeleton(n, rng):
    J = np.zeros((n, 127, 3))
    yaw = np.linspace(0, 2.5, n)
    for i in range(n):
        Rb = Rotation.from_euler('y', yaw[i]).as_matrix()
        base = np.array([.3 * i / n, 0, 3.0])
        J[i, PELVIS] = base
        J[i, L_HIP] = base + Rb @ [.1, 0, 0]
        J[i, R_HIP] = base + Rb @ [-.1, 0, 0]
        J[i, SPINE3] = base + Rb @ [0, -.4, 0]
    return J


class T2Diag(unittest.TestCase):
    def test_body_frames_are_rotations(self):
        B = body_frames(skeleton(5, np.random.default_rng(0)))
        for b in B:
            np.testing.assert_allclose(b.T @ b, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(b), 1.0)

    def test_first_frame_alignment_recovers_a_rigid_world_transform(self):
        J = skeleton(40, np.random.default_rng(1))
        B = body_frames(J)
        Rw = Rotation.from_rotvec([.2, -.7, .1]).as_matrix()
        tw = np.array([1., 2., -3.])
        C = Rotation.from_rotvec([0, np.pi / 2, .3]).as_matrix()       # reference root convention
        p_ref = (Rw @ J[:, PELVIS].T).T + tw
        B_ref = np.einsum('ij,tjk,kl->til', Rw, B, C)
        s, R0, t0, C_est = first_frame_alignment(B, J[:, PELVIS], B_ref, p_ref, Rw)
        np.testing.assert_allclose(C_est, C, atol=1e-9)
        np.testing.assert_allclose(R0, Rw, atol=1e-9)
        np.testing.assert_allclose(t0, tw, atol=1e-9)
        np.testing.assert_allclose(mean_rotation(np.stack([C, C])), C, atol=1e-12)

    def test_zbuffer_takes_the_front_surface_over_the_mask(self):
        K = {'fx': 100., 'fy': 100., 'cx': 50., 'cy': 50.}
        xs, ys = np.meshgrid(np.linspace(-.2, .2, 60), np.linspace(-.2, .2, 60))
        front = np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, 2.0)]
        back = front + [0, 0, .5]
        mask = np.zeros((100, 100), bool); mask[45:55, 45:55] = True
        self.assertAlmostEqual(zbuffer_median_depth(np.r_[back, front], K, mask), 2.0)
        self.assertIsNone(zbuffer_median_depth(front, K, np.zeros((100, 100), bool)))


if __name__ == '__main__':
    unittest.main()
