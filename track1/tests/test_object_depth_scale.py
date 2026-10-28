"""Scaling about the camera keeps every image projection and changes only depth and size."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
from object_depth_scale import scale_about_camera  # noqa: E402


class ObjectDepthScale(unittest.TestCase):
    def test_projection_unchanged_depth_and_size_scaled(self):
        R = Rotation.from_rotvec([[.2, .5, -.1], [0, 1, 0]]).as_matrix().astype(np.float32)
        d = {'object_rotation': R, 'object_translation': np.array([[.1, -.2, 3.], [.3, .1, 2.5]], np.float32),
             'object_scale': np.array(1.1, np.float32), 'pose': np.zeros((2, 136), np.float32)}
        out = scale_about_camera(d, 1.04)
        mesh = np.array([[.1, 0, 0], [0, .2, .05], [-.1, -.1, .1]])
        for f in range(2):
            a = d['object_scale'] * mesh @ R[f].T + d['object_translation'][f]
            b = out['object_scale'] * mesh @ R[f].T + out['object_translation'][f]
            np.testing.assert_allclose(b[:, :2] / b[:, 2:], a[:, :2] / a[:, 2:], rtol=1e-5)   # same pixels
            np.testing.assert_allclose(b, 1.04 * a, rtol=1e-5)
        np.testing.assert_array_equal(out['pose'], d['pose'])
        self.assertEqual(out['object_scale'].dtype, np.float32)
        with self.assertRaises(ValueError):
            scale_about_camera(d, 1.5)


if __name__ == '__main__':
    unittest.main()
