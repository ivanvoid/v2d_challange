"""joint_fit: rotation math, smoothing index, ray split, and the renderer's pixel convention."""
import sys
import unittest
from pathlib import Path

import numpy as np

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
from joint_fit import axis_angle_to_matrix, ray_split, second_diff_index  # noqa: E402


class JointFit(unittest.TestCase):
    def test_axis_angle_matches_scipy(self):
        import torch
        from scipy.spatial.transform import Rotation
        v = np.array([[.3, -.2, .9], [0, 0, 0], [1e-9, 0, 0]])
        np.testing.assert_allclose(axis_angle_to_matrix(torch.as_tensor(v)).numpy(),
                                   Rotation.from_rotvec(v).as_matrix(), atol=1e-9)

    def test_second_differences_only_inside_scored_stretches(self):
        np.testing.assert_array_equal(second_diff_index(np.array([5, 6, 7, 8, 20, 21, 22])), [1, 2, 5])

    def test_ray_split(self):
        import torch
        c = torch.tensor([[0., 0, 3]]); dlt = torch.tensor([[.1, 0, .2]])
        along, lat = ray_split(dlt, c)
        self.assertAlmostEqual(float(along[0]), .2)
        np.testing.assert_allclose(lat.numpy(), [[.1, 0, 0]], atol=1e-7)

    def test_renderer_puts_the_object_where_the_camera_projects_it(self):
        try:
            import nvdiffrast.torch  # noqa: F401
            import torch
            import trimesh
        except ImportError:
            self.skipTest('nvdiffrast only in /opt/venvs/foundation_pose')
        from joint_fit import Renderer
        K = {'fx': 400., 'fy': 400., 'cx': 160., 'cy': 120.}
        box = trimesh.creation.box(extents=(.2, .2, .2))
        faces = torch.as_tensor(np.asarray(box.faces), dtype=torch.int32, device='cuda')
        for c in ([.5, .3, 3.], [-.4, -.2, 2.5]):
            X = torch.as_tensor(np.asarray(box.vertices) + c, dtype=torch.float32, device='cuda')[None]
            sil = Renderer(K, 240, 320, faces)(X)[0].cpu().numpy() > .5
            ys, xs = np.nonzero(sil)
            u, v = K['fx'] * c[0] / c[2] + K['cx'], K['fy'] * c[1] / c[2] + K['cy']
            self.assertAlmostEqual(xs.mean(), u - .5, delta=1.5)
            self.assertAlmostEqual(ys.mean(), v - .5, delta=1.5)
        # straight ahead only the front face (z = 2.9) bounds the silhouette
        X = torch.as_tensor(np.asarray(box.vertices) + [0, 0, 3.], dtype=torch.float32, device='cuda')[None]
        sil = Renderer(K, 240, 320, faces)(X)[0].cpu().numpy() > .5
        xs = np.nonzero(sil)[1]
        self.assertAlmostEqual(xs.max() - xs.min() + 1, 400 * .2 / 2.9, delta=1.5)


if __name__ == '__main__':
    unittest.main()
