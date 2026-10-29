"""hoop_conic: a circle of known radius must be recovered from its rendered image ellipse."""
import sys
import unittest
from pathlib import Path

import numpy as np

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
from hoop_conic import (circle_candidates, circle_points, min_rotation, project,  # noqa: E402
                        rim_distance, ring_frame)

K = {'fx': 950., 'fy': 950., 'cx': 768., 'cy': 576.}


def render_ellipse(c, n, R):
    import cv2
    img = np.zeros((1152, 1536), np.uint8)
    uv = project(circle_points(c, n, R, 720), K).round().astype(np.int32)
    cv2.polylines(img, [uv], True, 255, 5)
    pts = np.c_[np.nonzero(img)[1], np.nonzero(img)[0]].astype(np.float32)
    return cv2.fitEllipse(pts)


class HoopConic(unittest.TestCase):
    def test_one_candidate_recovers_the_true_circle(self):
        for c, n in (([-.6, .3, 3.4], [.3, -.2, -.93]), ([.2, .4, 3.0], [-.6, .1, -.8]), ([0, 0, 4.0], [0, .8, -.6])):
            c, n = np.asarray(c), np.asarray(n) / np.linalg.norm(n)
            cands = circle_candidates(render_ellipse(c, n, .38), K, .38)
            errs = [(np.linalg.norm(cc - c), np.degrees(np.arccos(min(1, abs(nn @ n))))) for cc, nn, _ in cands]
            best = min(errs, key=lambda x: x[1])
            self.assertLess(best[0], 0.03, errs)                    # centre within 3 cm at 3-4 m
            self.assertLess(best[1], 3.0, errs)                     # normal within 3 deg
            self.assertTrue(all(r < 2.0 for _, _, r in cands))      # both candidates explain the ellipse

    def test_rim_distance_and_minimal_rotation(self):
        c, n = np.zeros(3), np.array([0, 0, 1.])
        self.assertAlmostEqual(rim_distance(np.array([[.5, 0, 0]]), c, n, .38)[0], .12)
        self.assertAlmostEqual(rim_distance(np.array([[.38, 0, .1]]), c, n, .38)[0], .1)
        a, b = np.array([1., 0, 0]), np.array([0, 1., 0])
        np.testing.assert_allclose(min_rotation(a, b) @ a, b, atol=1e-12)
        np.testing.assert_allclose(min_rotation(a, -a) @ a, -a, atol=1e-12)

    def test_ring_frame_of_a_torus(self):
        import trimesh
        t = trimesh.creation.torus(.35, .015)
        c, axis, r = ring_frame(np.asarray(t.vertices))
        self.assertAlmostEqual(r, .35, delta=.01)
        self.assertGreater(abs(axis[2]), .99)


if __name__ == '__main__':
    unittest.main()
