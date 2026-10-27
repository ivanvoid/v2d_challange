"""Hoop hollow-mask regression checks; no SAM2 or tracking rerun."""
import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import trimesh

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
import paths  # noqa: E402
from hoop_mask_repair import fill_fraction, largest_hole_fraction, pose_at, remove_filled_interior  # noqa: E402
from stationary_stool import mask_iou, rendered_mask  # noqa: E402


class HoopMaskRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = paths.DATA / 'out/episode_000000'

    def mask(self, frame):
        p = self.w / 'masks/1' / f'{frame:06d}.png'
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        return m > 0

    def test_opening_has_a_hole_and_late_sam2_mask_is_filled(self):
        early, late = self.mask(0), self.mask(600)
        self.assertGreater(largest_hole_fraction(early), .5)
        self.assertLess(fill_fraction(early), .5)
        self.assertEqual(largest_hole_fraction(late), 0.)
        self.assertGreater(fill_fraction(late), .85)

    def test_repair_only_removes_pixels_and_preserves_known_ring(self):
        early, late = self.mask(0), self.mask(600)
        repaired = remove_filled_interior(late, 8)
        self.assertFalse(np.any(repaired & ~late))
        self.assertGreater(largest_hole_fraction(repaired), .3)
        self.assertLess(int(repaired.sum()), int(late.sum()) // 2)
        self.assertLess(fill_fraction(self.mask(37)), .72)  # partial arc, do not hallucinate ring
        self.assertLess(fill_fraction(self.mask(570)), .72) # valid late arc, keep it
        self.assertLess(fill_fraction(early), .72)          # opening ring, keep it

    def test_saved_ring_pose_consistency_recovers_at_late_frame(self):
        m = trimesh.load(paths.DATA / 'submission/episodes/episode_000000_object.obj',
                         force='mesh', process=False)
        intr = json.loads((self.w / 'intrinsics_stable.json').read_text())
        K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]])
        R, t, s = pose_at(self.w, 600)
        pred = rendered_mask(m.vertices, m.faces, R, t, s, K, (288, 384))
        down = lambda mask: cv2.resize(mask.astype(np.uint8), (384, 288),
                                       interpolation=cv2.INTER_NEAREST) > 0
        before = mask_iou(pred, down(self.mask(600)))
        after = mask_iou(pred, down(remove_filled_interior(self.mask(600), 8)))
        self.assertGreater(after, before + .20)


if __name__ == '__main__':
    unittest.main()
