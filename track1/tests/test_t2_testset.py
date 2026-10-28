"""Test-set helpers: the registration and metrics must be right before we trust any number."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
from t2_testset import (accel, chamfer_sum, manipulation_frames, quat_wxyz_to_matrix,  # noqa: E402
                        reference_leaks, register, rotvec_to_matrix, umeyama)


class T2TestSet(unittest.TestCase):
    def test_quaternion_is_w_first_like_the_dataset_readme(self):
        r = Rotation.from_euler('xyz', [10, -40, 70], degrees=True)
        x, y, z, w = r.as_quat()
        np.testing.assert_allclose(quat_wxyz_to_matrix(np.array([w, x, y, z])), r.as_matrix(), atol=1e-12)
        v = np.array([[.3, -.2, 1.1], [0, 0, 0]])
        np.testing.assert_allclose(rotvec_to_matrix(v), Rotation.from_rotvec(v).as_matrix(), atol=1e-12)

    def test_umeyama_recovers_a_similarity(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(50, 3))
        R = Rotation.from_rotvec([.2, 1.0, -.3]).as_matrix()
        dst = (1.3 * (R @ src.T)).T + [1, 2, 3]
        s, R2, t = umeyama(src, dst)
        self.assertAlmostEqual(s, 1.3, 9)
        np.testing.assert_allclose(R2, R, atol=1e-9)
        np.testing.assert_allclose(t, [1, 2, 3], atol=1e-9)

    def test_registration_absorbs_a_root_offset_in_the_reference_body_frame(self):
        rng = np.random.default_rng(1)
        n = 200
        ref = np.cumsum(rng.normal(scale=.03, size=(n, 3)), 0)                  # wandering root
        ref_rot = Rotation.from_rotvec(np.c_[np.zeros(n), np.linspace(0, 3, n), np.zeros(n)]).as_matrix()
        o = np.array([.0, .08, -.05])                                            # skeleton root mismatch
        pelvis_world = ref + ref_rot @ o
        Rc = Rotation.from_rotvec([.1, -.4, .2]).as_matrix()
        ours = ((Rc.T @ (pelvis_world - [.5, 1, 2]).T) / 1.1).T                 # camera frame, wrong scale
        s, R, t, o2, res = register(ours, ref, ref_rot)
        self.assertLess(res.max(), 1e-6)
        # turning only about the vertical (y) leaves o_y indistinguishable from t: only x, z are identified
        np.testing.assert_allclose(o2[[0, 2]], o[[0, 2]], atol=1e-6)
        self.assertAlmostEqual(s, 1.1, 6)

    def test_manipulation_frames_cover_the_move_with_padding_and_visibility(self):
        n = 100
        t = np.zeros((n, 3)); t[40:60, 0] = np.linspace(0, .5, 20); t[60:, 0] = .5
        R = np.tile(np.eye(3), (n, 1, 1))
        vis = np.ones(n, bool); vis[50] = False
        f = manipulation_frames(t, R, vis, pad=5)
        # > 5 cm from rest from frame 42 (sustained 5-frame window centred at 44) until frame 57 (55)
        self.assertEqual((f[0], f[-1]), (39, 60))
        self.assertNotIn(50, f)
        noisy = np.random.default_rng(0).normal(scale=.015, size=(n, 3))    # capture noise at rest
        self.assertEqual(len(manipulation_frames(noisy, R, vis)), 0)

    def test_accel_uses_only_consecutive_triplets_and_chamfer_is_a_sum(self):
        frames = np.array([0, 1, 2, 3, 10, 11, 12])
        x = np.c_[np.array([0, 1, 4, 9, 0, 0, 0.]), np.zeros(7), np.zeros(7)]
        a = accel(x, frames)
        self.assertEqual(len(a), 3)                                              # (0,1,2) (1,2,3) (10,11,12)
        np.testing.assert_allclose(a[:, 0], [2, 2, 0])
        p = np.zeros((4, 3)); q = p + [0, 0, .1]
        self.assertAlmostEqual(chamfer_sum(p, q), .2)                             # 0.1 each way, summed


    def test_pipeline_root_must_not_hold_reference_data(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as d:
            root = Path(d)
            (root / 'challenge/track_1/videos').mkdir(parents=True)
            (root / 'challenge/track_1/videos/episode_000007.mp4').write_bytes(b'')
            (root / 'submission').mkdir()
            (root / 'submission/scored_spans.json').write_text('{}')
            self.assertEqual(reference_leaks(root), [])
            (root / 'out').mkdir()
            (root / 'out/generated.glb').write_bytes(b'')                       # our own mesh: fine
            self.assertEqual(reference_leaks(root), [])
            from t2_testset import T2
            (root / 'out/iron.glb').symlink_to(T2 / 'mesh/iron/iron.glb')          # reference mesh: leak
            (root / 'submission/x.parquet').write_bytes(b'')
            self.assertEqual(len(reference_leaks(root)), 2)

    def test_real_test_root_is_clean_if_built(self):
        root = Path('/local-data/ivan/v2d_t2')
        if not root.exists():
            self.skipTest('test root not built')
        self.assertEqual(reference_leaks(root), [])


if __name__ == '__main__':
    unittest.main()
