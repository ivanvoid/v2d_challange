"""Small, scorer-based tests for PEN; run with /opt/venvs/sam3d_body/bin/python -m unittest discover -s track1/tests."""
import json
import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
import paths  # noqa: E402
from pen_fast import PenetrationMesh  # noqa: E402
from stage8_depen import cap_displacement, scored_local_indices, solve_translation  # noqa: E402
from stage8_scale import contact_gap_from_depth  # noqa: E402
from validate_pen_candidate import compare  # noqa: E402
from blend_pen_correction import blend  # noqa: E402
from build_episode import load_object_track  # noqa: E402


class PenetrationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A solid 4 cm cube, unlike open/degenerate meshes whose winding number is undefined.
        mesh = trimesh.creation.box(extents=(0.04, 0.04, 0.04))
        cls.v, cls.f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        cls.pm = PenetrationMesh(cls.v, cls.f, device="cpu")

    def test_exact_scorer_agrees_with_fast_proxy_on_inside_and_outside(self):
        from v2dlb.mhr_submission import _penetration_depth

        points = np.array([[0, 0, 0], [0.015, 0, 0], [0.04, 0, 0], [0, 0.03, 0]])
        expected = _penetration_depth(points, self.v, self.f)
        np.testing.assert_allclose(self.pm.depth(points), expected, atol=1e-6)
        np.testing.assert_allclose(expected, [0.02, 0.005, 0, 0], atol=1e-6)

    def test_face_orientation_padding_and_scene_scale(self):
        from v2dlb.mhr_submission import _penetration_depth

        points = np.array([[0.005, 0, 0], [0.025, 0, 0]])
        baseline = _penetration_depth(points, self.v, self.f)
        reversed_faces = self.f[:, ::-1]
        padding = np.array([[0, 0, 0]])  # packer pads faces to a fixed budget
        np.testing.assert_allclose(_penetration_depth(points, self.v, reversed_faces), baseline, atol=1e-6)
        np.testing.assert_allclose(_penetration_depth(points, self.v, np.vstack([self.f, padding])),
                                   baseline, atol=1e-6)
        scale = 3.0
        np.testing.assert_allclose(_penetration_depth(points, self.v, self.f) * scale,
                                   _penetration_depth(points * scale, self.v * scale, self.f), atol=1e-6)

    def test_zero_contact_gap_can_be_a_penetrating_hand(self):
        depth = np.array([[0.01, 0.0]])
        surface_distance = np.array([[0.01, 0.12]])
        self.assertEqual(float(contact_gap_from_depth(depth, surface_distance)[0]), 0.0)
        self.assertGreater(float(depth.mean() * 100), 0.0)

    def test_solver_finds_collision_reduction_without_exceeding_total_cap(self):
        # An interior hand point near the positive-X surface can be freed by a small shift.
        hands = np.array([[0.015, 0.0, 0.0], [0.1, 0.1, 0.1]])
        for base in (np.array([0., 0., 0.]), np.array([-0.008, 0., 0.])):
            delta, cost = solve_translation(self.pm, hands, np.eye(3), np.zeros(3), 1.0,
                                            base, 0.03, iters=8)
            self.assertLessEqual(np.linalg.norm(base + delta), 0.03000001)
            before = self.pm.depth(hands - base).sum()
            after = self.pm.depth(hands - (base + delta)).sum()
            self.assertLessEqual(after, before + 1e-8)
            self.assertAlmostEqual(cost, after, places=6)
        np.testing.assert_allclose(cap_displacement(np.array([0.03, 0.04, 0]), 0.02),
                                   [0.012, 0.016, 0], atol=1e-8)

    def test_exact_scored_ids_and_no_cross_gap_acceleration(self):
        idx = scored_local_indices(np.array([10, 11, 12, 17, 18]), 10, 10)
        np.testing.assert_array_equal(idx, [0, 1, 2, 7, 8])
        with self.assertRaises(ValueError):
            scored_local_indices(np.array([9]), 10, 10)

    def test_active_stage8_does_not_run_collision_correction(self):
        runner = (T1.parent / "run_stage.sh").read_text()
        stage8 = runner.split('stage8_final() {', 1)[1].split('MHR_MODEL=', 1)[0]
        self.assertNotIn('stage8_depen.py', stage8)
        self.assertIn('--object_poses "$W/poses_final"', stage8)

    def test_current_production_penetration_is_concentrated_in_foam(self):
        # Observational regression: verify the identified target against the immutable control.
        p = paths.DATA / "logs/eval/beta/prodnow/proxies.json"
        if not p.exists():
            self.skipTest(f"production control not available: {p}")
        rows = {r['episode']: r['pen_cm'] for r in json.loads(p.read_text())['episodes']}
        self.assertGreater(sum(rows[e] for e in (15, 16, 17)) / sum(rows.values()), 0.65)
        self.assertGreater(rows[16], 0.4)

    def test_candidate_gate_rejects_jitter_even_if_penetration_is_zero(self):
        original = {"episode": 16, "pen_cm": 0.4, "acc_o_self": 0.18,
                    "gap_cm_median": 0.0, "contact_frac": 0.6}
        with_jitter = {**original, "pen_cm": 0.0, "acc_o_self": 0.40}
        result = compare({"episodes": [original]}, {"episodes": [with_jitter]})
        self.assertFalse(result["pass"])
        self.assertFalse(result["episodes"][0]["checks"]["object_acc"])
        modest = {**original, "pen_cm": 0.05, "acc_o_self": 0.19,
                  "gap_cm_median": 1.0, "contact_frac": 0.55}
        self.assertTrue(compare({"episodes": [original]}, {"episodes": [modest]})["pass"])

    def test_temporal_smoothing_reduces_the_impulse_from_per_frame_repairs(self):
        shifts = np.zeros((31, 3))
        shifts[15, 0] = 0.03
        smoothed = gaussian_filter1d(shifts, 3, axis=0, mode="nearest")
        jitter = lambda arr: np.linalg.norm(np.diff(arr, n=2, axis=0), axis=1).mean()
        self.assertLess(jitter(smoothed), jitter(shifts) / 4)
        self.assertLessEqual(np.linalg.norm(smoothed, axis=1).max(), 0.03)

    def test_trial_packing_respects_video_frame_offset(self):
        root = paths.DATA / "out/episode_000016"
        submitted = paths.DATA / "submission/episodes/episode_000016.npz"
        if not root.is_dir() or not submitted.is_file():
            self.skipTest("episode 16 production artifacts unavailable")
        meta = json.loads((root / "track1_meta.json").read_text())
        original = np.load(submitted)
        assert meta["frame_offset"] > 0
        R, t, s, _have = load_object_track(root / "poses_final", len(original["pose"]), meta["frame_offset"])
        np.testing.assert_allclose(R, original["object_rotation"], atol=1e-6)
        np.testing.assert_allclose(t, original["object_translation"], atol=1e-6)
        self.assertAlmostEqual(s, float(original["object_scale"]))
        _wrong_R, wrong_t, _s, _have = load_object_track(root / "poses_final", len(original["pose"]), 0)
        self.assertGreater(float(np.max(np.abs(wrong_t - original["object_translation"]))), 0.1)

    def test_independent_hand_spec_identifies_the_actual_mhr_model(self):
        weight = paths.DATA / "weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt"
        spec_path = Path('/workspace/v2d_cari4d/lib/cari4d/lib_mhr/assets/mhr_hand_surface_spec.npz')
        if not weight.exists() or not spec_path.exists():
            self.skipTest("anatomical hand spec not available")
        spec = np.load(spec_path)
        with weight.open('rb') as handle:
            self.assertEqual(hashlib.file_digest(handle, 'sha256').hexdigest(), str(spec['mhr_model_sha256']))
        self.assertEqual(spec['vertex_indices'].shape, (2, 2318))
        self.assertEqual(spec['sample_local_indices'].shape, (2, 256))

    def test_isolated_full_candidate_only_changes_foam_object_translation(self):
        candidate = paths.DATA / 'experiments/penfix_smooth/episodes'
        baseline = paths.DATA / 'submission/episodes'
        if not (candidate / 'episode_000016.npz').exists():
            self.skipTest('the isolated 30-episode candidate is not assembled')
        for e in range(30):
            name = f'episode_{e:06d}'
            a = np.load(baseline / f'{name}.npz')
            b = np.load(candidate / f'{name}.npz')
            self.assertEqual(set(a.files), set(b.files))
            for key in a.files:
                if key == 'object_translation' and e in (15, 16, 17):
                    self.assertGreater(float(np.max(np.abs(a[key] - b[key]))), 0.0)
                else:
                    np.testing.assert_array_equal(a[key], b[key], err_msg=f'{name} {key} must remain unchanged')
            # Mesh is unmodified: PEN improvement is not from damaging/replacing the object.
            self.assertEqual((baseline / f'{name}_object.obj').read_bytes(),
                             (candidate / f'{name}_object.obj').read_bytes())

    def test_blending_only_moves_object_translation(self):
        base = {"translation": [0, 0, 3], "rotation": [1, 0, 0, 0], "scale": [1, 1, 1]}
        moved = {**base, "translation": [0.02, 0, 3]}
        result = blend(base, moved, 0.25)
        np.testing.assert_allclose(result["translation"], [0.005, 0, 3])
        self.assertEqual(result["rotation"], base["rotation"])
        self.assertEqual(result["scale"], base["scale"])
        with self.assertRaises(ValueError):
            blend(base, {**moved, "scale": [2, 2, 2]}, 0.25)


if __name__ == '__main__':
    unittest.main()
