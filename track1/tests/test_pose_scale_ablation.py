"""Contracts and falsifiable assumptions for the isolated stage-7/8 experiments."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
import paths  # noqa: E402
from ablate_pose_scale import prepare_episode  # noqa: E402
from eval_pose_ablation import matched_summary  # noqa: E402
from stage7_smooth import smooth_series  # noqa: E402
from stage8_scale import candidate_scales  # noqa: E402


class PoseScaleAblation(unittest.TestCase):
    def test_scale_boundary_excludes_a_better_logged_candidate(self):
        p = paths.DATA / 'out_beta/ripple_stool/episode_000025/track/scale_match.json'
        if not p.exists():
            self.skipTest('VDA stool diagnostic unavailable')
        sm = json.loads(p.read_text())
        self.assertAlmostEqual(sm['best']['k'], 0.8)
        global_candidate = sm['legacy_variants']['global']
        self.assertLess(global_candidate['k'], sm['params']['k_min'])
        self.assertLess(global_candidate['score'], sm['best']['score'])
        self.assertFalse(any(abs(k - global_candidate['k']) < 0.01 for k in candidate_scales(0.8, 1.25, 0.025)))
        self.assertTrue(any(abs(k - global_candidate['k']) < 0.01 for k in candidate_scales(0.65, 1.25, 0.025)))

    def test_scale_candidates_validate_domain(self):
        self.assertAlmostEqual(candidate_scales(0.8, 1.25, 0.025)[0], 0.8)
        for values in ((0, 1, .1), (1.1, 0.8, .1), (.8, 1.2, 0)):
            with self.assertRaises(ValueError):
                candidate_scales(*values)

    def test_stronger_smoothing_reduces_self_acceleration_not_reference_error(self):
        t = np.arange(60, dtype=float)
        x = np.stack((t * 0.01, np.zeros_like(t), np.zeros_like(t)), axis=1)
        x[30, 0] += 0.05
        weak = smooth_series(x, np.ones(len(x)), 15.)
        strong = smooth_series(x, np.ones(len(x)), 150.)
        acc = lambda a: np.abs(np.diff(a[:, 0], n=2)).mean()
        self.assertLess(acc(strong), acc(weak))
        self.assertGreater(float(np.max(np.abs(strong-x))), 0)  # may lag genuine motion

    def test_isolated_trial_has_private_output_directories(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as tmp:
            base = Path(tmp)
            src = base / 'source'
            (src / 'body').mkdir(parents=True)
            (src / 'track').mkdir()
            for name in ('body/mhr_fit.npz', 'mesh_scaled.obj', 'intrinsics_stable.json',
                         'track1_meta.json', 'track/states.json'):
                (src / name).touch()
            for name in ('depth', 'poses', 'poses_smoothed'):
                (src / name).mkdir()
            scaled = base / 'scale'
            prepare_episode(src, scaled, smooth=False)
            self.assertTrue((scaled / 'poses_smoothed').is_symlink())
            self.assertFalse((scaled / 'track').is_symlink())
            self.assertFalse((scaled / 'poses_final').exists())
            tracked = base / 'smooth'
            prepare_episode(src, tracked, smooth=True)
            self.assertTrue((tracked / 'poses').is_symlink())
            self.assertTrue((tracked / 'track/states.json').is_symlink())
            self.assertFalse((tracked / 'track').is_symlink())
            self.assertFalse((tracked / 'poses_smoothed').is_symlink())
            self.assertFalse((tracked / 'poses_final').exists())

    def test_summary_is_matched_to_candidate_episodes_only(self):
        row = lambda e, p: {'episode': e, 'object': 'test', **{k: p for k in
            ('pen_cm', 'gap_cm_median', 'contact_frac', 'acc_h_self', 'acc_o_self',
             'object_size_m', 'body_minus_object_z_cm')}}
        control = {'episodes': [row(0, 100.), row(24, 2.), row(25, 4.)]}
        candidate = {'episodes': [row(24, 1.), row(25, 3.)]}
        result = matched_summary(control, candidate)
        self.assertEqual(result['episodes'], [24, 25])
        self.assertEqual(result['matched_mean']['control']['pen_cm'], 3.)
        self.assertEqual(result['matched_mean']['candidate']['pen_cm'], 2.)

    def test_scale_trial_preserves_all_non_object_pose_arrays(self):
        candidate = paths.DATA / 'experiments/scale_rescue_vda_stool/episodes'
        source = paths.DATA / 'submission_beta/ripple_stool/episodes'
        if not (candidate / 'episode_000026.npz').exists():
            self.skipTest('VDA scale trial not yet complete')
        for e in (24, 25, 26):
            stem = f'episode_{e:06d}'
            a, b = (np.load(folder / f'{stem}.npz') for folder in (source, candidate))
            for key in a.files:
                if key not in ('object_translation', 'object_scale'):
                    np.testing.assert_array_equal(a[key], b[key], err_msg=f'{stem} {key} changed')
            self.assertEqual((source / f'{stem}_object.obj').read_bytes(),
                             (candidate / f'{stem}_object.obj').read_bytes())


if __name__ == '__main__':
    unittest.main()
