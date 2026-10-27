"""Tests for the user-confirmed stationary stool prior and independent image gate."""
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
from stationary_stool import (accept_stationary_pose, choose_constant_pose, mask_iou,
                              load_local_poses, rendered_mask, visible_indices)  # noqa: E402


class StationaryStool(unittest.TestCase):
    def test_distributed_training_anchors_and_independent_holdout(self):
        rows = [{'state': 'observed', 'occlusion': 0.02, 'iou': 0.8,
                 'area_frac': 1.0, 'quality': 0.5 + (i % 3) * .1} for i in range(90)]
        anchors, holdout = visible_indices(rows, n_bins=9)
        self.assertEqual(len(anchors), 9)
        self.assertTrue(set(anchors).isdisjoint(holdout))
        self.assertGreaterEqual(len(holdout), 70)
        self.assertLess(min(anchors), 10)
        self.assertGreater(max(anchors), 70)

    def test_multiframe_silhouette_rejects_misplaced_pose(self):
        mesh = trimesh.creation.box(extents=[.2, .2, .2])
        K = np.array([[800., 0, 320.], [0, 800., 240.], [0, 0, 1]])
        rotations = np.tile(np.eye(3), (60, 1, 1))
        translations = np.tile(np.array([0., 0., 3.]), (60, 1))
        translations[15] = [0.2, 0.0, 3.]
        truth = rendered_mask(mesh.vertices, mesh.faces, rotations[0], translations[0], 1., K, (120, 160))
        wrong = rendered_mask(mesh.vertices, mesh.faces, rotations[15], translations[15], 1., K, (120, 160))
        self.assertGreater(mask_iou(truth, truth), mask_iou(truth, wrong) + 0.3)
        anchors = [0, 15, 30, 45]
        candidate, scores = choose_constant_pose(rotations, translations, anchors,
                         {i: truth for i in anchors}, mesh.vertices, mesh.faces, 1., K, (120, 160))
        self.assertNotEqual(candidate, 15)
        self.assertGreater(scores[candidate], scores[15])

    def test_holdout_and_motion_caps_reject_a_wrong_static_solution(self):
        self.assertTrue(accept_stationary_pose(0.7, 0.67, 4., max_iou_drop=.05, max_allowed_shift_cm=8.))
        self.assertFalse(accept_stationary_pose(0.7, 0.55, 4., max_iou_drop=.05, max_allowed_shift_cm=8.))
        self.assertFalse(accept_stationary_pose(0.7, 0.71, 20., max_iou_drop=.05, max_allowed_shift_cm=8.))

    def test_stationary_prior_supported_by_visible_masks_not_by_hand_contact(self):
        for e in (24, 26):
            w = paths.DATA / 'out' / f'episode_{e:06d}'
            if not (w / 'track/states.json').exists():
                self.skipTest('stool production outputs unavailable')
            rows = json.loads((w / 'track/states.json').read_text())['frames']
            anchors, holdout = visible_indices(rows)
            centroids = []
            for f in anchors + holdout:
                mask = cv2.imread(str(w / 'masks/1' / f'{f:06d}.png'), cv2.IMREAD_GRAYSCALE)
                ys, xs = np.nonzero(mask)
                centroids.append([xs.mean(), ys.mean()])
            center = np.median(centroids, axis=0)
            self.assertLess(np.percentile(np.linalg.norm(np.array(centroids)-center, axis=1), 90), 12)
        d = np.load(paths.DATA / 'submission/episodes/episode_000025.npz')
        self.assertGreater(float(np.ptp(d['object_translation'], axis=0).max()), .5)

    def test_full_trimmed_clip_can_exceed_packed_scored_video_length(self):
        for e in (24, 26):
            w = paths.DATA / 'out' / f'episode_{e:06d}'
            meta = json.loads((w / 'track1_meta.json').read_text())
            with np.load(paths.DATA / 'submission/episodes' / f'episode_{e:06d}.npz') as packed:
                off, n = meta['frame_offset'], meta['n_frames']
                R, t = load_local_poses(w / 'poses_final', n)
                self.assertEqual(len(t), n)
                self.assertGreater(n + off, len(packed['object_translation']))
                stop = len(packed['object_translation']) - off
                np.testing.assert_allclose(R[:stop], packed['object_rotation'][off:], atol=1e-6)
                np.testing.assert_allclose(t[:stop], packed['object_translation'][off:], atol=1e-6)

    def test_isolated_static_result_only_changes_object_pose(self):
        trial = paths.DATA / 'experiments/stool_static_v1/episodes'
        if not (trial / 'episode_000026.npz').exists():
            self.skipTest('stationary-stool trial not yet finished')
        for e in (24, 26):
            stem = f'episode_{e:06d}'
            source = paths.DATA / 'submission/episodes'
            with np.load(source / f'{stem}.npz') as original, np.load(trial / f'{stem}.npz') as fixed:
                self.assertEqual(set(original.files), set(fixed.files))
                for key in original.files:
                    if key not in ('object_rotation', 'object_translation'):
                        np.testing.assert_array_equal(original[key], fixed[key], err_msg=f'{stem} {key} changed')
                self.assertEqual(float(np.ptp(fixed['object_translation'], axis=0).max()), 0.)
                self.assertEqual(float(np.ptp(fixed['object_rotation'], axis=0).max()), 0.)
            self.assertEqual((source / f'{stem}_object.obj').read_bytes(),
                             (trial / f'{stem}_object.obj').read_bytes())
        self.assertFalse((trial / 'episode_000025.npz').exists())

    def test_full_candidate_changes_only_stationary_stool_episodes(self):
        candidate = paths.DATA / 'experiments/stool_static_plus_smooth_t150/episodes'
        control = paths.DATA / 'experiments/smooth_t150_full/episodes'
        if not (candidate / 'episode_000029.npz').exists():
            self.skipTest('full stationary-stool candidate not yet assembled')
        for e in range(30):
            stem = f'episode_{e:06d}'
            with np.load(control / f'{stem}.npz') as original, np.load(candidate / f'{stem}.npz') as fixed:
                self.assertEqual(set(original.files), set(fixed.files))
                for key in original.files:
                    if e in (24, 26) and key in ('object_rotation', 'object_translation'):
                        self.assertTrue(np.any(original[key] != fixed[key]))
                    else:
                        np.testing.assert_array_equal(original[key], fixed[key],
                                                      err_msg=f'{stem} {key} changed unexpectedly')
            self.assertEqual((control / f'{stem}_object.obj').read_bytes(),
                             (candidate / f'{stem}_object.obj').read_bytes())

    def test_existing_camera_initialization_samples_across_whole_video(self):
        log = paths.DATA / 'logs/stages/01_prep/episode_000024.debug.log'
        if not log.exists():
            self.skipTest('episode-24 calibration log unavailable')
        samples = [line for line in log.read_text(encoding='utf-8').splitlines() if 'sampled frame ' in line]
        self.assertEqual(len(samples), 16)
        self.assertIn('sampled frame 0 ', samples[0])
        self.assertIn('sampled frame 261 ', samples[-1])


if __name__ == '__main__':
    unittest.main()
