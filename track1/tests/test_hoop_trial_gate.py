"""The hoop stage-6 gate cannot pass only by scoring its own repaired masks."""
import sys
import unittest
from pathlib import Path

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
from hoop_trial_gate import compare_states  # noqa: E402
from finish_hoop_trial import require_stage6_gate  # noqa: E402
import tempfile
import json
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import paths  # noqa: E402


class HoopTrialGate(unittest.TestCase):
    def test_stronger_edited_fit_requires_unchanged_frame_stability(self):
        baseline = [{'iou': .40, 'state': 'observed'} for _ in range(20)]
        scored = list(range(20))
        edited = set(range(10, 20))
        trial = [dict(x) for x in baseline]
        for i in edited:
            trial[i]['iou'] = .70
        self.assertTrue(compare_states(baseline, trial, scored, edited)['pass'])
        for i in range(10):
            trial[i]['iou'] = .30
        self.assertFalse(compare_states(baseline, trial, scored, edited)['pass'])
        for i in range(10):
            trial[i]['iou'] = .40
        trial[0]['state'] = 'lost'
        self.assertFalse(compare_states(baseline, trial, scored, edited)['pass'])
        with self.assertRaises(ValueError):
            compare_states(baseline, trial, list(edited), edited)

    def test_downstream_stages_refuse_rejected_or_missing_tracking_gate(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as folder:
            p = Path(folder)
            with self.assertRaises(ValueError):
                require_stage6_gate(p)
            (p / 'stage6_gate.json').write_text(json.dumps({'pass': False}))
            with self.assertRaises(ValueError):
                require_stage6_gate(p)
            (p / 'stage6_gate.json').write_text(json.dumps({'pass': True}))
            self.assertTrue(require_stage6_gate(p)['pass'])

    def test_mask_improvement_is_not_a_new_3d_pose_for_the_symmetric_ring(self):
        trial = paths.DATA / 'experiments/hoop_mask_v1'
        if not (trial / 'stage6.exit').exists():
            self.skipTest('hoop stage-6 retrack has not completed')
        mask_report = json.loads((trial / 'report.json').read_text())
        self.assertGreater(mask_report['existing_pose_iou_median_after'] -
                           mask_report['existing_pose_iou_median_before'], .3)
        mesh = trimesh.load(paths.DATA / 'submission/episodes/episode_000000_object.obj',
                            force='mesh', process=False)
        sample = trimesh.sample.sample_surface(mesh, 3000, seed=13)[0]
        for frame in (582, 600, 619):
            clouds, translations = [], []
            for folder in (paths.DATA / 'out/episode_000000', trial / 'out/episode_000000'):
                row = json.loads((folder / 'poses' / f'{frame:06d}.json').read_text())
                w, x, y, z = row['rotation']
                rot = Rotation.from_quat([x, y, z, w]).as_matrix()
                t = np.asarray(row['translation'])
                translations.append(t)
                clouds.append(sample @ rot.T + t)
            self.assertLess(float(np.linalg.norm(translations[0]-translations[1])), .01)
            a, b = clouds
            symmetric_distance_m = (cKDTree(a).query(b)[0].mean() +
                                    cKDTree(b).query(a)[0].mean()) / 2
            self.assertLess(float(symmetric_distance_m), .01)


if __name__ == '__main__':
    unittest.main()
