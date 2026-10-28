"""The physical-plausibility assembler may only change object poses, and only on evidence."""
import sys
import unittest
from pathlib import Path

import numpy as np

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
from assemble_phys import (check_only_object_pose_changed, depen_accepted,  # noqa: E402
                           scored_rows, translation_deviation_cm)


def episode(n=6):
    return {'pose': np.zeros((n, 136), np.float32), 'scales': np.zeros(68, np.float32),
            'shape': np.zeros(45, np.float32), 'object_scale': np.array(1.05, np.float32),
            'object_rotation': np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
            'object_translation': np.zeros((n, 3), np.float32)}


class AssemblePhys(unittest.TestCase):
    def test_object_pose_may_change_but_body_scale_and_keys_may_not(self):
        base = episode()
        moved = {**base, 'object_translation': base['object_translation'] + .01}
        check_only_object_pose_changed(base, moved)
        for key, value in (('pose', np.ones((6, 136), np.float32)),
                           ('object_scale', np.array(1.10, np.float32))):
            with self.assertRaisesRegex(ValueError, key):
                check_only_object_pose_changed(base, {**base, key: value})
        with self.assertRaises(ValueError):
            check_only_object_pose_changed(base, {k: v for k, v in base.items() if k != 'shape'})
        bad = base['object_translation'].copy(); bad[2, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            check_only_object_pose_changed(base, {**base, 'object_translation': bad})

    def test_deviation_is_measured_on_scored_rows_only(self):
        base = episode(10)
        cand = {**base, 'object_translation': base['object_translation'].copy()}
        cand['object_translation'][9] = [1., 0, 0]            # 1 m jump on an unscored row
        rows = scored_rows({'3': {'frames': [2, 3, 4]}}, 3, 10)
        self.assertEqual(translation_deviation_cm(base, cand, rows), 0.0)
        with self.assertRaises(ValueError):
            scored_rows({'3': {'frames': [2, 12]}}, 3, 10)

    def test_depen_needs_fit_and_holdout_gain_without_more_jitter(self):
        good = {'pen_proxy_before_cm': .40, 'pen_proxy_after_cm': .30, 'pen_holdout_before_cm': .41,
                'pen_holdout_after_cm': .29, 'acc_o_before': .20, 'acc_o_after': .203}
        self.assertTrue(depen_accepted(good)[0])
        self.assertFalse(depen_accepted(None)[0])
        self.assertFalse(depen_accepted({**good, 'pen_holdout_after_cm': .41})[0])   # fit-only gain
        self.assertFalse(depen_accepted({**good, 'acc_o_after': .25})[0])            # jitter
        self.assertTrue(depen_accepted({**good, 'acc_o_before': .050, 'acc_o_after': .058})[0])  # ep 8-like
        self.assertFalse(depen_accepted({**good, 'pen_proxy_before_cm': 0., 'pen_proxy_after_cm': 0.,
                                         'pen_holdout_before_cm': 0., 'pen_holdout_after_cm': 0.})[0])


if __name__ == '__main__':
    unittest.main()
