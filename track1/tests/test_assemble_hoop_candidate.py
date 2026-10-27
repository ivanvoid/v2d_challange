"""The hoop candidate cannot silently replace the human, mesh, or physical size."""
import sys
import unittest
from pathlib import Path

import numpy as np

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
from assemble_hoop_candidate import validate_episode_delta  # noqa: E402


class AssembleHoopCandidate(unittest.TestCase):
    def test_only_object_pose_changes_at_a_plausible_shared_diameter(self):
        base = {'pose': np.zeros((2,136),np.float32), 'object_scale': np.array(.925,np.float32),
                'object_rotation': np.tile(np.eye(3,dtype=np.float32),(2,1,1)),
                'object_translation': np.array([[0,0,3.],[0,0,3.]],np.float32)}
        changed = {**base, 'object_translation': base['object_translation'] + [0.02,0,0]}
        validate_episode_delta(base,changed)
        with self.assertRaises(AssertionError):
            validate_episode_delta(base,{**changed, 'pose': np.ones((2,136),np.float32)})
        with self.assertRaises(ValueError):
            validate_episode_delta(base,{**changed,'object_scale':np.array(1.3,np.float32)})
        # Episode 1's mesh differs from episode 0's: use its ACTUAL mesh extent.
        validate_episode_delta(base,{**changed,'object_scale':np.array(1.1,np.float32)},
                               mesh_max_extent_m=.692)


if __name__ == '__main__':
    unittest.main()
