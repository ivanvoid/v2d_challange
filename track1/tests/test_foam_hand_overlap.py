"""Read-only diagnostic: deep palm overlap limits finger-only PEN correction on episode 16."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
import paths  # noqa: E402
from pen_fast import PenetrationMesh  # noqa: E402
from qa_episode import MHR  # noqa: E402


class FoamHandOverlap(unittest.TestCase):
    def test_frame_237_left_hand_is_embedded_beyond_finger_only_reach(self):
        extracted = paths.DATA / 'logs/eval/beta/penfix_smooth_full/episode_000016.npz'
        spec_path = Path('/workspace/v2d_cari4d/lib/cari4d/lib_mhr/assets/mhr_hand_surface_spec.npz')
        if not extracted.exists() or not spec_path.exists():
            self.skipTest('episode-16 packed candidate or anatomical hand spec unavailable')
        d, spec = np.load(extracted), np.load(spec_path)
        k = int(np.flatnonzero(d['frames'] == 237)[0])
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        mhr = MHR(str(paths.DATA / 'weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt'),
                  dev, 1, 'float32')
        pose = torch.as_tensor(d['pose'][k:k+1], dtype=torch.float64, device=dev)
        z = torch.as_tensor(np.concatenate([d['scales'], d['shape']])[None], dtype=torch.float64, device=dev)
        v = (mhr.run(pose, z)[0][0] / 1000).cpu().numpy()
        bumped = pose.clone()
        bumped[:, 68:122] += 0.3
        v_bumped = (mhr.run(bumped, z)[0][0] / 1000).cpu().numpy()
        ids = spec['vertex_indices'][0, spec['sample_local_indices'][0]]
        pm = PenetrationMesh(d['mesh_v'], d['mesh_f'], device=dev)
        points = v[ids]
        local = ((points - d['t'][k]) @ d['R'][k]) / float(d['s'])
        depth_cm = pm.depth(local) * float(d['s']) * 100
        movement_cm = np.linalg.norm(v_bumped[ids] - points, axis=1) * 100
        self.assertGreater(int((depth_cm > 0).sum()), 200)  # 239 / 256 observed
        self.assertGreater(float(depth_cm.mean()), 3.)
        self.assertGreater(int(((depth_cm > 0) & (movement_cm <= 0.2)).sum()), 40)


if __name__ == '__main__':
    unittest.main()
