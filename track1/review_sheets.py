#!/usr/bin/env python3
"""Human-review contact sheets for a packed episodes dir (camera view, body + object on the video).

  first   one panel per episode: the FIRST scored frame — the scorer fits its single alignment on
          our body here, so a wrong torso orientation rotates the whole episode (CD-H and CD-O)
  frames  for chosen episodes, N evenly spaced scored frames with body (blue), hands (red) and
          object (orange), plus the per-frame hand-object gap in the title

    /opt/venvs/sam3d_body/bin/python track1/review_sheets.py first --episodes_dir experiments/phys_v4/episodes \
        --out experiments/review/first_frames.png
    /opt/venvs/sam3d_body/bin/python track1/review_sheets.py frames --episodes_dir experiments/phys_v4/episodes \
        --episodes 0 2 18 19 26 --n 8 --out experiments/review/contact_failures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402


def project(x, K):
    return K['fx'] * x[:, 0] / x[:, 2] + K['cx'], K['fy'] * x[:, 1] / x[:, 2] + K['cy']


class Scene:
    def __init__(self, episodes_dir: Path, dev):
        import torch
        sys.path.insert(0, str(paths.KIT_TRACK1))
        from mesh_to_mhr_params import MHR
        from qa_episode import hand_vertex_ids
        self.torch, self.dir, self.dev = torch, episodes_dir, dev
        self.mhr = MHR(str(paths.DATA / 'weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt'), dev, 64, 'float32')
        self.hand_ids = hand_vertex_ids
        self.spans = json.loads((paths.DATA / 'submission/scored_spans.json').read_text())

    def load(self, e: int, frames):
        import trimesh
        torch = self.torch
        stem = f'episode_{e:06d}'
        d = np.load(self.dir / f'{stem}.npz')
        z = torch.as_tensor(np.concatenate([d['scales'], d['shape']])[None], dtype=torch.float64, device=self.dev)
        pose = torch.as_tensor(d['pose'][frames], dtype=torch.float64, device=self.dev)
        v = (self.mhr.run(pose, z)[0] / 1000).cpu().numpy()
        hid = self.hand_ids(self.mhr, pose[:1], z)
        mesh = trimesh.load(self.dir / f'{stem}_object.obj', force='mesh', process=False)
        P = trimesh.sample.sample_surface(mesh, 1500, seed=0)[0] * float(d['object_scale'])
        objs = [P @ d['object_rotation'][f].T + d['object_translation'][f] for f in frames]
        return v, hid, objs

    def draw(self, ax, e, f, v, hid, obj, title=''):
        from PIL import Image
        from scipy.spatial import cKDTree
        W = paths.OUT / f'episode_{e:06d}'
        K = json.loads((W / 'intrinsics_stable.json').read_text())
        off = json.loads((W / 'track1_meta.json').read_text())['frame_offset']
        ax.imshow(Image.open(W / 'frames' / f'{f - off:06d}.png'))
        ax.scatter(*project(v[::25], K), s=.2, c='deepskyblue', alpha=.5)
        ax.scatter(*project(v[hid][::6], K), s=.4, c='red', alpha=.7)
        ax.scatter(*project(obj[::2], K), s=.4, c='orange', alpha=.6)
        gap = cKDTree(obj).query(v[hid])[0].min() * 100
        ax.set_title(f'{title}ep {e} frame {f}  hand-object {gap:.0f} cm', fontsize=7)
        ax.axis('off')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=('first', 'frames'))
    ap.add_argument('--episodes_dir', type=Path, required=True)
    ap.add_argument('--episodes', type=int, nargs='*', default=list(range(30)))
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    sc = Scene(args.episodes_dir, 'cuda' if torch.cuda.is_available() else 'cpu')
    if args.mode == 'first':
        cols = 6
        rows = int(np.ceil(len(args.episodes) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis('off')
        for ax, e in zip(axes.ravel(), args.episodes):
            f = int(sc.spans[str(e)]['frames'][0])
            v, hid, objs = sc.load(e, [f])
            sc.draw(ax, e, f, v[0], hid, objs[0], 'FIRST scored: ')
        fig.suptitle(f'First scored frame per episode (the scorer aligns on our body here) — {args.episodes_dir}', fontsize=11)
        fig.tight_layout()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=70)
        print('wrote', args.out)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        for e in args.episodes:
            fr = np.asarray(sc.spans[str(e)]['frames'])
            pick = fr[np.linspace(0, len(fr) - 1, args.n).round().astype(int)]
            v, hid, objs = sc.load(e, pick)
            cols = 4
            fig, axes = plt.subplots(int(np.ceil(args.n / cols)), cols, figsize=(5 * cols, 4 * int(np.ceil(args.n / cols))), squeeze=False)
            for ax in axes.ravel():
                ax.axis('off')
            for k, (ax, f) in enumerate(zip(axes.ravel(), pick)):
                sc.draw(ax, e, int(f), v[k], hid, objs[k])
            fig.suptitle(f'episode {e}: scored (= contact) frames — body blue, hands red, object orange', fontsize=11)
            fig.tight_layout()
            path = args.out / f'episode_{e:06d}.png'
            fig.savefig(path, dpi=70)
            plt.close(fig)
            print('wrote', path)


if __name__ == '__main__':
    main()
