#!/usr/bin/env python3
"""Measure each object mesh's rotational symmetry (read-only; writes --out, default symmetry.json here).

For each principal axis through the surface centroid, rotate the surface samples by a set of
angles and measure the mean nearest-neighbour distance back to the unrotated surface, as a
fraction of the object size. Resampling noise (err at 0 deg) is the floor.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get('V2D_DATA', '/local-data/ivan/v2d_full')) / 'experiments/full_v2/pen_hands/merged/episodes'
ANGLES = [0, 30, 60, 90, 120, 150, 180]


def symmetry_errors(mesh: trimesh.Trimesh, n: int = 4000, n_ref: int = 30000):
    rng = np.random.default_rng(0)
    ref = trimesh.sample.sample_surface_even(mesh, n_ref, seed=1)[0]
    pts = trimesh.sample.sample_surface_even(mesh, n, seed=2)[0]
    centre = ref.mean(0)
    _, vec = np.linalg.eigh(np.cov((ref - centre).T))
    size = float(np.max(mesh.extents))
    tree = cKDTree(ref)
    rows = []
    for k in range(3):
        axis = vec[:, k] / np.linalg.norm(vec[:, k])
        errs = {}
        for ang in ANGLES:
            R = Rotation.from_rotvec(np.radians(ang) * axis).as_matrix()
            moved = (pts - centre) @ R.T + centre
            errs[ang] = float(tree.query(moved)[0].mean() / size)
        rows.append(dict(axis=axis.tolist(), errors=errs))
    del rng
    return centre, size, rows


def classify(rows, tol):
    """continuous: every tested angle within tol of the 0-deg floor; n-fold: 360/n angles within tol."""
    out = []
    for k, r in enumerate(rows):
        e = r['errors']; floor = e[0]
        ok = {a: e[a] - floor < tol for a in ANGLES[1:]}
        if all(ok.values()):
            kind, order = 'continuous', 0
        elif ok[90] and ok[180]:
            kind, order = 'discrete', 4
        elif ok[120]:
            kind, order = 'discrete', 3
        elif ok[180]:
            kind, order = 'discrete', 2
        else:
            continue
        out.append(dict(axis_index=k, axis=r['axis'], kind=kind, order=order))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    tol = 0.006  # 0.6 % of object size above the resampling floor
    result = {}
    for e in range(30):
        mesh = trimesh.load(SRC / f'episode_{e:06d}_object.obj', force='mesh', process=False)
        centre, size, rows = symmetry_errors(mesh)
        sym = classify(rows, tol)
        result[e] = dict(centre=centre.tolist(), size_m=size, axes=rows, symmetries=sym, tol=tol)
        txt = ' | '.join(' '.join(f"{r['errors'][a]*100:4.1f}" for a in ANGLES) for r in rows)
        print(f"{e:2d} size {size:.2f} m  err% per axis (0..180 deg): {txt}  -> "
              + (', '.join(f"ax{s['axis_index']}:{s['kind']}{s['order'] or ''}" for s in sym) or 'none'))
    json.dump(result, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
