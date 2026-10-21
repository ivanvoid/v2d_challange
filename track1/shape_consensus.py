#!/usr/bin/env python3
"""3D shape agreement between candidate meshes of one object (size-free).

Each mesh: surface samples, centred, scaled to unit RMS radius. Pairwise distance = symmetric mean
nearest-point distance after the best rigid alignment (PCA starts with all proper sign flips + the 24 cube
rotations, each refined by ICP). A candidate's "consensus" = median distance to the object's OTHER
candidates: independent reconstructions of the same object agree on its true shape; a blob, a ring made
from a shirt or a broken handle agrees with few of them.

    python shape_consensus.py mesh1.obj mesh2.obj ...      (prints the distance matrix and consensus)
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_points(path: str, n: int = 1500, seed: int = 0) -> np.ndarray:
    tm = trimesh.load(path, force="mesh", process=False)
    p, _ = trimesh.sample.sample_surface(tm, n, seed=seed)
    p = np.asarray(p, float)
    p -= p.mean(0)
    return p / np.sqrt((p ** 2).sum(1).mean())


def _rotations() -> list[np.ndarray]:
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, (j, s) in enumerate(zip(perm, signs)):
                R[i, j] = s
            if np.linalg.det(R) > 0:
                out.append(R)
    return out                                                    # the 24 proper cube rotations


ROTS = _rotations()


def _pca(p: np.ndarray) -> np.ndarray:
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    if np.linalg.det(vt) < 0:
        vt[2] *= -1
    return vt


def _chamfer(a: np.ndarray, b: np.ndarray, ta: cKDTree, tb: cKDTree) -> float:
    return float((tb.query(a)[0].mean() + ta.query(b)[0].mean()) / 2)


def _icp(src: np.ndarray, tree: cKDTree, dst: np.ndarray, R: np.ndarray, iters: int = 25) -> np.ndarray:
    for _ in range(iters):
        x = src @ R.T
        _, idx = tree.query(x)
        y = dst[idx]
        H = (src - src.mean(0)).T @ (y - y.mean(0))
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
    return R


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Size-free shape distance (unit RMS radius), minimised over rigid rotations."""
    ta, tb = cKDTree(a), cKDTree(b)
    Pa, Pb = _pca(a), _pca(b)
    starts = [Pb.T @ S @ Pa for S in (np.diag(s) for s in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]))]
    best = np.inf
    for R0 in starts + ROTS:
        R = _icp(a, tb, b, R0, iters=8)
        d = _chamfer(a @ R.T, b, cKDTree(a @ R.T), tb)
        best = min(best, d)
    return best


def consensus(points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = distance(points[i], points[j])
    med = np.array([np.median([D[i, j] for j in range(n) if j != i]) for i in range(n)])
    return D, med


if __name__ == "__main__":
    pts = [load_points(p) for p in sys.argv[1:]]
    D, med = consensus(pts)
    for p, m, row in zip(sys.argv[1:], med, D):
        print(f"{m:.3f}  {p}  " + " ".join(f"{x:.2f}" for x in row))
