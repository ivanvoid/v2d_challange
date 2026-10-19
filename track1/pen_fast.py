"""GPU version of the PEN scorer's penetration depth (v2dlb.mhr_submission._penetration_depth).

Same definition: a point is inside when |generalised winding number| > 0.5 (Jacobson et al. 2013,
the kit's exact formula), depth = distance to the closest surface point, 0 outside, points outside
the mesh's bounding box skipped. Winding numbers run in torch float64 on the GPU; distances (only
for inside points) use trimesh's exact closest-point query. Validated against the kit function.
"""
from __future__ import annotations

import numpy as np
import torch
import trimesh


class PenetrationMesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, device: str | None = None):
        tri = vertices[faces]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        faces = faces[area > 0]
        self.lo, self.hi = vertices.min(0), vertices.max(0)
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.as_tensor(vertices[faces], dtype=torch.float64, device=self.dev)   # [F, 3, 3]
        self.v0, self.v1, self.v2 = t[:, 0], t[:, 1], t[:, 2]
        self.mesh = trimesh.Trimesh(vertices, faces, process=False)

    def winding(self, points: np.ndarray, chunk_elems: int = 8_000_000) -> np.ndarray:
        out = np.zeros(len(points))
        per = max(1, chunk_elems // max(1, len(self.v0)))
        q_all = torch.as_tensor(points, dtype=torch.float64, device=self.dev)
        with torch.no_grad():
            for s in range(0, len(points), per):
                q = q_all[s:s + per, None, :]
                a, b, c = self.v0[None] - q, self.v1[None] - q, self.v2[None] - q
                la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
                num = (a * torch.cross(b, c, dim=-1)).sum(-1)
                den = (la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb)
                out[s:s + per] = (torch.atan2(num, den).sum(1) / (2 * np.pi)).cpu().numpy()
        return out

    def depth(self, points: np.ndarray, return_closest: bool = False):
        """Depth (>= 0) of each mesh-frame point inside the mesh; optionally the closest surface points."""
        depth = np.zeros(len(points))
        closest = np.zeros_like(points)
        near = np.flatnonzero(((points >= self.lo) & (points <= self.hi)).all(axis=1))
        if near.size:
            inside = near[np.abs(self.winding(points[near])) > 0.5]
            if inside.size:
                c, d, _ = self.mesh.nearest.on_surface(points[inside])
                depth[inside] = d
                closest[inside] = c
        return (depth, closest) if return_closest else depth
