#!/usr/bin/env python3
"""Independent PEN diagnostic on anatomical hand samples from the same MHR model.

This is NOT the hidden official hand-vertex index set. Use it alongside eval_local.py.
Input must be NPZ files extracted from the final packed Parquet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

import paths
from pen_fast import PenetrationMesh

sys.path.insert(0, str(paths.KIT_TRACK1))
from mesh_to_mhr_params import MHR  # noqa: E402 (run with sam3d_body venv)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extracted", type=Path, required=True)
    ap.add_argument("--spec", type=Path, default=Path("/workspace/v2d_cari4d/lib/cari4d/lib_mhr/assets/mhr_hand_surface_spec.npz"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    weight = paths.DATA / "weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt"
    spec = np.load(args.spec)
    with weight.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != str(spec["mhr_model_sha256"]):
        raise SystemExit("anatomical spec was built for a different MHR model")
    hand_ids = np.concatenate([spec["vertex_indices"][i, spec["sample_local_indices"][i]] for i in range(2)])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHR(str(weight), dev, 32, "float32")
    rows = []
    for path in sorted(args.extracted.glob("episode_*.npz")):
        d = np.load(path)
        pose = torch.as_tensor(d["pose"], dtype=torch.float64, device=dev)
        identity = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device=dev)
        hands = (mhr.run(pose, identity)[0][:, hand_ids] / 1000).cpu().numpy()
        pm = PenetrationMesh(d["mesh_v"], d["mesh_f"], device=dev)
        vals = []
        for i in range(len(hands)):
            loc = ((hands[i] - d["t"][i]) @ d["R"][i]) / float(d["s"])
            vals.append(float(pm.depth(loc).mean() * float(d["s"]) * 100))
        row = {"episode": int(path.stem.split("_")[1]), "pen_cm": float(np.mean(vals)),
               "frames_penetrating": sum(v > 0 for v in vals), "frames": len(vals)}
        rows.append(row)
        print(row, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"mhr_model_sha256": digest, "hand_points": len(hand_ids),
                                    "official_hand_indices": False, "episodes": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
