#!/usr/bin/env python3
"""Stage 5c — test each candidate mesh on frames it was NOT built from, keep the best.

For each candidate (mesh/cand_<frame>/mesh_scaled.obj), FoundationPose registers the mesh
independently on --n_check CLEAR frames (stage-5a eligible: stage-4 status 'ok', not occluded, not at
the border) spread over the contact span (plus its own frame), using that frame's SAM2 mask and MoGe
depth. On each frame:
  iou        rendered mesh silhouette vs the SAM2 mask
  coverage   share of the SAM2 mask covered by the rendered mesh: a mesh missing a part (a pan without
             its handle) leaves that part uncovered, which whole-outline IoU barely notices
  depth_cm   median |rendered depth - MoGe depth| where both exist
A mesh with the right shape and size fits every frame; a wrong one only fits its own.
Chosen = highest (median IoU + median coverage) / 2 over the check frames (its own frame excluded).

Writes mesh/cand_<frame>/check.json (poses as 4x4 object-to-camera matrices + scores) and
mesh/choice.json. It is cheap (no SAM3D): changing this scoring reruns only 5c and 5e, not generation.
The final mesh (out/episode_XXXXXX/mesh_scaled.obj) is written by stage 5e, across the object's videos.

Run with the foundation_pose venv:
    /opt/venvs/foundation_pose/bin/python track1/stage5_check.py --episode 28 --out_root out \
        --weights weights --debug_log ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logutil import log, to_debug  # noqa: E402

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask  # noqa: E402
from v2d.common.datatypes import Image as V2dImage  # noqa: E402
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker, nvdiffrast_render  # noqa: E402
from v2d.mesh.lib.mesh import Mesh  # noqa: E402
from PIL import Image  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--debug_log", required=True)
    ap.add_argument("--n_check", type=int, default=6)
    args = ap.parse_args()

    ep = f"episode_{args.episode:06d}"
    W = os.path.join(os.path.abspath(args.out_root), ep)
    meta = json.load(open(os.path.join(W, "track1_meta.json")))
    off, first, last = meta["frame_offset"], meta["scored_first"], meta["scored_last"]
    intr = CameraIntrinsics.load(os.path.join(W, "intrinsics_stable.json"))
    K = intr.to_matrix()
    cands = json.load(open(os.path.join(W, "mesh", "candidates.json")))["chosen"]
    lo, hi = first - off, last - off
    cj = json.load(open(os.path.join(W, "mesh", "candidates.json")))
    clear = sorted(set(cj.get("eligible_frames", [])) - set(cands))
    if len(clear) >= args.n_check:
        check = [clear[int(round(i))] for i in np.linspace(0, len(clear) - 1, args.n_check)]
    else:
        check = sorted({int(round(lo + (hi - lo) * (i + 0.5) / args.n_check)) for i in range(args.n_check)})

    results = []
    for c in cands:
        C = os.path.join(W, "mesh", f"cand_{c:06d}")
        mesh_path = os.path.join(C, "mesh_scaled.obj")
        if not os.path.exists(mesh_path):
            log(f"  candidate video frame {c + off}: no mesh, skipped")
            continue
        tm = trimesh.load(mesh_path, force="mesh", process=False)
        extent = float(np.ptp(np.asarray(tm.vertices), axis=0).max())
        textured = type(tm.visual).__name__ == "TextureVisuals"
        per = []
        with to_debug(args.debug_log, f"check candidate {c + off}"):
            tracker = FoundationPoseTracker(Mesh.load(mesh_path), os.path.join(os.path.abspath(args.weights),
                                                                              "foundation_pose"))
            for f in sorted(set(check + [c])):
                mpath = os.path.join(W, "masks", "1", f"{f:06d}.png")
                mask = Mask.load(mpath)
                if not mask.mask.astype(bool).any():
                    continue
                rgb = V2dImage(data=np.asarray(Image.open(os.path.join(W, "frames", f"{f:06d}.png")).convert("RGB")))
                depth = DepthImage.load(os.path.join(W, "depth", f"{f:06d}.png"))
                pose = tracker.register(rgb, depth, mask, intr, iteration=10)
                M = pose.to_matrix()
                # register() returns the pose of the ORIGINAL mesh, but mesh_tensors hold the CENTRED mesh:
                # render with pose @ inv(tf_to_center) (the wrapper's _mask_iou skips this and is offset).
                tfc = tracker._est.get_tf_to_centered_mesh().cpu().numpy()
                Mc = M @ np.linalg.inv(tfc)
                with torch.no_grad():
                    _, rd, _ = nvdiffrast_render(K, intr.height, intr.width,
                                                 torch.as_tensor(Mc[None], device="cuda", dtype=torch.float),
                                                 glctx=tracker._glctx, mesh_tensors=tracker._est.mesh_tensors,
                                                 get_normal=False)
                rd = rd[0].cpu().numpy()
                sil, obs = rd > 0.001, mask.mask.astype(bool)
                iou = float((sil & obs).sum() / max((sil | obs).sum(), 1))
                cov = float((sil & obs).sum() / max(obs.sum(), 1))
                both = sil & obs & (depth.depth > 0)
                dcm = float(np.median(np.abs(rd[both] - depth.depth[both])) * 100) if both.any() else None
                per.append({"frame": f, "video_frame": f + off, "own": f == c, "iou": iou, "coverage": cov, "depth_cm": dcm,
                            "pose": M.tolist()})
        others = [p for p in per if not p["own"]]
        own = next((p for p in per if p["own"]), None)
        r = {"candidate": c, "video_frame": c + off, "extent_m": extent, "textured": textured,
             "scale": json.load(open(os.path.join(C, "scale.json")))["scale"],
             "own_iou": own["iou"] if own else None,
             "median_iou": float(np.median([p["iou"] for p in others])) if others else 0.0,
             "median_coverage": float(np.median([p["coverage"] for p in others])) if others else 0.0,
             "min_iou": float(np.min([p["iou"] for p in others])) if others else 0.0,
             "median_depth_cm": float(np.median([p["depth_cm"] for p in others if p["depth_cm"] is not None]))
             if others else None, "frames": per}
        json.dump(r, open(os.path.join(C, "check.json"), "w"), indent=1)
        results.append(r)
        log(f"  candidate video frame {c + off}: {'textured' if textured else 'NO TEXTURE'}, "
            f"extent {extent:.2f} m, scale {r['scale']:.3f}, own-frame IoU "
            f"{r['own_iou']:.2f}, other frames IoU median {r['median_iou']:.2f} (min {r['min_iou']:.2f}), coverage "
            f"{r['median_coverage']:.2f}, "
            f"depth error {r['median_depth_cm']:.1f} cm")
    if not results:
        raise SystemExit(f"{ep}: no candidate mesh to check")
    for r in results:
        r["score"] = (r["median_iou"] + r["median_coverage"]) / 2
    best = max(results, key=lambda r: r["score"])
    json.dump({"chosen": best["candidate"], "chosen_video_frame": best["video_frame"],
               "candidates": [{k: v for k, v in r.items() if k != "frames"} for r in results],
               "check_frames": check}, open(os.path.join(W, "mesh", "choice.json"), "w"), indent=1)
    log(f"{ep}: stage 5c OK — best of this video's candidates: the mesh from video frame {best['video_frame']} "
        f"(cross-frame IoU {best['median_iou']:.2f}, coverage {best['median_coverage']:.2f}, extent {best['extent_m']:.2f} m)")


if __name__ == "__main__":
    main()
