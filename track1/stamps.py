#!/usr/bin/env python3
"""Stage stamps: re-run a stage exactly when its code, settings or inputs changed — and then from scratch.

After a stage succeeds for an episode, `write` stores out/<ep>/stamps/<stage>.json with
  code    sha1 of every file the stage runs: its entry scripts, the local modules they import
          (found by scanning `import x` / `from x import` lines, recursively) and fixed extras
          (kit modules, config files)
  params  settings that do not live in code (env SMOOTH_SIGMA, this episode's stage6_overrides entry)
  inputs  sha1 of the upstream data files the stage reads directly (track1_meta.json frame window
          only — the file also stores timings —, intrinsics, depth_scale.json, a manual seed)
  deps    the stamp ids of the stages it depends on, so any upstream change invalidates it
and `id` = sha1 of all of that.

  check   exit 0 if the stored id equals the id expected now and the outputs exist; else exit 1
          and print why (exit 2 if an upstream stage has no stamp: run that stage first)
  archive move this stage's outputs to out/<ep>/_archive/<stage>/ (only the latest archive is
          kept), so no "file exists -> skip" inside the stage can reuse stale results
  write   store the stamp (after the stage succeeded)
  adopt   store stamps for outputs that already exist and are known to be valid (migration)
  status  table of current / stale / missing stamps for episodes and stages

    python3 track1/stamps.py status --episodes 0-29
    python3 track1/stamps.py check --episode 6 --stage 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

T1 = Path(__file__).resolve().parent
OUT = paths.DATA / "out"
KIT_T1 = paths.KIT / "tools" / "track1"

# stage: entry scripts (track1/), extra files, direct dependencies, input files, outputs
STAGES = {
    "1": dict(code=["stage1_prep.py"], extra=[], deps=[],
              inputs=["META_WINDOW", "intrinsics_stable.json", "depth_scale.json", "floor.json"], outputs=[]),
    "2": dict(code=["stage2_track.py", "stage2_body.py", "fit_mhr.py", "stage2_depth.py"], extra=[KIT_T1 / "mesh_to_mhr_params.py"], deps=[],
              inputs=["META_WINDOW", "intrinsics_stable.json", "FRAMES"], outputs=["body"]),
    "3": dict(code=["stage3_select.py"], extra=[T1 / "object_priors.json"], deps=["1", "2"],
              inputs=["SEED"],
              outputs=["select", "object_track.json", "dino_detections.json", "dino_candidates", "near_frames"]),
    "4": dict(code=["stage4_prompts.py", "stage4_mask.py", "sam2_box_neg.py"], extra=[T1 / "object_priors.json"],
              deps=["1", "2", "3"], inputs=[],
              outputs=["masks", "masks_raw", "masks_fill", "masks_prob", "masks_pass", "masks_conf.json",
                       "masks_seed.json", "mask_prompts.json", "masks_repair.json", "masks_quality.json",
                       "sam2_prompts.json"]),
    "5": dict(code=["stage5_pick.py", "stage5_mesh.py", "stage5_scale.py", "stage5_check.py", "stage5_refine.py"],
              extra=[T1 / "object_priors.json"], deps=["1", "2", "4"], inputs=[],
              outputs=["mesh", "mesh_scaled.obj", "mesh_scaled_stage5c.obj", "mesh_scaled_stage5d.obj",
                       "material.mtl", "material_0.png"]),
    "5e": dict(code=["stage5_object.py", "stage5_validate.py"], extra=[], deps=["5"], inputs=["OBJECT_VIDEOS"], outputs=[]),
    "6": dict(code=["stage6_track.py"], extra=[], deps=["1", "2", "4", "5e"], inputs=["OVERRIDES"],
              outputs=["poses", "track/states.json"]),
    "7": dict(code=["stage7_smooth.py"], extra=[], deps=["6"], inputs=[],
              outputs=["poses_smoothed", "track/smooth.json"]),
    "8": dict(code=["stage8_scale.py", "build_episode.py", "qa_episode.py"],
              extra=[T1.parent / "render_results.py", KIT_T1 / "mesh_to_mhr_params.py"],
              deps=["1", "2", "7"], inputs=[],
              outputs=["poses_final", "track/scale_match.json", "SUB_EPISODE"]),
}
# outputs that must exist for a stamp to count (a stamp without its result is stale)
REQUIRED = {"1": ["depth_scale.json", "floor.json", "background.npz"], "2": ["body/mhr_fit.npz", "body/person_track.json", "body/depth_trend.json"], "3": ["object_track.json"],
            "4": ["masks_seed.json", "masks_quality.json"], "5": ["mesh/refine.json"], "5e": ["mesh/object_choice.json", "mesh/validate.json"],
            "6": ["track/states.json"], "7": ["track/smooth.json"], "8": ["track/scale_match.json"]}


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def sha1_file(p: Path) -> str | None:
    return sha1_bytes(p.read_bytes()) if p.exists() else None


def local_imports(script: Path, seen: set[Path]) -> None:
    if script in seen or not script.exists():
        return
    seen.add(script)
    if script.suffix != ".py":
        return
    for line in script.read_text(errors="replace").splitlines():
        m = re.match(r"\s*(?:from\s+([A-Za-z_]\w*)\s+import|import\s+([A-Za-z_]\w*))", line)
        if m:
            mod = T1 / f"{m.group(1) or m.group(2)}.py"
            local_imports(mod, seen)


def code_hashes(stage: str) -> dict[str, str]:
    files: set[Path] = set()
    for s in STAGES[stage]["code"]:
        local_imports(T1 / s, files)
    files |= {Path(p) for p in STAGES[stage]["extra"]}
    return {str(p.relative_to(T1.parent) if T1.parent in p.parents else p): sha1_file(p) for p in sorted(files)}


def episode_object(e: int) -> tuple[str, list[int]]:
    meta = [json.loads(l) for l in open(paths.META)]
    obj = next(m["object"] for m in meta if m["episode_index"] == e)
    return obj, sorted(m["episode_index"] for m in meta if m["object"] == obj)


def ep_dir(e: int) -> Path:
    return OUT / f"episode_{e:06d}"


def stamp_path(e: int, stage: str) -> Path:
    return ep_dir(e) / "stamps" / f"{stage}.json"


def read_stamp(e: int, stage: str) -> dict | None:
    p = stamp_path(e, stage)
    return json.load(open(p)) if p.exists() else None


def inputs_of(e: int, stage: str) -> dict:
    W = ep_dir(e)
    out = {}
    for name in STAGES[stage]["inputs"]:
        if name == "META_WINDOW":
            if not (W / "track1_meta.json").exists():
                out[name] = None
                continue
            m = json.load(open(W / "track1_meta.json"))
            out[name] = {k: m.get(k) for k in ("frame_offset", "n_frames", "scored_first", "scored_last", "margin")}
        elif name == "FRAMES":                     # the extracted frames: names, sizes, modification times
            fd = W / "frames"
            out[name] = sha1_bytes("".join(f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns};"
                                           for p in sorted(fd.glob("*.png"))).encode()) if fd.is_dir() else None
        elif name == "SEED":
            out[name] = sha1_file(paths.SEEDS / f"episode_{e:06d}.json")
        elif name == "OVERRIDES":
            o = json.load(open(T1 / "stage6_overrides.json")).get(str(e), {})
            out[name] = {k: v for k, v in o.items() if not k.startswith("_")}
        elif name == "OBJECT_VIDEOS":
            out[name] = episode_object(e)[1]
        else:
            out[name] = sha1_file(W / name)
    return out


def params_of(stage: str) -> dict:
    if stage == "2":
        return {"SMOOTH_SIGMA": os.environ.get("SMOOTH_SIGMA", "2.0"), "BODY_MODE": os.environ.get("BODY_MODE", "body"),
                "BODY_NO_MASK": os.environ.get("BODY_NO_MASK", "")}
    return {}


def dep_ids(e: int, stage: str) -> tuple[dict, list[str]]:
    """Ids of the upstream stamps; stage 5e depends on stage 5 of every video of the object."""
    ids, missing = {}, []
    for d in STAGES[stage]["deps"]:
        eps = episode_object(e)[1] if stage == "5e" else [e]
        for x in eps:
            s = read_stamp(x, d)
            key = d if len(eps) == 1 else f"{d}@{x}"
            if s is None:
                missing.append(key)
            else:
                ids[key] = s["id"]
    return ids, missing


def expected(e: int, stage: str) -> tuple[dict, list[str]]:
    deps, missing = dep_ids(e, stage)
    body = {"stage": stage, "code": code_hashes(stage), "params": params_of(stage),
            "inputs": inputs_of(e, stage), "deps": deps}
    body["id"] = sha1_bytes(json.dumps(body, sort_keys=True).encode())
    return body, missing


def outputs_exist(e: int, stage: str) -> bool:
    return all((ep_dir(e) / r).exists() for r in REQUIRED[stage])


def diff_reason(old: dict, new: dict) -> str:
    why = []
    for k in ("code", "params", "inputs", "deps"):
        a, b = old.get(k, {}), new.get(k, {})
        if a != b:
            keys = sorted(x for x in set(a) | set(b) if a.get(x) != b.get(x))
            why.append(f"{k}: {', '.join(keys)}")
    return "; ".join(why) or "unknown"


def check(e: int, stage: str, quiet: bool = False) -> int:
    new, missing = expected(e, stage)
    if missing:
        if not quiet:
            print(f"episode {e} stage {stage}: upstream stage(s) {', '.join(missing)} not stamped — run them first")
        return 2
    old = read_stamp(e, stage)
    if old is None:
        if not quiet:
            print(f"episode {e} stage {stage}: no stamp")
        return 1
    if old["id"] != new["id"]:
        if not quiet:
            print(f"episode {e} stage {stage}: changed ({diff_reason(old, new)})")
        return 1
    if not outputs_exist(e, stage):
        if not quiet:
            print(f"episode {e} stage {stage}: stamp current but outputs missing")
        return 1
    return 0


def write(e: int, stage: str, adopted: bool = False) -> None:
    new, missing = expected(e, stage)
    if missing:
        raise SystemExit(f"episode {e} stage {stage}: cannot stamp, upstream {missing} not stamped")
    if not outputs_exist(e, stage):
        raise SystemExit(f"episode {e} stage {stage}: cannot stamp, outputs {REQUIRED[stage]} missing")
    new["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    new["adopted"] = adopted
    stamp_path(e, stage).parent.mkdir(exist_ok=True)
    json.dump(new, open(stamp_path(e, stage), "w"), indent=1, sort_keys=True)


def archive(e: int, stage: str) -> None:
    W = ep_dir(e)
    dst = W / "_archive" / stage
    if dst.exists():
        shutil.rmtree(dst)
    moved = []
    for o in STAGES[stage]["outputs"]:
        if o == "SUB_EPISODE":
            srcs = [paths.DATA / "submission" / "episodes" / f"episode_{e:06d}{s}" for s in (".npz", "_object.obj")]
            srcs.append(paths.DATA / "submission" / "reports" / f"episode_{e:06d}.json")
        else:
            srcs = [W / o]
        for src in srcs:
            if src.exists():
                rel = src.name if o == "SUB_EPISODE" else o
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst / rel))
                moved.append(rel)
    stamp_path(e, stage).unlink(missing_ok=True)
    print(f"episode {e} stage {stage}: archived {', '.join(moved) if moved else 'nothing'} -> {dst.relative_to(OUT)}")


def expand(spec: list[str]) -> list[int]:
    out = []
    for t in spec:
        for part in t.split():
            if "-" in part:
                a, b = part.split("-")
                out += list(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
    return out


def status(eps: list[int]) -> None:
    order = ["1", "2", "3", "4", "5", "5e", "6", "7", "8"]
    sym = {0: "ok", 1: "STALE", 2: "-"}
    print("ep  " + "".join(f"{s:>7s}" for s in order))
    for e in eps:
        cells = []
        for s in order:
            rc = check(e, s, quiet=True)
            if rc == 1 and read_stamp(e, s) is None:
                cells.append("none")
            else:
                cells.append(sym[rc])
        print(f"{e:2d}  " + "".join(f"{c:>7s}" for c in cells))
    print("ok = up to date; STALE = code/settings/inputs/upstream changed; none = never stamped; "
          "- = an upstream stage is not stamped")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "write", "archive", "adopt", "status"])
    ap.add_argument("--episode", type=int)
    ap.add_argument("--episodes", nargs="+", default=["0-29"])
    ap.add_argument("--stage", choices=list(STAGES))
    a = ap.parse_args()
    if a.cmd == "status":
        status(expand(a.episodes))
    elif a.cmd == "adopt":
        for e in expand(a.episodes):
            write(e, a.stage, adopted=True)
            print(f"episode {e} stage {a.stage}: adopted existing outputs")
    elif a.cmd == "check":
        sys.exit(check(a.episode, a.stage))
    elif a.cmd == "write":
        write(a.episode, a.stage)
    elif a.cmd == "archive":
        archive(a.episode, a.stage)


if __name__ == "__main__":
    main()
