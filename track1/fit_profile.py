"""Profiling of the kit's mesh -> MHR fit (tools/track1/mesh_to_mhr_params.py), for fit_mhr.py --profile.

install(module) patches the kit module IN THIS PROCESS ONLY (the kit file is not changed):
  * cold_fit / lm_pose / lm_joint are wrapped with timers; each call is attributed to a phase:
      keyframes        cold_fit before tracking            identity      lm_joint
      tracking         lm_pose chunks (warm start)         stragglers    lm_pose of 1 frame during a sweep
      backward         lm_pose of 1 frame in the backward pass
      cold_refit       cold_fit during tracking (identity frozen)
      refine_retrack   lm_pose of all frames after the identity refinement
  * track() is replaced by an instrumented copy of the kit's track() (same logic, same order, same
    thresholds) that also records, for every frame the outlier rule flags, its error after tracking,
    after the backward re-solve and after the cold refit, and whether each step improved it.
report() returns {"phases": {name: {"calls", "seconds"}}, "flagged": [...], "median_error_mm", "rule"}.
"""
from __future__ import annotations

import time

import torch

_rec = {"phases": {}, "flagged": [], "phase": "keyframes", "in_track": False}


def _add(name: str, dt: float) -> None:
    p = _rec["phases"].setdefault(name, {"calls": 0, "seconds": 0.0})
    p["calls"] += 1
    p["seconds"] += dt


def install(m) -> None:
    cold_fit, lm_pose, lm_joint = m.cold_fit, m.lm_pose, m.lm_joint

    def t_cold_fit(*a, **k):
        t = time.time(); r = cold_fit(*a, **k)
        _add("cold_refit" if _rec["in_track"] else "keyframes", time.time() - t)
        return r

    def t_lm_pose(mhr, tgt, *a, **k):
        t = time.time(); r = lm_pose(mhr, tgt, *a, **k)
        _add(_rec["phase"] if _rec["in_track"] else ("refine_retrack" if _rec.get("tracked") else "anchors"),
             time.time() - t)
        return r

    def t_lm_joint(*a, **k):
        t = time.time(); r = lm_joint(*a, **k)
        _add("identity", time.time() - t)
        return r

    def track(mhr, tgt, z, anchors, wbody, chunk=8, iters=15, log=print):
        """Copy of the kit's track() with per-frame bookkeeping of the outlier re-solves."""
        _rec["in_track"] = True
        T = len(tgt); dd = dict(dtype=tgt.dtype, device=tgt.device)
        pose = torch.zeros(T, m.NP, **dd); err = torch.full((T,), float("inf"), **dd)
        ks = sorted(anchors)
        _rec["phase"] = "anchors"
        pose[ks], err[ks] = m.lm_pose(mhr, tgt[ks], torch.cat([anchors[k] for k in ks]), z, iters=iters)
        n_strag = 0

        def sweep(order):
            nonlocal n_strag
            prev = order[0]
            for a in range(1, len(order), chunk):
                fr = order[a:a + chunk]
                _rec["phase"] = "tracking"
                p, e = m.lm_pose(mhr, tgt[fr], pose[prev:prev + 1].expand(len(fr), -1), z, iters=iters)
                better = e < err[fr]
                pose[fr] = torch.where(better[:, None], p, pose[fr]); err[fr] = torch.minimum(e, err[fr])
                med = err[fr].median()
                for i, f in enumerate(fr):
                    nb = order[a + i - 1]
                    if m.is_outlier(err[f], med) and err[nb] < err[f]:
                        _rec["phase"] = "stragglers"; n_strag += 1
                        p1, e1 = m.lm_pose(mhr, tgt[f:f + 1], pose[nb:nb + 1], z, iters=2 * iters)
                        if e1[0] < err[f]:
                            pose[f], err[f] = p1[0], e1[0]
                prev = fr[-1]
                log(f"\r  tracking {a + len(fr)}/{len(order)} frames", end="")
            log("")
        sweep(list(range(ks[0], T)))
        if ks[0] > 0:
            sweep(list(range(ks[0], -1, -1)))
        med = err.median()
        _rec["median_error_mm"] = float(med)
        flagged = {}
        n_back = n_cold = 0
        for f in range(T - 2, -1, -1):
            if m.is_outlier(err[f], med):
                rec = flagged.setdefault(f, {"frame": f, "after_tracking": float(err[f])})
                _rec["phase"] = "backward"
                p1, e1 = m.lm_pose(mhr, tgt[f:f + 1], pose[f + 1:f + 2], z, iters=2 * iters); n_back += 1
                rec["backward"] = float(e1[0]); rec["backward_improved"] = bool(e1[0] < err[f])
                if e1[0] < err[f]:
                    pose[f], err[f] = p1[0], e1[0]
        for f in range(T):
            if m.is_outlier(err[f], med):
                rec = flagged.setdefault(f, {"frame": f, "after_tracking": float(err[f])})
                rec["before_cold"] = float(err[f])
                p1, _, e1 = m.cold_fit(mhr, tgt[f], wbody, z_fixed=z); n_cold += 1
                rec["cold"] = float(e1); rec["cold_improved"] = bool(e1 < err[f])
                if e1 < err[f]:
                    pose[f], err[f] = p1[0], e1
        for f, rec in flagged.items():
            rec["final"] = float(err[f]); rec["still_outlier"] = bool(m.is_outlier(err[f], med))
        _rec["flagged"] = sorted(flagged.values(), key=lambda r: r["frame"])
        _rec["stragglers"] = n_strag
        _rec["in_track"] = False; _rec["tracked"] = True
        return pose, err, dict(backward_resolves=n_back, cold_refits=n_cold)

    m.cold_fit, m.lm_pose, m.lm_joint, m.track = t_cold_fit, t_lm_pose, t_lm_joint, track
    _rec["rule"] = "outlier if error > 2 x median + 0.02 mm (kit is_outlier)"


def report() -> dict:
    return {k: v for k, v in _rec.items() if k not in ("phase", "in_track", "tracked")}
