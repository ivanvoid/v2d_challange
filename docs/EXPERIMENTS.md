# Experiment ledger — Track 1 submissions

One row per Kaggle submission. Change ONE thing per submission (about 5 submissions per week).
Kaggle scores: lower is better for all five. Local columns come from
`track1/eval_local.py` run on the exact parquet that was uploaded (scored frames only):

```bash
.venv-submit/bin/python track1/eval_local.py extract --parquet <file> --out logs/eval/<name>
CUDA_VISIBLE_DEVICES=0 /opt/venvs/sam3d_body/bin/python track1/eval_local.py proxies --out logs/eval/<name>
/opt/venvs/sam3d_body/bin/python track1/eval_local.py compare logs/eval/<before> logs/eval/<name>
```

Local measurements (no reference needed): `PEN` is the scorer's PEN computed locally (512 sampled MHR
hand vertices, alignment scale 1); `gap` = median over scored frames of the closest hand-object
distance (scored frames are contact frames, so this should be ~0); `contact` = fraction of scored
frames with gap < 2 cm; `accH`/`accO` = our own mean |second difference| of the 22 scored joints /
the object translation per scored stretch (jitter, not the error).

| # | Date | Name | Commit | Change (only this) | CD-H | CD-O | ACC-H | ACC-O | PEN | local PEN | gap cm | contact | accH | accO | Files |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2026-10-15 | baseline | `20b2d5c` | stages 1–8 as in HANDOFF.md | 15.716 | 37.420 | 0.1745 | 0.1274 | 0.1007 | 0.0966 | 12.38 | 0.454 | 0.131 | 0.115 | `submission/track1.parquet` |
| 1 | 2026-10-19 | p1_global | `899c80e` | episodes 0 1 8 14 25 27 28 29: stage-8 scale variant "none" → "global" (object translation and size × s_global 1.07–1.16) | 15.716 | **43.294** | 0.1745 | 0.1306 | 0.0889 | 0.0850 | 16.06 | 0.417 | 0.131 | 0.118 | `experiments/p1_global/track1.parquet` |
| 2 | 2026-10-22 | stage67_v2 | `8873d07` | stages 5e-8 re-run: new 5e meshes (iron forced to ep 6 frame 723), new stage 6 tracker (re-find, states), stage 7 hidden frames interpolated, stage 8 fixed/follow scale | _not submitted_ | _not submitted_ | _not submitted_ | _not submitted_ | _not submitted_ | 0.0555 | 4.65 | 0.489 | 0.130 | 0.112 | `experiments/stage67_v2/track1.parquet` |

## Notes per experiment

**1 · p1_global** — question: should the object live in the body's depth scale ("global") or in
MoGe's ("none")? The baseline chose by the smallest hand–object gap over the whole first..last
range (the contact span; gaps between scored stretches are frames with invalid ground truth, not
non-contact frames — see v2dlb/mhr_submission.py:34). Projective consistency and CARI4D (which anchors
monocular depth to the human body, `models/CARI4D/prep/align_monod2hum.py`) both argue for the body
scale. Read CD-O first. If CD-O improves, follow-ups: "global" everywhere vs per-frame (CARI4D
aligns per frame with a robust scale + shift); choose the variant on scored (contact) frames only.

Local prediction (`eval_local.py compare logs/eval/baseline logs/eval/p1_global`): the two local
signals disagree, which is what makes this submission informative.
* contact on scored frames gets WORSE on most changed episodes (28: gap 0.3 → 9.7 cm, contact
  58 → 31 %; 29: 0.0 → 3.3 cm; 8: 4.2 → 5.9 cm; 25: 0.0 → 2.1 cm); better only on 14 (12.3 → 8.7 cm)
* object size agrees BETTER across the 3 videos of each object (largest ratio 1.28 → 1.21; carts
  0.88/0.72/0.80 → 0.96/0.82/0.87 m, hoop 0.72/0.72/0.80 → 0.83/0.84/0.80 m)
* local PEN 0.0966 → 0.0850
If CD-O improves: trust the body's scale (and size consistency); the hands are what is misplaced.
If CD-O gets worse: the hand-contact signal is the better guide for object placement.

**Result: CD-O 37.42 → 43.29 (worse by 5.87), ACC-O 0.1274 → 0.1306, PEN 0.1007 → 0.0889, CD-H and
ACC-H unchanged.** The contact signal predicted the direction; size consistency did not. Lessons:
* place the object where the hands touch it on scored (contact) frames — not by the body's scale
* object depth placement is a big lever: 8 episodes moved by 7–16 % in depth changed the
  30-episode CD-O by ~6 cm (≈ 22 cm per changed episode)
* local PEN is calibrated: Kaggle / local = 1.046 here, 1.042 for the baseline
Next: choose the object's depth scale per episode continuously (not 3 variants) by minimising the
hand–object gap on scored frames only, and check the gain locally before submitting.

**2 · stage67_v2 (local only, not yet submitted)** — a bundle, not a one-variable change: everything from stage 5e on was
re-run (see HANDOFF.md). Local proxies vs the baseline (`eval_local.py compare logs/eval/baseline logs/eval/stage67_v2`):
* local PEN 0.0966 -> 0.0555, median hand-object gap 12.4 -> 4.65 cm, contact 0.454 -> 0.489,
  object steps > 10 cm per frame 0.233 -> 0, accO(self) 0.1145 -> 0.1122, accH(self) unchanged (0.131 -> 0.130)
* worse: iron PEN ep 7 0.041 -> 0.093, ep 8 0.021 -> 0.128; foam block ep 15 0.22 -> 0.42 (ep 16/17 better); contact on
  bowls 0.94 -> 0.57 in ep 4 (gap still < 1 cm)
* unresolved: hula hoop (ep 0 gap 42 cm / ep 2 22 cm, tracker median IoU 0.28-0.39), paint roller ep 18/19 gap 11-16 cm,
  stool ep 26 gap 15.6 cm (seated); size spread across videos of one object up to 1.28x
* all 30 episodes packed: 740,780 rows, row ids/order equal to the sample, no non-finite values
Local proxies are not CD-H/CD-O/ACC errors (see AGENT_HANDOFF.md section 6); only Kaggle scores say whether CD-O improved.

## Beta models (local only, not submitted) — `run_beta.sh`, branch `beta-models`

Candidates: Video Depth Anything depth (`BETA_DEPTH=vda`), TRELLIS.2 / Hunyuan3D-2 meshes (`BETA_MESH`), CoTracker3
pose refinement (`BETA_TRACK=cotracker`). Run on the iron episodes 6 7 8 only.

**Compare against the control `logs/eval/beta/prodnow`** (current `submission/episodes`, i.e. experiment 2 = stage67_v2,
all 30 episodes), NOT against `logs/eval/baseline`: that one is experiment 0 (Oct 15), and `run_beta.sh eval` still
compares with it by default (set `BETA_BASELINE=logs/eval/beta/prodnow`). Against the stale baseline CoTracker looked like a
big win (gap 12.4 -> 0.1 cm); that was the pipeline changes since experiment 0, not the model. Also `eval_local.py compare` prints
the mean over each run's own episodes, so its last line is not comparable when the runs cover different episode sets.

Mean over episodes 6 7 8 (iron):

| run | gap cm | contact | PEN | accO | object size m |
|---|---|---|---|---|---|
| exp 0 baseline (Oct 15) | 18.08 | 0.316 | 0.0455 | 0.0969 | 0.305 |
| prodnow = exp 2 | 0.00 | 0.704 | 0.0924 | 0.1000 | 0.285 |
| warp (CoTracker3) | 0.10 | 0.695 | 0.0839 | 0.1234 | 0.285 |
| ember (Hunyuan3D-2 meshes) | 1.63 | 0.548 | 0.0896 | 0.1142 | 0.390 |

* CoTracker3: reprojection error of the refined poses drops (1.99 -> 1.21, 1.88 -> 1.08, 3.68 -> 1.60 px), but no proxy improves
  clearly over exp 2 (PEN -9 %, accO +23 % worse). No evidence for including it.
* Hunyuan3D-2: real 3D but lumpy, oversized meshes (0.35-0.44 m; iron prior 0.18-0.40); worse than exp 2 on episodes 6 and 7.
* TRELLIS.2: flat sheets (thickness 0.2-5 % of width) for the iron (47 x 82 px in the frame); the official example image works,
  so it is the input, not the install. Larger objects (stool, cart) are probed separately.
* Environment notes: `transformers` must be 4.56.2 (5.x breaks both DINO loaders); Hunyuan also needs `diffusers==0.35.2`.

**CoTracker3 on 12 episodes** (iron 6 7 8, foam block 15 16 17, stool 24 25 26, cart 27 28 29; `warp` vs the control `prodnow`,
means over the same 12 episodes):

| run | gap cm | contact | PEN | accH | accO | object size m |
|---|---|---|---|---|---|---|
| prodnow (exp 2) | 1.851 | 0.595 | 0.1283 | 0.1352 | 0.1263 | 0.491 |
| warp (CoTracker3) | 1.856 | 0.596 | 0.1122 | 0.1352 | 0.1403 | 0.489 |

PEN better on 9 of 12 episodes (-12.5 % mean), accO worse on 9 of 12 (+11 %), gap/contact unchanged (gap worse on 5, better on 2).
It does not repair the weak episodes (stool ep 26 gap 15.6 -> 15.9 cm). A trade (PEN vs ACC-O), not a gain: not included
in the submission. If a submission slot is spare it could be run as its own experiment (exp 2 + CoTracker, one change).
