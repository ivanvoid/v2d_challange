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
| 2 | 2026-10-22 | stage67_v2 | `8873d07` | stages 5e-8 re-run: new 5e meshes (iron forced to ep 6 frame 723), new stage 6 tracker (re-find, states), stage 7 hidden frames interpolated, stage 8 fixed/follow scale | 15.716 | 30.956 | 0.1742 | 0.1235 | 0.0451 | 0.0555 | 4.65 | 0.489 | 0.130 | 0.112 | `experiments/stage67_v2/track1.parquet` |
| 3 | 2026-10-27 | penfix_smooth | `fd79f96` | stage-8-only bounded smooth object translation on foam episodes 15–17; scores reported by user, receipt/upload hash pending | 15.9737* | 30.9978 | 0.17421 | 0.12355 | 0.03349 | 0.0396 | 4.70 | 0.487 | 0.130 | 0.112 | `experiments/penfix_smooth/track1.parquet` |

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

**2 · stage67_v2 (submitted Oct 22; scores from the Oct 27 public leaderboard)** — a bundle, not a one-variable change: everything from stage 5e on was
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

## PEN correction trial (local, October 27; not uploaded)

The active stage-8 runner uses `poses_final`, so the pre-existing `stage8_depen.py` was not in any submitted
pipeline. Tests in `track1/tests/test_penetration_contract.py` verify the scorer kernel on a synthetic solid,
the 40-frame trim offset in foam episode 16, the zero-gap-inside-object ambiguity, and a hard total-shift cap.
No production output or scorer code was modified by this experiment.

An isolated stage-8-only trial under `experiments/penfix_smooth` uses the existing MHR fit, budgeted mesh
and final poses. It fits object translations on 256 sampled hand vertices, applies Gaussian smoothing
(sigma=5) without unsmoothed residual repairs, and caps total shifts at 3 cm. It keeps all object rotations,
scale, body parameters and mesh unchanged. It then rebuilds and evaluates the actual packed Parquet, not
just the fitting objective. One initial unsmoothed attempt reduced PEN but raised object self-acceleration
from 0.184 to 0.528 cm/frame^2 and was rejected. An initial preview packed with frame_offset=0 was invalid;
the scored video in episode 16 starts at local offset 40. All reported results below use the correct offset.

| foam episode | local PEN (control → trial, cm) | gap (cm) | contact | accO self (cm/frame²) |
|---|---|---|---|---|
| 15 | 0.4170 → 0.2879 | 0 → 0.2 | 0.58 → 0.56 | 0.236 → 0.234 |
| 16 | 0.4584 → 0.2886 | 0 → 0.7 | 0.68 → 0.65 | 0.184 → 0.186 |
| 17 | 0.2637 → 0.0862 | 1.0 → 1.5 | 0.56 → 0.55 | 0.176 → 0.180 |

Independent anatomical hand samples from an MHR-model-hash-matched CARI4D spec (not the secret Kaggle
hand indices) also improve on all three: ep 15 0.326→0.235, ep 16 0.367→0.227, ep 17 0.159→0.049 cm.
These are local proxies; only Kaggle evaluates reference alignment and the hidden hand indices. Do not
interpret zero median contact gap as no penetration. A one-episode trial with stronger shifts had gap 3.7 cm
and contact 40 % and was rejected. Recheck the full packed candidate before any upload, and keep it separate
from both `submission/track1.parquet` and `experiments/stage67_v2/track1.parquet`.

**Full packed trial:** `experiments/penfix_smooth/track1.parquet` (740,780 rows), containing only 691 changed
`object_translation` rows in episodes 15-17; every other array and episode matches experiment 2 exactly.
Matched 30-episode local mean PEN is **0.0555 -> 0.0396 cm**; contact 0.4889 -> 0.4872, object
self-acceleration 0.1122 -> 0.1123 cm/frame². All three changed episodes pass the guardrails in
`experiments/penfix_smooth/full_gate.json`. The kit's local schema validator passes. The packed file
references code commit `fd79f965f212345b9bca3ca15c2587a2bb243ccf`, which is currently on local
`beta-models` only: **do not upload until the commit is pushed and the GitHub URL is accessible**.
The file SHA-256 is `415332f98891baf3af8e7fbb6c6ec9559af06dfacfe9b6be7d1e2c91bfafc652`.
The local trial alone did not establish its Kaggle scores or guarantee improvements on the other four metrics;
the subsequently user-reported Kaggle result is recorded below.

**3 · penfix_smooth user-reported Kaggle result (Oct 27, receipt not yet inspected):** PEN 0.04506 →
0.03349 cm (−25.7 %), CD-O 30.95621 → 30.99780 cm (+0.04159 cm), ACC-O 0.12346 → 0.12355
cm/frame² (+0.00009), ACC-H unchanged at 0.17421. User reports CD-H 15.97370 versus
15.71611 for experiment 2 (+0.25759 cm). **This CD-H difference is not attributable to this
local artifact:** compared against `experiments/stage67_v2/track1.parquet`, its only changed
`x/y/z` values are 691 `object_translation` rows in episodes 15–17. The Track 1 scorer's
`cd_h_cm` evaluates human surface points only and uses an episode-indexed sampling seed.
Check that all five score entries belong to the same Kaggle upload and compare the submitted
file's SHA-256 with `415332f98891baf3af8e7fbb6c6ec9559af06dfacfe9b6be7d1e2c91bfafc652`.
Do not treat the reported CD-H change as an algorithmic regression until provenance is verified.

**CARI4D follow-up feasibility (no pretrained inference yet):** the native-MHR implementation under
`/workspace/v2d_cari4d` has a model-matched anatomical hand surface asset and differentiable
`object_sdf_penetration_loss`; `track1/tests/test_cari4d_feasibility.py` confirms the loss produces
the expected gradient for hand points inside an analytic solid. However, the default parity
refinement penalizes **object samples inside the human**, not Kaggle's **hand points inside the
object**. This must be evaluated or augmented before drawing any conclusion about PEN. A real
CoCoNet experiment needs the separate gated `nvidia/cari4d_commercial` step-200000 checkpoint,
the CARI4D image/runtime, and accepted model terms. Neither this checkpoint nor a Docker runtime
is currently present in this container; `HF_TOKEN` is unset. The public upstream step-031397
checkpoint in `weights/cari4d/hf/` is not a drop-in replacement for the native-MHR model.
As a separate one-frame feasibility probe on the existing episode-16 submission, the installed
TorchScript MHR model produced nonzero gradients in 53 of its 54 finger-control columns from
a **box-proxy** hand-inside-object loss (53/54, gradient L2 0.00452); this is not a real-mesh
penetration improvement and did not change any submission. It indicates that bounded, temporal
MHR finger-pose refinement is technically possible without installing another model. The proper
next experiment needs actual budgeted-mesh signed depth, finger-pose and reprojection priors,
and the same packed-output guardrails as the object-only trial.

**Stop test for a finger-only pilot:** `track1/tests/test_foam_hand_overlap.py` checks
episode 16, video frame 237, against the *packed* `penfix_smooth` mesh/body and the
model-matched anatomical hand sample. **239 of 256 left-hand vertices are inside**;
mean left-hand depth is about 4.3 cm, maximum about 10 cm. Perturbing all finger
controls by 0.3 rad still moves 63 penetrating left-hand points no more than
0.2 cm (many palm/wrist points). Only 14/256 right-hand points penetrate.
Consequently, a small finger-only pose correction cannot remove this collision;
the next physical-consistency experiment should jointly diagnose body-hand/object
placement and geometry under image constraints. A full CARI4D run remains blocked
by its separate gated checkpoint/runtime and cannot be inferred from this diagnostic.

## VDA depth ablation (local only; stool episodes 24–26)

`ripple_stool`: Video Depth Anything depth with production SAM3D mesh and no CoTracker. All
stages 1–8 completed; the stage-8 review initially crashed on obsolete `variants`
in `scale_match.json`. The generator was subsequently fixed and its episode pages rebuilt.
Compare to the **current** control `logs/eval/beta/prodnow`, not the runner's old-baseline compare log.

| matched mean over ep 24–26 | current production | VDA |
|---|---:|---:|
| local PEN (cm) | 0.0181 | 0.0022 |
| hand–object gap (cm) | 6.95 | **12.57** |
| contact fraction | 0.328 | **0.167** |
| object size (m) | 0.552 | 0.629 |
| body minus object depth (cm) | −8.8 | −28.1 |

Episode 25's gap worsened 0 → 9.7 cm and contact 63% → 26%; episode 26's gap
15.6 → 19.3 cm. The PEN reduction is not evidence of improvement when the object moves
away from the hands. Do **not** substitute VDA into a submission on this evidence.
Review: `logs/beta/ripple_stool/07_smooth/index.html` and
`logs/beta/ripple_stool/08_final/review/episode_000025_result.png`.

**Scale-boundary follow-up (isolated stage-8-only ablation):** the VDA stool search
selected its lower bound k=0.800 on all three episodes despite a better logged
off-grid diagnostic on episode 25. Tests in `track1/tests/test_pose_scale_ablation.py`
confirm this before running. Re-searching k=0.65..1.25 in steps of 0.025 on the
same VDA outputs (`experiments/scale_rescue_vda_stool/`, partial/non-uploadable):

| matched mean, episodes 24–26 | production | original VDA | VDA + expanded search |
|---|---:|---:|---:|
| PEN cm | 0.0181 | 0.0022 | 0.0144 |
| hand–object gap cm | **6.95** | 12.57 | 8.03 |
| contact fraction | **0.328** | 0.167 | 0.268 |
| object self-acceleration | 0.118 | 0.116 | 0.104 |
| max object size m, mean | 0.552 | 0.629 | 0.582 |

Episode 25 gap 9.73→0.53 cm and contact 26%→53%, but production remains at
0 cm/63%; episode 26 remains disconnected (15.9 cm gap). The expanded scale
search **rescued a real integration bug but did not make VDA competitive as a
whole**. Its 745 changed packed rows are exclusively object translation/scale;
body, mesh, rotation and frame IDs are unchanged. Object self-acceleration is
not official ACC-O error. The stage-8 review generator was repaired and its
VDA stool page rebuilt at `logs/beta/ripple_stool/08_final/index.html`.

## Stage-7 translation smoothing ablation (local only; episodes 8, 16, 25, 28)

Current `stage7_smooth.py` uses `--lam_t 50`. Isolated `smooth_t15` and
`smooth_t150` trials use the same production stage-6 poses, the same rotation
smoothing, and re-run stage 8; their partial Parquets are **not uploadable**.
Matched control is `logs/eval/beta/prodnow` on exactly these four episodes.

| matched mean | production λ=50 | weaker λ=15 | stronger λ=150 |
|---|---:|---:|---:|
| object self-acceleration cm/frame² | 0.1672 | **0.2300 (worse)** | 0.1225 |
| PEN cm | 0.1639 | 0.1582 | 0.1665 |
| gap cm | 0.10 | 0.29 | 0.13 |
| contact fraction | 0.668 | 0.645 | 0.665 |

Weaker smoothing changed episode 25's object size by −0.075 (stage-8 scale
reselection) and moved its object up to 26 cm on scored frames. **Reject it.**
Stronger smoothing preserved all object sizes, body parameters, meshes,
rotations and frame IDs; only object translations changed. It lowered local
self-acceleration on all four episodes, but this is **not the reference-based
Kaggle ACC-O error**. Episode 25 still has a 7.6 cm maximum displacement
(3.8 cm scored-frame p95) relative to production, so review transitions.
The six-frame sheet and stage-7 checks are under `experiments/smooth_t150/review7/`;
episodes 8, 16, 28 pass; episode 25 has a warning from a large correction to
already-bad stage-6 outlier frames. Result sheets for 16 and 25 are under
`experiments/smooth_t150/review_final/`. No change is promoted or submitted yet.

A *separate* full-file ACC-O ablation is packed at
`experiments/smooth_t150_full/track1.parquet` (SHA-256
`4cdab16befc4dfa725c643a5479493d6c0b4756b8a91de2bd7b633ee64ff430a`).
It uses **experiment 2 as its control**, not the later PEN correction: all
740,780 rows are present; exactly 1,060 object-translation rows differ, in
episodes 8, 16, 25 and 28 only. The body, meshes, object rotations, scales
and all other episodes are byte-equivalent to experiment 2. The kit's local
schema validator passes and tests verify the episode NPZs. It embeds commit
`1b915846561405db43156418f104caad051eceab`, which must be pushed before
any submission. At the time of local validation this ablation had no Kaggle
score; the proxy decrease did not establish an ACC-O gain against the hidden reference.

**Full 30-episode check completed** (`experiments/smooth_t150_full/compare_production.json`):
local object self-acceleration 0.112164→0.106191 cm/frame² (−0.005973), PEN
0.055463→0.055814 cm (+0.000352), contact 0.488920→0.488524 and gap
4.651899→4.656379 cm. Only the four intended episodes change. This is a
clean ACC-O benchmark ablation, but self-acceleration is not the scored ACC-O
error. The subsequently user-reported Kaggle result appears below; keep it
separate from `penfix_smooth` when attributing that score.

**Later user-reported Kaggle result for `smooth_t150_full` (receipt not inspected):**
CD-H 15.97370, CD-O 30.92902, ACC-H 0.17421, ACC-O **0.11826**, PEN 0.04505.
Against experiment 2 ACC-O is 0.12346→0.11826, CD-O 30.95621→30.92902,
PEN approximately unchanged; against `penfix_smooth` ACC-O improves but PEN
returns to the experiment-2 level. The local CD-H-provenance anomaly remains.

## Stationary-stool multi-frame pose trial (local only; episodes 24 and 26)

The user confirmed that the stool does **not move throughout either video**;
episode 25 is not assumed stationary, and the camera is fixed only per video.
`track1/stationary_stool.py` uses 5 high-visibility anchors distributed across
each clip to select one existing final mesh pose by median mask IoU at *other*
anchor frames, then validates it on 66/71 held-out clear frames. It does not
change the camera or any body parameter, object mesh or scale. It writes one
constant object R,t for the whole video directly into isolated submission NPZs,
so later stage-8 scale selection cannot undo the stationarity.

| Episode | held-out IoU original → stationary | max translation change | local object self-acceleration |
|---|---:|---:|---:|
| 24 | 0.819 → 0.833 | 5.48 cm | 0.034 → 0 |
| 26 | 0.774 → 0.786 | 7.41 cm | 0.039 → 0 |

The episode-24 hand-gap proxy rises 5.19→7.02 cm, but continuous hand contact
is **not** a valid stool prior when sitting, so do not retune the stool position
to minimize it. Episode 25 is untouched. The result is a *physical-stationarity
hypothesis*, not proof of improvement against hidden ACC-O/CD-O reference data;
check packed structure, the reviewed overlays and Kaggle metrics before promotion.
Files and frame-index/holdout tests: `experiments/stool_static_v1/` and
`track1/tests/test_stationary_stool.py`.

**Full-file stationary-stool candidate based on the separately scored λ=150
smoothing variant:** `experiments/stool_static_plus_smooth_t150/track1.parquet`
(SHA-256 `db83a85d5379b78934287b21493149d0ff5a0215a9b17ba43aed6c16bc795f0a`).
Compared with `experiments/smooth_t150_full/track1.parquet`, only object
rotation and translation rows in episodes 24 and 26 differ; the other 28
episodes, all human parameters, object meshes, scales and scored frame IDs
are identical. The full 740,780-row kit validator and tests pass. Matched
30-episode local self-acceleration 0.106191→0.103777 cm/frame², PEN
0.055814→0.055558 cm; contact 0.488524→0.486457, gap 4.656379→4.710636.
Because sitting is not continuous hand contact and ACC-O scores error against
the hidden reference, these proxies alone do not prove a benchmark gain.
The file embeds commit `126c6c86a655ea9645ae6869786d7d8667baac89`;
push that commit so the organizers can inspect it.

**Later user-reported Kaggle result (receipt not inspected):** CD-H 15.97370,
CD-O 30.95619, ACC-H 0.17421, ACC-O **0.11673**, PEN 0.04496. Relative to
`smooth_t150_full`, ACC-O 0.11826→0.11673 (−0.00153 cm/frame²), CD-O
30.92902→30.95619 (+0.02717 cm), PEN 0.04505→0.04496 (−0.00009 cm),
human metrics unchanged. This **is** an incremental ACC-O gain, not evidence
of a large overall reconstruction improvement. For a larger effect, investigate
the hula hoop's poor stage-6 tracking and the filled-in late SAM2 masks.

**Camera-prior check:** Stage 1 already uses AnyCalib predictions from 16
frames evenly distributed across each video (episode 24 logged samples 0,
17, ..., 261), aggregates intrinsics/distortion by a robust median, and
stores fixed per-video intrinsics. No single-frame camera-initialization
bug has been demonstrated, so do not rerun camera calibration without an
independent residual or image evidence of a calibration error.

## Hula hoop: geometry versus segmentation (isolated local probes)

The user estimates the same rigid hoop's outer diameter at roughly **67–72 cm
from the person's shoe** (a soft prior, not a measured caliper value). The
episode-0 hoop lies on the floor in the first clear second, is subsequently
lifted, and lies on the floor again at the end; episode 1 is held throughout;
episode 2 is held until its final seconds. `floor.json` supplies an independently
fitted plane per video. AnyCalib already uses 16 distributed frames per video.

**Do not blindly replace the object mesh with a torus.**
`track1/hoop_torus_probe.py` generated clean watertight genus-one meshes of
one shared *physical* diameter and evaluated them against multi-frame masks
with the original camera and pose held fixed. No high-fit stage-6 training
anchors exist for episode 1. The best common ring (diameter 0.66 m, tube
0.03 m) gave held-out IoU ep 0 **0.668→0.721** but ep 2 **0.226→0.000**.
Ep 2 alone prefers a hoop near 0.82–0.86 m in its CURRENT pose/scale.
This violates the shared-size prior and implies that pose/scale also needs
repair. The geometry-only candidate failed its held-out guard, produced no
submission, and the original meshes/poses remain unchanged.

**Stage-4 mask root cause discovered in episode 0:** early floor-frame mask
000000 has an inner hole covering 64% of its outer contour; late floor-frame
000600 has **no hole**—SAM2 produced a filled disk. `masks_raw/1/000600.png`
and the final mask are identical, so downstream mask repair did not introduce
the fill. A conservative offline repair of high-fill masks only in the
user-confirmed final floor interval changed **48/630 frames**, never adds
pixels or affects the opening masks, and raised IoU of the existing saved
ring poses on those frames from **0.279→0.636** (48 improved, 0 worsened).
The masks reside only under `experiments/hoop_mask_v1/masks/1/`; the test is
not ground truth and the independent gate is an isolated stage-6 rerun,
logged at `experiments/hoop_mask_v1/stage6.log`. The rerun **failed the
predeclared unaffected-frame gate**: repaired scored frames 38/169 improved
IoU 0.210→0.623 and lost share 28.9%→0%, but untouched scored frames 131/169
worsened lost share 2.3%→22.1%. Newly tracked late poses are physically
almost the same as production (only ~0.2–0.3 cm translation difference;
~180° symmetric ring orientation, ~0.4–0.6 cm surface difference). A shape-
corrected mask helped the metric's target, not the 3D trajectory. **No hoop
submission assembled** from this rejected run. Full verdict in
`docs/STAGE_AUDIT_2026-10-27.md`.

**Held-hoop episode-1 visibility-anchor pilot:** no mask or mesh was changed.
Seven anchors were proposed, but the first three early attempts had best IoU
0.00/0.13/0.23 and were rejected. Scored-frame lost share worsened
32.8%→33.9% with no meaningful IoU improvement. The independent gate at
`experiments/hoop_vis_v1/stage6_gate.json` rejected the run; no submission
file was produced. Future anchor search must validate registration fit rather
than rely on maximum visible mask area.
