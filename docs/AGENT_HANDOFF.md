# Agent Handoff: Correctness-First V2D Track 1 Stabilization

## 1. Mission and scope

The user's highest priority is **making sure everything works as expected**. Do not optimize leaderboard scores, throughput, or visual smoothness at the expense of valid outputs, reproducibility, or motion correctness.

Continue the existing Track 1 pipeline. First establish trustworthy tests and evaluation, then make small, attributable improvements to penetration, scale reconciliation, and stationary-object handling.

Read this document together with `docs/HANDOFF.md`:

- `docs/HANDOFF.md` contains the operational history, current pipeline, baseline scores, and known difficult episodes. It labels its snapshot as 2026-10-16; verify live state rather than inferring freshness from dates.
- This document refines that plan around correctness, acceptance tests, and safeguards. It does **not** claim that the proposed fixes are already implemented.
- Older audit documents describe earlier versions of the workflow. Current code and observed artifacts take precedence.

### State at preparation

- Repository: `/local-data/ivan/v2d/v2d_challange`
- Remote: `https://github.com/ivanvoid/v2d_challange` (private)
- HEAD: `20b2d5cab3d19a53c497f1e42e470f26146c9b6f`
- Before creating this document, `docs/HANDOFF.md` was already staged as a new file. Do not overwrite, unstage, or discard that user work.
- No implementation changes, GPU jobs, retraining, or new submissions were performed for this handoff.

## 2. Non-negotiable operating rules

1. **Use only GPUs 0 and 2. GPU 1 belongs to someone else.**
2. **One GPU job per GPU.** Check existing utilization and jobs before starting. Do not run a test beside a large model stage on the same GPU.
3. Run long jobs in `tmux`. Do not edit an actively executing shell script.
4. Preserve the baseline submission, meshes, poses, body fits, and diagnostic reports before any rerun that would overwrite them. Use versioned experiment output directories where possible.
5. Do not reset or discard unfamiliar Git changes. Inspect `git status` before and after work. Do not commit, push, invite collaborators, or upload submissions without user authorization.
6. Do not modify challenge inputs, model weights, or the official scorer to improve a result.
7. A plaintext Hugging Face token was previously committed. It has reportedly been removed from files, but external revocation must be confirmed by the user. Never print, recover, or copy the old token. New downloads use `HF_TOKEN` only.
8. Do not call a result "exact leaderboard evaluation" if official point indices, reference alignment, scored-frame selection, or packed geometry are missing.
9. No guaranteed IoU, acceleration, PEN, or leaderboard improvement. Measure results and label proxies clearly.
10. Avoid broad recomputation. First test CPU-only contracts, then one episode, then a representative subset, then all 30.

## 3. Environment and entrypoints

```text
CODE: /local-data/ivan/v2d/v2d_challange
DATA: $V2D_DATA, default /local-data/ivan/v2d
MODEL IMPLEMENTATIONS: /workspace/v2d_*
MODEL INTERPRETERS: /opt/venvs/<model>/bin/python
SUBMISSION INTERPRETER: $V2D_DATA/.venv-submit/bin/python
```

`track1/paths.py` resolves code and data locations. Old paths such as `$V2D_DATA/track1` and `$V2D_DATA/run_stage.sh` are symlinks into the repository. Edit the canonical repository files.

Relevant interpreters:

| Work | Interpreter/environment |
|---|---|
| Pipeline wrappers | `/opt/conda` Python; `V2D_BACKEND=venv` |
| MHR inference, fitting, reviews, current QA | `/opt/venvs/sam3d_body/bin/python` |
| Object registration/tracking | `/opt/venvs/foundation_pose/bin/python` |
| Rendering | `/opt/venvs/wilor/bin/python` |
| Parquet/schema/packing | `$V2D_DATA/.venv-submit/bin/python` |

Operational commands, **not an instruction to run them immediately**:

```bash
export V2D_DATA=/local-data/ivan/v2d
export V2D_BACKEND=venv
cd /local-data/ivan/v2d/v2d_challange

# Single stage, explicit episode and GPU:
GPU=0 EPISODES="16" ./run_stage.sh 8

# Larger runs, only after checking available GPUs and preserving outputs:
./run_stage_2gpu.sh 6 "0 2" "0-29"
```

The example stage 8 command overwrites final outputs; preserve baseline artifacts first. Do not use `run_rest.sh` as the first validation step.

## 4. Task and scoring contract

Track 1 contains 30 static-camera third-person videos, documented as 1536×1152 at 30 FPS. There are 10 physical objects, each represented in three episodes.

Each episode supplies:

```text
pose                [T,136]      per-frame MHR parameters
scales              [68]         one shared body-scale vector
shape               [45]         one shared identity vector
object_rotation     [T,3,3]
object_translation  [T,3]
object_scale        scalar      positive and constant per episode
object mesh                     one fixed triangle mesh
```

Arrays are indexed by original VIDEO frame. Intermediate filenames use LOCAL trimmed-clip frames:

```text
video_frame = local_frame + track1_meta.json["frame_offset"]
```

The baseline sample contains **740,780 rows** and must be matched exactly by the packed submission. Confirm against the live kit before relying on the count.

### Authoritative scoring behavior

- One Sim(3), fitted from the human at the first scored frame, aligns both human and object for the episode.
- CD-H/CD-O evaluate body/object surfaces.
- ACC-H uses the scorer's 22 selected body joints.
- ACC-O evaluates second differences of the submitted object **translation**, not a generic geometric center trajectory.
- Acceleration errors are computed against the hidden reference, without FPS scaling, and reported in cm/frame².
- Acceleration is evaluated within contiguous scored stretches, with the scorer's endpoint handling and weighting.
- PEN tests the scorer's fixed 512 MHR hand points against the participant's own submitted mesh, using winding-number inside tests and exact triangle distance. The result includes object scale and the hidden body-alignment scale.

Read:

```text
$V2D_DATA/v2d_submission_kit/v2dlb/mhr_submission.py
$V2D_DATA/v2d_submission_kit/v2dlb/mhr_metrics.py
$V2D_DATA/v2d_submission_kit/v2dlb/mesh_budget.py
$V2D_DATA/v2d_submission_kit/tools/pack_reconstruction.py
```

## 5. Current pipeline map

| Stage | Purpose | Main code |
|---|---|---|
| 1 | Trim, undistort, extract, depth, intrinsics | `track1/stage1_prep.py` |
| 2 | Person track, SAM3D Body, one-identity MHR fit | `run_stage.sh`, `track1/fit_mhr.py` |
| 3 | Actor-aware object selection | `track1/stage3_select.py` |
| 4 | Object masks, negative prompts, repair | `track1/stage4_mask.py`, `stage4_prompts.py`, `sam2_box_neg.py` |
| 5a–d | Generate candidate meshes and scale/check them | `track1/stage5_*.py` |
| 5e | Select one geometry across an object's three videos | `track1/stage5_object.py` |
| 6 | Multi-anchor bidirectional tracking and Viterbi selection | `track1/stage6_track.py`, `stage6_overrides.json` |
| 7 | Outlier-weighted whole-sequence smoothing | `track1/stage7_smooth.py` |
| 8 | Reconcile object/body scale, build episode, QA | `track1/stage8_scale.py`, `build_episode.py`, `qa_episode.py` |

Stage reviews are in `$V2D_DATA/logs/stages/<stage>/index.html`. Inspect trajectories and overlays, not only summary status.

## 6. Confirmed issues that must guide the work

### A. Local PEN is currently approximate

`track1/qa_episode.py`:

- `hand_vertex_ids()` selects a heuristic set by perturbing finger parameters.
- `signed_dist()` uses nearest surface samples and normals, not the scorer's winding-number test.
- Evaluation occurs on the episode OBJ, before the packer's mesh-budget transformation.

The kit exports `penetration_depth` and `episode_penetration`, but their existence does not make current QA exact. The full score requires the official hand roles and the hidden alignment scale.

`track1_vertex_indices.json` is referenced in kit documentation but was **not found during inspection**. Verify whether it can be obtained from an authorized release. Do not fabricate official roles or assume the heuristic matches them.

Use explicitly named outputs, for example:

```text
pen_approx_cm
pen_triangle_cm_on_selected_hand_points
pen_official_points_cm              # only if official indices are available
```

Any unaligned local result must be identified as such. Exact zero geometric penetration is unaffected by a positive alignment scale, but nonzero magnitudes still depend on that scale and the chosen hand points.

### B. Scored spans are not exact scored-frame lists

`track1/scored_spans.py` saves only `first`, `last`, and `n_scored`. Current QA and scale matching iterate over the complete first-to-last interval.

The sample was checked: episodes **0, 1, 2, 3, 4, 5, 12, 13, 14, and 20** have gaps in their scored frames.

Preserve context frames for inference, but derive evaluation from exact frame IDs in the sample. Do not bridge scored gaps in acceleration diagnostics. Match the kit's endpoint weighting as well as the second-difference formula.

### C. Stage 8 can undo stage 7's stability

`stage7_smooth.py` smooths translation, then `stage8_scale.py` can apply a time-varying multiplier to translation while retaining a global object-size factor.

```text
t_final[t] = position_scale[t] * t_smoothed[t]
```

Even constant `t_smoothed` moves if `position_scale` varies. For a moving trajectory, scale variation also changes velocity and acceleration.

The inspected choices were 8 `none`, 10 `global`, and 12 `per_frame` episodes. Preserve this baseline before testing alternatives.

Plan final motion refinement in the reconciled body/object scale, or optimize scale and motion jointly. Do not add a stationary lock before a later operation that moves the object again.

### D. Contact proxies are not universal supervision

The current scale objective uses a minimum hand-surface distance and approximate penetration. One nearby hand point does not establish a correct grasp.

- Episode 10 includes foot interaction.
- Stool episodes include sitting rather than continuous hand contact.
- A released object can continue moving while hands are far away.

Do not force every scored frame into hand contact. Use supported interaction states and preserve valid alternatives.

### E. Stage 7's outlier rejection is not absolute rejection

Outliers currently receive weight `1e-4`, not zero. Its rotation reference is derived from the raw sequence using component-wise quaternion medians and Slerp. Long incorrect pose segments can survive local-median checks and influence the reference.

Do not assume outlier filtering repairs a prolonged tracking failure. Test residual rotation smoothing on large turns, quaternion sign changes, and near-180° transitions before describing it as exact manifold optimization.

### F. Mesh-origin changes alter ACC-O, not necessarily for the better

For scene geometry:

```text
p = s R v + t
v_new = v - c
t_new = t + s R c
```

This preserves the posed mesh, but changes the trajectory whose acceleration the scorer compares. Reference-origin compatibility matters.

Several current submitted meshes already have bbox-center offsets of only approximately 0.3–8 mm before object scale. Re-centering is not the leading explanation for large tracking failures.

Vertex mean, bbox center, and physical center of mass are different. Keep origin changes as a separate ablation, with a strict posed-geometry invariance test.

### G. MHR fit error is not body accuracy

The approximately 1 mm fit error describes how closely the converter reproduces the predicted input meshes. It does not establish 1 mm error against the real human or hidden reference.

### H. Exact matching of all metrics is limited by hidden reference data

Self-acceleration magnitude is a jitter proxy, **not ACC-H/ACC-O error**. Silhouette/depth agreement against predictions used during reconstruction is a consistency check, not independent ground truth.

## 7. Proposed plan, in order

### Phase 0 — Establish a reproducible baseline

Tasks:

1. Inspect live Git status, running tmux sessions, available GPUs, model environment, and output state.
2. Read `docs/HANDOFF.md` and verify which artifacts belong to the baseline commit.
3. Preserve baseline episode NPZs/meshes, final poses, body fits, and QA reports before overwriting anything.
4. Record code revision, uncommitted changes, effective parameters, model/environment identity, and artifact hashes in an experiment manifest.
5. Inventory required files, dimensions, finite values, scale consistency, and local/video frame mapping across all 30 episodes using CPU-only reads first.

Acceptance:

- Baseline can be identified and restored without reconstructing models.
- No existing user work is discarded.
- Missing/stale artifacts are reported, not silently filled and called observed.
- GPU 1 is not used.

### Phase 1 — Build trustworthy evaluation and regression tests

Tasks:

1. Preserve exact scored frame IDs and contiguous stretches alongside span metadata; keep compatibility with existing readers.
2. Add scorer-compatible local acceleration diagnostics on those frames and the scorer's 22 joints, where the necessary joint mapping is available.
3. Integrate the kit's triangle-distance/winding kernel on legal MHR-generated hand points. Clearly identify whether points are heuristic or official.
4. Evaluate final budgeted/packed geometry. Prefer a round-trip from the submission representation for final checks.
5. Keep approximate metrics for speed, but label them and compare against the more faithful checks.
6. Make missing/non-finite QA values explicit failures or unknowns, not automatic passes. Ensure stage exit status propagates failed required validation.

Acceptance:

- CPU tests cover metric units, exact scored-frame selection, scored gaps, and endpoint weighting.
- Penetration fixtures cover inside/outside, face orientation reversal, nonzero scale, degenerate padding faces, and surface points.
- Final packed rows match the sample exactly, with finite values and valid rotations.
- Any inability to reproduce official point roles or alignment is documented as a blocker, not hidden.

### Phase 2 — Add bounded, legal penetration correction

Start with one affected episode, such as episode 16. Do not directly edit hand vertices: the host regenerates geometry from MHR parameters.

Initial approach:

- Keep body parameters and object rotation/scale fixed.
- Optimize a small object-translation correction in the final shared metric frame.
- Penalize penetration, correction magnitude, temporal variation, and reprojection disagreement.
- Preserve contact when it is supported; do not minimize penetration by moving the object away from the interaction.
- Use a configurable correction cap; around 3 cm is an experiment, not an established safe threshold.
- Recheck after smoothing/projection, since either operation can reintroduce penetration.

Acceptance:

- Zero correction is the baseline candidate.
- Faithful penetration checks improve on the targeted frames/points.
- No invalid geometry, scale changes, frame shifts, or non-finite parameters.
- Reprojection, contact, motion, and acceleration diagnostics remain within explicitly reviewed regression budgets.
- Unsolved opposing-hand constraints are reported. Do not claim that translation alone must solve them.

### Phase 3 — Test scale policy, then stationary-object handling

Run global versus current scale selection as an isolated ablation. Do not combine it immediately with several unrelated changes.

Then implement stationary intervals using visible feature/mask evidence, support state, sufficient visibility, and hysteresis. Fit one pose across each verified stationary interval.

Important constraints:

- A stable silhouette alone does not establish zero 3D motion; optical-axis motion and symmetric rotations can be weakly observable.
- Hands being far away does not prove stationarity.
- Stationarity must persist through final scale reconciliation and penetration correction, or those constraints must be solved jointly.
- Avoid pose snapping at lock/unlock boundaries.
- During moving-object occlusion use bounded prediction with increasing uncertainty, not an indefinite frozen pose presented as observed.

Acceptance:

- A synthetic stationary object remains fixed after the entire pipeline, including stage 8.
- Moving, released, and foot/body-contact cases are not incorrectly frozen.
- Lock/unlock transitions do not create unexplained velocity/acceleration spikes.
- Real starts, stops, rotations, and impacts remain visible in image-consistent trajectories.

### Phase 4 — Repair the demonstrated geometry/tracking failures

#### Hula hoop: episodes 0–2

- Use an explicit ring/torus primitive with dimensions shared across the three videos.
- Fit rendered silhouette or inner/outer boundaries over reliable frames, using perspective geometry.
- Treat ellipse axis-ratio tilt as an approximation, not a complete calibrated 3D solution.
- Keep normal-direction ambiguity until temporal/depth/support evidence resolves it.
- Monocular depth on thin structures can be unreliable; do not assume it resolves every ambiguous frame.
- Ring-axis rotation is unobservable for a perfectly symmetric torus without asymmetric texture/cues.

Do not promise IoU above 0.85 or a specific leaderboard improvement before measurement.

#### Episode 7: iron tracking

- Diagnose the lost interval with independent forward/backward tracks, visibility, reprojection, and interaction evidence.
- Track into the failure interval from reliable frames on both sides.
- Add only a small number of high-quality/reviewed hypotheses where needed.
- Preserve successful overrides, particularly episode 17.
- Do not blindly insert dense fresh registrations: prior adaptive anchoring increased flips.

#### Paint roller: episodes 18–20

- Measure missed **mask pixels**, not missed bbox area.
- Add thin-structure/boundary or skeleton-weighted coverage so the handle matters.
- Respect occlusion: absent observed pixels are not proof of missing geometry.
- If no candidate contains a handle, scoring cannot invent one; try a composite primitive or new geometry candidate.

Acceptance:

- Changed episodes have improved reviewed geometry/trajectory consistency, not just a larger average IoU.
- Unaffected episodes retain their previous outputs unless a shared dependency intentionally changed.
- All upstream mesh changes invalidate dependent pose, scale, and submission artifacts.

### Phase 5 — Body and research experiments

Only after the preceding correctness work:

- Test root translation smoothing separately from joint smoothing.
- Apply support-foot constraints only when contact is supported; do not pin both feet during stepping or hoop interactions.
- Check the first scored frame and shared-scale effects on the object.
- Test person-mask prompting for SAM3D Body on the debug set.
- Compare CARI4D using the same masks/meshes and evaluation protocol if environment access, licensing, weights, time, and compute permit.

Research attributions to CHOIR/PHOSA in prior discussion were not independently verified during the audit. Treat their general ideas as hypotheses, not validated implementation recipes or guaranteed gains.

## 8. Minimum test matrix

Write tests before changing core behavior. Keep most tests independent of model loading and CUDA.

| Test | Expected behavior |
|---|---|
| Local/video frame mapping | Round-trips for nonzero trim offsets; scored frames match original video IDs |
| Scored-frame gaps | No acceleration triple crosses an unscored gap |
| Acceleration formulas | Constant position/velocity have zero second differences; quadratic motion has the expected value |
| Units | Distinguish cm/frame² from m/s²; FPS factor is applied only for physical diagnostics |
| Metric weighting | Matches the kit's contiguous-stretch and endpoint aggregation |
| Pose conventions | Rotation matrices are orthonormal with determinant +1; quaternion ordering is explicit |
| Mesh re-centering | Before/after posed vertices agree after scale-aware translation compensation |
| Scale reconciliation | Global scaling preserves projection; per-frame scaling's motion effects are explicitly measured |
| Translation correction | Body remains valid MHR; corrected object remains rigid and bounded |
| Penetration kernel | Inside/outside, reversed faces, scaled meshes, degenerate padding, and boundary cases behave as documented |
| Stationary interval | Fixed object stays fixed after final scale and correction stages |
| Moving/released object | Real motion is not suppressed because hands are distant |
| Occlusion | Uncertainty increases; predictions are bounded and marked as inferred |
| Rotation smoothing | Handles quaternion sign flips, large turns, and near-180° cases without unsupported discontinuities |
| Packed geometry | Final geometry passes the same budget and schema path used for submission |
| Cache invalidation | Changed mesh/mask/calibration/configuration invalidates dependent artifacts |
| Failure behavior | Missing files, NaNs, corrupt inputs, and failed QA cannot return a successful required stage |
| Rerun behavior | Unchanged runs reuse validated outputs; no stale review sheets or mixed experiment versions |

Use the official kit as an oracle for formulas, not the current QA implementation. Never make tests pass by changing the official scorer.

## 9. Motion and acceleration safeguards

For constant sampling interval:

```text
delta2_p[t] = p[t+1] - 2*p[t] + p[t-1]
challenge-style quantity = 100 * delta2_p       [cm/frame², positions in metres]
physical acceleration = delta2_p / dt²         [m/s²]
```

The challenge evaluates **error against reference acceleration**, not low self-acceleration. A frozen but wrong trajectory can have perfect smoothness and bad reconstruction accuracy.

- Preserve timestamps and actual frame gaps.
- Reject tracking resets before differentiating positions.
- Evaluate acceleration after every operation that changes translation, including scale and penetration correction.
- Use motion-aware regularization; do not assume stronger smoothing is always better.
- Inspect final rotation and origin conventions together, since changing origin changes translation dynamics.

## 10. Regression subset and reporting

Use a small targeted subset before running all episodes:

| Episodes | Reason |
|---|---|
| 0–2 | Failed thin-ring geometry; scored-frame gaps |
| 3 or 4 | Good bowl behavior to protect; scored-frame selection |
| 6, 7, 8 | Iron baseline versus prolonged tracking failure |
| 10 | Foot interaction; catches incorrect hand-only assumptions |
| 16, 17 | Penetration and a successful tracking override |
| 18 or 19 | Roller handle completeness |
| 24 or 26 | Sitting/support interaction, not continuous hand grasp |
| 28 | Cart orientation and geometry consistency |

Not every experiment needs this entire subset. Select episodes relevant to the changed behavior, then broaden the regression run before promotion.

For each variant report baseline and candidate values for:

- Required scored-frame coverage and schema validity.
- Clearly labeled penetration measure and hand-point definition.
- Median/tail contact gaps with supported contact frames identified.
- Object silhouette/boundary agreement and valid-depth residuals.
- Stationary drift, recovery jumps, and prediction/interpolation fraction.
- Object and 22-joint self-acceleration proxies on exact scored stretches.
- Maximum position/rotation corrections and confidence states.
- First-scored-frame body reprojection/support consistency.
- Runtime, peak RAM/VRAM, and failed/retried stages when GPU work is used.

Review all affected temporal intervals. A six-frame sheet can miss a long tracking failure; include trajectories and short clips around difficult transitions.

Thresholds must be chosen and documented before promotion. Hard structural tests can use explicit numeric tolerances; quality budgets require reviewed validation evidence. Do not silently promote a change merely because one proxy improved.

## 11. Experiment and submission discipline

1. One conceptual change per experiment whenever practical.
2. Keep the baseline immutable and identify every candidate by code/config/artifact hashes.
3. Preserve complete run provenance, including uncommitted dependency changes under `/workspace` if any.
4. Do not automatically pick each episode's best variant from repeated leaderboard feedback without documenting the selection methodology.
5. Before any upload, validate the final packed Parquet and review changed episodes.
6. Upload only with user authorization. The same Parquet goes to all five Track 1 competitions.
7. Record the actual commit URL, Parquet SHA-256, configuration, changed episodes, local diagnostics, and all five scores together.
8. Confirm Kaggle's current submission limits rather than assuming them.

Historical baseline reported in `docs/HANDOFF.md`:

| Metric | Score, lower is better |
|---|---:|
| CD-H | 15.716 cm |
| CD-O | 37.420 cm |
| ACC-H | 0.174 cm/frame² |
| ACC-O | 0.127 cm/frame² |
| PEN | 0.1007 cm |

These are historical user-provided results, not newly verified live leaderboard values.

## 12. Performance work comes after correctness

Possible later optimizations:

- Load/reuse frame data rather than repeatedly decoding it.
- Batch pose hypotheses and reduce CPU/GPU transfer in scoring.
- Reuse validated registration artifacts across mesh-check stages.
- Investigate bypassing 5d where 5e supersedes it, while retaining required candidate/check artifacts and reviews.
- Compare warm-started scale refinement with independent registration.

Do not label these quality-neutral without regression tests. Warm starts can retain incorrect local minima, and changed batch/precision behavior can alter outputs.

## 13. What the next agent should deliver

Before implementation, return a concise live-state check and identify blockers. Then work through Phase 0 and Phase 1 first unless the user explicitly changes priorities.

After each implementation increment, report:

```text
What changed and why
Files changed
Tests executed and results
What was not tested
Baseline-versus-candidate diagnostics
Known limitations / missing official assets
Artifacts and review locations
Rollback path
Recommended next step
```

### Completion standard

"Works as expected" means:

- Required artifacts are valid and frame-aligned.
- Evaluation semantics and proxy limitations are explicit.
- Outputs are reproducible and traceable to code/configuration/model identities.
- Stationary objects do not drift through the final pipeline.
- Real motion is not hidden by freezing or excessive smoothing.
- Corrections stay within legal MHR/object representations.
- Penetration, reprojection, contact, and acceleration are checked together.
- Uncertain or failed results are visible rather than silently accepted.
- Successful baseline episodes do not regress unnoticed.

**Start with trustworthy evaluation and regression protection, not another large model run.**
