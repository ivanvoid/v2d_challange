# V2D Track 1: Pipeline Review, Weak Points, and Model Recommendations

## Overall assessment

**The pipeline is now a complete Track 1 reconstruction workflow. The biggest opportunity is not replacing every model—it is making geometry, tracking, scale, and contact agree with each other.**

This review covers the active stages, orchestration, model adapters, submission path, current diagnostics, and the newer evaluation/penetration drafts. Official documentation for several model alternatives was also checked.

The analysis was read-only: **no pipeline files changed and no GPU inference was launched**. This document records that analysis; results and working-tree observations below are a snapshot from the inspection, not a new benchmark.

All code paths below are relative to `/local-data/ivan/v2d/v2d_challange` unless stated otherwise.

---

## 1. Current pipeline, summarized

```text
Video + official scored frames
        │
        ▼
1. Trim → AnyCalib undistortion → MoGe depth → fixed intrinsics
        │
        ├──────────────────────────────────┐
        ▼                                  │
2. Person detection/tracking                │
   → SAM3D Body                            │
   → shared-identity MHR fit                │
        │                                  │
        ▼                                  │
3. Actor-aware object detection ◄───────────┘
   → prompt aliases, mirror/zoom, depth/size gates
        │
        ▼
4. SAM2 object masks
   → negative body prompts
   → re-detection and leak repair
        │
        ▼
5. SAM3D object candidates
   → metric scale search
   → cross-frame checks
        │
        ▼
5e. Select one geometry across the object's three videos
        │
        ▼
6. FoundationPose
   → multiple anchors and orientations
   → forward/backward tracking
   → Viterbi trajectory selection
        │
        ▼
7. Outlier-weighted temporal smoothing
        │
        ▼
8. Body/object scale reconciliation
   → episode arrays → QA → Parquet
```

### What is already good

- Full-body output matches the required **MHR representation**.
- A shared body identity avoids frame-by-frame shape changes.
- Object selection is much stronger than a single text detection.
- Sharing geometry across three videos is sensible.
- Bidirectional tracking and orientation hypotheses address real FoundationPose failures.
- Correct handling of FoundationPose’s centered-mesh convention is important.
- Per-stage review pages make failures inspectable.
- The new local evaluator reads **packed geometry**, which is a useful improvement.

Preserve these foundations.

---

## 2. Where the results are weakest

### Current saved tracking diagnostics

These are diagnostics over trimmed clips, **not ground-truth accuracy measurements**.

| Episode | Object | Median mask IoU | Frames with IoU < 0.5 | Interpretation |
|---:|---|---:|---:|---|
| 0 | Hoop | 0.471 | 348 / 630 | Geometry/tracking unreliable |
| 1 | Hoop | **0.118** | **330 / 331** | Almost complete tracking failure |
| 2 | Hoop | 0.317 | 510 / 571 | Geometry/tracking unreliable |
| 3 | Bowl | 0.915 | 0 / 432 | Strong baseline to protect |
| 6 | Iron | 0.794 | 8 / 656 | Mostly consistent |
| 7 | Iron | 0.654 | **290 / 617** | Long failure interval |
| 17 | Foam block | 0.857 | 0 / 297 | Strong tracking baseline |
| 26 | Stool | 0.667 | 13 / 301 | Usable, with geometry/visibility limitations |

**Priority:** fix the hoop and episode 7 before spending substantial compute improving already-good bowls.

Smoothing diagnostics reinforce this:

- Episode 2: maximum translation correction of approximately **101 cm**.
- Episode 7: approximately **64 cm**.
- Hoop rotation corrections approach **180°**.

Those are not ordinary jitter corrections. They indicate that smoothing is compensating for seriously inconsistent input poses.

---

## 3. Main weak parts and concrete improvements

### A. Too many stages optimize different objectives

**Files:** `track1/stage5_object.py`, `track1/stage6_track.py`, `track1/stage7_smooth.py`, `track1/stage8_scale.py`

Currently:

- Mesh selection prioritizes silhouette IoU.
- Tracking uses IoU minus colour error.
- Smoothing minimizes deviation and second differences.
- Scale selection prioritizes hand distance and approximate penetration.

A trajectory can improve one objective while becoming worse under another.

**Improvement:** add a final shared refinement that evaluates:

- Reprojection.
- Reliable depth.
- Object rigidity and fixed dimensions.
- Supported human–object contact.
- Penetration.
- Temporal consistency.
- Stationary intervals.

Start by optimizing **small object-pose corrections**, keeping body identity and mesh geometry fixed. Expand the optimization only when the simpler version is validated.

### B. Stage 8 can undo stabilization

**File:** `track1/stage8_scale.py`

Stage 7 smooths translation, then stage 8 may multiply it by a time-varying scale.

```text
stationary smoothed position × varying scale = moving final position
```

Also, only the global variant uniformly scales both geometry and position. The per-frame variant changes position with a varying factor while retaining a global size factor; it does not generally preserve image projection.

**Improvement:**

```text
scale reconciliation
→ final stationary/contact-aware refinement
→ final acceleration and reprojection checks
```

Compare global scaling against the current policy as an isolated experiment. Do not assume either is universally correct.

### C. Generic mesh generation is the wrong tool for some objects

**Files:** `track1/stage5_mesh.py`, `track1/stage5_object.py`

SAM3D is useful for many objects, but the current results show predictable failures:

- Hoop topology.
- Thin roller handles.
- Occluded components.
- Symmetric geometry.

**Improvement: introduce a geometry router.**

| Object | Preferred geometry approach |
|---|---|
| Hoop | Parametric torus/ring |
| Foam roll | Cylinder/capsule candidate |
| Foam block | Box candidate |
| Paint roller | Composite handle, bent connector, roller |
| Bowl | SAM3D versus a bowl-shaped parametric candidate |
| Complex desk/cart | SAM3D with multi-frame completeness checks |

These primitives should be **additional hypotheses selected against observations**, not mandatory shapes that override the images.

For the hoop, share dimensions across all three videos and fit perspective projections over time. Do not expect monocular depth to resolve every orientation ambiguity.

### D. Tracking confidence is too dependent on silhouette and generated colour

**File:** `track1/stage6_track.py:174–189`

The tracking score is:

```text
fit = IoU − colour_error / 255
```

Depth error is calculated but does not directly influence that score.

Problems:

- A wrong orientation may have the same silhouette.
- Generated texture can be wrong on unseen surfaces.
- Specular highlights change on the iron.
- A plausible silhouette can coexist with incorrect depth.
- Full-mesh rendering is compared with partially occluded observed masks.

**Improvement:** combine visibility-aware silhouette, boundaries, reliable depth, tracked points, and temporal agreement.

Do not use depth blindly either—weight it by validity and model disagreement.

### E. Temporal repair cannot recover a missing correct hypothesis

**Files:** `track1/stage6_track.py`, `track1/stage7_smooth.py`

Viterbi selects the best sequence **among available candidates**. If all candidates are wrong for 200 frames, sequence selection cannot invent the correct track.

Likewise, a local median can accept a long, consistently wrong interval.

**Improvement:**

1. Detect prolonged low-confidence intervals.
2. Recover from reliable observations on both sides.
3. Add a small number of useful hypotheses.
4. Re-run selection.
5. Preserve uncertainty where recovery remains ambiguous.

Avoid blindly increasing anchor density; existing experiments showed that this can increase orientation flips.

### F. Body verification is not sufficiently independent

**Files:** `track1/fit_mhr.py`, `track1/stage3_select.py`, `track1/stage4_prompts.py`

SAM3D Body supplies:

- The body estimate.
- Wrist locations used for object detection.
- Negative prompts for masks.
- Body occlusion evidence.
- Hand geometry for scale selection.

One body error can therefore reinforce itself across several stages.

**Improvement:** add an independent 2D whole-body estimator to check wrists, feet, elbows, and the target person. Use disagreements to reduce confidence or trigger review.

The ~1 mm MHR fitting error measures reproduction of the predicted meshes—not accuracy against the real person.

### G. Mask repairs can erase genuine motion

**File:** `track1/stage4_prompts.py`

Area-based leak repair intersects a suspicious mask with dilated neighboring masks.

That can help transient leakage, but a legitimate rotation or fast movement can also produce a large area or position change.

Colour-based checks can similarly reject genuine changes of visible surface.

**Improvement:**

- Motion-compensate neighboring masks before comparison.
- Use point/flow evidence.
- Separate “occluded,” “moving quickly,” and “leaking.”
- Keep raw and repaired masks with provenance.
- Evaluate thin boundaries and handles, not just area.

### H. Evaluation still has limitations

The new `track1/eval_local.py` is a worthwhile step: it uses packed geometry and the official penetration kernel. Its documentation correctly acknowledges approximate hand-point selection and unknown alignment scale.

However:

- Standard QA still uses approximate surface-normal distances.
- Some calculations use every frame between first and last scored frames.
- Self-acceleration is not reference acceleration error.
- Official hand-point roles remain unavailable in the inspected files.

**Improvement:** keep explicit metric names and exact scored-frame lists. Separate:

```text
structural validation
geometric consistency
approximate quality proxies
actual leaderboard measurements
```

A local PEN proxy around **0.0966 cm** is reasonably close to the historical **0.1007 cm** leaderboard value, but that does not prove per-episode equivalence.

---

## 4. New models worth trying

### 1. SAM 3.1 — detection and segmentation challenger

**Where:** stages 3–4.

[SAM 3 / 3.1](https://github.com/facebookresearch/sam3) supports text/visual prompting, detection, segmentation, and video tracking. SAM 3.1 is an available newer checkpoint family.

**Why it fits:**

- Could reduce the prompt-ladder/re-detection complexity.
- Could improve detection-to-segmentation consistency.
- Could produce a useful independent mask proposal.

**Suggested integration:**

- Run alongside—not immediately replace—Grounding DINO + SAM2.
- Retain actor-aware instance selection.
- Compare on hoops, rollers, and similar background furniture.
- Route only validated improvements into reconstruction.

**Cautions:** concept matching does not identify the manipulated instance automatically. It also needs a separate, newer environment and gated checkpoints. Its multi-object efficiency improvements may not translate into a major gain for these single-object episodes.

**SAM 3.1 is not SAM3D:** this is a segmentation alternative, not a mesh generator.

### 2. CoTracker3 — temporal point evidence

**Where:** stage 6 and stationary detection.

[CoTracker3](https://github.com/facebookresearch/co-tracker) provides point trajectories and visibility.

**Useful roles:**

- Identify stationary intervals.
- Detect mask/pose drift.
- Check forward/backward agreement.
- Help bridge episode 7’s difficult turn.
- Constrain pose using visible object features.

At reliable anchors, mesh/depth associations can provide object-space points for robust PnP-style constraints.

**Cautions:** it produces 2D tracks, not metric 6D poses. Reflective, tiny, occluded, and symmetric objects remain challenging. The main repository license is noncommercial; check competition and deployment compatibility.

### 3. RTMW or RTMPose whole-body — independent body observations

**Where:** body QA, detection guidance, contact-state estimation.

[RTMPose/RTMW](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) provides body or whole-body 2D keypoints.

**Useful roles:**

- Verify anatomical wrist identity.
- Check foot support.
- Detect target-person errors.
- Add independent keypoint reprojection constraints to MHR fitting.

**Caution:** explicitly map its keypoints to MHR conventions. It is not a replacement for full-body MHR reconstruction, and tiny hand keypoints should remain confidence-gated.

### 4. Depth Anything 3 Metric — alternative depth branch

**Where:** stage 1, initially as a diagnostic comparison.

[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) has a monocular metric variant and multi-view/nested variants.

**First choice here:** `DA3METRIC-LARGE`, conditioned on validated camera geometry.

Compare against MoGe on:

- Body/object depth-scale disagreement.
- Temporal stability in background regions.
- Thin structures and object boundaries.
- Downstream pose consistency.

For multi-view/nested experiments, be careful: the camera is static while the person and object move. Ordinary static-scene assumptions do not automatically apply.

**Important:** the official repository now advises refreshed `-1.1` checkpoints for affected any-view/nested models. Verify wrapper compatibility rather than assuming older defaults are current.

Licensing differs by checkpoint; the metric model is listed as Apache 2.0, while several larger variants are noncommercial.

### 5. UniDepth V2 — depth and confidence cross-check

**Where:** stage 1.

[UniDepth V2](https://github.com/lpiccinelli-eth/UniDepth) supports metric depth, camera conditioning, and confidence output.

Its main value would be **disagreement detection**, not indiscriminate averaging:

```text
MoGe and UniDepth disagree strongly
→ reduce depth trust
→ rely more on image/temporal constraints
→ flag ambiguous geometry
```

A local wrapper already exists under `/workspace/v2d_unidepth`, but it needs validation for camera conditioning, confidence export, and frame handling.

**Caution:** noncommercial licensing and additional compute.

### 6. CARI4D / CoCoNet — joint HOI experiment

**Where:** a separate reconstruction/refinement branch.

The local implementation under `/workspace/v2d_cari4d` already combines:

- SAM3D Body.
- MoGe.
- FoundationPose.
- Learned temporal human–object refinement.
- Contact-guided optimization.

This is the most directly relevant alternative to the current **decoupled body/object composition**.

**Recommended experiment:** compare on one good episode and several failures using the same validated masks and meshes.

**Cautions:**

- Source availability does not mean the runtime and weights are installed.
- It shares initializers with the current pipeline, so errors may be correlated.
- It cannot reliably rescue a hoop with fundamentally wrong geometry.
- Do not assume its refinement is a drop-in function without checking its input/output contracts.

---

## 5. Correctness and engineering fixes before model changes

| Priority | Finding | Recommended change |
|---|---|---|
| **P0** | `run_stage.sh` can continue after QA/render failure and print stage success | Explicitly propagate every required failure |
| **P0** | `run_rest.sh` does not reliably aggregate child-job failures | Capture each PID and exit status; stop dependent work |
| **P0** | Two-GPU launcher kills existing named tmux sessions | Refuse collisions or use unique run IDs; never silently kill active work |
| **P0** | Reuse relies on timestamps/file existence | Include code, configuration, model, calibration, mask, and mesh identities |
| **P1** | Packing can nearest-fill missing poses | Reject missing scored poses or explicitly classify an approved fallback |
| **P1** | Final pose files lose observation confidence | Carry observed/tracked/interpolated/uncertain states into QA |
| **P1** | SAM2 buffers all full-resolution masks | Stream output masks or use bounded storage |
| **P1** | No repository test suite found | Add CPU contract tests and a small GPU regression set |
| **P1** | New penetration draft has incremental rather than guaranteed total shift bounds | Test the final total correction cap and acceleration after residual repair |

At inspection, `track1/eval_local.py` and `track1/stage8_depen.py` were untracked drafts and were not wired into the standard stage-8 runner. Distinguish their proposed behavior from the deployed path.

Syntax was checked without running the pipeline:

- **40 Python files parsed successfully.**
- **Three active shell scripts passed `bash -n`.**
- **No repository test suite was found in the inspected test locations.**

Syntax success is useful, but it does not validate behavior.

---

## 6. Recommended architectural direction

```text
                       Shared episode state
                 camera + timing + units + identity
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
       Body proposals    Object geometry    Image evidence
       SAM3D Body        SAM3D/primitives   masks/keypoints/tracks
             └────────────────┼────────────────┘
                              ▼
                Joint confidence-aware refinement
                ├── Reprojection
                ├── Reliable depth
                ├── Contact / support
                ├── Penetration
                ├── Stationary intervals
                └── Motion consistency
                              ▼
                  Packed-output validation
```

Keep model proposals separate from accepted estimates. Preserve competing hypotheses and uncertainty instead of silently forcing a single answer early.

**No model is a source of truth by itself.** The strongest evidence here is reviewed image correspondence, known camera/timing contracts, supported geometry, and agreement across independent observations.

---

## 7. Recommended next steps

### First: protect correctness

- Build exact scored-frame and packed-output checks.
- Fix stage failure propagation.
- Preserve baseline artifacts and experiment identities.
- Validate the new penetration draft before integrating it.

### Second: fix the demonstrated failures

- Parametric hoop geometry.
- Localized episode 7 recovery.
- Roller-handle completeness.
- Final-scale stationary constraints.

### Third: introduce complementary models

Preferred first experiments:

1. **CoTracker3** for temporal evidence.
2. **SAM 3.1** as a segmentation challenger.
3. **RTMW** for independent body/keypoint checks.

Then compare **DA3 Metric / UniDepth V2** for depth and **CARI4D** for joint refinement.

### Bottom line

**Keep the current backbone, repair the known geometry/tracking failures, and add independent evidence rather than more post-processing.** That is more likely to improve correctness than replacing the entire stack or increasing smoothing strength.
