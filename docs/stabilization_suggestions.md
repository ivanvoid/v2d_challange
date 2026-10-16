# Object Stabilization and Motion-Accuracy Suggestions

## 1. Objective

Make the reconstruction stable **without sacrificing correct object identity, geometry, motion, or acceleration**.

The key distinctions are:

- **Correctness:** the reconstruction represents the intended object in the correct coordinate system and scale.
- **Stability:** stationary objects do not drift, and moving objects do not exhibit unsupported jitter or jumps.
- **Motion accuracy:** genuine starts, stops, rotations, flips, and impacts remain represented.
- **Acceleration accuracy:** changes in velocity are temporally and geometrically consistent with the observations and, where available, independent reference measurements.

> Smoothing removes shaking. Anchoring prevents drift. Neither guarantees correctness on its own.

These recommendations consolidate the preceding discussion and the findings in `report.md`. They are an implementation proposal, not a claim that the suggested changes have already been implemented or benchmarked.

## 2. Why static objects can appear to move

The current pipeline estimates object poses frame by frame and then smooths them. Each new estimate can be slightly different even when the physical object is stationary.

Common mechanisms include:

```text
Stationary object
    → monocular depth or depth scale changes
    → pose estimator changes object translation
    → output object drifts
```

```text
Stationary object
    → segmentation deteriorates or object becomes occluded
    → mask IoU falls
    → tracker re-registers
    → replacement pose jumps
```

Other causes include inconsistent camera calibration, incorrect target selection, object symmetries, coordinate mistakes, stale cached artifacts, and changing object geometry or scale.

The existing error-state Kalman filter and RTS smoother can make a drifting trajectory smoother. They do not explicitly establish that a stationary object must retain one fixed pose.

### Correct-target prerequisite

The audit found that episode 9's saved object mask selects a **background desk**, rather than the foreground desk being manipulated:

- `out/episode_000009/frames/000207.png`
- `out/episode_000009/masks/1/000207.png`

Anchoring that prediction would make it stable but still wrong. Verify the target before applying stabilization.

## 3. Top three recommendations

### 1. Lock the pose during verified stationary intervals

Add stationary and moving tracking states. When multiple reliable observations support that the object is stationary, fit one pose and use it throughout that stationary interval.

Unlock only after sustained motion evidence. Use separate entry and exit thresholds so the state does not flicker.

**Expected impact:** prevents static drift instead of merely reducing its speed.

### 2. Stabilize camera calibration and depth scale

For the documented static-camera videos, use a validated fixed camera model. Estimate monocular depth-scale drift from static background regions, excluding the person and manipulated object.

Apply corrections only when supported by sufficient valid evidence. Do not let noisy depth updates move an otherwise stationary object.

**Expected impact:** removes a major source of artificial 3D motion.

### 3. Gate pose updates and re-registration

Do not replace a stable pose because one frame has poor mask IoU. Require adequate visibility, valid depth, multi-frame agreement, and an improvement over the current hypothesis.

During occlusion, retain a verified stationary anchor with explicit uncertainty. For moving objects, predict only for a bounded duration before declaring the estimate unreliable.

**Expected impact:** prevents sudden jumps and repeated tracking resets.

## 4. Ten detailed improvements

### 1. Verify the manipulated object before tracking

Do not automatically accept the highest Grounding DINO score.

- Compare candidates across several frames.
- Prefer the object interacting with the target person's hands, while accounting for resting intervals with no contact.
- Use target-person proximity, visible interaction, and temporal continuity as evidence.
- Reject ambiguous candidates or request a reviewed seed mask.
- Render the object mask, not only hand tracks, in verification overlays.

**Benefit:** prevents a wrong-object error from contaminating every downstream stage.

**Relevant implementation:** `/workspace/v2d_pipelines/run_ego_wilor.py`, object detection and verification-render stages.

### 2. Select initialization frames by quality

Replace the median-hand-frame heuristic in `pick_reference_frame.py`.

Rank frames using:

- Object visibility and visible extent.
- Image sharpness.
- Segmentation quality.
- Valid depth coverage.
- Occlusion level.
- Hand/body visibility when those branches need initialization.

Choose several strong candidates and evaluate their initializations against neighboring frames. Object and hand initialization do not necessarily need the same frame.

**Benefit:** reduces incorrect mesh generation, scale estimates, and initial poses.

### 3. Use one consistent camera, coordinate, and timing contract

For these static-camera videos, estimate and validate a fixed camera model rather than allowing inconsistent calibration across stages.

Every artifact should explicitly identify:

- Original versus undistorted image coordinates.
- Camera or world coordinate frame.
- Units.
- Quaternion ordering.
- Source frame ID and timestamp.
- Calibration and preprocessing identity.

Add projection and transform round-trip tests. Ensure the intrinsics match the exact image geometry consumed by the stage.

**Benefit:** eliminates systematic misalignment that no filter can repair.

**Relevant implementations:** `/workspace/v2d_common/datatypes.py`, calibration/depth adapters, and `render_results.py`.

### 4. Stabilize depth scale without flattening real motion

Independent MoGe predictions can introduce apparent motion through changes in estimated depth scale.

- Measure drift using verified static background regions.
- Exclude the person and manipulated object from background scale estimation.
- Use robust statistics or fitting with invalid-pixel and outlier rejection.
- Apply scale correction only when background evidence supports it.
- Preserve genuine foreground depth changes.
- Do not independently rescale the object every frame.

**Benefit:** reduces artificial movement along the camera's depth axis.

**Caution:** stationary image features alone do not prove zero 3D motion. Motion along the optical axis and some rotations can be weakly observable from one view.

### 5. Keep object geometry and scale fixed throughout the episode

A rigid object should not change shape or size frame to frame.

- Validate one canonical mesh and preserve its origin.
- Estimate one episode-level scale using several reliable observations.
- Keep mesh, scale, and pose conventions consistent.
- Refine geometry offline if needed, then rerun tracking with the finalized mesh.
- Keep human and object geometry in a shared, consistent scale system.

**Benefit:** prevents pose estimation from compensating for inconsistent geometry or scale.

### 6. Make tracking recovery evidence-based

FoundationPose's mask-IoU recovery is useful, but silhouette agreement alone cannot establish a correct 3D pose.

Combine:

- Mask agreement and visibility.
- Valid-depth residuals.
- Visible feature or point continuity.
- Translation and rotation plausibility.
- Support/contact evidence, where reliable.

When confidence falls, test multiple pose hypotheses. Accept re-registration only when it improves the evidence. Account for object symmetries rather than treating every orientation change as real motion.

**Benefit:** reduces drift and abrupt recovery jumps.

**Relevant implementations:** `/workspace/v2d_foundation_pose/lib/run_video_to_poses.py` and `/workspace/v2d_foundation_pose/lib/foundation_pose_tracker.py`.

### 7. Use adaptive, time-aware SE(3) smoothing

Keep rotation operations on SO(3). Do not independently average Euler angles or quaternion components.

Improve the existing filter/smoother by:

- Using timestamps and actual frame gaps.
- Increasing measurement uncertainty for unreliable observations.
- Allowing greater motion during fast actions.
- Rejecting outliers before smoothing.
- Incorporating verified stationary intervals.
- Preserving raw poses and quality diagnostics for comparison.

**Benefit:** reduces jitter without excessively flattening flips, impacts, or other genuine motion.

**Relevant implementation:** `/workspace/v2d_foundation_pose/lib/run_ekf_smoothing.py`.

### 8. Stabilize the human and hand tracks too

Stable objects can still look wrong beside jittering hands or an unstable body.

- Add persistent target-person tracking and full-body MHR estimation.
- Keep body identity parameters stable across the episode.
- Associate hands using anatomical body wrists and temporal evidence, not image-left/image-right heuristics.
- Bound interpolation and extrapolation by time and confidence.
- Label synthesized frames explicitly.

**Benefit:** produces stable human geometry and prevents hand identity switches.

**Relevant implementations:** `/workspace/v2d_wilor/lib/tracks_from_wilor_masks.py`, `/workspace/v2d_wilor/lib/tracks_interpolate.py`, and the SAM3D Body/CARI4D branches.

### 9. Refine human–object relationships jointly

Use the human and object in the same coordinate system and shared scale.

Apply confidence-gated constraints for:

- Hand–object contact.
- Excessive penetration.
- Floor/table support.
- Foot sliding.
- Relative motion during a grasp.

Activate contact constraints only when supported by observations. Proximity is not proof of contact, and a stable grasp does not mean the object is stationary in the world.

**Benefit:** makes interaction physically plausible rather than merely visually smooth.

### 10. Add automatic acceptance checks

An episode should not pass simply because pose files exist.

Check:

- Required-frame coverage and timestamp alignment.
- Finite values and valid rotations.
- Fixed object scale.
- Correct target identity.
- Reprojection and depth residuals.
- Unexplained pose, velocity, or acceleration jumps.
- Stationary drift.
- Observation provenance and confidence.

Export states such as `observed`, `tracked`, `interpolated`, `extrapolated`, `uncertain`, and `rejected`.

Compare raw and stabilized results. Lower jitter must not come at the expense of worse image alignment or lost genuine motion.

**Benefit:** prevents incomplete or smooth-but-wrong outputs from being accepted.

## 5. Stationary-object strategy

### 5.1 State machine

```text
STATIC
  → retain one anchored pose
  → switch to MOVING only after sustained, reliable motion evidence

MOVING
  → estimate pose using time-aware motion dynamics
  → return to STATIC after sustained stationary evidence

UNRELIABLE / OCCLUDED
  → avoid accepting unsupported pose updates
  → use bounded prediction or a previously verified stationary anchor
  → preserve uncertainty until reliable visibility returns
```

Use hysteresis: the evidence required to enter a stationary state should differ from the evidence required to leave it. State decisions should consider multiple frames and observation quality.

Do not use estimated 3D translation alone to detect motion; that estimate is itself a source of drift.

### 5.2 Fit one pose across a stationary interval

Instead of smoothing a collection of drifting poses, jointly fit one rotation and translation against several reliable frames using robust losses on:

- Object silhouettes.
- Valid depth pixels.
- Visible feature correspondences.
- Known support geometry, where available.

Assign that pose to the verified stationary interval. Keep the anchor's confidence and supporting frame IDs in the output.

This provides an explicit stability constraint rather than merely reducing drift speed.

### 5.3 Separate background objects from manipulated objects

- **Verified background object:** reconstruct and anchor once, subject to a validated static camera/world frame.
- **Manipulated object:** anchor only during resting intervals; preserve real movement during manipulation.
- **Ambiguous target or visibility:** reject or mark uncertain instead of forcing a stable output.

For a moving camera, a stationary world object changes pose in camera coordinates. Stabilization must therefore operate in a validated world frame. For the documented static-camera dataset, adding SLAM by default is not necessary; begin with a validated fixed camera.

### 5.4 Avoid transition artifacts

Hard switching can introduce a discontinuity when a pose is locked or unlocked.

- Anchor to a pose consistent with observations and the trajectory boundary.
- Initialize the moving model from the anchored position and appropriate uncertainty.
- Use dynamics-aware transition handling rather than arbitrary pose snapping.
- Preserve genuine impacts; do not automatically smooth every abrupt change away.

## 6. Stabilization algorithms

These algorithms stabilize **3D object poses**, rather than merely stabilizing image motion.

| Algorithm | What it solves | Limitation |
|---|---|---|
| Stationary detection + pose anchoring | Holds one fixed pose while an object is stationary | Requires reliable stationary/moving classification |
| Interacting Multiple Model filter (IMM) | Switches between stationary and moving motion models | Drifting predictions alone can be mistaken for motion; incorporate visual evidence |
| SE(3) Kalman filtering + RTS smoothing | Reduces translation/rotation jitter using temporal evidence | Can smooth drift without eliminating it |
| Robust multi-frame pose optimization | Fits consistent geometry using silhouettes, depth, and feature tracks | More computation and implementation effort; systematic input errors still matter |

Huber or other robust losses reduce the influence of bad residuals. RANSAC is useful when the chosen feature/correspondence formulation supports it; it is not a universal replacement for pose optimization.

### Recommended combination

1. **Stationary detection with a fixed-pose anchor** to address static drift.
2. **IMM stationary and moving models** to handle transitions between rest and manipulation.
3. **Robust, gated SE(3) filtering and offline smoothing** to handle observation noise and motion dynamics.

The pipeline already has an ESKF + RTS smoother in `/workspace/v2d_foundation_pose/lib/run_ekf_smoothing.py`. Improve its observation gating, timing, and stationary handling before adding another generic smoothing filter.

## 7. Acceleration correctness

### 7.1 Smooth motion is not necessarily correct acceleration

Excessive smoothing can erase genuine acceleration. Small position errors can create large false acceleration. A stationary but incorrect reconstruction can have perfectly zero acceleration.

Therefore, acceleration must be evaluated together with object identity, reprojection, geometry, and observation quality.

### 7.2 Estimate position, velocity, and acceleration together

A translational motion state can contain:

```text
State = position + velocity + acceleration
```

Use different dynamics according to reliable evidence:

- **Stationary:** anchored position, zero velocity and acceleration.
- **Moving:** constant-velocity and/or constant-acceleration models that permit genuine motion.
- **Occluded:** bounded prediction with increasing uncertainty.

An IMM filter followed by offline smoothing is suitable for this design. Model assumptions are priors, not proof of the physical acceleration.

For rotation, estimate angular motion on SO(3), using relative rotations and a consistent reference frame. Do not obtain angular acceleration by differentiating wrapped Euler angles.

### 7.3 Use correct timestamps

For constant frame spacing:

```text
a_t = (p_(t+1) - 2 p_t + p_(t-1)) / Δt²
```

Here, positions in metres and time in seconds produce acceleration in `m/s²`.

The dataset is documented as 30 FPS, so `Δt = 1/30` second when frames are consecutive and the timing contract is satisfied.

For unequal time gaps, a local second-derivative estimate is:

```text
h_prev = t_t - t_(t-1)
h_next = t_(t+1) - t_t

a_t ≈ 2 / (h_prev + h_next)
      × [(p_(t+1) - p_t) / h_next
         - (p_t - p_(t-1)) / h_prev]
```

This estimate is exact for quadratic position trajectories and is an approximation for general motion. Prefer a timestamp-aware state estimator when observations are noisy or gaps are substantial.

Missing frames must not be treated as consecutive observations.

### 7.4 Separate physical acceleration from the challenge metric

Track 1's ACC-O metric compares object-translation second frame differences in **cm/frame²**, not physical acceleration in **m/s²**. ACC-H similarly evaluates body-joint second differences.

For consecutive frames and positions stored in metres:

```text
Second frame difference:
    δ²p_t = p_(t+1) - 2 p_t + p_(t-1)

Challenge-style quantity:
    100 × δ²p_t                     [cm/frame²]

Physical acceleration at FPS f:
    f² × δ²p_t                      [m/s²]
```

The challenge score is an **error against the reference**, not a preference for low acceleration magnitude. Follow the scorer's exact frame-consecutiveness and averaging rules rather than substituting a generic physical-acceleration calculation.

Relevant reference: `v2d_submission_kit/tracks/track_1/README.md` and `v2d_submission_kit/v2dlb/mhr_submission.py`.

### 7.5 Reject jumps before they become acceleration spikes

Common sources of false acceleration include:

- Re-registration jumps.
- Object/person identity switches.
- Depth or mesh scale changes.
- Coordinate changes.
- Missing-frame indexing errors.
- Abrupt stationary-anchor transitions.

For independent position noise with standard deviation `σ_p`, central second differences have approximate acceleration-noise standard deviation:

```text
σ_a ≈ √6 × σ_p / Δt²
```

At 30 FPS, independent position noise of 5 mm corresponds to approximately **11 m/s²** of acceleration noise per coordinate. Actual noise may be correlated; this example illustrates the amplification rather than predicting the measured pipeline error.

Gate unreliable observations before filtering. Preserve rejected measurements and reasons in diagnostics.

### 7.6 Validate dynamics rather than minimizing acceleration

Check:

- Near-zero motion and acceleration during independently verified stationary intervals.
- No unexplained acceleration spikes at tracking resets.
- Preservation of genuine starts, stops, flips, and impacts.
- Agreement between projected motion and visible object features.
- Acceleration error against independent reference trajectories, when available.
- Correct body-joint motion, not only object motion.
- Uncertainty growth during occlusion and unsupported prediction.

Do not simply minimize acceleration: that would favor motionless objects even when they should move.

Temporal acceleration or jerk penalties can be useful regularizers, but they must be confidence- and motion-aware. Strong penalties can erase genuine dynamics, especially near contact events and impacts.

Without independent reference measurements, the pipeline can establish plausibility and consistency, but cannot guarantee exact acceleration.

## 8. Recommended pipeline

```text
Validated video, timestamps, and camera model
    ▼
Verified person and object identity
    ▼
Reliable masks, depth, and feature observations
    ▼
Outlier rejection and observation-confidence estimation
    ▼
Stationary/moving state estimation
    ├── STATIC: robust fixed-pose anchor
    ├── MOVING: time-aware position/velocity/acceleration dynamics
    └── UNRELIABLE: bounded prediction + increasing uncertainty
    ▼
SE(3)-aware filtering / offline smoothing
    ▼
Confidence-gated human–object refinement
    ▼
Reprojection, stationary-drift, and acceleration validation
    ▼
Accepted output + confidence/provenance + raw comparison
```

Offline smoothing may use future observations. A causal streaming mode must use a separate contract and should not claim equivalence to full-clip RTS smoothing.

## 9. Validation and implementation priorities

### First milestone: correctness and observability

- Verify target identity and render object masks.
- Validate camera, coordinate, scale, and timestamp contracts.
- Reject stale or incomplete artifacts; cache identity must include changed stabilization configuration.
- Add raw-versus-stabilized diagnostic plots for position, rotation, velocity, and acceleration.
- Establish reviewed stationary and moving intervals on representative episodes.

### Second milestone: eliminate static drift

- Fit stationary anchors across reliable frames.
- Add hysteresis-based stationary/moving decisions.
- Gate updates and recovery using multiple evidence sources.
- Stabilize supported depth-scale drift.
- Test occlusion and lock/unlock transitions.

### Third milestone: preserve correct dynamics

- Add timestamp-aware velocity/acceleration states and motion models.
- Integrate adaptive filtering and offline smoothing.
- Stabilize anatomical hand/body tracks.
- Validate shared-scale human–object interactions.
- Evaluate acceleration error when reference trajectories are available.

### Suggested benchmark cases

Use representative bowl, desk, pan, and hoop episodes, including:

- An object resting before and after manipulation.
- A stationary background object.
- Occlusion followed by reappearance.
- A fast flip or abrupt contact event.
- A thin or symmetric object.

Record:

| Measure | Purpose |
|---|---|
| Correct-target rate | Reject stable but semantically incorrect results |
| Required-frame coverage | Detect missing predictions |
| Stationary translation/rotation drift | Measure unwanted movement |
| Reprojection and valid-depth residuals | Ensure stabilization remains observation-consistent |
| Recovery jump magnitude | Evaluate re-registration behavior |
| Acceleration/jerk traces | Expose resets, over-smoothing, and timing errors |
| Reference acceleration error | Measure dynamics accuracy where reference data exists |
| Interpolation/prediction fraction | Expose synthetic coverage |
| Confidence and rejection states | Make unsupported outputs visible |

Tune thresholds on reviewed validation data. Universal pixel-motion, acceleration, or IoU thresholds should not be assumed correct across object sizes, camera views, and action speeds.

## 10. Bottom line

The most important combination is:

1. **Correct target and consistent geometry.**
2. **Verified stationary intervals with explicit pose anchoring.**
3. **Confidence-gated, timestamp-aware motion estimation that preserves real acceleration.**

Do not solve static drift by only increasing smoothing strength, and do not solve acceleration error by forcing acceleration toward zero.
