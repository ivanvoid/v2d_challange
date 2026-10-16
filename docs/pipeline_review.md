# Technical Audit: V2D Reconstruction Workspace

## 1. Executive summary

**The workspace is a useful research prototype, but it is not yet a reliable Track 1 submission pipeline or production reconstruction service.**

The underlying model stack is capable: it includes camera calibration, monocular geometry, segmentation, hand reconstruction, object mesh generation, 6D pose tracking, and optional refinement. However, the current workflow can:

- Reconstruct the **wrong object** while reporting successful execution.
- Reuse outputs generated with an **old reference frame**.
- Treat **partial output directories as completed stages**.
- Produce hand/object results without the **full-body MHR parameters required by Track 1**.
- Present synthetic or interpolated estimates as valid without preserving their uncertainty.
- Exhaust memory on longer videos.

**Recommendation:** prioritize target identity, data contracts, completion validation, and full-body integration before adding models or optimizing GPU performance.

### Verified observations

At the time of inspection:

| Observation | Evidence |
|---|---|
| Dataset contains 30 episodes and 16,563 frames | `challenge/track_1/meta/info.json` |
| Videos are static third-person views, 1536×1152 at 30 FPS | Same metadata |
| Batch summary contained 27 recorded episodes | `out/_summary.tsv` |
| Summary reported 18 `DONE(no-overlay)`, 8 failures, and 1 skipped episode | Same summary |
| No episode-level `mhr_params.pt` or `mhr_mesh.pt` was present | `out/episode_*/` |
| Episode 9 segments a background desk rather than the manipulated foreground desk | `out/episode_000009/frames/000207.png` and `masks/1/000207.png` |
| Two saved reference frames disagree with batch-selected references | Episode 3: batch 324, artifacts 0; episode 26: batch 40, artifacts 0 |
| A partial-batch loss bug is reproducible without loading models | Isolated MoGe early-EOF test |
| Main pipeline function has 125 arguments and spans 1,658 lines | `/workspace/v2d_pipelines/run_ego_wilor.py` |

The summary is a snapshot, not a completed evaluation of all 30 episodes. Artifact presence and process success are not evidence of reconstruction accuracy.

### Audit scope and limitations

This audit inspected:

- The current workspace’s scripts, documentation, dataset metadata, logs, and selected outputs.
- The active implementations and supporting modules under `/workspace`.
- The submission schema and evaluation implementation.
- CARI4D training configuration and selected training/runtime code.
- Relevant container definitions and tests.

GPU inference, model retraining, submission, and model-accuracy benchmarking were **not** performed. Performance gains below are engineering expectations, not measured speedups. The audit was read-only; this report records the findings from that inspection rather than a new snapshot of the outputs.

---

## 2. System architecture

### 2.1 What problem does the project solve?

The intended task is **4D human–object reconstruction**:

> Recover a human’s articulated geometry, an object’s geometry, and their motion over time from video.

There are three distinct systems in this installation:

1. **Active local workflow:** monocular hand–object reconstruction driven by `run_all.sh`.
2. **Available full-body workflows:** SAM3D Body and CARI4D.
3. **Available multi-camera workflow:** calibrated stereo/multi-view reconstruction.

The active workflow is primarily an **egocentric hand–object pipeline being applied to third-person challenge videos**. That distinction explains several failures and fragile assumptions.

### 2.2 Architecture diagram

```text
/local-data/ivan/v2d
│
├── challenge/                 Videos + episode/task metadata
├── weights/                   Local checkpoints and model caches
├── run_all.sh                 Sequential batch controller
├── pick_reference_frame.py    Reference-frame heuristic
├── progress.sh                File/log-based dashboard
├── render_results.py          Body/object visualization
├── out/                       Intermediate and final artifacts
└── v2d_submission_kit/         Schema validation, packing, evaluation
           │
           ▼
/workspace/v2d_pipelines
           │
           ├── Active: run_ego_wilor.py
           ├── Alternative: run_ego_reconstruction.py
           ├── Alternative: run_mv_hoi_reconstruction.py
           └── Supporting pipeline entrypoints
           │
           ▼
/workspace/v2d_docker/container.py
           │
           ├── V2D_BACKEND=venv
           │     └── /opt/venvs/<model>/bin/python -m <module>
           │
           └── Docker backend
                 └── Model-specific container
           │
           ▼
Model libraries + native CUDA/C++ extensions
           │
           ▼
PNG / JSON / PT / OBJ / NPZ artifacts
           │
           ├── Visualization
           ├── Portable hand–object bundle
           └── Track 1 exporter — not connected by the local batch workflow
```

### 2.3 Module dependency map

```text
v2d_pipelines
├── v2d_common
│   ├── datatypes
│   ├── video/frame I/O
│   ├── transform conventions
│   └── result_bundle
├── v2d_docker
│   └── subprocess/container execution
├── v2d_anycalib
├── v2d_depth
├── v2d_moge
├── v2d_grounding_dino
├── v2d_sam2
├── v2d_wilor
├── v2d_hamer
├── v2d_hawor
├── v2d_sam3d
├── v2d_foundation_pose
│   ├── v2d_mesh
│   ├── TensorRT or PyTorch scorer/refiner
│   └── native rendering/geometry extensions
├── v2d_geocalib             [optional]
├── v2d_droid_slam           [optional]
└── v2d_gsplat_refinement    [optional]

Full-body alternative:
v2d_cari4d
├── MoGe 2
├── SAM3D Body / MHR
├── FoundationPose
├── CoCoNet temporal refinement
└── Contact-guided optimization
```

There is no demonstrated circular dependency in the inspected active path. The more significant issue is **high-level orchestration coupling**, including WiLoR relying on HaMeR-named alignment utilities.

### 2.4 Execution flow diagram

```text
For each configured episode:
    │
    ├── Locate video and object prompt
    │
    ├── Pass 1: run complete pipeline with reference frame 0
    │     ├── Calibration
    │     ├── Depth
    │     ├── WiLoR detections
    │     └── Continue downstream if frame 0 has hands
    │
    ├── Choose median frame containing any hand detection
    │
    ├── Pass 2: rerun pipeline with selected reference
    │     ├── Reuse cached artifacts
    │     └── On any pipeline failure:
    │           run SAM2 recovery and retry
    │
    ├── Attempt full-body/object rendering
    │
    └── Append status to summary
          └── Batch exits 0, including when episodes failed
```

Pass 1 is not a dedicated detection pass. If frame 0 contains hands, it can perform expensive reconstruction before the final reference is chosen.

### 2.5 Data flow diagram

```text
MP4 + episode metadata
        │
        ▼
AnyCalib
        ├── Intrinsics JSON
        ├── Distortion JSON
        └── Undistorted MP4
                  │
          ┌───────┴─────────┐
          ▼                 ▼
   Extracted RGB PNGs   MoGe per-frame inference
          │                 ├── Depth PNGs
          │                 └── Intrinsics JSONs
          │                           │
          │                     Temporal median K
          │
    ┌─────┴───────────────────┐
    ▼                         ▼
WiLoR                     Grounding DINO
MANO detections           Reference-frame boxes
    └─────────┬───────────────┘
              ▼
         SAM2 prompts
              ▼
      Object/hand masks
          │          │
          │          └── Hand association → depth alignment
          │                              → gap filling
          ▼
SAM3D object mesh
          ▼
Transform + scale estimation
          ▼
FoundationPose tracking
          ▼
ESKF + backward smoothing
          │
          └───────────────┬──────────────────┐
                          ▼                  ▼
                  Hand–object bundle     Overlays

Missing active connection:
RGB/person masks → SAM3D Body or CARI4D → MHR params
                                              │
                                              ▼
                                  Track 1 episode NPZ + mesh
                                              ▼
                                    Submission Parquet
```

### 2.6 Communication and critical path

Components communicate through **filesystem artifacts and CLI subprocesses**, not a persistent inference service.

Advantages:

- Different dependency environments can coexist.
- Stages are inspectable and individually executable.
- Process boundaries release model memory.

Costs:

- Repeated model initialization.
- Repeated video decoding and encoding.
- Large numbers of small files.
- No shared transaction or completion contract.
- Weak cross-stage provenance.

The active critical path is:

```text
Calibration → depth/hand detections → seed selection → masks
→ generated mesh → scale search → object tracking
→ smoothing/alignment → packaging
```

Depth and hand detection could be scheduled independently after preprocessing. Body and object branches could also run independently before joint refinement, subject to GPU-memory scheduling.

---

## 3. End-to-end AI pipeline review

### 3.1 Stage inventory

| Stage | Inputs → outputs | Models/features | Transformations and post-processing | Main dependencies |
|---|---|---|---|---|
| Ingestion | MP4 + JSONL metadata → episode configuration | None | Index lookup, object prompt selection | Bash, Python |
| Calibration | MP4 → intrinsics/distortion | AnyCalib learned camera estimation | Up to 16 sampled frames; parameter aggregation | PyTorch, OpenCV |
| Undistortion | MP4 + camera estimate → rectified MP4 | None | Image remapping, interpolation, H.264 re-encoding | OpenCV, FFmpeg |
| Frame extraction | MP4 → numbered PNGs | None | Decode and sequential indexing | FFmpeg |
| Depth | MP4 → depth PNGs + per-frame K | MoGe V2 dense geometry | RGB normalization; batch inference; inverse-depth encoding | PyTorch, MoGe |
| Intrinsics stabilization | Per-frame K → single K | None | Temporal medians | NumPy |
| Hand detection/reconstruction | Frames → MANO JSON records | WiLoR internal YOLO detector and hand regressor | Left-hand convention conversion; camera metadata | WiLoR, PyTorch |
| Object detection | Reference RGB + text → boxes | Grounding DINO Swin-T visual/text features | Thresholding; confidence sorting; select first result | Grounding DINO, torchvision |
| Segmentation | Video + boxes/masks → masks | SAM2.1 Hiera-L; temporal memory | Reverse/forward propagation; logits thresholded at zero | PyTorch, SAM2 |
| Mesh generation | Reference RGB/mask/depth → object mesh/transform | SAM3D Objects latent 3D reconstruction | Pointmap conversion, mesh post-processing, texture baking | Hydra, PyTorch3D, gsplat |
| Scale estimation | Mesh + reference RGB/depth/mask → scaled mesh | FoundationPose registration/scoring | Coarse-to-fine log-scale search | TensorRT/PyTorch, CUDA rendering |
| Object tracking | Video + depth/masks/mesh/K → pose JSONs | FoundationPose scorer/refiner | Register at reference; forward/backward tracking; IoU recovery | OpenCV, native CUDA/C++ |
| Object smoothing | Pose sequence + masks → smoothed poses | Error-state KF and RTS backward smoother | SO(3) rotation handling; IoU-weighted measurement noise | NumPy, SciPy |
| Hand association | WiLoR records + masks → hand tracks | MANO silhouettes | Per-frame greedy IoU assignment | MANO, pyrender |
| Hand alignment | Hands + depth/K → aligned records | None beyond geometry | Reprojection, median depth offset, scale aggregation | MANO, pyrender |
| Gap filling | Aligned records + mask filenames → filled records | None | Rotation SLERP; translation interpolation; boundary clamping | NumPy |
| Packaging | Mesh/poses/hands → NPZ + manifest | None | Transform composition; mesh scale baking | NumPy, SciPy |
| Visualization | Frames/meshes/poses → MP4 | None | CV-to-GL conversion, rendering, compositing | pyrender, FFmpeg |
| Full-body reconstruction | RGB/person prompts → MHR parameters/mesh | SAM3D Body, or CARI4D | **Available but absent from active batch** | PyTorch, MHR |
| Submission | MHR/object episode arrays → Parquet | None | Mesh budgeting, schema/rotation/finite-value checks | pandas, pyarrow, trimesh |

### 3.2 Data ingestion

Current validation is insufficient for a multi-stage reconstruction system.

Missing or inconsistent checks include:

- Successfully opening the video.
- Decoded frame count versus dataset frame count.
- Frame dimensions and orientation.
- Timestamp continuity.
- Required artifact coverage.
- Model/checkpoint availability before expensive work starts.

Several readers interpret a failed frame read as ordinary EOF. A corrupted or truncated file can therefore become a shorter, apparently successful prediction.

**Fix:** introduce a preflight record containing:

```text
episode_id
source_video_hash
decoded_frame_count
expected_frame_count
width / height
frame timestamps
camera_id
object identity
object prompt
dataset revision
```

Require every stage to report the exact frame IDs it consumed and produced.

### 3.3 Preprocessing

**Information loss:**

- Undistortion performs interpolation.
- Rectified video is re-encoded as H.264/YUV420.
- MoGe depth is quantized to uint16 inverse depth.
- SAM2 probabilities are reduced to binary masks.
- Learned internal features are not retained for downstream association.

PNG storage itself is lossless. The lossy operations are upstream resampling, video compression, depth quantization, and binary thresholding.

**Important geometry distinction:** the MoGe wrapper correctly forwards calibrated intrinsics. MoGe uses a focal/FOV prior rather than preserving the entire camera model unchanged; its output convention fixes the principal point at image center. A median camera matrix is then used downstream.

That is not automatically incorrect, but the resulting depth/K pairing needs validation rather than an assumption of exact calibration equivalence.

**Leakage:** full-clip medians and backward smoothing use future frames. This is acceptable for offline reconstruction, but incompatible with a claimed causal streaming pipeline.

### 3.4 Model-stage weaknesses

The central weakness is **unverified composition of plausible estimates**:

```text
Wrong target box
    → wrong mask
    → wrong generated mesh
    → wrong pose sequence
    → smooth, internally consistent wrong reconstruction
```

Other limitations:

- Independent per-frame depth lacks explicit temporal consistency.
- Independent per-frame hand inference loses identity.
- Single-image mesh completion invents unseen geometry.
- One reference frame determines initialization quality.
- Contact, body pose, and object pose are largely separate in the active path.

The stack is not uniformly outdated. MoGe V2, SAM3D, and TensorRT FoundationPose support are already present. Replacing everything with newer models would not solve these architectural failures.

### 3.5 Post-processing and output

The smoother can make a wrong trajectory look stable. Hand interpolation can create apparent coverage without observations. Packaging preserves some validity masks, which is positive, but its validity semantics are too coarse.

A production output should distinguish:

```text
observed
tracked
interpolated
extrapolated
missing
rejected
```

It should also retain confidence diagnostics and the reason for each fallback.

---

## 4. Top 10 priority issues

“Critical” here means an immediate security or deliverable-integrity blocker. Not all ten issues have Critical severity.

### 1. Plaintext credential exposure

**Severity: Critical**

**Affected file:** `setup_me.sh:12`

A non-placeholder Hugging Face token is stored in executable documentation.

**Impact:** potential unauthorized use of the account and access to gated resources.

**Fix:**

- Revoke/rotate the token.
- Remove it from scripts and any shared copies/history.
- Use environment injection or a secret manager.
- Add automated secret scanning.
- Redact credentials from diagnostic output.

Do not assume the token is inactive without checking.

### 2. Active workflow does not satisfy the Track 1 contract

**Severity: Critical for Track 1 delivery**

**Affected files:**

- `run_all.sh:80–116`
- `/workspace/v2d_pipelines/run_ego_wilor.py`
- `v2d_submission_kit/tools/pack_reconstruction.py:16–25`

The batch produces hand/object artifacts, but the submission requires:

- `pose [T,136]`
- `scales [68]`
- `shape [45]`
- Object rotation, translation, scale, and mesh.

No episode-level MHR outputs were present in the inspected output tree.

**Impact:** expensive runs do not yield a complete submission.

**Fix:** create a dedicated third-person Track 1 pipeline with a SAM3D Body or CARI4D branch and an explicit submission adapter. Validate the full contract before marking an episode complete.

### 3. Wrong-object reconstruction is already occurring

**Severity: High**

**Affected files:**

- `/workspace/v2d_pipelines/run_ego_wilor.py:921–942`
- `out/episode_000009/dino_detections.json`
- `out/episode_000009/masks/1/000207.png`

The code chooses the highest-confidence Grounding DINO detection without interaction or identity verification.

In episode 9, the saved mask selects a background desk rather than the foreground desk being manipulated.

**Impact:** every downstream object result is semantically invalid, even if its geometry is internally consistent.

**Fix:**

- Score candidates using target-person proximity, hand interaction, temporal evidence, and prompt matching.
- Verify candidates across multiple frames.
- Reject ambiguous selection instead of silently choosing the first.
- Support reviewed seed masks for difficult episodes.
- Include the object in prompt/mask verification overlays.

### 4. Cache invalidation is unsafe

**Severity: High**

**Affected file:** `/workspace/v2d_pipelines/run_ego_wilor.py:270,352–396,926,1098,1138,2084`

Completion commonly means “a file/directory exists.” The prompt matcher does not compare the requested reference frame or object prompt.

**Observed impact:** episode 3 and episode 26 retained reference-frame-0 artifacts after the batch selected different references.

Changing calibration, model weights, a prompt, or a reference can leave downstream artifacts stale.

**Fix:** compute a stage identity from:

```text
input hashes + configuration + model hash + code revision + schema version
```

Reuse outputs only if identity and output validation pass. Invalidate dependent stages through an explicit DAG.

### 5. Partial outputs can masquerade as completed inference

**Severity: High**

**Affected files:**

- `/workspace/v2d_moge/lib/video_to_depth.py:100–158`
- `/workspace/v2d_unidepth/lib/video_to_depth.py:43–87`
- `/workspace/v2d_foundation_pose/lib/run_video_to_poses.py:163–185`
- `/workspace/v2d_pipelines/run_ego_wilor.py:270–271`

The MoGe final batch is processed only when the batch fills or the frame index equals the declared last frame.

**Reproduced:** a source declaring 10 frames but decoding 7, with batch size 8, executes zero inference batches.

FoundationPose also breaks on missing RGB/depth and subsequently logs completion using the declared frame count.

**Impact:** silently missing predictions; incomplete directories are then cached.

**Fix:**

- Flush buffered frames at EOF.
- Validate decoded count against the ingestion contract.
- Report truncation explicitly.
- Require exact expected frame coverage before committing stage completion.
- Use transactional output publication.

### 6. Third-person videos inherit egocentric identity assumptions

**Severity: High**

**Affected files:**

- `pick_reference_frame.py:31–45`
- `/workspace/v2d_wilor/lib/tracks_from_wilor_masks.py:128–137,207–225`
- `/workspace/v2d_wilor/lib/tracks_interpolate.py:219–288`

Hand association is greedy per-frame silhouette IoU. A handedness tiebreak assumes image-right corresponds to the right hand.

That assumption is unsuitable for third-person views, crossed arms, and changing body orientation.

**Impact:** identity switches, incorrect handedness, and downstream mirroring errors.

**Fix:** associate hands through a persistent person track, body wrist keypoints, temporal motion, and assignment optimization. Do not use image side as an anatomical label.

Decouple object initialization from the requirement that WiLoR detects a hand.

### 7. Long-video memory is not bounded

**Severity: High**

**Affected files:**

- `/workspace/v2d_sam2/lib/video_to_masks.py:82–124`
- `/workspace/v2d_sam3d_body/lib/estimate_mhr_params.py:344–425`
- `/workspace/v2d_depth_anything/lib/video_to_depth.py:116–128`

SAM2 buffers every full-resolution mask before writing. SAM3D Body accumulates all outputs, including device tensors. Depth Anything 3 loads the entire RGB video before chunked inference.

**Impact:** RAM/VRAM grows with video length; chunked inference does not necessarily mean chunked storage.

**Fix:** incremental writers, bounded decode queues, chunked CPU output transfer, and overlap-aware state persistence.

### 8. Confidence and provenance are lost at the output boundary

**Severity: High**

**Affected files:**

- `/workspace/v2d_wilor/lib/_wilor.py:75–92`
- `/workspace/v2d_common/result_bundle.py:174–220,698–760`

WiLoR records have `score=1.0` regardless of detection quality. Bundle validity is largely driven by file presence, and hand interpolation provenance is not exported.

Without camera poses, identity camera transforms are marked valid. That can be appropriate for a verified static camera, but is unsafe as an undocumented general fallback.

**Impact:** downstream consumers cannot distinguish measured, inferred, assumed, and synthesized data.

**Fix:** export source state, uncertainty, quality diagnostics, and explicit camera assumptions. Preserve interpolation metadata.

### 9. Batch failure handling masks operational failures

**Severity: High**

**Affected files:** `run_all.sh:125–196`, `progress.sh`

Problems include:

- First-pass failures ignored broadly.
- Any second-pass failure triggers SAM2 recovery.
- Batch always exits zero.
- Summary is overwritten on each invocation.
- Resume depends on an overlay file.
- `FORCE=1` bypasses the batch skip but does not invalidate pipeline caches.
- No per-episode lock prevents concurrent writers.

**Impact:** schedulers can report success while work failed; retries waste compute; parallel execution risks corruption.

**Fix:** typed stage failures, targeted retries, persistent run IDs, artifact-based acceptance checks, file locks, and a nonzero batch exit when required episodes fail.

### 10. Experiment reproducibility is inconsistent

**Severity: High**

**Affected files:**

- `fetch_repos.sh`
- `setup_me.sh`
- `/workspace/v2d_sam3d/docker/run_image_to_mesh.py`
- `/workspace/v2d_sam3d_body/lib/download_weights.py`
- Model Dockerfiles and package definitions

The local workflow uses editable source, machine-specific paths, partially pinned dependencies, and model downloads without uniform revision/hash requirements. Active SAM3D generation does not receive an explicit seed from `run_ego_wilor`.

**Impact:** model comparisons can conflate code, checkpoint, environment, and randomness changes.

**Fix:** immutable experiment manifests, pinned model revisions and hashes, locked environments, explicit seed policy, and promoted runtime images.

Some modules already implement good provenance checks; extend those patterns rather than replacing them.

---

## 5. Additional code-quality and reliability findings

| Issue | Severity | Affected files | Impact and actionable fix |
|---|---|---|---|
| Oversized orchestration function | High | `/workspace/v2d_pipelines/run_ego_wilor.py` | 125 arguments and a 1,658-line function combine configuration, inference, rendering, cache logic, and packaging. Split into typed stage adapters and task-specific DAGs. |
| Duplicate/misnamed setup script | Medium | `fetch_weights.sh`, `fetch_repos.sh` | Files are byte-identical; the former installs source rather than downloading weights. Separate installation and weight provisioning. |
| Machine-specific setup instructions | Medium | `setup_me.sh`, `how_to_run_it.md` | Paths, repeated commands, and malformed example syntax make setup unreliable. Replace with one validated configuration and a clean runbook. |
| Non-finite depth enters hand alignment | High | `/workspace/v2d_hamer/lib/align_hands.py:343–378` | Median depth residuals do not explicitly filter non-finite/invalid observed depth. Apply validity masks and reject inadequate valid-pixel support. |
| Mask existence mistaken for visibility | High | `/workspace/v2d_wilor/lib/tracks_interpolate.py:175–183,350–352` | Empty mask files still make frames interpolation candidates. Gate on actual mask area/quality. |
| Boundary extrapolation is unbounded | High | Same file:291–305,389–405 | Default clamping can persist the nearest hand pose arbitrarily far. Limit extrapolation by elapsed time and confidence. |
| Non-standard JSON NaN | Medium | Same file:160–165 | Interpolated diagnostics include `NaN`, which strict JSON clients reject. Use `null` plus an explicit state. |
| Smoother ignores elapsed time and missing-frame gaps | Medium | `/workspace/v2d_foundation_pose/lib/run_ekf_smoothing.py` | Sparse poses are treated as equally spaced measurements. Use timestamps and time-dependent process noise. |
| Missing mask gets maximal smoothing trust | Medium | Same file:71–77 | IoU defaults to 1 when a mask is absent. Represent unknown quality separately and reduce trust. |
| Quaternion/transform validation inconsistent | Medium | `/workspace/v2d_common/datatypes.py:145–158`, `render_results.py:52–57` | Transform construction assumes unit quaternions. Normalize/reject and validate scale, units, and SO(3). |
| Global runtime monkey patches | Medium | `/workspace/v2d_sam3d_body/lib/estimate_mhr_params.py:336–338`; SAM3D initialization | Globally replacing `torch.cuda.empty_cache` or upstream functions affects all code in-process. Use scoped adapters or upstream-compatible configuration. |
| Cleanup not guaranteed on exception | Medium | `render_results.py`, body estimation debug path | Renderer/writer closure can be skipped after failure. Use context managers or `try/finally`. |
| Model singleton ignores checkpoint identity | Medium | MoGe, WiLoR, SAM2 loader globals | Reusing a process with new weight paths can return the old model. Key caches by checkpoint, device, and precision. |
| Unsafe generalization of trusted-file loading | Medium | `render_results.py:96`, body bbox loading | `torch.load(..., weights_only=False)` should not accept arbitrary uploads. Restrict to trusted artifacts or use safer tensor serialization. |
| Training optimizer configuration is misleading | Medium | CARI4D `trainer.py:106` and training YAML:98–102 | YAML says `optimizer: adam`, `weight_decay: 0`; code instantiates AdamW without passing weight decay. Honor config or record actual optimizer parameters. |

There is no evidence proving that all old scripts or optional modules are dead code, so they should not be deleted without usage investigation.

### SOLID assessment

- **Single responsibility:** substantially violated in the main orchestrator.
- **Open/closed:** adding model alternatives requires editing many branches and argument mappings.
- **Interface segregation:** callers receive a very large configuration surface.
- **Dependency inversion:** orchestration directly imports implementation-specific wrappers.
- **Liskov substitution:** depth/pose alternatives lack sufficiently uniform validated contracts to assess safe interchangeability.

---

## 6. Computer vision and video-specific assessment

### 6.1 Frame extraction and FPS

The released dataset documents 30 FPS, so the current hardcoded visualization rate matches these inputs. It is not portable to variable-FPS or other video sources.

Current extraction:

- Materializes all frames as PNG.
- Does not preserve presentation timestamps in a central frame manifest.
- Can leave old frames behind when rerun into an existing directory.
- Uses different readers across stages.
- Relies on sequential numbering to align predictions.

`render_results.py` and mask overlays use enumeration indexes rather than consistently deriving IDs from filenames. Missing or stale files can therefore shift correspondence.

**Recommended contract:** preserve `frame_id`, source timestamp, and preprocessing identity independently of storage order.

### 6.2 Where temporal information is lost

| Location | Information lost |
|---|---|
| MoGe per-frame inference | Joint temporal geometry constraints |
| WiLoR per-frame inference | Persistent identity and motion context |
| Grounding DINO on one frame | Evidence that the detected object is the manipulated target |
| Greedy hand matching | Global sequence-consistent assignment |
| Binary mask export | Uncertainty around boundaries and occlusion |
| Final hand bundle | Observed versus interpolated/extrapolated state |
| Fixed-step smoother | Actual elapsed time and missing-frame duration |

SAM2 and FoundationPose retain temporal state, so the pipeline is not entirely frame-independent. The problem is that temporal information is not preserved consistently across boundaries.

### 6.3 Where identity is lost

1. Selecting the first object detection.
2. Seeding only the hands visible at one reference frame.
3. Associating detections independently in each frame.
4. Relabeling handedness using an egocentric position heuristic.
5. Packaging tracks without complete observation provenance.

Use a hierarchy:

```text
episode target
    ├── persistent person identity
    │      ├── anatomical left hand
    │      └── anatomical right hand
    └── persistent manipulated-object identity
```

Track IDs should not be defined merely by detection order.

### 6.4 Feature extraction and embeddings

There is **no general embedding-generation or retrieval subsystem** in the active workflow.

Models internally extract visual/text/temporal features, but the interfaces mostly exchange boxes, binary masks, depth, and poses.

That simplifies interoperability, but it prevents:

- Appearance-assisted re-identification.
- Reuse of image features.
- Cross-model feature consistency checks.
- Confidence-aware association.

If appearance embeddings are added, use them as one association signal—not as a geometric source of truth.

### 6.5 Where compression hurts

- **Binary masks:** useful uncertainty is discarded.
- **Low-resolution model crops:** fine geometry of hoops, handles, rollers, and fingers can disappear.
- **Single-image latent mesh reconstruction:** small features and hidden geometry may be replaced by learned priors.
- **Submission mesh simplification:** thin structures can be damaged.
- **Inverse-depth quantization:** error increases with distance.

Approximate inverse-depth quantization step:

```text
Δd ≈ (d + 1)² / 65535
```

At ordinary indoor distances this may be much smaller than monocular model error. It should not be assumed to be the dominant accuracy bottleneck.

### 6.6 Memory per video

At 1536×1152:

| Representation | Approximate storage for 1,000 frames |
|---|---:|
| RGB uint8 frames | 5.31 GB |
| One uint8 mask stream | 1.77 GB |
| Object + two hand mask streams | 5.31 GB |
| Float32 depth | 7.08 GB |
| MHR vertices, 18,439×3 float32 | 0.22 GB, excluding other outputs |

These are uncompressed payload estimates, not measured process peaks. Predictor state, tensors, batching, rendering, and Python overhead add memory.

### 6.7 Long-video scalability and streaming

The active system is fundamentally offline:

- Calibration samples the whole clip.
- Reference selection inspects all detections.
- SAM2 propagates backward and forward.
- Object smoothing uses an RTS backward pass.
- Several stages accumulate whole-video data.
- Outputs are committed after large jobs.

For long videos, use overlapping chunks and explicit state transfer. For streaming, provide a separate causal mode rather than silently changing the semantics of the offline pipeline.

### 6.8 Multi-camera support

Multi-camera support exists under:

`/workspace/v2d_pipelines/run_mv_hoi_reconstruction.py`

It integrates calibration, stereo depth, object/human segmentation, multi-view pose tracking, MHR optimization, and consistency evaluation.

However, Track 1 currently exposes **one static third-person view per episode**. Metadata naming front/back/left/right cameras does not provide synchronized multi-view inputs for every episode.

Do not treat frames of a moving object from a fixed camera as ordinary static-scene multi-view geometry without motion compensation.

---

## 7. Machine learning assessment

### 7.1 Dataset quality, bias, and labels

The challenge set is small and organized around repeated object groups. The prompts also contain misspellings and inconsistent descriptions.

Verified failures include:

- Episode 15: no detection for the long foam-block prompt.
- Episode 18: no detection for the misspelled paint-roller prompt.
- Episodes 12, 13, 16: no hand detections usable by the active workflow.

Risks:

- Similar furniture competes with the target object.
- Small hands have poor effective image resolution.
- Thin structures challenge segmentation and mesh generation.
- Reflective metal objects challenge monocular depth.
- Occlusion can produce plausible but incorrect hand poses.
- Repeated objects can leak across naïve train/validation splits.

**Recommendation:** maintain normalized detection aliases while preserving the original metadata. Split evaluation by sequence, object instance/category, subject, and camera—not by random frames.

The metadata’s `train` label does not by itself establish permission to train on challenge evaluation targets.

### 7.2 Training flow

The active local batch performs **pretrained inference only**. It does not train WiLoR, SAM2, MoGe, or FoundationPose.

A separate training system exists in CARI4D:

```text
External RGB/depth/masks/MHR/contact data
        ▼
Trim-aware, identity-validated dataset indexes
        ▼
Temporal window sampling
        ▼
RGB / XYZ / mask / pose initialization features
        ▼
CoCoNet / HORefine
        ▼
Human, object, and contact losses
        ▼
Accelerate distributed training
        ▼
Checkpoint + optimizer/scheduler/scaler/RNG state
        ▼
Validation + best-model tracking + W&B provenance
```

The inspected configuration uses:

- 96-frame clips.
- DINOv2 feature encoders.
- Spatial-temporal attention.
- Object pose, human pose/shape, and contact prediction.
- FP16.
- 200,000 optimization steps.
- External Daniel/BEHAVE datasets.

### 7.3 Training strengths

The CARI4D training implementation is significantly more mature than the local batch runner:

- Resumable optimizer/scheduler/scaler state.
- RNG capture/restoration.
- Checkpoint-bound supervision contracts.
- Data-sampling contracts.
- Distributed checkpoint loading.
- W&B/source/config provenance.
- Finite-loss guards.
- Timing and speed-benchmark support.

Those capabilities should not be described as absent merely because the local workflow does not use them.

### 7.4 Training risks

- Training data is external and was not available for a quality audit.
- Absolute cluster paths impede portability.
- `pred_uncertainty: false` means the configured model does not supply a learned uncertainty output.
- Optimizer YAML does not match actual construction.
- Strong initialization dependencies can propagate depth/mask/pose bias.
- Contact-derived pseudo-labels need quality stratification and provenance.
- Thin-object and rare-contact performance requires targeted evaluation.

Avoid changing the optimizer semantics for an existing checkpoint without recording the change and validating a new training run.

### 7.5 Evaluation

Track 1 evaluates:

- Human Chamfer distance.
- Object Chamfer distance.
- Human acceleration error.
- Object acceleration error.
- Hand–object penetration.

The scorer fits **one human-derived Sim(3) alignment at the first scored frame** and applies it across the episode. Independently rescaling the human and object can therefore still harm the result.

Local evaluation should additionally measure:

| Metric | Why it matters |
|---|---|
| Required-frame coverage | Prevent silent truncation |
| Target identity accuracy | Catch wrong-object reconstruction |
| Mask IoU/boundary quality | Assess segmentation |
| 2D keypoint/silhouette reprojection | Detect alignment problems |
| Depth residuals on valid pixels | Assess geometric consistency |
| Identity switches | Assess hand/person association |
| Contact precision/recall | Avoid false contact constraints |
| Foot sliding/support violations | Assess body motion |
| Confidence calibration | Make rejection thresholds meaningful |
| Runtime, RAM, VRAM, retries | Measure operational cost |

**Caution:** evaluating a reconstruction against the same predicted depth and masks used to create it measures self-consistency, not independent correctness.

### 7.6 Benchmark methodology

Use a fixed benchmark with:

- Difficult and easy object categories.
- Occluded and visible interactions.
- Small and large hands.
- Reflective and thin objects.
- Separate cold-start and cached runs.
- Fixed checkpoint/code/environment identities.
- Per-episode results, not only averages.
- A held-out group split.
- One component changed at a time.

Do not claim an accuracy improvement from smoother videos alone.

---

## 8. MLOps assessment

| Area | Current state | Required improvement |
|---|---|---|
| Environments | Model-specific venv/container isolation; partial version pins | Lock dependencies and native ABI compatibility; record image digests |
| Model versioning | Local weight directories; some modules have manifests/hashes | Uniform immutable checkpoint identity |
| Deployment | CLI/subprocess execution | Health checks, resource limits, timeouts, graceful cancellation |
| Rollback | Manual source/weight changes | Promote complete pipeline versions and retain previous artifacts |
| CI/CD | Module tests exist; no workflow found in inspected trees | CPU contracts plus GPU smoke tests and regression gates |
| Logging | Human-readable logs and artifact dashboard | Structured stage events and persistent run IDs |
| Failure monitoring | Status strings; broad retry logic | Typed failures, failure rates, retry counts, timeouts |
| Quality/drift monitoring | No integrated operational quality monitor found | Track mask quality, rejection rate, residuals, model disagreement, input changes |
| Horizontal scaling | Sequential episode loop | Episode job queue with isolated outputs |
| Multi-GPU inference | No explicit placement in local runner; Docker uses all GPUs | Resource-aware worker/device assignment |
| Distributed training | Available in CARI4D | Preserve its existing checkpoint/provenance contracts |
| Security/privacy | Plaintext token; unrestricted artifact assumptions | Secret scanning, trusted model loading, access controls, retention/redaction policy |

The dataset includes bystanders. A face-blurring workflow exists in the multi-view system, but the active local rendering path does not use it. Privacy controls should be part of output publication.

---

## 9. Performance optimization opportunities

The logged long completions ranged from approximately **49–94 minutes per episode**, with a median near **62.5 minutes** among 13 long `DONE(no-overlay)` entries. These are whole-run times with mixed cache states, not per-stage GPU measurements.

### Ranked by likely operational gain

| Rank | Opportunity | Expected gain | Complexity |
|---|---|---|---|
| 1 | Correct stage caching and selective retries | Avoid repeated expensive work and unnecessary recovery | Medium |
| 2 | Make diagnostic renders optional/asynchronous | Remove several whole-video render/encode passes from the required path | Low–medium |
| 3 | Stream SAM2 masks to storage | Replace output-mask memory growth with bounded buffering | Medium |
| 4 | Reuse canonical decoded/rectified frames | Reduce repeated decoding, temporary PNG generation, and I/O | Medium |
| 5 | Sequential FoundationPose decoding | Avoid `cap.set()` seeking for every forward frame; backward path needs separate design | Medium |
| 6 | Separate detection-only initialization | Avoid full first-pass reconstruction before reference selection | Low–medium |
| 7 | Separate rendering from FoundationPose model initialization | Smoothing currently constructs a tracker/scorer/refiner to obtain rendered IoUs | Medium |
| 8 | Adaptive scale initialization/search | Reduce the configured 9×4 = 36 scale-candidate registrations | Medium |
| 9 | Tuned batches and bounded prefetch | Improve GPU utilization without exceeding memory | Medium |
| 10 | Persistent model workers | Amortize initialization across episodes | High |

FoundationPose already defaults to a TensorRT backend in this installation. “Enable TensorRT” is not a new optimization here; confirm actual runtime manifests and profile preprocessing, rendering, and transfer overhead.

**First profiling target:** measure decode, model initialization, GPU inference, rendering, serialization, and synchronization separately.

---

## 10. AI/ML improvements and models to try

### Ranked by expected reconstruction-quality impact

| Rank | Improvement/model | Integration guidance | Important limitation |
|---|---|---|---|
| 1 | Target-identity verification | Multi-frame candidate selection using person/hand interaction and ambiguity rejection | Better model scores do not solve wrong-target selection by themselves |
| 2 | CARI4D full-body HOI baseline | Use the installed module with human/object masks and a validated object mesh | Not independent ground truth; shares initialization models |
| 3 | SAM3D Body MHR branch | Track the target person; use masks/bboxes and matched camera intrinsics | Verify units/topology/export with the reference MHR forward model |
| 4 | Persistent anatomical hand association | Fuse body wrists, temporal motion, and assignment optimization | Avoid egocentric image-side heuristics |
| 5 | Depth model ablation: MoGe V2 vs UniDepth V2 | Compare on scale, reprojection, and temporal residuals | Different depth conventions require adapters |
| 6 | Depth Anything 3 metric/nested experiments | Existing wrapper supports variants and overlap alignment | Wrapper loads whole video; fixed-camera dynamic scenes challenge multi-view assumptions |
| 7 | Multi-frame object geometry refinement | Motion-compensated RGB/depth fusion or BundleSDF | Errors in monocular depth/pose can contaminate fusion |
| 8 | HaMeR/HaWoR comparison | Existing pipeline branches reduce integration cost | Hand estimators still do not replace full-body reconstruction |
| 9 | CoTracker3 auxiliary point tracking | Track selected object/person points and visibility to verify continuity | 2D tracks do not establish metric 3D; repository’s main license is noncommercial |
| 10 | Confidence-aware joint optimization | Robust silhouette, keypoint, depth, contact, and temporal losses | Must avoid forcing false contact or fitting bad masks |

[CoTracker3’s official repository](https://github.com/facebookresearch/co-tracker) supports online and offline point tracking; review checkpoint and code licensing before commercial integration.

### Sources of truth versus model evidence

Use an explicit evidence hierarchy:

1. **Authoritative contracts:** episode/frame IDs, submission schema, permitted dataset metadata.
2. **Independent observations:** reviewed target masks/keypoints, known calibration, measured dimensions, calibrated stereo where available.
3. **Geometric consistency:** reprojection, visibility, support, motion, and contact constraints.
4. **Model proposals:** depth, pose, masks, generated meshes, learned refinement.

Model agreement is helpful, but correlated models can agree on the same error. A VLM may help with prompt aliases or semantic review; it should not be treated as a source of metric geometry.

---

## 11. Improvement recommendations ranked by ROI

### Quick wins

| Priority | Problem/root cause | Proposed solution | Expected impact | Complexity |
|---|---|---|---|---|
| P0 | Credential stored in script | Rotate and externalize secrets | Immediate security improvement | Low |
| P0 | “Done” lacks deliverable validation | Validate MHR/object schema and required frames | Prevent false completion | Low–medium |
| P0 | Wrong target hidden in QA | Draw object prompts/masks and review ambiguous seeds | Catch catastrophic semantic errors | Low |
| P0 | Partial final depth batches lost | Flush on EOF and compare frame counts | Prevent silent missing predictions | Low |
| P1 | Arbitrary failures invoke SAM2 recovery | Retry only the failed typed stage | Reduce wasted work | Low–medium |
| P1 | Batch always exits zero | Aggregate required-episode failures into exit status | Reliable scheduler integration | Low |
| P1 | Setup scripts misnamed/duplicated | One validated setup/provisioning workflow | Reduce setup mistakes | Low |

### Medium-effort improvements

| Priority | Problem/root cause | Proposed solution | Expected impact | Complexity |
|---|---|---|---|---|
| P0 | Stale/incomplete cache reuse | Stage identity manifests and transactional completion | Reliable reruns and experiments | Medium |
| P0 | Missing full-body branch | SAM3D Body/CARI4D integration and Track 1 adapter | Complete intended deliverable | Medium |
| P1 | Wrong target/hand identity | Person-linked temporal association and multi-frame selection | Major quality improvement | Medium–high |
| P1 | Unbounded mask/body outputs | Incremental storage and CPU transfer | Long-video support | Medium |
| P1 | Confidence stripped away | Observation-state and diagnostics schema | Safer outputs and review routing | Medium |
| P1 | No trustworthy model comparison | Fixed regression suite and controlled ablations | Evidence-based model selection | Medium |

### Major architectural changes

| Priority | Problem/root cause | Proposed solution | Expected impact | Complexity |
|---|---|---|---|---|
| P1 | Monolithic procedural orchestration | Typed stage DAG with task-specific pipelines | Maintainability and selective recomputation | High |
| P2 | Sequential machine-bound batch | Queue-based episode workers with GPU resource limits | Horizontal throughput scaling | High |
| P2 | Offline-only whole-video design | Overlapping chunks and explicit state transfer | Long-video resilience | High |
| P2 | Independent estimates lack interaction reasoning | Confidence-aware joint human/object refinement | Better relative geometry/contact | High |

### Recommended stage contract

A minimal contract should look conceptually like:

```text
StageResult
├── episode_id
├── stage_name
├── status: complete | partial | failed | rejected
├── input_identity
├── code_revision
├── model_identity
├── configuration_hash
├── frame_ids
├── artifact_checksums
├── quality_metrics
├── duration / peak RAM / peak VRAM
└── failure_type / retryability
```

Publish outputs to a temporary run-specific location, validate them, then atomically publish a completion marker. Never equate directory existence with success.

---

## 12. Refactoring roadmap

Assuming a small engineering team, this is a reasonable six-week sequence.

| Week | Work | Acceptance criteria |
|---|---|---|
| **1 — Safety and correctness** | Rotate token; add ingestion/frame checks; fix EOF batching; show object masks; correct exit/retry semantics | Truncated inputs fail clearly; wrong-target example is flagged; required outputs gate completion |
| **2 — Reproducible artifacts** | Stage manifests, reference/config invalidation, atomic publication, per-episode locks, run history | Changing reference/prompt/calibration invalidates dependent artifacts; interrupted runs cannot pass as complete |
| **3 — Full-body delivery** | Add target-person tracking and SAM3D Body/CARI4D branch; implement Track 1 adapter; coordinate/unit round-trip tests | Representative episodes pack successfully with MHR/object arrays matching the reference forward-model convention |
| **4 — Identity and quality** | Multi-frame target selection, anatomical hand association, confidence states, reviewed validation subset | Wrong-target and identity-switch regression tests pass; outputs preserve observation provenance |
| **5 — Performance and memory** | Incremental masks/body outputs, optional renders, canonical frame access, profiling, improved decode path | Peak memory remains bounded for longer clips; cold/cached stage timings are recorded |
| **6 — Deployment and scale** | Locked runtime versions, GPU smoke CI, queue workers, monitoring, promoted pipeline versions | Reproducible clean deployment; safe concurrent episodes; rollback restores a known pipeline version |

Do not start with persistent GPU services or distributed orchestration before the output and caching contracts are reliable.

---

## 13. Production readiness score

These scores assess the **active local workflow**, with credit for reusable infrastructure already present under `/workspace`.

| Category | Score / 10 | Rationale |
|---|---:|---|
| Architecture | **4** | Useful modular model boundaries, but monolithic orchestration and task mismatch |
| Code quality | **4** | Readable components and some good tests; substantial duplication/configuration debt |
| Reliability | **2** | Wrong-target acceptance, stale caches, silent partial results, misleading batch success |
| ML design | **5** | Capable pretrained stack and strong alternative HOI models; weak identity/evidence integration |
| MLOps | **3** | Isolation and some provenance exist; active workflow lacks uniform versioning, monitoring, and safe job control |
| Maintainability | **3** | Large configuration surface, machine-specific setup, and intertwined concerns |

## Overall Production Readiness Score: **3.5/10**

**Bottom line:** the highest-value next step is a validated third-person reconstruction pipeline that produces the exact required MHR/object outputs, rejects wrong targets, and only reuses provenance-matched complete artifacts. Model upgrades should follow—not precede—that work.
