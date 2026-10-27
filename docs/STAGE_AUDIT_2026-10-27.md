# Track 1 cross-stage audit — 2026-10-27

**Purpose:** find errors large enough to matter to the challenge metrics, rather
than make another small proxy-only submission. Source and reproducible JSON:
`track1/stage_audit.py` and `$V2D_DATA/experiments/hoop_mask_v1/stage_audit.json`.
All status statistics below refer to the **exact scored video frames**, mapped
through `track1_meta.json.frame_offset`, unless stated as full-video counts.

## Most important findings

| Episode(s) | Observed issue | Evidence | Action |
|---|---|---|---|
| 0, 1 hula hoop | Stage 4 accepts masks while stage 6 cannot fit the ring | Scored-frame stage-6 median IoU **0.484 / 0.355**; lost **8.3% / 32.8%**. The full trimmed videos have 332/630 and 169/331 lost poses; do not confuse full-video with scored-frame counts. | Shape-aware mask diagnostics, then alternative ring registration/tracking on versioned outputs. |
| 0 hula hoop, late floor interval | SAM2 fills the ring's centre; not caused by later repair | Original frame 0 has hole/outer-contour ratio 0.64; frame 600 has no hole. Raw and final masks are identical. Stage 4 nevertheless reports `accepted: true`, zero scored-frame mask `lost`. | Narrow isolated pilot corrected 48 late masks (38/169 scored frames) by **removing** interior pixels; no new visible pixels synthesized. |
| 0 hoop mask pilot | Ring-shaped target better matches the existing ring geometry | Median existing-pose IoU on 48 altered masks 0.279→0.636; all 48 changed frames improve this consistency check. | An isolated stage-6 re-track runs in tmux `v2d-hoop-stage6`. This saved-pose metric is **not** independent ground truth; compare unchanged frames and output trajectories before promotion. |
| 2 hoop | Geometry-only torus replacement is not safe | One-size 0.66 m torus fits ep 0 held-out masks 0.668→0.721 but ep 2 0.226→0.000 under fixed poses. Ep 1 has no high-fit anchors. User estimates physical diameter **67–72 cm by shoe comparison**, soft prior. | Do not swap geometry without testing per-video pose/scale. The circle is obscured while held; floor-supported anchors can constrain ep 0 opening and ep 2 end. |
| 15–17 foam block and 8 iron | Penetration despite high 2D fit | Local PEN ~0.417/0.458/0.264/0.128 cm, while stage-6 median scored IoU ≥0.74. | Existing `penfix_smooth` improved reported PEN 0.04506→0.03349 cm but did not solve deep palm overlap. Do not expect image tracker swaps to fix physical contact. |

## Other checks and fixes

- **Stage-8 QA/render failure propagation:** `run_stage.sh` previously proceeded
  to `stage 8 OK` without checking the QA pipeline's producer exit or result
  rendering. Now those failures prevent stage-8 success; `bash -n` and
  `track1/tests/test_stage_contracts.py` cover the contract. QA **warning flags**
  remain warnings; they are not converted into hard failures by this change.
- **Camera calibration:** `/workspace/v2d_anycalib/lib/video_to_calibration.py`
  already samples 16 frames evenly over each video, takes a robust median and
  supplies one per-video camera calibration. Episode 24 logs frames 0..261
  across the whole clip. No single-frame camera-parameter problem was found.
  Do not rerun stage 1 without an independent calibration residual.
- **Stool prior:** the user confirmed episodes 24 and 26 are stationary for
  their whole videos, episode 25 is not. Multi-frame stationary fitting on
  held-out visible frames improved IoU 0.819→0.833 and 0.774→0.786.
  The separate full submission candidate
  `experiments/stool_static_plus_smooth_t150/track1.parquet` reportedly
  scored ACC-O 0.11673 vs 0.11826 for smoothing alone. These are small gains;
  concentrate future effort on hoop tracking.

## Acceptance before a hoop submission

1. Compare the isolated `experiments/hoop_mask_v1/out/episode_000000/track/states.json`
   to production on the **169 scored** and all 630 local frames. A changed
   mask's own IoU is not independent validation; unchanged visible frames must
   not regress.
2. Verify registration and object silhouette at the repaired final floor
   interval, at the unobscured opening, and while the hoop is held. Inspect
   trajectories and handoffs, not just six selected frames.
3. If stage 6 improves without damaging unaffected frames, run isolated stages
   7 and 8, pack the episode and compare with the current exact-frame control.
   Only object pose/scale is allowed to change in the full submission.
4. Validate the complete 740,780-row packed Parquet, preserve every other
   episode and the body's parameters, then request Kaggle scoring. Do not
   claim CD-O or ACC-O improvements from local IoU/self-acceleration alone.

**Provenance:** user-provided physical priors, local images, public scorer and
control proxies; no organizer reference poses/meshes were available. Production
`out/`, `submission/`, baseline Parquets and the official scorer were not edited.
