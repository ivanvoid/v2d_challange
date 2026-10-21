# Handoff — V2D Challenge Track 1 pipeline (state as of 2026-10-21: full rerun "v2", stages 1–5e done, stage 6 next)

v1 results (the baseline submission) are kept in `out_v1/`, `logs/stages_v1/`, `submission_v1/`; the rerun
writes `out/`, `logs/stages/`. Each stage was rerun after reading the reviews in `$V2D_DATA/review/stageN*.md`
and choosing fixes; hardest episodes first, then all.

This document is everything needed to continue the work: what the pipeline does, how to run it,
what we learned the hard way, where the results stand, and what to do next.

Repository: <https://github.com/ivanvoid/v2d_challange> (private; invite `Nvidia-V2D-Challenge` if not yet).
Code lives in the repo; data and results live in `V2D_DATA` (default `/local-data/ivan/v2d`).

---

## 1. Task in one paragraph

Track 1: 30 videos (`challenge/track_1`), each one static third-person camera at 1536×1152, 30 fps, one
person manipulating one object. For every **scored frame** we submit the person as **MHR body
parameters** (`pose [T,136]` per frame, `scales [68]` and `shape [45]` per episode) and the object as
**one mesh + per-frame rotation, translation and one scale**. The kit packs this into one Parquet
(740,780 rows) uploaded to **5 Kaggle competitions** (CD-H, CD-O, ACC-H, ACC-O, PEN). There are 10
objects, **each filmed in 3 videos** (episodes 0–2 hula hoop, 3–5 bowl, 6–8 iron, 9–11 desk, 12–14
pan, 15–17 foam block, 18–20 paint roller, 21–23 pink foam roll, 24–26 stool, 27–29 cart).

### How the scorer works (read from `v2d_submission_kit/v2dlb/mhr_submission.py`, `mhr_metrics.py`)
* **One** similarity transform (scale, rotation, translation) per episode, fitted on the **body at the
  first scored frame only**, then applied to body AND object for the whole episode. A bad first frame
  or a body that drifts later hurts everything.
* **CD-H / CD-O**: symmetric Chamfer (cm) between our and the reference body / posed object surface.
* **ACC-H**: error of the 2nd frame difference of 22 body joints. **ACC-O**: same for the object
  **translation vector** (i.e. the mesh origin) — if our mesh origin is not at the object centre,
  rotations add fake acceleration.
* **PEN**: depth of our 512 MHR hand points inside **our own** object mesh (winding numbers), mean,
  scaled by the alignment scale. **Computable locally with the kit's own function.**
* Only "valid frames inside the contact span" are scored, grouped in contiguous stretches; per-episode
  first/last scored frame is in `submission/scored_spans.json`.

---

## 2. Environment

| Thing | Where |
|---|---|
| Model stack | NVIDIA video-to-data `v2d.*` packages, one venv per model in `/opt/venvs/<model>`, selected by `V2D_BACKEND=venv` (code under `/workspace/v2d_*`, not a git repo) |
| Venvs used | `/opt/conda` python (pipeline driver), `sam3d_body` (body, reviews, QA), `foundation_pose`, `sam2`, `sam3d`, `grounding_dino`, `moge`, `anycalib`, `detectron2`, `wilor` (rendering) |
| Kit | `V2D_DATA/v2d_submission_kit` (evaluator commit `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`) + its venv `V2D_DATA/.venv-submit` |
| Weights | `V2D_DATA/weights/{anycalib,moge,detectron2,sam3d_body,grounding_dino,sam2,sam3d,foundation_pose,wilor}` |
| MHR model | `weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt` (sha256 matches the kit's required lod1 model) |
| GPUs | A40 ×3 on this host. **We use GPU 0 and GPU 2.** GPU 1 is someone else's — do not use it. |
| Disk | `out/` is ~36 GB for 30 episodes |

**Security**: a Hugging Face token was committed in plaintext in the old `setup_me.sh`. It has been
removed from the files, but **it must be revoked on huggingface.co** (if not done yet). New downloads
(e.g. CARI4D weights) need a fresh token via the `HF_TOKEN` environment variable only.

---

## 3. Pipeline

Run one stage at a time; each writes `logs/stages/<NN_name>/` with `index.html` (open this),
`episode_XXXXXX.log` (short progress), `episode_XXXXXX.debug.log` (all model output),
`episode_XXXXXX.checks.json` and `review/*.png`. Every stage **skips work whose outputs are complete
and up to date**, and the review pages reuse unchanged sheets.

| # | Stage | Code | Key outputs in `out/episode_XXXXXX/` | Time / video |
|---|---|---|---|---|
| 1 | trim to scored span ±10, AnyCalib undistort, frames, MoGe depth + intrinsics; per-frame depth-scale stabilisation (`depth_raw/` → `depth/`), static background and floor plane | `stage1_prep.py` | `frames/ depth/ depth_raw/ intrinsics_stable.json track1_meta.json background.npz floor.json` | ~6 min |
| 2 | person tracking with masks (one detectron2 pass, own tracking with fragment joining, actor choice, acceptance check) → SAM3D-Body prompted with box + person mask (body decoder) → one-identity MHR fit (kit's `mesh_to_mhr_params`, σ=2, odd-reflection edge padding) → body depth follows the stabilised MoGe slow trend (unadjusted fit kept as `mhr_fit_raw.npz`) | `stage2_track.py stage2_body.py fit_mhr.py stage2_depth.py` | `body/{person_track.json,mhr_mesh.pt,mhr_fit.npz,depth_trend.json}` | ~7 min |
| 3 | object selection (see below), incl. a size check against the same object's other videos | `stage3_select.py` | `object_track.json select/` | 1–3 min |
| 4 | object mask: SAM2 with boxes + negative points on the body (person masks), backward and forward passes averaged, per-frame quality classes (`mask_quality.py`), recovery gated by wrist distance, leak repair, acceptance check | `stage4_prompts.py stage4_mask.py sam2_box_neg.py mask_quality.py` | `masks/1/ masks_prob/ masks_conf.json masks_quality.json masks_repair.json` | 1–5 min |
| 5 | **generation**: 3 diverse clear views per video → textured SAM3D meshes (all in one process) → metric scale (FoundationPose search around a mask-based initial scale) | `stage5_pick.py stage5_mesh.py sam3d_batch.py stage5_scale.py` | `mesh/candidates.json mesh/cand_*/` | ~15 min |
| 5c | **per-video selection**: each candidate fitted on frames it was not built from, (IoU + coverage) / 2 | `stage5_check.py` | `mesh/choice.json mesh/cand_*/check.json` | ~2 min |
| 5e | **one mesh per object** (see below) + **validation** of the exported mesh on held-out frames | `stage5_object.py stage5_validate.py` | `mesh_scaled.obj mesh/object_choice.json mesh/validate.json` | ~30 min / object |
| 6 | FoundationPose tracking: anchors every 60 frames (best-quality frame), up to 3 orientation hypotheses each, forward+backward chains with hand-over through anchors; **re-find**: a visible stretch of >= 30 frames where no candidate fits gets a fresh registration (up to 2 rounds); global Viterbi selection with fit = IoU (body pixels in front of the object not counted against a pose: a handle under the hand) − colour − depth (valid pixels; off for thin objects) and translation cost relative to object size; states observed / tracked / lost / occluded | `stage6_track.py` (+ `stage6_overrides.json`) | `poses/ track/states.json` | 6–22 min |
| 7 | whole-sequence smoothing: outliers (>8 cm / >30° from local median) ignored, IoU-weighted, second-difference penalty (λ=50) on translation and rotation | `stage7_smooth.py` | `poses_smoothed/ track/smooth.json` | seconds |
| 8 | object moved into the body's metric scale (none / global / per-frame, smallest hand–object gap wins), submission files, QA, 6-frame result sheet | `stage8_scale.py build_episode.py qa_episode.py` | `poses_final/`, `submission/episodes/`, `submission/reports/` | ~2 min |

### Stage 3 details (object selection) — the part that took most iterations
* **V-wedge cover**: seen from above, two vertical planes meet behind the actor and open toward the
  camera; pixels outside are greyed **for detection only**. The wedge size follows the object's size
  prior (bigger for desk/cart). The actor's own pixels are never hidden (checked per frame).
* **Prompt ladder**: cleaned prompt from `object_priors.json` first (metadata prompts have typos:
  "stipped", "pixalated", "An metallic", "white_laptop_cart."), then the metadata prompt, shorter
  phrases, aliases.
* **Test-time views**: image + horizontal mirror + (small objects) a 2.5× zoom around the wrists.
* **Gates**: gap to the actor ≤ 0.1 actor heights, |depth − actor depth| ≤ 0.6 m, metric size inside the
  prior ×1.6, not the actor itself; score = confidence × closeness to a wrist × size consensus.
* **Cross-video size check**: every object is in 3 videos. An option (prompt + threshold) is rejected unless
  its typical size agrees within ×1.6 with the object's stage-3 size in **at least one** of its other videos
  ("at least one", not the median: one wrong video must not move the reference for the other two). Ep 19:
  the generic prompt "roller." found the white table top on all 9 frames (0.45 m vs 0.21/0.23 m in ep 18/20)
  and won on frame count; now the specific "a white paint roller with a blue handle." (0.27 m) is chosen.
  Uses the siblings' existing `object_track.json`; with none available the check is skipped.
* Of the remaining options, the one seen on the most frames wins; then the box is grown to the whole
  object on the uncovered frame.
* Manual override if ever needed: `seeds/episode_XXXXXX.json` = `{"video_frame": N, "box": [x0,y0,x1,y1]}`.

### Stage 5 details (object mesh) — generate, select, choose per object, validate
* **5 generate** (expensive, the only step that runs SAM3D): `stage5_pick.py` picks 3 clearly different
  views from clear stage-4 "ok" frames, preferring frames where the object looks complete (extent = mask's
  metric size / the object's typical size). Iron has `"view_extent": false` in `object_priors.json`: lying
  flat on the table it looks short (grazing angle), which the extent term took for a hidden part, dropping
  ep 6's sharp table view (video frame 723, v1's best-looking iron mesh) for blurry upright views. The rule
  barely changed the pan's or hoop's views; the bowl's views changed (current meshes fit well);
  `stage5_mesh.py` sends all missing candidates to `sam3d_batch.py`, ONE process that loads SAM3D once
  (the library wrapper reloaded it per mesh, ~1 min each); `stage5_scale.py` searches the metric size
  (coarse-to-fine; a size already fitted on an earlier level is reused).
* **5c select** (cheap): `stage5_check.py` fits each candidate on clear frames it was not built from:
  silhouette IoU, coverage (share of the mask the mesh covers: a missing handle shows here), depth error.
  It writes only `mesh/choice.json`. Changing a scoring rule re-runs 5c + 5e, never SAM3D.
* **5e one mesh per object**: `stage5_object.py` tests the top 2 candidates of each video by its own 5c
  ranking (6 of 9; outlines alone cannot see a wrong 3D shape, see the pan below) on frames of all
  3 videos at sizes ×0.9/1.0/1.1 per video (pass 1, so a good shape with a slightly wrong size is not
  dropped), then the 2 finalists at finer sizes around each video's best (pass 2, pass-1 trials reused).
  Score = mean over videos of the best trial's (median IoU + median coverage) / 2; colour is not used.
  The winning trial's per-frame numbers, poses and render tiles are kept: the reports, the review sheet
  and the tracking reference frame (`track_ref_frame`, used by stage 6) all come from it — nothing is
  refitted. 5e is the **only writer** of `out/<ep>/mesh_scaled.obj` (+ material).
* **Validation** (`stage5_validate.py`, run by `run_stage.sh 5e`): the exported mesh is fitted on 8 held-out
  clear frames per video; accepted if median IoU and coverage pass the thresholds. `mesh/validate.json` is
  the final acceptance summary (with the mesh file's sha1). Only accepted videos get a 5e stamp, so
  stage 6 refuses to track a rejected mesh.
* Review: `logs/stages/05c_check/index.html` (each video's own candidates) and
  `logs/stages/05e_object/index.html` (choice across videos + final validation per video). Renders are
  drawn on a background opposite to the object's mean colour (`track1/viz.py`), so a black pan is visible.
* The old 5d multi-frame refine (`stage5_refine.py`) is no longer run: 5e chose again from the original
  candidates and ignored its result.
* Pan test (ep 12–14), testing all 9: 5e chose cand_000010 of ep 14 (score 0.87), whose handle is bent up
  into a thick hook (fine from above, wrong from the side), tied with the clean straight-handle pan
  cand_000301 of ep 13 (0.87). The hook covers the mask (hand on the handle) a bit better; 5c had ranked it
  3rd of 3 in its own video. Hence the top-2 rule. Validation of that run: IoU 0.78–0.81, coverage 0.94,
  depth 1.5–2.6 cm. 5e took 32 min + 3 min validation. Single-frame fits still flip the pan (upside down /
  handle mirrored) on some frames — orientation is stage 6's job. Compare sheet:
  `logs/analysis/stage5/pan_meshes_compare.png`.
* Naming: `cand_<N>` folders use the LOCAL frame index of each video, so different videos have different
  meshes with the same name (ep 13 and ep 14 both have a `cand_000301`). Sheets and logs show
  "cand_<N> = video frame <N + frame_offset>".

### Stage 5 results of the 2026-10-20 rerun, and what was learned
* 5e accepted 9 of 10 objects (27 videos); the hoop is rejected on all 3 videos (thin ring: IoU 0.30–0.48).
* Same validation (same held-out frames) run on v1's final meshes: the new meshes are equal or better on 9 of
  10 objects (mean IoU change +0.02 to +0.03 for pan, iron, cart, bowl, pink roll, roller; stool −0.01, mainly ep
  26 which had a single candidate; hoop equally bad). Coverage is higher almost everywhere. This metric does
  not see 3D shape.
* Galleries (`logs/stages/05e_object/review/<object>_gallery.png`): good choices for bowl, pan, desk, roller,
  cart, stool, foam block, pink roll. **Iron: 5e chose a handle-less blob (ep 7, video frame 597; v1 chose the
  same) over a clean iron (ep 6, frame 723): 0.90 vs 0.87.** The same mesh at x0.99 vs x1.00 already differs by
  up to 0.02, so this is a tie the outline score cannot break.
* Tried and NOT used as tie-breaks (both are in `track1/` as diagnostics): 3D agreement with the object's other
  candidates (`shape_consensus.py`: noise ±0.005, as large as the iron's gap 0.125 vs 0.132) and colour
  agreement of the textured render with the photo (`colour_check.py`: 0.127 vs 0.127). 5e now records
  finalists within 0.03 as a **near-tie** (in `<object>.json` and on the review page). Every object has one, so
  the gallery is the check. Next step if needed: an "essential part" check (is the handle opening visible
  where the photo shows it?), as suggested in `review/stage5_fix_suggestions.md`.
* **Texture tie-break (5e):** among near-ties, a mesh whose rendered texture matches the photos clearly better
  (colour error lower by >= 0.01 after matching brightness; per frame the best of the orientations that fit
  the outline about equally) wins. Foam block: the outline winner (ep 15, video frame 147) had green paint on
  the dirt sides (0.093) vs the correct ep 16 frame 89 (0.071); a cube fits its outline in many orientations,
  so tracking relies on the texture. Iron: 0.127 vs 0.127, so the outline winner stays.
* **Thin objects in validation (hoop):** outlines compared with a 3 px tolerance (a ~4 px rim halves the plain
  IoU of a perfect fit), and the mesh is accepted if it matches on >= 3 of 8 held-out frames: a ring fitted
  from scratch on one frame often lands flat on the floor or tilted (its pixels carry the wall's depth), which
  is a pose problem for stage 6, not a mesh problem. Ep 0 passes outright, ep 1/2 with 3 and 4 frames.
* View diversity: on a static object every view looks alike, so the mask-shape filter left ep 16 with 1
  candidate of 80 clear frames (also ep 26: 1, ep 15 and 28: 2). `stage5_pick.py` now relaxes the shape
  limit 0.8 -> 0.9 -> off when fewer than 3 views remain (frames stay 30 apart). Ep 15, 16, 26 got the extra
  views without regenerating their existing meshes; ep 28 has no further view (its 16 clear frames are close).

### Running
```bash
export V2D_DATA=/local-data/ivan/v2d
cd $V2D_DATA                                   # old paths are symlinks into the repo
EPISODES="6 8 17 26 28" ./run_stage.sh 3       # one stage, chosen episodes (default: the debug set)
./run_stage_2gpu.sh 6 "0 2" "0-29"             # one stage split over GPUs 0 and 2 (tmux s6gpu0/s6gpu2)
WORKERS="0 2" EPISODES="0-29" ./run_stage_pool.sh 5   # SAM3D generation, 1 worker per GPU
WORKERS="0 0 0 2 2 2" EPISODES="0-29" ./run_stage_pool.sh 2   # stages 1-2: 3 workers per GPU; stage 4: ~6 GB each
GPUS="0 2" FROM=5 ./run_rest.sh                # stage 5 (missing) -> 5c -> 5e -> 6 -> 7 -> 8 on all 30
python3 track1/stamps.py status                # what is up to date per episode (1 2 3 4 5 5c 5e 6 7 8)
```
**Always run long jobs inside tmux** (`tmux attach -t <name>`, detach with Ctrl-b d). A background job
tied to a shell session was killed once by a session reset.

### Submission
```bash
git -C v2d_challange commit -am "..." && git -C v2d_challange push
.venv-submit/bin/python track1/pack_submission.py --commit https://github.com/ivanvoid/v2d_challange/commit/$(git -C v2d_challange rev-parse HEAD)
# -> submission/track1.parquet ; upload the SAME file to all five Track 1 competitions (Kaggle web page)
```
`run_stage.sh 8` also writes `submission/track1_partial.parquet` with a placeholder commit — for checks only.

---

## 4. Lessons learned (do not repeat)

* **Stage 5 generation: one job per GPU.** Running a test next to stage 5 ran GPU 0 out of memory and
  killed an episode. Other stages pack several workers per GPU (stages 1–2: 3, stage 4: ~6 GB each).
* **A generic prompt can win by finding the background.** Ep 19's "roller." found the white table on every
  frame and beat the real roller, which the hand often hides. Frame count alone is not identity; stage 3
  now cross-checks size against the object's other videos.
* **Re-run only what changed.** A code change marks a stage stale for every episode; when only some
  episodes' results change, check that and `stamps.py adopt` the others instead of re-running all 30.
* **Fresh registration flips orientation.** FoundationPose fitted from scratch on a single frame picks a
  180°-flipped orientation on 0–7 of 7 frames depending on the object. Never trust single-frame
  registration for orientation; carry it by tracking (stage 6).
* **Smart (motion-adaptive) anchors were worse** (v3: 5 flips vs 0 on ep 17): more anchors = more fresh
  orientation guesses, placed on the hardest frames. Kept: fixed 60-frame windows, ≤3 hypotheses,
  quality-based frame choice inside the window, hand-over through anchors. Ep 17 uses plain v2
  (`stage6_overrides.json`) because it reviewed best.
* **Smoothing does not fix wrong poses** — it hides them. Reject outliers first (stage 7 does).
* **FoundationPose's own mask-IoU check is offset** (renders the centred mesh with the un-centred pose);
  our scoring uses `pose @ inv(tf_to_center)`.
* **Texture matters for tracking** (FoundationPose compares colour); texture baking is on in stage 5.
  Colour does not rank meshes (a wrong texture must not beat a better shape); it only breaks outline
  near-ties, and only when it is clearly better (foam block: green paint on the dirt sides).
* **The outline score cannot see 3D shape.** The same mesh at x0.99 vs x1.00 differs by up to 0.02; every
  object's two 5e finalists are within 0.03. Iron: a handle-less blob beat a clean iron 0.90 vs 0.87. Look at
  the 05e gallery; 3D agreement between candidates and colour did not separate the iron (too noisy / equal).
* **A view rule must not confuse "seen at an angle" with "partly hidden".** The 'complete object' extent term
  dropped the iron's best view (lying flat it looks short); it is off for the iron (`view_extent: false`).
* **Thin objects need tolerant measures.** A hoop rim is ~4 px wide: a 1–2 px offset halves the IoU of a
  perfect fit; single-frame ring fits also land flat on the floor (wall depth on the rim pixels).
* **Stopping `run_stage.sh` mid-stage leaves the old outputs archived** (`out/<ep>/_archive/<stage>/`);
  move them back (and `stamps.py adopt`) if the old result should stay.
* **`pgrep -f` / `pkill -f` match their own command line** (and a tmux command that contains the
  pattern); wait on a PID or a tmux session instead.
* **SAM3D builds small objects 2–3× too small**; stage 5 starts the scale search from a mask-based
  estimate, otherwise the search hits its upper bound.
* **Frame numbering**: folder/file names use the LOCAL index inside the trimmed clip; review sheets show
  the VIDEO frame = local + `frame_offset` (`track1_meta.json`). `cand_000517` of ep 7 = video frame 597.
* **Review pages must not show stale sheets**: they are deleted when an episode has no current result,
  and every sheet shows when it was made.

---

## 5. Current results

### Baseline submission (commit `20b2d5c`, file `submission/track1.parquet`)

| Metric (lower is better) | Ours | Rank | Leader | CARI4D baseline |
|---|---|---|---|---|
| CD-H (cm) | 15.716 | 16 | 13.936 | 15.575 |
| CD-O (cm) | **37.420** | 14 / 14 | 18.541 | 23.162 |
| ACC-H (cm/frame²) | 0.174 | 11 | 0.137 | – |
| ACC-O (cm/frame²) | 0.127 | 9 | 0.071 | – |
| PEN (cm) | **0.1007** | 20 / 20 | 0.0000 | 0.0042 |

### Kaggle submissions (details in `docs/EXPERIMENTS.md`)
| # | Change | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|---|
| 0 baseline (v1) | — | 15.716 | 37.420 | 0.1745 | 0.1274 | 0.1007 |
| 1 p1_global | 8 episodes' object scale "none" → "global" | 15.716 | **43.294** (worse) | 0.1745 | 0.1306 | 0.0889 |

Lesson of P1: place the object where the hands touch it on scored frames, not by the body's scale.
Local PEN is calibrated (Kaggle / local ≈ 1.045).

### v2 rerun status per object (stage 5e; stage 6 not run yet)
Final ids, scores and galleries: `logs/stages/05e_object/index.html`. Validation = the exported mesh fitted on 8
held-out frames per video; compared with v1's final meshes on the same frames, v2 is equal or better on 9 of 10.

| Object | Videos | Mesh (gallery review) | Validation | Open issue |
|---|---|---|---|---|
| red bowl, white desk, cart, stool, pink foam roll | — | good | accepted | — |
| foam block | 15–17 | good shape; texture: rerun with the colour tie-break | accepted | check the new choice |
| pan | 12–14 | clean pan with straight handle (top-2 rule) | accepted | orientation (upside down / handle side): stage 6 |
| paint roller | 18–20 | roller with frame and handle | accepted | handle is thin: outline score barely sees it |
| iron | 6–8 | handle-less blob chosen (near-tie with a clean iron) | accepted | needs an "essential part" check |
| hula hoop | 0–2 | clean rings | accepted as thin object (ep 1/2: 3–4 of 8 frames fit) | single-frame ring poses fail: stage 6 |

### v1 per object (baseline run; stage 5e choice, stage 6 median mask IoU, stage 8 hand–object gap)

| Object | Videos | Mesh source (video frame) | Tracking IoU | Hand gap | Status |
|---|---|---|---|---|---|
| red bowl | 3–5 | ep 4 (641), score 0.92 | 0.87–0.92 | 0 cm | good |
| foam block | 15–17 | ep 17 (153), 0.88 | 0.86–0.88 | 0 cm | good; ep 16 penetration 1.38 cm |
| pink foam roll | 21–23 | ep 21 (330), 0.86 | 0.82–0.91 | 0–2 cm | good |
| white desk | 9–11 | ep 10 (450), 0.80 | 0.78–0.83 | 0.4–2.4 cm (ep 10 pushed with a foot) | good |
| iron | 6–8 | ep 7 (597), 0.80 | 6: 0.79, 8: 0.77, **7: 0.65** | 6/8: 2–4 cm, **7: 48 cm** | ep 7 tracking lost ~290 frames (rotated mid-air) |
| pan | 12–14 | ep 14 (141), 0.76 | 0.68–0.79 | 0.1–7 cm | ok |
| cart | 27–29 | ep 29 (128), 0.71 | 0.66–0.69 | ~0.1 cm | ok |
| stool | 24–26 | ep 26 (121), 0.69 | 0.64–0.67 | seated (no hand contact) | ok |
| paint roller | 18–20 | ep 18 (438), 0.78 | 0.68–0.75 | **14–18 cm** | mesh likely missing the handle |
| hula hoop | 0–2 | ep 0 (699), **0.43** | **0.12–0.47** | up to 123 cm | **bad**: SAM3D cannot make a thin ring |

Body: all 30 fits OK (fit error ≈1 mm, heights 1.70–1.76 m on the debug set). Body depth is 16–49 cm
deeper than MoGe on the body's own pixels (MoGe/body ratio 0.87–0.95); stage 8 handles this per episode.

---

## 6. Plan (in order; one bundle per Kaggle submission, record commit ↔ scores)

| Round | Changes | Targets | Compute |
|---|---|---|---|
| **R1** | (a) **PEN**: per frame push the object off the hand points (+2 mm, smoothed, capped ~3 cm), checked with the kit's exact PEN function; (b) **mesh origin at the object centre** (ACC-O); (c) **hold the object still** when hands are far and the mask is static (ACC-O, CD-O); (d) **body root smoothing** (σ≈5 on root translation, esp. depth) + **feet on the floor plane** from MoGe (ACC-H, CD-H) | PEN, ACC-O, ACC-H | ~1 h (stages 7–8; root smoothing works on parameters directly) |
| **R2** | DONE (P1): force "global" scale matching for the 8 "none" episodes — CD-O worse (37.42 → 43.29). Stage 8 now places the object by hand contact on scored frames (continuous factor 0.80–1.25; written, not run yet) | CD-O | minutes |
| **R3** | (b) DONE: coverage in 5c/5e + per-object choice (pan has its handle); (c) stage 6 re-find of a lost object built (ep 7), not run yet; (a) open: **hula hoop as a torus**: diameter from the mask's major axis × depth (median over 3 videos), per-frame tilt from an ellipse fit (depth resolves the two-fold ambiguity), smoothed; (b) **mesh completeness** in 5e ("mask not covered" penalty) → rollers get the handle; (c) **episode 7** tracking (more anchors where lost, re-detection; fallback hold) | CD-O, ACC-O | 3–4 h on 2 GPUs |
| **R4** | ACC-H σ test (joints σ=2 vs 3); SAM3D-Body prompted with person masks (needs a SAM2 person track) and first-scored-frame robustness | ACC-H, CD-H | ~4 h |
| **R5** | **CARI4D** (`/workspace/v2d_cari4d`, not installed: needs a container/venv + weights via a NEW `HF_TOKEN`): run on the debug set with our masks and 5e meshes; if it wins, add its joint human–object refinement after stage 8 | CD-O, PEN, CD-H | 1–2 days |

**Before every submission**: exact PEN per episode (kit function), jitter/contact/IoU/depth proxies per
episode compared with the previous round, review sheets for changed videos. Kaggle only reports totals.

Speed-ups (no quality loss, see discussion): stage 6 — load frames once, score on GPU, batch hypotheses;
stage 5 — done: 5d dropped, SAM3D loaded once per video, repeated scale trials cached, generation and
selection stamped separately, 5e tests only each video's 5c top 2 (6 meshes instead of 9, ~1/3 less 5e
time) (still open: register once then refine per scale).

### Research notes (pasted, not verified)

Based on recent state-of-the-art research in 4D Human-Object Interaction (HOI) reconstruction (such as CHOIR, PHOSA, and Monocular HOI Optimization Frameworks), here is a breakdown of advanced research methods tailored specifically to address your pipeline's remaining failure points before the freeze date.   1. Contact Penetration (PEN) & Physical ConsistencyState-of-the-Art Research Insights:Contact-Aware Joint Optimization (CHOIR Framework): Rather than performing decoupled pose tracking followed by a post-hoc push, state-of-the-art pipelines use barycentric contact anchors. When hand vertices come within a threshold distance of the object, correspondence pairs are dynamically formed.   SDF-Based Loss Functions: Modern HOI approaches compute a Signed Distance Field (SDF) around the object mesh $\mathcal{O}$. For any hand vertex $v_i \in \mathcal{H}$ inside the object (where $\text{SDF}_{\mathcal{O}}(v_i) < 0$), an explicit loss term penalizes penetration:$$\mathcal{L}_{\text{pen}} = \sum_{v_i \in \mathcal{H}} \max(0, -\text{SDF}_{\mathcal{O}}(v_i))^2$$Implementation in Your Pipeline (R1 Refinement):Instead of an isotropic (+2 mm) push in Stage 8, implement a fast SDF / Surface-Normal Repulsion Step:Compute the local surface normals $\hat{n}_j$ of the closest object face for any hand vertex $v_i$ with a negative distance (or positive winding number) inside the object.Push the hand vertex outward along $\hat{n}_j$ by $d_{\text{pen}} + \delta$ (where $\delta = 2\text{ mm}$ safety margin).Smooth the resulting hand joint trajectories using a temporal Gaussian kernel ($\sigma=2$) to ensure no high-frequency jitter is introduced into ACC-H.2. Thin Topologies & Primitive Degeneracy (Hula Hoop & Paint Roller)State-of-the-Art Research Insights:Parametric Geometric Priors vs. Mesh Generation: Generic single-view mesh generation models (like SAM3D) consistently break on non-solid or high-genus topologies (e.g., rings, tori, thin handles) due to resolution limits in implicit representations. SOTA HOI literature replaces feedforward generation with parametric fitting whenever a object category exhibits strong geometric symmetry.Aspect-Ratio & Mask Coverage Penalty: Methods handling thin objects penalize missing mask coverage explicitly during mesh selection:$$\mathcal{L}_{\text{coverage}} = 1.0 - \frac{\text{Area}(\text{Rendered Mask} \cap \text{Detected Mask})}{\text{Area}(\text{Detected Mask})}$$Implementation in Your Pipeline (R3 Refinement):Hula Hoop (Videos 0–2): Completely replace SAM3D with an analytical torus model:Fit an ellipse to the 2D segmentation mask to extract the major axis $a$ (radius $R = a / 2$) and minor axis $b$.Estimate tube thickness $r$ using the mask boundaries.Resolve the 2D-to-3D normal direction tilt ambiguity using MoGe depth map gradients inside the hoop mask region.Construct an explicit parametric OBJ torus mesh. This will instantly jump your IoU from 0.43 to > 0.85.Paint Roller (Videos 18–20): In Stage 5e, incorporate a mask non-coverage penalty. If the rendered silhouette misses more than 15% of the SAM2 mask bounding box (i.e., the handle region), lower its score so a candidate containing the full handle geometry is selected.3. Object Tracking Drift & Rotational Instability (Episode 7 Iron)State-of-the-Art Research Insights:Bidirectional Keyframe Hand-Over with Confidence Thresholding: In monocular RGB tracking, specular highlights or rapid turns cause visual trackers (like FoundationPose) to accumulate orientation drift. SOTA trackers enforce a forward-backward cycle consistency check:$$\Delta T = \Vert{} T_{t \to t+1} \cdot T_{t+1 \to t} - \mathbf{I} \Vert{}_F$$If $\Delta T$ exceeds a preset threshold, the frame is flagged as tracking failure.Kinematic Contact Holding: When the human hand is moving, but its speed relative to the object is near zero, the object trajectory is constrained to be rigidly attached to the hand:$$T_{\text{object}}^{(t)} = T_{\text{hand}}^{(t)} \cdot T_{\text{rel}}$$Implementation in Your Pipeline (R3 & R6 Refinement):Dynamic Anchor Insertion for Ep 7: Automatically trigger dense anchors (every 10–15 frames instead of 60) when the tracking confidence score or mask IoU drops below 0.65.Contact Holding / Stationarity Lock:When hand-object distance is small ($<3\text{ cm}$) and hand velocity is high, lock object pose to the hand transform $T_{\text{rel}}$.When hands are far ($>20\text{ cm}$) and the mask is static across consecutive frames, set object velocity to zero to eliminate fake acceleration noise (ACC-O).4. Origin Centering & Alignment (ACC-O Optimization)Research Recommendation for R1:In Track 1 metrics, object acceleration (ACC-O) evaluates the 2nd-difference error of the mesh origin. If the origin of your CAD model is offset from its center of mass, pure rotations introduce artificial translation vectors, severely degrading your ACC-O score.Action: Before running Stage 6 tracking, re-center all generated meshes so that their coordinate origin $(0,0,0)$ lies exactly at the bounding box centroid or center of mass:$$\mathbf{x}_{\text{centered}} = \mathbf{x} - \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i$$Update your pose transformation matrices accordingly: $T_{\text{new}} = T_{\text{old}} \cdot T_{\text{center}}^{-1}$.Recommended Execution PriorityPriorityAction ItemAffected MetricEstimated TimeExpected Gain1Re-center mesh origins to geometry centroidsACC-O~15 minsHigh drop in ACC-O error2Add Surface Normal/SDF Hand RepulsionPEN~1 hourPEN $\to 0.00x$ cm3Parametric Torus fitting for Hula HoopCD-O, ACC-O~2 hoursEp 0–2 IoU jumps from $0.43 \to >0.85$4Stationarity/Contact-holding constraintsACC-O, CD-O~1 hourRemoves jitter during static phases5Dense anchor recovery for Ep 7CD-O~2 hoursRecovers ~290 lost tracking framesFor a broader conceptual understanding of contact-aware tracking methods in human-object interactions, you can review this research video.


