# V2D Challenge — Track 1 (4D human + object reconstruction)

Reconstructs, from one static third-person video per episode, the **human body** (MHR parameters)
and the **manipulated object** (mesh + per-frame pose), and packs them into the Track 1 submission
Parquet. Everything runs as numbered stages; each stage writes review sheets so a person can check
every video before the next stage.

## Layout

| Path | What |
|---|---|
| `track1/` | all pipeline code (stages, review sheets, packing) |
| `track1/paths.py` | where code and data live (`V2D_DATA`) |
| `track1/object_priors.json` | per object: plausible size, cleaned detector prompt, aliases |
| `track1/stage6_overrides.json` | per-episode tracking settings chosen after review |
| `run_stage.sh` | run ONE stage on a set of episodes + build its review page |
| `run_stage_2gpu.sh` | the same, split over two GPUs (two tmux sessions) |
| `run_stage_pool.sh` | one per-episode stage as a queue over several workers (`WORKERS="0 0 2 2"`) |
| `run_rest.sh` | a range of stages (default 3 → 4 → 5 → 5c → 5e → 6 → 7 → 8) on all episodes, two GPUs |
| `render_results.py` | render body + object over frames (video or review sheet) |
| `legacy/` | earlier drivers kept for reference (not used by the current pipeline) |
| `docs/` | setup notes, audit and stabilisation notes |

Data is **not** in the repository. `V2D_DATA` (default `/local-data/ivan/v2d`) must contain:
`challenge/` (the released dataset), `weights/`, `v2d_submission_kit/` (official kit),
and receives `out/` (per-episode results), `logs/stages/` (review pages), `submission/`.

## Dependencies

* The NVIDIA video-to-data model stack (`v2d.*` Python packages, one venv per model under
  `/opt/venvs/<model>`, `V2D_BACKEND=venv`): AnyCalib, MoGe, detectron2, SAM3D-Body, Grounding DINO,
  SAM2, SAM3D, FoundationPose. See `docs/setup_me.sh` and `docs/fetch_repos.sh`.
* The official submission kit (`v2d_submission_kit/`, evaluator commit
  `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`) and a venv with its requirements at
  `$V2D_DATA/.venv-submit`.

## Stages

| # | Stage | Main outputs (`out/episode_XXXXXX/`) |
|---|---|---|
| 1 | trim to the scored span, AnyCalib undistortion, frames, MoGe depth (per-frame scale stabilised), stable intrinsics, static background and floor plane | `frames/ depth/ intrinsics_stable.json background.npz floor.json` |
| 2 | person tracking with masks (detectron2, fragment joining, actor choice) → SAM3D-Body (box + person mask) → one-identity MHR fit → body depth follows the stabilised MoGe trend | `body/mhr_fit.npz` |
| 3 | object selection: V-wedge background cover, prompt ladder + mirror + hand zoom views, gates on size / depth / distance to the actor, size-consistent supports; an option must also agree in size (×1.6) with the same object in at least one of its other videos; whole-object box | `object_track.json` |
| 4 | SAM2 masks: boxes from stage 3 + negative points on the body, backward/forward passes averaged, per-frame quality classes, gated recovery and leak repair, acceptance check | `masks/1/ masks_quality.json` |
| 5 | **generation** (expensive): 3 diverse clear views per video → SAM3D textured meshes (one process, model loaded once) → metric scale search (each size fitted once) | `mesh/candidates.json mesh/cand_*/` |
| 5c | **per-video selection** (cheap): every candidate fitted on frames it was not built from, scored by (IoU + coverage) / 2 | `mesh/choice.json` |
| 5e | **one mesh per physical object**: each video's top 2 candidates by 5c, at sizes ×0.9/1.0/1.1 on frames of all 3 videos, finalists refined; then the exported mesh is validated on held-out frames of each video (accept: median IoU and coverage thresholds) | `mesh_scaled.obj mesh/object_choice.json mesh/validate.json` |
| 6 | FoundationPose tracking: anchors, bidirectional chains with hand-over, re-finding a lost object (fresh registration where nothing fits), global (Viterbi) selection with occlusion-aware IoU, depth and size-aware motion; states observed / tracked / lost / occluded | `poses/ track/states.json` |
| 7 | whole-sequence smoothing with outlier rejection | `poses_smoothed/` |
| 8 | object scale matched to the body, submission files, QA | `poses_final/`, `submission/episodes/` |

## Running

```bash
export V2D_DATA=/local-data/ivan/v2d
./run_stage.sh 1                                   # stage 1 on the debug set (6 8 17 26 28)
EPISODES="0-29" ./run_stage.sh 2                   # one stage, chosen episodes
./run_stage_2gpu.sh 6 "0 2" "0-29"                 # one stage on two GPUs
WORKERS="0 2" EPISODES="0-29" ./run_stage_pool.sh 5   # SAM3D generation (1 worker per GPU: memory)
EPISODES="0-29" ./run_stage.sh 5c                  # per-video selection (re-run alone after a scoring change)
EPISODES="0-29" ./run_stage.sh 5e                  # one mesh per object (objects of these episodes) + validation
FROM=3 TO=8 GPUS="0 2" ./run_rest.sh               # stages 3 → 4 → 5 → 5c → 5e → 6 → 7 → 8, two GPUs (tmux!)
python3 track1/stamps.py status                    # which stages are up to date per episode
```
A stage is skipped only when its stamp (`out/<ep>/stamps/<stage>.json`) matches its current code,
settings, inputs and upstream stamps; otherwise its old outputs move to `out/<ep>/_archive/<stage>/`
and it runs from scratch. Upstream stages must be up to date first. `FORCE=1` re-runs anyway.
Review pages: `$V2D_DATA/logs/stages/<NN_stage>/index.html`.

## Local evaluation and experiments

`track1/eval_local.py` reads a packed parquet exactly as the scorer does (scored frames only) and
reports per episode: PEN (the scorer's method), hand–object gap and contact on scored frames,
jitter of the 22 scored joints and of the object translation per stretch, object size across its
3 videos. `compare` shows two runs side by side. Every Kaggle submission is recorded in
`docs/EXPERIMENTS.md` with its commit, scores and local values.

## Submission

```bash
.venv-submit/bin/python track1/pack_submission.py --commit https://github.com/ivanvoid/v2d_challange/commit/<sha>
# -> $V2D_DATA/submission/track1.parquet: upload the same file to all five Track 1 competitions
```
