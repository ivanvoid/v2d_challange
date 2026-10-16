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
| `run_rest.sh` | stage 5 (missing) → 5e → 6 → 7 → 8 on all episodes, two GPUs |
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
| 1 | trim to the scored span, AnyCalib undistortion, frames, MoGe depth, stable intrinsics | `frames/ depth/ intrinsics_stable.json` |
| 2 | person track (detectron2) → SAM3D-Body (real intrinsics) → one-identity MHR fit | `body/mhr_fit.npz` |
| 3 | object selection: V-wedge background cover, prompt ladder + mirror + hand zoom views, gates on size / depth / distance to the actor, whole-object box | `object_track.json` |
| 4 | SAM2 masks: boxes from stage 3 + negative points on the body, gap fill by re-detection, leak repair | `masks/1/` |
| 5 | two SAM3D textured meshes per video, metric scale search, cross-frame check, multi-frame scale refine | `mesh/` |
| 5e | one mesh per physical object: every candidate tested on frames of all its 3 videos | `mesh_scaled.obj`, `mesh/object_choice.json` |
| 6 | FoundationPose tracking: anchors, bidirectional chains with hand-over, global (Viterbi) selection | `poses/ track/states.json` |
| 7 | whole-sequence smoothing with outlier rejection | `poses_smoothed/` |
| 8 | object scale matched to the body, submission files, QA | `poses_final/`, `submission/episodes/` |

## Running

```bash
export V2D_DATA=/local-data/ivan/v2d
./run_stage.sh 1                                   # stage 1 on the debug set (6 8 17 26 28)
EPISODES="0-29" ./run_stage.sh 2                   # one stage, chosen episodes
./run_stage_2gpu.sh 6 "0 2" "0-29"                 # one stage on two GPUs
GPUS="0 2" ./run_rest.sh                           # stages 5 → 5e → 6 → 7 → 8, two GPUs
```
Each stage skips work whose output is complete and up to date. Review pages:
`$V2D_DATA/logs/stages/<NN_stage>/index.html`.

## Submission

```bash
.venv-submit/bin/python track1/pack_submission.py --commit https://github.com/ivanvoid/v2d_challange/commit/<sha>
# -> $V2D_DATA/submission/track1.parquet: upload the same file to all five Track 1 competitions
```
