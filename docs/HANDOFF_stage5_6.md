# Handoff — analyse the Stage 5 results, then run Stage 6 (2026-10-21)

Read first: `docs/HANDOFF.md` (pipeline, lessons, results) and `README.md`. This note only covers the open
work. Code: `$V2D_DATA/v2d_challange` (`V2D_DATA=/local-data/ivan/v2d`); results: `$V2D_DATA/out`,
review pages: `$V2D_DATA/logs/stages/<NN_stage>/index.html`. Times below are the machine clock.

## Rules the user set (keep them)
* GPUs **0 and 2 only**. GPU 1 belongs to someone else (Triton, ~16 GB).
* Long jobs in **tmux**. Stage 5e and stage 6 use ~30 GB each: **one job per GPU**.
* Go stage by stage; hardest episodes first, then all. Before a stage: read `$V2D_DATA/review/stageN*.md`,
  verify its claims on data, ask the user which fixes to include.
* **Don't redo completed work.** Stamps (`track1/stamps.py`) decide reruns; when a code change does not change
  an episode's result, `python3 track1/stamps.py adopt --stage N --episodes ...` instead of rerunning.
* Fully automatic and reproducible: no manual mesh picks or per-episode hand edits.
* Simple, short explanations for the user. Local commits only; never push unless asked.
* `pgrep -f` / `pkill -f` match their own command line: wait on a PID or a tmux session.
* Stopping `run_stage.sh` mid-stage leaves the old outputs in `out/<ep>/_archive/<stage>/`.
* The image viewer can show an outdated file: copy images to view into `logs/stages/_viewer/`, never into a
  stage's `review/` folder.

## 1. DONE (04:21): 5e rerun for all 10 objects — all 30 videos accepted
Final choices: pan cand_000301 ep 13 (unchanged), bowl cand_000454 ep 5 (texture), hoop cand_000220 ep 0
(unchanged), foam block cand_000103 ep 16 (frame 143, correct texture), pink roll cand_000250 ep 21 (texture),
iron cand_000517 ep 7 (unchanged, blob), desk cand_000334 ep 10, roller cand_000172 ep 20, cart cand_000250
ep 29 (all unchanged), stool cand_000070 ep 26 (frame 145, an added view; good). The rest of this section is
the record of the run.
`FROM=5e TO=5e GPUS='0 2' ./run_rest.sh`, started 01:38, ~25 min per object, both GPUs: expect it done
around 04:15. Log: `/tmp/opencode/s5e_all.log`, per object `logs/stages/05e_object/<object>.log`.
Why: 5e got a new rule — near-tied finalists (within 0.03) are decided by texture agreement with the photos
when it is clearly better (colour error lower by >= 0.01). Already done in this run:
* black_pan: unchanged (cand_000301 of ep 13, video frame 366 — clean pan with a straight handle)
* big_red_bowl: **switched** by texture to cand_000454 of ep 5 (video frame 534): colour error 0.064 vs 0.106
* hula_hoop: unchanged (cand_000220 of ep 0, video frame 300)

Previous choices (before this rerun), to compare against:

| Object | Previous choice | Expected / to check |
|---|---|---|
| foam_grass_block | cand_000076 of ep 15 (green paint on the dirt sides) | should switch to cand_000049 of ep 16 (colour 0.071 vs 0.093) |
| iron | cand_000517 of ep 7 (handle-less blob) | stays (colour 0.127 vs 0.127); a clean iron is cand_000643 of ep 6 |
| paint_roller | cand_000172 of ep 20 (roller with frame and handle) | near-tie with cand_000187 of ep 19 (a plain cylinder) |
| pink_foam_roll | cand_000436 of ep 22 | near-tie with cand_000250 of ep 21 |
| white_desk | cand_000334 of ep 10 | near-tie with cand_000370 of ep 10 |
| white_laptop_cart | cand_000250 of ep 29 | near-tie with cand_000100 of ep 29 |
| short_wood_stool | cand_000040 of ep 24 (good) | near-tie with cand_000262 of ep 26 |

## 2. Analyse the Stage 5 results (when `s5e_all` is done)
1. `python3 track1/stamps.py status` — 5e should be `ok` for all 30 (the hoop is accepted as a thin object).
2. Open `logs/stages/05e_object/index.html`: per object the choice, near-tie, validation per video, the
   **gallery** (all candidates, 4 views each, 5c rank, 5e score) and validation sheets.
3. For every object: did the texture rule switch the choice? Is the new mesh at least as good in shape
   (gallery)? A texture win must not bring a worse shape (e.g. roller: a plain cylinder must not beat the
   roller with its handle). Report switches with the colour numbers from `<object>.log`
   ("near-tie broken by texture").
4. Known weak spots (do not hand-pick; report):
   * iron: handle-less blob (no automatic signal separates it; idea: "essential part" check — is the handle
     opening visible where the photo shows it?)
   * roller: the handle is thin, the outline score barely sees it
   * hoop: meshes are fine; single-frame ring fits often land flat on the floor (pose = stage 6)
   * ep 28 (cart) has only 2 candidates (no further clear view)
5. Update `docs/HANDOFF.md` section "v2 rerun status per object" (foam block row says "check the new choice")
   and commit locally.

Stage 5 facts for the analysis:
* v1 vs v2 on the same held-out frames (v1's final meshes revalidated in `/tmp/opencode/v1val/`): v2 equal or
  better on 9 of 10 objects; this metric only sees outlines.
* Tried and NOT used: 3D shape agreement between candidates (`track1/shape_consensus.py`, noise ±0.005) and
  colour on the iron (`track1/colour_check.py`, equal).
* `review/stage5_fix_suggestions.md` (other reviewer) — points 1–2 (v1 candidates in the pool, old/new
  matrix) not done; 4, 5, 7, 8 done.

## 3. Run Stage 6 (code changed, NOT run yet — commit `cacf4bd`)
Changes (chosen by the user from `review/stage6.md`, verified on v1 data: ep 1 had 330 of 331 frames
uncertain, ep 7 lost the object for 286 frames, anchors 455 frames apart):
1. **Re-find a lost object**: after normal tracking, a visible stretch of >= `--lost_after` (30) frames where
   NO candidate fits (IoU < 0.5) gets a fresh registration on its best frame and is tracked both ways (up to
   `--recover_rounds` 2). Re-found anchors are marked `refound` in `track/states.json`.
2. **Hand contact**: model pixels where the body is IN FRONT of the object (projected MHR body, MoGe depth
   5 cm closer than the object, within 50% of the object box around it) do not count against a pose — a pan
   handle under the holding hand is not penalised. Stored per frame as packed bits.
3. **Size-aware motion + depth**: translation cost per unit = 0.25 x object size; fit = IoU − colour/255 −
   0.2 x min(depth error / 10 cm, 1), depth error on valid depth pixels only; depth off for thin objects.
4. **States**: observed / tracked (<= 30 frames from an observed frame) / lost / occluded (small mask). Review
   checks: lost frames <= 5 % (+ longest stretch), translation steps > half the object size, rotation flips.
   Unused options removed (`--max_gap` etc.). `stage6_overrides.json`: ep 17 keeps `anchor_select: vis`.

None of this has run. Risks to watch: run time of the re-find step, the occluder letting a wrong pose hide
behind the body, the depth weight (0.2) and the size-relative translation cost changing the Viterbi path.

Test first (hardest), one job per GPU:
```bash
cd $V2D_DATA
tmux new -d -s s6a "GPU=0 EPISODES='7 13' ./run_stage.sh 6 > /tmp/opencode/s6_gpu0.log 2>&1"
tmux new -d -s s6b "GPU=2 EPISODES='1' ./run_stage.sh 6 > /tmp/opencode/s6_gpu2.log 2>&1"
```
* ep 7 (iron): v1 lost 286 frames — check the re-find lines and the lost-frame count.
* ep 13 (pan): handle side / upside down — look at the track sheet frames.
* ep 1 (hoop): v1 uncertain for all 330 frames; ring poses are hard (an ellipse-based ring pose was offered
  and NOT chosen by the user).
Compare with v1: `out_v1/episode_0000NN/track/states.json`, `logs/stages_v1/06_track/`. Per-episode log:
`logs/stages/06_track/episode_0000NN.log` (look for "re-find round", "after re-finding", "stage 6 OK").
Review page: `logs/stages/06_track/index.html`. Show the user the numbers and the sheets, fix, then run all 30:
`FROM=6 TO=6 GPUS='0 2' ./run_rest.sh` (in tmux). Then stage 7 (CPU, seconds) and stage 8 (new contact-based
placement, written but untested) follow the same per-stage process (`review/` has no stage 7/8 notes yet;
stage 7 needed no fix last time).
