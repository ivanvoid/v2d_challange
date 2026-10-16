#!/usr/bin/env bash
# =============================================================================
# run_all.sh — batch-run the ego-wilor pipeline over track_1 (exo videos).
#
# For every episode it:
#   1. looks up the object_prompt from episodes_metadata.jsonl
#   2. pass 1: run_ego_wilor (ref frame 0) to populate wilor_raw/
#   3. auto-picks the median frame that has hands (pick_reference_frame.py)
#   4. pass 2: re-run with that --reference_frame (reuses cached wilor_raw)
#   5. auto-recovers the known SAM2 crash, then retries pass 2 once
#   6. renders body_object_overlay.mp4
#
# Safe to re-run: finished episodes are skipped, in-progress work is reused,
# and one episode's failure never aborts the batch.
#
# Usage:
#   ./run_all.sh                 # all 30 track_1 episodes
#   EPISODES="3 11 14" ./run_all.sh
#   EPISODES="0-9" ./run_all.sh
#   FORCE=1 ./run_all.sh         # ignore the resume check, recompute
#
# Knobs (env):
#   EPISODES        range "A-B" and/or space list "N N N"  (default 0-29)
#   FORCE           1 = recompute even if overlay exists    (default 0)
#   UNDISTORT       1 = pass --undistort                     (default 1)
#   GRAVITY_ALIGN   1 = pass --gravity_align                 (default 0)
#   BASEPATH        repo root                 (default /local-data/ivan/v2d)
# =============================================================================
set -uo pipefail

# --- config ------------------------------------------------------------------
BASEPATH="${BASEPATH:-/local-data/ivan/v2d}"
EPISODES="${EPISODES:-0-29}"
FORCE="${FORCE:-0}"
UNDISTORT="${UNDISTORT:-1}"
GRAVITY_ALIGN="${GRAVITY_ALIGN:-0}"

export V2D_BACKEND=venv
export WEIGHTS="$BASEPATH/weights"
OUT="$BASEPATH/out"
VIDEO_DIR="$BASEPATH/challenge/track_1/videos/chunk-000/observation.images.exo_camera"
META="$BASEPATH/challenge/track_1/meta/episodes_metadata.jsonl"
HERE="$(cd "$(dirname "$0")" && pwd)"
SUMMARY="$OUT/_summary.tsv"

unset CONFIG_FILE CHECKPOINT_FILE
mkdir -p "$OUT"

# sam3d dino-weights fix (see setup_me.sh): some venvs look under /data/weights_dir.
if [[ -d "$WEIGHTS/sam3d/torch_home/hub/checkpoints" ]]; then
    mkdir -p /data/weights_dir/sam3d/torch_home/hub/checkpoints 2>/dev/null || true
    cp -n "$WEIGHTS"/sam3d/torch_home/hub/checkpoints/dinov2_*.pth \
          /data/weights_dir/sam3d/torch_home/hub/checkpoints/ 2>/dev/null || true
fi

# --- expand EPISODES into a list of integers ---------------------------------
expand_episodes() {
    local tok lo hi
    for tok in $1; do
        if [[ "$tok" == *-* ]]; then
            lo="${tok%-*}"; hi="${tok#*-}"
            for ((n = lo; n <= hi; n++)); do echo "$n"; done
        else
            echo "$tok"
        fi
    done
}

prompt_for() {  # $1 = episode index -> prints object_prompt (may be empty)
    python3 -c "
import json, sys
idx = int(sys.argv[2])
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get('episode_index') == idx:
        print(r.get('object_prompt', '')); break
" "$META" "$1"
}

run_pipeline() {  # $1 = video  $2 = outdir  $3 = prompt  $4 = reference_frame
    local extra=()
    [[ "$UNDISTORT" == "1" ]] && extra+=(--undistort)
    [[ "$GRAVITY_ALIGN" == "1" ]] && extra+=(--gravity_align)
    python -m v2d.pipelines.run_ego_wilor \
        --video_path "$1" \
        --output_dir "$2" \
        --object_prompt "$3" \
        --reference_frame "$4" \
        --anycalib_weights        "$WEIGHTS/anycalib" \
        --moge_weights            "$WEIGHTS/moge" \
        --wilor_weights           "$WEIGHTS/wilor" \
        --sam2_weights            "$WEIGHTS/sam2" \
        --grounding_dino_weights  "$WEIGHTS/grounding_dino" \
        --sam3d_weights           "$WEIGHTS/sam3d" \
        --foundation_pose_weights "$WEIGHTS/foundation_pose" \
        "${extra[@]}"
}

recover_sam2() {  # $1 = outdir
    [[ -f "$1/sam2_prompts.json" ]] || return 1
    /opt/venvs/sam2/bin/python -m v2d.sam2.lib.video_to_masks \
        --video_path   "$1/frames" \
        --prompts_path "$1/sam2_prompts.json" \
        --weights_dir  "$WEIGHTS/sam2" \
        --masks_dir    "$1/masks"
}

render_overlay() {  # $1 = outdir
    /opt/venvs/wilor/bin/python "$HERE/render_results.py" \
        --frames_dir       "$1/frames" \
        --intrinsics       "$1/intrinsics_stable.json" \
        --body_mesh        "$1/mhr_mesh.pt" \
        --object_mesh      "$1/mesh_scaled.obj" \
        --object_poses_dir "$1/poses_smoothed" \
        --output           "$1/body_object_overlay.mp4" \
        --fps 30
}

record() {  # $1=N $2=ref $3=status $4=seconds
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$SUMMARY"
    printf '[episode %06d] ref=%s  %s  (%ss)\n' "$1" "$2" "$3" "$4"
}

# --- main loop ---------------------------------------------------------------
printf 'episode\tref_frame\tstatus\tseconds\n' > "$SUMMARY"

for N in $(expand_episodes "$EPISODES"); do
    EP="$(printf 'episode_%06d' "$N")"
    VIDEO="$VIDEO_DIR/$EP.mp4"
    ODIR="$OUT/$EP"
    LOG="$ODIR/run.log"
    start=$SECONDS

    if [[ ! -f "$VIDEO" ]]; then
        record "$N" "-" "FAIL(no-video)" 0
        continue
    fi

    if [[ -f "$ODIR/body_object_overlay.mp4" && "$FORCE" != "1" ]]; then
        record "$N" "-" "SKIP(done)" 0
        continue
    fi

    mkdir -p "$ODIR"
    echo "===== $(date -Is) $EP =====" >> "$LOG"

    PROMPT="$(prompt_for "$N")"
    if [[ -z "$PROMPT" ]]; then
        record "$N" "-" "FAIL(no-prompt)" $((SECONDS - start))
        continue
    fi
    echo "prompt: $PROMPT" >> "$LOG"

    # Pass 1 — fill wilor_raw/. Expected to exit at the "no hands in ref frame"
    # check; we only need the per-frame detections, so ignore its exit status.
    echo "--- pass 1 (ref 0) ---" >> "$LOG"
    run_pipeline "$VIDEO" "$ODIR" "$PROMPT" 0 >> "$LOG" 2>&1 || true

    # Pick the reference frame.
    REF="$(python3 "$HERE/pick_reference_frame.py" "$ODIR/wilor_raw" 2>>"$LOG")"
    if [[ -z "$REF" ]]; then
        record "$N" "-" "FAIL(no-hands)" $((SECONDS - start))
        continue
    fi

    # Pass 2 — real run at the chosen reference frame.
    echo "--- pass 2 (ref $REF) ---" >> "$LOG"
    if ! run_pipeline "$VIDEO" "$ODIR" "$PROMPT" "$REF" >> "$LOG" 2>&1; then
        echo "--- pass 2 failed; attempting sam2 recovery ---" >> "$LOG"
        if recover_sam2 "$ODIR" >> "$LOG" 2>&1 \
           && run_pipeline "$VIDEO" "$ODIR" "$PROMPT" "$REF" >> "$LOG" 2>&1; then
            : # recovered
        else
            record "$N" "$REF" "FAIL(pipeline)" $((SECONDS - start))
            continue
        fi
    fi

    # Render overlay (best-effort; pipeline success is the real deliverable).
    echo "--- render ---" >> "$LOG"
    if render_overlay "$ODIR" >> "$LOG" 2>&1; then
        record "$N" "$REF" "DONE" $((SECONDS - start))
    else
        record "$N" "$REF" "DONE(no-overlay)" $((SECONDS - start))
    fi
done

# --- summary -----------------------------------------------------------------
echo
echo "===== summary ($SUMMARY) ====="
column -t -s $'\t' "$SUMMARY"
echo
awk -F'\t' 'NR>1 {c[$3]++} END {for (s in c) printf "  %-18s %d\n", s, c[s]}' "$SUMMARY"
failed="$(awk -F'\t' 'NR>1 && $3 ~ /FAIL/ {printf "%d ", $1}' "$SUMMARY")"
[[ -n "$failed" ]] && echo && echo "re-run failures with:  EPISODES=\"$failed\" ./run_all.sh"
exit 0
