#!/usr/bin/env bash
# =============================================================================
# run_stage.sh — run ONE pipeline stage on a set of episodes, then build its review.
#
#   ./run_stage.sh 1                      # stage 1 on the debug set (6 8 17 26 28)
#   EPISODES="0-29" ./run_stage.sh 1      # stage 1 on everything
#
# Output: logs/stages/<NN_name>/
#   batch.log                       everything this script printed
#   episode_XXXXXX.log              short progress log per episode
#   episode_XXXXXX.debug.log        full model output per episode
#   episode_XXXXXX.checks.json      automatic checks
#   review/                         review images
#   index.html                      all episodes side by side — open this
# Results go to out/episode_XXXXXX/ (stages skip work that is already complete).
# =============================================================================
set -uo pipefail

STAGE="${1:?usage: run_stage.sh <stage number>}"
CODE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"          # this repository
BASEPATH="${V2D_DATA:-${BASEPATH:-/local-data/ivan/v2d}}"       # data: challenge/, weights/, out/, logs/, ...
export V2D_DATA="$BASEPATH"
EPISODES="${EPISODES:-6 8 17 26 28}"
GPU="${GPU:-0}"

export V2D_BACKEND=venv
export CUDA_VISIBLE_DEVICES="$GPU"
unset CONFIG_FILE CHECKPOINT_FILE
T1="$CODE/track1"
OUT="$BASEPATH/out"
WEIGHTS="$BASEPATH/weights"
SPANS="$BASEPATH/submission/scored_spans.json"
PY_PIPE=python
PY_REVIEW=/opt/venvs/sam3d_body/bin/python
PY_SUB="$BASEPATH/.venv-submit/bin/python"
PY_FP=/opt/venvs/foundation_pose/bin/python
PY_RENDER=/opt/venvs/wilor/bin/python
SUB="$BASEPATH/submission"

stage4_mask() {  # $1 = episode: prompts (boxes + negative body points) -> SAM2 -> gap fill -> SAM2 -> leak repair
    local N="$1" EP PLOG DLOG rc
    EP="$(printf 'episode_%06d' "$N")"; PLOG="$SDIR/$EP.log"; DLOG="$SDIR/$EP.debug.log"
    p4() { "$PY_REVIEW" "$T1/stage4_prompts.py" --episode "$N" --out_root "$OUT" --weights "$WEIGHTS" --mode "$1" \
               2>> "$DLOG" | tee -a "$PLOG"; return ${PIPESTATUS[0]}; }
    sam() { "$PY_PIPE" "$T1/stage4_mask.py" --episode "$N" --out_root "$OUT" --weights "$WEIGHTS" \
               --debug_log "$DLOG" 2>&1 | tee -a "$PLOG"; return ${PIPESTATUS[0]}; }
    p4 init || return 1
    sam || return 1
    p4 fill; rc=$?
    if [[ $rc -eq 10 ]]; then sam || return 1; elif [[ $rc -ne 0 ]]; then return 1; fi
    p4 repair || return 1
}

stage8_final() {  # $1 = episode index: scale match -> build -> QA -> result frames
    local N="$1" EP W PLOG DLOG OFF FIRST LAST FR
    EP="$(printf 'episode_%06d' "$N")"; W="$OUT/$EP"
    PLOG="$SDIR/$EP.log"; DLOG="$SDIR/$EP.debug.log"
    OFF=$(python3 -c "import json;print(json.load(open('$W/track1_meta.json'))['frame_offset'])")
    FIRST=$(python3 -c "import json;print(json.load(open('$W/track1_meta.json'))['scored_first'])")
    LAST=$(python3 -c "import json;print(json.load(open('$W/track1_meta.json'))['scored_last'])")
    mkdir -p "$SUB/episodes" "$SUB/reports" "$SDIR/review"
    "$PY_REVIEW" "$T1/stage8_scale.py" --episode "$N" --out_root "$OUT" --model "$MHR_MODEL" \
        2>> "$DLOG" | tee -a "$PLOG"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || return 1
    echo "[$(date +%T)]   build submission files" | tee -a "$PLOG"
    "$PY_SUB" "$T1/build_episode.py" --episode "$N" --ep_dir "$W" --out_dir "$SUB/episodes" \
        --object_poses "$W/poses_final" --frame_offset "$OFF" >> "$DLOG" 2>&1 || return 1
    tail -1 "$DLOG" | tee -a "$PLOG"
    echo "[$(date +%T)]   QA" | tee -a "$PLOG"
    "$PY_REVIEW" "$T1/qa_episode.py" --episode "$N" --ep_dir "$W" --sub_dir "$SUB/episodes" --spans "$SPANS" \
        --model "$MHR_MODEL" --frame_offset "$OFF" --out "$SUB/reports/$EP.json" \
        --export_body "$W/body/submitted_body.pt" 2>> "$DLOG" | grep '^\[qa' | tee -a "$PLOG"
    FR="$(python3 -c "import sys; a,b=int(sys.argv[1]),int(sys.argv[2]); print(','.join(str(round(a+(b-a)*i/5)) for i in range(6)))" "$((FIRST - OFF))" "$((LAST - OFF))")"
    echo "[$(date +%T)]   render 6 result frames" | tee -a "$PLOG"
    "$PY_RENDER" "$CODE/render_results.py" --frames_dir "$W/frames" --intrinsics "$W/intrinsics_stable.json" \
        --body_mesh "$W/body/submitted_body.pt" --object_mesh "$SUB/episodes/${EP}_object.obj" \
        --object_poses_dir "$W/poses_final" --frames "$FR" \
        --title "$EP submitted body (blue) + object (orange)" --output "$SDIR/review/${EP}_result.png" >> "$DLOG" 2>&1
    echo "[$(date +%T)] $EP: stage 8 OK" | tee -a "$PLOG"
}
MHR_MODEL="$WEIGHTS/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt"
SMOOTH_SIGMA="${SMOOTH_SIGMA:-2.0}"

# step <episode log> <debug log> <name> <done-test> -- command...   (timed; output -> debug log)
step() {
    local plog="$1" dlog="$2" name="$3" done="$4"; shift 5
    if eval "$done"; then echo "[$(date +%T)]   [skip] $name (already complete)" | tee -a "$plog"; return 0; fi
    echo "[$(date +%T)]   [run ] $name" | tee -a "$plog"
    local t=$SECONDS
    echo "===== $(date +%T) $name =====" >> "$dlog"
    "$@" >> "$dlog" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[$(date +%T)]   [FAIL] $name (exit $rc) after $((SECONDS - t))s — see $(basename "$dlog")" | tee -a "$plog"
        return $rc
    fi
    echo "[$(date +%T)]   [done] $name in $((SECONDS - t))s" | tee -a "$plog"
    STEP_SECONDS+="\"$name\": $((SECONDS - t)), "
}

stage2_body() {  # $1 = episode index
    local N="$1" EP W B PLOG DLOG OFF FIRST LAST
    EP="$(printf 'episode_%06d' "$N")"; W="$OUT/$EP"; B="$W/body"
    PLOG="$SDIR/$EP.log"; DLOG="$SDIR/$EP.debug.log"
    [[ -f "$W/intrinsics_stable.json" ]] || { echo "[$(date +%T)] $EP: stage 1 output missing" | tee -a "$PLOG"; return 1; }
    OFF=$(python3 -c "import json;print(json.load(open('$W/track1_meta.json'))['frame_offset'])")
    FIRST=$(python3 -c "import json;print(json.load(open('$W/track1_meta.json'))['scored_first'])")
    LAST=$(python3 -c "import json;print(json.load(open('$W/track1_meta.json'))['scored_last'])")
    echo "[$(date +%T)] $EP: frames $(ls "$W/frames" | wc -l), offset $OFF, scored $FIRST..$LAST" | tee -a "$PLOG"
    mkdir -p "$B"; STEP_SECONDS=""
    step "$PLOG" "$DLOG" "person track (detectron2)" "[[ -f '$B/bbox_track.pt' ]]" -- \
        "$PY_PIPE" -m v2d.detectron2.docker.run_track_bboxes --rgb_path "$W/frames" \
        --weights_dir "$WEIGHTS/detectron2" --output_path "$B/bbox_track.pt" --batch_size 8 || return 1
    step "$PLOG" "$DLOG" "SAM3D-Body (real intrinsics)" "[[ -f '$B/mhr_mesh.pt' ]]" -- \
        "$PY_PIPE" -m v2d.sam3d_body.docker.run_estimate_mhr_params --rgb_path "$W/frames" \
        --cam_intrinsics_path "$W/intrinsics_stable.json" --weights_dir "$WEIGHTS/sam3d_body" \
        --bbox_path "$B/bbox_track.pt" --output_params_path "$B/mhr_params.pt" \
        --output_mesh_path "$B/mhr_mesh.pt" --batch_size 8 || return 1
    step "$PLOG" "$DLOG" "MHR fit (one identity, sigma $SMOOTH_SIGMA)" "[[ -f '$B/mhr_fit.npz' ]]" -- \
        "$PY_REVIEW" "$T1/fit_mhr.py" --mesh "$B/mhr_mesh.pt" --model "$MHR_MODEL" \
        --first "$FIRST" --last "$LAST" --frame_offset "$OFF" --smooth_sigma "$SMOOTH_SIGMA" \
        --out "$B/mhr_fit.npz" || return 1
    python3 - "$W/track1_meta.json" "{${STEP_SECONDS%, }}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); new = json.loads(sys.argv[2])
m["stage2_seconds"] = {**m.get("stage2_seconds", {}), **new}; json.dump(m, open(sys.argv[1], "w"), indent=1)
PY
    echo "[$(date +%T)] $EP: stage 2 OK — $(python3 -c "import json,numpy" 2>/dev/null; "$PY_REVIEW" -c "import numpy as n,json;r=json.loads(str(n.load('$B/mhr_fit.npz')['report']));print('fit error mean %.2f mm' % r['vertex_error_mm']['mean'])" 2>/dev/null)" | tee -a "$PLOG"
}

case "$STAGE" in
    1) NAME=01_prep ;;
    2) NAME=02_body ;;
    3) NAME=03_select ;;
    4) NAME=04_mask ;;
    5) NAME=05_mesh ;;
    6) NAME=06_track ;;
    7) NAME=07_smooth ;;
    8) NAME=08_final ;;
    *) echo "unknown stage $STAGE"; exit 2 ;;
esac
SDIR="$BASEPATH/logs/stages/$NAME"
mkdir -p "$SDIR/review" "$OUT"
exec > >(tee -a "$SDIR/batch.log") 2>&1

expand() { local t; for t in $1; do if [[ "$t" == *-* ]]; then seq "${t%-*}" "${t#*-}"; else echo "$t"; fi; done; }
EP_LIST="$(expand "$EPISODES" | tr '\n' ' ')"

build_review() {  # $1 = episodes to include; rebuilds review/ sheets and index.html
    # Include every episode that already has checks in this stage, so parallel runs (e.g. one per GPU)
    # do not overwrite each other's index.html; one rebuild at a time (flock).
    local L e
    L="$(for e in $1 $(ls "$SDIR"/episode_*.checks.json 2>/dev/null | sed -E 's/.*episode_0*([0-9]+)\.checks\.json/\1/'); do echo $e; done | sort -n | uniq | tr '\n' ' ')"
    [[ -n "${L// }" ]] || return 0
    exec 9> "$SDIR/.review.lock"
    flock 9
    case "$STAGE" in
        1) "$PY_REVIEW" "$T1/review_stage1.py" --episodes $L --out_root "$OUT" --stage_dir "$SDIR" ;;
        2) "$PY_REVIEW" "$T1/review_stage2.py" --episodes $L --out_root "$OUT" --stage_dir "$SDIR" \
               --model "$MHR_MODEL" ;;
        3) "$PY_REVIEW" "$T1/review_stage3.py" --episodes $L --out_root "$OUT" --stage_dir "$SDIR" ;;
        4) "$PY_REVIEW" "$T1/review_stage4.py" --episodes $L --out_root "$OUT" --stage_dir "$SDIR" ;;
        5) "$PY_REVIEW" "$T1/review_stage5.py" --episodes $L --out_root "$OUT" --stage_dir "$SDIR" ;;
        6) "$PY_REVIEW" "$T1/review_stage6.py" --episodes $L --out_root "$OUT" --stage_dir "$SDIR" ;;
        7) "$PY_REVIEW" "$T1/review_stage7.py" --episodes $L --out_root "$OUT" --stage_dir "$SDIR" ;;
        8) "$PY_REVIEW" "$T1/review_stage8.py" --episodes $L --out_root "$OUT" \
               --sub_dir "$BASEPATH/submission" --stage_dir "$SDIR" ;;
    esac 2>&1 | grep -v pynvml
    flock -u 9
}

echo "===== stage $NAME  $(date -Is)  episodes: $EP_LIST ====="
[[ -f "$SPANS" ]] || "$PY_SUB" "$T1/scored_spans.py" \
    --sample "$BASEPATH/v2d_submission_kit/data/track_1_sample_submission.parquet" --out "$SPANS" || exit 1

N_FAIL=0
DONE_LIST=""
T0=$SECONDS
for N in $EP_LIST; do
    EP="$(printf 'episode_%06d' "$N")"
    t=$SECONDS
    # Stages 6-8: skip when the output is newer than everything it depends on.
    W_="$OUT/$EP"
    newer() { [[ -e "$1" ]] && for x in "${@:2}"; do [[ ! -e "$x" || "$1" -nt "$x" ]] || return 1; done; }
    if { [[ "$STAGE" == 6 ]] && newer "$W_/track/states.json" "$W_/mesh_scaled.obj" "$W_/masks_seed.json"; } \
       || { [[ "$STAGE" == 7 ]] && newer "$W_/track/smooth.json" "$W_/track/states.json"; } \
       || { [[ "$STAGE" == 8 ]] && newer "$BASEPATH/submission/reports/$EP.json" "$W_/track/smooth.json" \
                                         "$W_/body/mhr_fit.npz" "$W_/mesh_scaled.obj"; }; then
        echo "[$(date +%T)] $EP: [skip] stage $STAGE output is up to date" | tee -a "$SDIR/$EP.log"
        DONE_LIST+="$N "
        continue
    fi
    case "$STAGE" in
        1) "$PY_PIPE" "$T1/stage1_prep.py" --episode "$N" --out_root "$OUT" --weights "$WEIGHTS" \
               --spans "$SPANS" --debug_log "$SDIR/$EP.debug.log" 2>&1 | tee -a "$SDIR/$EP.log"
           rc=${PIPESTATUS[0]} ;;
        2) stage2_body "$N"; rc=$? ;;
        3) "$PY_REVIEW" "$T1/stage3_select.py" --episode "$N" --out_root "$OUT" --weights "$WEIGHTS" \
               2>> "$SDIR/$EP.debug.log" | tee -a "$SDIR/$EP.log"
           rc=${PIPESTATUS[0]} ;;
        4) stage4_mask "$N"; rc=$? ;;
        5) { "$PY_REVIEW" "$T1/stage5_pick.py" --episode "$N" --out_root "$OUT" 2>> "$SDIR/$EP.debug.log" \
             && "$PY_PIPE" "$T1/stage5_mesh.py" --episode "$N" --out_root "$OUT" --weights "$WEIGHTS" \
                    --debug_log "$SDIR/$EP.debug.log" \
             && "$PY_FP" "$T1/stage5_check.py" --episode "$N" --out_root "$OUT" --weights "$WEIGHTS" \
                    --debug_log "$SDIR/$EP.debug.log" \
             && "$PY_FP" "$T1/stage5_refine.py" --episode "$N" --out_root "$OUT" --weights "$WEIGHTS" \
                    --debug_log "$SDIR/$EP.debug.log" --sheet "$SDIR/review/${EP}_refine.png"; } 2>&1 | tee -a "$SDIR/$EP.debug.log" \
             | grep --line-buffered -E '^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\]' | tee -a "$SDIR/$EP.log"
           rc=${PIPESTATUS[0]} ;;
        6) "$PY_FP" "$T1/stage6_track.py" --episode "$N" --out_root "$OUT" --weights "$WEIGHTS" \
               --debug_log "$SDIR/$EP.debug.log" 2>> "$SDIR/$EP.debug.log" | tee -a "$SDIR/$EP.log"
           rc=${PIPESTATUS[0]} ;;
        7) "$PY_REVIEW" "$T1/stage7_smooth.py" --episode "$N" --out_root "$OUT" \
               2>> "$SDIR/$EP.debug.log" | tee -a "$SDIR/$EP.log"
           rc=${PIPESTATUS[0]} ;;
        8) stage8_final "$N"; rc=$? ;;
    esac
    if [[ $rc -ne 0 ]]; then
        N_FAIL=$((N_FAIL + 1))
        echo "[$(date +%T)] $EP FAILED (exit $rc) after $((SECONDS - t))s — see $SDIR/$EP.log and $EP.debug.log"
    else
        echo "[$(date +%T)] $EP finished in $((SECONDS - t))s"
    fi
    DONE_LIST+="$N "
    build_review "$DONE_LIST" > /dev/null 2>&1   # index.html is refreshed after every episode
done

echo "[$(date +%T)] building review"
build_review "$EP_LIST"
if [[ "$STAGE" == "8" ]]; then
    echo "[$(date +%T)] packing a validation parquet for these episodes (placeholder commit; not uploadable)"
    "$PY_SUB" "$T1/pack_submission.py" --partial 2>&1 | tail -4
fi
echo "===== stage $NAME done in $(( (SECONDS - T0) / 60 )) min, $N_FAIL failure(s). Review: $SDIR/index.html ====="
[[ "$N_FAIL" -eq 0 ]]
