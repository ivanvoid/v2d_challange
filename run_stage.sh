#!/usr/bin/env bash
# =============================================================================
# run_stage.sh — run ONE pipeline stage on a set of episodes, then build its review.
#
#   ./run_stage.sh 1                      # stage 1 on the debug set (6 8 17 26 28)
#   EPISODES="0-29" ./run_stage.sh 1      # stage 1 on everything
#   EPISODES="0-29" ./run_stage.sh 5      # stage 5 = SAM3D generation (pick + mesh + scale)
#   EPISODES="0-29" ./run_stage.sh 5c     # per-video selection (cheap; re-run after a scoring change)
#   EPISODES="0-29" ./run_stage.sh 5e     # one mesh per object, for the objects of these episodes, + validation
#   FORCE=1 ...                           # run even if the stage is up to date
#   NO_REVIEW=1 ...                       # skip building the review (run_stage_pool.sh does it per episode)
#   REVIEW_ONLY=1 ...                     # only build the review of these episodes (no lock, no index merge)
#
# Output: logs/stages/<NN_name>/
#   batch.log                       everything this script printed
#   episode_XXXXXX.log              short progress log per episode
#   episode_XXXXXX.debug.log        full model output per episode
#   episode_XXXXXX.checks.json      automatic checks
#   review/                         review images
#   index.html                      all episodes side by side — open this
# Results go to out/episode_XXXXXX/. A stage is skipped only when its stamp (out/<ep>/stamps/<stage>.json,
# track1/stamps.py) matches its current code, settings, inputs and upstream stamps; otherwise its old
# outputs are moved to out/<ep>/_archive/<stage>/ and it runs from scratch. Upstream stages must be
# stamped first (bottom-up). `python3 track1/stamps.py status` shows what is up to date.
#
# Beta (run_beta.sh sets these; unset = production): V2D_OUT / V2D_SUB / V2D_LOGS move out/, submission/
# and logs/stages/; BETA_DEPTH=vda (stage 1), BETA_MESH=trellis2|hunyuan (stage 5), BETA_TRACK=cotracker
# (stage 7) switch in the new models (track1/beta/, venvs from setup_beta_envs.sh).
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
OUT="${V2D_OUT:-$BASEPATH/out}"
WEIGHTS="$BASEPATH/weights"
SPANS="$BASEPATH/submission/scored_spans.json"
PY_PIPE=python
PY_REVIEW=/opt/venvs/sam3d_body/bin/python
PY_SUB="$BASEPATH/.venv-submit/bin/python"
PY_FP=/opt/venvs/foundation_pose/bin/python
PY_RENDER=/opt/venvs/wilor/bin/python
SUB="${V2D_SUB:-$BASEPATH/submission}"
VENVS="${V2D_VENV_ROOT:-/opt/venvs}"
beta_on() { local v="${!1:-}"; [[ -n "$v" && "$v" != "$2" ]]; }   # beta_on BETA_MESH sam3d

stage4_mask() {  # $1 = episode: prompts -> SAM2 -> (re-detect -> SAM2) x <= 2 -> leak repair -> assess
    local N="$1" EP PLOG DLOG rc
    EP="$(printf 'episode_%06d' "$N")"; PLOG="$SDIR/$EP.log"; DLOG="$SDIR/$EP.debug.log"
    p4() { "$PY_REVIEW" "$T1/stage4_prompts.py" --episode "$N" --out_root "$OUT" --weights "$WEIGHTS" --mode "$1" \
               2>> "$DLOG" | tee -a "$PLOG"; return ${PIPESTATUS[0]}; }
    sam() { "$PY_PIPE" "$T1/stage4_mask.py" --episode "$N" --out_root "$OUT" --weights "$WEIGHTS" \
               --debug_log "$DLOG" 2>&1 | tee -a "$PLOG"; return ${PIPESTATUS[0]}; }
    p4 init || return 1
    sam || return 1
    local round
    for round in 1 2; do                     # recovery: re-detect on suspicious / lost stretches, at most twice
        p4 fill; rc=$?
        if [[ $rc -eq 10 ]]; then sam || return 1; elif [[ $rc -ne 0 ]]; then return 1; else break; fi
    done
    p4 repair || return 1
    p4 assess || return 1                    # final per-frame status; exit 3 = too many scored frames lost
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
    step "$PLOG" "$DLOG" "actor track + person masks (one detector pass)" "[[ -f '$B/person_track.json' && -f '$B/bbox_track.pt' ]]" -- \
        /opt/venvs/detectron2/bin/python "$T1/stage2_track.py" --episode "$N" --out_root "$OUT" \
        --weights "$WEIGHTS" --batch 2 || { grep "ACCEPTANCE\|actor track" "$DLOG" | tail -2 | tee -a "$PLOG"; return 1; }
    grep "actor track —" "$DLOG" | tail -1 | sed 's/^/    /' | tee -a "$PLOG"
    step "$PLOG" "$DLOG" "SAM3D-Body (box + person mask)" "[[ -f '$B/mhr_mesh.pt' ]]" -- \
        "$PY_REVIEW" "$T1/stage2_body.py" --episode "$N" --out_root "$OUT" --weights "$WEIGHTS" \
        --mode "${BODY_MODE:-body}" ${BODY_NO_MASK:+--no_mask} || return 1
    grep "SAM3D-Body (" "$DLOG" | tail -1 | sed 's/^.*\] /    /' | tee -a "$PLOG"
    step "$PLOG" "$DLOG" "MHR fit (one identity, sigma $SMOOTH_SIGMA)" "[[ -f '$B/mhr_fit_raw.npz' || -f '$B/mhr_fit.npz' ]]" -- \
        "$PY_REVIEW" "$T1/fit_mhr.py" --mesh "$B/mhr_mesh.pt" --model "$MHR_MODEL" \
        --first "$FIRST" --last "$LAST" --frame_offset "$OFF" --smooth_sigma "$SMOOTH_SIGMA" \
        --out "$B/mhr_fit.npz" || return 1
    step "$PLOG" "$DLOG" "body depth follows the MoGe trend" "[[ -f '$B/depth_trend.json' ]]" -- \
        "$PY_REVIEW" "$T1/stage2_depth.py" --episode "$N" --out_root "$OUT" --model "$MHR_MODEL" || return 1
    grep "body depth follows" "$DLOG" | tail -1 | sed 's/^.*\] /    /' | tee -a "$PLOG"
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
    5c) NAME=05c_check ;;
    5e) NAME=05e_object ;;
    6) NAME=06_track ;;
    7) NAME=07_smooth ;;
    8) NAME=08_final ;;
    *) echo "unknown stage $STAGE"; exit 2 ;;
esac
SDIR="$BASEPATH/logs/${V2D_LOGS:-stages}/$NAME"
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
        5c) "$PY_REVIEW" "$T1/review_stage5.py" --episodes $L --out_root "$OUT" --stage_dir "$SDIR" ;;
        6) "$PY_REVIEW" "$T1/review_stage6.py" --episodes $L --out_root "$OUT" --stage_dir "$SDIR" ;;
        7) "$PY_REVIEW" "$T1/review_stage7.py" --episodes $L --out_root "$OUT" --stage_dir "$SDIR" ;;
        8) "$PY_REVIEW" "$T1/review_stage8.py" --episodes $L --out_root "$OUT" \
               --sub_dir "$SUB" --stage_dir "$SDIR" ;;
    esac 2>&1 | grep -v pynvml
    flock -u 9
}

if [[ -n "${REVIEW_ONLY:-}" ]]; then         # this episode's review only; the full index is rebuilt later
    case "$STAGE" in
        1) "$PY_REVIEW" "$T1/review_stage1.py" --episodes $EP_LIST --out_root "$OUT" --stage_dir "$SDIR" ;;
        2) "$PY_REVIEW" "$T1/review_stage2.py" --episodes $EP_LIST --out_root "$OUT" --stage_dir "$SDIR" --model "$MHR_MODEL" ;;
        3) "$PY_REVIEW" "$T1/review_stage3.py" --episodes $EP_LIST --out_root "$OUT" --stage_dir "$SDIR" ;;
        4) "$PY_REVIEW" "$T1/review_stage4.py" --episodes $EP_LIST --out_root "$OUT" --stage_dir "$SDIR" ;;
        5c) "$PY_REVIEW" "$T1/review_stage5.py" --episodes $EP_LIST --out_root "$OUT" --stage_dir "$SDIR" ;;
        6) "$PY_REVIEW" "$T1/review_stage6.py" --episodes $EP_LIST --out_root "$OUT" --stage_dir "$SDIR" ;;
        7) "$PY_REVIEW" "$T1/review_stage7.py" --episodes $EP_LIST --out_root "$OUT" --stage_dir "$SDIR" ;;
        8) "$PY_REVIEW" "$T1/review_stage8.py" --episodes $EP_LIST --out_root "$OUT" --sub_dir "$SUB" --stage_dir "$SDIR" ;;
    esac 2>&1 | grep -v pynvml
    exit 0
fi
echo "===== stage $NAME  $(date -Is)  episodes: $EP_LIST ====="
[[ -f "$SPANS" ]] || "$PY_SUB" "$T1/scored_spans.py" \
    --sample "$BASEPATH/v2d_submission_kit/data/track_1_sample_submission.parquet" --out "$SPANS" || exit 1

N_FAIL=0
DONE_LIST=""
T0=$SECONDS
STAMPS="python3 $T1/stamps.py"

if [[ "$STAGE" == "5e" ]]; then
    # one mesh per object: the objects of the given episodes, each over all of its videos
    OBJ_LIST="$(python3 -c "
import json, sys
meta = [json.loads(l) for l in open('$BASEPATH/challenge/track_1/meta/episodes_metadata.jsonl')]
want = set(int(x) for x in '$EP_LIST'.split())
print(' '.join(sorted({m['object'] for m in meta if m['episode_index'] in want})))")"
    for O in $OBJ_LIST; do
        VIDS="$(python3 -c "
import json
meta = [json.loads(l) for l in open('$BASEPATH/challenge/track_1/meta/episodes_metadata.jsonl')]
print(' '.join(str(m['episode_index']) for m in meta if m['object'] == '$O'))")"
        t=$SECONDS; ok=1; blocked=0
        for V in $VIDS; do
            $STAMPS check --episode "$V" --stage 5e > /dev/null; rc=$?
            [[ $rc -eq 0 ]] || ok=0
            [[ $rc -eq 2 ]] && blocked=1
        done
        if [[ $ok -eq 1 && -z "${FORCE:-}" ]]; then
            echo "[$(date +%T)] $O: [skip] stage 5e is up to date (videos $VIDS)" | tee -a "$SDIR/$O.log"; continue
        fi
        if [[ $blocked -eq 1 ]]; then
            for V in $VIDS; do $STAMPS check --episode "$V" --stage 5e; done | tee -a "$SDIR/$O.log"
            N_FAIL=$((N_FAIL + 1)); echo "[$(date +%T)] $O FAILED: stage 5 not done for all of its videos"; continue
        fi
        echo "[$(date +%T)] $O: stage 5e on videos $VIDS" | tee -a "$SDIR/$O.log"
        for V in $VIDS; do $STAMPS archive --episode "$V" --stage 5e | tee -a "$SDIR/$O.log"; done
        /opt/venvs/foundation_pose/bin/python "$T1/stage5_object.py" --object "$O" --out_root "$OUT" \
            --weights "$WEIGHTS" --stage_dir "$SDIR" 2>> "$SDIR/$O.debug.log" | tee -a "$SDIR/$O.log"
        rc=${PIPESTATUS[0]}
        if [[ $rc -eq 0 ]]; then
            for V in $VIDS; do               # validate the exact final model per video; stamp only accepted ones
                VE="$(printf 'episode_%06d' "$V")"
                CUDA_VISIBLE_DEVICES="$GPU" "$PY_FP" "$T1/stage5_validate.py" --episode "$V" --out_root "$OUT" \
                    --weights "$WEIGHTS" --debug_log "$SDIR/$O.debug.log" --sheet "$SDIR/review/${VE}_validate.png" \
                    2>> "$SDIR/$O.debug.log" | tee -a "$SDIR/$O.log"
                if [[ ${PIPESTATUS[0]} -eq 0 ]]; then $STAMPS write --episode "$V" --stage 5e || rc=1
                else rc=3; echo "[$(date +%T)] $O: episode $V final model REJECTED (see ${VE}_validate.png)" | tee -a "$SDIR/$O.log"; fi
            done
        fi
        if [[ $rc -ne 0 ]]; then N_FAIL=$((N_FAIL + 1)); echo "[$(date +%T)] $O FAILED (exit $rc) after $((SECONDS - t))s"
        else echo "[$(date +%T)] $O finished in $((SECONDS - t))s"; fi
    done
    [[ -n "${NO_REVIEW:-}" ]] || "$PY_REVIEW" "$T1/review_stage5e.py" --out_root "$OUT" --stage_dir "$SDIR" \
        > "$SDIR/index_build.log" 2>&1 || echo "review index failed (see $SDIR/index_build.log)"
    echo "===== stage $NAME done in $(( (SECONDS - T0) / 60 )) min, $N_FAIL failure(s). Review: $SDIR/ ====="
    [[ "$N_FAIL" -eq 0 ]]; exit $?
fi

for N in $EP_LIST; do
    EP="$(printf 'episode_%06d' "$N")"
    t=$SECONDS
    # Skip only when this stage's stamp matches its current code, settings, inputs and upstream stamps
    # (track1/stamps.py). Otherwise its old outputs are archived and it runs from scratch.
    $STAMPS check --episode "$N" --stage "$STAGE" > "$SDIR/.stamp_$N" 2>&1; src=$?
    if [[ $src -eq 0 && -z "${FORCE:-}" ]]; then
        echo "[$(date +%T)] $EP: [skip] stage $STAGE output is up to date" | tee -a "$SDIR/$EP.log"
        DONE_LIST+="$N "
        continue
    fi
    if [[ $src -eq 2 ]]; then
        cat "$SDIR/.stamp_$N" | tee -a "$SDIR/$EP.log"
        N_FAIL=$((N_FAIL + 1)); echo "[$(date +%T)] $EP FAILED: run the upstream stage(s) first"
        continue
    fi
    echo "[$(date +%T)] $EP: $(cat "$SDIR/.stamp_$N")${FORCE:+ (FORCE=1)}" | tee -a "$SDIR/$EP.log"
    if [[ "$STAGE" != 1 ]]; then $STAMPS archive --episode "$N" --stage "$STAGE" | tee -a "$SDIR/$EP.log"; fi
    rm -f "$SDIR/.stamp_$N"
    case "$STAGE" in
        1) if beta_on BETA_DEPTH moge; then       # beta: depth_raw/ from Video Depth Anything instead of MoGe
               "$VENVS/vda/bin/python" "$T1/beta/depth_vda.py" --episode "$N" --out_root "$OUT" --weights "$WEIGHTS" \
                   --model "$BETA_DEPTH" --debug_log "$SDIR/$EP.debug.log" 2>> "$SDIR/$EP.debug.log" | tee -a "$SDIR/$EP.log"
               rc=${PIPESTATUS[0]}
               if [[ $rc -ne 0 ]]; then
                   N_FAIL=$((N_FAIL + 1)); echo "[$(date +%T)] $EP FAILED: beta depth ($BETA_DEPTH), see $EP.debug.log"
                   continue
               fi
           fi
           "$PY_PIPE" "$T1/stage1_prep.py" --episode "$N" --out_root "$OUT" --weights "$WEIGHTS" \
               --spans "$SPANS" --debug_log "$SDIR/$EP.debug.log" 2>&1 | tee -a "$SDIR/$EP.log"
           rc=${PIPESTATUS[0]} ;;
        2) stage2_body "$N"; rc=$? ;;
        3) "$PY_REVIEW" "$T1/stage3_select.py" --episode "$N" --out_root "$OUT" --weights "$WEIGHTS" \
               2>> "$SDIR/$EP.debug.log" | tee -a "$SDIR/$EP.log"
           rc=${PIPESTATUS[0]} ;;
        4) stage4_mask "$N"; rc=$? ;;
        5) if beta_on BETA_MESH sam3d; then       # beta: candidate meshes from TRELLIS.2 / Hunyuan3D-2
               MESH_CMD=("$VENVS/$BETA_MESH/bin/python" "$T1/beta/mesh_gen.py" --model "$BETA_MESH")
           else
               MESH_CMD=("$PY_PIPE" "$T1/stage5_mesh.py")
           fi
           { "$PY_REVIEW" "$T1/stage5_pick.py" --episode "$N" --out_root "$OUT" 2>> "$SDIR/$EP.debug.log" \
             && "${MESH_CMD[@]}" --episode "$N" --out_root "$OUT" --weights "$WEIGHTS" \
                    --debug_log "$SDIR/$EP.debug.log" \
             && "$PY_FP" "$T1/stage5_scale.py" --episode "$N" --out_root "$OUT" --weights "$WEIGHTS" \
                    --debug_log "$SDIR/$EP.debug.log"; } 2>&1 | tee -a "$SDIR/$EP.debug.log" \
             | grep --line-buffered -E '^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\]' | tee -a "$SDIR/$EP.log"
           rc=${PIPESTATUS[0]} ;;
        5c) "$PY_FP" "$T1/stage5_check.py" --episode "$N" --out_root "$OUT" --weights "$WEIGHTS" \
               --debug_log "$SDIR/$EP.debug.log" 2>&1 | tee -a "$SDIR/$EP.debug.log" \
               | grep --line-buffered -E '^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\]' | tee -a "$SDIR/$EP.log"
           rc=${PIPESTATUS[0]} ;;
        6) "$PY_FP" "$T1/stage6_track.py" --episode "$N" --out_root "$OUT" --weights "$WEIGHTS" \
               --debug_log "$SDIR/$EP.debug.log" 2>> "$SDIR/$EP.debug.log" | tee -a "$SDIR/$EP.log"
           rc=${PIPESTATUS[0]} ;;
        7) "$PY_REVIEW" "$T1/stage7_smooth.py" --episode "$N" --out_root "$OUT" \
               2>> "$SDIR/$EP.debug.log" | tee -a "$SDIR/$EP.log"
           rc=${PIPESTATUS[0]}
           if [[ $rc -eq 0 ]] && beta_on BETA_TRACK off; then   # beta: object point tracks refine the smoothed poses
               { "$VENVS/$BETA_TRACK/bin/python" "$T1/beta/point_tracks.py" --episode "$N" --out_root "$OUT" \
                     --weights "$WEIGHTS" --model "$BETA_TRACK" --debug_log "$SDIR/$EP.debug.log" \
                 && "$PY_REVIEW" "$T1/beta/track_refine.py" --episode "$N" --out_root "$OUT"; } \
                 2>> "$SDIR/$EP.debug.log" | tee -a "$SDIR/$EP.log"
               rc=${PIPESTATUS[0]}
           fi ;;
        8) stage8_final "$N"; rc=$? ;;
    esac
    if [[ $rc -ne 0 ]]; then
        N_FAIL=$((N_FAIL + 1))
        echo "[$(date +%T)] $EP FAILED (exit $rc) after $((SECONDS - t))s — see $SDIR/$EP.log and $EP.debug.log"
    elif ! $STAMPS write --episode "$N" --stage "$STAGE" 2>&1 | tee -a "$SDIR/$EP.log"; then
        N_FAIL=$((N_FAIL + 1))
        echo "[$(date +%T)] $EP FAILED: finished but could not be stamped (outputs missing?)"
    else
        echo "[$(date +%T)] $EP finished in $((SECONDS - t))s"
    fi
    DONE_LIST+="$N "
    [[ -n "${NO_REVIEW:-}" ]] || build_review "$DONE_LIST" > /dev/null 2>&1   # index.html refreshed per episode
done

if [[ -z "${NO_REVIEW:-}" ]]; then echo "[$(date +%T)] building review"; build_review "$EP_LIST"; fi
if [[ "$STAGE" == "8" && -z "${NO_REVIEW:-}" ]]; then
    echo "[$(date +%T)] packing a validation parquet for these episodes (placeholder commit; not uploadable)"
    "$PY_SUB" "$T1/pack_submission.py" --partial --episodes "$SUB/episodes" --out "$SUB/track1_partial.parquet" 2>&1 | tail -4
fi
echo "===== stage $NAME done in $(( (SECONDS - T0) / 60 )) min, $N_FAIL failure(s). Review: $SDIR/index.html ====="
[[ "$N_FAIL" -eq 0 ]]
