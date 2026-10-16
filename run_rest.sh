#!/usr/bin/env bash
# =============================================================================
# run_rest.sh — finish the pipeline on all 30 videos, one job per GPU, step by step.
#   1. wait for the running stage-5 sessions (s5gpu0, s5gpu2)
#   2. stage 5 again for any episode without a stage-5 result (e.g. 11, failed on GPU memory)
#   3. stage 5e: one mesh per object (10 objects, split over the two GPUs)
#   4. stage 6 tracking on all 30 (skips episodes whose tracking is newer than mesh + masks)
#   5. stage 7 smoothing, stage 8 scale match + submission files + QA (skips up-to-date ones)
# Log: logs/stages/run_rest.log. Usage: GPUS="0 2" ./run_rest.sh   (run inside tmux)
# =============================================================================
set -uo pipefail
CODE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
export V2D_DATA="${V2D_DATA:-/local-data/ivan/v2d}"
cd "$V2D_DATA"
GPUS=(${GPUS:-0 2})
LOG=logs/stages/run_rest.log
ALL="$(seq -s ' ' 0 29)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True V2D_BACKEND=venv
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

split2() {  # $@ = episodes -> two lines, balanced by frame count
    python3 - "$@" <<'PY'
import json, sys
eps = [int(x) for x in sys.argv[1:]]
fr = lambda e: json.load(open(f"out/episode_{e:06d}/track1_meta.json"))["n_frames"]
a, b, la, lb = [], [], 0, 0
for e in sorted(eps, key=fr, reverse=True):
    if la <= lb: a.append(e); la += fr(e)
    else: b.append(e); lb += fr(e)
print(" ".join(map(str, sorted(a)))); print(" ".join(map(str, sorted(b))))
PY
}

run_on_two() {  # $1 = stage, $2.. = episodes: one run_stage.sh per GPU, wait for both
    local st="$1"; shift
    [[ $# -gt 0 ]] || return 0
    mapfile -t L < <(split2 "$@")
    say "stage $st: GPU ${GPUS[0]} <- ${L[0]:-none} | GPU ${GPUS[1]} <- ${L[1]:-none}"
    [[ -n "${L[0]:-}" ]] && { GPU=${GPUS[0]} EPISODES="${L[0]}" "$CODE/run_stage.sh" "$st" > /dev/null 2>&1 & }
    [[ -n "${L[1]:-}" ]] && { GPU=${GPUS[1]} EPISODES="${L[1]}" "$CODE/run_stage.sh" "$st" > /dev/null 2>&1 & }
    wait
    say "stage $st done — $(grep -c FAILED logs/stages/0${st}_*/batch.log) FAILED lines in total in its batch.log"
}

say "===== start (GPUs ${GPUS[*]}) ====="
# 1. wait for the stage-5 sessions already running
while tmux has-session -t s5gpu0 2>/dev/null || tmux has-session -t s5gpu2 2>/dev/null; do
    if ! pgrep -f "run_stage.sh 5" > /dev/null; then break; fi
    sleep 30
done
say "stage-5 sessions finished"

# 2. stage 5 for episodes still without a result
MISSING="$(for e in $ALL; do [[ -f out/$(printf episode_%06d $e)/mesh/refine.json ]] || echo $e; done | tr '\n' ' ')"
for e in $MISSING; do rm -rf "out/$(printf episode_%06d $e)/mesh"; done
run_on_two 5 $MISSING

# 3. stage 5e: one mesh per object
OBJS=(hula_hoop big_red_bowl iron white_desk black_pan foam_grass_block paint_roller pink_foam_roll short_wood_stool white_laptop_cart)
mkdir -p logs/stages/05e_object
obj_worker() {  # $1 = GPU, rest = objects
    local g="$1"; shift
    for o in "$@"; do
        CUDA_VISIBLE_DEVICES=$g /opt/venvs/foundation_pose/bin/python "$CODE/track1/stage5_object.py" --object "$o" \
            --out_root out --weights weights --stage_dir logs/stages/05e_object 2>> logs/stages/05e_object/$o.debug.log \
            | tee -a logs/stages/05e_object/$o.log | grep -E "stage 5e OK|pass 2" >> "$LOG"
        [[ ${PIPESTATUS[0]} -eq 0 ]] || say "stage 5e FAILED for $o (see logs/stages/05e_object/$o.debug.log)"
    done
}
say "stage 5e: one mesh per object"
obj_worker "${GPUS[0]}" "${OBJS[@]:0:5}" &
obj_worker "${GPUS[1]}" "${OBJS[@]:5:5}" &
wait

# 4-5. tracking, smoothing, final
run_on_two 6 $ALL
EPISODES="$ALL" "$CODE/run_stage.sh" 7 > /dev/null 2>&1; say "stage 7 done"
run_on_two 8 $ALL
say "===== done ====="
