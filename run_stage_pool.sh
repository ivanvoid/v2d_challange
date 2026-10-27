#!/usr/bin/env bash
# =============================================================================
# run_stage_pool.sh — run ONE stage on many episodes with several workers per GPU.
#
#   WORKERS="0 0 0 2 2 2" EPISODES="0-29" ./run_stage_pool.sh 1      (run inside tmux)
#
# WORKERS lists one GPU id per worker (here 3 workers on GPU 0 and 3 on GPU 2). Episodes go into one
# shared queue, longest first; each worker takes the next episode under a lock and runs
# `run_stage.sh <stage>` on it (NO_REVIEW=1), then builds that episode's review sheet (REVIEW_ONLY=1,
# no lock, so reviews run in parallel instead of queueing); one full review page at the end. No
# episode is ever processed twice at once; episodes whose stamp is up to date are skipped.
#
# How many workers per GPU (A40, 46 GB): stage 1 (MoGe ~12 GB peak, mostly CPU work) 3; stage 2
# (detector batches 4/2, SAM3D-Body; peak ~12 GB) 3; stages 3-4 2; stage 5 (SAM3D object generation) 1 — two ran out of memory
# before; stage 6 (FoundationPose) 2; stages 7-8 2. Never use GPU 1.
# Not for stage 5e (one job per object): use run_stage.sh 5e or run_rest.sh.
# Log: logs/stages/run_pool_stage<N>.log (one line per episode).
# =============================================================================
set -uo pipefail
STAGE="${1:?usage: run_stage_pool.sh <stage>}"
[[ "$STAGE" != "5e" ]] || { echo "stage 5e runs per object: use run_stage.sh 5e"; exit 2; }
CODE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
export V2D_DATA="${V2D_DATA:-/local-data/ivan/v2d}"
cd "$V2D_DATA"
WORKERS=(${WORKERS:?set WORKERS, e.g. WORKERS=\"0 0 2 2\"})
for g in "${WORKERS[@]}"; do [[ "$g" != 1 ]] || { echo "GPU 1 is not ours"; exit 2; }; done
EPS="${EPISODES:-0-29}"
LOGD="logs/${V2D_LOGS:-stages}"                        # run_beta.sh: logs/beta/<tag>
LOG="$LOGD/run_pool_stage${STAGE}${POOL_TAG:-}.log"
QUEUE="$LOGD/.queue_stage${STAGE}${POOL_TAG:-}"   # POOL_TAG=_b: a second pool with its own queue
mkdir -p "$LOGD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

expand() { local t; for t in $1; do if [[ "$t" == *-* ]]; then seq "${t%-*}" "${t#*-}"; else echo "$t"; fi; done; }
python3 - $(expand "$EPS") > "$QUEUE" <<'PY'
import json, sys, os
eps = [int(x) for x in sys.argv[1:]]
def frames(e):
    try:
        return json.load(open(f"out/episode_{e:06d}/track1_meta.json"))["n_frames"]
    except Exception:
        return json.load(open("submission/scored_spans.json"))[str(e)]["n_scored"] + 20
for e in sorted(eps, key=frames, reverse=True):
    print(e)
PY
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "stage $STAGE: $(wc -l < "$QUEUE") episodes, workers on GPUs ${WORKERS[*]}"

pop() {  # next episode from the queue, under a lock
    ( flock 8; e="$(head -n 1 "$QUEUE")"; [[ -n "$e" ]] && sed -i '1d' "$QUEUE"; echo "$e" ) 8> "$QUEUE.lock"
}
worker() {  # $1 = worker id, $2 = GPU
    local id="$1" g="$2" e t rc
    while :; do
        e="$(pop)"; [[ -n "$e" ]] || break
        t=$SECONDS
        NO_REVIEW=1 GPU="$g" EPISODES="$e" "$CODE/run_stage.sh" "$STAGE" > /dev/null 2>&1; rc=$?
        say "worker $id (GPU $g): episode $e $([[ $rc -eq 0 ]] && echo OK || echo "FAILED (exit $rc)") in $((SECONDS - t))s"
        t=$SECONDS                           # this episode's review sheet + checks, in parallel with the others
        REVIEW_ONLY=1 GPU="$g" EPISODES="$e" "$CODE/run_stage.sh" "$STAGE" > /dev/null 2>&1
        say "worker $id (GPU $g): episode $e review built in $((SECONDS - t))s"
    done
}
i=0
for g in "${WORKERS[@]}"; do
    worker "$i" "$g" &
    i=$((i + 1))
    sleep 5                                  # stagger model loading
done
wait
rm -f "$QUEUE" "$QUEUE.lock"
say "building the full review page (reuses the per-episode results)"
ALL_REVIEWED="$( { expand "$EPS"; ls "$LOGD"/0${STAGE}_*/episode_*.checks.json 2>/dev/null \
    | sed -E 's/.*episode_0*([0-9]+)\.checks\.json/\1/'; } | sort -n | uniq | tr '\n' ' ')"
REVIEW_ONLY=1 GPU="${WORKERS[0]}" EPISODES="$ALL_REVIEWED" "$CODE/run_stage.sh" "$STAGE" > /dev/null 2>&1
say "stage $STAGE done: $(grep -c "FAILED" "$LOG") FAILED lines in $LOG so far"
python3 "$CODE/track1/stamps.py" status --episodes $(expand "$EPS" | tr '\n' ' ') | tee -a "$LOG"
