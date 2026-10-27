#!/usr/bin/env bash
# =============================================================================
# run_beta.sh — the pipeline with new models switched in, next to production (out/ is never written).
#
#   ./run_beta.sh                 seed + run every stage from the first one a switch changes, then eval
#   ./run_beta.sh status          stamp table of the beta results
#   ./run_beta.sh seed            only copy the production results the beta keeps
#   ./run_beta.sh 6               one stage (any of 1 2 3 4 5 5c 5e 6 7 8)
#   ./run_beta.sh eval            pack the beta submission and compare its proxies with production
#
# Switches (production value in brackets = switch off):
#   BETA_DEPTH  vda | vda-small    [moge]     stage 1: Video Depth Anything metric depth (temporally stable)
#   BETA_MESH   trellis2 | hunyuan [sam3d]    stage 5: candidate meshes from TRELLIS.2-4B / Hunyuan3D-2
#   BETA_TRACK  cotracker          [off]      stage 7: CoTracker3 point tracks refine the smoothed poses
# Defaults: BETA_DEPTH=vda BETA_MESH=trellis2 BETA_TRACK=cotracker. Example — only the new tracking:
#   BETA_DEPTH=moge BETA_MESH=sam3d BETA_TRACK=cotracker EPISODES="6 8 17" ./run_beta.sh
#
# Where things go (tag = <depth>-<mesh>-<track>, or BETA_TAG):
#   out_beta/<tag>/episode_XXXXXX/   seeded from out/: unchanged stages are copied (stamps included) and
#                                    skipped; the first changed stage and everything after reruns
#   submission_beta/<tag>/           episodes/, reports/, track1_partial.parquet
#   logs/beta/<tag>/<NN_stage>/      the usual per-stage logs and review pages
#   logs/eval/beta/<tag>/            proxies; compared with current production
#                                    logs/eval/beta/prodnow (override: BETA_BASELINE)
# EPISODES (default the debug set 6 8 17 26 28) is widened to every video of the same objects, since
# stage 5e builds one mesh per object over all its videos. With BETA_DEPTH on, the body depth trend
# (stage2_depth.py, cheap) is redone on the new depth; the rest of stage 2 is kept.
#
# GPU=0 (default) runs each stage on one GPU; WORKERS="0 0 2 2" uses run_stage_pool.sh instead
# (5e always runs on GPU). Never GPU 1. New venvs: ./setup_beta_envs.sh (once).
# =============================================================================
set -uo pipefail
CODE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
BASEPATH="${V2D_DATA:-${BASEPATH:-/local-data/ivan/v2d}}"
export V2D_DATA="$BASEPATH"
T1="$CODE/track1"
CMD="${1:-run}"

export BETA_DEPTH="${BETA_DEPTH:-vda}" BETA_MESH="${BETA_MESH:-trellis2}" BETA_TRACK="${BETA_TRACK:-cotracker}"
case "$BETA_DEPTH" in vda|vda-small|moge) ;; *) echo "BETA_DEPTH must be vda, vda-small or moge"; exit 2 ;; esac
case "$BETA_MESH" in trellis2|hunyuan|sam3d) ;; *) echo "BETA_MESH must be trellis2, hunyuan or sam3d"; exit 2 ;; esac
case "$BETA_TRACK" in cotracker|off) ;; *) echo "BETA_TRACK must be cotracker or off"; exit 2 ;; esac
TAG="${BETA_TAG:-$BETA_DEPTH-$BETA_MESH-$BETA_TRACK}"
FIRST=""
[[ "$BETA_TRACK" != off ]] && FIRST=7
[[ "$BETA_MESH" != sam3d ]] && FIRST=5
[[ "$BETA_DEPTH" != moge ]] && FIRST=1
[[ -n "$FIRST" || "$CMD" == status ]] || { echo "all switches are at their production value: nothing to test"; exit 2; }

export V2D_OUT="$BASEPATH/out_beta/$TAG" V2D_SUB="$BASEPATH/submission_beta/$TAG" V2D_LOGS="beta/$TAG"
VENVS="${V2D_VENV_ROOT:-/opt/venvs}"
PY_REVIEW="$VENVS/sam3d_body/bin/python"
PY_SUB="$BASEPATH/.venv-submit/bin/python"
MHR_MODEL="$BASEPATH/weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt"
mkdir -p "$V2D_OUT" "$V2D_SUB" "$BASEPATH/logs/beta/$TAG"
LOG="$BASEPATH/logs/beta/$TAG/run_beta.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

expand() { local t; for t in $1; do if [[ "$t" == *-* ]]; then seq "${t%-*}" "${t#*-}"; else echo "$t"; fi; done; }
EPS="$(python3 - $(expand "${EPISODES:-6 8 17 26 28}") <<PY
import json, sys
meta = [json.loads(l) for l in open("$BASEPATH/challenge/track_1/meta/episodes_metadata.jsonl")]
want = {int(x) for x in sys.argv[1:]}
objs = {m["object"] for m in meta if m["episode_index"] in want}
print(" ".join(str(e) for e in sorted(m["episode_index"] for m in meta if m["object"] in objs)))
PY
)"
[[ -n "$EPS" ]] || { echo "no episodes"; exit 2; }
export EPISODES="$EPS"

for v in vda:$BETA_DEPTH:moge "$BETA_MESH":$BETA_MESH:sam3d "$BETA_TRACK":$BETA_TRACK:off; do
    IFS=: read -r venv val prod <<< "$v"
    [[ "$val" == "$prod" || "$CMD" == status || "$CMD" == eval || -x "$VENVS/$venv/bin/python" ]] \
        || { echo "missing venv $VENVS/$venv — run ./setup_beta_envs.sh first"; exit 2; }
done

seed() {
    python3 "$T1/beta/seed.py" --src "$BASEPATH/out" --dst "$V2D_OUT" --first "$FIRST" --episodes $EPS | tee -a "$LOG"
    return "${PIPESTATUS[0]}"
}

stage() {  # $1 = stage
    local S="$1"
    say "stage $S ($TAG) on episodes $EPS"
    if [[ -n "${WORKERS:-}" && "$S" != 5e ]]; then "$CODE/run_stage_pool.sh" "$S"
    else "$CODE/run_stage.sh" "$S"; fi
}

redo_body_depth() {  # the body depth trend follows the scene depth: redo it on the new depth
    local N W rc=0
    for N in $EPS; do
        W="$V2D_OUT/$(printf 'episode_%06d' "$N")"
        [[ -f "$W/depth_scale.json" && -f "$W/body/mhr_fit_raw.npz" ]] || continue
        [[ -f "$W/body/depth_trend.json" && "$W/body/depth_trend.json" -nt "$W/depth_scale.json" ]] && continue
        say "episode $N: body depth trend on the $BETA_DEPTH depth"
        CUDA_VISIBLE_DEVICES="${GPU:-0}" "$PY_REVIEW" "$T1/stage2_depth.py" --episode "$N" --out_root "$V2D_OUT" \
            --model "$MHR_MODEL" >> "$LOG" 2>&1 || { say "episode $N: stage2_depth FAILED"; rc=1; }
    done
    return $rc
}

evaluate() {
    local EV="$BASEPATH/logs/eval/beta/$TAG" BASE="${BETA_BASELINE:-$BASEPATH/logs/eval/beta/prodnow}"
    [[ -f "$BASE/proxies.json" ]] || { say "current-production control missing in $BASE: create it first or set BETA_BASELINE"; return 1; }
    mkdir -p "$EV"
    say "eval: pack + proxies -> $EV"
    "$PY_SUB" "$T1/pack_submission.py" --partial --episodes "$V2D_SUB/episodes" --out "$V2D_SUB/track1_partial.parquet" \
        >> "$LOG" 2>&1 || { say "pack FAILED"; return 1; }
    rm -f "$EV"/episode_*.npz
    "$PY_SUB" "$T1/eval_local.py" extract --parquet "$V2D_SUB/track1_partial.parquet" --out "$EV" >> "$LOG" 2>&1 \
        && "$PY_REVIEW" "$T1/eval_local.py" proxies --out "$EV" 2>&1 | tee "$EV/proxies.log" | tail -3 \
        || { say "eval FAILED (see $LOG)"; return 1; }
    if [[ -f "$BASE/proxies.json" ]]; then
        python3 "$T1/eval_local.py" compare "$BASE" "$EV" 2>/dev/null \
            || "$PY_REVIEW" "$T1/eval_local.py" compare "$BASE" "$EV"
    else say "no baseline proxies in $BASE: not compared"; fi | tee "$EV/compare.log"
}

case "$CMD" in
    status) python3 "$T1/stamps.py" status --episodes $EPS ;;
    seed) seed ;;
    eval) evaluate ;;
    1|2|3|4|5|5c|5e|6|7|8)
        seed || exit 1
        stage "$CMD"; rc=$?
        [[ "$CMD" != 1 || "$BETA_DEPTH" == moge ]] || redo_body_depth || rc=1
        exit $rc ;;
    run)
        say "beta $TAG: episodes $EPS, rerun from stage $FIRST (out: $V2D_OUT)"
        seed || exit 1
        ORDER=(1 2 3 4 5 5c 5e 6 7 8)
        on=0
        for S in "${ORDER[@]}"; do
            [[ "$S" == "$FIRST" ]] && on=1
            [[ $on -eq 1 ]] || continue
            stage "$S" || say "stage $S had failures (continuing with the episodes that passed)"
            [[ "$S" != 1 || "$BETA_DEPTH" == moge ]] || redo_body_depth
        done
        python3 "$T1/stamps.py" status --episodes $EPS | tee -a "$LOG"
        evaluate ;;
    *) echo "usage: run_beta.sh [run|status|seed|eval|<stage>]"; exit 2 ;;
esac
