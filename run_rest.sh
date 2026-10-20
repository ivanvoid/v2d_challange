#!/usr/bin/env bash
# =============================================================================
# run_rest.sh — run a range of stages on all 30 videos, one job per GPU, stage by stage.
#
#   FROM=3 TO=8 GPUS="0 2" ./run_rest.sh          (run inside tmux)
#
# Stage order: 3 4 5 5c 5e 6 7 8 (stages 1-2 are run with run_stage.sh / run_stage_2gpu.sh).
# Every stage goes through run_stage.sh, so it is skipped per episode when its stamp is up to date
# and otherwise re-run from scratch (old outputs archived) — see track1/stamps.py. Episodes are
# split over the two GPUs by frame count; for 5e by object, so an object's videos stay together.
# Log: logs/stages/run_rest.log
# =============================================================================
set -uo pipefail
CODE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
export V2D_DATA="${V2D_DATA:-/local-data/ivan/v2d}"
cd "$V2D_DATA"
GPUS=(${GPUS:-0 2})
FROM="${FROM:-3}"; TO="${TO:-8}"
EPS="${EPISODES:-$(seq -s ' ' 0 29)}"
LOG=logs/stages/run_rest.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True V2D_BACKEND=venv
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

split2() {  # $1 = stage, rest = episodes -> two lines (balanced by frames; 5e: whole objects)
    python3 - "$@" <<'PY'
import json, sys
st, eps = sys.argv[1], [int(x) for x in sys.argv[2:]]
fr = lambda e: json.load(open(f"out/episode_{e:06d}/track1_meta.json"))["n_frames"]
if st == "5e":
    meta = [json.loads(l) for l in open("challenge/track_1/meta/episodes_metadata.jsonl")]
    obj = {m["episode_index"]: m["object"] for m in meta}
    groups = {}
    for e in eps:
        groups.setdefault(obj[e], []).append(e)
    units = list(groups.values())
else:
    units = [[e] for e in eps]
a, b, la, lb = [], [], 0, 0
for u in sorted(units, key=lambda u: -sum(fr(e) for e in u)):
    if la <= lb: a += u; la += sum(fr(e) for e in u)
    else: b += u; lb += sum(fr(e) for e in u)
print(" ".join(map(str, sorted(a)))); print(" ".join(map(str, sorted(b))))
PY
}

run_on_two() {  # $1 = stage: one run_stage.sh per GPU, wait for both
    local st="$1" rc=0 pa pb
    mapfile -t L < <(split2 "$st" $EPS)
    say "stage $st: GPU ${GPUS[0]} <- ${L[0]:-none} | GPU ${GPUS[1]} <- ${L[1]:-none}"
    pa=""; pb=""
    [[ -n "${L[0]:-}" ]] && { GPU=${GPUS[0]} EPISODES="${L[0]}" "$CODE/run_stage.sh" "$st" > /dev/null 2>&1 & pa=$!; }
    [[ -n "${L[1]:-}" ]] && { GPU=${GPUS[1]} EPISODES="${L[1]}" "$CODE/run_stage.sh" "$st" > /dev/null 2>&1 & pb=$!; }
    [[ -n "$pa" ]] && { wait "$pa" || rc=1; }
    [[ -n "$pb" ]] && { wait "$pb" || rc=1; }
    say "stage $st done ($([[ $rc -eq 0 ]] && echo 'no failures' || echo 'FAILURES — see its batch.log'))"
    return $rc
}

ORDER=(3 4 5 5c 5e 6 7 8)
say "===== start: stages $FROM..$TO on GPUs ${GPUS[*]} ====="
on=0
for st in "${ORDER[@]}"; do
    [[ "$st" == "$FROM" ]] && on=1
    [[ $on -eq 1 ]] || continue
    if [[ "$st" == 7 ]]; then        # CPU only, seconds per episode
        EPISODES="$EPS" "$CODE/run_stage.sh" 7 > /dev/null 2>&1; rc=$?
        say "stage 7 done ($([[ $rc -eq 0 ]] && echo 'no failures' || echo 'FAILURES'))"
    else
        run_on_two "$st"; rc=$?
    fi
    # failed episodes stay unstamped, so later stages refuse to run on them; keep going for the rest
    [[ "$st" == "$TO" ]] && break
done
python3 "$CODE/track1/stamps.py" status --episodes $EPS | tee -a "$LOG"
say "===== done ====="
