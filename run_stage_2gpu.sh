#!/usr/bin/env bash
# =============================================================================
# run_stage_2gpu.sh — run ONE stage on two GPUs at once, in two tmux sessions.
#
#   ./run_stage_2gpu.sh 5 "0 1" "4 5 7 9 10 13-16 21-23 25 27"
#        stage   GPUs     episodes
#
# Episodes are split into two lists with about the same number of frames (longest first,
# each to the lighter GPU). Sessions: <stage>gpu<A> and <stage>gpu<B>, e.g. s5gpu0 / s5gpu1.
# Logs: logs/stages/<NN_stage>/ (shared review page, per-episode logs), plus
#       logs/stages/run_stage<N>_gpu<G>.log (one line when each half finishes).
# =============================================================================
set -uo pipefail
CODE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
export V2D_DATA="${V2D_DATA:-/local-data/ivan/v2d}"
cd "$V2D_DATA"
STAGE="${1:?stage}"; GPUS=(${2:?"two GPU ids, e.g. \"0 1\""}); EPS="${3:?episodes}"
[[ ${#GPUS[@]} -eq 2 ]] || { echo "need exactly two GPU ids"; exit 2; }

expand() { local t; for t in $1; do if [[ "$t" == *-* ]]; then seq "${t%-*}" "${t#*-}"; else echo "$t"; fi; done; }
read -r LIST_A LIST_B < <(python3 - "$(expand "$EPS" | tr '\n' ' ')" <<'PY'
import json, sys
eps = [int(x) for x in sys.argv[1].split()]
def frames(e):
    try:
        return json.load(open(f"out/episode_{e:06d}/track1_meta.json"))["n_frames"]
    except Exception:
        return 400
a, b, la, lb = [], [], 0, 0
for e in sorted(eps, key=frames, reverse=True):
    if la <= lb:
        a.append(e); la += frames(e)
    else:
        b.append(e); lb += frames(e)
print(",".join(map(str, sorted(a))), ",".join(map(str, sorted(b))))
PY
)
LIST_A="${LIST_A//,/ }"; LIST_B="${LIST_B//,/ }"
for i in 0 1; do
    G="${GPUS[$i]}"; L=$([[ $i -eq 0 ]] && echo "$LIST_A" || echo "$LIST_B")
    S="s${STAGE}gpu${G}"
    tmux kill-session -t "$S" 2>/dev/null
    tmux new-session -d -s "$S" -c "$V2D_DATA" \
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; GPU=$G EPISODES='$L' V2D_DATA='$V2D_DATA' '$CODE/run_stage.sh' $STAGE; \
         echo \"[\$(date +%T)] stage $STAGE gpu $G exit=\$? episodes: $L\" >> logs/stages/run_stage${STAGE}_gpu${G}.log; \
         echo 'finished - press Enter'; read"
    echo "started tmux session $S on GPU $G: episodes $L"
done
