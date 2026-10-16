#!/usr/bin/env bash
# =============================================================================
# progress.sh — live dashboard for run_all.sh.
#
#   ./progress.sh        print one snapshot and exit
#   ./progress.sh -w     refresh every few seconds (Ctrl-C to quit)
#   ./progress.sh -n 5   watch with a 5s interval
#
# Shows: whether the batch is alive, GPU memory, a status table of every
# episode under out/, and a detail block for the episode currently running
# (elapsed time, stage checklist from on-disk artifacts, and the tail of its
# log).
# =============================================================================
set -uo pipefail

BASEPATH="${BASEPATH:-/local-data/ivan/v2d}"
OUT="$BASEPATH/out"
SUMMARY="$OUT/_summary.tsv"

WATCH=0
INTERVAL=3
while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--watch) WATCH=1; shift ;;
        -n) INTERVAL="$2"; WATCH=1; shift 2 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Ordered pipeline artifacts -> short stage label.
STAGE_FILES=(
    "frames:frames"
    "depth:depth"
    "wilor_raw:wilor"
    "sam2_prompts.json:prompts"
    "masks:masks"
    "mesh_scaled.obj:mesh"
    "poses:pose"
    "poses_smoothed:smooth"
    "mhr_mesh.pt:body"
    "body_object_overlay.mp4:overlay"
)

status_of() {  # $1 = episode name -> status from summary, or "" if not finished
    [[ -f "$SUMMARY" ]] || { echo ""; return; }
    local n="${1#episode_}"; n="$((10#$n))"
    awk -F'\t' -v n="$n" 'NR>1 && $1==n {print $3; exit}' "$SUMMARY"
}

count() {  # $1 = dir -> file count (0 if missing)
    [[ -d "$1" ]] && find "$1" -maxdepth 1 -type f 2>/dev/null | wc -l || echo 0
}

snapshot() {
    local now alive gpu
    now="$(date '+%H:%M:%S')"

    if pgrep -f "run_all.sh" >/dev/null 2>&1; then
        alive="\033[32mRUNNING\033[0m"
    else
        alive="\033[31mnot running (run_all.sh finished or stopped)\033[0m"
    fi
    gpu="$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
           --format=csv,noheader 2>/dev/null | head -1)"

    echo -e "═══ batch @ $now — $alive"
    [[ -n "$gpu" ]] && echo "    GPU: $gpu"
    echo

    # Per-episode status table.
    echo "  episode   status"
    echo "  ────────  ──────────────────"
    local active="" d ep st
    for d in "$OUT"/episode_*/; do
        [[ -d "$d" ]] || continue
        ep="$(basename "$d")"
        st="$(status_of "$ep")"
        if [[ -z "$st" ]]; then
            if [[ -f "$d/run.log" ]]; then
                st="… in progress"
                active="$ep"
            else
                st="pending"
            fi
        fi
        printf "  %s  %b\n" "${ep#episode_}" "$st"
    done
    echo

    # Detail for the active episode (pick most recently touched unfinished log).
    local log newest=""
    for log in $(ls -t "$OUT"/episode_*/run.log 2>/dev/null); do
        ep="$(basename "$(dirname "$log")")"
        [[ -z "$(status_of "$ep")" ]] && { newest="$log"; active="$ep"; break; }
    done
    [[ -z "$newest" ]] && { echo "  (no active episode — batch idle or done)"; return; }

    local d_ep="$OUT/$active"
    local started elapsed=""
    started="$(grep -m1 '^===== ' "$d_ep/run.log" | sed -E 's/^===== ([^ ]+) .*/\1/')"
    if [[ -n "$started" ]]; then
        local s0 s1
        s0="$(date -d "$started" +%s 2>/dev/null)"; s1="$(date +%s)"
        [[ -n "$s0" ]] && elapsed="$(( (s1 - s0) / 60 ))m$(( (s1 - s0) % 60 ))s"
    fi

    echo -e "─── active: \033[1m$active\033[0m   elapsed ${elapsed:-?}"

    # Stage checklist from artifacts.
    local entry f label mark line="   "
    for entry in "${STAGE_FILES[@]}"; do
        f="${entry%%:*}"; label="${entry##*:}"
        if [[ -e "$d_ep/$f" ]]; then mark="\033[32m✓\033[0m"; else mark="·"; fi
        line+=" ${mark} ${label}"
    done
    echo -e "$line"

    # Frames-vs-depth progress (coarse % for the long MoGe stage).
    local nf nd
    nf="$(count "$d_ep/frames")"; nd="$(count "$d_ep/depth")"
    [[ "$nf" -gt 0 ]] && echo "    frames: $nf   depth: $nd/$nf"

    # Last meaningful log lines. Split carriage-return progress bars into real
    # lines, drop ffmpeg/tqdm/warning spam, keep the last few, bound the width.
    echo "    log tail:"
    tr '\r' '\n' < "$d_ep/run.log" 2>/dev/null \
        | grep -avE 'frame=|pynvml|FutureWarning|UserWarning|Press \[q\]|Stream #|Metadata|handler_name|vendor_id|major_brand|minor_version|compatible_brands|encoder|Duration:|bitrate=|muxing overhead|video:[0-9]' \
        | grep -avE '^\s*$' \
        | sed -E 's/(:[[:space:]]*[0-9]+%)\|[^0-9]*/\1 /' \
        | tail -n 4 | cut -c1-110 | sed 's/^/      /'
}

if [[ "$WATCH" == "1" ]]; then
    while true; do
        clear
        snapshot
        sleep "$INTERVAL"
    done
else
    snapshot
fi
