#!/usr/bin/env bash
# Runs stages 1-4 one after another on EPISODES; detached-safe (start with setsid nohup).
cd /local-data/ivan/v2d
export EPISODES="${EPISODES:-0-5 7 9-16 18-25 27 29}"
LOG=logs/stages/run_1to4.log
echo "===== $(date -Is) start, episodes: $EPISODES =====" >> "$LOG"
for s in 1 2 3 4; do
    ./run_stage.sh "$s" > /dev/null 2>&1
    echo "[$(date +%T)] stage $s exit=$? — $(tail -1 logs/stages/0${s}_*/batch.log)" >> "$LOG"
done
echo "===== $(date -Is) done =====" >> "$LOG"
