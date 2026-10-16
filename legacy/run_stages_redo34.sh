#!/usr/bin/env bash
# Redo: stage 2 for episode 0 (body fit was killed), then stages 3-4 with the new detection/mask fixes.
cd /local-data/ivan/v2d
LOG=logs/stages/run_redo34.log
echo "===== $(date -Is) start =====" >> "$LOG"
rm -f out/episode_000000/body/mhr_fit.npz
EPISODES="0" ./run_stage.sh 2 > /dev/null 2>&1
echo "[$(date +%T)] stage 2 ep0 exit=$? — $(tail -1 logs/stages/02_body/batch.log)" >> "$LOG"
export EPISODES="${EPISODES:-0-5 7 9-16 18-25 27 29}"
for e in $EPISODES; do :; done
for e in $(for t in $EPISODES; do [[ $t == *-* ]] && seq ${t%-*} ${t#*-} || echo $t; done); do
    rm -rf "out/$(printf episode_%06d $e)/select"            # redo stage 3 with the new views
done
for s in 3 4; do
    ./run_stage.sh "$s" > /dev/null 2>&1
    echo "[$(date +%T)] stage $s exit=$? — $(tail -1 logs/stages/0${s}_*/batch.log)" >> "$LOG"
done
echo "===== $(date -Is) done =====" >> "$LOG"
