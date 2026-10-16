#!/usr/bin/env bash
# Stages 3-4 only for the episodes flagged in review (12 already redone).
cd /local-data/ivan/v2d
LOG=logs/stages/run_fix_flagged.log
echo "===== $(date -Is) start =====" >> "$LOG"
for e in 11 18 19 20 29; do rm -rf "out/$(printf episode_%06d $e)/select"; done
EPISODES="11 18 19 20 29" ./run_stage.sh 3 > /dev/null 2>&1
echo "[$(date +%T)] stage 3 exit=$? — $(tail -1 logs/stages/03_select/batch.log)" >> "$LOG"
EPISODES="11 18 19 20 24 29" ./run_stage.sh 4 > /dev/null 2>&1
echo "[$(date +%T)] stage 4 exit=$? — $(tail -1 logs/stages/04_mask/batch.log)" >> "$LOG"
echo "===== $(date -Is) done =====" >> "$LOG"
