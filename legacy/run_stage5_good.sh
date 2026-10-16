#!/usr/bin/env bash
# Stage 5 (mesh) for the episodes whose stages 3-4 were approved in review.
cd /local-data/ivan/v2d
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/stages/run_stage5_good.log
echo "===== $(date -Is) start =====" >> "$LOG"
EPISODES="0-5 7 9 10 13-16 21-23 25 27" ./run_stage.sh 5 > /dev/null 2>&1
echo "[$(date +%T)] stage 5 exit=$? — $(tail -1 logs/stages/05_mesh/batch.log)" >> "$LOG"
echo "===== $(date -Is) done =====" >> "$LOG"
