#!/usr/bin/env bash
# Smoke test: run OUR method on a few images per task, then evaluate. Fast sanity
# that the whole pipeline (model load -> operator -> sampler -> metrics) works on
# THIS machine. Does NOT overwrite previous results. Default 3 images/task.
#
#   bash scripts/run_local_smoke_test.sh                 # 3 imgs, all tasks, default GPU
#   NUM_IMAGES=5 GPU=3 bash scripts/run_local_smoke_test.sh
#   TASKS=super_resolution bash scripts/run_local_smoke_test.sh   # one task
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

: "${NUM_IMAGES:=3}"
: "${TASKS:=all}"

log_cmd "SMOKE start: NUM_IMAGES=${NUM_IMAGES} TASKS=${TASKS} GPU=${GPU}"
CMD="${PY} benchmark/reproduce_previous_results.py --paths ${PATHS_YAML} \
  --num_images ${NUM_IMAGES} --tasks ${TASKS} --gpu ${GPU} --overwrite"
log_cmd "${CMD}"
${CMD} 2>&1 | tee -a "${LOG}"
echo "[smoke] done. Per-task summary in results/metrics/reproduction_smoke${NUM_IMAGES}.json" | tee -a "${LOG}"
