#!/usr/bin/env bash
# Reproduce OUR previous results. Default = smoke (5 imgs/task). Pass --full for
# all 100 DAPS images per task (slow: ~10-60 min/task on one GPU).
#
#   bash scripts/reproduce_previous_results.sh            # smoke, 5 imgs/task
#   bash scripts/reproduce_previous_results.sh --full     # full 100/task
#   GPU=3 bash scripts/reproduce_previous_results.sh --full --overwrite
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

FULL=""
OVERWRITE=""
NUM_IMAGES="${NUM_IMAGES:-5}"
for a in "$@"; do
  case "$a" in
    --full) FULL="--full" ;;
    --overwrite) OVERWRITE="--overwrite" ;;
    *) echo "unknown arg: $a"; exit 2 ;;
  esac
done

MODE="smoke${NUM_IMAGES}"; [ -n "${FULL}" ] && MODE="full100"
log_cmd "REPRODUCE start: MODE=${MODE} GPU=${GPU}"
CMD="${PY} benchmark/reproduce_previous_results.py --paths ${PATHS_YAML} \
  --num_images ${NUM_IMAGES} --gpu ${GPU} ${FULL} ${OVERWRITE}"
log_cmd "${CMD}"
${CMD} 2>&1 | tee -a "${LOG}"

# refresh the cross-method comparison from whatever results exist
log_cmd "${PY} benchmark/compare_baselines.py --paths ${PATHS_YAML}"
${PY} benchmark/compare_baselines.py --paths "${PATHS_YAML}" 2>&1 | tee -a "${LOG}"
echo "[reproduce] done (${MODE}). See results/metrics/reproduction_${MODE}.json" | tee -a "${LOG}"
