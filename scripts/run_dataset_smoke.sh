#!/usr/bin/env bash
# Smoke test for the EXTRA datasets (AFHQ-512, CelebA-256, MRI-384).
# AFHQ + CelebA actually sample a couple of images; MRI prints its reference
# metrics (the smoke stays fast — re-sampling with the restored prior is opt-in:
# run benchmark/run_mri.py without --summarize_reference). Does not overwrite
# previous results.
#
#   bash scripts/run_dataset_smoke.sh                 # all 3, default GPU
#   DATASETS="afhq celeba" NUM_IMAGES=1 GPU=4 bash scripts/run_dataset_smoke.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

: "${NUM_IMAGES:=1}"
: "${DATASETS:=afhq celeba mri}"
OUT="${PROJECT_ROOT}/results/reproduced_results/dataset_smoke"

log_cmd "DATASET SMOKE: DATASETS='${DATASETS}' NUM_IMAGES=${NUM_IMAGES} GPU=${GPU}"
for ds in ${DATASETS}; do
  case "$ds" in
    afhq)
      log_cmd "${PY} benchmark/run_afhq.py --config configs/afhq/superresolution.yaml --num_images ${NUM_IMAGES}"
      ${PY} benchmark/run_afhq.py --config configs/afhq/superresolution.yaml --paths "${PATHS_YAML}" \
        --num_images "${NUM_IMAGES}" --output_dir "${OUT}/afhq" --gpu "${GPU}" 2>&1 | tee -a "${LOG}" ;;
    celeba)
      log_cmd "${PY} benchmark/run_celeba.py --config configs/celeba/superresolution.yaml --num_images ${NUM_IMAGES} --group_id 0"
      ${PY} benchmark/run_celeba.py --config configs/celeba/superresolution.yaml --paths "${PATHS_YAML}" \
        --num_images "${NUM_IMAGES}" --group_id 0 --output_dir "${OUT}/celeba" --gpu "${GPU}" 2>&1 | tee -a "${LOG}" ;;
    mri)
      log_cmd "${PY} benchmark/run_mri.py --paths ${PATHS_YAML} --summarize_reference  (reference summary)"
      ${PY} benchmark/run_mri.py --paths "${PATHS_YAML}" --summarize_reference 2>&1 | tee -a "${LOG}" ;;
    *) echo "unknown dataset: $ds" ;;
  esac
done
echo "[dataset-smoke] done. Outputs under ${OUT}" | tee -a "${LOG}"
