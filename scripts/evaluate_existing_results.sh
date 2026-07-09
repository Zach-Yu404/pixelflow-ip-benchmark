#!/usr/bin/env bash
# Re-evaluate the EXISTING previous results (no GPU sampling, just metrics) to
# confirm the recorded numbers reproduce with this project's metric stack, then
# rebuild the cross-method comparison. This is the "re-evaluation" the repro spec
# requires — it touches only existing recons/gts, never regenerates them.
#
#   bash scripts/evaluate_existing_results.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PREV="${PROJECT_ROOT}/results/previous_results/ours_rerun_best"
mkdir -p "${PROJECT_ROOT}/results/metrics" "${PROJECT_ROOT}/results/figures/existing_eval"

log_cmd "EVALUATE-EXISTING start: GPU=${GPU}"
CSVS=""
for d in "${PREV}"/group100_*_best; do
  [ -d "${d}/recons" ] || continue
  tag="$(basename "${d}")"
  csv="${PROJECT_ROOT}/results/metrics/existing_${tag}.csv"
  log_cmd "${PY} benchmark/evaluate.py --gt_dir ${d}/gts --result_dir ${d}/recons --output_csv ${csv}"
  ${PY} benchmark/evaluate.py --paths "${PATHS_YAML}" \
    --gt_dir "${d}/gts" --result_dir "${d}/recons" --output_csv "${csv}" --gpu "${GPU}" 2>&1 | tee -a "${LOG}"
  # compare re-eval mean vs recorded meta.json mean
  ${PY} - "${d}/meta.json" "${csv}" <<'PYEOF' 2>&1 | tee -a "${LOG}"
import json, csv, sys
meta = json.load(open(sys.argv[1]))
rec = None
for r in csv.DictReader(open(sys.argv[2])):
    if str(r["idx"]).upper() == "MEAN": rec = r
rp, rs, rl = float(rec["psnr"]), float(rec["ssim"]), float(rec["lpips"])
mp, ms, ml = meta["psnr_mean"], meta["ssim_mean"], meta["lpips_mean"]
dp = abs(rp-mp)
flag = "OK" if dp < 0.05 else ("~" if dp < 0.5 else "DRIFT")
print(f"[check] {meta['task']:<18} recorded PSNR={mp:.3f} reeval PSNR={rp:.3f}  dPSNR={dp:.3f} [{flag}]")
PYEOF
  CSVS="${CSVS}${CSVS:+,}${csv}"
done

log_cmd "${PY} benchmark/visualize.py --metrics_csv ${CSVS} --output_dir results/figures/existing_eval"
${PY} benchmark/visualize.py --metrics_csv "${CSVS}" --output_dir results/figures/existing_eval 2>&1 | tee -a "${LOG}"
log_cmd "${PY} benchmark/compare_baselines.py --paths ${PATHS_YAML}"
${PY} benchmark/compare_baselines.py --paths "${PATHS_YAML}" 2>&1 | tee -a "${LOG}"
echo "[evaluate-existing] done. CSVs in results/metrics/existing_*.csv" | tee -a "${LOG}"
