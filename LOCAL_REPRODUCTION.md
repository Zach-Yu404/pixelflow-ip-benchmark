# LOCAL_REPRODUCTION.md

The exact contract for reproducing our results on **this machine**. Everything
machine-specific is centralized in `configs/paths.local.yaml`; this doc explains
the moving parts and the commands.

## 0. Prerequisites
- Interpreter: `~/anaconda3/envs/pixelflow/bin/python` (conda env
  `pixelflow`; torch 2.6.0+cu124, piq 0.8.0, lpips 0.1.4). See `requirements.txt`.
- A free GPU (default `0` — on the CBIG cluster request one via Slurm; the index is
  within your allocation). Override with `GPU=<n>`.
- The original IP_package present at the path in `paths.local.yaml:original_ip_package`
  (we import the verified sampler/operators/metrics from it).

## 1. The path contract (`configs/paths.local.yaml`)
Single source of truth. Key entries:
- `original_ip_package`, `original_rerun_imagenet`, `daps_repo` — code we reuse.
- `model_dir` = `checkpoints/ours` (symlink → c2img: `config.yaml` + `model.pt`).
- `groups_dir` = `rerun_imageNet/groups`; `daps_group_id: 100` = DAPS's 100 images.
- `output_root`, `metrics_root`, `figures_root` — where reproduced artifacts land.
- `python_bin`, `default_gpu`, `seed: 42`, `batch_size: 8`.
No other file contains an absolute path.

## 2. The seed contract (all-seeds-42)
`rerun_imageNet/seeding.py`: every RNG is reset to **42** before each batch's
measurement and before each sampler call. Cross-config fairness holds because image
order is fixed and **`batch_size=8`** pins each image to the same RNG-stream position.
**Do not change `batch_size`** when comparing configs.

## 3. The operator contract (DAPS-aligned)
See `docs/operator_alignment_notes.md`. SR `antialias:true` (≈DAPS Resizer, |Δ|0.0002),
gaussian identical (|Δ|0), motion injects DAPS `motionblur.Kernel` (|Δ|0, seed 42),
box uses DAPS `random_sq_bbox`, random = 70% missing. Implemented by the reused
`operators.py`; the YAML `operator:` block carries the alignment flags.

## 4. The model contract
`checkpoints/ours/model.pt` is a **flat state_dict** (2.7 GB), loaded `strict=True`.
Class-conditional (1000 classes) + classifier-free guidance (`kw.guidance_scale=2.0`).
DAPS is unconditional — we hold a class-info advantage and still trail PSNR
(perception–distortion). See `docs/checkpoint_manifest.md`.

## 5. Commands

### 5.1 Smoke test (fast, a few images/task)
```bash
bash scripts/run_local_smoke_test.sh                 # NUM_IMAGES=3, all tasks
NUM_IMAGES=5 GPU=3 bash scripts/run_local_smoke_test.sh
```
Outputs `results/reproduced_results/smoke<N>/group100_<task>_best/` and
`results/metrics/reproduction_smoke<N>.json`.

### 5.2 Re-evaluate existing results (no sampling)
```bash
bash scripts/evaluate_existing_results.sh
```
Re-scores the reference recons with this project's metric stack and checks them
against the recorded `meta.json` means (flags any DRIFT). Also rebuilds
`results/metrics/baseline_comparison.md`.

### 5.3 Full reproduction (all 100 images/task — slow)
```bash
GPU=3 bash scripts/reproduce_previous_results.sh --full --overwrite
```

### 5.4 Manual single task
```bash
PY=~/anaconda3/envs/pixelflow/bin/python
$PY benchmark/generate_measurements.py --config configs/imagenet/deblur.yaml \
    --paths configs/paths.local.yaml --num_images 5 \
    --output_dir results/reproduced_results/meas/deblur --gpu 2
$PY benchmark/run_ours.py --config configs/imagenet/deblur.yaml \
    --model_config configs/models/ours.yaml --paths configs/paths.local.yaml \
    --checkpoint checkpoints/ours/model.pt --num_images 5 \
    --output_dir results/reproduced_results/demo --gpu 2
$PY benchmark/evaluate.py --config configs/eval/metrics.yaml \
    --gt_dir results/reproduced_results/demo/group100_gaussian_blur_best/gts \
    --result_dir results/reproduced_results/demo/group100_gaussian_blur_best/recons \
    --output_csv results/metrics/demo_deblur.csv
$PY benchmark/visualize.py --metrics_csv results/metrics/demo_deblur.csv \
    --output_dir results/figures/demo
```

### 5.5 MRI-384 (prior restored)
Task configs live in `configs/mri/` (`sense_acc8.yaml` = the exact val_final_8
reference protocol; `sense_acc4.yaml` = exploratory, no reference). `run_mri.py`
translates the YAML to the verified `val_final_8/run.py` CLI (tag → `--config`,
`operator.mask` paths-key → `--mask_path`). One process per slice:
```bash
$PY benchmark/run_mri.py --slice_idx 0 --gpu 0             # full protocol: 16 samples, acc8
$PY benchmark/run_mri.py --slice_idx 0 --gpu 0 --nsamples 1  # smoke
$PY benchmark/run_mri.py --paths configs/paths.local.yaml --summarize_reference
```
Outputs land in `results/reproduced_results/mri/<acc>/slice<idx>/` (gt/mmse/std/
samples + meta.json); compare against `results/previous_results/mri/val_final_8`.

## 6. What "reproduced" means here (honesty)
- **GTs / measurements / masks**: bit-exact (data pipeline fully deterministic).
- **Deterministic tasks** (gaussian, inpainting): recons bit-exact vs the reference
  (verified — see `docs/local_reproduction_report.md`, kept locally, not in the repo).
- **SR**: per-image varies ~0.5 dB run-to-run because
  `upsample_bicubic2d_aa_backward` is non-deterministic on CUDA (PyTorch limitation,
  present in the original runs too). The 100-image **mean** is stable. This is a
  hardware/kernel property, **not** a pipeline bug.
- A smoke subset's mean ≠ the 100-image mean; compare per-image (idx-matched) or run
  `--full` for mean-level comparison.

## 7. Troubleshooting
- `CUDA out of memory` → pick a free GPU (`nvidia-smi`), `GPU=<n>`.
- `missing config.yaml/model.pt` → the checkpoint symlink is broken; re-point it at
  `PixelFlow/pretrained_models/c2img` (next to `paths.local.yaml:original_ip_package`).
- import errors for `pixelflow`/`ms_posterior_sampling_*` → `original_ip_package` in
  `paths.local.yaml` is wrong; `bootstrap()` adds it to `sys.path`.
- A task SKIPs → output already exists; pass `--overwrite`.
