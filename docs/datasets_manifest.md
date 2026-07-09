# Datasets manifest

The project now covers **four** modalities. All share the same PixelFlow posterior
sampler; they differ in prior/checkpoint, resolution, data, and forward operator.

| dataset | res | prior (checkpoint) | data | operators | reproducible here? |
|---|---|---|---|---|---|
| **ImageNet-256** | 256 | c2img (class-cond, 1000) | DAPS 100 val imgs (group_100) | SR/gaussian/motion/box/random (DAPS-aligned) | ✅ full |
| **AFHQ-512** | 512 | afhq (cat/dog/wild, 3) `model_epoch80_fid39.02.pt` (7.3 GB) | demo_afhq (18 imgs) | SR/gaussian/motion/box/random | ✅ full |
| **CelebA-256** | 256 | exp_155024 (2-class) `model_epoch0020.pt` (7.3 GB) | celeba groups (5×100 `.pt`) | SR(×2)/gaussian/box/random | ✅ full |
| **MRI-384** | 384 | MRI prior **(MISSING)** | brain AXT2 slices + SENSE maps | SENSE k-space (acc8) | ⚠️ reference-only |

## Per-dataset entry points
| dataset | configs | runner | reference results |
|---|---|---|---|
| ImageNet | `configs/imagenet/*.yaml` | `benchmark/run_ours.py` | `results/previous_results/ours_rerun_best/` |
| AFHQ | `configs/afhq/*.yaml` | `benchmark/run_afhq.py` (reuses `demo_runner`) | `results/previous_results/afhq/demo_runs_afhq/` |
| CelebA | `configs/celeba/*.yaml` | `benchmark/run_celeba.py` (reuses `celeba_results/code/evaluate.py`) | `results/previous_results/celeba/{runs,metrics}/` |
| MRI | `configs/mri/*` + `configs/models/mri.yaml` | `benchmark/run_mri.py` (reference summary / re-sample if prior restored) | `results/previous_results/mri/val_final_8/` |

## AFHQ-512 notes
- Same `run_posterior_sampling` + `demo_runner.{load_demo_images,build_setup_and_measurement}`
  as ImageNet, but model_dir/res/demo set from `paths.local.yaml` (`afhq_*`). batch_size 1.
- Operators scaled for 512² (e.g. gaussian k=121 σ=6, box 256²) — see each config's comment.
- Reference per-image recons live under `demo_runs_afhq/<task>/<tag>/<image_name>/`; aggregate
  metrics in `metrics_all.csv` / `best_summary.md`.

## CelebA-256 notes
- Self-contained harness reused verbatim: `celeba_results/code/{evaluate,run_all,prepare_groups,
  metrics,aggregate}.py`. Groups are `group_{0..4}.pt` tensors (fnames/labels/gts/masks).
- **SR is ×2** (not ×4). Labels all class 0. Metrics are COMPLETE (500/config, 5 groups) in
  `results/previous_results/celeba/metrics/metrics_by_config.csv`.

## MRI-384 notes (reference-only)
- PixelFlow MRI prior + `mri_operator.MRIReconstructionOperator` (SENSE, cgIter 50) + `run_ip4`.
- **Prior checkpoint is missing** → cannot re-sample; `run_mri.py` prints the recorded reference
  metrics (val_final_8 slices: per-sample + MMSE, nsamples=16, acc8).
- **Normalization (verified):** GT `x` is already peak-normalized (|x|max=1), csm unit-power;
  `scale` (~1300) applies to the measurement only. **Do not re-normalize.** Full contract:
  `docs/mri_normalization_notes.md`.
