# Config manifest

All configs are plain YAML. Absolute paths live ONLY in `paths.local.yaml`.

| file | role |
|---|---|
| `configs/paths.local.yaml` | **single source of truth for every absolute path** (original code dirs, checkpoint, data, output roots, python_bin, default GPU, seed, batch_size) |
| `configs/models/ours.yaml` | model card: checkpoint location, load convention, guidance |
| `configs/eval/metrics.yaml` | the one metric stack (piq PSNR/SSIM + piq-VGG LPIPS = "LPIPS_king"; official LPIPS as cross-check) |
| `configs/baselines/baselines.yaml` | baseline registry: per-method result dirs, task-slug map, recon path patterns, caveats |
| `configs/imagenet/super_resolution.yaml` | SR ×4, `operator.antialias: true` (DAPS Resizer-aligned) |
| `configs/imagenet/deblur.yaml` | gaussian k=61 σ=3 (operator IDENTICAL to DAPS) |
| `configs/imagenet/motion_deblur.yaml` | motion k=61, `use_daps_kernel: true`, `kernel_seed: 42` |
| `configs/imagenet/inpainting.yaml` | box 128² (DAPS `random_sq_bbox`), noiseless |
| `configs/imagenet/inpainting_random.yaml` | random 70% missing, noiseless |

## Provenance of the task configs
Each `imagenet/*.yaml` was **auto-generated** (no hand transcription) from the
canonical DAPS-aligned JSON in
`rerun_imageNet/configs_best_out/<task>_best.json` — the exact configs that
produced `results/previous_results/ours_rerun_best/`. The full sampler `kw`
(num_langevin, ode_steps_per_stage, h_x, lambda_reg/prox, guidance_scale=2.0,
scheduled_h_x, …) is embedded verbatim so the YAML is self-contained.

## Mapping: local YAML → previous-result subdir
| YAML | task | reproduces |
|---|---|---|
| super_resolution.yaml | superresolution | `group100_superresolution_best` |
| deblur.yaml | gaussian_blur | `group100_gaussian_blur_best` |
| motion_deblur.yaml | motion_blur | `group100_motion_blur_best` |
| inpainting.yaml | box_inpainting | `group100_box_inpainting_best` |
| inpainting_random.yaml | random_inpainting | `group100_random_inpainting_best` |

`validation_tag` defaults to `<task>_best`; output subdir = `group<group_id>_<validation_tag>`.
`group_id: 100` = DAPS's exact 100 ImageNet-val images. `batch_size: 8` is FIXED
(it sets the per-image seed stream under all-seeds-42; do not change for comparison).
