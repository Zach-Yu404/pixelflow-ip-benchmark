# Release migration notes (local → official)

This local project is intentionally **machine-bound** (absolute paths, symlinks,
on-disk 2.7 GB checkpoint, code reused from the original IP_package) so it
reproduces the prior results exactly. To turn it into a clean public release:

## 1. Cut the symlinks
| local symlink | release action |
|---|---|
| `checkpoints/ours/model.pt` (→ c2img blob) | provide a `download_checkpoint.sh` (HF/GCS); ship `config.yaml` + a model card, not the blob |
| `results/baselines/*`, `results/previous_results/*`, `results/figures/*` | publish a results tarball / release asset; or regenerate via scripts |
| `results/previous_results/groups_used` (→ rerun groups) | ship a `build_groups.py` that rebuilds group_100 from ImageNet-val (uses `imagenet_val_*` in paths) |

## 2. Vendor the reused code
`src/` currently re-exports the original `rerun_imageNet/{runner,operators,metrics,
seeding}.py` and `ms_posterior_sampling_*` + the `pixelflow` package via
`sys.path` bootstrap. For release, **vendor** those modules into `src/` (copy the
files, drop the `bootstrap()` sys.path hack) and add `pixelflow` as a dependency or
submodule. The DAPS motion kernel needs the DAPS repo (`daps_repo`) — vendor
`forward_operator/motionblur/` or gate motion behind an optional extra.

## 3. De-absolutize paths
`configs/paths.local.yaml` is the ONLY file with absolute paths. Replace it with a
`paths.example.yaml` of relative defaults + env-var overrides; nothing else in the
codebase hard-codes a path, so this is a one-file change.

## 4. Keep
- `configs/imagenet/*.yaml` (self-contained, no paths) — ship as-is.
- `configs/{models,eval,baselines}/*.yaml` — ship (drop any absolute hints).
- `benchmark/*.py`, `scripts/*.sh` — ship (they read paths from the yaml).
- The honesty docs (operator alignment, perception–distortion framing, OOD baseline
  caveats) — ship; they're the credibility of the comparison.

## 5. Reproducibility caveats to document publicly
- **SR is not bit-reproducible** on CUDA: `upsample_bicubic2d_aa_backward` is
  non-deterministic (PyTorch). Per-image SR varies ~0.5 dB run-to-run; the 100-image
  **mean** is stable. Deterministic tasks (gaussian/inpaint) reproduce bit-exactly.
- Pin torch==2.6.0+cu124; different CUDA/cuDNN can shift the 3rd decimal.
- `batch_size=8` is part of the seed contract under all-seeds-42; don't change it
  for cross-config comparison.
