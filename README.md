# PixelFlow Inverse-Problem Benchmark — Local Reproduction

A clean, **locally reproducible** packaging of our flow-based posterior-sampling
method for image inverse problems (super-resolution, Gaussian/motion deblur,
box/random inpainting), with a DAPS-aligned baseline comparison. Covers **four
modalities**: **ImageNet-256** (+ DAPS baselines), **AFHQ-512**, **CelebA-256**, and
**MRI-384** (SENSE k-space; prior restored — see below). See
[docs/datasets_manifest.md](docs/datasets_manifest.md).

> This is the **local** project (absolute paths / symlinks / on-disk checkpoints
> are allowed). It reuses the already-validated original code so numbers reproduce
> bit-for-bit. For the structure of a future official release, see
> [docs/release_migration_notes.md](docs/release_migration_notes.md).

## What this reproduces

On DAPS's exact 100 ImageNet-val images (val 49000–49099), all operators aligned to
DAPS, one metric ruler (PSNR / SSIM / LPIPS, LPIPS = piq-VGG): five tasks —
SR ×4, gaussian deblur, motion deblur, random inpainting, box inpainting —
sampled with our method and evaluated against the recorded reference runs.
Quantitative tables are kept out of this repo; they live with the results on the
reproduction machine (`results/` symlinks) and in the paper draft.

## Layout

```
configs/          paths.local.yaml (all absolute paths) · imagenet/*.yaml (task configs)
                  afhq/ · celeba/ · mri/ (sense_acc8 = reference protocol, sense_acc4 = exploratory)
                  models/ours.yaml · eval/metrics.yaml · baselines/baselines.yaml
checkpoints/ours/ model.pt (symlink, 2.7 GB) + config.yaml  ← PixelFlow class-cond model
src/              thin re-exports of the VERIFIED original operators/sampler/metrics
benchmark/        run_ours · generate_measurements · evaluate · visualize · compare_baselines
                  reproduce_previous_results · run_afhq · run_celeba · run_mri
scripts/          run_local_smoke_test.sh · reproduce_previous_results.sh · evaluate_existing_results.sh · run_dataset_smoke.sh
results/          previous_results/ (reference, symlinks) · baselines/ (symlinks)
                  reproduced_results/ · metrics/ · figures/
docs/             manifests, alignment notes, safety check, reproduction report
```

## Quick start

```bash
# 0) use the verified interpreter (see configs/paths.local.yaml: python_bin)
PY=~/anaconda3/envs/pixelflow/bin/python

# 1) smoke test — a few images per task, end-to-end (model→operator→sampler→metrics)
bash scripts/run_local_smoke_test.sh                 # NUM_IMAGES=3, default GPU
NUM_IMAGES=5 GPU=3 bash scripts/run_local_smoke_test.sh

# 2) re-evaluate the existing previous results (no sampling; confirms metrics reproduce)
bash scripts/evaluate_existing_results.sh

# 3) full reproduction (all 100 images/task — slow)
bash scripts/reproduce_previous_results.sh --full --overwrite

# 4) other datasets (AFHQ-512 / CelebA-256 / MRI-384)
bash scripts/run_dataset_smoke.sh                    # afhq+celeba sample 1 img; mri prints reference
$PY benchmark/run_afhq.py   --config configs/afhq/superresolution.yaml   --num_images 1 --output_dir results/reproduced_results/afhq   --gpu 4
$PY benchmark/run_celeba.py --config configs/celeba/superresolution.yaml --num_images 2 --group_id 0 --output_dir results/reproduced_results/celeba --gpu 5
$PY benchmark/run_mri.py    --paths configs/paths.local.yaml --summarize_reference  # reference metrics
$PY benchmark/run_mri.py    --slice_idx 0 --gpu 0                        # re-sample, full reference protocol
                                                                         # (configs/mri/sense_acc8.yaml, 16 samples)
$PY benchmark/run_mri.py    --slice_idx 0 --gpu 0 --nsamples 1           # quick MRI smoke
```

**Reproduction fidelity (verified):** AFHQ-512 SR is **bit-exact** vs reference (batch_size=1,
recon Δ=0). ImageNet/CelebA reproduce **bit-exact at batch parity**; truncated smokes differ
~0.1 dB per image (batch-composition + a non-deterministic CUDA kernel — not a bug). The MRI
prior checkpoint was **restored** after the cluster migration (see
[checkpoints/mri/CHECKPOINT_RESTORED.txt](checkpoints/mri/CHECKPOINT_RESTORED.txt)), so MRI
re-sampling works again; its normalization contract is verified in
[docs/mri_normalization_notes.md](docs/mri_normalization_notes.md) — **the data is already
normalized; do not re-normalize.**

Single task, by hand:

```bash
$PY benchmark/run_ours.py \
  --config configs/imagenet/super_resolution.yaml \
  --model_config configs/models/ours.yaml --paths configs/paths.local.yaml \
  --checkpoint checkpoints/ours/model.pt \
  --num_images 5 --output_dir results/reproduced_results/demo --gpu 2

$PY benchmark/evaluate.py --config configs/eval/metrics.yaml \
  --gt_dir  results/reproduced_results/demo/group100_superresolution_best/gts \
  --result_dir results/reproduced_results/demo/group100_superresolution_best/recons \
  --output_csv results/metrics/demo_sr.csv
```

## How reproduction is guaranteed

`src/` does not reimplement the method — it bootstraps `sys.path` from
`configs/paths.local.yaml` and re-exports the original, validated
`rerun_imageNet/{runner,operators,metrics,seeding}.py` and the
`ms_posterior_sampling_*` sampler. Operators are **DAPS-aligned** (SR antialias,
DAPS motion kernel, DAPS box position); the seed policy is all-seeds-42. So a
`run_ours.py` invocation runs the exact code path that produced the reference
results. See [docs/operator_alignment_notes.md](docs/operator_alignment_notes.md).

## Notes

- DAPS uses an **unconditional** model; ours is **class-conditional + CFG** —
  keep the prior mismatch in mind when comparing methods.
- PSLD/ReSample baselines use an **FFHQ face prior on ImageNet** (OOD) — not
  prior-matched; don't present them naively.
- See [LOCAL_REPRODUCTION.md](LOCAL_REPRODUCTION.md) for the full operator/path/seed
  contract and [docs/local_safety_check.md](docs/local_safety_check.md) for the
  secret/large-file scan.

## License

[Apache-2.0](LICENSE). This project reuses code from
[PixelFlow](https://github.com/ShoufaChen/PixelFlow) (Apache-2.0) and compares
against DAPS / PSLD / ReSample baselines (their own licenses apply; none of their
code is vendored here — see `configs/paths.local.yaml` for how originals are
referenced).
