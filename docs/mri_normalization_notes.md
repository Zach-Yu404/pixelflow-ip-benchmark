# MRI normalization — verified contract (DO NOT double-normalize)

**Question raised:** the MRI data at `test_data_n_mask/subset_test_data384/` may already be
normalized — confirm so we don't normalize it again.

**Answer: YES, the data is already normalized. Do not re-normalize the GT or the coil maps.**
Empirically verified by loading `group_0/processed_file_brain_AXT2_200_2000021_slice0.pt`:

| tensor | key | dtype | finding |
|---|---|---|---|
| GT image | `x` | complex64 `(1,1,384,384)` | **`|x|` max = 1.000, mean = 0.102**; real,imag ∈ ~[-1,1] → **peak-normalized to 1** |
| coil maps | `csm` | complex64 `(1,16,384,384)` | **unit-power**: `(|csm|²).sum(coil) ≈ 1` on support |
| k-space meas. | `b_hat` | complex64 `(1,16,384,384)` | tiny scale (`|·|max≈0.005`) — raw measurement scale |

## The only rescaling that happens (and it's on the MEASUREMENT, not the GT)
`val_final_8/run.py` (the verified runner):
```python
x_gt = d["x"]                      # already |x|max=1  -> fed to the prior AS-IS
pred_full = op_full.forward(x_gt)  # clean k-space from the normalized GT
scale = pred_full.abs().mean() / (y_full.abs().mean() + 1e-12)   # ~1385 per slice
y_ksp = (y_full * scale) * mask    # bring the measurement up to the GT-forward scale
gt_real = operator.complex_to_real(x_gt)   # [real,imag] at the normalized scale
... run_posterior_sampling(model, config, gt_real, y_ksp, operator, SIGMA_N, ...)
```
- `scale` (per-slice, e.g. `1385.05` for slice0; ranges ~1302–1385 across slices) multiplies **`y` (the measurement)**, never the GT.
- The GT `x` and `csm` go into the prior/operator **unchanged** — because the MRI flow prior
  was trained on this exact normalized scale (`|x|max=1`, csm unit-power).
- `mri_operator.py` docstring is explicit: *"csm is assumed unit-power normalized … Do NOT
  re-normalize csm internally."*

## What this means for reproduction
- **Reuse `run.py` / `ip4_mri` / `mri_operator` verbatim** — they encode the correct contract.
  Our integration does exactly this; it adds **no** normalization of its own.
- **Pitfall to avoid:** if you (or a new operator) divide `x` by `|x|.max()` again, or
  re-normalize `csm`, you double-normalize → wrong data-fidelity weighting → corrupted recon
  and meaningless PSNR. **Don't.**
- Metrics are magnitude-domain with `data_range = gt_mag.max()` (run.py `psnr_mag`), so they are
  invariant to a global GT rescale — but the **prior** is not, which is why the as-stored
  normalized scale must be preserved.

## Status
MRI is **reference-only** here (the prior checkpoint is missing — see
`checkpoints/mri/CHECKPOINT_MISSING.txt`). This contract is documented so that, once the prior
is restored, re-sampling is correct on the first try.
