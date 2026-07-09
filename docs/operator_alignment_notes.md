# Operator alignment notes (vs DAPS)

The forward operators in this project are **aligned pixel-for-pixel to DAPS**
(Decoupled Annealing Posterior Sampling), so the ours-vs-DAPS comparison isolates
the *method*, not operator mismatch. The implementation is the original
`rerun_imageNet/operators.py`, re-exported by `src/operators` — it **references the
DAPS repo directly** for the motion kernel (`configs/paths.local.yaml: daps_repo`).

| task | our operator | DAPS operator | alignment | max&#124;Δ&#124; |
|---|---|---|---|---|
| SR ×4 | `F.interpolate(bicubic, antialias=True)` | antialiased `Resizer` | `operator.antialias: true` | **≈0.0002** |
| gaussian | `GaussianBlurOperator` (Blurkernel, ReflectionPad) | same Gaussian Blurkernel | identical | **0.00000** |
| motion | `MotionBlurOperator(use_daps_kernel=True)` | `motionblur.Kernel` | **inject DAPS kernel** (seed=42) | **0.0** |
| box-inp | `random_sq_bbox` position | DAPS `random_sq_bbox` | same position, noiseless | exact |
| random-inp | 70% **missing** | DAPS convention (70% missing) | matched (after fixing inverted mask) | exact |

## How the DAPS motion kernel is injected
`MotionBlurOperator(..., use_daps_kernel=True)` calls a module-level helper that does
`from forward_operator.motionblur.motionblur import Kernel` (with `daps_repo` on
`sys.path`), `np.random.seed(kernel_seed=42)`, builds `Kernel(size, intensity).kernelMatrix`,
and overwrites the conv weights (`self._op.conv.update_weights(K)`). Verified `max|Δ|=0.0`
against DAPS's own kernel. Pure Gaussian-line fallback (`use_daps_kernel=False`) is the
non-aligned legacy path — **not** used by the `*best` configs.

## Two historical defects this fixed (see FINAL_REPORT §1)
- **SR**: plain bicubic (ours, legacy) vs antialiased Resizer (DAPS) differed by 0.044 →
  inflated the SR gap. `antialias=true` closes the operator mismatch to 0.0002.
- **random inpainting**: our demo mask had been **inverted** (70% *observed*, noiseless),
  inflating the score. Corrected to 70% missing (DAPS convention).

## DAPS reference fidelity
Our DAPS-100 reference reproduces the DAPS paper (Table 3) within tolerance on
every task — confirming both the exact image set and a faithful DAPS run. The
per-task comparison lives with the results on the reproduction machine.

> DAPS uses an **unconditional** model; ours is class-conditional + CFG. Operator
> alignment is exact regardless; the model conditioning asymmetry is documented in
> the (local) report.
