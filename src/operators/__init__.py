"""Forward operators — re-exported from the original rerun_imageNet/operators.py,
which is the DAPS-ALIGNED implementation:

  * SROperator(antialias=True)  -> matches DAPS antialiased Resizer family (|Δ|≈0.0002)
  * GaussianBlurOperator        -> identical to DAPS Blurkernel (|Δ|=0.0)
  * MotionBlurOperator(use_daps_kernel=True) injects DAPS forward_operator.motionblur.Kernel
    (DAPS repo: configs/paths.local.yaml: daps_repo); verified max|Δ|=0.0
  * inpainting via get_operator("inpainting", ...) — random 70% missing / box random_sq_bbox

We REUSE this file rather than fork it so operator behavior cannot silently drift
from what produced the previous results. See docs/operator_alignment_notes.md.
"""
from src.utils.local_paths import bootstrap as _bootstrap

_bootstrap()

import operators as _ops          # noqa: E402  (original rerun_imageNet/operators.py)
from operators import *           # noqa: E402,F401,F403

# Explicit handles for the common symbols (so `from src.operators import SROperator` works).
SROperator = _ops.SROperator
GaussianBlurOperator = _ops.GaussianBlurOperator
MotionBlurOperator = _ops.MotionBlurOperator
get_operator = _ops.get_operator
