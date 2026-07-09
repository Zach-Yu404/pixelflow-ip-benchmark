"""Metrics — re-exported from the original rerun_imageNet/metrics.py.

  psnr / ssim  : piq, data_range=1.0
  lpips        : piq.LPIPS(replace_pooling=True)  == the headline "LPIPS_king" ruler
  lpips_alex / lpips_vgg : official lpips package (cross-check rulers)

Same ruler as every previous run -> numbers are directly comparable."""
from src.utils.local_paths import bootstrap as _bootstrap

_bootstrap()

import metrics as _met  # noqa: E402  (original rerun_imageNet/metrics.py)

psnr = _met.psnr
ssim = _met.ssim
lpips = _met.lpips
lpips_alex = _met.lpips_alex   # official AlexNet ruler (cross-check)
lpips_vgg = _met.lpips_vgg     # official VGG ruler (cross-check)

__all__ = ["psnr", "ssim", "lpips", "lpips_alex", "lpips_vgg"]
