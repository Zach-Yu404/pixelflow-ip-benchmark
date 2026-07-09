#!/usr/bin/env python
"""Evaluate recons against GTs with the ONE canonical metric stack (piq PSNR/SSIM +
piq-VGG LPIPS == the headline ruler). Same functions runner.py used, so numbers match.

Example:
  benchmark/evaluate.py --config configs/eval/metrics.yaml \
      --gt_dir results/reproduced_results/smoke/group100_superresolution_best/gts \
      --result_dir results/reproduced_results/smoke/group100_superresolution_best/recons \
      --output_csv results/metrics/smoke_super_resolution.csv
"""
import argparse
import csv
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.local_paths import bootstrap, resolve  # noqa: E402


def _imgs(d):
    return sorted(glob.glob(os.path.join(d, "*.png")))


EVAL_RES = 256  # metrics are computed at 256x256; non-256 recons are resized to match


def _t01(path, device):
    import numpy as np
    import torch
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), dtype="float32") / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)
    if t.shape[-1] != EVAL_RES:
        import torch.nn.functional as F
        t = F.interpolate(t, size=(EVAL_RES, EVAL_RES), mode="bilinear", align_corners=False)
    return t


def main():
    ap = argparse.ArgumentParser(description="Evaluate recons vs GTs (canonical metric stack).")
    ap.add_argument("--config", default="configs/eval/metrics.yaml",
                    help="documentation-only; the metric stack is fixed in src/metrics (kept for CLI parity)")
    ap.add_argument("--paths", default="configs/paths.local.yaml")
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--result_dir", required=True, help="dir of recon pngs (or a run dir containing recons/)")
    ap.add_argument("--measurement_dir", default=None, help="optional; recorded in summary only")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    paths = bootstrap(resolve(args.paths))
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from src import metrics as M  # noqa: E402

    gpu = args.gpu if args.gpu is not None else int(paths.get("default_gpu", 0))
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"

    rdir = resolve(args.result_dir)
    if os.path.isdir(os.path.join(rdir, "recons")):
        rdir = os.path.join(rdir, "recons")
    gdir = resolve(args.gt_dir)
    recons, gts = _imgs(rdir), _imgs(gdir)
    n = min(len(recons), len(gts))
    assert n > 0, f"no paired images: recons={len(recons)} in {rdir}, gts={len(gts)} in {gdir}"

    rows = []
    for i in range(n):
        r, g = _t01(recons[i], device), _t01(gts[i], device)
        with torch.no_grad():
            psnr_v = float(M.psnr(r, g))
            ssim_v = float(M.ssim(r, g))
            lpips_v = float(M.lpips(r, g, device))
        rows.append(dict(idx=i, recon=os.path.basename(recons[i]), psnr=psnr_v, ssim=ssim_v, lpips=lpips_v))
        print(f"[eval] {i:3d} PSNR={psnr_v:.2f} SSIM={ssim_v:.3f} LPIPS={lpips_v:.3f}", flush=True)

    means = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "lpips")}
    out_csv = os.path.abspath(resolve(args.output_csv))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "recon", "psnr", "ssim", "lpips"])
        w.writeheader()
        w.writerows(rows)
        w.writerow(dict(idx="MEAN", recon=f"n={n}", **{k: round(v, 4) for k, v in means.items()}))

    summary = dict(n=n, result_dir=rdir, gt_dir=gdir, measurement_dir=args.measurement_dir,
                   ruler="piq psnr/ssim (data_range=1) + piq-VGG LPIPS (replace_pooling)",
                   means=means)
    json.dump(summary, open(out_csv.replace(".csv", "_summary.json"), "w"), indent=2)
    print(f"\n[eval] MEAN PSNR={means['psnr']:.3f} SSIM={means['ssim']:.4f} LPIPS={means['lpips']:.4f}  (n={n})")
    print(f"[eval] wrote {out_csv}")


if __name__ == "__main__":
    main()
