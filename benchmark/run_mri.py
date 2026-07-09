#!/usr/bin/env python
"""MRI-384 SENSE reconstruction.

The MRI flow prior checkpoint is RESTORED (see checkpoints/mri/CHECKPOINT_RESTORED.txt;
configs/paths.local.yaml: mri_prior_ckpt). Two modes:

  (default, prior available) : subprocess the VERIFIED val_final_8/run.py, which
            uses the operator + normalization contract in docs/mri_normalization_notes.md
            (GT/csm are ALREADY normalized — it does NOT re-normalize; scale is on y only).
  --summarize_reference      : read the reference val_final_8/slice*/meta.json
            and print the recorded per-sample + MMSE metrics (PSNR/SSIM/LPIPS).

Task configs live in configs/mri/*.yaml (default: sense_acc8.yaml = the exact
val_final_8 reference protocol: acc8 mask, winning sweep tag, nsamples 16).
The YAML is translated to the verified run.py CLI: validation_tag -> --config,
operator.mask (a paths.local.yaml key) -> --mask_path, group/nsamples likewise.

Example:
  benchmark/run_mri.py --paths configs/paths.local.yaml --summarize_reference  # reference summary
  benchmark/run_mri.py --slice_idx 0 --gpu 0            # full reference protocol (16 samples, GPU)
  benchmark/run_mri.py --slice_idx 0 --gpu 0 --nsamples 1                      # quick smoke
  benchmark/run_mri.py --config configs/mri/sense_acc4.yaml --slice_idx 0      # acc4 (exploratory)
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
from src.utils.local_paths import load_paths, resolve


def summarize_reference(p):
    ref = p["mri_reference_runs"]
    slices = sorted(glob.glob(os.path.join(ref, "slice*")))
    if not slices:
        print(f"[mri] no reference slices under {ref}"); return
    print(f"[mri] reference reconstructions: {ref}  (acc8, res 384)")
    print(f"{'slice':>6} | {'config':<32} | {'persample PSNR/SSIM/LPIPS':>28} | {'MMSE PSNR/SSIM/LPIPS':>24}")
    for s in slices:
        mp = os.path.join(s, "meta.json")
        if not os.path.isfile(mp): continue
        m = json.load(open(mp))
        ps = m.get("per_sample_mean", {}); mm = m.get("mmse", {})
        print(f"{os.path.basename(s):>6} | {m.get('config','?'):<32} | "
              f"{ps.get('psnr',0):.2f}/{ps.get('ssim',0):.3f}/{ps.get('lpips',0):.3f}{'':>10} | "
              f"{mm.get('psnr',0):.2f}/{mm.get('ssim',0):.3f}/{mm.get('lpips',0):.3f}")
    print(f"[mri] nsamples per slice = {m.get('nsamples','?')}; scale (per-slice, applied to y) e.g. {m.get('scale','?'):.1f}")
    print("[mri] normalization: GT |x|max=1 & csm unit-power are PRE-normalized — not re-normalized. "
          "See docs/mri_normalization_notes.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.local.yaml")
    ap.add_argument("--config", default="configs/mri/sense_acc8.yaml",
                    help="task YAML (configs/mri/*.yaml); sense_acc8 = reference protocol")
    ap.add_argument("--summarize_reference", action="store_true")
    ap.add_argument("--ckpt", default=None, help="MRI prior dir (config.yaml + model_epoch40.pt) to RE-SAMPLE")
    ap.add_argument("--slice_idx", type=int, default=0)
    ap.add_argument("--nsamples", type=int, default=None,
                    help="posterior samples per slice (default: config value; reference protocol = 16)")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--output_dir", default="results/reproduced_results/mri")
    args = ap.parse_args()
    p = load_paths(resolve(args.paths))

    cfg_path = resolve(args.config)
    if not os.path.isfile(cfg_path):
        sys.exit(f"[mri] task config not found: {cfg_path}")
    cfg = yaml.safe_load(open(cfg_path))
    nsamples = args.nsamples if args.nsamples is not None else int(cfg.get("nsamples", 1))
    # operator.mask is a KEY into paths.local.yaml (e.g. mri_mask_acc8) or a literal path
    mask_ref = (cfg.get("operator") or {}).get("mask")
    mask_path = p.get(mask_ref, mask_ref) if mask_ref else None
    accel = (cfg.get("operator") or {}).get("acceleration")

    prior = args.ckpt or (None if str(p.get("mri_prior_ckpt", "")) == "__MISSING__" else p.get("mri_prior_ckpt"))
    if args.summarize_reference:
        prior = None
    if prior and os.path.isdir(prior) and os.path.isfile(os.path.join(prior, "config.yaml")):
        gpu = args.gpu if args.gpu is not None else int(p.get("default_gpu", 0))
        out = os.path.abspath(resolve(args.output_dir))
        if accel:  # keep acc8/acc4 outputs apart
            out = os.path.join(out, str(accel))
        os.makedirs(out, exist_ok=True)
        cmd = [p.get("python_bin", sys.executable), p["mri_runner"],
               "--slice_idx", str(args.slice_idx), "--gpu", str(gpu),
               "--nsamples", str(nsamples), "--ckpt", prior, "--out_dir", out]
        if cfg.get("validation_tag"):
            cmd += ["--config", str(cfg["validation_tag"])]
        if mask_path:
            cmd += ["--mask_path", mask_path]
        if cfg.get("group") is not None:
            cmd += ["--group", str(cfg["group"])]
        print(f"[mri] RE-SAMPLING via verified run.py (prior restored): {' '.join(cmd)}", flush=True)
        sys.exit(subprocess.run(cmd).returncode)

    # reference-only path
    print("=" * 78)
    if args.summarize_reference:
        print("[mri] --summarize_reference: showing the recorded reference metrics.")
    else:
        print("[mri] MRI prior checkpoint not found -> cannot re-sample.")
        print("      Check configs/paths.local.yaml: mri_prior_ckpt (dir with config.yaml")
        print("      + model_epoch40.pt) or pass --ckpt <prior_dir> explicitly.")
        print("      Showing the REFERENCE reconstruction metrics instead.")
    print("=" * 78)
    summarize_reference(p)


if __name__ == "__main__":
    main()
