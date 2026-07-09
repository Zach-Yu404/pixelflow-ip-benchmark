#!/usr/bin/env python
"""Run OUR method on AFHQ-512 (cat/dog/wild PixelFlow prior).

Faithful to the demo_runner pipeline that produced demo_runs_afhq/: reuses
demo_runner.{load_demo_images, build_setup_and_measurement, to_img01} + the same
run_posterior_sampling, with the AFHQ model (res 512, num_classes 3) loaded from
configs/paths.local.yaml. Per-image, seed=42 — identical to running demo_runner.py
once per image, just batched-load.

Example (smoke, 1 img):
  benchmark/run_afhq.py --config configs/afhq/superresolution.yaml \
     --paths configs/paths.local.yaml --num_images 1 \
     --output_dir results/reproduced_results/afhq_smoke --gpu 2
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
import yaml
from src.utils.local_paths import bootstrap, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--paths", default="configs/paths.local.yaml")
    ap.add_argument("--num_images", type=int, default=1)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = bootstrap(resolve(args.paths))
    import numpy as np, torch, piq
    from omegaconf import OmegaConf
    from PIL import Image
    from pixelflow.utils import config as config_utils
    from ms_posterior_sampling_article_version_final import run_posterior_sampling
    import demo_runner as DR

    tc = yaml.safe_load(open(resolve(args.config)))
    task, op_cfg, kw = tc["task"], tc["operator"], dict(tc["kw"])
    tag = tc.get("validation_tag", "LPIPS_king")
    res = int(tc.get("resolution", paths.get("afhq_resolution", 512)))
    sigma_n = float(tc.get("sigma_n", 0.05))
    gpu = args.gpu if args.gpu is not None else int(paths.get("default_gpu", 0))
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    SEED = int(args.seed)

    model_dir = paths["afhq_model_dir"]; model_ckpt = paths["afhq_model_ckpt"]
    demo_dir = paths["afhq_demo_dir"]
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model = config_utils.instantiate_from_config(config.model).to(device)
    ck = torch.load(os.path.join(model_dir, model_ckpt), map_location="cpu", weights_only=False)
    sd = ck["ema"] if isinstance(ck, dict) and "ema" in ck else (ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    model.load_state_dict(sd, strict=True); model.eval()
    lpips_fn = piq.LPIPS(replace_pooling=True).to(device)

    demos = DR.load_demo_images(resolution=res, demo_dir=demo_dir)
    N = min(args.num_images, len(demos)) if args.num_images > 0 else len(demos)
    out = os.path.join(os.path.abspath(resolve(args.output_dir)), f"{task}_{tag}")
    os.makedirs(os.path.join(out, "recons"), exist_ok=True); os.makedirs(os.path.join(out, "gts"), exist_ok=True)

    per_image = []; t0 = time.time()
    for i in range(N):
        demo = demos[i]
        operator, mask, y, meas_label, meas_panel, make_Ak, term = \
            DR.build_setup_and_measurement(task, op_cfg, demo, sigma_n, res, device)
        gt = demo["gt"].unsqueeze(0).to(device)
        torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
        xf = run_posterior_sampling(model, config, gt, y, operator, sigma_n, device,
                                    class_label=[demo["class_idx"]], seed=SEED,
                                    make_Ak_fns_fn=make_Ak, terminal_replacement_fn=term,
                                    record_trajectory=False, **kw).to(device)
        r01, g01 = DR.to_img01(xf), DR.to_img01(gt)
        psnr_v = piq.psnr(r01, g01, data_range=1.0).item(); ssim_v = piq.ssim(r01, g01, data_range=1.0).item()
        lpips_v = lpips_fn(r01, g01).item()
        Image.fromarray((r01[0].permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)).save(os.path.join(out, "recons", f"{i:03d}.png"))
        Image.fromarray((g01[0].permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)).save(os.path.join(out, "gts", f"{i:03d}.png"))
        per_image.append(dict(idx=i, short=demo["short_name"], psnr=psnr_v, ssim=ssim_v, lpips=lpips_v))
        print(f"[afhq {i:3d}] {task} {demo['short_name']} PSNR={psnr_v:.2f} SSIM={ssim_v:.3f} LPIPS={lpips_v:.3f}", flush=True)

    meta = dict(dataset="afhq512", task=task, validation_tag=tag, num_images=N, resolution=res, sigma_n=sigma_n,
                kw=kw, operator=op_cfg, psnr_mean=float(np.mean([m["psnr"] for m in per_image])),
                ssim_mean=float(np.mean([m["ssim"] for m in per_image])), lpips_mean=float(np.mean([m["lpips"] for m in per_image])),
                time_total_s=time.time()-t0, per_image=per_image)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2, default=str)
    print(f"[afhq done] {task}/{tag} PSNR={meta['psnr_mean']:.3f} LPIPS={meta['lpips_mean']:.4f} n={N} -> {out}")
    print(f"OUTPUT_DIR={out}")


if __name__ == "__main__":
    main()
