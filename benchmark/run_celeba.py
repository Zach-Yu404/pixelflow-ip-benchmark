#!/usr/bin/env python
"""Run OUR method on CelebA-256 (PixelFlow exp_155024 prior).

Reuses the VERIFIED celeba_results/code/evaluate.py (which loads the precomputed
CelebA .pt groups + the exp_155024 model). This wrapper just assembles a JSON
config from the local YAML + paths.local.yaml (so model/data/output paths stay
centralized) and subprocesses evaluate.py for one (task, group).

Example (smoke, 2 imgs):
  benchmark/run_celeba.py --config configs/celeba/superresolution.yaml \
     --paths configs/paths.local.yaml --num_images 2 --group_id 0 \
     --output_dir results/reproduced_results/celeba_smoke --gpu 2
"""
import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
import yaml
from src.utils.local_paths import load_paths, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--paths", default="configs/paths.local.yaml")
    ap.add_argument("--num_images", type=int, default=2, help="max_n; 0 = full group (100)")
    ap.add_argument("--group_id", type=int, default=0)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    p = load_paths(resolve(args.paths))
    tc = yaml.safe_load(open(resolve(args.config)))
    gpu = args.gpu if args.gpu is not None else int(p.get("default_gpu", 0))
    out_root = os.path.abspath(resolve(args.output_dir))
    os.makedirs(out_root, exist_ok=True)

    cfg = dict(
        task=tc["task"], validation_tag=tc.get("validation_tag", "LPIPS_king"),
        operator=tc["operator"], kw=tc["kw"],
        sigma_n=float(tc.get("sigma_n", 0.05)), resolution=int(tc.get("resolution", 256)),
        batch_size=int(tc.get("batch_size", 8)),
        model_dir=p["celeba_model_dir"], model_ckpt=p["celeba_model_ckpt"],
        groups_dir=p["celeba_groups_dir"], output_dir=out_root, device=f"cuda:{gpu}",
    )
    assert os.path.isfile(os.path.join(cfg["model_dir"], "config.yaml")), f"no config.yaml in {cfg['model_dir']}"
    assert os.path.isfile(os.path.join(cfg["model_dir"], cfg["model_ckpt"])), f"no {cfg['model_ckpt']}"
    gpt = os.path.join(cfg["groups_dir"], cfg["task"], f"group_{args.group_id}.pt")
    assert os.path.isfile(gpt), f"missing CelebA group: {gpt}"

    tmp_json = os.path.join(out_root, f"_cfg_{cfg['task']}_{cfg['validation_tag']}_g{args.group_id}.json")
    json.dump(cfg, open(tmp_json, "w"), indent=2)

    evaluate = os.path.join(p["celeba_code_dir"], "evaluate.py")
    py = p.get("python_bin", sys.executable)
    cmd = [py, evaluate, "--config", tmp_json, "--group_id", str(args.group_id),
           "--device", f"cuda:{gpu}", "--max_n", str(args.num_images), "--batch_size", str(cfg["batch_size"])]
    print(f"[run_celeba] {cfg['task']}/{cfg['validation_tag']} group={args.group_id} "
          f"max_n={args.num_images} gpu={gpu}\n  -> {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    out_dir = os.path.join(out_root, cfg["task"], cfg["validation_tag"], f"group_{args.group_id}")
    print(f"OUTPUT_DIR={out_dir}")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
