#!/usr/bin/env python
"""Run OUR method (PixelFlow posterior sampler) on a task.

Reproduces results/previous_results/ours_rerun_best/group100_<task>_best/ by
building a runner config from a task YAML + paths.local.yaml and calling the
VERIFIED original rerun_imageNet/runner.main (no reimplementation).

Example (smoke, 5 imgs):
  benchmark/run_ours.py --config configs/imagenet/super_resolution.yaml \
      --model_config configs/models/ours.yaml --paths configs/paths.local.yaml \
      --checkpoint checkpoints/ours/model.pt --num_images 5 \
      --output_dir results/reproduced_results/smoke --gpu 2

Output: <output_dir>/group<gid>_<validation_tag>/{recons/, gts/, grid.png, meta.json}
"""
import argparse
import os
import shutil
import sys

# project root on path so `import src...` works regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import yaml  # noqa: E402
from src.utils.local_paths import bootstrap, resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Run our PixelFlow posterior sampler on one task.")
    ap.add_argument("--config", required=True, help="task YAML, e.g. configs/imagenet/super_resolution.yaml")
    ap.add_argument("--model_config", default="configs/models/ours.yaml",
                    help="documentation-only; model is resolved from --checkpoint or paths.model_dir (kept for CLI parity)")
    ap.add_argument("--paths", default="configs/paths.local.yaml")
    ap.add_argument("--checkpoint", default=None, help="path to model.pt (its dir must hold config.yaml)")
    ap.add_argument("--num_images", type=int, default=5, help="N images; <=0 or >=100 means full 100-image set")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--group_id", type=int, default=None, help="override group (default: task YAML / paths daps_group_id)")
    ap.add_argument("--overwrite", action="store_true", help="delete an existing output subdir and rerun")
    args = ap.parse_args()

    paths = bootstrap(resolve(args.paths))
    import runner  # noqa: E402  (rerun_imageNet/runner.py — now importable)

    task_cfg = yaml.safe_load(open(resolve(args.config)))

    # --- resolve checkpoint / model_dir -------------------------------------
    if args.checkpoint:
        ckpt = resolve(args.checkpoint)
        model_dir = os.path.dirname(ckpt)
        model_ckpt = os.path.basename(ckpt)
    else:
        model_dir = paths["model_dir"]
        model_ckpt = "model.pt"
    assert os.path.isfile(os.path.join(model_dir, "config.yaml")), f"missing config.yaml in {model_dir}"
    assert os.path.isfile(os.path.join(model_dir, model_ckpt)), f"missing {model_ckpt} in {model_dir}"

    # --- group + smoke truncation -------------------------------------------
    group_id = args.group_id if args.group_id is not None else int(
        task_cfg.get("group_id", paths.get("daps_group_id", 100)))
    n = args.num_images
    max_n = 0 if (n is None or n <= 0 or n >= 100) else int(n)   # 0 => full group, resume-guarded
    gpu = args.gpu if args.gpu is not None else int(paths.get("default_gpu", 0))
    out_root = os.path.abspath(resolve(args.output_dir))
    tag = str(task_cfg.get("validation_tag", task_cfg["task"] + "_best"))

    cfg = dict(
        task=task_cfg["task"],
        operator=task_cfg["operator"],
        kw=task_cfg["kw"],
        sigma_n=float(task_cfg.get("sigma_n", 0.05)),
        resolution=int(task_cfg.get("resolution", 256)),
        batch_size=int(task_cfg.get("batch_size", paths.get("batch_size", 8))),
        group_id=group_id,
        validation_tag=tag,
        groups_dir=os.path.abspath(paths["groups_dir"]),   # absolute -> overrides HERE-relative default
        output_dir=out_root,                               # absolute -> recons land here
        model_dir=model_dir,
        model_ckpt=model_ckpt,
        device=f"cuda:{gpu}",
        max_n=max_n,
    )

    target = os.path.join(out_root, f"group{group_id}_{tag}")
    if args.overwrite and os.path.isdir(target):
        print(f"[run_ours] --overwrite: removing {target}", flush=True)
        shutil.rmtree(target)
    elif os.path.isfile(os.path.join(target, "meta.json")):
        print(f"[run_ours] SKIP (already done): {target}  (use --overwrite to rerun)", flush=True)
        print(f"OUTPUT_DIR={target}")
        return

    print(f"[run_ours] task={cfg['task']} group={group_id} N={'full(100)' if max_n==0 else max_n} "
          f"gpu={gpu} -> {target}", flush=True)
    runner.main(resolve(args.config), cfg=cfg)
    print(f"OUTPUT_DIR={target}")


if __name__ == "__main__":
    main()
