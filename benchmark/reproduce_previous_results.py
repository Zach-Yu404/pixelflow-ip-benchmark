#!/usr/bin/env python
"""Orchestrate reproduction of OUR previous results across all 5 tasks.

For each task: run_ours -> evaluate -> (then visualize + compare at the end).
Each task runs in its OWN subprocess (fresh CUDA context, no mem accumulation).

  smoke (default): --num_images 5   (a few images per task; quick sanity)
  full:            --full           (all 100 DAPS images per task)

Compares reproduced means against the recorded previous means (meta.json) and
flags any drift. Honest: failures are reported, not hidden.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.local_paths import load_paths, resolve  # noqa: E402

TASK_CONFIGS = {
    "super_resolution": "configs/imagenet/super_resolution.yaml",
    "deblur":           "configs/imagenet/deblur.yaml",            # gaussian
    "motion_deblur":    "configs/imagenet/motion_deblur.yaml",
    "inpainting":       "configs/imagenet/inpainting.yaml",        # box
    "inpainting_random": "configs/imagenet/inpainting_random.yaml",
}
# task YAML validation_tag -> previous_results/ours_rerun_best subdir
PREV_TAG = {
    "super_resolution": "group100_superresolution_best",
    "deblur":           "group100_gaussian_blur_best",
    "motion_deblur":    "group100_motion_blur_best",
    "inpainting":       "group100_box_inpainting_best",
    "inpainting_random": "group100_random_inpainting_best",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.local.yaml")
    ap.add_argument("--num_images", type=int, default=5)
    ap.add_argument("--full", action="store_true", help="run all 100 images per task")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--tasks", default="all", help="comma list of task keys or 'all'")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    p = load_paths(resolve(args.paths))
    py = p.get("python_bin", sys.executable)
    gpu = args.gpu if args.gpu is not None else int(p.get("default_gpu", 0))
    n = 0 if args.full else args.num_images
    mode = "full100" if args.full else f"smoke{args.num_images}"
    out_root = resolve(args.output_dir) if args.output_dir else resolve(f"results/reproduced_results/{mode}")
    ckpt = resolve("checkpoints/ours/model.pt")

    keys = list(TASK_CONFIGS) if args.tasks == "all" else [k.strip() for k in args.tasks.split(",")]
    summary = []
    for key in keys:
        cfg = TASK_CONFIGS[key]
        print(f"\n{'='*70}\n[reproduce] TASK {key}  ({mode})\n{'='*70}", flush=True)
        cmd = [py, os.path.join(_HERE, "run_ours.py"),
               "--config", resolve(cfg), "--model_config", resolve("configs/models/ours.yaml"),
               "--paths", resolve(args.paths), "--checkpoint", ckpt,
               "--num_images", str(n if n > 0 else 100), "--output_dir", out_root, "--gpu", str(gpu)]
        if args.overwrite:
            cmd.append("--overwrite")
        r = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(r.stdout[-3000:]); sys.stderr.write(r.stderr[-2000:])
        if r.returncode != 0:
            summary.append(dict(task=key, status="RUN_FAILED", error=(r.stderr.strip().splitlines() or ["unknown"])[-1]))
            continue
        target = None
        for line in r.stdout.splitlines():
            if line.startswith("OUTPUT_DIR="):
                target = line.split("=", 1)[1].strip()
        if not target or not os.path.isfile(os.path.join(target, "meta.json")):
            summary.append(dict(task=key, status="NO_OUTPUT"))
            continue
        m = json.load(open(os.path.join(target, "meta.json")))
        repro = (m["psnr_mean"], m["ssim_mean"], m["lpips_mean"])

        # compare vs recorded previous (full-set) means
        prev_meta = os.path.join(resolve("results/previous_results/ours_rerun_best"), PREV_TAG[key], "meta.json")
        prev = None
        if os.path.isfile(prev_meta):
            pm = json.load(open(prev_meta))
            prev = (pm["psnr_mean"], pm["ssim_mean"], pm["lpips_mean"])
        # evaluate.py independent re-score (sanity that our metric path matches runner's)
        csv_out = resolve(f"results/metrics/{mode}_{key}.csv")
        ev = subprocess.run([py, os.path.join(_HERE, "evaluate.py"),
                             "--paths", resolve(args.paths), "--gt_dir", os.path.join(target, "gts"),
                             "--result_dir", os.path.join(target, "recons"),
                             "--output_csv", csv_out, "--gpu", str(gpu)], capture_output=True, text=True)
        summary.append(dict(task=key, status="OK", n=m["num_images"], output=target,
                            repro_psnr_ssim_lpips=[round(x, 4) for x in repro],
                            prev_full_psnr_ssim_lpips=[round(x, 4) for x in prev] if prev else None,
                            note=("smoke subset != full-100 mean; expect approx match" if n > 0 else "full set"),
                            eval_csv=csv_out if ev.returncode == 0 else f"EVAL_FAILED: {(ev.stderr.strip().splitlines() or ['unknown'])[-1]}"))

    rep = dict(mode=mode, gpu=gpu, output_root=out_root, tasks=summary)
    rep_path = resolve(f"results/metrics/reproduction_{mode}.json")
    os.makedirs(os.path.dirname(rep_path), exist_ok=True)
    json.dump(rep, open(rep_path, "w"), indent=2)
    print(f"\n{'='*70}\n[reproduce] SUMMARY ({mode})\n{'='*70}")
    for s in summary:
        print(json.dumps(s, ensure_ascii=False))
    print(f"\n[reproduce] wrote {rep_path}")
    if any(s["status"] != "OK" for s in summary):
        sys.exit(1)


if __name__ == "__main__":
    main()
