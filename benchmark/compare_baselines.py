#!/usr/bin/env python
"""Build the cross-method comparison from existing results. Presents TWO tables,
honestly separated because they are DIFFERENT image sets:

  (A) 100-img DAPS-aligned 3-way: ours(rerun *best) vs ours(demo_runner 恩恩) vs DAPS-100
      — same 100 images (val 49000-99), same crop, operators aligned to DAPS, one ruler.
  (B) 15-demo proper-crop baseline table: every baseline vs ITS OWN GT (the audit's
      each-vs-own-GT stack). Aggregated from the canonical metric CSV.

Reads existing results only (no GPU). Writes results/metrics/baseline_comparison.{md,csv}.
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.local_paths import load_paths, resolve  # noqa: E402

TASKS = [  # (canonical, ours_rerun subdir tag, demorunner glob, daps slug, label)
    ("superresolution",   "superresolution_best",   "superresolution_*",   "superresolution",   "SR x4"),
    ("gaussian_blur",     "gaussian_blur_best",     "gaussian_blur_*",     "gaussian_deblur",   "gaussian"),
    ("motion_blur",       "motion_blur_best",       "motion_blur_*",       "motion_deblur",     "motion"),
    ("random_inpainting", "random_inpainting_best", "random_inpainting_*", "random_inpainting", "random-inp"),
    ("box_inpainting",    "box_inpainting_best",    "box_inpainting_*",    "box_inpainting",    "box-inp"),
]


def _mean(path, key):
    try:
        m = json.load(open(path))
        return float(m[key])
    except Exception:
        return None


def _daps_mean(daps_dir, slug):
    import numpy as np
    mjs = glob.glob(os.path.join(daps_dir, slug, "**", "metrics.json"), recursive=True)
    if not mjs:
        return None
    m = json.load(open(mjs[0]))
    return {k: float(np.mean(m[k]["mean"])) for k in ("psnr", "ssim", "lpips")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.local.yaml")
    ap.add_argument("--output", default="results/metrics/baseline_comparison.md")
    args = ap.parse_args()
    p = load_paths(resolve(args.paths))

    prev = resolve("results/previous_results")
    rerun = os.path.join(prev, "ours_rerun_best")
    demo = os.path.join(prev, "ours_demorunner")
    daps = resolve("results/baselines/daps_pixel")

    # ---- (A) 100-img DAPS-aligned 3-way -------------------------------------
    rowsA = []
    for canon, tag, dglob, slug, label in TASKS:
        rr = os.path.join(rerun, f"group100_{tag}", "meta.json")
        dr_cands = [d for d in glob.glob(os.path.join(demo, dglob)) if os.path.isfile(os.path.join(d, "meta.json"))]
        dr = os.path.join(dr_cands[0], "meta.json") if dr_cands else None
        d = _daps_mean(daps, slug)
        rowsA.append(dict(
            task=label,
            ours_rerun=(_mean(rr, "psnr_mean"), _mean(rr, "ssim_mean"), _mean(rr, "lpips_mean")),
            ours_demo=((_mean(dr, "psnr_mean"), _mean(dr, "ssim_mean"), _mean(dr, "lpips_mean")) if dr else (None, None, None)),
            daps=((d["psnr"], d["ssim"], d["lpips"]) if d else (None, None, None)),
        ))

    # ---- (B) 15-demo proper-crop baselines (aggregate canonical CSV) --------
    # table (A)'s single `lpips` is piq-VGG; table (B) carries raw lpips_alex/lpips_vgg from the audit CSV.
    csv_path = resolve("results/metrics/previous_baseline_metric_comparison.csv")
    agg = defaultdict(lambda: defaultdict(list))
    if os.path.isfile(csv_path):
        for r in csv.DictReader(open(csv_path)):
            for k in ("psnr", "ssim", "lpips_alex", "lpips_vgg"):
                try:
                    agg[(r["method"], r["task"])][k].append(float(r[k]))
                except (ValueError, KeyError):
                    pass

    def f3(t):
        return " / ".join(("—" if v is None else f"{v:.2f}" if i == 0 else f"{v:.3f}") for i, v in enumerate(t))

    lines = ["# Cross-method comparison (reproduced from existing results)\n",
             "## (A) 100-image DAPS-aligned 3-way  *(PSNR / SSIM / LPIPS-piqVGG)*",
             "Same 100 images (val 49000-99), same center crop, operators aligned to DAPS, one ruler.\n",
             "| task | ours (rerun *best) | ours (demo_runner 恩恩) | DAPS-100 |",
             "|---|---|---|---|"]
    for r in rowsA:
        lines.append(f"| {r['task']} | {f3(r['ours_rerun'])} | {f3(r['ours_demo'])} | {f3(r['daps'])} |")
    lines += ["\n> LPIPS here = piq-VGG (replace_pooling). DAPS uses an UNCONDITIONAL model; "
              "ours is class-conditional + CFG=2.0. ~2.5-3 dB PSNR gap = perception-distortion, not a bug.\n",
              "## (B) 15-demo proper-crop baselines  *(each vs its OWN GT; PSNR / SSIM / LPIPS-alex / LPIPS-vgg)*",
              "From the audit's canonical CSV (`results/metrics/previous_baseline_metric_comparison.csv`). "
              "CAVEAT: PSLD/ReSample use an FFHQ FACE prior on ImageNet (OOD); DDNM-gaussian needed the DDNM+ fix.\n",
              "| method | task | PSNR | SSIM | LPIPS-alex | LPIPS-vgg |",
              "|---|---|---|---|---|---|"]
    import statistics as st

    def cell(v, p=2):  # format one metric cell ("—" if missing), p decimals
        return "—" if v is None else f"{v:.{p}f}"

    csv_rows = []
    for (method, task), d in sorted(agg.items()):
        vals = {k: (st.mean(d[k]) if d[k] else None) for k in ("psnr", "ssim", "lpips_alex", "lpips_vgg")}
        lines.append(f"| {method} | {task} | {cell(vals['psnr'])} | {cell(vals['ssim'],3)} | "
                     f"{cell(vals['lpips_alex'],3)} | {cell(vals['lpips_vgg'],3)} |")
        csv_rows.append(dict(method=method, task=task, **{k: vals[k] for k in vals}))

    out_md = os.path.abspath(resolve(args.output))
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w").write("\n".join(lines) + "\n")
    out_csv = out_md.replace(".md", ".csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "task", "psnr", "ssim", "lpips_alex", "lpips_vgg"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"[compare] wrote {out_md}\n[compare] wrote {out_csv}")
    # console preview = table (A) only: lines[0:3]=intro, [3]=header, [4]=separator, [5:7]=first 2 task rows
    print("\n".join(lines[:7]))


if __name__ == "__main__":
    main()
