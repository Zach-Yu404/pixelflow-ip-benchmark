#!/usr/bin/env python
"""Visualize metric CSV(s) produced by evaluate.py: grouped bar chart of mean
PSNR / SSIM / LPIPS, plus a per-image strip. Accepts one or more CSVs (comma-sep)
to compare configs/tasks side by side.

Example:
  benchmark/visualize.py --metrics_csv results/metrics/smoke_super_resolution.csv \
      --output_dir results/figures/smoke
"""
import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.local_paths import resolve  # noqa: E402


def _read(csv_path):
    rows, mean = [], None
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if str(r["idx"]).upper() == "MEAN":
                mean = {k: float(r[k]) for k in ("psnr", "ssim", "lpips")}
            else:
                rows.append({k: float(r[k]) for k in ("psnr", "ssim", "lpips")})
    if mean is None and rows:
        import statistics as st
        mean = {k: st.mean(r[k] for r in rows) for k in ("psnr", "ssim", "lpips")}
    return rows, mean


def main():
    ap = argparse.ArgumentParser(description="Plot metric CSV(s).")
    ap.add_argument("--metrics_csv", required=True, help="one or more CSVs, comma-separated")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csvs = [resolve(c.strip()) for c in args.metrics_csv.split(",") if c.strip()]
    labels = [os.path.splitext(os.path.basename(c))[0] for c in csvs]
    data = [_read(c) for c in csvs]
    out = os.path.abspath(resolve(args.output_dir))
    os.makedirs(out, exist_ok=True)

    metrics = ["psnr", "ssim", "lpips"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for mi, m in enumerate(metrics):
        vals = [(d[1][m] if d[1] else 0.0) for d in data]
        axes[mi].bar(range(len(vals)), vals, color=[["#3b6", "#36b", "#b63", "#963"][i % 4] for i in range(len(vals))])
        axes[mi].set_xticks(range(len(vals)))
        axes[mi].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        axes[mi].set_title(f"mean {m.upper()}" + (" (lower=better)" if m == "lpips" else " (higher=better)"))
        for i, v in enumerate(vals):
            axes[mi].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    p1 = os.path.join(out, "metrics_mean_bars.png")
    plt.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close()

    # per-image strip for the first CSV
    rows0 = data[0][0]
    if rows0:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3))
        for mi, m in enumerate(metrics):
            axes[mi].plot([r[m] for r in rows0], marker="o", ms=3)
            axes[mi].set_title(f"{labels[0]} — per-image {m.upper()}")
            axes[mi].set_xlabel("image idx")
        plt.tight_layout()
        plt.savefig(os.path.join(out, "metrics_per_image.png"), dpi=120, bbox_inches="tight")
        plt.close()

    print(f"[visualize] wrote {p1}")
    for lbl, d in zip(labels, data):
        if d[1]:
            print(f"  {lbl}: PSNR={d[1]['psnr']:.3f} SSIM={d[1]['ssim']:.4f} LPIPS={d[1]['lpips']:.4f}")
    print(f"OUTPUT_DIR={out}")


if __name__ == "__main__":
    main()
