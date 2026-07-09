#!/usr/bin/env python
"""Generate measurements (y), GTs, and masks for a task — DAPS-aligned operators,
all-seeds-42. Reuses the VERIFIED runner helpers (_setup_task / _build_inpaint_operator
/ load_rerun_group / seeding) so the saved y matches what run_ours regenerates internally.

Example:
  benchmark/generate_measurements.py --config configs/imagenet/deblur.yaml \
      --paths configs/paths.local.yaml --num_images 5 \
      --output_dir results/reproduced_results/measurements/deblur

Output: <output_dir>/{gts/, measurements/, masks/ (inpaint only), meta.json}
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import yaml  # noqa: E402
from src.utils.local_paths import bootstrap, resolve  # noqa: E402


def _save01(t, path):
    import numpy as np
    from PIL import Image
    x = ((t + 1) / 2).clamp(0, 1) if t.min() < 0 else t.clamp(0, 1)
    a = (x.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(a).save(path)


def main():
    ap = argparse.ArgumentParser(description="Generate y/GT/mask for a task.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--paths", default="configs/paths.local.yaml")
    ap.add_argument("--num_images", type=int, default=5)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--group_id", type=int, default=None)
    args = ap.parse_args()

    paths = bootstrap(resolve(args.paths))
    import torch  # noqa: E402
    import runner  # noqa: E402
    import seeding  # noqa: E402

    tc = yaml.safe_load(open(resolve(args.config)))
    task, op_cfg = tc["task"], tc["operator"]
    sigma_n = float(tc.get("sigma_n", 0.05))
    resolution = int(tc.get("resolution", 256))
    bs_size = int(tc.get("batch_size", 8))
    gpu = args.gpu if args.gpu is not None else int(paths.get("default_gpu", 0))
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    group_id = args.group_id if args.group_id is not None else int(tc.get("group_id", paths.get("daps_group_id", 100)))
    groups_dir = os.path.abspath(paths["groups_dir"])

    operator, mask_subdir, meas_label, *_ = runner._setup_task(task, op_cfg, device, sigma_n, resolution)
    gmeta, gts, masks = runner.load_rerun_group(groups_dir, group_id, mask_subdir)
    gts = gts.to(device)
    if masks is not None:
        masks = masks.to(device)
    N = min(args.num_images, gts.shape[0]) if args.num_images and args.num_images > 0 else gts.shape[0]

    out = os.path.abspath(resolve(args.output_dir))
    for sub in ("gts", "measurements") + (("masks",) if mask_subdir else ()):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    is_inpaint = task in ("random_inpainting", "box_inpainting")
    for bs in range(0, N, bs_size):
        be = min(bs + bs_size, N)
        gt_b = gts[bs:be]
        seeding.set_all_seeds(seeding.SEED)
        if is_inpaint:
            op_b = runner._build_inpaint_operator(task, op_cfg, device, sigma_n, resolution)
            mb = masks[bs:be]
            runner._force_mask(op_b, mb)
            y_b = op_b(gt_b).detach()
        else:
            y_clean = operator(gt_b).detach()
            y_b = (y_clean + sigma_n * torch.randn_like(y_clean)).detach()
        for j in range(be - bs):
            i = bs + j
            _save01(gt_b[j], os.path.join(out, "gts", f"{i:03d}.png"))
            _save01(y_b[j], os.path.join(out, "measurements", f"{i:03d}.png"))
            if is_inpaint:
                _save01(masks[i].repeat(3, 1, 1), os.path.join(out, "masks", f"{i:03d}.png"))
        print(f"[gen_meas] {task} [{bs}:{be}] saved", flush=True)

    meta = dict(task=task, group_id=group_id, num_images=N, sigma_n=sigma_n,
                operator=op_cfg, measurement_label=meas_label, noiseless=is_inpaint,
                fnames=gmeta["fnames"][:N], labels=[int(x) for x in gmeta["labels"][:N]])
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(f"[gen_meas] done: {N} imgs -> {out}")
    print(f"OUTPUT_DIR={out}")


if __name__ == "__main__":
    main()
