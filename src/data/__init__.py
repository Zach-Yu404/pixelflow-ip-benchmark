"""Data loading — re-exported from the original code.

  load_rerun_group(groups_dir, group_id, mask_subdir) -> (meta, gts[-1,1], masks|None)
  seeding : the all-seeds-42 policy MODULE (use seeding.set_all_seeds / seeding.sampler_seed);
            SEED (==42) is also re-exported directly for convenience.

group_100 is DAPS's exact 100 ImageNet-val images (val 49000..49099), GTs built with
center_crop_arr (ADM-style proper crop), masks pre-generated. Reused verbatim."""
from src.utils.local_paths import bootstrap as _bootstrap

_bootstrap()

import seeding  # noqa: E402  (rerun_imageNet/seeding.py: all-seeds-42)
from runner import load_rerun_group, to_img01  # noqa: E402,F401

SEED = seeding.SEED

__all__ = ["load_rerun_group", "to_img01", "seeding", "SEED"]
