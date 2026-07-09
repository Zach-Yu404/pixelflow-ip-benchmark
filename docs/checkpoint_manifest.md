# Checkpoint manifest

| item | value |
|---|---|
| logical name | `checkpoints/ours/model.pt` (symlink) |
| resolves to | `/sfs/ceph/standard/CBIG-Standard-ECE/Zach/MSFlow/PixelFlow/pretrained_models/c2img/model.pt` |
| size | 2,706,601,240 bytes (~2.7 GB) |
| format | **flat `state_dict`** (top keys e.g. `patch_embed.proj.weight`) — NO `ema`/`model` wrapper |
| load branch | `state_dict = ckpt` (the else-branch in runner.py:295-300), `strict=True` |
| config | `checkpoints/ours/config.yaml` (symlink → same c2img dir) |

## Model (`config.yaml`)
```yaml
target: pixelflow.model.PixelFlowModel
params: {num_attention_heads: 16, attention_head_dim: 72, in_channels: 3,
         out_channels: 3, depth: 28, num_classes: 1000, patch_size: 4, attention_bias: true}
scheduler: {num_train_timesteps: 1000, num_stages: 4, pyramid_shift: false}
```
- **Class-conditional** (`num_classes: 1000`) → used with classifier-free guidance
  (`kw.guidance_scale: 2.0`). This is the class-info advantage over DAPS's unconditional model.
- **4-stage** pyramid → the multi-stage Langevin posterior sampler.
- `data.root` in the config is a *training* path (`/public/datasets/ILSVRC2012/train`),
  irrelevant to inference/reproduction; not used by this project.

## Why a symlink (not a copy)
2.7 GB; copying into the project would bloat it and risk drift from the canonical
weights. The symlink is validated (`test -f` passes; see local_reproduction_report.md, kept locally,
Q-checks). `.gitignore` excludes `*.pt` so the blob never enters version control.
For release, swap the symlink for a download script (release_migration_notes.md).
