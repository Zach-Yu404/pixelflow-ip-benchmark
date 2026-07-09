# Local safety check

Scan of `github_project_local/` for secrets, credential files, surprise symlink
targets, and large blobs. Re-run with the scan block in
`docs/reproduction_commands.log` / the README. **Status: CLEAN.**

## 1. Secrets / tokens / keys (grep over *.py/yaml/sh/md/txt/json)
Patterns: `api_key`, `secret`, `password`, `token`, `bearer`, `aws_*`, private-key
headers, `wandb.*key`, `hf_…`, `sk-…`, `ghp_…`.
- **0 real secrets.** The only match is documentation text in `README.md`
  ("secret/large-file scan") and config **field names** (`secret`/`token` do not
  appear as values anywhere). No API keys, no W&B keys, no SSH/PGP private keys.

## 2. Credential / key files
`find` for `*.pem`, `id_rsa*`, `*.key`, `.netrc`, `credentials`, `wandb-*`.
- **None present.**

## 3. Symlink targets (all 33 verified)
Every symlink points into the original tree (read-only reference):
- `checkpoints/ours/{model.pt,config.yaml}` → `pretrained_models/c2img/…`
- `checkpoints/afhq/{model_epoch80…pt,config.yaml}` → `pretrained_models/afhq/…`
- `checkpoints/celeba/{model_epoch0020.pt,config.yaml}` → `exp_155024/…`
- `results/baselines/*` → `baselines/results/{updated_all_results,/}…`
- `results/previous_results/*`, `results/figures/*`, `results/metrics/previous_*` → audit outputs
- `results/previous_results/groups_used` → `rerun_imageNet/groups`
- `results/previous_results/{afhq,celeba,mri}/*` → `demo_runs_afhq` / `celeba_results{runs,metrics}` / `debug_IP4/MRI/val_final_8`
- **No symlink escapes to `$HOME`, `/etc`, `.ssh`, or anything unexpected.**

## 4. Large files
- **No real (non-symlink) file >5 MB is committed.** The 2.7 GB `model.pt` is a
  **symlink**, not a blob; `.gitignore` also excludes `*.pt`/`model.pt` defensively.

## 5. Recommendations before any public release
- Replace symlinks with a download script / model card (the blob must not ship).
- Strip the absolute paths in `configs/paths.local.yaml` (they encode the local
  filesystem) — see [release_migration_notes.md](release_migration_notes.md).
- `docs/reproduction_commands.log` may contain local paths; do not publish verbatim.
