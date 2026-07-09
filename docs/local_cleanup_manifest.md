# Local cleanup manifest

What this project created, what it points at, and what is safe to delete.

## Created fresh (this project; safe to regenerate)
- `configs/**` — all YAML (paths.local, models, eval, baselines, imagenet/*)
- `src/**` — re-export package (no original code copied; pure thin wrappers)
- `benchmark/**`, `scripts/**` — entry scripts
- `README.md`, `LOCAL_REPRODUCTION.md`, `requirements.txt`, `environment.yml`, `.gitignore`
- `docs/**` — manifests + reports
- `results/reproduced_results/**`, `results/metrics/{smoke,full100,existing,reproduction,baseline_comparison}*`,
  `results/figures/{smoke,existing_eval}/**` — regenerable outputs

## Symlinks (33 total; point INTO the original tree — never modified)
- `checkpoints/ours/{model.pt,config.yaml}` → `pretrained_models/c2img/`
- `checkpoints/afhq/*` → `pretrained_models/afhq/`; `checkpoints/celeba/*` → `exp_155024/`
- `results/baselines/*` → `baselines/results/{updated_all_results,/}`
- `results/previous_results/*`, `results/figures/{final_3way_grids,metric_visualization_grids}`,
  `results/metrics/previous_*` → audit outputs
- `results/previous_results/groups_used` → `rerun_imageNet/groups`
- `results/previous_results/{afhq,celeba,mri}/*` → `demo_runs_afhq` / `celeba_results/{runs,metrics}` / `debug_IP4/MRI/val_final_8`
  (MRI is reference-only; `checkpoints/mri/` is a MISSING marker, not a symlink)

## NOT touched (original repo integrity)
Per the task constraint **"不要破坏原始 repo"**: nothing outside
`github_project_local/` was created, moved, or edited by this project. All
references to the original tree are read-only (symlinks / sys.path imports).
The one pre-existing change in the original tree — a `--seed` arg added to
`IP_package/demo_runner_batch.py` — was made by the user/linter, not here.

## Safe to delete (and how to rebuild)
| delete | rebuild with |
|---|---|
| `results/reproduced_results/` | `bash scripts/reproduce_previous_results.sh [--full]` |
| `results/metrics/{smoke,full100,existing,reproduction}*`, `figures/{smoke,existing_eval}` | the smoke / evaluate scripts |
| `docs/reproduction_commands.log` | regenerated on next script run |
| `__pycache__/` | transient |

## If re-running this builder
The project dir already exists. Per the constraint **"如果该目录已存在，先备份或
增量更新，不要直接删除"**: do an incremental update (overwrite individual files)
or back up to `github_project_local.bak.<date>` first — do **not** `rm -rf` it.
