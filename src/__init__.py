"""Local reproduction package. Thin re-export layer over the VERIFIED original
IP_package code (sampler / operators / metrics / runner). Nothing here
reimplements the sampling method — it bootstraps sys.path from configs/paths.local.yaml
and re-exports, so reproduction is bit-for-bit. See docs/release_migration_notes.md."""
