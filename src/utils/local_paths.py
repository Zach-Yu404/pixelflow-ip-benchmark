"""Single bootstrap used by every benchmark/ script and src/ module.

Loads configs/paths.local.yaml and puts the ORIGINAL, already-validated code on
sys.path (the sampler + pixelflow live in original_ip_package; runner/operators/
metrics/seeding live in original_rerun_imagenet). This is the ONLY place that
touches sys.path / absolute paths, keeping the rest of the project release-clean.
"""
import os
import sys
import functools

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError("PyYAML required: pip install pyyaml (see requirements.txt)") from e


def project_root():
    """github_project_local/ — two levels up from this file (src/utils/)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@functools.lru_cache(maxsize=1)
def load_paths(paths_yaml=None):
    """Load configs/paths.local.yaml -> dict. Override path via arg or
    $PATHS_LOCAL_YAML; otherwise default to <project_root>/configs/paths.local.yaml."""
    if paths_yaml is None:
        paths_yaml = os.environ.get(
            "PATHS_LOCAL_YAML",
            os.path.join(project_root(), "configs", "paths.local.yaml"),
        )
    with open(paths_yaml) as f:
        p = yaml.safe_load(f)
    p = {k: (os.path.expanduser(v) if isinstance(v, str) and v.startswith("~") else v)
         for k, v in p.items()}
    p["_paths_yaml"] = os.path.abspath(paths_yaml)
    return p


def bootstrap(paths_yaml=None):
    """Put the original code dirs on sys.path and chdir into IP_package (runner.py
    expects CWD == IP_package because the sampler resolves a few relative paths).
    Returns the loaded paths dict. Idempotent."""
    p = load_paths(paths_yaml)
    ipp = p["original_ip_package"]
    rr = p["original_rerun_imagenet"]
    for d in (rr, ipp):                 # rr first so its operators/metrics/seeding win
        if d not in sys.path:
            sys.path.insert(0, d)
    # runner.py itself does os.chdir(IPP); do it here too for standalone reuse.
    if os.path.isdir(ipp):
        os.chdir(ipp)
    return p


def resolve(path):
    """Resolve a possibly-relative project path against project_root."""
    if os.path.isabs(path):
        return path
    return os.path.join(project_root(), path)
