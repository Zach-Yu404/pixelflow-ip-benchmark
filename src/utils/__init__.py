"""Utilities for the local reproduction package.

The only module here is ``local_paths``, the single bootstrap that loads
``configs/paths.local.yaml`` and puts the original ``IP_package`` /
``rerun_imageNet`` dirs on ``sys.path`` (rerun first). Callers import it as
``from src.utils.local_paths import bootstrap, resolve, load_paths`` — this
package ``__init__`` intentionally re-exports nothing.
"""
