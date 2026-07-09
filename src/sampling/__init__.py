"""Posterior sampler — re-exported from the original IP_package. This is the
multi-stage Langevin / ODE flow posterior sampler that produced all results.
Reused verbatim (NOT reimplemented) so reproduction is exact."""
from src.utils.local_paths import bootstrap as _bootstrap

_bootstrap()

from ms_posterior_sampling_article_version_final import run_posterior_sampling  # noqa: E402,F401

__all__ = ["run_posterior_sampling"]
