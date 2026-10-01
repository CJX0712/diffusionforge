"""评测子包。"""

from .metrics import (
    mmd,
    wasserstein2,
    mode_coverage,
    energy_distance,
    evaluate_samples,
)

__all__ = [
    "mmd",
    "wasserstein2",
    "mode_coverage",
    "energy_distance",
    "evaluate_samples",
]
