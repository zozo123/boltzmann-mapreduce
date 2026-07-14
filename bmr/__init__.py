"""Uncertainty-aware confidence-distribution reduction for independent workers."""
from bmr.core import (
    CD, CanonicalSummary, Pooled, finalize_canonical,
    map_mean, map_linreg, map_logistic, reduce_partition,
    reduce_naive, byzantine_clip, oracle_mean, oracle_linreg, oracle_logistic,
)

__all__ = [
    "CD", "CanonicalSummary", "Pooled", "finalize_canonical",
    "map_mean", "map_linreg", "map_logistic", "reduce_partition",
    "reduce_naive", "byzantine_clip", "oracle_mean", "oracle_linreg", "oracle_logistic",
]
__version__ = "0.1.0"
