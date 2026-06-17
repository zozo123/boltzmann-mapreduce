"""Boltzmann MapReduce: distributed statistical inference as Gibbs-ensemble computation."""
from bmr.core import (
    CD, Pooled, map_mean, map_linreg, map_logistic, reduce_partition,
    reduce_naive, byzantine_clip, oracle_mean, oracle_linreg, oracle_logistic,
)

__all__ = [
    "CD", "Pooled", "map_mean", "map_linreg", "map_logistic", "reduce_partition",
    "reduce_naive", "byzantine_clip", "oracle_mean", "oracle_linreg", "oracle_logistic",
]
__version__ = "0.1.0"
