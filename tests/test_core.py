"""Correctness tests for the Boltzmann MapReduce core."""
import numpy as np
import pytest

from bmr.core import (
    CD, map_mean, reduce_partition, byzantine_clip,
    oracle_mean, oracle_linreg, map_linreg,
    map_logistic, oracle_logistic, reduce_naive,
)


def test_reduce_matches_closed_form_inverse_variance():
    # Two scalar-mean shards: pooled mean/precision must equal the analytic formula.
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    x2 = np.array([2.0, 2.5, 3.5, 4.0])
    c1, c2 = map_mean(x1, 0), map_mean(x2, 1)
    pooled = reduce_partition([c1, c2])

    p1 = c1.n / np.var(x1, ddof=1)
    p2 = c2.n / np.var(x2, ddof=1)
    expected_theta = (p1 * x1.mean() + p2 * x2.mean()) / (p1 + p2)
    expected_var = 1.0 / (p1 + p2)
    assert pooled.theta[0] == pytest.approx(expected_theta, rel=1e-12)
    assert pooled.cov[0][0] == pytest.approx(expected_var, rel=1e-12)


def test_pooled_tracks_oracle_mean_homoscedastic():
    # Under equal variance, precision-weighting tracks the full-data mean (the
    # efficiency ceiling) up to finite-sample variance-estimation noise; they are
    # asymptotically equivalent, not bit-identical.
    rng = np.random.default_rng(1)
    parts = [rng.normal(3.0, 1.5, n) for n in (400, 600, 300, 500)]
    cds = [map_mean(p, i) for i, p in enumerate(parts)]
    pooled = reduce_partition(cds)
    oracle = oracle_mean(np.concatenate(parts))
    assert pooled.theta[0] == pytest.approx(oracle.theta_hat[0], rel=2e-2)
    assert abs(pooled.theta[0] - 3.0) < 0.3  # both consistent for the truth


def test_pooled_precision_matches_oracle_asymptotically():
    # Pooling recovers full-data precision (no free information). With plug-in
    # variance estimates the two plug-in precisions differ by O(1/sqrt(n)) in
    # either direction, converging as n grows -- an asymptotic statement.
    rng = np.random.default_rng(11)
    parts = [rng.normal(0.0, 1.0, n) for n in (4000, 4000, 4000)]
    cds = [map_mean(p, i) for i, p in enumerate(parts)]
    pooled = reduce_partition(cds)
    oracle = oracle_mean(np.concatenate(parts))
    pooled_prec = 1.0 / pooled.cov[0][0]
    oracle_prec = float(oracle.total_precision()[0, 0])
    assert pooled_prec == pytest.approx(oracle_prec, rel=3e-2)


def test_pooled_close_to_oracle_linreg():
    rng = np.random.default_rng(2)
    beta = np.array([1.0, -1.5, 0.5])
    Xs, ys = [], []
    cds = []
    for k in range(6):
        n = 200
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n), rng.normal(0, 1, n)])
        y = X @ beta + rng.normal(0, 1.0, n)
        Xs.append(X); ys.append(y)
        cds.append(map_linreg(X, y, k))
    pooled = reduce_partition(cds)
    oracle = oracle_linreg(np.vstack(Xs), np.concatenate(ys))
    assert np.allclose(pooled.theta, oracle.theta_hat, atol=0.05)


def test_byzantine_clip_beats_naive_and_flags_liar():
    rng = np.random.default_rng(3)
    cds = [map_mean(rng.normal(5.0, 1.0, 150), i) for i in range(10)]
    liar = map_mean(rng.normal(17.0, 0.02, 2000), "LIAR")
    liar.info_per_obs = [[liar.info_per_obs[0][0] * 50.0]]  # false high precision
    attacked = cds + [liar]

    naive = reduce_partition(attacked)
    kept, flagged = byzantine_clip(attacked, kappa=3.0)
    robust = reduce_partition(kept)

    assert abs(robust.theta[0] - 5.0) < abs(naive.theta[0] - 5.0)
    assert "LIAR" in flagged


def test_logistic_pooled_recovers_oracle_and_beats_naive():
    # Non-linear estimating function (logistic), HETEROGENEOUS shard sizes: the
    # informative regime. Precision-weighted pooling should recover the full-data
    # MLE and beat naive equal-weight coefficient-averaging.
    rng = np.random.default_rng(7)
    beta = np.array([0.5, -1.2, 2.0])
    sizes = [2000, 400, 120, 60, 5000]  # strongly heterogeneous
    Xs, ys, cds = [], [], []
    for k, n in enumerate(sizes):
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n), rng.normal(0, 1, n)])
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        y = (rng.uniform(size=n) < p).astype(float)
        Xs.append(X); ys.append(y); cds.append(map_logistic(X, y, k))
    pooled = np.array(reduce_partition(cds).theta)
    oracle = np.array(oracle_logistic(np.vstack(Xs), np.concatenate(ys)).theta_hat)
    naive = reduce_naive(cds)
    err_pooled = np.linalg.norm(pooled - oracle)
    err_naive = np.linalg.norm(naive - oracle)
    assert err_pooled < 0.05                 # pooled recovers the full-data MLE
    assert err_pooled < err_naive            # and beats naive equal-weight averaging


def test_cd_roundtrip_and_gibbs_identity():
    cd = map_mean(np.array([1.0, 2.0, 3.0, 4.0]), 7)
    again = CD.from_dict(cd.to_dict())
    assert again.theta_hat == cd.theta_hat
    assert again.n == cd.n
    theta = [2.5]
    assert cd.neg_log_density(theta) == pytest.approx(cd.beta * cd.energy(theta), rel=1e-12)
    assert cd.temperature() == pytest.approx(1.0 / cd.n, rel=1e-12)
