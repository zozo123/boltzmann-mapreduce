"""Correctness tests for the Boltzmann MapReduce core."""
import numpy as np
import pytest

from bmr.core import (
    CD, CanonicalSummary, finalize_canonical,
    map_mean, reduce_partition, byzantine_clip,
    oracle_mean, oracle_linreg, map_linreg,
    map_logistic, oracle_logistic, reduce_naive,
)
from bmr.worker import run_one


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
    # centralized comparison) up to finite-sample variance-estimation noise; they are
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


def test_logistic_pooled_tracks_oracle_and_beats_equal_weighting():
    # Non-linear estimating function (logistic) with strongly unequal IID shard
    # sizes. Information weighting should track the full-data MLE more closely
    # than the deliberately weak equal-weight coefficient average.
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
    assert err_pooled < 0.05                 # pooled tracks the full-data MLE here
    assert err_pooled < err_naive            # and beats naive equal-weight averaging


def test_cd_roundtrip_and_gibbs_identity():
    cd = map_mean(np.array([1.0, 2.0, 3.0, 4.0]), 7)
    cd.lineage = ["root", "branch-7"]
    cd.evidence_ids = ["sample-1", "sample-2"]
    again = CD.from_dict(cd.to_dict())
    assert again.theta_hat == cd.theta_hat
    assert again.n == cd.n
    assert again.lineage == cd.lineage
    assert again.evidence_ids == cd.evidence_ids
    theta = [2.5]
    assert cd.neg_log_density(theta) == pytest.approx(cd.beta * cd.energy(theta), rel=1e-12)
    assert cd.temperature() == pytest.approx(1.0 / cd.n, rel=1e-12)
    with pytest.raises(ValueError):
        CD.from_dict({"theta_hat": [0.0], "info_per_obs": [[1.0]], "n": 1.5})


def test_partition_normalizer_includes_worker_disagreement():
    agree = [CD([0.0], [[1.0]], 1), CD([0.0], [[1.0]], 1)]
    disagree = [CD([0.0], [[1.0]], 1), CD([10.0], [[1.0]], 1)]
    pooled_agree = reduce_partition(agree)
    pooled_disagree = reduce_partition(disagree)

    # Moving the two unit-precision modes ten units apart adds 25 to -log Z.
    assert pooled_disagree.neg_log_Z - pooled_agree.neg_log_Z == pytest.approx(25.0)
    assert pooled_agree.disagreement_energy == pytest.approx(0.0)
    assert pooled_disagree.disagreement_energy == pytest.approx(25.0)


def test_finalization_clips_only_roundoff_scale_negative_disagreement():
    near_consistent = CanonicalSummary(
        precision=[[1.0]],
        info_theta=[1.0],
        quadratic_constant=1.0 - 1e-15,
        n_total=1,
    )
    pooled = finalize_canonical(near_consistent)
    assert pooled.disagreement_energy == pytest.approx(0.0)

    inconsistent = CanonicalSummary(
        precision=[[1.0]],
        info_theta=[10.0],
        quadratic_constant=0.0,
        n_total=1,
    )
    with pytest.raises(ValueError, match="quadratic constant is inconsistent"):
        finalize_canonical(inconsistent)


def test_canonical_merge_is_associative_and_matches_flat_reduce():
    cds = [
        CD([float(k), -float(k)], [[2.0, 0.2], [0.2, 1.0]], k + 2,
           lineage=["root", f"branch-{k}"], evidence_ids=[f"batch-{k}"])
        for k in range(3)
    ]
    summaries = [CanonicalSummary.from_cd(cd) for cd in cds]
    left = summaries[0].merge(summaries[1]).merge(summaries[2])
    right = summaries[0].merge(summaries[1].merge(summaries[2]))
    hierarchical = finalize_canonical(left)
    flat = reduce_partition(cds)

    assert np.allclose(left.precision, right.precision)
    assert np.allclose(left.info_theta, right.info_theta)
    assert left.quadratic_constant == pytest.approx(right.quadratic_constant)
    assert np.allclose(hierarchical.theta, flat.theta)
    assert np.allclose(hierarchical.cov, flat.cov)
    assert hierarchical.neg_log_Z == pytest.approx(flat.neg_log_Z)
    assert set(left.evidence_ids) == {"batch-0", "batch-1", "batch-2"}


def test_final_pooled_result_preserves_provenance():
    cds = [
        CD([0.0], [[1.0]], 10,
           lineage=["snapshot-a", "branch-a"], evidence_ids=["evaluation-a"]),
        CD([1.0], [[1.0]], 20,
           lineage=["snapshot-a", "branch-b"], evidence_ids=["evaluation-b"]),
    ]

    flat = reduce_partition(cds)
    summary = CanonicalSummary.from_cd(cds[0]).merge(CanonicalSummary.from_cd(cds[1]))
    hierarchical = finalize_canonical(summary)

    expected_evidence = ("evaluation-a", "evaluation-b")
    expected_lineage = ("snapshot-a", "branch-a", "branch-b")
    assert flat.evidence_ids == expected_evidence
    assert flat.lineage == expected_lineage
    assert hierarchical.evidence_ids == expected_evidence
    assert hierarchical.lineage == expected_lineage


def test_canonical_numeric_merge_is_commutative_but_lineage_order_is_not():
    a = CanonicalSummary.from_cd(
        CD([0.0], [[2.0]], 3,
           lineage=["root", "branch-a"], evidence_ids=["evaluation-a"])
    )
    b = CanonicalSummary.from_cd(
        CD([2.0], [[1.0]], 5,
           lineage=["root", "branch-b"], evidence_ids=["evaluation-b"])
    )

    ab = a.merge(b)
    ba = b.merge(a)

    assert np.allclose(ab.precision, ba.precision)
    assert np.allclose(ab.info_theta, ba.info_theta)
    assert ab.quadratic_constant == pytest.approx(ba.quadratic_constant)
    assert ab.n_total == ba.n_total
    assert set(ab.evidence_ids) == set(ba.evidence_ids)
    assert ab.lineage == ("root", "branch-a", "branch-b")
    assert ba.lineage == ("root", "branch-b", "branch-a")
    assert ab != ba


def test_duplicate_evidence_is_rejected_unless_explicitly_allowed():
    first = CD([0.0], [[1.0]], 10, evidence_ids=["shared-eval"])
    second = CD([1.0], [[1.0]], 10, evidence_ids=["shared-eval"])
    with pytest.raises(ValueError, match="overlapping evidence_ids"):
        reduce_partition([first, second])
    allowed = reduce_partition([first, second], allow_evidence_overlap=True)
    assert allowed.theta[0] == pytest.approx(0.5)


def test_multivariate_clip_detects_attack_outside_first_coordinate():
    honest = [
        CD([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 100, {"shard_id": i})
        for i in range(10)
    ]
    liar = CD([0.0, 1000.0], [[1.0, 0.0], [0.0, 1.0]], 100, {"shard_id": "LIAR"})
    attacked = reduce_partition(honest + [liar])
    kept, flagged = byzantine_clip(honest + [liar])
    bounded = reduce_partition(kept)

    assert "LIAR" in flagged
    assert abs(bounded.theta[1]) < abs(attacked.theta[1]) * 1e-3


def test_worker_transports_provenance_and_backend(monkeypatch):
    monkeypatch.setenv("BMR_BACKEND", "islo")
    cd = run_one({
        "scenario": "mean", "x": [1.0, 2.0, 3.0], "shard_id": 4,
        "lineage": ["snapshot-a", "branch-4"],
        "evidence_ids": ["evaluation-batch-4"],
    })
    assert cd.meta["backend"] == "islo"
    assert cd.lineage == ["snapshot-a", "branch-4"]
    assert cd.evidence_ids == ["evaluation-batch-4"]


def test_map_estimators_reject_statistically_unidentified_inputs():
    with pytest.raises(ValueError, match="at least two"):
        map_mean([1.0])
    with pytest.raises(ValueError, match="sample variance"):
        map_mean([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="n > p"):
        map_linreg(np.eye(2), np.ones(2))
    with pytest.raises(ValueError, match="both classes"):
        map_logistic(np.ones((5, 2)), np.ones(5))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"theta_hat": [0.0], "info_per_obs": [[1.0]], "n": 0},
        {"theta_hat": [0.0], "info_per_obs": [[-1.0]], "n": 1},
        {"theta_hat": [0.0, 1.0], "info_per_obs": [[1.0]], "n": 1},
        {"theta_hat": [float("nan")], "info_per_obs": [[1.0]], "n": 1},
    ],
)
def test_cd_rejects_malformed_worker_summaries(kwargs):
    with pytest.raises(ValueError):
        CD(**kwargs)
