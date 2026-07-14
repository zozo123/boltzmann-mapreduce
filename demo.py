"""Boltzmann MapReduce demo.

Generates synthetic data with a known truth, shards it, maps each shard to a
confidence density via the chosen backend, and reduces by information-weighted
pooling. Prints a per-shard table (with the T = 1/n convention), compares pooled
vs the full-data estimator, optionally runs an outlier stress test, and saves
three figures.

    uv run --locked python demo.py --scenario mean --byzantine
    uv run --locked python demo.py --scenario linreg --shards 16
    uv run --locked python demo.py --backend islo --scenario mean  # requires: islo login
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from bmr.core import (
    reduce_partition, reduce_naive, byzantine_clip, oracle_mean, oracle_linreg,
    oracle_logistic, map_mean,
)
from bmr.backends import get_backend, IsloUnavailable
from bmr import viz


def make_mean_shards(rng, K, n, true_mu, sd=1.5):
    """K homoscedastic IID shards centered at true_mu, with unequal sizes n_k.

    Equal variance keeps the precision-weighted pool asymptotically equal to the
    full-data estimator (the efficiency ceiling); the varying n_k is what makes
    each shard's temperature T = 1/n_k --- and thus its weight --- differ.
    """
    shards, all_x = [], []
    for k in range(K):
        nk = int(rng.integers(max(20, n // 2), n * 2))
        x = rng.normal(true_mu, sd, size=nk)
        shards.append({
            "scenario": "mean", "x": x.tolist(), "shard_id": k,
            "lineage": ["demo-root", f"mean-branch:{k}"],
            "evidence_ids": [f"mean-shard:{k}"],
        })
        all_x.append(x)
    return shards, np.concatenate(all_x)


def make_linreg_shards(rng, K, n, true_beta):
    p = len(true_beta)
    shards, Xs, ys = [], [], []
    for k in range(K):
        nk = int(rng.integers(max(30, n // 2), n * 2))
        X = np.column_stack([np.ones(nk)] + [rng.normal(0, 1, nk) for _ in range(p - 1)])
        noise = rng.normal(0, float(rng.uniform(0.5, 2.0)), nk)
        y = X @ np.asarray(true_beta) + noise
        shards.append({
            "scenario": "linreg", "X": X.tolist(), "y": y.tolist(), "shard_id": k,
            "lineage": ["demo-root", f"linreg-branch:{k}"],
            "evidence_ids": [f"linreg-shard:{k}"],
        })
        Xs.append(X); ys.append(y)
    return shards, np.vstack(Xs), np.concatenate(ys)


def make_logistic_shards(rng, true_beta, sizes):
    """Strongly unequal-size IID logistic shards for a non-linear sanity check."""
    p = len(true_beta)
    shards, Xs, ys = [], [], []
    for k, nk in enumerate(sizes):
        X = np.column_stack([np.ones(nk)] + [rng.normal(0, 1, nk) for _ in range(p - 1)])
        prob = 1.0 / (1.0 + np.exp(-(X @ np.asarray(true_beta))))
        y = (rng.uniform(size=nk) < prob).astype(float)
        shards.append({
            "scenario": "logistic", "X": X.tolist(), "y": y.tolist(), "shard_id": k,
            "lineage": ["demo-root", f"logistic-branch:{k}"],
            "evidence_ids": [f"logistic-shard:{k}"],
        })
        Xs.append(X); ys.append(y)
    return shards, np.vstack(Xs), np.concatenate(ys)


def _fmt_vec(v):
    return "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"


def run(args):
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    if args.scenario == "mean":
        true_theta = 5.0
        shards, all_x = make_mean_shards(rng, args.shards, args.n_per_shard, true_theta)
        oracle = oracle_mean(all_x)
    elif args.scenario == "logistic":
        true_beta = [0.5, -1.2, 2.0]
        sizes = [2000, 400, 120, 60, 5000]  # strongly unequal IID shard sizes
        shards, allX, ally = make_logistic_shards(rng, true_beta, sizes)
        oracle = oracle_logistic(allX, ally)
    else:
        true_beta = [1.5, -2.0, 0.75]
        shards, allX, ally = make_linreg_shards(rng, args.shards, args.n_per_shard, true_beta)
        oracle = oracle_linreg(allX, ally)

    backend = get_backend(args.backend, harness=args.harness)
    try:
        cds = backend.map_shards(shards)
        used = args.backend
    except (IsloUnavailable, RuntimeError, OSError) as exc:
        print(f"\n[islo backend unavailable] {exc}\n"
              f"  --> falling back to the local backend "
              f"(fix islo + `islo login`, then retry --backend islo).\n")
        cds = get_backend("local").map_shards(shards)
        used = "local (islo fallback)"

    pooled = reduce_partition(cds)

    print("=" * 74)
    print(f"  Boltzmann MapReduce  |  scenario={args.scenario}  backend={used}  "
          f"shards={len(cds)}")
    print("=" * 74)
    if args.scenario == "mean":
        print(f"{'shard':>5} {'n':>6} {'theta_hat':>11} {'se':>8} {'T=1/n':>9} {'weight%':>8}")
        for cd in sorted(cds, key=lambda c: c.meta.get('shard_id', 0)):
            print(f"{cd.meta.get('shard_id'):>5} {cd.n:>6} {cd.theta_hat[0]:>11.4f} "
                  f"{cd.se()[0]:>8.4f} {cd.temperature():>9.5f} "
                  f"{100*pooled.weights.get(cd.meta.get('shard_id'),0):>7.1f}%")
        (lo, hi) = pooled.ci()[0]
        print("-" * 74)
        print(f"  TRUE theta      = {true_theta:.4f}")
        print(f"  POOLED theta    = {pooled.theta[0]:.4f}   95% CI = [{lo:.4f}, {hi:.4f}]")
        (olo, ohi) = oracle.ci()[0]
        print(f"  ORACLE theta    = {oracle.theta_hat[0]:.4f}   95% CI = [{olo:.4f}, {ohi:.4f}]")
        print(f"  |pooled-oracle| = {abs(pooled.theta[0]-oracle.theta_hat[0]):.2e}  "
              f"(one deterministic sanity check; not an efficiency estimate)")
        viz.fig_gibbs(cds, pooled, oracle, true_theta, os.path.join(args.out, "fig_gibbs.png"))
        viz.fig_cooling(cds, os.path.join(args.out, "fig_cooling.png"), seed=args.seed)
    else:
        print(f"  TRUE   beta   = {_fmt_vec(true_beta)}")
        print(f"  POOLED beta   = {_fmt_vec(pooled.theta)}")
        print(f"  ORACLE beta   = {_fmt_vec(oracle.theta_hat)}")
        pooled_err = float(np.linalg.norm(np.array(pooled.theta) - np.array(oracle.theta_hat)))
        print(f"  ||pooled-full||    = {pooled_err:.2e}  (information-weighted sanity check)")
        if args.scenario == "logistic":
            naive = reduce_naive(cds)
            naive_err = float(np.linalg.norm(naive - np.array(oracle.theta_hat)))
            print(f"  NAIVE  beta   = {_fmt_vec(naive)}  (equal-weight average)")
            print(f"  ||naive-oracle||   = {naive_err:.3f}  --> naive is "
                  f"{naive_err/max(pooled_err,1e-12):.0f}x worse "
                  f"(non-linear, unequal-size IID shards)")

    if args.byzantine and args.scenario == "mean":
        liar_theta = true_theta + 12.0
        liar = map_mean(rng.normal(liar_theta, 0.02, size=2000), shard_id="LIAR")
        liar.info_per_obs = [[v * 50.0 for v in liar.info_per_obs[0]]]  # false high precision
        attacked = cds + [liar]
        naive = reduce_partition(attacked)
        kept, flagged = byzantine_clip(attacked, kappa=3.0)
        robust = reduce_partition(kept)
        print("-" * 74)
        print("  OUTLIER STRESS TEST: one false-precision worker injected "
              f"(theta={liar_theta:.1f}, falsely tiny variance)")
        print(f"    naive  pooled = {naive.theta[0]:.4f}   (pulled toward the injected worker)")
        print(f"    bounded pool  = {robust.theta[0]:.4f}   (heuristic; flagged={flagged})")
        viz.fig_byzantine(true_theta, naive, robust, liar_theta,
                          os.path.join(args.out, "fig_byzantine.png"))

    print("=" * 74)
    print(f"  figures written to: {args.out}/")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(description="Boltzmann MapReduce demo")
    ap.add_argument("--backend", choices=["local", "islo"], default="local")
    ap.add_argument("--harness", default=None, metavar="AGENT",
                    help="On-box AI harness profile as the islo fork entry point "
                         "(e.g. 'claude'); omit to run the worker directly.")
    ap.add_argument("--scenario", choices=["mean", "linreg", "logistic"], default="mean")
    ap.add_argument("--shards", type=int, default=12)
    ap.add_argument("--n-per-shard", type=int, default=200)
    ap.add_argument("--byzantine", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figs")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
