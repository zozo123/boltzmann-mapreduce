# Research roadmap

## Current evidence

| Component | Status |
|---|---|
| Validated worker-result schema | Implemented and tested |
| Gaussian canonical merge | Flat and hierarchical APIs implemented and tested |
| Exact unnormalized product normalizer | Implemented and tested |
| Lineage and evidence identifiers | Literal overlap rejected; lineage correlation not modeled |
| Local execution | Python process pool |
| Named-snapshot execution | One four-shard islo run |
| Daytona/Tensorlake observations | Default sandbox creation only |
| Outlier handling | Multivariate stress-test heuristic; no robustness guarantee |
| Agent workload | Research agenda only |

## What the artifact establishes

- Independent Gaussian factors merge through additive natural parameters.
- The exact normalizer includes disagreement energy
  `1/2 * (c - q.T @ inv(P) @ q)`.
- Malformed sample sizes and information matrices are rejected.
- Hierarchical canonical merging matches flat reduction, and literal duplicate
  evidence IDs are rejected unless the caller explicitly overrides the guard.
- The outlier heuristic examines all parameter coordinates and bounds the supplied
  multivariate regression attack.
- The islo adapter can execute deterministic workers from a named snapshot.

These are algebraic, software, and execution-path checks. They are not evidence of
general statistical efficiency, Byzantine tolerance, or a systems performance win.

## Experiments needed for a full research paper

1. **Statistical calibration.** At least 500 repetitions across `K`, local `n`, and
   dimension `p`; report bias, MSE, interval coverage, width, and convergence failures.
2. **Serious baselines.** Exact sufficient statistics, sample-size weighting,
   conventional meta-analysis/Rao-CD, surrogate likelihood, and distributed Newton.
3. **Non-IID targets.** Covariate shift, heteroskedasticity, overlapping evidence,
   correlated workers, and genuinely different local parameters with random effects.
4. **Robustness threat model.** Adaptive and colluding attacks on both location and
   covariance; formal breakdown or influence analysis; affine-invariant multivariate
   baselines.
5. **Controlled systems comparison.** Identical image and workload on processes,
   containers, cold microVMs, and named-snapshot restores; concurrent fanout,
   p50/p95/p99, page faults, memory sharing, failures, cost, and utility per second.
6. **Agent evidence.** A real forked coding/evaluation workload where confidence is
   derived from held-out tests or independent verification rather than model self-report.
7. **Lineage-aware inference.** Use the fork DAG and evidence identifiers to avoid
   double counting shared observations and common ancestors.
8. **Adaptive selection.** Correct confidence after branch pruning, tournament
   selection, and repeated use of the same tests.

## Engineering priorities

1. Replace the demonstration heuristic with a separately evaluated robust estimator.
2. Store raw structured benchmark traces with timestamps and environment manifests.
3. Add an explicit correlation model over the lineage DAG.
4. Add numerical conditioning diagnostics and shifted canonical summaries for extreme
   parameter offsets.
5. Extend CI with rendered-page regression checks once a stable renderer is pinned.
