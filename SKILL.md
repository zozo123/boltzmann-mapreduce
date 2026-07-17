---
name: boltzmann-mapreduce
description: Use for combining independent, non-overlapping statistical summaries that target the same parameter and carry estimates plus valid uncertainty. Do not use for correlated agent votes, overlapping evidence, different local truths, or adversarial workers without a separately justified model.
---

# Boltzmann MapReduce

Use this package when independent workers return estimates with unequal information and
you need an evidence-aware Gaussian/confidence-distribution merge. The method is
established inverse-information pooling; the optional thermodynamic notation is a naming
convention, not a general trust score or an automatic defense against confident agents.

## Preconditions

Use `reduce_partition` only when all are true:

- Workers target the same parameter and use compatible units.
- Evidence is disjoint and statistical dependence is negligible or modeled elsewhere.
- Each information matrix is a defensible uncertainty estimate.
- A regular asymptotic Gaussian approximation is suitable, or the model is exactly quadratic.
- Exact model-specific sufficient-statistic aggregation is unavailable or inappropriate.

Do not multiply factors merely because workers ran in separate sandboxes. Isolation does
not imply independence. Repeated non-empty `evidence_ids` are rejected by default, but
distinct identifiers are not proof of independence; inspect `lineage` too.

## API

```python
from bmr import map_mean, map_linreg, reduce_partition, byzantine_clip

cds = [map_mean(x_k, shard_id=k) for k, x_k in enumerate(shards)]
pooled = reduce_partition(cds)

pooled.theta                 # information-weighted mode
pooled.cov                   # inverse summed precision
pooled.neg_log_Z             # closed-form -log integral of unit-height factors
pooled.disagreement_energy   # Delta/2, the minimum summed quadratic energy
pooled.weights               # scalar precision shares only when p == 1
pooled.evidence_ids          # canonicalized evidence labels
pooled.lineage               # order-preserving lineage union
```

A `CD` carries `theta_hat`, per-observation `info_per_obs`, positive `n`, `meta`,
`lineage`, and `evidence_ids`. Construction rejects malformed or non-positive-definite
summaries.

For streaming or tree reduction, convert workers with
`CanonicalSummary.from_cd(cd)`, merge the summaries, and call
`finalize_canonical(summary)`. The same evidence-overlap guard applies.

## Thermodynamic convention

For `P_k = n_k J_k`, the Gaussian kernel is

```text
g_k(theta) = exp(-1/2 (theta-theta_hat_k)' P_k (theta-theta_hat_k))
           = exp(-beta_k E_k(theta)), beta_k = n_k.
```

`beta = n` follows from defining energy per observation. The invariant object is total
precision `P_k`, not sample size alone. The reducer adds Gaussian natural parameters;
this is standard confidence-distribution/inverse-covariance pooling.

## Outlier heuristic

`byzantine_clip(cds)` caps unusual precision volume and applies a redescending,
coordinate-scale-aware multivariate location weight. It returns clipped copies and
flagged IDs without mutating inputs.

This function is a stress-test heuristic, not a Byzantine guarantee. It is not rotation
invariant and has not been evaluated against collusion, adaptive attacks, or genuine
cross-worker heterogeneity.

## Demo

```bash
uv sync --locked
uv run --locked python scripts/verify_e2e.py
uv run --locked python demo.py --scenario mean --byzantine
uv run --locked python demo.py --scenario logistic
```

The local backend uses a platform-dependent Python process pool. The islo backend can
restore workers from a named snapshot when authenticated.

## Forward-looking use with agents

Agent-reported confidence is not automatically calibrated. For agent workloads, derive
uncertainty from external evidence such as held-out tests, repeated execution, or an
independent verifier. Shared model weights, prompts, repositories, tests, and fork
ancestors create correlation; use lineage and evidence identifiers to avoid double
counting. Correlation-aware reduction over a fork DAG remains future research.
