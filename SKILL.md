---
name: boltzmann-mapreduce
description: Use for combining independent, non-overlapping statistical summaries that target the same parameter and carry estimates plus valid uncertainty. Do not use for correlated agent votes, overlapping evidence, different local truths, or adversarial workers without a separately justified model.
---

# Boltzmann MapReduce

Use this package when independent workers return estimates with unequal information and
you need an auditable Gaussian/confidence-distribution merge. The method is established
inverse-information pooling expressed through a thermodynamic convention; it is not a
general trust score or an automatic defense against confident agents.

## Preconditions

Use `reduce_partition` only when all are true:

- Workers target the same parameter and use compatible units.
- Evidence is disjoint and statistical dependence is negligible or modeled elsewhere.
- Each information matrix is a defensible uncertainty estimate.
- Local Gaussian/LAN approximation is suitable, or the model is exactly quadratic.
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
pooled.neg_log_Z             # exact -log integral of unnormalized factors
pooled.disagreement_energy   # minimum summed quadratic energy
pooled.weights               # scalar precision shares only when p == 1
```

A `CD` carries `theta_hat`, per-observation `info_per_obs`, positive `n`, `meta`,
`lineage`, and `evidence_ids`. Construction rejects malformed or non-positive-definite
summaries.

For streaming or tree reduction, convert workers with
`CanonicalSummary.from_cd(cd)`, merge the summaries, and call
`finalize_canonical(summary)`. The same evidence-overlap guard applies.

## Thermodynamic convention

For `P_k = n_k J_k`, the Gaussian factor is

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
uv sync
uv run pytest -q
uv run python demo.py --scenario mean --byzantine
uv run python demo.py --scenario logistic
```

The local backend uses a platform-dependent Python process pool. The islo backend can
restore workers from a named snapshot when authenticated.

## Forward-looking use with agents

Agent-reported confidence is not automatically calibrated. For agent workloads, derive
uncertainty from external evidence such as held-out tests, repeated execution, or an
independent verifier. Shared model weights, prompts, repositories, tests, and fork
ancestors create correlation; use lineage and evidence identifiers to avoid double
counting. Correlation-aware reduction over a fork DAG remains future research.
