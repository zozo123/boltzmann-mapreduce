---
name: boltzmann-mapreduce
description: Use for evidence-aware MapReduce over independent, non-overlapping statistical summaries that target the same parameter and carry estimates plus valid uncertainty. Correlated votes, overlapping evidence, different local truths, and adversarial workers require a separately justified model.
---

# Evidence-Aware MapReduce for Forkable Compute

Use this package when independent workers return estimates with unequal information and
you need an evidence-aware Gaussian/confidence-distribution merge. The method is
established inverse-information pooling; the optional thermodynamic notation is a naming
convention. The evidence contract makes trust assumptions explicit, and robustness
requires a separately justified defense.

## Preconditions

Use `reduce_partition` only when all are true:

- Workers target the same parameter and use compatible units.
- Evidence is disjoint and statistical dependence is negligible or modeled elsewhere.
- Each information matrix is a defensible uncertainty estimate.
- A regular asymptotic Gaussian approximation is suitable, or the model is exactly quadratic.
- Exact model-specific sufficient-statistic aggregation is unavailable or inappropriate.

Multiply factors after establishing that their evidence is independent. Separate
sandboxes provide execution isolation, while `evidence_ids` and `lineage` record declared
evidence boundaries and ancestry. The reducer rejects repeated non-empty `evidence_ids`
by default; callers should also validate distinct identifiers and inspect shared lineage.

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

`beta = n` follows from defining energy per observation. Total precision `P_k` is the
invariant quantity; sample size contributes through the chosen decomposition. The reducer
adds Gaussian natural parameters; this is standard confidence-distribution/inverse-covariance
pooling.

## Outlier heuristic

`byzantine_clip(cds)` caps unusual precision volume and applies a redescending,
coordinate-scale-aware multivariate location weight. It returns clipped copies and
flagged IDs while preserving the inputs.

This function is a stress-test heuristic with coordinate-dependent behavior. Its current
evaluation covers the supplied attack scenario; collusion, adaptive attacks, genuine
cross-worker heterogeneity, and certified Byzantine robustness remain open work.

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

For agent workloads, calibrate uncertainty from external evidence such as held-out tests,
repeated execution, or an independent verifier. Shared model weights, prompts,
repositories, tests, and fork ancestors create correlation; lineage and evidence
identifiers help detect repeated evidence. Correlation-aware reduction over a fork DAG
remains future research.
