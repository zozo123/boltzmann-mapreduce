# Evidence-Aware MapReduce for Forkable Compute: Separating Execution Fan-Out from Independent Evidence

Forkable compute can launch many workers from one snapshot. Those workers often
inherit the same model, prompt, repository, tests, observations, or ancestor. A
reducer therefore needs to know where each answer came from and how much evidence
supports it. Otherwise, one repeated mistake can look like confident agreement.

This repository pairs a small evidence-aware reducer with deterministic checks,
raw execution traces, and the paper:

- [Evidence-Aware MapReduce for Forkable Compute: Separating Execution Fan-Out from Independent Evidence](https://raw.githubusercontent.com/zozo123/boltzmann-mapreduce/main/docs/evidence-aware-reduction.pdf)
- [Interactive fusion explorer](https://zozo123.github.io/boltzmann-mapreduce/#explorer)

## Worker contract

A `CD` worker record carries:

- `theta_hat`: local estimate.
- `info_per_obs`: positive-definite per-observation information $J_k$.
- `n`: positive literal sample count.
- `evidence_ids`: caller-supplied identifiers for observations or evaluations.
- `lineage`: caller-supplied fork ancestry.
- `meta`: worker-level execution and application metadata.

The implementation validates dimensions, finite values, integer sample size,
ID and lineage shape, symmetry, and positive definiteness. It rejects overlap among
nonempty `evidence_ids` unless the caller uses an explicit unsafe override.

> Evidence IDs and lineage are declarations supplied by the caller. They make
> repeated identifiers auditable. The application must establish reliable identity,
> account for hidden data reuse and shared noise, and ensure that workers estimate
> the same target.

For total precision $P_k=n_kJ_k$, information vector
$q_k=P_k\hat\theta_k$, and quadratic constant
$c_k=\hat\theta_k^\top P_k\hat\theta_k$, numeric summaries merge by addition:

```text
(P, q, c, N) = sum_k (P_k, q_k, c_k, n_k)
theta_pool   = solve(P, q)
Sigma_pool   = inverse(P)
Delta        = c - q' solve(P, q)
```

The numeric state is associative and commutative in exact arithmetic. Evidence IDs
are sorted into a stable order, while lineage keeps the first occurrence of each
ancestor. `finalize_canonical(...)` returns both fields with the pooled statistics.
Worker `meta` stays attached to its original worker record.

$\Delta$ is the weighted residual heterogeneity statistic—Cochran's $Q$ in
the scalar inverse-variance case. For Gaussian factors whose peak value is one,

```text
-log Z = -p/2 log(2*pi) + 1/2 log|P| + 1/2 Delta.
```

The API field `disagreement_energy` stores $\Delta/2$. The implementation uses
Cholesky solves and log-determinants for numerical stability. It clips a negative
computed $\Delta$ only when the value falls within a scale-aware roundoff
tolerance; a larger violation signals an inconsistent summary and raises an error.

## What the artifact establishes

| Layer | Evidence in this repository | Boundary |
|---|---|---|
| Gaussian algebra | Closed-form, flat, and tree reductions agree | Common target and independent estimation noise |
| Record validation | Malformed inputs and exact repeated nonempty IDs are rejected | IDs are caller-supplied and unverified |
| Provenance | Evidence IDs and lineage reach the finalized pooled result | Applications must supply any covariance model implied by lineage |
| Logistic sanity check | Expected direction under severe IID size imbalance | Equal averaging is a deliberately weak baseline |
| Forged-precision trace | Vulnerability and a transparent stress-test clip | The trace covers this case; broader robustness guarantees need a different reducer |
| islo integration | One four-worker named-snapshot end-to-end trace | Timing covers the full restore–run–capture path |
| Provider sweeps | Three exercised API paths with raw timing logs | Operation boundaries differ, so each timing characterizes its own path |

## Quickstart

The locked local workflow is:

```bash
uv sync --locked
uv run --locked python -m pytest -q
uv run --locked python demo.py --scenario mean --byzantine
uv run --locked python demo.py --scenario linreg
uv run --locked python demo.py --scenario logistic
```

The local backend uses Python's process pool. Its `fork()` and copy-on-write
behavior varies by platform.

## Named-snapshot execution

The documented islo path prepares a named environment before normal workers
restore from it:

```bash
islo login
islo use bmr-base --source github://zozo123/boltzmann-mapreduce -- true
islo snapshot save bmr-base --name bmr-base
uv run --locked python demo.py --backend islo --scenario mean
```

[`runs/islo_run.txt`](runs/islo_run.txt) records one four-shard execution from a
named 141 MB snapshot. The recorded 6.70 s is client-observed
restore–run–capture round-trip time.

Optional Daytona and Tensorlake traces exercise default sandbox creation:

```bash
uv run --locked --extra daytona python scripts/daytona_fanout.py
uv run --locked --extra tensorlake python scripts/tensorlake_fanout.py
```

These commands require provider credentials and may consume quota. The automated
local workflow verifies the locked SDK imports; cloud sweeps run only through the
explicit commands above.

| Platform | State source | Measured operation | Client concurrency | Batch | Success | p50 | p95 | Total wall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Daytona | default environment | create | 8 | 1024 | 1024/1024 | 0.20 s | 1.12 s | 190.3 s |
| Tensorlake | default environment | create | 8 | 256 | 256/256 | 3.44 s | 9.00 s | 387.3 s |
| islo | named 141 MB snapshot | restore + run + capture | 12 | 256 | 255/256 | 6.87 s | 9.04 s | 240.3 s |

Daytona and Tensorlake p50/p95 values are create latency; their total wall time
covers the complete create–run–delete batch. The islo percentiles cover
restore–run–capture. Failed higher-concurrency exploratory sweeps remain in the
raw logs as capacity observations.

## Build the paper

The paper builder uses a temporary TeX tree, stops on release-blocking warnings,
and publishes identical bytes under the current and legacy filenames:

```bash
uv run --locked python scripts/build_paper.py
```

Canonical outputs:

- `submission/evidence-aware-reduction.pdf`
- `output/pdf/evidence-aware-reduction.pdf`
- `docs/evidence-aware-reduction.pdf`

All arXiv inputs sit directly under `submission/`: `main.tex`, `body.tex`,
`fig_contract.tex`, and `refs.bib`.

Historical PDF paths serve the same bytes, which keeps existing links current.
The full local release workflow is:

```bash
uv run --locked python scripts/verify_e2e.py
```

Cloud-provider sweeps stay separate and require the explicit commands shown above.

## Public paper status

The repository paper is the candidate next revision titled **Evidence-Aware
MapReduce for Forkable Compute: Separating Execution Fan-Out from Independent
Evidence**.

The public [arXiv:2607.09689](https://arxiv.org/abs/2607.09689) record was last
revised as v2 on 14 July 2026 and still uses the earlier title and thermodynamic
abstract. The repository and arXiv records will match after this revision is
uploaded as the next arXiv version. See
[`paper/RELEASE_CHECKLIST.md`](paper/RELEASE_CHECKLIST.md).

## License status

Default copyright currently governs the code because the repository has no
software license. Choose a code license before describing the implementation as
open source. The code needs its own license in addition to the arXiv document
license.
