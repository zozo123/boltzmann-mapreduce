# Evidence-Aware Reduction for Forkable Sandboxes

Snapshot-backed sandboxes make execution fan-out cheap. They do not make worker
outputs independent evidence. Branches can share a model, prompt, repository,
tests, observations, or ancestor; reducing them as independent votes can turn one
repeated error into high-confidence consensus.

This repository contains a small evidence-aware reducer, deterministic checks, raw
execution traces, and the paper:

- [Evidence-Aware Reduction for Forkable Sandboxes](https://raw.githubusercontent.com/zozo123/boltzmann-mapreduce/main/docs/evidence-aware-reduction.pdf)
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
provenance shape, symmetry, and positive definiteness. It rejects overlap among
nonempty `evidence_ids` unless the caller uses an explicit unsafe override.

> Evidence labels and lineage are transport fields, not verified identities or a
> correlation model. Empty IDs, different labels for the same data, shared noise,
> and mismatched estimands remain the caller's responsibility.

For total precision $P_k=n_kJ_k$, information vector
$q_k=P_k\hat\theta_k$, and quadratic constant
$c_k=\hat\theta_k^\top P_k\hat\theta_k$, numeric summaries merge by addition:

```text
(P, q, c, N) = sum_k (P_k, q_k, c_k, n_k)
theta_pool   = solve(P, q)
Sigma_pool   = inverse(P)
Delta        = c - q' solve(P, q)
```

The numeric state is associative and commutative in exact arithmetic. Provenance
has separate semantics: evidence IDs are canonicalized, while lineage uses
order-preserving deduplication. `finalize_canonical(...)` returns `evidence_ids`
and `lineage` with the pooled statistics; worker `meta` remains a worker-level
field.

$\Delta$ is the weighted residual heterogeneity statistic—Cochran's $Q$ in
the scalar inverse-variance case. The unit-height Gaussian product has

```text
-log Z = -p/2 log(2*pi) + 1/2 log|P| + 1/2 Delta.
```

The API field `disagreement_energy` stores $\Delta/2$. Cholesky solves and
log-determinants avoid explicit numerical inversion. A negative computed $\Delta$
is clipped only within a scale-aware roundoff tolerance; a larger violation is
rejected as an inconsistent canonical summary.

## What the artifact establishes

| Layer | Evidence in this repository | Boundary |
|---|---|---|
| Gaussian algebra | Closed-form, flat, and tree reductions agree | Common target and independent estimation noise |
| Record validation | Malformed inputs and exact repeated nonempty IDs are rejected | IDs are caller-supplied and unverified |
| Provenance | Evidence IDs and lineage reach the finalized pooled result | Lineage is not yet a fork-DAG covariance model |
| Logistic sanity check | Expected direction under severe IID size imbalance | Equal averaging is a deliberately weak baseline |
| Forged-precision trace | Vulnerability and a transparent stress-test clip | No Byzantine or affine-invariant guarantee |
| islo integration | One four-worker named-snapshot end-to-end trace | No isolated restore-speed claim |
| Provider sweeps | Three exercised API paths with raw timing logs | Different operation boundaries; no provider ranking |

## Quickstart

The locked local workflow is:

```bash
uv sync --locked
uv run --locked python -m pytest -q
uv run --locked python demo.py --scenario mean --byzantine
uv run --locked python demo.py --scenario linreg
uv run --locked python demo.py --scenario logistic
```

The local backend uses Python's platform-dependent process pool and makes no
copy-on-write or guaranteed `fork()` claim.

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

They require provider credentials and may consume quota. The automated local
workflow imports their locked SDKs but does not execute cloud sweeps.

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

The paper builder uses a temporary TeX tree, rejects release-blocking warnings,
and publishes byte-identical canonical and compatibility copies:

```bash
uv run --locked python scripts/build_paper.py
```

Canonical outputs:

- `submission/evidence-aware-reduction.pdf`
- `output/pdf/evidence-aware-reduction.pdf`
- `docs/evidence-aware-reduction.pdf`

All arXiv inputs are flat under `submission/`: `main.tex`, `body.tex`,
`fig_contract.tex`, and `refs.bib`. There are no nested figure or source
directories.

Historical PDF paths are retained as byte-identical aliases so existing links do
not serve stale manuscripts. The full local release workflow is:

```bash
uv run --locked python scripts/verify_e2e.py
```

Cloud-provider sweeps are excluded from that command.

## Public paper status

The repository paper is the candidate next revision titled **Evidence-Aware
Reduction for Forkable Sandboxes: Separating Execution Fan-Out from Independent
Evidence**.

The public [arXiv:2607.09689](https://arxiv.org/abs/2607.09689) record was last
revised as v2 on 14 July 2026 and still uses the earlier title and thermodynamic
abstract. Uploading this repository revision as the next arXiv version is the one
remaining external synchronization step. See
[`paper/RELEASE_CHECKLIST.md`](paper/RELEASE_CHECKLIST.md).

## License status

No software license file is currently present. Public visibility does not grant
reuse rights; select a code license before presenting the implementation as open
source. The arXiv document license does not automatically license the code.
