# Uncertainty-Aware Reduction for Forkable Sandboxes

**Boltzmann MapReduce** is a small reference implementation and research paper about
what parallel workers should return when their outputs are noisy: an estimate,
uncertainty, evidence provenance, and fork lineage rather than an unexplained scalar.

Explore the idea interactively in the
[live fusion explorer](https://zozo123.github.io/boltzmann-mapreduce/#explorer),
including false-precision and repeated-evidence failure modes, or read the
[corrected repository PDF](https://raw.githubusercontent.com/zozo123/boltzmann-mapreduce/main/docs/uncertainty-aware-reduction.pdf).

For disjoint evidence about a common parameter, the statistical core uses established
inverse-information/confidence-distribution pooling. Under local asymptotic normality,
the Gaussian factor can be written

```text
g_k(theta) = exp(-beta_k * E_k(theta))
beta_k     = n_k
E_k(theta) = 1/2 (theta_hat_k-theta)' J_k (theta_hat_k-theta)
```

when energy is defined per observation. `beta = n` is a useful thermodynamic
convention, not a new estimator; the invariant statistical quantity is total precision
`P_k = n_k * J_k`.

## What is new - and what is not

This project does **not** claim to invent inverse-variance pooling, Rao-type confidence
distributions, or MapReduce statistical inference. Its contributions are:

- An uncertainty- and provenance-carrying worker-result contract.
- An associative Gaussian canonical merge `(P, q, c, N)`.
- The exact Gaussian product normalizer, including worker-disagreement energy.
- A runnable snapshot-backed worker path and carefully scoped platform observations.
- A research agenda for lineage-aware reduction of correlated forked AI agents.

The key boundary is equally important: process or sandbox isolation does not make
worker evidence statistically independent.

## Result contract

A `CD` worker result carries:

- `theta_hat`: local estimate.
- `info_per_obs`: positive-definite per-observation information `J`.
- `n`: positive local sample size.
- `lineage`: fork ancestry identifiers.
- `evidence_ids`: identifiers for underlying observations/evaluations.
- `meta`: runtime and application metadata.

The reducer validates dimensions, finite values, positive sample size, provenance,
symmetry, and positive definiteness before combining summaries. Non-empty duplicate
`evidence_ids` are rejected by default; an explicit override exists for callers that
model overlap outside this package.

For `P_k = n_k J_k`, `q_k = P_k theta_hat_k`, and
`c_k = theta_hat_k' P_k theta_hat_k`, independent factors merge by addition. The
pooled mode is `P^-1 q`; covariance is `P^-1`; and the unnormalized product has

```text
-log Z = -p/2 log(2*pi) + 1/2 log|P| + 1/2 (c - q' P^-1 q).
```

The final term is exposed as `disagreement_energy`.

`CanonicalSummary.from_cd(cd)` and `CanonicalSummary.merge(...)` expose the same
associative operation for streaming or tree reduction. `finalize_canonical(...)`
converts a merged summary to a pooled result. Shared lineage is preserved, but it is
not interpreted as an independence model.

## Quickstart

All Python commands run from the checked-in [`uv`](https://docs.astral.sh/uv/)
lockfile. The first command below verifies tests, compilation, package builds, and every
local demo:

```bash
uv sync --locked
uv run --locked python scripts/verify_e2e.py --skip-paper
uv run --locked python demo.py --scenario mean --byzantine
uv run --locked python demo.py --scenario linreg
uv run --locked python demo.py --scenario logistic
```

The local backend uses Python's platform-dependent process pool. It does not claim
copy-on-write or `fork()` semantics.

## Snapshot-backed execution on islo

```bash
islo login
islo use bmr-base --source github://zozo123/boltzmann-mapreduce -- true
islo snapshot save bmr-base --name bmr-base
uv run --locked python demo.py --backend islo --scenario mean
```

The committed [`runs/islo_run.txt`](runs/islo_run.txt) records one four-shard execution
from a named 141 MB snapshot. It is an execution-path existence proof, not an isolated
microVM-restore benchmark.

The Daytona and Tensorlake SDKs are optional extras resolved in `uv.lock`; no mutable
per-provider installation step is needed:

```bash
uv run --locked --extra daytona python scripts/daytona_fanout.py
uv run --locked --extra tensorlake python scripts/tensorlake_fanout.py
```

These scripts create default sandboxes and run a trivial command. They do **not**
restore the named islo snapshot, so their logs are platform-specific creation
observations rather than comparable snapshot-fork measurements. They require provider
credentials and may consume quota, so E2E checks compile the scripts and CI imports the
locked SDKs but does not execute provider sweeps.

## Robustness scope

Gaussian precision pooling is not robust: even a fixed-precision location outlier has
unbounded influence, and forged high precision adds another attack surface.
`byzantine_clip` is retained as a transparent stress-test heuristic. It now handles all
parameter coordinates, caps relative precision volume, and applies a redescending
multivariate location weight, but it is not a certified Byzantine defense.

Do not multiply factors when workers reuse evidence, share correlated noise, or target
different parameters. In those settings use a correlation-aware or hierarchical model.

## Paper

The revised paper is:

> **Uncertainty-Aware Reduction for Forkable Sandboxes: A Thermodynamic View of
> Confidence-Distribution Pooling**

Sources are in `paper/`. The uv-driven builder isolates TeX intermediates in a temporary
directory, rejects release-blocking warnings, and publishes identical PDF copies to
`output/pdf/` and `docs/`:

```bash
uv run --locked python scripts/build_paper.py
```

The complete release check uses that builder after tests, compilation, and all local
demos. It requires `pdflatex` and `bibtex` on `PATH`:

```bash
uv run --locked python scripts/verify_e2e.py
```

CI runs the Python path on Python 3.10 and 3.12 and runs this complete release check in
the paper job.

## Evidence boundaries

| Component | What the repository establishes | What it does not establish |
|---|---|---|
| Statistical core | Gaussian algebra, exact normalizer, schema validation | General finite-sample efficiency |
| Logistic check | Expected direction under severe IID size imbalance | Non-IID robustness or baseline superiority |
| Outlier heuristic | One scalar stress test and multivariate regression test | Byzantine tolerance |
| islo | One named-snapshot end-to-end execution | In-host restore latency advantage |
| Daytona/Tensorlake | Batched default sandbox creation | Shared-snapshot restore performance |
| Provenance | Literal evidence-ID overlap rejection and lineage transport | Correlation-aware inference |

See [`PLAN.md`](PLAN.md) for the experiments required for a full systems/statistics paper.

## License status

No software license file is currently present. Public visibility does not itself grant
reuse rights; select and add a code license before presenting the repository as open
source. The arXiv paper's document license does not automatically license this code.
