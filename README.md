# Boltzmann MapReduce

**Distributed statistical inference as Gibbs-ensemble computation over forkable sandboxes.**

The reduce step of classic MapReduce is an arithmetic aggregator — it sums, counts,
and concatenates, but cannot reason about sampling variability or efficiency. This
project reorganizes the model around one observation:

> To leading asymptotic order, the confidence density a statistical worker emits is a
> **Gibbs–Boltzmann measure** `h_k(θ) ∝ exp(−β_k · E_k(θ))` whose inverse temperature is
> the local sample size, **β_k = n_k**, with energy `E_k(θ) = ½(θ̂_k−θ)ᵀ J_k (θ̂_k−θ)`.

So distributed inference becomes the **equilibration of a statistical ensemble**
(*Boltzmann computing*): the unit of execution is not one deterministic path but an
ensemble of cheaply forked replicas, and the output is a *distribution* that collapses to
a point only in the cold (`n → ∞`) limit. The **reduce is a partition-function product**
of Gibbs factors — precision-weighted (inverse-variance) pooling — and frequentist
consistency is the zero-temperature limit.

This repo holds the **paper** and a **runnable demo** that makes the picture concrete.

## What it gives you

- A `map` that returns a *confidence density* (point estimate + per-observation
  information `J` + `n`), not a scalar.
- A `reduce` (`reduce_partition`) that multiplies Gibbs factors → precision-weighted
  pooling, recovering the full-data estimator's efficiency from summaries alone.
- The honest Byzantine story: precision-weighted pooling has an **unbounded influence
  function**, so a confident liar (tiny variance, far estimate) hijacks the consensus;
  `byzantine_clip` bounds reported precision and down-weights location outliers.
- Two execution backends behind one interface:
  - **`local`** — forks OS processes (runs today).
  - **`islo`** — *snapshot once, fork per shard* on [islo.dev](https://islo.dev)
    microVM sandboxes (the paper's substrate).

## Snapshot-once, fork-per-shard → islo primitives

```
parent sandbox  ──(islo snapshot save … --name bmr-base)──▶  warm snapshot
                                                                  │
        per shard k:  islo use bmr-shard-k --snapshot bmr-base    │  (fork)
                       -e SHARD_B64=… -- python3 -m bmr.worker  ◀─┘
                       └─▶ emits  __BMR__{confidence density}__BMR__
reduce: multiply Gibbs factors  ⇒  precision-weighted pooled CD
```

Each fork is a replica of the canonical ensemble; copy-on-write microVM forking is what
makes drawing the replicas cheap enough to treat the ensemble as the unit of computation.

## Quickstart (local backend)

All Python runs under [`uv`](https://docs.astral.sh/uv/) (no manual venv):

```bash
uv sync                                   # uv-managed Python + locked deps
uv run pytest -q                          # correctness tests (7/7)
uv run python demo.py --scenario mean --byzantine
uv run python demo.py --scenario linreg
```

Or just `./run.sh` (installs uv if missing). The local backend forks OS processes
directly — no AI harness needed.

## Run on islo (entry point: an on-box AI harness)

```bash
islo login
# build the parent: clones the repo; islo.yaml's setup_script installs uv and runs `uv sync`.
islo use bmr-base --source github://zozo123/boltzmann-mapreduce -- true
islo snapshot save bmr-base --name bmr-base               # snapshot once

# Each shard is forked from the snapshot; its ENTRY POINT is an on-box AI agent
# harness that runs the worker (the agentic map of the paper, made real):
uv run python demo.py --backend islo --scenario mean --harness claude
```

With `--harness claude`, each fork launches as
`islo use bmr-shard-k --snapshot bmr-base --workdir /workspace/boltzmann-mapreduce
-e SHARD_B64=… --agent claude --task "…"`, so the **entry point is an on-box AI harness**
(verified: islo reports `claude is ready in sandbox`). That harness is an *interactive*
agent session, not a one-shot pipe — it's the right entry point for a *reasoning* map (let
the agent pick/adapt the method), but the deterministic reduce we benchmark captures each
worker's confidence density from the direct `-- uv run python -m bmr.worker` path (the
"unless not needed" fallback). Omit `--harness` for that direct path.

**The substrate has been run for real on islo** (see `runs/islo_run.txt`): a 4-shard job
forked from the 141 MB uv-locked OCI snapshot reduced to a pooled mean of `4.942` vs the
full-data oracle `4.945` (gap `2.9e-3`), each fork dispatched in ~6.7 s (round-trip).
Because the workers are deterministic, the islo and local backends produce identical
statistical output; the islo run evidences the snapshot-once/fork-per-shard substrate
executing end-to-end. The `islo` backend falls back to `local` (with a notice) if the CLI is
unauthenticated or capacity is unavailable, so the demo always completes.

## Figures (written to `figs/`)

- **`fig_gibbs.png`** — per-shard Gibbs densities (hot replicas) collapsing into the sharp
  pooled density; the full-data oracle overlaps the pool.
- **`fig_cooling.png`** — pooled CI half-width vs cumulative `N`, tracking `∝ 1/√N`: the
  effective temperature `T ∝ 1/n → 0` as data accrues.
- **`fig_byzantine.png`** — a confident liar pulls naive pooling; `byzantine_clip`
  recovers the truth.

## Paper

`paper/` contains the LaTeX source and the compiled PDF:

- **`boltzmann-mapreduce.pdf`** — *Boltzmann MapReduce: A Partition-Function Reduce for
  Forkable Sandboxes* (≤5 pp body + refs). Built from `main_5pp.tex` + `body_5pp.tex`
  + `refs.bib`; the one figure is `figs/fig_gibbs.png`. Leads with measured results
  (local, plus real runs on two forkable-sandbox clouds, islo and Daytona); the snapshot
  layer is framed as OCI images.

Rebuild: `cd paper && pdflatex main_5pp && bibtex main_5pp && pdflatex main_5pp && pdflatex main_5pp`.
See `paper/REVIEWS.md` for the expert reviews that shaped the paper.

## Honest scope

The paper's reported systems/robustness numbers are **projections from published figures
and design targets, not measurements**. This demo computes the *statistics* exactly and
locally; the islo path is real when authenticated. The Gibbs identification is
leading-order (LAN); consistency assumes correct specification and a common parameter
across shards. See `PLAN.md` for what would make the empirical claims real.
