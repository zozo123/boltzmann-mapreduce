---
name: boltzmann-mapreduce
description: Use when an agent must combine a forked ensemble of noisy/uncertain results from parallel sandboxes or agent runs into a trustworthy pooled estimate with calibrated uncertainty, instead of naive averaging --- the partition-function (precision-weighted) reduce with a Byzantine clip for confident liars.
---

# Boltzmann MapReduce

A principled `reduce` for the outputs of many parallel, fire-and-forget runs. When you fan a job out across forked sandboxes/agents and each one emits a noisy estimate, naive averaging throws away the single most important signal: *how much each result should be trusted*. Boltzmann MapReduce treats every result as a **confidence density** and pools them by the partition-function (precision-weighted) reduce, recovering full-data efficiency from per-shard summaries alone --- with a Byzantine clip that stops one overconfident liar from hijacking the consensus.

Install as an agent skill via skills.sh:

```bash
npx skills add zozo123/boltzmann-mapreduce
```

Repo: https://github.com/zozo123/boltzmann-mapreduce · Directory: https://www.skills.sh/

## What this skill does

- Provides a `map` that returns a *confidence density* (`CD`: point estimate `theta_hat` + per-observation information `J` + sample size `n`), not a bare scalar.
- Provides a `reduce` (`reduce_partition`) that multiplies the per-shard Gibbs factors --- precision and energies add for disjoint shards --- yielding precision-weighted (inverse-variance) pooling that asymptotically equals the full-data (oracle) estimator. The pooled result carries a covariance and calibrated CIs.
- Provides `byzantine_clip`, because precision-weighted pooling has an **unbounded influence function**: a confident liar (tiny reported variance, far estimate) can dominate. The clip caps reported precision and down-weights location outliers, returning `(kept, flagged)` without mutating inputs.
- Ships numpy-only core (no heavy deps) with two execution backends behind one interface: `local` (forks OS processes, runs today) and `islo` (snapshot-once, fork-per-shard on islo.dev microVMs; falls back to `local` with a notice if unauthenticated).

## When an agent should use it

Reach for this whenever you are aggregating results from many parallel sandbox/agent runs and the results are noisy or carry uneven trust:

- Pooling **test pass-rates** or **build/patch success** across forked branches/hypotheses.
- Combining **eval scores** or **benchmark deltas** from independent runs.
- Averaging **coefficient / parameter estimates** (means, regressions) computed on disjoint data shards.

Use it instead of an arithmetic mean any time some runs saw more data, ran longer, or are more concentrated than others --- precision-weighting is the statistically correct combiner, and the clip is the defense when one fork may be wrong-but-confident.

## Core idea (one paragraph)

To leading asymptotic order the confidence density each worker emits is a Gibbs–Boltzmann measure `h_k(θ) ∝ exp(−β_k · E_k(θ))`, with inverse temperature equal to the local sample size `β_k = n_k` and energy `E_k(θ) = ½(θ̂_k−θ)ᵀ J_k (θ̂_k−θ)`, where `J_k` is the per-observation information so total shard precision is `n_k · J_k`. Combining independent shards is then a **partition-function product** of these Gibbs factors: precisions and information-weighted estimates add, which is exactly precision-weighted pooling, and frequentist consistency is the zero-temperature (`n → ∞`) limit. Because that reduce has unbounded influence, a **Byzantine clip** bounds each shard's reported precision and removes location outliers before pooling, so a confident liar cannot capture the result.

## Usage --- the `bmr` package API

```python
from bmr import map_mean, map_linreg, reduce_partition, byzantine_clip

# 1) MAP: each parallel run turns its data into a confidence density (CD)
cds = [map_mean(x_k, shard_id=k) for k, x_k in enumerate(shards)]
# regression shards:  map_linreg(X_k, y_k, shard_id=k)

# 2) (optional) Byzantine clip before reducing: cap confident liars, flag outliers
kept, flagged = byzantine_clip(cds, kappa=3.0)   # returns clipped copies + flagged ids

# 3) REDUCE: partition-function / precision-weighted pooling
pooled = reduce_partition(kept)
pooled.theta       # pooled estimate
pooled.ci()        # calibrated 95% CIs (precision-weighted)
pooled.weights     # each shard's share of total precision
```

`CD` carries `theta_hat`, `info_per_obs` (per-obs `J`), `n`, and `meta`; it exposes `cov()`, `se()`, `ci()`, `temperature()` (= 1/n), and `to_dict`/`from_dict` for serializing across sandboxes. `oracle_mean` / `oracle_linreg` give the full-data ceiling the reduce recovers.

### Demo

```bash
uv sync                                              # uv-managed Python + deps
uv run python demo.py --scenario mean --byzantine
# also: --scenario linreg --shards 16
# on islo, with an on-box AI harness as the entry point (requires `islo login`):
uv run python demo.py --backend islo --scenario mean --harness claude
```

The demo shards synthetic data with a known truth, maps each shard to a `CD`, reduces by partition-function pooling, prints a per-shard table (with temperature `T = 1/n` and weight%), compares pooled vs the full-data oracle, optionally injects one confident liar to show `byzantine_clip` recovering the truth, and writes `fig_gibbs.png`, `fig_cooling.png`, `fig_byzantine.png` to `figs/`.

## The vision (forward-looking thesis, not a measured result)

We have not yet figured out how to work in parallel, fire-and-forget sandboxes at scale. This is the new Hadoop-like moment --- the era of AI lambda servers. What is missing is a real theory and paradigm for **code evolution via snapshots**: what Hadoop/MapReduce did for distributed *data*, but for *coding and system-building*. Picture an army of AI coding agents building products in parallel: each snapshots a warm repo/environment (an OCI image), forks per task/branch/hypothesis, evolves code fire-and-forget, and is reaped. The **map** step (fork, run, isolate, evolve) is commoditizing fast. The **reduce** --- how you combine the noisy, uncertain outcomes those agents emit (test pass-rates, eval scores, build/patch successes, benchmark deltas) into a trustworthy decision --- has no principled theory yet.

Boltzmann MapReduce is a **first principled primitive** for that reduce: the outcome each agent/fork emits carries an implicit confidence (how much it ran, how concentrated, how trustworthy), and the right combination is the partition-function (precision-weighted) reduce with a Byzantine clip --- not an arithmetic average. "Distributed software engineering for AI coding agents" is the larger program this work opens: reinventing how software is built when the builders are agents and the machine is a forkable sandbox.

**Honesty guard.** The paper *measures* the statistical reduce (mean and regression, on both the `local` backend and a real `islo` run). The software-engineering-by-agents program above is an **outlook and research agenda we motivate, not something we benchmark** --- there are no SWE numbers here, and none are claimed.
