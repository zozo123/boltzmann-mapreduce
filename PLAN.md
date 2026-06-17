# Plan & roadmap

## Where this stands

| Component | Status |
|---|---|
| Paper (concise ≤5pp + full) | Done; 5 expert reviews applied (`paper/REVIEWS.md`) |
| `bmr.core` (CD, reduce, Gibbs view, Byzantine clip) | Done; 6 passing tests |
| Local backend (process fork) | Done; runs the full demo |
| islo backend (snapshot + fork) | Done; reaches real allocation, graceful fallback |
| Figures (Gibbs collapse, cooling, Byzantine) | Done |

## What the demo proves vs. what is still a projection

**Proven, locally and exactly:**
- `reduce_partition` = partition-function product = precision-weighted pooling
  (matches the closed-form inverse-variance estimator to machine precision).
- Pooled estimate is asymptotically equivalent to the full-data oracle (efficiency
  recovered), for both the mean and linear-regression scenarios.
- `β = n` as inverse temperature; CI half-width cools as `∝ 1/√N`.
- Precision-weighted pooling is *not* Byzantine-robust; clipping bounds a confident liar.

**Still a projection (would require real experiments):**
- The latency table (3–8 ms restore, sub-20 ms p99) — needs microbenchmarks of islo
  snapshot/restore for warmed Python working sets.
- The robustness table (mid-90% accuracy under 30% Byzantine, ~⅓ comms) — needs a real
  poisoning experiment with a defined attack model, dataset split, and model.

## Next steps (in priority order)

1. **Real islo latency benchmark.** Measure `islo snapshot save` + `islo use --snapshot`
   round-trip for the bmr base across fanouts (1, 10, 100), warmed NumPy resident set.
   Replace the projected latency row with measured numbers (and keep the caveat where it
   still applies).
2. **Real Byzantine experiment.** Fix an attack model (sign-flip / label-flip / confident
   liar), sweep the Byzantine fraction, and report clipped-pooling accuracy vs Krum/FLAME
   on a *matched* protocol — or clearly keep it as a design target.
3. **Specify the detection layer.** The Shapley-based detector is currently a heuristic;
   define the value function, the game, and the discard/clip threshold, then evaluate.
4. **Heterogeneous-truth handling.** Today consistency assumes `θ_{k,0} = θ_0`. Add a
   test for / treatment of genuine cross-shard heterogeneity (the pooled target becomes a
   precision-weighted average of differing truths).
5. **Higher-order CD corrections.** The Gibbs identity is leading-order; add the
   `O(n^{-1/2})` skewness/curvature term a confidence distribution captures.
6. **Live branching demo.** Use islo fork-at-a-decision-node to test competing
   hyperparameterizations in parallel and keep the low-energy branch (simulated annealing
   over the agent's hypothesis landscape).

## Repo layout

```
bmr/            core stats, worker entrypoint, backends, figures
demo.py         CLI orchestrator (local | islo; mean | linreg; --byzantine)
tests/          correctness tests (pytest)
paper/          LaTeX sources + compiled PDFs + REVIEWS.md
figs/           generated figures
islo.yaml       islo sandbox config (pip install -r requirements.txt)
run.sh          one-shot: venv + tests + demo
```
