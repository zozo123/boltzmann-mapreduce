# Review history and revision decisions

## Revision standard

The current paper was rewritten against four demanding review lenses: distributed
systems, mathematical statistics, agent research, and adversarial robustness. The goal
was not to make every dimension look complete; it was to make every claim traceable to
the artifact or explicitly mark it as a research agenda.

| Lens | Blocking concern | Revision |
|---|---|---|
| Distributed systems | MapReduce mischaracterized; no controlled substrate comparison | Reduce is correctly described as user-defined; cloud operations and timing boundaries are separated |
| Statistical novelty | Rao-CD and inverse-information pooling predate the work | Prior work is credited in the abstract, introduction, theory, and bibliography |
| Mathematical correctness | Exactness too broad; partition normalizer incomplete | Exactness narrowed; full disagreement term derived, implemented, and tested |
| Robustness | Exponential suppression claim false; scalar clip missed other coordinates | Claim removed; multivariate heuristic added and explicitly scoped as non-certified |
| Heterogeneity | Unequal shard sizes presented as non-IID heterogeneity | Renamed unequal-size IID; common-target assumption made explicit |
| Systems evidence | Daytona/Tensorlake creates labeled snapshot forks; islo rounded to 100% | Operations corrected; islo reported as 255/256; batched totals distinguished from concurrency |
| Local backend | Process pool described as guaranteed `fork()`/CoW | Wording corrected; no CoW claim |
| Agent relevance | Fork isolation treated as statistical independence | Literal evidence reuse is rejected; evidence IDs and lineage reach the pooled result; correlation-aware reduction remains future work |
| Artifact/API | Associativity stated but only a flat loop exposed | Public canonical summary, tree merge, provenance-carrying finalization, and algebra/semantics tests added |
| Reproducibility | No automated verification | CI now tests two Python versions and rebuilds the paper |

## Claims intentionally removed

- A new statistical estimator or identity.
- Robustness through "Boltzmann suppression."
- A Byzantine defense or accuracy target.
- Three-cloud shared-snapshot equivalence.
- Single-digit microVM restore projections.
- Full-data efficiency based on one point-estimate gap.
- A measured AI-agent or software-engineering result.

## Remaining limitations

- No repeated-sampling coverage study or serious distributed-inference baseline suite.
- No genuine non-IID/random-effects evaluation.
- No controlled process/container/cold-VM/snapshot benchmark.
- No formal robust-aggregation guarantee.
- Evidence and lineage fields reach the finalized result but are not yet used in a
  correlation model; distinct
  evidence IDs are not proof of independence.
- Only one named-snapshot end-to-end statistical execution is committed.

These limitations are now part of the paper's scientific boundary rather than being
hidden behind projected tables.

## Issue #2 resolution (17 July 2026)

| Critical review item | Resolution |
|---|---|
| End-to-end lineage claim | `Pooled` now returns evidence IDs and lineage; worker metadata remains worker-local |
| Commutativity claim | Paper, README, website, docstring, and tests distinguish commutative numeric state from order-sensitive lineage |
| Platform table | Added state source, measured boundary, client concurrency, batch, success, p50, p95, and total wall time |
| Heterogeneity statistic | Derived the weighted-residual identity, identified scalar Cochran $Q$, stated calibration, and added classical citations |
| Public version synchronization | Repository sources and every tracked PDF alias are synchronized; arXiv v2 still requires an external next-version upload |

The rewrite also centers evidence-aware reduction, replaces the loose JSON-schema
claim with a serialized-record contract, uses regular asymptotic normality rather
than calling an estimator CLT “LAN,” and adds a compact related-work section.
