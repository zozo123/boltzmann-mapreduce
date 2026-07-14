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
| Agent relevance | Fork isolation treated as statistical independence | Literal evidence reuse is rejected; lineage is preserved; correlation-aware reduction remains future work |
| Artifact/API | Associativity stated but only a flat loop exposed | Public canonical summary, tree merge, finalization, and associativity test added |
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
- Lineage fields are transported but not yet used in a correlation model; distinct
  evidence IDs are not proof of independence.
- Only one named-snapshot end-to-end statistical execution is committed.

These limitations are now part of the paper's scientific boundary rather than being
hidden behind projected tables.
