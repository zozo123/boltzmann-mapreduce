# Paper release checklist

Status: repository candidate for the next arXiv revision

Prepared: 17 July 2026
Release parent: `83b694974e2e2381d09ce8b01393fd8dd6a5d939`

## Candidate manuscript

- Title: **Evidence-Aware MapReduce for Forkable Compute: Separating
  Execution Fan-Out from Independent Evidence**
- Canonical submission PDF: `submission/evidence-aware-reduction.pdf`
- Pages: 5
- SHA-256: `222e358f06c595cd6751e0c0a730aab78938ea2ce8bed989cb407041bdead43e`
- Size: 409,112 bytes
- Reproducible metadata epoch: 17 July 2026 UTC
- Canonical and historical PDF paths: byte-identical

## Reproducibility hashes

| Artifact | SHA-256 |
|---|---|
| `uv.lock` | `fe8195b97d834687ed1614ddafb8ce35fc10877c2de9ae9397873ff766d1995a` |
| `runs/daytona_fanout.txt` | `31403ed242ddc38e525b687f6892cfa3f4492ac06fd23a1bba02b61701af962a` |
| `runs/tensorlake_fanout.txt` | `093b3f4b35e15829f659541799b09a5c6844219a4a838b449c083afc3b32839e` |
| `runs/fanout_sweep.txt` | `17ff046e5d5eba2f4fdbeeaffcf8aeafa376979a7f72bac129920a9873285439` |
| `runs/islo_run.txt` | `baae024c0d8b77b39a018675d955e06046847726d594a1c85b3725f396817133` |
| `runs/logistic_sweep.txt` | `608d2c3b372219fa08537a419d59ebed535a5faa2f58ab6cbf0aa576fc998d02` |

## Repository checks

- [x] Repository title, abstract, equations, claims, bibliography, README, and
  website agree on the candidate revision.
- [x] Numeric commutativity and provenance semantics are stated separately.
- [x] Evidence IDs and lineage reach the finalized pooled result.
- [x] The platform table matches committed raw logs and labels operation boundaries.
- [x] $\Delta$ is derived as a residual heterogeneity statistic and attributed to
  Cochran's $Q$ in one dimension.
- [x] The TeX build completes without overfull boxes, undefined citations, or
  undefined references.
- [x] All tracked PDF aliases have the checksum above.

## External release steps

- [ ] Upload the flat `submission/` package (`main.tex`, `body.tex`,
  `fig_contract.tex`, and `refs.bib`) as the next arXiv version.
- [ ] Verify the public title and abstract match this candidate.
- [ ] Verify the platform table retains its operation boundaries, concurrency,
  batch size, success count, p50/p95, and total wall time.
- [ ] Verify the product-normalizer equation includes the residual
  heterogeneity term `exp(-Delta/2)`.
- [ ] Verify the robustness language describes the stress clip as a heuristic with
  explicitly limited scope.
- [ ] Verify the bibliography includes the cited statistical, systems, and AI
  literature.
- [ ] Verify the artifact URL resolves to the final commit/tag, then record both
  identifiers in the release notes.
- [ ] Verify the rendered PDF is five pages, dated 17 July 2026, and has SHA-256
  `222e358f06c595cd6751e0c0a730aab78938ea2ce8bed989cb407041bdead43e`.
- [ ] Replace the website's candidate-version note after the public record updates.
- [ ] After the public record updates, create a paper tag and record the final commit
  and tag in the GitHub release notes.

As of 17 July 2026, arXiv:2607.09689 v2 (14 July 2026) still exposes the older
title and thermodynamic abstract. That upload is the only remaining
claim-synchronization step outside this repository.
