# Traffic-light reviews

The concise paper was hardened by five expert-persona reviews. All returned
**🟡 yellow** (accept with fixes); every blocking issue below was applied.

| Reviewer (lens) | Verdict | Headline blocking issue (fixed) |
|---|---|---|
| **Jeff Dean** — distributed systems at scale | 🟡 | Number provenance mismatched; KVM-isolation vs nanosecond-IPC contradiction; novelty over-claimed vs parameter servers / one-shot estimation / bag-of-little-bootstraps |
| **Mathematical statistician** (Efron/Reid/Meng) | 🟡 | Efficiency overclaim — Rao-CD *recovers*, never *exceeds*, the full-data oracle; `J` must be per-observation for `β=n`; `exactly` → `to leading order under LAN` |
| **OS / virtualization** | 🟡 | It's a snapshot/UFFD restore, **not** a `fork()` of a live parent; the 2× single-address-space throughput claim was uncited |
| **ML-security / federated learning** | 🟡 | Precision-weighted pooling is **not** Byzantine-robust (confident liar gets *more* weight); SHAP is attribution, not a defense; FLAME comparison not head-to-head |
| **Scientific-writing / integrity editor** | 🟡 | Several figures read as original measurements; stripped false-precision table numbers; framed all numbers as projected/illustrative |

## Key changes applied

- Removed every "improves on / exceeds the full-data benchmark" claim; reduce is now
  stated as asymptotically first-order efficient, *recovering* the oracle, strictly
  dominating only naive equal-weight pooling.
- Defined `J_{n_k}` as per-observation (averaged) information so total precision is
  `n_k J_{n_k}` and `β_k = n_k` is a genuine inverse temperature.
- Softened the Gibbs identity to leading-order/LAN (exact only in the Gaussian case).
- Rewrote the microVM clone path as snapshot + UFFD demand paging, not a live `fork()`.
- Scoped the nanosecond zero-copy hand-off to a co-located single-address-space tier that
  trades the KVM boundary for software-fault isolation (with the side-channel cost noted).
- Reframed the robustness story: thermal suppression handles honest outliers; a separate
  detection layer is required for confident adversaries.
- Reframed both tables as illustrative design-target profiles; stripped false precision;
  added real prior-art citations (parameter server, Zhang–Duchi–Wainwright, CSL,
  bag-of-little-bootstraps, Faasm, Morph).

## Residual risks (carried forward in `PLAN.md`)

- Robustness/latency numbers remain projections, not experiments.
- The Gibbs identity is leading-order only.
- The Shapley detection layer is an unvalidated heuristic.
- Consistency assumes a common parameter `θ_0` across shards.
