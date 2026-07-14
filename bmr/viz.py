"""Figures that make the Boltzmann picture visible."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bmr.core import CD, reduce_partition  # noqa: E402


def fig_gibbs(cds: list[CD], pooled, oracle: CD, true_theta: float, path: str) -> str:
    """Per-shard Gibbs densities collapsing onto the pooled / oracle density (p == 1)."""
    lo = min([c.theta_hat[0] for c in cds] + [true_theta]) - 3
    hi = max([c.theta_hat[0] for c in cds] + [true_theta]) + 3
    grid = np.linspace(lo, hi, 800)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c in cds:
        ax.plot(grid, c.density1d(grid), color="0.7", lw=1.0, zorder=1)
    ax.plot([], [], color="0.7", lw=1.0, label=r"per-shard $h_k\propto e^{-\beta_k E_k}$ ($\beta_k=n_k$)")
    pooled_var = pooled.cov[0][0]
    pooled_d = np.exp(-0.5 * (grid - pooled.theta[0]) ** 2 / pooled_var) / np.sqrt(2 * np.pi * pooled_var)
    ax.plot(grid, pooled_d, color="C0", lw=2.6, zorder=3, label="information-weighted product")
    ax.plot(grid, oracle.density1d(grid), color="C3", lw=1.8, ls="--", zorder=2, label="full-sample estimate")
    ax.axvline(true_theta, color="k", lw=1.0, ls=":", label=r"true $\theta$")
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel("confidence density")
    ax.set_title("Independent Gaussian factors combine into a sharper product")
    ax.legend(fontsize=8, loc="upper right"); fig.tight_layout()
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def fig_cooling(cds: list[CD], path: str, seed: int = 0) -> str:
    """CI half-width vs cumulative N as shards accrue --- the T = 1/n cooling curve."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cds))
    Ns, widths = [], []
    acc: list[CD] = []
    for idx in order:
        acc.append(cds[idx])
        pooled = reduce_partition(acc)
        lo, hi = pooled.ci()[0]
        Ns.append(pooled.n_total); widths.append((hi - lo) / 2.0)
    Ns = np.array(Ns, dtype=float); widths = np.array(widths)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.loglog(Ns, widths, "o-", color="C0", label="pooled 95% CI half-width")
    ref = widths[0] * np.sqrt(Ns[0]) / np.sqrt(Ns)
    ax.loglog(Ns, ref, "--", color="0.5", label=r"$\propto 1/\sqrt{N}$ reference")
    ax.set_xlabel("cumulative samples $N$ (shards accrued)")
    ax.set_ylabel("CI half-width")
    ax.set_title(r"Concentration: interval width tracks $1/\sqrt{N}$")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def fig_byzantine(true_theta: float, naive_pooled, robust_pooled,
                  liar_theta: float, path: str) -> str:
    """False-precision outlier pulls pooling; the heuristic limits its influence."""
    labels = ["truth", "unprotected\npool", "bounded\nheuristic", "outlier"]
    vals = [true_theta, naive_pooled.theta[0], robust_pooled.theta[0], liar_theta]
    colors = ["k", "C3", "C0", "0.6"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(true_theta, color="k", lw=0.8, ls=":")
    ax.set_ylabel(r"estimated $\theta$")
    ax.set_title("Precision pooling is non-robust; a heuristic limits this stress test")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path
