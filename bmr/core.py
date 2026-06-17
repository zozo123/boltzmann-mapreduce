"""Boltzmann MapReduce --- statistical core.

Each map worker emits a *confidence density*

    h_k(theta)  proportional to  exp{ -beta_k * E_k(theta) },
    beta_k = n_k                       (inverse temperature = local sample size)
    E_k(theta) = 1/2 (theta_hat_k - theta)^T J_k (theta_hat_k - theta)

where ``J_k`` is the **per-observation** information, so the *total* precision of
a shard is ``n_k * J_k``.  The reduce phase combines independent shards as a
partition-function product of Gibbs factors, which is exactly precision-weighted
(inverse-variance) pooling.  See the paper, Sec. "The Boltzmann Reduce".
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

Z95 = 1.959963984540054  # standard normal 97.5% quantile
_VAR_FLOOR = 1e-12


def _as2d(M) -> np.ndarray:
    A = np.asarray(M, dtype=float)
    if A.ndim == 0:
        A = A.reshape(1, 1)
    elif A.ndim == 1:
        A = np.diag(A)
    return A


@dataclass
class CD:
    """Confidence density / Gibbs factor for a ``p``-dimensional parameter."""

    theta_hat: list[float]
    info_per_obs: list[list[float]]  # per-observation information J (p x p); total precision = n * J
    n: int
    meta: dict = field(default_factory=dict)

    # ---- basic geometry ----
    @property
    def beta(self) -> float:
        return float(self.n)

    @property
    def p(self) -> int:
        return len(self.theta_hat)

    def _theta(self) -> np.ndarray:
        return np.asarray(self.theta_hat, dtype=float)

    def _J(self) -> np.ndarray:
        return _as2d(self.info_per_obs)

    def total_precision(self) -> np.ndarray:
        return self.n * self._J()

    def cov(self) -> np.ndarray:
        return np.linalg.inv(self.total_precision())

    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.cov()), 0.0, None))

    # ---- Gibbs / thermodynamic view ----
    def energy(self, theta) -> float:
        d = self._theta() - np.asarray(theta, dtype=float)
        return float(0.5 * d @ self._J() @ d)

    def neg_log_density(self, theta) -> float:
        return self.beta * self.energy(theta)

    def gibbs_factor(self, theta) -> float:
        return float(np.exp(-self.neg_log_density(theta)))

    def temperature(self) -> float:
        return 1.0 / self.n

    def density1d(self, grid) -> np.ndarray:
        """Normalized 1-D confidence density N(theta_hat, cov) over ``grid`` (p == 1)."""
        if self.p != 1:
            raise ValueError("density1d requires p == 1")
        g = np.asarray(grid, dtype=float)
        mu = self.theta_hat[0]
        var = float(self.cov()[0, 0])
        return np.exp(-0.5 * (g - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)

    # ---- intervals ----
    def ci(self, level: float = 0.95):
        z = Z95 if abs(level - 0.95) < 1e-9 else _z_for(level)
        se = self.se()
        th = self._theta()
        return [(float(th[i] - z * se[i]), float(th[i] + z * se[i])) for i in range(self.p)]

    # ---- serialization ----
    def to_dict(self) -> dict:
        return {
            "theta_hat": list(map(float, self.theta_hat)),
            "info_per_obs": _as2d(self.info_per_obs).tolist(),
            "n": int(self.n),
            "meta": dict(self.meta),
        }

    @staticmethod
    def from_dict(d: dict) -> "CD":
        return CD(
            theta_hat=[float(v) for v in d["theta_hat"]],
            info_per_obs=[[float(v) for v in row] for row in d["info_per_obs"]],
            n=int(d["n"]),
            meta=dict(d.get("meta", {})),
        )


def _z_for(level: float) -> float:
    # inverse normal CDF via a rational approximation (Acklam) for arbitrary levels
    p = 0.5 + level / 2.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = np.sqrt(-2 * np.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


# ----------------------------------------------------------------------------
# Map: per-shard local estimators that emit a CD
# ----------------------------------------------------------------------------
def map_mean(x, shard_id=None, backend="local") -> CD:
    """Scalar-mean worker. theta_hat = mean(x); per-obs info J = 1/var(x)."""
    a = np.asarray(x, dtype=float).ravel()
    n = int(a.size)
    var = float(np.var(a, ddof=1)) if n > 1 else _VAR_FLOOR
    var = max(var, _VAR_FLOOR)
    return CD(
        theta_hat=[float(a.mean())],
        info_per_obs=[[1.0 / var]],
        n=n,
        meta={"shard_id": shard_id, "backend": backend, "n": n, "scenario": "mean"},
    )


def map_linreg(X, y, shard_id=None, backend="local") -> CD:
    """OLS worker. total precision = X'X / sigma2; per-obs info J = (X'X / n) / sigma2."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    resid = y - X @ beta
    dof = max(n - p, 1)
    sigma2 = max(float(resid @ resid) / dof, _VAR_FLOOR)
    J = (XtX / n) / sigma2  # per-observation information so that n * J = X'X / sigma2
    return CD(
        theta_hat=[float(v) for v in beta],
        info_per_obs=J.tolist(),
        n=int(n),
        meta={"shard_id": shard_id, "backend": backend, "n": int(n),
              "scenario": "linreg", "sigma2": sigma2},
    )


# ----------------------------------------------------------------------------
# Reduce: partition-function combination == precision-weighted pooling
# ----------------------------------------------------------------------------
@dataclass
class Pooled:
    theta: list[float]
    cov: list[list[float]]
    precision: list[list[float]]
    n_total: int
    neg_log_Z: float
    weights: dict

    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(np.asarray(self.cov)), 0.0, None))

    def ci(self, level: float = 0.95):
        z = Z95 if abs(level - 0.95) < 1e-9 else _z_for(level)
        se = self.se()
        return [(float(self.theta[i] - z * se[i]), float(self.theta[i] + z * se[i]))
                for i in range(len(self.theta))]


def reduce_partition(cds: list[CD]) -> Pooled:
    """Multiply Gibbs factors: precision and energies add (disjoint shards)."""
    if not cds:
        raise ValueError("no confidence densities to reduce")
    p = cds[0].p
    precision = np.zeros((p, p))
    info_theta = np.zeros(p)
    traces = {}
    for cd in cds:
        Tp = cd.total_precision()
        precision += Tp
        info_theta += Tp @ cd._theta()
        sid = cd.meta.get("shard_id", id(cd))
        traces[sid] = traces.get(sid, 0.0) + float(np.trace(Tp))
    cov = np.linalg.inv(precision)
    theta = cov @ info_theta
    tot_trace = sum(traces.values()) or 1.0
    weights = {k: v / tot_trace for k, v in traces.items()}
    _, logdet = np.linalg.slogdet(precision)
    neg_log_Z = float(-(0.5 * p * np.log(2 * np.pi) - 0.5 * logdet))
    return Pooled(
        theta=[float(v) for v in theta],
        cov=cov.tolist(),
        precision=precision.tolist(),
        n_total=int(sum(cd.n for cd in cds)),
        neg_log_Z=neg_log_Z,
        weights=weights,
    )


# ----------------------------------------------------------------------------
# Byzantine clip: bound a confident liar's influence (the paper's honest point)
# ----------------------------------------------------------------------------
def _mad(v: np.ndarray) -> float:
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med)))


def byzantine_clip(cds: list[CD], kappa: float = 3.0):
    """Cap reported precision and down-weight location-outliers.

    Returns ``(kept, flagged)`` where ``kept`` are clipped copies (inputs are not
    mutated) and ``flagged`` lists the shard ids that were precision-clipped or
    flagged as location outliers.  Precision-weighted pooling has an *unbounded*
    influence function, so a confident liar (tiny variance, far estimate) must be
    bounded here rather than absorbed by the reduce.
    """
    if not cds:
        return [], []
    taus = np.array([float(np.trace(cd.total_precision())) for cd in cds])
    center, scale = float(np.median(taus)), max(_mad(taus), 1e-12)
    cap = center + kappa * scale

    thetas0 = np.array([cd.theta_hat[0] for cd in cds])
    tmed, tscale = float(np.median(thetas0)), max(_mad(thetas0), 1e-12)

    kept, flagged = [], []
    for cd, tau in zip(cds, taus):
        sid = cd.meta.get("shard_id")
        J = cd._J()
        scale_factor = 1.0
        is_flagged = False
        if tau > cap:                                  # confident-liar precision clip
            scale_factor *= cap / tau
            is_flagged = True
        rz = abs(cd.theta_hat[0] - tmed) / tscale      # location-outlier test
        if rz > kappa:
            scale_factor *= 1e-6                       # effectively remove its pull
            is_flagged = True
        kept.append(CD(theta_hat=list(cd.theta_hat),
                       info_per_obs=(J * scale_factor).tolist(),
                       n=cd.n, meta=dict(cd.meta)))
        if is_flagged and sid is not None:
            flagged.append(sid)
    return kept, flagged


# ----------------------------------------------------------------------------
# Oracle: full-data estimator (the efficiency ceiling reduce recovers)
# ----------------------------------------------------------------------------
def oracle_mean(x_all) -> CD:
    return map_mean(x_all, shard_id="oracle", backend="oracle")


def oracle_linreg(X_all, y_all) -> CD:
    return map_linreg(X_all, y_all, shard_id="oracle", backend="oracle")


# ----------------------------------------------------------------------------
# Non-linear estimating function: logistic regression (IRLS / Fisher scoring).
# Here the Gibbs energy is only the LEADING-ORDER (LAN) quadratic, so this is the
# informative regime: the identity is exact only asymptotically.
# ----------------------------------------------------------------------------
def _irls_logistic(X: np.ndarray, y: np.ndarray, iters: int = 50, ridge: float = 1e-8):
    p = X.shape[1]
    beta = np.zeros(p)
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        XtWX = X.T @ (w[:, None] * X) + ridge * np.eye(p)
        grad = X.T @ (y - mu)
        step = np.linalg.solve(XtWX, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = np.clip(X @ beta, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(mu * (1 - mu), 1e-9, None)
    XtWX = X.T @ (w[:, None] * X) + ridge * np.eye(p)  # observed Fisher information
    return beta, XtWX


def map_logistic(X, y, shard_id=None, backend="local") -> CD:
    """Logistic-regression worker. Total Fisher info = X'WX; per-obs info J = X'WX / n."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[0]
    beta, XtWX = _irls_logistic(X, y)
    J = XtWX / n
    return CD(theta_hat=[float(v) for v in beta], info_per_obs=J.tolist(), n=int(n),
              meta={"shard_id": shard_id, "backend": backend, "n": int(n), "scenario": "logistic"})


def oracle_logistic(X_all, y_all) -> CD:
    return map_logistic(X_all, y_all, shard_id="oracle", backend="oracle")


def reduce_naive(cds: list[CD]) -> np.ndarray:
    """Equal-weight average of point estimates --- the naive baseline the
    precision-weighted partition-function reduce is meant to beat under
    heterogeneity. Returns the averaged parameter vector."""
    if not cds:
        raise ValueError("no estimates")
    return np.mean([np.asarray(cd.theta_hat, dtype=float) for cd in cds], axis=0)
