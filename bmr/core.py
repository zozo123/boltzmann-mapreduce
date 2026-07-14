"""Uncertainty-aware reduction for independent worker summaries.

Each map worker emits a *confidence density*

    h_k(theta)  proportional to  exp{ -beta_k * E_k(theta) },
    beta_k = n_k                       (under the per-observation convention)
    E_k(theta) = 1/2 (theta_hat_k - theta)^T J_k (theta_hat_k - theta)

where ``J_k`` is the **per-observation** information, so the *total* precision of
a shard is ``n_k * J_k``.  For disjoint, statistically independent shards with a
common target, the reduce phase combines Gaussian factors by adding their natural
parameters.  This is established inverse-information/confidence-distribution
pooling; the paper contributes an execution contract and thermodynamic reading,
not a new estimator.
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
    lineage: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Reject malformed or non-identifiable summaries at the trust boundary."""
        theta = np.asarray(self.theta_hat, dtype=float)
        if theta.ndim != 1 or theta.size == 0 or not np.all(np.isfinite(theta)):
            raise ValueError("theta_hat must be a finite, non-empty vector")
        try:
            n_int = int(self.n)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("n must be a positive integer") from exc
        if isinstance(self.n, bool) or n_int != self.n or n_int <= 0:
            raise ValueError("n must be a positive integer")
        if not isinstance(self.meta, dict):
            raise ValueError("meta must be a dictionary")
        for name, values in (("lineage", self.lineage), ("evidence_ids", self.evidence_ids)):
            if not isinstance(values, (list, tuple)) or any(
                not isinstance(v, str) or not v.strip() for v in values
            ):
                raise ValueError(f"{name} must contain only non-empty strings")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids must not contain duplicates")
        J = _as2d(self.info_per_obs)
        if J.shape != (theta.size, theta.size):
            raise ValueError(f"information matrix must have shape {(theta.size, theta.size)}")
        if not np.all(np.isfinite(J)):
            raise ValueError("information matrix must contain only finite values")
        if not np.allclose(J, J.T, rtol=1e-10, atol=1e-12):
            raise ValueError("information matrix must be symmetric")
        try:
            np.linalg.cholesky(J)
        except np.linalg.LinAlgError as exc:
            raise ValueError("information matrix must be positive definite") from exc

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
        P = self.total_precision()
        chol = np.linalg.cholesky(P)
        return np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(self.p)))

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
        _validate_level(level)
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
            "lineage": list(self.lineage),
            "evidence_ids": list(self.evidence_ids),
        }

    @staticmethod
    def from_dict(d: dict) -> "CD":
        return CD(
            theta_hat=[float(v) for v in d["theta_hat"]],
            info_per_obs=[[float(v) for v in row] for row in d["info_per_obs"]],
            n=d["n"],
            meta=d.get("meta", {}),
            lineage=d.get("lineage", []),
            evidence_ids=d.get("evidence_ids", []),
        )


def _z_for(level: float) -> float:
    # inverse normal CDF via a rational approximation (Acklam) for arbitrary levels
    _validate_level(level)
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


def _validate_level(level: float) -> None:
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence level must be strictly between 0 and 1")


# ----------------------------------------------------------------------------
# Map: per-shard local estimators that emit a CD
# ----------------------------------------------------------------------------
def map_mean(x, shard_id=None, backend="local") -> CD:
    """Scalar-mean worker. theta_hat = mean(x); per-obs info J = 1/var(x)."""
    a = np.asarray(x, dtype=float).ravel()
    n = int(a.size)
    if n < 2 or not np.all(np.isfinite(a)):
        raise ValueError("mean worker requires at least two finite observations")
    var = float(np.var(a, ddof=1))
    if not np.isfinite(var) or var <= _VAR_FLOOR:
        raise ValueError("mean worker requires positive, estimable sample variance")
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
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix")
    n, p = X.shape
    if y.size != n or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must be finite and have matching rows")
    if n <= p:
        raise ValueError("linear-regression worker requires n > p")
    XtX = X.T @ X
    try:
        beta = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError as exc:
        raise ValueError("linear-regression design matrix must have full column rank") from exc
    resid = y - X @ beta
    dof = max(n - p, 1)
    sigma2 = float(resid @ resid) / dof
    if not np.isfinite(sigma2) or sigma2 <= _VAR_FLOOR:
        raise ValueError("linear-regression residual variance must be positive and estimable")
    J = (XtX / n) / sigma2  # per-observation information so that n * J = X'X / sigma2
    return CD(
        theta_hat=[float(v) for v in beta],
        info_per_obs=J.tolist(),
        n=int(n),
        meta={"shard_id": shard_id, "backend": backend, "n": int(n),
              "scenario": "linreg", "sigma2": sigma2},
    )


# ----------------------------------------------------------------------------
# Reduce: Gaussian natural-parameter / inverse-information pooling
# ----------------------------------------------------------------------------
@dataclass
class CanonicalSummary:
    """Mergeable Gaussian summary ``(P, q, c, N)`` plus provenance unions.

    ``merge`` is associative and commutative algebraically. Literal overlap in
    non-empty evidence identifiers is rejected by default; callers must opt in
    explicitly when overlap is intentional and statistically modeled elsewhere.
    Shared lineage is retained but not interpreted as an independence model.
    """

    precision: list[list[float]]
    info_theta: list[float]
    quadratic_constant: float
    n_total: int
    evidence_ids: tuple[str, ...] = ()
    lineage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        P = _as2d(self.precision)
        q = np.asarray(self.info_theta, dtype=float)
        if q.ndim != 1 or q.size == 0 or P.shape != (q.size, q.size):
            raise ValueError("canonical precision and information vector dimensions disagree")
        if not np.all(np.isfinite(P)) or not np.all(np.isfinite(q)):
            raise ValueError("canonical summary must contain only finite values")
        if not np.allclose(P, P.T, rtol=1e-10, atol=1e-12):
            raise ValueError("canonical precision must be symmetric")
        try:
            np.linalg.cholesky(P)
        except np.linalg.LinAlgError as exc:
            raise ValueError("canonical precision must be positive definite") from exc
        if not np.isfinite(self.quadratic_constant):
            raise ValueError("canonical quadratic constant must be finite")
        try:
            n_int = int(self.n_total)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("canonical sample size must be a positive integer") from exc
        if isinstance(self.n_total, bool) or n_int != self.n_total or n_int <= 0:
            raise ValueError("canonical sample size must be a positive integer")
        for name, values in (("evidence_ids", self.evidence_ids), ("lineage", self.lineage)):
            if any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError(f"canonical {name} must contain non-empty strings")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("canonical evidence_ids must not contain duplicates")

    @classmethod
    def from_cd(cls, cd: CD) -> "CanonicalSummary":
        P = cd.total_precision()
        theta = cd._theta()
        return cls(
            precision=P.tolist(),
            info_theta=(P @ theta).tolist(),
            quadratic_constant=float(theta @ P @ theta),
            n_total=cd.n,
            evidence_ids=tuple(cd.evidence_ids),
            lineage=tuple(cd.lineage),
        )

    @property
    def p(self) -> int:
        return len(self.info_theta)

    def merge(
        self,
        other: "CanonicalSummary",
        *,
        allow_evidence_overlap: bool = False,
    ) -> "CanonicalSummary":
        if self.p != other.p:
            raise ValueError("canonical summaries must have the same dimension")
        overlap = set(self.evidence_ids).intersection(other.evidence_ids)
        if overlap and not allow_evidence_overlap:
            labels = ", ".join(sorted(overlap))
            raise ValueError(f"overlapping evidence_ids: {labels}")
        P = _as2d(self.precision) + _as2d(other.precision)
        q = np.asarray(self.info_theta, dtype=float) + np.asarray(other.info_theta, dtype=float)
        evidence = tuple(sorted(set(self.evidence_ids).union(other.evidence_ids)))
        lineage = tuple(dict.fromkeys(self.lineage + other.lineage))
        return CanonicalSummary(
            precision=P.tolist(),
            info_theta=q.tolist(),
            quadratic_constant=self.quadratic_constant + other.quadratic_constant,
            n_total=self.n_total + other.n_total,
            evidence_ids=evidence,
            lineage=lineage,
        )


@dataclass
class Pooled:
    theta: list[float]
    cov: list[list[float]]
    precision: list[list[float]]
    n_total: int
    neg_log_Z: float
    disagreement_energy: float
    weights: dict

    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(np.asarray(self.cov)), 0.0, None))

    def ci(self, level: float = 0.95):
        _validate_level(level)
        z = Z95 if abs(level - 0.95) < 1e-9 else _z_for(level)
        se = self.se()
        return [(float(self.theta[i] - z * se[i]), float(self.theta[i] + z * se[i]))
                for i in range(len(self.theta))]


def finalize_canonical(summary: CanonicalSummary, *, weights: dict | None = None) -> Pooled:
    """Convert a mergeable canonical summary into a pooled Gaussian result."""
    precision = _as2d(summary.precision)
    info_theta = np.asarray(summary.info_theta, dtype=float)
    chol = np.linalg.cholesky(precision)
    theta = np.linalg.solve(chol.T, np.linalg.solve(chol, info_theta))
    cov = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(summary.p)))
    logdet = 2.0 * float(np.log(np.diag(chol)).sum())
    disagreement = max(
        0.0,
        float(summary.quadratic_constant - info_theta @ theta),
    )
    log_Z = 0.5 * summary.p * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * disagreement
    return Pooled(
        theta=[float(v) for v in theta],
        cov=cov.tolist(),
        precision=precision.tolist(),
        n_total=int(summary.n_total),
        neg_log_Z=float(-log_Z),
        disagreement_energy=float(0.5 * disagreement),
        weights=dict(weights or {}),
    )


def reduce_partition(cds: list[CD], *, allow_evidence_overlap: bool = False) -> Pooled:
    """Combine independent unnormalized Gaussian factors.

    For ``g_k(theta) = exp(-1/2 (theta-mu_k)' P_k (theta-mu_k))``, this returns
    the product mode/covariance and the exact ``-log integral(prod_k g_k)``.
    Multiplication is justified only when summaries are based on disjoint,
    statistically independent evidence for a common parameter.
    """
    if not cds:
        raise ValueError("no confidence densities to reduce")
    p = cds[0].p
    summary = CanonicalSummary.from_cd(cds[0])
    scalar_precisions = {}
    for cd in cds:
        if cd.p != p:
            raise ValueError("all confidence densities must have the same dimension")
        Tp = cd.total_precision()
        sid = cd.meta.get("shard_id", id(cd))
        if p == 1:
            scalar_precisions[sid] = scalar_precisions.get(sid, 0.0) + float(Tp[0, 0])
    for cd in cds[1:]:
        summary = summary.merge(
            CanonicalSummary.from_cd(cd),
            allow_evidence_overlap=allow_evidence_overlap,
        )
    total_scalar_precision = sum(scalar_precisions.values()) or 1.0
    weights = {k: v / total_scalar_precision for k, v in scalar_precisions.items()}
    return finalize_canonical(summary, weights=weights)


# ----------------------------------------------------------------------------
# Outlier stress-test heuristic (not a certified Byzantine defense)
# ----------------------------------------------------------------------------
def _mad(v: np.ndarray) -> float:
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med)))


def byzantine_clip(cds: list[CD], kappa: float = 3.0):
    """Apply a scale-aware multivariate precision/location heuristic.

    Returns ``(kept, flagged)`` where ``kept`` are clipped copies (inputs are not
    mutated) and ``flagged`` lists the shard ids that were precision-clipped or
    location-downweighted. Precision is summarized by log determinant per
    dimension, whose relative values are invariant to a common invertible
    reparameterization. Location is scored with coordinate-wise median/MAD scaling
    and a redescending weight. The latter is scale-aware but not rotation invariant.
    This is a transparent stress-test heuristic, not a Byzantine-robust guarantee.
    """
    if not cds:
        return [], []
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be positive and finite")
    p = cds[0].p
    if any(cd.p != p for cd in cds):
        raise ValueError("all confidence densities must have the same dimension")

    log_precision = []
    for cd in cds:
        sign, logdet = np.linalg.slogdet(cd.total_precision())
        if sign <= 0:
            raise ValueError("worker precision must be positive definite")
        log_precision.append(float(logdet) / p)
    log_precision = np.asarray(log_precision)
    lp_center = float(np.median(log_precision))
    lp_scale = max(_mad(log_precision), 1e-12)
    log_cap = lp_center + kappa * lp_scale

    thetas = np.asarray([cd.theta_hat for cd in cds], dtype=float)
    location_center = np.median(thetas, axis=0)
    location_scale = 1.4826 * np.median(np.abs(thetas - location_center), axis=0)
    median_se = np.median(np.asarray([cd.se() for cd in cds]), axis=0)
    location_scale = np.maximum(location_scale, np.maximum(median_se, 1e-12))
    location_scores = np.linalg.norm((thetas - location_center) / location_scale, axis=1)
    location_cap = kappa * np.sqrt(p)

    kept, flagged = [], []
    for cd, lp, location_score in zip(cds, log_precision, location_scores):
        sid = cd.meta.get("shard_id")
        J = cd._J()
        scale_factor = 1.0
        is_flagged = False
        if lp > log_cap:
            scale_factor *= float(np.exp(log_cap - lp))
            is_flagged = True
        if location_score > location_cap:
            scale_factor *= float((location_cap / location_score) ** 2)
            is_flagged = True
        kept.append(CD(theta_hat=list(cd.theta_hat),
                       info_per_obs=(J * scale_factor).tolist(),
                       n=cd.n, meta=dict(cd.meta), lineage=list(cd.lineage),
                       evidence_ids=list(cd.evidence_ids)))
        if is_flagged and sid is not None:
            flagged.append(sid)
    return kept, flagged


# ----------------------------------------------------------------------------
# Centralized comparison estimators on the same synthetic data
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
    converged = False
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        XtWX = X.T @ (w[:, None] * X) + ridge * np.eye(p)
        grad = X.T @ (y - mu)
        step = np.linalg.solve(XtWX, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            converged = True
            break
    if not converged:
        raise RuntimeError("logistic IRLS did not converge")
    eta = np.clip(X @ beta, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(mu * (1 - mu), 1e-9, None)
    XtWX = X.T @ (w[:, None] * X)  # data Fisher information; ridge is solver-only
    return beta, XtWX


def map_logistic(X, y, shard_id=None, backend="local") -> CD:
    """Logistic-regression worker. Total Fisher info = X'WX; per-obs info J = X'WX / n."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix")
    n = X.shape[0]
    p = X.shape[1]
    if y.size != n or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must be finite and have matching rows")
    if n <= p:
        raise ValueError("logistic worker requires n > p")
    if not np.all(np.isin(y, [0.0, 1.0])) or np.unique(y).size < 2:
        raise ValueError("logistic worker requires binary outcomes containing both classes")
    beta, XtWX = _irls_logistic(X, y)
    J = XtWX / n
    return CD(theta_hat=[float(v) for v in beta], info_per_obs=J.tolist(), n=int(n),
              meta={"shard_id": shard_id, "backend": backend, "n": int(n), "scenario": "logistic"})


def oracle_logistic(X_all, y_all) -> CD:
    return map_logistic(X_all, y_all, shard_id="oracle", backend="oracle")


def reduce_naive(cds: list[CD]) -> np.ndarray:
    """Equal-weight point-estimate average, used as a naive baseline when
    shards have unequal sizes or information. Returns the averaged vector."""
    if not cds:
        raise ValueError("no estimates")
    return np.mean([np.asarray(cd.theta_hat, dtype=float) for cd in cds], axis=0)
