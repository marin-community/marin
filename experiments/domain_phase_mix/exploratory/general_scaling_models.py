"""General scaling models for arbitrary M domains and N phases.

Generalizes the 24 scaling models from scaling_models.py (which are hardcoded
for 2 phases, 2 domains) to work with any (R, N, M) weight array.

Models are organized as:
  - Feature-based (OLS / regularized regression on a design matrix)
  - SOE family (satiety + overfit + epoch, design matrix)
  - Nonlinear (CES, multi-start scipy.optimize)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPS = 1e-8
CrossPhase = Literal["adjacent", "all_pairs"]
WithinPhaseInteraction = Literal["none", "small_x_nonsmall", "small_x_all_others"]


# ---------------------------------------------------------------------------
# Data specifications
# ---------------------------------------------------------------------------
@dataclass
class DatasetSpec:
    """Everything a model needs to know about one dataset."""

    weights: np.ndarray  # (R, N, M) per-run weights
    y: np.ndarray  # (R,) target values
    epoch_multipliers: np.ndarray  # (M,) or (N, M)
    domain_names: list[str]
    phase_names: list[str]
    small_domains: list[int] | None = None  # indices of epochable domains
    name: str = ""

    @property
    def R(self) -> int:
        return self.weights.shape[0]

    @property
    def N(self) -> int:
        return self.weights.shape[1]

    @property
    def M(self) -> int:
        return self.weights.shape[2]

    def subset(self, idx: np.ndarray) -> "DatasetSpec":
        return DatasetSpec(
            weights=self.weights[idx],
            y=self.y[idx],
            epoch_multipliers=self.epoch_multipliers,
            domain_names=self.domain_names,
            phase_names=self.phase_names,
            small_domains=self.small_domains,
            name=self.name,
        )


def _always_applicable(_spec: DatasetSpec) -> bool:
    return True


@dataclass
class GeneralModelSpec:
    name: str
    fit_fn: Callable[[DatasetSpec], tuple[Callable[[np.ndarray], np.ndarray], dict]]
    applicable: Callable[[DatasetSpec], bool] = field(default=_always_applicable)
    description: str = ""


# ---------------------------------------------------------------------------
# SOE helpers (adapted from user-provided general implementations)
# ---------------------------------------------------------------------------
def _as_3d(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    if W.ndim == 2:
        W = W[None, :, :]
    if W.ndim != 3:
        raise ValueError(f"weights must have shape (R,N,M) or (N,M); got {W.shape}")
    return W


def _broadcast_epoch_mult(epoch_multipliers: np.ndarray, N: int, M: int) -> np.ndarray:
    C = np.asarray(epoch_multipliers, dtype=float)
    if C.ndim == 1:
        if C.shape[0] != M:
            raise ValueError(f"epoch_multipliers shape {C.shape} != (M={M},)")
        C = np.tile(C[None, :], (N, 1))
    elif C.ndim == 2:
        if C.shape != (N, M):
            raise ValueError(f"epoch_multipliers shape {C.shape} != ({N},{M})")
    else:
        raise ValueError(f"epoch_multipliers must be 1D or 2D; got {C.ndim}D")
    return C


def _parse_small(domains: list[int] | None, M: int) -> list[int]:
    if domains is None:
        return list(range(M))
    return sorted(set(domains))


def _phase_pairs(N: int, cross_phase: CrossPhase) -> list[tuple[int, int]]:
    if N < 2:
        return []
    if cross_phase == "adjacent":
        return [(k, k + 1) for k in range(N - 1)]
    return [(k1, k2) for k1 in range(N) for k2 in range(k1 + 1, N)]


def _compute_epochs(W: np.ndarray, C: np.ndarray) -> np.ndarray:
    """(R,N,M) weights × (N,M) multipliers → (R,N,M) epochs."""
    return W * C[None, :, :]


# ---------------------------------------------------------------------------
# SOE design matrices
# ---------------------------------------------------------------------------
def soe_base_design(
    W: np.ndarray,
    epoch_multipliers: np.ndarray,
    small_domains: list[int] | None = None,
    cross_phase: CrossPhase = "adjacent",
) -> tuple[np.ndarray, list[str], int]:
    W = _as_3d(W)
    R, N, M = W.shape
    C = _broadcast_epoch_mult(epoch_multipliers, N, M)
    E = _compute_epochs(W, C)
    G = np.log(E + EPS)
    S = _parse_small(small_domains, M)
    small_mask = np.zeros(M, dtype=bool)
    for s in S:
        small_mask[s] = True

    cols: list[np.ndarray] = [np.ones(R)]
    names: list[str] = ["1"]

    for k in range(N):
        for d in range(M):
            cols.append(G[:, k, d])
            names.append(f"sat[p{k},d{d}]")

    for k in range(N):
        for d in range(M):
            if not small_mask[d]:
                continue
            cols.append(E[:, k, d] ** 2)
            names.append(f"epoch2[p{k},d{d}]")

    for d in S:
        for k1, k2 in _phase_pairs(N, cross_phase):
            cols.append(E[:, k1, d] * E[:, k2, d])
            names.append(f"reuse[d{d},p{k1}-p{k2}]")

    Phi = np.column_stack(cols)
    return Phi, names, Phi.shape[1]


def soe_plus_design(
    W: np.ndarray,
    epoch_multipliers: np.ndarray,
    small_domains: list[int] | None = None,
    cross_phase: CrossPhase = "adjacent",
    within_phase: WithinPhaseInteraction = "small_x_nonsmall",
) -> tuple[np.ndarray, list[str], int]:
    W = _as_3d(W)
    R, N, M = W.shape
    C = _broadcast_epoch_mult(epoch_multipliers, N, M)
    E = _compute_epochs(W, C)
    G = np.log(E + EPS)
    S = _parse_small(small_domains, M)
    small_mask = np.zeros(M, dtype=bool)
    for s in S:
        small_mask[s] = True
    nonsmall = [d for d in range(M) if not small_mask[d]]

    cols: list[np.ndarray] = [np.ones(R)]
    names: list[str] = ["1"]

    for k in range(N):
        for d in range(M):
            cols.append(G[:, k, d])
            names.append(f"sat[p{k},d{d}]")

    for k in range(N):
        for d in S:
            cols.append(E[:, k, d])
            names.append(f"epoch[p{k},d{d}]")
            cols.append(E[:, k, d] ** 2)
            names.append(f"epoch2[p{k},d{d}]")

    for d in S:
        for k1, k2 in _phase_pairs(N, cross_phase):
            cols.append(E[:, k1, d] * E[:, k2, d])
            names.append(f"reuse[d{d},p{k1}-p{k2}]")

    if within_phase == "small_x_nonsmall":
        for k in range(N):
            for sd in S:
                for d in nonsmall:
                    cols.append(G[:, k, sd] * G[:, k, d])
                    names.append(f"satx[p{k},d{sd}*d{d}]")
    elif within_phase == "small_x_all_others":
        for k in range(N):
            for sd in S:
                for d in range(M):
                    if d == sd:
                        continue
                    cols.append(G[:, k, sd] * G[:, k, d])
                    names.append(f"satx[p{k},d{sd}*d{d}]")

    Phi = np.column_stack(cols)
    return Phi, names, Phi.shape[1]


def soe_curric_design(
    W: np.ndarray,
    epoch_multipliers: np.ndarray,
    small_domains: list[int] | None = None,
    cross_phase: CrossPhase = "adjacent",
    within_phase: WithinPhaseInteraction = "small_x_nonsmall",
    include_nonsmall_persistence: bool = True,
) -> tuple[np.ndarray, list[str], int]:
    W = _as_3d(W)
    R, N, M = W.shape
    C = _broadcast_epoch_mult(epoch_multipliers, N, M)
    E = _compute_epochs(W, C)
    G = np.log(E + EPS)
    S = _parse_small(small_domains, M)
    small_mask = np.zeros(M, dtype=bool)
    for s in S:
        small_mask[s] = True
    nonsmall = [d for d in range(M) if not small_mask[d]]
    pairs = _phase_pairs(N, cross_phase)

    cols: list[np.ndarray] = [np.ones(R)]
    names: list[str] = ["1"]

    # satiety for all (k,d)
    for k in range(N):
        for d in range(M):
            cols.append(G[:, k, d])
            names.append(f"sat[p{k},d{d}]")

    # small-domain per-phase: epoch, epoch², sat*epoch
    for k in range(N):
        for d in S:
            cols.append(E[:, k, d])
            names.append(f"epoch[p{k},d{d}]")
            cols.append(E[:, k, d] ** 2)
            names.append(f"epoch2[p{k},d{d}]")
            cols.append(G[:, k, d] * E[:, k, d])
            names.append(f"sat_epoch[p{k},d{d}]")

    # cross-phase curriculum
    for d in S:
        for k1, k2 in pairs:
            cols.append(E[:, k1, d] * E[:, k2, d])
            names.append(f"reuse[d{d},p{k1}-p{k2}]")
            cols.append(G[:, k1, d] * E[:, k2, d])
            names.append(f"carry[d{d},p{k1}->p{k2}]")
            cols.append(G[:, k2, d] * E[:, k1, d])
            names.append(f"carry[d{d},p{k2}->p{k1}]")
            cols.append(G[:, k1, d] * G[:, k2, d])
            names.append(f"persist[d{d},p{k1}-p{k2}]")

    # within-phase coupling
    if within_phase == "small_x_nonsmall":
        for k in range(N):
            for sd in S:
                for d in nonsmall:
                    cols.append(G[:, k, sd] * G[:, k, d])
                    names.append(f"satx[p{k},d{sd}*d{d}]")
    elif within_phase == "small_x_all_others":
        for k in range(N):
            for sd in S:
                for d in range(M):
                    if d == sd:
                        continue
                    cols.append(G[:, k, sd] * G[:, k, d])
                    names.append(f"satx[p{k},d{sd}*d{d}]")

    # non-small cross-phase persistence
    if include_nonsmall_persistence:
        for d in nonsmall:
            for k1, k2 in pairs:
                cols.append(G[:, k1, d] * G[:, k2, d])
                names.append(f"persist[d{d},p{k1}-p{k2}]")

    Phi = np.column_stack(cols)
    return Phi, names, Phi.shape[1]


# ---------------------------------------------------------------------------
# General feature builders
# ---------------------------------------------------------------------------
def _flat_weights(spec: DatasetSpec) -> np.ndarray:
    """(R, N*M) flattened weight features."""
    return spec.weights.reshape(spec.R, -1)


def _flat_weight_epoch(spec: DatasetSpec) -> np.ndarray:
    """(R, 2*N*M) = [weights, log(epochs+eps)] per (phase, domain)."""
    C = _broadcast_epoch_mult(spec.epoch_multipliers, spec.N, spec.M)
    E = _compute_epochs(spec.weights, C)
    w_flat = spec.weights.reshape(spec.R, -1)
    e_flat = np.log(E.reshape(spec.R, -1) + EPS)
    return np.column_stack([w_flat, e_flat])


def _quadratic_expand(X: np.ndarray) -> np.ndarray:
    """Add intercept, linear, squared, and cross terms."""
    R, d = X.shape
    parts = [np.ones((R, 1))]
    for j in range(d):
        parts.append(X[:, j : j + 1])
    for j in range(d):
        parts.append(X[:, j : j + 1] ** 2)
    for j in range(d):
        for k in range(j + 1, d):
            parts.append((X[:, j] * X[:, k]).reshape(-1, 1))
    return np.hstack(parts)


def _quadratic_nparams(d: int) -> int:
    return 1 + d + d + d * (d - 1) // 2


def _vdom_features(spec: DatasetSpec) -> np.ndarray:
    """(R, N*M) virtual domain volumes = phase_fraction * weight."""
    phase_fracs = np.ones(spec.N) / spec.N
    V = spec.weights * phase_fracs[None, :, None]
    return V.reshape(spec.R, -1)


# ---------------------------------------------------------------------------
# OLS / ridge helpers
# ---------------------------------------------------------------------------
def _fit_ols(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return beta


def _fit_ridge(Phi: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """OLS with ridge penalty (intercept in column 0 is not penalized)."""
    penalty = np.eye(Phi.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(Phi.T @ Phi + ridge * penalty, Phi.T @ y)


# ---------------------------------------------------------------------------
# Model: Linear (weight-only)
# ---------------------------------------------------------------------------
def _fit_linear(spec: DatasetSpec):
    X = np.column_stack([np.ones(spec.R), _flat_weights(spec)])
    beta = _fit_ols(X, spec.y)
    n_params = X.shape[1]

    def predict(W_new):
        sp = _as_3d(W_new)
        Xn = np.column_stack([np.ones(sp.shape[0]), sp.reshape(sp.shape[0], -1)])
        return Xn @ beta

    return predict, {"n_params": n_params}


def _applicable_linear(spec: DatasetSpec) -> bool:
    return spec.R > spec.N * spec.M + 1


# ---------------------------------------------------------------------------
# Model: Quadratic (weight-only), unregularized
# ---------------------------------------------------------------------------
def _fit_quadratic_weight(spec: DatasetSpec):
    X = _flat_weights(spec)
    Z = _quadratic_expand(X)
    n_params = Z.shape[1]
    beta = _fit_ols(Z, spec.y)

    def predict(W_new):
        sp = _as_3d(W_new)
        Xn = sp.reshape(sp.shape[0], -1)
        return _quadratic_expand(Xn) @ beta

    return predict, {"n_params": n_params}


def _applicable_quadratic_weight(spec: DatasetSpec) -> bool:
    d = spec.N * spec.M
    return spec.R > _quadratic_nparams(d)


# ---------------------------------------------------------------------------
# Model: Quadratic(w-e), unregularized
# ---------------------------------------------------------------------------
def _fit_quadratic_we(spec: DatasetSpec):
    X = _flat_weight_epoch(spec)
    Z = _quadratic_expand(X)
    n_params = Z.shape[1]
    beta = _fit_ols(Z, spec.y)
    emult = spec.epoch_multipliers

    def predict(W_new):
        sp = _as_3d(W_new)
        C = _broadcast_epoch_mult(emult, sp.shape[1], sp.shape[2])
        E = _compute_epochs(sp, C)
        w_flat = sp.reshape(sp.shape[0], -1)
        e_flat = np.log(E.reshape(sp.shape[0], -1) + EPS)
        Xn = np.column_stack([w_flat, e_flat])
        return _quadratic_expand(Xn) @ beta

    return predict, {"n_params": n_params}


def _applicable_quadratic_we(spec: DatasetSpec) -> bool:
    d = 2 * spec.N * spec.M
    return spec.R > _quadratic_nparams(d)


# ---------------------------------------------------------------------------
# Model: RidgeQuad(w-e) — always applicable
# ---------------------------------------------------------------------------
def _fit_ridge_quad_we(spec: DatasetSpec):
    from sklearn.linear_model import RidgeCV

    X = _flat_weight_epoch(spec)
    Z = _quadratic_expand(X)
    n_params = Z.shape[1]

    mu, sigma = Z.mean(0), Z.std(0)
    mu[0], sigma[0] = 0.0, 1.0
    sigma[sigma < 1e-12] = 1.0
    Z_std = (Z - mu) / sigma

    ridge = RidgeCV(alphas=np.logspace(-4, 4, 50), fit_intercept=False)
    ridge.fit(Z_std, spec.y)

    coef = ridge.coef_ / sigma
    coef[0] -= np.sum(ridge.coef_ * mu / sigma)
    emult = spec.epoch_multipliers

    def predict(W_new):
        sp = _as_3d(W_new)
        C = _broadcast_epoch_mult(emult, sp.shape[1], sp.shape[2])
        E = _compute_epochs(sp, C)
        w_flat = sp.reshape(sp.shape[0], -1)
        e_flat = np.log(E.reshape(sp.shape[0], -1) + EPS)
        Xn = np.column_stack([w_flat, e_flat])
        return _quadratic_expand(Xn) @ coef

    return predict, {"n_params": n_params, "alpha": float(ridge.alpha_)}


# ---------------------------------------------------------------------------
# Model: LogQuad(w-e), unregularized
# ---------------------------------------------------------------------------
def _fit_logquad_we(spec: DatasetSpec):
    X = _flat_weight_epoch(spec)
    Z = _quadratic_expand(X)
    n_params = Z.shape[1]
    log_y = np.log(spec.y)
    beta = _fit_ols(Z, log_y)
    emult = spec.epoch_multipliers

    def predict(W_new):
        sp = _as_3d(W_new)
        C = _broadcast_epoch_mult(emult, sp.shape[1], sp.shape[2])
        E = _compute_epochs(sp, C)
        w_flat = sp.reshape(sp.shape[0], -1)
        e_flat = np.log(E.reshape(sp.shape[0], -1) + EPS)
        Xn = np.column_stack([w_flat, e_flat])
        return np.exp(_quadratic_expand(Xn) @ beta)

    return predict, {"n_params": n_params}


# ---------------------------------------------------------------------------
# Model: BayesLogQuad(w-e) — always applicable
# ---------------------------------------------------------------------------
def _fit_bayes_logquad_we(spec: DatasetSpec):
    from sklearn.linear_model import BayesianRidge

    X = _flat_weight_epoch(spec)
    Z = _quadratic_expand(X)
    n_params = Z.shape[1]
    log_y = np.log(spec.y)

    mu, sigma = Z.mean(0), Z.std(0)
    mu[0], sigma[0] = 0.0, 1.0
    sigma[sigma < 1e-12] = 1.0
    Z_std = (Z - mu) / sigma

    br = BayesianRidge(max_iter=1000, tol=1e-8, fit_intercept=False, compute_score=True)
    br.fit(Z_std, log_y)

    coef = br.coef_ / sigma
    coef[0] -= np.sum(br.coef_ * mu / sigma)
    emult = spec.epoch_multipliers

    def predict(W_new):
        sp = _as_3d(W_new)
        C = _broadcast_epoch_mult(emult, sp.shape[1], sp.shape[2])
        E = _compute_epochs(sp, C)
        w_flat = sp.reshape(sp.shape[0], -1)
        e_flat = np.log(E.reshape(sp.shape[0], -1) + EPS)
        Xn = np.column_stack([w_flat, e_flat])
        return np.exp(_quadratic_expand(Xn) @ coef)

    return predict, {"n_params": n_params}


# ---------------------------------------------------------------------------
# Model: ElasticLogQuad(w-e) — always applicable
# ---------------------------------------------------------------------------
def _fit_elastic_logquad_we(spec: DatasetSpec):
    from sklearn.linear_model import ElasticNetCV

    X = _flat_weight_epoch(spec)
    Z = _quadratic_expand(X)
    n_params = Z.shape[1]
    log_y = np.log(spec.y)

    mu, sigma = Z.mean(0), Z.std(0)
    mu[0], sigma[0] = 0.0, 1.0
    sigma[sigma < 1e-12] = 1.0
    Z_std = (Z - mu) / sigma

    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
        alphas=np.logspace(-5, 1, 50),
        cv=min(5, spec.R),
        fit_intercept=False,
        max_iter=10000,
    )
    enet.fit(Z_std, log_y)

    coef = enet.coef_ / sigma
    coef[0] -= np.sum(enet.coef_ * mu / sigma)
    emult = spec.epoch_multipliers

    def predict(W_new):
        sp = _as_3d(W_new)
        C = _broadcast_epoch_mult(emult, sp.shape[1], sp.shape[2])
        E = _compute_epochs(sp, C)
        w_flat = sp.reshape(sp.shape[0], -1)
        e_flat = np.log(E.reshape(sp.shape[0], -1) + EPS)
        Xn = np.column_stack([w_flat, e_flat])
        return np.exp(_quadratic_expand(Xn) @ coef)

    return predict, {"n_params": n_params}


# ---------------------------------------------------------------------------
# Model: HuberLogQuad(w-e) — always applicable
# ---------------------------------------------------------------------------
def _fit_huber_logquad_we(spec: DatasetSpec):
    from sklearn.linear_model import HuberRegressor

    X = _flat_weight_epoch(spec)
    Z = _quadratic_expand(X)
    n_params = Z.shape[1]
    log_y = np.log(spec.y)

    # Standardize features for stable Huber fitting
    mu, sigma = Z.mean(0), Z.std(0)
    mu[0], sigma[0] = 0.0, 1.0
    sigma[sigma < 1e-12] = 1.0
    Z_std = (Z - mu) / sigma

    log_y_median = np.median(log_y)
    mad = np.median(np.abs(log_y - log_y_median))
    eps_huber = max(1.35, 1.35 * mad / 0.6745)

    # Use small alpha for regularization in high-d
    alpha = 1e-4 if Z.shape[1] > Z.shape[0] // 2 else 0.0
    huber = HuberRegressor(epsilon=float(eps_huber), max_iter=50000, fit_intercept=False, alpha=alpha)
    huber.fit(Z_std, log_y)

    coef = huber.coef_ / sigma
    coef[0] -= np.sum(huber.coef_ * mu / sigma)
    emult = spec.epoch_multipliers

    def predict(W_new):
        sp = _as_3d(W_new)
        C = _broadcast_epoch_mult(emult, sp.shape[1], sp.shape[2])
        E = _compute_epochs(sp, C)
        w_flat = sp.reshape(sp.shape[0], -1)
        e_flat = np.log(E.reshape(sp.shape[0], -1) + EPS)
        Xn = np.column_stack([w_flat, e_flat])
        return np.exp(_quadratic_expand(Xn) @ coef)

    return predict, {"n_params": n_params}


# ---------------------------------------------------------------------------
# Model: BayesLogQuad(weight) — Bayesian Ridge on quadratic weight features (no epoch)
# ---------------------------------------------------------------------------
def _fit_bayes_logquad_weight(spec: DatasetSpec):
    from sklearn.linear_model import BayesianRidge

    X = _flat_weights(spec)
    Z = _quadratic_expand(X)
    n_params = Z.shape[1]
    log_y = np.log(spec.y)

    mu, sigma = Z.mean(0), Z.std(0)
    mu[0], sigma[0] = 0.0, 1.0
    sigma[sigma < 1e-12] = 1.0
    Z_std = (Z - mu) / sigma

    br = BayesianRidge(max_iter=1000, tol=1e-8, fit_intercept=False, compute_score=True)
    br.fit(Z_std, log_y)

    coef = br.coef_ / sigma
    coef[0] -= np.sum(br.coef_ * mu / sigma)

    def predict(W_new):
        sp = _as_3d(W_new)
        Xn = sp.reshape(sp.shape[0], -1)
        return np.exp(_quadratic_expand(Xn) @ coef)

    return predict, {"n_params": n_params}


# ---------------------------------------------------------------------------
# Model: ElasticLogQuad(weight) — ElasticNet on quadratic weight features (no epoch)
# ---------------------------------------------------------------------------
def _fit_elastic_logquad_weight(spec: DatasetSpec):
    from sklearn.linear_model import ElasticNetCV

    X = _flat_weights(spec)
    Z = _quadratic_expand(X)
    n_params = Z.shape[1]
    log_y = np.log(spec.y)

    mu, sigma = Z.mean(0), Z.std(0)
    mu[0], sigma[0] = 0.0, 1.0
    sigma[sigma < 1e-12] = 1.0
    Z_std = (Z - mu) / sigma

    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
        alphas=np.logspace(-5, 1, 50),
        cv=min(5, spec.R),
        fit_intercept=False,
        max_iter=10000,
    )
    enet.fit(Z_std, log_y)

    coef = enet.coef_ / sigma
    coef[0] -= np.sum(enet.coef_ * mu / sigma)

    def predict(W_new):
        sp = _as_3d(W_new)
        Xn = sp.reshape(sp.shape[0], -1)
        return np.exp(_quadratic_expand(Xn) @ coef)

    return predict, {"n_params": n_params}


# ---------------------------------------------------------------------------
# Model: SOE-Base
# ---------------------------------------------------------------------------
def _fit_soe_base(spec: DatasetSpec):
    Phi, feat_names, n_params = soe_base_design(
        spec.weights, spec.epoch_multipliers, small_domains=spec.small_domains
    )
    beta = _fit_ols(Phi, spec.y)
    emult = spec.epoch_multipliers
    sd = spec.small_domains

    def predict(W_new):
        Phi_new, _, _ = soe_base_design(W_new, emult, small_domains=sd)
        return Phi_new @ beta

    return predict, {"n_params": n_params, "features": feat_names}


def _applicable_soe_base(spec: DatasetSpec) -> bool:
    _, _, n = soe_base_design(spec.weights[:1], spec.epoch_multipliers, small_domains=spec.small_domains)
    return spec.R > n


# ---------------------------------------------------------------------------
# Model: SOE-Plus
# ---------------------------------------------------------------------------
def _fit_soe_plus(spec: DatasetSpec):
    Phi, feat_names, n_params = soe_plus_design(
        spec.weights, spec.epoch_multipliers, small_domains=spec.small_domains
    )
    beta = _fit_ols(Phi, spec.y)
    emult = spec.epoch_multipliers
    sd = spec.small_domains

    def predict(W_new):
        Phi_new, _, _ = soe_plus_design(W_new, emult, small_domains=sd)
        return Phi_new @ beta

    return predict, {"n_params": n_params, "features": feat_names}


def _applicable_soe_plus(spec: DatasetSpec) -> bool:
    _, _, n = soe_plus_design(spec.weights[:1], spec.epoch_multipliers, small_domains=spec.small_domains)
    return spec.R > n


# ---------------------------------------------------------------------------
# Model: SOE-Curric
# ---------------------------------------------------------------------------
def _fit_soe_curric(spec: DatasetSpec):
    Phi, feat_names, n_params = soe_curric_design(
        spec.weights, spec.epoch_multipliers, small_domains=spec.small_domains
    )
    beta = _fit_ols(Phi, spec.y)
    emult = spec.epoch_multipliers
    sd = spec.small_domains

    def predict(W_new):
        Phi_new, _, _ = soe_curric_design(W_new, emult, small_domains=sd)
        return Phi_new @ beta

    return predict, {"n_params": n_params, "features": feat_names}


def _applicable_soe_curric(spec: DatasetSpec) -> bool:
    _, _, n = soe_curric_design(spec.weights[:1], spec.epoch_multipliers, small_domains=spec.small_domains)
    return spec.R > n


# ---------------------------------------------------------------------------
# Model: SOE-Plus(ridge) — ridge-regularized SOE-Plus
# ---------------------------------------------------------------------------
def _fit_soe_plus_ridge(spec: DatasetSpec):
    Phi, feat_names, n_params = soe_plus_design(
        spec.weights, spec.epoch_multipliers, small_domains=spec.small_domains
    )
    beta = _fit_ridge(Phi, spec.y, ridge=1.0)
    emult = spec.epoch_multipliers
    sd = spec.small_domains

    def predict(W_new):
        Phi_new, _, _ = soe_plus_design(W_new, emult, small_domains=sd)
        return Phi_new @ beta

    return predict, {"n_params": n_params, "ridge": 1.0, "features": feat_names}


# ---------------------------------------------------------------------------
# Model: SOE-Curric(ridge) — ridge-regularized SOE-Curric
# ---------------------------------------------------------------------------
def _fit_soe_curric_ridge(spec: DatasetSpec):
    Phi, feat_names, n_params = soe_curric_design(
        spec.weights, spec.epoch_multipliers, small_domains=spec.small_domains
    )
    beta = _fit_ridge(Phi, spec.y, ridge=1.0)
    emult = spec.epoch_multipliers
    sd = spec.small_domains

    def predict(W_new):
        Phi_new, _, _ = soe_curric_design(W_new, emult, small_domains=sd)
        return Phi_new @ beta

    return predict, {"n_params": n_params, "ridge": 1.0, "features": feat_names}


# ---------------------------------------------------------------------------
# Model: Threshold Overfit (general)
# ---------------------------------------------------------------------------
def _build_threshold_design_general(
    W: np.ndarray,
    epoch_multipliers: np.ndarray,
    tau: float,
    small_domains: list[int] | None = None,
) -> np.ndarray:
    W = _as_3d(W)
    R, N, M = W.shape
    C = _broadcast_epoch_mult(epoch_multipliers, N, M)
    E = _compute_epochs(W, C)
    G = np.log(E + EPS)
    S = _parse_small(small_domains, M)

    cols: list[np.ndarray] = [np.ones(R)]
    # satiety for all (k, d)
    for k in range(N):
        for d in range(M):
            cols.append(G[:, k, d])
    # linear epoch + hinge + squared hinge for small domains
    for k in range(N):
        for d in S:
            cols.append(E[:, k, d])
            h = np.maximum(0.0, E[:, k, d] - tau)
            cols.append(h)
            cols.append(h**2)
    # cross-phase reuse for small domains
    for d in S:
        for k1, k2 in _phase_pairs(N, "adjacent"):
            cols.append(E[:, k1, d] * E[:, k2, d])
    return np.column_stack(cols)


def _threshold_nparams(spec: DatasetSpec) -> int:
    N, M = spec.N, spec.M
    S = len(_parse_small(spec.small_domains, M))
    pairs = len(_phase_pairs(N, "adjacent"))
    return 1 + N * M + 3 * S * N + S * pairs


def _fit_threshold_overfit(spec: DatasetSpec):
    # Determine tau search range from max epoch across training data
    C = _broadcast_epoch_mult(spec.epoch_multipliers, spec.N, spec.M)
    E = _compute_epochs(spec.weights, C)
    max_epoch = float(E.max())
    tau_grid = np.linspace(0.5, max(1.0, max_epoch * 0.95), 80)

    best_tau, best_beta, best_sse = 0.0, None, float("inf")
    emult = spec.epoch_multipliers
    sd = spec.small_domains
    for tau in tau_grid:
        Phi = _build_threshold_design_general(spec.weights, emult, tau, sd)
        beta = _fit_ols(Phi, spec.y)
        sse = float(np.sum((Phi @ beta - spec.y) ** 2))
        if sse < best_sse:
            best_sse, best_tau, best_beta = sse, float(tau), beta

    n_params = _threshold_nparams(spec) + 1  # +1 for tau

    def predict(W_new):
        Phi_new = _build_threshold_design_general(W_new, emult, best_tau, sd)
        return Phi_new @ best_beta

    return predict, {"n_params": n_params, "tau": best_tau}


def _applicable_threshold(spec: DatasetSpec) -> bool:
    return spec.R > _threshold_nparams(spec)


# ---------------------------------------------------------------------------
# Model: Scheffé+log (general, on vdom features)
# ---------------------------------------------------------------------------
def _scheffe_log_features(V: np.ndarray) -> np.ndarray:
    """Scheffé+log design: linear + pairwise + x*ln(x) terms on volumes."""
    R, d = V.shape
    safe_V = np.maximum(V, 1e-10)
    parts: list[np.ndarray] = []
    for j in range(d):
        parts.append(V[:, j : j + 1])
    for j in range(d):
        for k in range(j + 1, d):
            parts.append((V[:, j] * V[:, k]).reshape(-1, 1))
    for j in range(d):
        parts.append((safe_V[:, j] * np.log(safe_V[:, j])).reshape(-1, 1))
    return np.hstack(parts)


def _scheffe_nparams(d: int) -> int:
    return d + d * (d - 1) // 2 + d


def _fit_scheffe_log(spec: DatasetSpec):
    V = _vdom_features(spec)
    d = V.shape[1]
    n_params = _scheffe_nparams(d)

    Z = _scheffe_log_features(V)

    if spec.R > n_params:
        coef = _fit_ols(Z, spec.y)
    else:
        # Fall back to Ridge for high-d
        from sklearn.linear_model import RidgeCV

        mu, sigma = Z.mean(0), Z.std(0)
        sigma[sigma < 1e-12] = 1.0
        Z_std = (Z - mu) / sigma
        ridge = RidgeCV(alphas=np.logspace(-4, 4, 50), fit_intercept=False)
        ridge.fit(Z_std, spec.y)
        coef = ridge.coef_ / sigma
        coef -= ridge.coef_ * mu / sigma

    phase_fracs = np.ones(spec.N) / spec.N

    def predict(W_new):
        sp = _as_3d(W_new)
        Vn = (sp * phase_fracs[None, :, None]).reshape(sp.shape[0], -1)
        return _scheffe_log_features(Vn) @ coef

    return predict, {"n_params": n_params}


# ---------------------------------------------------------------------------
# Model: CES (general, on vdom features)
# ---------------------------------------------------------------------------
def _fit_ces(spec: DatasetSpec, n_restarts: int = 40, seed: int = 42):
    V = _vdom_features(spec)
    y = spec.y
    rng = np.random.default_rng(seed)
    d = V.shape[1]
    n_params = 3 + d  # C, log_A, rho, log_a[d]

    def model(Vmat, p):
        C, log_A, rho = p[0], p[1], np.clip(p[2], -10, 0.99)
        log_a = p[3 : 3 + d]
        a = np.exp(log_a)
        a = a / a.sum()
        inner = np.sum(a * np.power(np.maximum(Vmat, 1e-10), rho), axis=1)
        return C - np.exp(log_A) * np.power(np.maximum(inner, 1e-10), 1.0 / rho)

    def loss(p):
        return float(np.sum((model(V, p) - y) ** 2))

    best_l, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = [y.max() + rng.normal(0, 0.1), rng.normal(-1, 1), rng.uniform(-5, 0.5)]
        p0.extend(rng.normal(0, 0.5, d).tolist())
        try:
            res = minimize(loss, np.array(p0), method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-10})
            if res.fun < best_l:
                best_l, best_p = res.fun, res.x
        except Exception:
            continue

    if best_p is None:
        raise RuntimeError("CES optimization failed to converge")

    phase_fracs = np.ones(spec.N) / spec.N
    final_p = best_p

    def predict(W_new):
        sp = _as_3d(W_new)
        Vn = (sp * phase_fracs[None, :, None]).reshape(sp.shape[0], -1)
        return model(Vn, final_p)

    return predict, {"n_params": n_params}


def _applicable_ces(spec: DatasetSpec) -> bool:
    d = spec.N * spec.M
    return spec.R > 3 + d


# ---------------------------------------------------------------------------
# Model list
# ---------------------------------------------------------------------------
GENERAL_MODELS: list[GeneralModelSpec] = [
    GeneralModelSpec("Linear", _fit_linear, _applicable_linear, "OLS linear on weights"),
    GeneralModelSpec("Quadratic", _fit_quadratic_weight, _applicable_quadratic_weight, "OLS quadratic on weights"),
    GeneralModelSpec("Quadratic(w-e)", _fit_quadratic_we, _applicable_quadratic_we, "OLS quadratic on weight+epoch"),
    GeneralModelSpec("RidgeQuad(w-e)", _fit_ridge_quad_we, lambda _: True, "Ridge quadratic on weight+epoch"),
    GeneralModelSpec("LogQuad(w-e)", _fit_logquad_we, _applicable_quadratic_we, "exp(OLS quad) on weight+epoch"),
    GeneralModelSpec("BayesLogQuad(w-e)", _fit_bayes_logquad_we, lambda _: True, "Bayesian Ridge logquad"),
    GeneralModelSpec("ElasticLogQuad(w-e)", _fit_elastic_logquad_we, lambda _: True, "ElasticNet logquad"),
    GeneralModelSpec("HuberLogQuad(w-e)", _fit_huber_logquad_we, lambda _: True, "Huber logquad"),
    GeneralModelSpec("BayesLogQuad(wt)", _fit_bayes_logquad_weight, _always_applicable, "Bayesian Ridge logquad on weights only"),
    GeneralModelSpec("ElasticLogQuad(wt)", _fit_elastic_logquad_weight, _always_applicable, "ElasticNet logquad on weights only"),
    GeneralModelSpec("SOE-Base", _fit_soe_base, _applicable_soe_base, "Satiety+overfit, OLS"),
    GeneralModelSpec("SOE-Plus", _fit_soe_plus, _applicable_soe_plus, "SOE-Base + within-phase coupling"),
    GeneralModelSpec("SOE-Plus(ridge)", _fit_soe_plus_ridge, _applicable_soe_plus, "SOE-Plus with ridge=1"),
    GeneralModelSpec("SOE-Curric", _fit_soe_curric, _applicable_soe_curric, "Full curriculum interactions"),
    GeneralModelSpec("SOE-Curric(ridge)", _fit_soe_curric_ridge, _applicable_soe_curric, "SOE-Curric with ridge=1"),
    GeneralModelSpec(
        "Threshold Overfit", _fit_threshold_overfit, _applicable_threshold, "Hinge penalty at learned threshold"
    ),
    GeneralModelSpec("Scheffé+log", _fit_scheffe_log, lambda _: True, "Scheffé polynomial + log terms"),
    GeneralModelSpec("CES", _fit_ces, _applicable_ces, "Constant Elasticity of Substitution"),
]
