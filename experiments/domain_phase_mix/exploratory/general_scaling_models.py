# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
from typing import Literal
from collections.abc import Callable

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

    def subset(self, idx: np.ndarray) -> DatasetSpec:
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
# Model: LogLinear (weight-only) — exp(OLS linear on weights)
# ---------------------------------------------------------------------------
def _fit_loglinear(spec: DatasetSpec):
    X = np.column_stack([np.ones(spec.R), _flat_weights(spec)])
    log_y = np.log(spec.y)
    beta = _fit_ols(X, log_y)
    n_params = X.shape[1]

    def predict(W_new):
        sp = _as_3d(W_new)
        Xn = np.column_stack([np.ones(sp.shape[0]), sp.reshape(sp.shape[0], -1)])
        return np.exp(Xn @ beta)

    return predict, {"n_params": n_params}


# ---------------------------------------------------------------------------
# Model: LogLinear(w-e) — exp(OLS linear on weight+epoch features)
# ---------------------------------------------------------------------------
def _fit_loglinear_we(spec: DatasetSpec):
    X = np.column_stack([np.ones(spec.R), _flat_weight_epoch(spec)])
    log_y = np.log(spec.y)
    beta = _fit_ols(X, log_y)
    n_params = X.shape[1]
    emult = spec.epoch_multipliers

    def predict(W_new):
        sp = _as_3d(W_new)
        C = _broadcast_epoch_mult(emult, sp.shape[1], sp.shape[2])
        E = _compute_epochs(sp, C)
        w_flat = sp.reshape(sp.shape[0], -1)
        e_flat = np.log(E.reshape(sp.shape[0], -1) + EPS)
        Xn = np.column_stack([np.ones(sp.shape[0]), w_flat, e_flat])
        return np.exp(Xn @ beta)

    return predict, {"n_params": n_params}


def _applicable_loglinear_we(spec: DatasetSpec) -> bool:
    return spec.R > 2 * spec.N * spec.M + 1


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
    Phi, feat_names, n_params = soe_base_design(spec.weights, spec.epoch_multipliers, small_domains=spec.small_domains)
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
    Phi, feat_names, n_params = soe_plus_design(spec.weights, spec.epoch_multipliers, small_domains=spec.small_domains)
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
    Phi, feat_names, n_params = soe_curric_design(spec.weights, spec.epoch_multipliers, small_domains=spec.small_domains)
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
    Phi, feat_names, n_params = soe_plus_design(spec.weights, spec.epoch_multipliers, small_domains=spec.small_domains)
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
    Phi, feat_names, n_params = soe_curric_design(spec.weights, spec.epoch_multipliers, small_domains=spec.small_domains)
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
    """Scheffé + (w log w) mixture model.

    Notes
    -----
    * Uses v-dom fractions V (N*M features on the simplex: sum(V)=1).
    * Always fits an intercept (either via explicit column in OLS, or via
      RidgeCV(fit_intercept=True) in the underdetermined regime).
    """
    V = _vdom_features(spec)
    d = V.shape[1]

    Z = _scheffe_log_features(V)  # (R, d_features)
    n_params = 1 + Z.shape[1]  # intercept + coefficients

    if spec.R > n_params:
        X = np.column_stack([np.ones(spec.R), Z])
        beta = _fit_ols(X, spec.y)
        intercept, coef = float(beta[0]), beta[1:]
    else:
        from sklearn.linear_model import RidgeCV

        ridge = RidgeCV(alphas=np.logspace(-4, 4, 50), fit_intercept=True)
        ridge.fit(Z, spec.y)
        intercept, coef = float(ridge.intercept_), ridge.coef_

    phase_fracs = np.ones(spec.N) / spec.N

    def predict(W_new):
        sp = _as_3d(W_new)
        Vn = (sp * phase_fracs[None, :, None]).reshape(sp.shape[0], -1)
        return intercept + _scheffe_log_features(Vn) @ coef

    return predict, {"n_params": n_params}
def _epoch_entropy_features(spec: DatasetSpec) -> np.ndarray:
    """(R, N) epoch-entropy per phase: H_k = sum_d q_{k,d} * ln(q_{k,d})."""
    C = _broadcast_epoch_mult(spec.epoch_multipliers, spec.N, spec.M)
    E = _compute_epochs(spec.weights, C)  # (R, N, M)
    denom = E.sum(axis=2, keepdims=True)  # (R, N, 1)
    q = np.clip(E / np.maximum(denom, 1e-10), 1e-10, 1.0)
    H = np.sum(q * np.log(q), axis=2)  # (R, N)
    return H


def _fit_scheffe_log_epoch_entropy(spec: DatasetSpec):
    """Scheffé+log with an added *epoch-entropy* term per phase.

    Uses epoch-based entropy:
      H_k = -sum_d q_{k,d} log q_{k,d},  q_{k,d} ∝ w_{k,d} * C_{k,d}.
    """
    V = _vdom_features(spec)

    Z_s = _scheffe_log_features(V)
    H = _epoch_entropy_features(spec)  # (R, N)
    Z = np.column_stack([Z_s, H])
    n_params = 1 + Z.shape[1]

    if spec.R > n_params:
        X = np.column_stack([np.ones(spec.R), Z])
        beta = _fit_ols(X, spec.y)
        intercept, coef = float(beta[0]), beta[1:]
    else:
        from sklearn.linear_model import RidgeCV

        ridge = RidgeCV(alphas=np.logspace(-4, 4, 50), fit_intercept=True)
        ridge.fit(Z, spec.y)
        intercept, coef = float(ridge.intercept_), ridge.coef_

    phase_fracs = np.ones(spec.N) / spec.N

    def predict(W_new):
        sp = _as_3d(W_new)
        Vn = (sp * phase_fracs[None, :, None]).reshape(sp.shape[0], -1)
        Zsn = _scheffe_log_features(Vn)

        tmp_spec = DatasetSpec(
            weights=sp,
            y=np.zeros(sp.shape[0]),
            epoch_multipliers=spec.epoch_multipliers,
            domain_names=spec.domain_names,
            phase_names=spec.phase_names,
            small_domains=spec.small_domains,
            name=spec.name,
        )
        Hn = _epoch_entropy_features(tmp_spec)
        Zn = np.column_stack([Zsn, Hn])
        return intercept + Zn @ coef

    return predict, {"n_params": n_params}
def _scheffe_tied_entropy_features(V: np.ndarray, N: int, M: int) -> np.ndarray:
    """Scheffé design with phase-tied entropy: linear + pairwise + N phase-tied entropy terms."""
    d = V.shape[1]
    safe_V = np.maximum(V, 1e-10)
    parts: list[np.ndarray] = []
    for j in range(d):
        parts.append(V[:, j : j + 1])
    for j in range(d):
        for k in range(j + 1, d):
            parts.append((V[:, j] * V[:, k]).reshape(-1, 1))
    for ph in range(N):
        cols = safe_V[:, ph * M : (ph + 1) * M]
        ent = np.sum(cols * np.log(cols), axis=1, keepdims=True)
        parts.append(ent)
    return np.hstack(parts)


def _scheffe_tied_nparams(d: int, N: int) -> int:
    return d + d * (d - 1) // 2 + N


def _fit_scheffe_tied_entropy(spec: DatasetSpec):
    """Scheffé+log + tied epoch-entropy + adjacent epoch-symKL.

    Features:
      - Scheffé+log mixture terms on v-dom fractions V
      - sum_k H_k where H_k is epoch-entropy per phase
      - sum_k symKL(q_k, q_{k+1}) where q_k is the epoch distribution over domains
        in phase k (curriculum smoothness)
    """
    W = spec.weights
    V = _vdom_features(spec)

    Z_s = _scheffe_log_features(V)
    H_sum = _epoch_entropy_features(spec).sum(axis=1, keepdims=True)  # (R,1)

    C = _broadcast_epoch_mult(spec.epoch_multipliers, spec.N, spec.M)

    def _adjacent_epoch_symkl(W_in: np.ndarray) -> np.ndarray:
        eps = 1e-12
        E = _compute_epochs(W_in, C)
        q = E / (E.sum(axis=2, keepdims=True) + eps)  # (R,N,M)
        Rn, Nn, Mn = q.shape
        out = np.zeros(Rn)
        if Nn <= 1:
            return out
        for k in range(Nn - 1):
            p = np.clip(q[:, k, :], eps, 1.0)
            r = np.clip(q[:, k + 1, :], eps, 1.0)
            kl_pr = np.sum(p * (np.log(p) - np.log(r)), axis=1)
            kl_rp = np.sum(r * (np.log(r) - np.log(p)), axis=1)
            out += 0.5 * (kl_pr + kl_rp)
        return out

    KL_adj = _adjacent_epoch_symkl(W).reshape(-1, 1)

    Z = np.column_stack([Z_s, H_sum, KL_adj])
    n_params = 1 + Z.shape[1]

    if spec.R > n_params:
        X = np.column_stack([np.ones(spec.R), Z])
        beta = _fit_ols(X, spec.y)
        intercept, coef = float(beta[0]), beta[1:]
    else:
        from sklearn.linear_model import RidgeCV

        ridge = RidgeCV(alphas=np.logspace(-4, 4, 50), fit_intercept=True)
        ridge.fit(Z, spec.y)
        intercept, coef = float(ridge.intercept_), ridge.coef_

    phase_fracs = np.ones(spec.N) / spec.N

    def predict(W_new):
        sp = _as_3d(W_new)
        Vn = (sp * phase_fracs[None, :, None]).reshape(sp.shape[0], -1)
        Zsn = _scheffe_log_features(Vn)

        tmp_spec = DatasetSpec(
            weights=sp,
            y=np.zeros(sp.shape[0]),
            epoch_multipliers=spec.epoch_multipliers,
            domain_names=spec.domain_names,
            phase_names=spec.phase_names,
            small_domains=spec.small_domains,
            name=spec.name,
        )
        Hs = _epoch_entropy_features(tmp_spec).sum(axis=1, keepdims=True)
        KL = _adjacent_epoch_symkl(sp).reshape(-1, 1)
        Zn = np.column_stack([Zsn, Hs, KL])
        return intercept + Zn @ coef

    return predict, {"n_params": n_params}
def _epoch_overfit_quadratic_raw(
    W: np.ndarray,
    epoch_multipliers: np.ndarray,
    small_domains: list[int] | None,
    N: int,
    M: int,
) -> tuple[np.ndarray, int]:
    """Epoch² + cross-phase epoch products for small domains.

    For each small domain d: N per-phase e²_{k,d} + S cross-phase e_{k1,d}*e_{k2,d}.
    Returns (features, n_cols).
    """
    W3d = _as_3d(W)
    R = W3d.shape[0]
    C = _broadcast_epoch_mult(epoch_multipliers, N, M)
    E = _compute_epochs(W3d, C)
    S_list = _parse_small(small_domains, M)
    pairs = _phase_pairs(N, "adjacent")
    cols: list[np.ndarray] = []
    for k in range(N):
        for d in S_list:
            cols.append(E[:, k, d] ** 2)
    for d in S_list:
        for k1, k2 in pairs:
            cols.append(E[:, k1, d] * E[:, k2, d])
    n_cols = len(cols)
    features = np.column_stack(cols) if cols else np.empty((R, 0))
    return features, n_cols


def _epoch_overfit_nparams(N: int, M: int, small_domains: list[int] | None) -> int:
    S = len(_parse_small(small_domains, M))
    P = len(_phase_pairs(N, "adjacent"))
    return N * S + S * P


def _fit_sheq(spec: DatasetSpec):
    """SHEQ: Scheffé+log + tied epoch-entropy + quadratic epoch overfit (+ reuse)."""
    W = spec.weights
    V = _vdom_features(spec)

    Z_s = _scheffe_log_features(V)

    # epoch-entropy (tied across phases)
    H_sum = _epoch_entropy_features(spec).sum(axis=1, keepdims=True)

    # epoch overfit features for small domains only (quadratic + adjacent reuse)
    Z_eq, _ = _epoch_overfit_quadratic_raw(
        W,
        spec.epoch_multipliers,
        spec.small_domains,
        spec.N,
        spec.M,
    )

    Z = np.column_stack([Z_s, H_sum, Z_eq])
    n_params = 1 + Z.shape[1]

    if spec.R > n_params:
        X = np.column_stack([np.ones(spec.R), Z])
        beta = _fit_ols(X, spec.y)
        intercept, coef = float(beta[0]), beta[1:]
    else:
        from sklearn.linear_model import RidgeCV

        ridge = RidgeCV(alphas=np.logspace(-4, 4, 50), fit_intercept=True)
        ridge.fit(Z, spec.y)
        intercept, coef = float(ridge.intercept_), ridge.coef_

    phase_fracs = np.ones(spec.N) / spec.N

    def predict(W_new):
        sp = _as_3d(W_new)
        Vn = (sp * phase_fracs[None, :, None]).reshape(sp.shape[0], -1)
        Zsn = _scheffe_log_features(Vn)

        tmp_spec = DatasetSpec(
            weights=sp,
            y=np.zeros(sp.shape[0]),
            epoch_multipliers=spec.epoch_multipliers,
            domain_names=spec.domain_names,
            phase_names=spec.phase_names,
            small_domains=spec.small_domains,
            name=spec.name,
        )
        Hs = _epoch_entropy_features(tmp_spec).sum(axis=1, keepdims=True)

        Z_eqn, _ = _epoch_overfit_quadratic_raw(
            sp,
            spec.epoch_multipliers,
            spec.small_domains,
            spec.N,
            spec.M,
        )

        Zn = np.column_stack([Zsn, Hs, Z_eqn])
        return intercept + Zn @ coef

    return predict, {"n_params": n_params}
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
# Model: CES-Overfit (per-phase CES on log1p epochs + overtraining penalty)
# ---------------------------------------------------------------------------
def _softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def _fit_ces_overfit(spec: DatasetSpec, n_restarts: int = 15, seed: int = 0):
    """CES-Overfit: per-phase CES on log1p(epochs) + softplus overfit penalty.

    L = C - sum_k A_k * CES_k(log1p(e)) + sum_{d in S} beta_d * softplus(E_d - tau_d)^2

    Parameters: C (1) + per-phase logA_k, rho_k, (M-1) logit shares (N*(M+1)) +
                per-small-domain logbeta_d, logtau_d (2*S).
    """
    rng = np.random.default_rng(seed)
    W = spec.weights  # (R, N, M)
    y = spec.y
    _, N, M = W.shape
    C_mult = _broadcast_epoch_mult(spec.epoch_multipliers, N, M)
    E = _compute_epochs(W, C_mult)  # (R, N, M)
    S_list = _parse_small(spec.small_domains, M)
    S = len(S_list)

    n_per_phase = 2 + (M - 1)  # logA, rho, (M-1) logit shares
    n_params = 1 + N * n_per_phase + 2 * S

    def _softmax(logits):
        x = logits - np.max(logits)
        e = np.exp(x)
        return e / e.sum()

    def model(E_data, p):
        result = np.full(E_data.shape[0], p[0])  # C
        idx = 1
        for k in range(N):
            logA = p[idx]
            rho = np.clip(p[idx + 1], -10, 0.99)
            logit_a = np.concatenate([p[idx + 2 : idx + 2 + (M - 1)], [0.0]])
            a = _softmax(logit_a)
            idx += n_per_phase

            sat = np.maximum(np.log1p(E_data[:, k, :]), 1e-10)  # (R, M)
            inner = np.sum(a[None, :] * np.power(sat, rho), axis=1)
            ces_val = np.power(np.maximum(inner, 1e-10), 1.0 / rho)
            result -= np.exp(np.clip(logA, -20, 20)) * ces_val

        for d in S_list:
            logbeta = p[idx]
            logtau = p[idx + 1]
            idx += 2
            beta = np.exp(np.clip(logbeta, -10, 10))
            tau = np.exp(np.clip(logtau, -5, 5))
            E_d_total = E_data[:, :, d].sum(axis=1)
            result += beta * _softplus(E_d_total - tau) ** 2

        return result

    def loss(p):
        r = model(E, p) - y
        return float(np.sum(np.clip(r * r, 0, 1e10)))

    # Precompute median total epochs for small domains (for tau init)
    med_epochs = {d: float(np.median(E[:, :, d].sum(axis=1))) + 1e-6 for d in S_list}

    best_l, best_p = np.inf, None
    for _ in range(n_restarts):
        p0_list: list[float] = [float(y.max()) + rng.normal(0, 0.05)]
        for _k in range(N):
            p0_list.append(float(rng.normal(0, 1)))  # logA
            p0_list.append(float(rng.uniform(-2, 0.8)))  # rho
            p0_list.extend(rng.normal(0, 0.5, M - 1).tolist())  # logit shares
        for d in S_list:
            p0_list.append(float(rng.normal(-3, 1)))  # logbeta
            p0_list.append(float(np.log(med_epochs[d]) + rng.normal(0, 0.5)))  # logtau
        try:
            res = minimize(
                loss,
                np.array(p0_list),
                method="L-BFGS-B",
                options={"maxiter": 800, "ftol": 1e-10},
            )
            if np.isfinite(res.fun) and res.fun < best_l:
                best_l, best_p = res.fun, res.x
        except Exception:
            continue

    if best_p is None:
        raise RuntimeError("CES-Overfit optimization failed to converge")

    emult = spec.epoch_multipliers
    final_p = best_p

    def predict(W_new):
        sp = _as_3d(W_new)
        C_new = _broadcast_epoch_mult(emult, sp.shape[1], sp.shape[2])
        E_new = _compute_epochs(sp, C_new)
        return model(E_new, final_p)

    return predict, {"n_params": n_params}


def _applicable_ces_overfit(spec: DatasetSpec) -> bool:
    S = len(_parse_small(spec.small_domains, spec.M))
    n_params = 1 + spec.N * (spec.M + 1) + 2 * S
    return spec.R > n_params



# ---------------------------------------------------------------------------
# CEQ family: low-parameter CES-like models with explicit overtraining penalty
# ---------------------------------------------------------------------------

def _mad_sigma(x: np.ndarray) -> float:
    """Robust scale estimate via MAD."""
    x = np.asarray(x, float)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad < 1e-12:
        return float(np.std(x) + 1e-12)
    return mad / 0.6745


def _huber_delta(y: np.ndarray) -> float:
    """Delta for (pseudo-)Huber based on robust scale, with sane bounds."""
    sigma = _mad_sigma(y)
    return float(np.clip(1.345 * sigma, 0.02, 0.30))


def _pseudo_huber(resid: np.ndarray, delta: float) -> np.ndarray:
    resid = np.asarray(resid, float)
    d = float(delta)
    return d * d * (np.sqrt(1.0 + (resid / d) ** 2) - 1.0)


def _softplus_scaled(x: np.ndarray, k: float) -> np.ndarray:
    """softplus(k*x)/k, with k>0; k->inf recovers hinge."""
    kk = float(max(k, 1e-8))
    return _softplus(kk * x) / kk


def _ces_mean_stable(X: np.ndarray, w: np.ndarray, rho: float) -> np.ndarray:
    """Stable CES mean over the last axis."""
    X = np.maximum(X, 1e-12)
    r = float(rho)
    if abs(r) < 1e-4:
        return np.exp(np.sum(w * np.log(X), axis=-1))
    inner = np.sum(w * np.power(X, r), axis=-1)
    inner = np.maximum(inner, 1e-12)
    return np.power(inner, 1.0 / r)


def _phase_softmax_weights(gamma: float, N: int) -> np.ndarray:
    if N <= 1:
        return np.ones(1)
    t = np.linspace(0.0, 1.0, N)
    z = gamma * t
    z = z - np.max(z)
    w = np.exp(z)
    return w / w.sum()


def _phase_quad_weights(beta1: float, beta2: float, N: int) -> np.ndarray:
    """Quadratic-softmax phase weights: pi_k proportional to exp(beta1*t + beta2*t^2).

    Recovers linear softmax when beta2=0.  beta2<0 gives a mid-phase hump,
    beta2>0 gives a U-shape.  Removes the constant adjacent-ratio restriction
    of the 1-parameter gamma-softmax.
    """
    if N <= 1:
        return np.ones(1)
    t = np.linspace(0.0, 1.0, N)
    z = beta1 * t + beta2 * (t * t)
    z = z - np.max(z)
    w = np.exp(z)
    return w / w.sum()


def _phase_beta_weights(log_alpha: float, log_beta: float, N: int, eps: float = 1e-3) -> np.ndarray:
    """Beta-distribution phase weights: pi_k proportional to t^(alpha-1) * (1-t)^(beta-1).

    alpha=beta=1 is uniform.  alpha>1,beta=1 is increasing.
    alpha>1,beta>1 gives a mid-phase peak (warmup/decay schedules).
    alpha<1,beta<1 is endpoint-heavy.
    """
    if N <= 1:
        return np.ones(1)
    t = np.linspace(0.0, 1.0, N)
    tt = np.clip(t, eps, 1.0 - eps)
    alpha = float(np.exp(np.clip(log_alpha, -5.0, 5.0)))
    beta = float(np.exp(np.clip(log_beta, -5.0, 5.0)))
    logw = (alpha - 1.0) * np.log(tt) + (beta - 1.0) * np.log(1.0 - tt)
    logw = logw - np.max(logw)
    w = np.exp(logw)
    return w / w.sum()


def _unpack_pi(p: np.ndarray, idx: int, N: int, pi_type: str):
    """Unpack phase weight parameters and return (pi, new_idx)."""
    if pi_type == "linear":
        gamma = float(p[idx])
        return _phase_softmax_weights(gamma, N), idx + 1
    elif pi_type == "quad":
        beta1 = float(p[idx])
        beta2 = float(p[idx + 1])
        return _phase_quad_weights(beta1, beta2, N), idx + 2
    elif pi_type == "beta":
        log_alpha = float(p[idx])
        log_beta = float(p[idx + 1])
        return _phase_beta_weights(log_alpha, log_beta, N), idx + 2
    else:
        raise ValueError(f"Unknown pi_type: {pi_type}")


def _init_pi(p0: np.ndarray, idx: int, rng: np.random.Generator, pi_type: str) -> int:
    """Initialize phase weight parameters in p0, return new idx."""
    if pi_type == "linear":
        p0[idx] = float(rng.normal(0.0, 1.0))
        return idx + 1
    elif pi_type == "quad":
        p0[idx] = float(rng.normal(0.0, 1.0))      # beta1
        p0[idx + 1] = float(rng.normal(0.0, 0.5))   # beta2
        return idx + 2
    elif pi_type == "beta":
        p0[idx] = float(rng.normal(0.0, 0.5))       # log_alpha
        p0[idx + 1] = float(rng.normal(0.0, 0.5))   # log_beta
        return idx + 2
    else:
        raise ValueError(f"Unknown pi_type: {pi_type}")


def _n_pi_params(pi_type: str) -> int:
    """Number of parameters for a given phase weight type."""
    if pi_type == "linear":
        return 1
    elif pi_type in ("quad", "beta"):
        return 2
    raise ValueError(f"Unknown pi_type: {pi_type}")


def _fit_ceq_sum(
    spec: DatasetSpec,
    *,
    nested: bool,
    learn_k: bool,
    k_fixed: float,
    n_restarts: int,
    seed: int,
    maxiter: int,
    reg: float,
):
    """Core CEQ fitter (CEQ-SUM and NCEQ)."""
    rng = np.random.default_rng(seed)
    W = spec.weights
    y = spec.y
    R, N, M = W.shape

    # Epoch exposures and satiety transform.
    C = _broadcast_epoch_mult(spec.epoch_multipliers, N, M)
    E = _compute_epochs(W, C)  # (R,N,M)
    sat = np.maximum(np.log1p(E), 1e-12)

    S_list = _parse_small(spec.small_domains, M)
    E_small_total = E[:, :, S_list].sum(axis=(1, 2))

    delta = _huber_delta(y)

    n_params = 3 + (M - 1) + 1 + 1 + 1 + (1 if learn_k else 0) + (1 if nested else 0)

    def _softmax_logits(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        e = np.exp(z)
        return e / e.sum()

    def unpack(p: np.ndarray):
        idx = 0
        c0 = float(p[idx])
        logA = float(p[idx + 1])
        logB = float(p[idx + 2])
        idx += 3

        logits = np.zeros(M)
        if M > 1:
            logits[: M - 1] = p[idx : idx + (M - 1)]
            idx += M - 1
        a = _softmax_logits(logits)

        rho = float(np.clip(5.0 * np.tanh(p[idx]), -10.0, 0.99))
        idx += 1

        gamma = float(p[idx])
        idx += 1

        tau = float(np.exp(np.clip(p[idx], -5.0, 8.0)))
        idx += 1

        if learn_k:
            k = float(np.exp(np.clip(p[idx], -5.0, 8.0)))
            idx += 1
        else:
            k = float(k_fixed)

        if nested:
            rho_p = float(np.clip(5.0 * np.tanh(p[idx]), -10.0, 0.99))
            idx += 1
        else:
            rho_p = None

        return c0, logA, logB, a, rho, gamma, tau, k, rho_p

    def forward(p: np.ndarray, W_in: np.ndarray) -> np.ndarray:
        c0, logA, logB, a, rho, gamma, tau, k, rho_p = unpack(p)
        A = float(np.exp(np.clip(logA, -10.0, 10.0)))
        B = float(np.exp(np.clip(logB, -10.0, 10.0)))

        E_in = _compute_epochs(W_in, C)
        sat_in = np.maximum(np.log1p(E_in), 1e-12)
        E_small = E_in[:, :, S_list].sum(axis=(1, 2))

        U_k = np.zeros((W_in.shape[0], N))
        for ph in range(N):
            U_k[:, ph] = _ces_mean_stable(sat_in[:, ph, :], a[None, :], rho)

        pi = _phase_softmax_weights(gamma, N)

        if nested:
            if abs(float(rho_p)) < 1e-4:
                U = np.exp(np.sum(pi[None, :] * np.log(np.maximum(U_k, 1e-12)), axis=1))
            else:
                inner = np.sum(pi[None, :] * np.power(np.maximum(U_k, 1e-12), float(rho_p)), axis=1)
                U = np.power(np.maximum(inner, 1e-12), 1.0 / float(rho_p))
        else:
            U = U_k @ pi

        h = _softplus_scaled(E_small - tau, k)
        P = h * h

        return c0 - A * U + B * P

    def obj(p: np.ndarray) -> float:
        yhat = forward(p, W)
        loss = float(np.sum(_pseudo_huber(yhat - y, delta)))
        loss += float(reg) * float(np.sum(p * p))
        return loss

    medE = float(np.median(E_small_total) + 1e-6)
    best_val, best_p = np.inf, None

    for r in range(n_restarts):
        p0 = np.zeros(n_params)
        idx = 0
        p0[idx] = float(np.median(y))
        p0[idx + 1] = float(rng.normal(0.0, 1.0))   # logA
        p0[idx + 2] = float(rng.normal(-2.0, 1.0))  # logB
        idx += 3

        if M > 1:
            p0[idx : idx + (M - 1)] = rng.normal(0.0, 0.5, M - 1)
            idx += M - 1

        p0[idx] = float(rng.normal(0.0, 0.7))  # rho raw
        idx += 1
        p0[idx] = float(rng.normal(0.0, 1.0))  # gamma
        idx += 1
        p0[idx] = float(np.log(medE) + rng.normal(0.0, 0.3))  # logtau
        idx += 1
        if learn_k:
            p0[idx] = float(np.log(10.0) + rng.normal(0.0, 0.4))  # logk
            idx += 1
        if nested:
            p0[idx] = float(rng.normal(0.0, 0.7))  # rho_p raw
            idx += 1

        try:
            res = minimize(obj, p0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-10})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_p = float(res.fun), res.x
        except Exception:
            continue

    if best_p is None:
        raise RuntimeError("CEQ optimization failed to converge")

    final_p = best_p.copy()

    def predict(W_new: np.ndarray) -> np.ndarray:
        sp = _as_3d(W_new)
        return forward(final_p, sp)

    return predict, {"n_params": n_params}


def _fit_ceq_sum_soft(spec: DatasetSpec):
    return _fit_ceq_sum(
        spec,
        nested=False,
        learn_k=False,
        k_fixed=1.0,
        n_restarts=12,
        seed=0,
        maxiter=700,
        reg=1e-4,
    )


def _fit_ceq_sum_hinge(spec: DatasetSpec):
    return _fit_ceq_sum(
        spec,
        nested=False,
        learn_k=False,
        k_fixed=50.0,
        n_restarts=12,
        seed=0,
        maxiter=700,
        reg=1e-4,
    )


def _fit_ceq_sum_learnk(spec: DatasetSpec):
    return _fit_ceq_sum(
        spec,
        nested=False,
        learn_k=True,
        k_fixed=1.0,
        n_restarts=12,
        seed=0,
        maxiter=700,
        reg=1e-4,
    )


def _fit_nceq_fixedk(spec: DatasetSpec):
    return _fit_ceq_sum(
        spec,
        nested=True,
        learn_k=False,
        k_fixed=1.0,
        n_restarts=12,
        seed=0,
        maxiter=700,
        reg=1e-4,
    )


def _fit_nceq_learnk(spec: DatasetSpec):
    return _fit_ceq_sum(
        spec,
        nested=True,
        learn_k=True,
        k_fixed=1.0,
        n_restarts=12,
        seed=0,
        maxiter=700,
        reg=1e-4,
    )

# ---------------------------------------------------------------------------
# Model: FM-CEQ (Forgetting Marginal CEQ)
# ---------------------------------------------------------------------------
def _fit_fmceq(
    spec: DatasetSpec,
    *,
    n_restarts: int = 12,
    seed: int = 0,
    maxiter: int = 700,
    reg: float = 1e-4,
):
    """FM-CEQ: Forgetting Marginal CEQ.

    Sequential state dynamics with a retention factor delta:
      h_{-1} = 0
      h_prior_k = delta * h_{k-1}   (partial forgetting between phases)
      h_k = h_prior_k + E_k         (apply phase k's data)
      DU_k = CES(log1p(h_k)) - CES(log1p(h_prior_k))   (marginal gain)
      U_total = sum_k DU_k          (unweighted sum — no backloading incentive)

    Phasewise overfit penalty:
      P = sum_k softplus(E_{k,small} - tau)^2

    Parameters: c0, logA, logB, (M-1) domain logits, rho, delta_raw, tau
    = M + 5 total (same as CEQ-SUM soft).
    """
    rng = np.random.default_rng(seed)
    W = spec.weights
    y = spec.y
    _, N, M = W.shape

    C = _broadcast_epoch_mult(spec.epoch_multipliers, N, M)
    E = _compute_epochs(W, C)  # (R, N, M)

    S_list = _parse_small(spec.small_domains, M)
    E_small_total = E[:, :, S_list].sum(axis=(1, 2))

    huber_d = _huber_delta(y)

    # c0, logA, logB, (M-1) logits, rho, delta_raw, tau = M + 5
    n_params = 3 + (M - 1) + 1 + 1 + 1

    def _softmax_logits(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        e = np.exp(z)
        return e / e.sum()

    def unpack(p: np.ndarray):
        idx = 0
        c0 = float(p[idx])
        logA = float(p[idx + 1])
        logB = float(p[idx + 2])
        idx += 3

        logits = np.zeros(M)
        if M > 1:
            logits[: M - 1] = p[idx : idx + (M - 1)]
            idx += M - 1
        a = _softmax_logits(logits)

        rho = float(np.clip(5.0 * np.tanh(p[idx]), -10.0, 0.99))
        idx += 1

        # Retention factor delta in [0, 1] via sigmoid
        delta = float(1.0 / (1.0 + np.exp(-np.clip(p[idx], -10.0, 10.0))))
        idx += 1

        tau = float(np.exp(np.clip(p[idx], -5.0, 8.0)))
        idx += 1

        return c0, logA, logB, a, rho, delta, tau

    def forward(p: np.ndarray, W_in: np.ndarray) -> np.ndarray:
        c0, logA, logB, a, rho, delta, tau = unpack(p)
        A = float(np.exp(np.clip(logA, -10.0, 10.0)))
        B = float(np.exp(np.clip(logB, -10.0, 10.0)))

        R_in = W_in.shape[0]
        E_in = _compute_epochs(W_in, C)  # (R_in, N, M)
        a_row = a[None, :]

        # Sequential state dynamics with forgetting
        h = np.zeros((R_in, M))
        U_total = np.zeros(R_in)
        for k in range(N):
            h_prior = np.zeros_like(h) if k == 0 else delta * h
            h = h_prior + E_in[:, k, :]

            U_state = _ces_mean_stable(np.maximum(np.log1p(h), 1e-12), a_row, rho)
            U_prior = _ces_mean_stable(np.maximum(np.log1p(h_prior), 1e-12), a_row, rho)
            U_total += U_state - U_prior

        # Phasewise overfit penalty
        P = np.zeros(R_in)
        for k in range(N):
            E_k_small = E_in[:, k, S_list].sum(axis=1)
            hk = _softplus_scaled(E_k_small - tau, 1.0)
            P += hk * hk

        return c0 - A * U_total + B * P

    def obj(p: np.ndarray) -> float:
        yhat = forward(p, W)
        loss = float(np.sum(_pseudo_huber(yhat - y, huber_d)))
        loss += float(reg) * float(np.sum(p * p))
        return loss

    medE = float(np.median(E_small_total) + 1e-6)
    best_val, best_p = np.inf, None

    for r in range(n_restarts):
        p0 = np.zeros(n_params)
        idx = 0
        p0[idx] = float(np.median(y))
        p0[idx + 1] = float(rng.normal(0.0, 1.0))   # logA
        p0[idx + 2] = float(rng.normal(-2.0, 1.0))  # logB
        idx += 3

        if M > 1:
            p0[idx : idx + (M - 1)] = rng.normal(0.0, 0.5, M - 1)
            idx += M - 1

        p0[idx] = float(rng.normal(0.0, 0.7))  # rho raw
        idx += 1
        p0[idx] = float(rng.normal(0.0, 1.5))  # delta_raw (sigmoid -> [0,1])
        idx += 1
        p0[idx] = float(np.log(medE) + rng.normal(0.0, 0.3))  # logtau
        idx += 1

        try:
            res = minimize(obj, p0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-10})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_p = float(res.fun), res.x
        except Exception:
            continue

    if best_p is None:
        raise RuntimeError("FM-CEQ optimization failed to converge")

    final_p = best_p.copy()

    def predict(W_new: np.ndarray) -> np.ndarray:
        sp = _as_3d(W_new)
        return forward(final_p, sp)

    return predict, {"n_params": n_params}


# ---------------------------------------------------------------------------
# Model: CEQ-Washpen (CEQ-SUM soft utility + washout-weighted overfit penalty)
# ---------------------------------------------------------------------------
def _fit_ceq_washpen(
    spec: DatasetSpec,
    *,
    n_restarts: int = 12,
    seed: int = 0,
    maxiter: int = 700,
    reg: float = 1e-4,
):
    """CEQ-SUM soft utility + washout-weighted phasewise overfit penalty.

    The overfit penalty for each phase is discounted by how much broad-domain
    training follows it:  omega_k = exp(-lambda * E^big_after_k).
    This makes early-phase overfit less costly when later phases wash it out.

    Parameters: c0, logA, logB, (M-1) domain logits, rho, gamma, tau, lambda
    = M + 6 total (one more than CEQ-SUM soft).
    """
    rng = np.random.default_rng(seed)
    W = spec.weights
    y = spec.y
    R, N, M = W.shape

    C = _broadcast_epoch_mult(spec.epoch_multipliers, N, M)
    E = _compute_epochs(W, C)

    S_list = _parse_small(spec.small_domains, M)
    B_list = [d for d in range(M) if d not in set(S_list)]
    E_small_total = E[:, :, S_list].sum(axis=(1, 2))

    huber_d = _huber_delta(y)

    # c0, logA, logB, (M-1) logits, rho, gamma, tau, loglambda = M + 6
    n_params = 3 + (M - 1) + 1 + 1 + 1 + 1

    def _softmax_logits(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        e = np.exp(z)
        return e / e.sum()

    def unpack(p: np.ndarray):
        idx = 0
        c0 = float(p[idx])
        logA = float(p[idx + 1])
        logB = float(p[idx + 2])
        idx += 3

        logits = np.zeros(M)
        if M > 1:
            logits[: M - 1] = p[idx : idx + (M - 1)]
            idx += M - 1
        a = _softmax_logits(logits)

        rho = float(np.clip(5.0 * np.tanh(p[idx]), -10.0, 0.99))
        idx += 1

        gamma = float(p[idx])
        idx += 1

        tau = float(np.exp(np.clip(p[idx], -5.0, 8.0)))
        idx += 1

        lam = float(np.exp(np.clip(p[idx], -8.0, 8.0)))
        idx += 1

        return c0, logA, logB, a, rho, gamma, tau, lam

    def forward(p: np.ndarray, W_in: np.ndarray) -> np.ndarray:
        c0, logA, logB, a, rho, gamma, tau, lam = unpack(p)
        A = float(np.exp(np.clip(logA, -10.0, 10.0)))
        B = float(np.exp(np.clip(logB, -10.0, 10.0)))

        E_in = _compute_epochs(W_in, C)  # (R_in, N, M)
        sat_in = np.maximum(np.log1p(E_in), 1e-12)

        # CEQ-SUM utility (same as CEQ-SUM soft)
        U_k = np.zeros((W_in.shape[0], N))
        for ph in range(N):
            U_k[:, ph] = _ces_mean_stable(sat_in[:, ph, :], a[None, :], rho)
        pi = _phase_softmax_weights(gamma, N)
        U = U_k @ pi

        # Washout-weighted overfit penalty
        E_small = E_in[:, :, S_list].sum(axis=2)  # (R_in, N)
        if B_list:
            E_big = E_in[:, :, B_list].sum(axis=2)  # (R_in, N)
        else:
            E_big = E_in.sum(axis=2)

        # E^big_after_k = sum_{j>k} E_big_j
        Big_after = np.zeros_like(E_big)
        if N > 1:
            suffix = np.cumsum(E_big[:, ::-1], axis=1)[:, ::-1]
            Big_after[:, :-1] = suffix[:, 1:]
        # Last phase: Big_after = 0, so omega = 1 (full penalty)

        omega = np.exp(-lam * Big_after)  # (R_in, N)
        h = _softplus_scaled(E_small - tau, 1.0)
        P = np.sum(omega * (h * h), axis=1)

        return c0 - A * U + B * P

    def obj(p: np.ndarray) -> float:
        yhat = forward(p, W)
        loss = float(np.sum(_pseudo_huber(yhat - y, huber_d)))
        loss += float(reg) * float(np.sum(p * p))
        return loss

    medE = float(np.median(E_small_total) + 1e-6)
    best_val, best_p = np.inf, None

    for r in range(n_restarts):
        p0 = np.zeros(n_params)
        idx = 0
        p0[idx] = float(np.median(y))
        p0[idx + 1] = float(rng.normal(0.0, 1.0))   # logA
        p0[idx + 2] = float(rng.normal(-2.0, 1.0))   # logB
        idx += 3

        if M > 1:
            p0[idx : idx + (M - 1)] = rng.normal(0.0, 0.5, M - 1)
            idx += M - 1

        p0[idx] = float(rng.normal(0.0, 0.7))  # rho raw
        idx += 1
        p0[idx] = float(rng.normal(0.0, 1.0))  # gamma
        idx += 1
        p0[idx] = float(np.log(medE) + rng.normal(0.0, 0.3))  # logtau
        idx += 1
        p0[idx] = float(rng.normal(-1.0, 1.0))  # loglambda
        idx += 1

        try:
            res = minimize(obj, p0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-10})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_p = float(res.fun), res.x
        except Exception:
            continue

    if best_p is None:
        raise RuntimeError("CEQ-Washpen optimization failed to converge")

    final_p = best_p.copy()

    def predict(W_new: np.ndarray) -> np.ndarray:
        sp = _as_3d(W_new)
        return forward(final_p, sp)

    return predict, {"n_params": n_params}


# ---------------------------------------------------------------------------
# Model: CR-CEQ (Cumulative-Recency CEQ + washpen)
# ---------------------------------------------------------------------------
def _fit_ceq_cumrecency(
    spec: DatasetSpec,
    *,
    pi_type: str = "linear",
    n_restarts: int = 12,
    seed: int = 0,
    maxiter: int = 700,
    reg: float = 1e-4,
):
    """CR-CEQ: Cumulative-Recency CEQ with washout-weighted overfit penalty.

    Instead of computing per-phase CES utilities and linearly combining them,
    CR-CEQ first weights the *epochs* by phase importance (recency), then takes
    CES of the aggregate.  This is a static "effective recency-weighted mixture"
    model: the final model behaves as if trained on F_d = sum_k pi_k * E_{k,d}.

    pi_type controls phase weighting:
      "linear" (1p): pi_k ~ exp(gamma * t_k)           — M+6 params
      "quad"   (2p): pi_k ~ exp(beta1*t + beta2*t^2)   — M+7 params
      "beta"   (2p): pi_k ~ t^(a-1) * (1-t)^(b-1)     — M+7 params
    """
    rng = np.random.default_rng(seed)
    W = spec.weights
    y = spec.y
    R, N, M = W.shape

    C = _broadcast_epoch_mult(spec.epoch_multipliers, N, M)
    E = _compute_epochs(W, C)

    S_list = _parse_small(spec.small_domains, M)
    B_list = [d for d in range(M) if d not in set(S_list)]
    E_small_total = E[:, :, S_list].sum(axis=(1, 2))

    huber_d = _huber_delta(y)

    n_pi = _n_pi_params(pi_type)
    n_params = 3 + (M - 1) + 1 + n_pi + 1 + 1

    def _softmax_logits(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        e = np.exp(z)
        return e / e.sum()

    def unpack(p: np.ndarray):
        idx = 0
        c0 = float(p[idx])
        logA = float(p[idx + 1])
        logB = float(p[idx + 2])
        idx += 3
        logits = np.zeros(M)
        if M > 1:
            logits[: M - 1] = p[idx : idx + (M - 1)]
            idx += M - 1
        a = _softmax_logits(logits)
        rho = float(np.clip(5.0 * np.tanh(p[idx]), -10.0, 0.99))
        idx += 1
        pi, idx = _unpack_pi(p, idx, N, pi_type)
        tau = float(np.exp(np.clip(p[idx], -5.0, 8.0)))
        idx += 1
        lam = float(np.exp(np.clip(p[idx], -8.0, 8.0)))
        idx += 1
        return c0, logA, logB, a, rho, pi, tau, lam

    def forward(p: np.ndarray, W_in: np.ndarray) -> np.ndarray:
        c0, logA, logB, a, rho, pi, tau, lam = unpack(p)
        A = float(np.exp(np.clip(logA, -10.0, 10.0)))
        B = float(np.exp(np.clip(logB, -10.0, 10.0)))

        E_in = _compute_epochs(W_in, C)
        F = np.sum(E_in * pi[None, :, None], axis=1)
        U = _ces_mean_stable(np.maximum(np.log1p(F), 1e-12), a[None, :], rho)

        E_small = E_in[:, :, S_list].sum(axis=2)
        E_big = E_in[:, :, B_list].sum(axis=2) if B_list else E_in.sum(axis=2)
        Big_after = np.zeros_like(E_big)
        if N > 1:
            suffix = np.cumsum(E_big[:, ::-1], axis=1)[:, ::-1]
            Big_after[:, :-1] = suffix[:, 1:]
        omega = np.exp(-lam * Big_after)
        h = _softplus_scaled(E_small - tau, 1.0)
        P = np.sum(omega * (h * h), axis=1)

        return c0 - A * U + B * P

    def obj(p: np.ndarray) -> float:
        yhat = forward(p, W)
        loss = float(np.sum(_pseudo_huber(yhat - y, huber_d)))
        loss += float(reg) * float(np.sum(p * p))
        return loss

    medE = float(np.median(E_small_total) + 1e-6)
    best_val, best_p = np.inf, None

    for r in range(n_restarts):
        p0 = np.zeros(n_params)
        idx = 0
        p0[idx] = float(np.median(y))
        p0[idx + 1] = float(rng.normal(0.0, 1.0))
        p0[idx + 2] = float(rng.normal(-2.0, 1.0))
        idx += 3
        if M > 1:
            p0[idx : idx + (M - 1)] = rng.normal(0.0, 0.5, M - 1)
            idx += M - 1
        p0[idx] = float(rng.normal(0.0, 0.7))
        idx += 1
        idx = _init_pi(p0, idx, rng, pi_type)
        p0[idx] = float(np.log(medE) + rng.normal(0.0, 0.3))
        idx += 1
        p0[idx] = float(rng.normal(-1.0, 1.0))
        idx += 1
        try:
            res = minimize(obj, p0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-10})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_p = float(res.fun), res.x
        except Exception:
            continue

    if best_p is None:
        raise RuntimeError("CR-CEQ optimization failed to converge")
    final_p = best_p.copy()

    def predict(W_new: np.ndarray) -> np.ndarray:
        return forward(final_p, _as_3d(W_new))

    return predict, {"n_params": n_params}


def _fit_ceq_cumrecency_quad(spec: DatasetSpec, **kw):
    return _fit_ceq_cumrecency(spec, pi_type="quad", **kw)


def _fit_ceq_cumrecency_beta(spec: DatasetSpec, **kw):
    return _fit_ceq_cumrecency(spec, pi_type="beta", **kw)


# ---------------------------------------------------------------------------
# Model: IS-CEQ (Interference-State CEQ + washpen)
# ---------------------------------------------------------------------------
def _fit_isceq(
    spec: DatasetSpec,
    *,
    pi_type: str = "linear",
    n_restarts: int = 12,
    seed: int = 0,
    maxiter: int = 700,
    reg: float = 1e-4,
):
    """IS-CEQ: Interference-State CEQ with washout-weighted overfit penalty.

    Maintains a per-domain "remembered learning state" S_d that decays when
    other domains are trained:
        S_{k,d} = S_{k-1,d} * exp(-lambda_learn * (1 - W_{k,d})) + pi_k * log1p(E_{k,d})

    This is a leaky integrator / exponential survival model of memory under
    interference: early domain signal decays when later phases emphasize other
    domains.  The decay uses token-mixture weights (not epoch counts), so it
    models order effects even without epoching.

    pi_type controls phase weighting:
      "linear" (1p): pi_k ~ exp(gamma * t_k)           — M+7 params
      "quad"   (2p): pi_k ~ exp(beta1*t + beta2*t^2)   — M+8 params
      "beta"   (2p): pi_k ~ t^(a-1) * (1-t)^(b-1)     — M+8 params
    """
    rng = np.random.default_rng(seed)
    W = spec.weights
    y = spec.y
    R, N, M = W.shape

    C = _broadcast_epoch_mult(spec.epoch_multipliers, N, M)
    E = _compute_epochs(W, C)

    S_list = _parse_small(spec.small_domains, M)
    B_list = [d for d in range(M) if d not in set(S_list)]
    E_small_total = E[:, :, S_list].sum(axis=(1, 2))

    huber_d = _huber_delta(y)

    n_pi = _n_pi_params(pi_type)
    # c0, logA, logB, (M-1) logits, rho, [pi params], tau, loglam_pen, loglam_learn
    n_params = 3 + (M - 1) + 1 + n_pi + 1 + 1 + 1

    def _softmax_logits(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        e = np.exp(z)
        return e / e.sum()

    def unpack(p: np.ndarray):
        idx = 0
        c0 = float(p[idx])
        logA = float(p[idx + 1])
        logB = float(p[idx + 2])
        idx += 3

        logits = np.zeros(M)
        if M > 1:
            logits[: M - 1] = p[idx : idx + (M - 1)]
            idx += M - 1
        a = _softmax_logits(logits)

        rho = float(np.clip(5.0 * np.tanh(p[idx]), -10.0, 0.99))
        idx += 1

        pi, idx = _unpack_pi(p, idx, N, pi_type)

        tau = float(np.exp(np.clip(p[idx], -5.0, 8.0)))
        idx += 1

        lam_pen = float(np.exp(np.clip(p[idx], -8.0, 8.0)))
        idx += 1

        lam_learn = float(np.exp(np.clip(p[idx], -8.0, 8.0)))
        idx += 1

        return c0, logA, logB, a, rho, pi, tau, lam_pen, lam_learn

    def forward(p: np.ndarray, W_in: np.ndarray) -> np.ndarray:
        c0, logA, logB, a, rho, pi, tau, lam_pen, lam_learn = unpack(p)
        A = float(np.exp(np.clip(logA, -10.0, 10.0)))
        B = float(np.exp(np.clip(logB, -10.0, 10.0)))

        R_in = W_in.shape[0]
        E_in = _compute_epochs(W_in, C)  # (R_in, N, M)
        sat = np.log1p(E_in)  # (R_in, N, M)

        # Sequential memory state update
        S = np.zeros((R_in, M))
        for k in range(N):
            decay = np.exp(-lam_learn * (1.0 - W_in[:, k, :]))  # (R_in, M)
            S = S * decay + pi[k] * sat[:, k, :]  # (R_in, M)

        U = _ces_mean_stable(np.maximum(S, 1e-12), a[None, :], rho)  # (R_in,)

        # Washout-weighted overfit penalty
        E_small = E_in[:, :, S_list].sum(axis=2)  # (R_in, N)
        E_big = E_in[:, :, B_list].sum(axis=2) if B_list else E_in.sum(axis=2)

        Big_after = np.zeros_like(E_big)
        if N > 1:
            suffix = np.cumsum(E_big[:, ::-1], axis=1)[:, ::-1]
            Big_after[:, :-1] = suffix[:, 1:]

        omega = np.exp(-lam_pen * Big_after)
        h = _softplus_scaled(E_small - tau, 1.0)
        P = np.sum(omega * (h * h), axis=1)

        return c0 - A * U + B * P

    def obj(p: np.ndarray) -> float:
        yhat = forward(p, W)
        loss = float(np.sum(_pseudo_huber(yhat - y, huber_d)))
        loss += float(reg) * float(np.sum(p * p))
        return loss

    medE = float(np.median(E_small_total) + 1e-6)
    best_val, best_p = np.inf, None

    for r in range(n_restarts):
        p0 = np.zeros(n_params)
        idx = 0
        p0[idx] = float(np.median(y))
        p0[idx + 1] = float(rng.normal(0.0, 1.0))   # logA
        p0[idx + 2] = float(rng.normal(-2.0, 1.0))   # logB
        idx += 3

        if M > 1:
            p0[idx : idx + (M - 1)] = rng.normal(0.0, 0.5, M - 1)
            idx += M - 1

        p0[idx] = float(rng.normal(0.0, 0.7))  # rho raw
        idx += 1
        idx = _init_pi(p0, idx, rng, pi_type)
        p0[idx] = float(np.log(medE) + rng.normal(0.0, 0.3))  # logtau
        idx += 1
        p0[idx] = float(rng.normal(-1.0, 1.0))  # loglam_pen
        idx += 1
        p0[idx] = float(rng.normal(-1.0, 1.0))  # loglam_learn
        idx += 1

        try:
            res = minimize(obj, p0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-10})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_p = float(res.fun), res.x
        except Exception:
            continue

    if best_p is None:
        raise RuntimeError("IS-CEQ optimization failed to converge")

    final_p = best_p.copy()

    def predict(W_new: np.ndarray) -> np.ndarray:
        return forward(final_p, _as_3d(W_new))

    return predict, {"n_params": n_params}


def _fit_isceq_quad(spec: DatasetSpec, **kw):
    return _fit_isceq(spec, pi_type="quad", **kw)


def _fit_isceq_beta(spec: DatasetSpec, **kw):
    return _fit_isceq(spec, pi_type="beta", **kw)


# ---------------------------------------------------------------------------
# Model: IS-CEQ-Toxic (source-toxicity asymmetric interference + washpen)
# ---------------------------------------------------------------------------
def _fit_isceq_toxic(
    spec: DatasetSpec,
    *,
    pi_type: str = "beta",
    n_restarts: int = 16,
    seed: int = 0,
    maxiter: int = 700,
    reg: float = 1e-4,
):
    """IS-CEQ-Toxic: source-toxicity asymmetric interference + washpen.

    Like IS-CEQ but the interference decay is asymmetric: each domain has a
    learned "toxicity" t_d >= 0 (sum to 1).  Training on toxic domains erases
    other memories more.  The decay exponent for memory of domain d in phase k
    is:
        lambda_learn * sum_{j != d} t_j * W_{k,j}

    This encodes "general CC later in training washes out specialized code
    skill" (high toxicity on CC), while "code training doesn't overwrite
    general ability as much" (low toxicity on StarCoder).

    Parameters: c0, logA, logB, (M-1) domain logits, rho, [pi params], tau,
                lam_pen, lam_learn, (M-1) toxicity logits
    = 2M + 5 + n_pi params.  For M=2: 11 (beta/quad) or 10 (linear).
    """
    rng = np.random.default_rng(seed)
    W = spec.weights
    y = spec.y
    R, N, M = W.shape

    C = _broadcast_epoch_mult(spec.epoch_multipliers, N, M)
    E = _compute_epochs(W, C)

    S_list = _parse_small(spec.small_domains, M)
    B_list = [d for d in range(M) if d not in set(S_list)]
    E_small_total = E[:, :, S_list].sum(axis=(1, 2))

    huber_d = _huber_delta(y)

    n_pi = _n_pi_params(pi_type)
    # c0, logA, logB, (M-1) logits, rho, [pi], tau, lam_pen, lam_learn, (M-1) tox logits
    n_params = 3 + (M - 1) + 1 + n_pi + 1 + 1 + 1 + (M - 1)

    def _softmax_logits(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        e = np.exp(z)
        return e / e.sum()

    def unpack(p: np.ndarray):
        idx = 0
        c0 = float(p[idx])
        logA = float(p[idx + 1])
        logB = float(p[idx + 2])
        idx += 3

        logits = np.zeros(M)
        if M > 1:
            logits[: M - 1] = p[idx : idx + (M - 1)]
            idx += M - 1
        a = _softmax_logits(logits)

        rho = float(np.clip(5.0 * np.tanh(p[idx]), -10.0, 0.99))
        idx += 1

        pi, idx = _unpack_pi(p, idx, N, pi_type)

        tau = float(np.exp(np.clip(p[idx], -5.0, 8.0)))
        idx += 1
        lam_pen = float(np.exp(np.clip(p[idx], -8.0, 8.0)))
        idx += 1
        lam_learn = float(np.exp(np.clip(p[idx], -8.0, 8.0)))
        idx += 1

        tox_logits = np.zeros(M)
        if M > 1:
            tox_logits[: M - 1] = p[idx : idx + (M - 1)]
            idx += M - 1
        tox = _softmax_logits(tox_logits)

        return c0, logA, logB, a, rho, pi, tau, lam_pen, lam_learn, tox

    def forward(p: np.ndarray, W_in: np.ndarray) -> np.ndarray:
        c0, logA, logB, a, rho, pi, tau, lam_pen, lam_learn, tox = unpack(p)
        A = float(np.exp(np.clip(logA, -10.0, 10.0)))
        B = float(np.exp(np.clip(logB, -10.0, 10.0)))

        R_in = W_in.shape[0]
        E_in = _compute_epochs(W_in, C)
        sat = np.log1p(E_in)

        # Sequential state with source-toxicity asymmetric interference
        S = np.zeros((R_in, M))
        for k in range(N):
            src = W_in[:, k, :] * tox[None, :]            # (R_in, M)
            src_tot = np.sum(src, axis=1, keepdims=True)   # (R_in, 1)
            # Decay for domain d: exp(-lam * sum_{j!=d} tox_j * W_{k,j})
            decay = np.exp(-lam_learn * (src_tot - src))   # (R_in, M)
            S = S * decay + pi[k] * sat[:, k, :]

        U = _ces_mean_stable(np.maximum(S, 1e-12), a[None, :], rho)

        # Washout-weighted overfit penalty
        E_small = E_in[:, :, S_list].sum(axis=2)
        E_big = E_in[:, :, B_list].sum(axis=2) if B_list else E_in.sum(axis=2)
        Big_after = np.zeros_like(E_big)
        if N > 1:
            suffix = np.cumsum(E_big[:, ::-1], axis=1)[:, ::-1]
            Big_after[:, :-1] = suffix[:, 1:]
        omega = np.exp(-lam_pen * Big_after)
        h = _softplus_scaled(E_small - tau, 1.0)
        P = np.sum(omega * (h * h), axis=1)

        return c0 - A * U + B * P

    def obj(p: np.ndarray) -> float:
        yhat = forward(p, W)
        loss = float(np.sum(_pseudo_huber(yhat - y, huber_d)))
        loss += float(reg) * float(np.sum(p * p))
        return loss

    medE = float(np.median(E_small_total) + 1e-6)
    best_val, best_p = np.inf, None

    for r in range(n_restarts):
        p0 = np.zeros(n_params)
        idx = 0
        p0[idx] = float(np.median(y))
        p0[idx + 1] = float(rng.normal(0.0, 1.0))   # logA
        p0[idx + 2] = float(rng.normal(-2.0, 1.0))   # logB
        idx += 3
        if M > 1:
            p0[idx : idx + (M - 1)] = rng.normal(0.0, 0.5, M - 1)
            idx += M - 1
        p0[idx] = float(rng.normal(0.0, 0.7))  # rho raw
        idx += 1
        idx = _init_pi(p0, idx, rng, pi_type)
        p0[idx] = float(np.log(medE) + rng.normal(0.0, 0.3))  # logtau
        idx += 1
        p0[idx] = float(rng.normal(-1.0, 1.0))  # loglam_pen
        idx += 1
        p0[idx] = float(rng.normal(-1.0, 1.0))  # loglam_learn
        idx += 1
        if M > 1:
            p0[idx : idx + (M - 1)] = rng.normal(0.0, 0.5, M - 1)
            idx += M - 1

        try:
            res = minimize(obj, p0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-10})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_p = float(res.fun), res.x
        except Exception:
            continue

    if best_p is None:
        raise RuntimeError("IS-CEQ-Toxic optimization failed to converge")

    final_p = best_p.copy()

    def predict(W_new: np.ndarray) -> np.ndarray:
        return forward(final_p, _as_3d(W_new))

    return predict, {"n_params": n_params}


# ---------------------------------------------------------------------------
# Model: PCEQ (Phase CES utility + entropy + epoch-overfit quadratic)
# ---------------------------------------------------------------------------
def _ces_utility_features_general(
    W: np.ndarray,
    epoch_multipliers: np.ndarray,
    rho: np.ndarray,
    a_small: np.ndarray,
    small_domains: list[int],
    N: int,
    M: int,
) -> np.ndarray:
    """Per-phase CES utility on log1p(epochs).  Returns (R, N).

    CES weights: a_small[k, i] for small domain i in phase k,
    remainder split evenly among non-small domains.
    """
    W3d = _as_3d(W)
    R = W3d.shape[0]
    C = _broadcast_epoch_mult(epoch_multipliers, N, M)
    E = _compute_epochs(W3d, C)
    nonsmall = [d for d in range(M) if d not in small_domains]
    n_nonsmall = len(nonsmall)

    U = np.zeros((R, N))
    for k in range(N):
        rho_k = np.clip(rho[k], -10, 0.99)
        a_full = np.zeros(M)
        a_small_sum = 0.0
        for i, d in enumerate(small_domains):
            a_full[d] = a_small[k, i]
            a_small_sum += a_small[k, i]
        if n_nonsmall > 0:
            remaining = max(1.0 - a_small_sum, 0.0)
            for d in nonsmall:
                a_full[d] = remaining / n_nonsmall

        sat = np.maximum(np.log1p(E[:, k, :]), 1e-10)
        inner = np.sum(a_full[None, :] * np.power(sat, rho_k), axis=1)
        U[:, k] = np.power(np.maximum(inner, 1e-10), 1.0 / rho_k)

    return U


def _pceq_design_general(
    W: np.ndarray,
    epoch_multipliers: np.ndarray,
    small_domains: list[int],
    N: int,
    M: int,
    rho: np.ndarray,
    a_small: np.ndarray,
) -> np.ndarray:
    """Build PCEQ design: [1, U_k, U_k*U_l, H_k, epoch_quad]."""
    W3d = _as_3d(W)
    R = W3d.shape[0]

    U = _ces_utility_features_general(W3d, epoch_multipliers, rho, a_small, small_domains, N, M)

    # Epoch entropy per phase
    C = _broadcast_epoch_mult(epoch_multipliers, N, M)
    E = _compute_epochs(W3d, C)
    denom = E.sum(axis=2, keepdims=True)
    q = np.clip(E / np.maximum(denom, 1e-10), 1e-10, 1.0)
    H = np.sum(q * np.log(q), axis=2)

    # Epoch-overfit quadratic
    Z_eq, _ = _epoch_overfit_quadratic_raw(W3d, epoch_multipliers, small_domains, N, M)

    cols: list[np.ndarray] = [np.ones((R, 1))]
    for k in range(N):
        cols.append(U[:, k : k + 1])
    for k1 in range(N):
        for k2 in range(k1 + 1, N):
            cols.append((U[:, k1] * U[:, k2]).reshape(-1, 1))
    cols.append(H)
    if Z_eq.shape[1] > 0:
        cols.append(Z_eq)

    return np.hstack(cols)


def _pceq_nparams(N: int, M: int, small_domains: list[int] | None) -> int:
    """Number of linear coefficients in PCEQ."""
    S = len(_parse_small(small_domains, M))
    P = len(_phase_pairs(N, "adjacent"))
    return 1 + N + N * (N - 1) // 2 + N + N * S + S * P


def _fit_pceq(spec: DatasetSpec, n_folds: int = 5, seed: int = 42):
    """PCEQ: Phase CES utility + entropy + epoch-overfit quadratic.

    Grid search over CES hyperparams (rho per phase, a per small domain per phase),
    then OLS.  Best hyperparams by k-fold CV.
    """
    from itertools import product as cart_product

    from sklearn.model_selection import KFold

    N, M = spec.N, spec.M
    S_list = _parse_small(spec.small_domains, M)
    S = len(S_list)
    n_linear = _pceq_nparams(N, M, spec.small_domains)

    rho_values = np.linspace(-5.0, 0.9, 12)
    a_values = np.linspace(0.1, 0.9, 8)

    kf = KFold(n_folds, shuffle=True, random_state=seed)
    folds = list(kf.split(spec.weights))

    rho_combos = list(cart_product(rho_values, repeat=N))
    a_combos = list(cart_product(a_values, repeat=N * S)) if S > 0 else [()]

    best_cv_sse = np.inf
    best_rho = np.zeros(N)
    best_a = np.zeros((N, max(S, 1)))

    for rho_tuple in rho_combos:
        rho = np.array(rho_tuple)
        for a_tuple in a_combos:
            a_small = np.array(a_tuple).reshape(N, S) if S > 0 else np.empty((N, 0))

            cv_sse = 0.0
            valid = True
            for tr, te in folds:
                try:
                    Phi_tr = _pceq_design_general(spec.weights[tr], spec.epoch_multipliers, S_list, N, M, rho, a_small)
                    beta = _fit_ols(Phi_tr, spec.y[tr])
                    Phi_te = _pceq_design_general(spec.weights[te], spec.epoch_multipliers, S_list, N, M, rho, a_small)
                    cv_sse += float(np.sum((Phi_te @ beta - spec.y[te]) ** 2))
                except Exception:
                    valid = False
                    break

            if valid and cv_sse < best_cv_sse:
                best_cv_sse = cv_sse
                best_rho = rho.copy()
                best_a = a_small.copy() if S > 0 else np.empty((N, 0))

    # Refit on all data
    Phi = _pceq_design_general(spec.weights, spec.epoch_multipliers, S_list, N, M, best_rho, best_a)
    coef = _fit_ols(Phi, spec.y)

    emult = spec.epoch_multipliers
    sd_list = S_list
    final_rho = best_rho
    final_a = best_a

    def predict(W_new):
        return _pceq_design_general(W_new, emult, sd_list, N, M, final_rho, final_a) @ coef

    return predict, {
        "n_params": n_linear,
        "rho": final_rho.tolist(),
        "a_small": final_a.tolist(),
    }


def _applicable_pceq(spec: DatasetSpec) -> bool:
    n_linear = _pceq_nparams(spec.N, spec.M, spec.small_domains)
    return spec.R > n_linear


# ---------------------------------------------------------------------------
# Model list
# ---------------------------------------------------------------------------
GENERAL_MODELS: list[GeneralModelSpec] = [
    GeneralModelSpec("Linear", _fit_linear, _applicable_linear, "OLS linear on weights"),
    GeneralModelSpec("LogLinear", _fit_loglinear, _applicable_linear, "exp(OLS linear) on weights"),
    GeneralModelSpec("LogLinear(w-e)", _fit_loglinear_we, _applicable_loglinear_we, "exp(OLS linear) on weight+epoch"),
    GeneralModelSpec("Quadratic", _fit_quadratic_weight, _applicable_quadratic_weight, "OLS quadratic on weights"),
    GeneralModelSpec("Quadratic(w-e)", _fit_quadratic_we, _applicable_quadratic_we, "OLS quadratic on weight+epoch"),
    GeneralModelSpec("RidgeQuad(w-e)", _fit_ridge_quad_we, lambda _: True, "Ridge quadratic on weight+epoch"),
    GeneralModelSpec("LogQuad(w-e)", _fit_logquad_we, _applicable_quadratic_we, "exp(OLS quad) on weight+epoch"),
    GeneralModelSpec("BayesLogQuad(w-e)", _fit_bayes_logquad_we, lambda _: True, "Bayesian Ridge logquad"),
    GeneralModelSpec("ElasticLogQuad(w-e)", _fit_elastic_logquad_we, lambda _: True, "ElasticNet logquad"),
    GeneralModelSpec("HuberLogQuad(w-e)", _fit_huber_logquad_we, lambda _: True, "Huber logquad"),
    GeneralModelSpec(
        "BayesLogQuad(wt)", _fit_bayes_logquad_weight, _always_applicable, "Bayesian Ridge logquad on weights only"
    ),
    GeneralModelSpec(
        "ElasticLogQuad(wt)", _fit_elastic_logquad_weight, _always_applicable, "ElasticNet logquad on weights only"
    ),
    GeneralModelSpec("SOE-Base", _fit_soe_base, _applicable_soe_base, "Satiety+overfit, OLS"),
    GeneralModelSpec("SOE-Plus", _fit_soe_plus, _applicable_soe_plus, "SOE-Base + within-phase coupling"),
    GeneralModelSpec("SOE-Plus(ridge)", _fit_soe_plus_ridge, _applicable_soe_plus, "SOE-Plus with ridge=1"),
    GeneralModelSpec("SOE-Curric", _fit_soe_curric, _applicable_soe_curric, "Full curriculum interactions"),
    GeneralModelSpec("SOE-Curric(ridge)", _fit_soe_curric_ridge, _applicable_soe_curric, "SOE-Curric with ridge=1"),
    GeneralModelSpec(
        "Threshold Overfit", _fit_threshold_overfit, _applicable_threshold, "Hinge penalty at learned threshold"
    ),
    GeneralModelSpec("Scheffé+log", _fit_scheffe_log, lambda _: True, "Scheffé polynomial + log terms"),
    GeneralModelSpec(
        "Scheffé+EpEnt", _fit_scheffe_log_epoch_entropy, lambda _: True, "Scheffé+log + epoch-entropy per phase"
    ),
    GeneralModelSpec("Scheffé+TiedEnt", _fit_scheffe_tied_entropy, lambda _: True, "Scheffé with phase-tied entropy"),
    GeneralModelSpec("CES", _fit_ces, _applicable_ces, "Constant Elasticity of Substitution"),
    GeneralModelSpec(
        "CES-Overfit", _fit_ces_overfit, _applicable_ces_overfit, "CES utility + softplus overtraining penalty"
    ),
    GeneralModelSpec("CEQ-SUM soft", _fit_ceq_sum_soft, lambda _: True, "CEQ-SUM (k=1) satiety+overfit"),
    GeneralModelSpec("CEQ-SUM hinge", _fit_ceq_sum_hinge, lambda _: True, "CEQ-SUM (k=50) satiety+overfit"),
    GeneralModelSpec("CEQ-SUM(k)", _fit_ceq_sum_learnk, lambda _: True, "CEQ-SUM with learnable sharpness k"),
    GeneralModelSpec("NCEQ", _fit_nceq_fixedk, lambda _: True, "Nested CEQ across phases (k=1 fixed)"),
    GeneralModelSpec("NCEQ(k)", _fit_nceq_learnk, lambda _: True, "Nested CEQ across phases + learnable k"),
    GeneralModelSpec("FM-CEQ", _fit_fmceq, lambda _: True, "Forgetting Marginal CEQ (sequential + retention)"),
    GeneralModelSpec("CEQ-Washpen", _fit_ceq_washpen, lambda _: True, "CEQ-SUM soft + washout-weighted overfit penalty"),
    GeneralModelSpec("CR-CEQ", _fit_ceq_cumrecency, lambda _: True, "Cumulative-recency CEQ + washpen"),
    GeneralModelSpec("CR-CEQ(quad)", _fit_ceq_cumrecency_quad, lambda _: True, "CR-CEQ + quadratic phase weights"),
    GeneralModelSpec("CR-CEQ(beta)", _fit_ceq_cumrecency_beta, lambda _: True, "CR-CEQ + beta phase weights"),
    GeneralModelSpec("IS-CEQ", _fit_isceq, lambda _: True, "Interference-state CEQ + washpen"),
    GeneralModelSpec("IS-CEQ(quad)", _fit_isceq_quad, lambda _: True, "IS-CEQ + quadratic phase weights"),
    GeneralModelSpec("IS-CEQ(beta)", _fit_isceq_beta, lambda _: True, "IS-CEQ + beta-distribution phase weights"),
    GeneralModelSpec("IS-CEQ-Toxic", _fit_isceq_toxic, lambda _: True, "IS-CEQ + source-toxicity asymmetric forgetting"),
    # Hybrid models: mixture design + economics utility + information theory
    GeneralModelSpec("SHEQ", _fit_sheq, lambda _: True, "Scheffé+log + epoch-entropy + epoch-overfit quadratic"),
    GeneralModelSpec("PCEQ", _fit_pceq, _applicable_pceq, "Phase CES utility + entropy + epoch-overfit quadratic"),
]