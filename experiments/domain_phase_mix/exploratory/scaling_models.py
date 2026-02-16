# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "scikit-learn"]
# ///
"""Pure library of parametric scaling-law models for two-phase training experiments.

This module contains:
- 20 parametric model fit functions (each returns (pred_fn, params_array))
- 20 LaTeX label generators (for visualizing fitted forms on the p0=0 slice)
- ModelSpec dataclass for clean registry
- Feature construction and slice builders
- Cross-validation metrics function

No data loading, no matplotlib, no module-level side effects.
"""

from dataclasses import dataclass
from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

# =========================================================================
# Constants
# =========================================================================
EPS = 1e-8
SC_EPOCH_MULT = 13.2289  # StarCoder epochs per weight fraction
NEM_EPOCH_MULT = 0.5  # Nemotron epochs per weight fraction

FeatureKind = Literal["weight", "vdom", "wt_epoch"]


# =========================================================================
# ModelSpec dataclass
# =========================================================================
@dataclass(frozen=True)
class ModelSpec:
    """Specification for a parametric scaling-law model.

    Attributes:
        name: Display name (e.g., "Linear", "Quadratic(w-e)")
        fit_fn: Function (X, y) -> (pred_fn, params_array)
        feature_kind: One of "weight", "vdom", "wt_epoch"
        label_fn: Function params_array -> LaTeX string (for p0=0 slice)
        color: Matplotlib color for plots
        linestyle: Matplotlib linestyle ("-", "--", "-.", ":")
    """

    name: str
    fit_fn: Callable
    feature_kind: FeatureKind
    label_fn: Callable
    color: str
    linestyle: str


# =========================================================================
# Feature construction
# =========================================================================
def build_features(kind: FeatureKind, p0_sc: np.ndarray, p1_sc: np.ndarray) -> np.ndarray:
    """Build feature matrix from raw StarCoder weight fractions.

    Args:
        kind: Feature type to construct
        p0_sc: Phase 0 StarCoder weight fractions (N,)
        p1_sc: Phase 1 StarCoder weight fractions (N,)

    Returns:
        Feature matrix (N, d) where d depends on kind:
        - "weight": (N, 2) = [p0_sc, p1_sc]
        - "vdom": (N, 4) = [nem0_vol, sc0_vol, nem1_vol, sc1_vol]
        - "wt_epoch": (N, 4) = [p0_sc, p1_sc, log(ep0), log(ep1)]
    """
    if kind == "weight":
        return np.column_stack([p0_sc, p1_sc])
    elif kind == "vdom":
        # Virtual domain: volume fractions (each phase contributes 0.5 total)
        return np.column_stack([
            0.5 * (1 - p0_sc),  # Nemotron phase 0
            0.5 * p0_sc,  # StarCoder phase 0
            0.5 * (1 - p1_sc),  # Nemotron phase 1
            0.5 * p1_sc,  # StarCoder phase 1
        ])
    elif kind == "wt_epoch":
        # Weight-epoch features: proportions + log-transformed epoch counts
        # Epochs scale linearly with weight: epoch_j = SC_EPOCH_MULT * p_j
        return np.column_stack([
            p0_sc,
            p1_sc,
            np.log(SC_EPOCH_MULT * p0_sc + EPS),
            np.log(SC_EPOCH_MULT * p1_sc + EPS),
        ])
    else:
        raise ValueError(f"Unknown feature kind: {kind}")


# =========================================================================
# Cross-validation
# =========================================================================
def cv_metrics(fit_fn, X, y, n_folds=5, seed=42):
    """5-fold cross-validation metrics for a model.

    Args:
        fit_fn: Model fit function (X, y) -> (pred_fn, params)
        X: Feature matrix (N, d)
        y: Target values (N,)
        n_folds: Number of CV folds
        seed: Random seed for reproducibility

    Returns:
        dict with keys: "R2", "RMSE", "Spearman", "RMSE_bot"
    """
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    r2s, rmses, spearmans, rmse_bots = [], [], [], []
    median_y = np.median(y)
    for tr, te in kf.split(X):
        pred_fn, _ = fit_fn(X[tr], y[tr])
        pred = pred_fn(X[te])
        ss_res = np.sum((y[te] - pred) ** 2)
        ss_tot = np.sum((y[te] - y[te].mean()) ** 2)
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        rmses.append(np.sqrt(np.mean((y[te] - pred) ** 2)))
        if len(y[te]) > 2:
            sp, _ = spearmanr(y[te], pred)
            spearmans.append(sp if not np.isnan(sp) else 0.0)
        bot = y[te] < median_y
        if bot.sum() > 1:
            rmse_bots.append(np.sqrt(np.mean((y[te][bot] - pred[bot]) ** 2)))
    return {
        "R2": np.mean(r2s),
        "RMSE": np.mean(rmses),
        "Spearman": np.mean(spearmans) if spearmans else 0.0,
        "RMSE_bot": np.mean(rmse_bots) if rmse_bots else np.nan,
    }


# =========================================================================
# Model definitions — each returns (pred_fn, params_array)
# =========================================================================
def _softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def fit_linear(X, y):
    """y = c0 + c1*x0 + c2*x1.  Features: [p0_sc, p1_sc]."""
    X_aug = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    return lambda Xn: np.column_stack([np.ones(len(Xn)), Xn]) @ coef, coef


def fit_quadratic(X, y):
    """y = c0 + c1*x0 + c2*x1 + c3*x0^2 + c4*x1^2 + c5*x0*x1.  Features: [p0_sc, p1_sc]."""

    def _build(X):
        x0, x1 = X[:, 0], X[:, 1]
        return np.column_stack([np.ones(len(X)), x0, x1, x0**2, x1**2, x0 * x1])

    coef, _, _, _ = np.linalg.lstsq(_build(X), y, rcond=None)
    return lambda Xn: _build(Xn) @ coef, coef


def fit_quadratic_4d(X, y):
    """Full quadratic on [p0_sc, p1_sc, log_sc_ep0, log_sc_ep1].  15 params."""

    def _build(X):
        n = len(X)
        parts = [np.ones((n, 1))]
        nf = X.shape[1]
        for i in range(nf):
            parts.append(X[:, i : i + 1])
        for i in range(nf):
            parts.append(X[:, i : i + 1] ** 2)
        for i in range(nf):
            for j in range(i + 1, nf):
                parts.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.hstack(parts)

    coef, _, _, _ = np.linalg.lstsq(_build(X), y, rcond=None)
    return lambda Xn: _build(Xn) @ coef, coef


def fit_loglinear(X, y):
    """log(y) = c0 + c1*x0 + ... + cn*xn  =>  y = exp(linear).
    Features: whatever is passed.  Params: [c0, c1, ..., cn].
    """
    log_y = np.log(y)
    X_aug = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(X_aug, log_y, rcond=None)
    return lambda Xn: np.exp(np.column_stack([np.ones(len(Xn)), Xn]) @ coef), coef


def fit_powerlaw(X, y, n_restarts=50, seed=42):
    """y = sum_i softplus(a_i + b_i @ x)^(-g_i) + c.  Features: [p0_sc, p1_sc], 2 terms."""
    rng = np.random.default_rng(seed)
    nf, nt = X.shape[1], 2

    def model(X, p):
        r = np.full(len(X), p[-1])
        for i in range(nt):
            b = i * (nf + 2)
            raw = p[b] + X @ p[b + 1 : b + 1 + nf]
            r += (_softplus(raw) + 0.1) ** (-p[b + 1 + nf])
        return r

    def loss(p):
        return float(np.sum((model(X, p) - y) ** 2))

    best_l, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = []
        for _ in range(nt):
            p0.extend([rng.uniform(0.5, 3), *rng.uniform(-2, 2, nf), rng.uniform(0.1, 1.5)])
        p0.append(y.mean())
        bnd = []
        for _ in range(nt):
            bnd += [(None, None)] * (nf + 1) + [(0.01, 3.0)]
        bnd.append((None, None))
        try:
            res = minimize(loss, np.array(p0), method="L-BFGS-B", bounds=bnd, options={"maxiter": 1000, "ftol": 1e-10})
            if res.fun < best_l:
                best_l, best_p = res.fun, res.x
        except Exception:
            continue
    return lambda Xn: model(Xn, best_p), best_p


def fit_dml_m1(X, y, n_restarts=40, seed=42):
    """DML M1 (Sum-Exp): y = c + sum_j k_j * exp(t_j * r_j).  Features: vdom."""
    rng = np.random.default_rng(seed)
    nf = X.shape[1]

    def model(X, p):
        r = np.full(len(X), p[0])
        for j in range(nf):
            r += p[1 + 2 * j] * np.exp(np.clip(p[2 + 2 * j] * X[:, j], -20, 20))
        return r

    def loss(p):
        return float(np.sum((model(X, p) - y) ** 2))

    best_l, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = [y.mean()]
        for _ in range(nf):
            p0.extend([rng.uniform(-1, 1), rng.uniform(-8, 8)])
        p0 = np.array(p0)
        try:
            res = minimize(loss, p0, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-10})
            if res.fun < best_l:
                best_l, best_p = res.fun, res.x
        except Exception:
            continue
    return lambda Xn: model(Xn, best_p), best_p


def fit_slodm(X, y, n_restarts=40, seed=42):
    """SLODM: y = E + 1/sum(C_j * r_j^g_j).  Features: vdom."""
    rng = np.random.default_rng(seed)
    nf = X.shape[1]

    def model(X, p):
        denom = np.full(len(X), 1e-10)
        for j in range(nf):
            Cj = np.exp(np.clip(p[1 + 2 * j], -10, 10))
            gj = np.exp(np.clip(p[2 + 2 * j], -3, 3))
            denom += Cj * np.power(np.maximum(X[:, j], 1e-8), gj)
        return p[0] + 1.0 / denom

    def loss(p):
        return float(np.sum((model(X, p) - y) ** 2))

    best_l, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = [y.min() + rng.normal(0, 0.05)]
        for _ in range(nf):
            p0.extend([rng.normal(0, 1.5), rng.normal(0, 0.5)])
        p0 = np.array(p0)
        try:
            res = minimize(loss, p0, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-10})
            if res.fun < best_l:
                best_l, best_p = res.fun, res.x
        except Exception:
            continue
    return lambda Xn: model(Xn, best_p), best_p


def fit_bimix(X, y, n_restarts=40, seed=42):
    """BiMix: y = sum_j A_j/(r_j+eps)^a_j + C.  Features: vdom."""
    rng = np.random.default_rng(seed)
    nf = X.shape[1]
    EPS_B = 1e-3

    def model(X, p):
        r = np.full(len(X), p[0])
        for j in range(nf):
            Aj = np.exp(np.clip(p[1 + 2 * j], -10, 10))
            aj = np.exp(np.clip(p[2 + 2 * j], -3, 3))
            r += Aj / np.power(X[:, j] + EPS_B, aj)
        return r

    def loss(p):
        return float(np.sum((model(X, p) - y) ** 2))

    best_l, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = [y.min() + rng.normal(0, 0.05)]
        for _ in range(nf):
            p0.extend([rng.normal(-4, 1.5), rng.normal(-1, 0.5)])
        p0 = np.array(p0)
        try:
            res = minimize(loss, p0, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-10})
            if res.fun < best_l:
                best_l, best_p = res.fun, res.x
        except Exception:
            continue
    return lambda Xn: model(Xn, best_p), best_p


def fit_cobb_douglas(X, y, n_restarts=40, seed=42):
    """Cobb-Douglas: L = C - A * prod(r_j^alpha_j).  Features: vdom."""
    rng = np.random.default_rng(seed)
    nf = X.shape[1]

    def model(X, p):
        C, log_A = p[0], p[1]
        alphas = np.abs(p[2 : 2 + nf]) + 1e-6
        log_prod = np.sum(alphas * np.log(np.maximum(X, 1e-10)), axis=1)
        return C - np.exp(log_A + log_prod)

    def loss(p):
        return float(np.sum((model(X, p) - y) ** 2))

    best_l, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = [y.max() + rng.normal(0, 0.1), rng.normal(-1, 1)]
        p0.extend(rng.uniform(0.1, 2.0, nf))
        try:
            res = minimize(loss, np.array(p0), method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-10})
            if res.fun < best_l:
                best_l, best_p = res.fun, res.x
        except Exception:
            continue
    return lambda Xn: model(Xn, best_p), best_p


def fit_translog(X, y, **_kwargs):
    """Translog: L = exp(a0 + sum a_j*ln(r_j) + 0.5*sum b_jk*ln(r_j)*ln(r_k)).  Features: vdom."""
    log_X = np.log(np.maximum(X, 1e-10))
    log_y = np.log(y)
    n, nf = X.shape
    parts = [np.ones((n, 1))]
    for j in range(nf):
        parts.append(log_X[:, j : j + 1])
    for j in range(nf):
        parts.append(log_X[:, j : j + 1] ** 2)
    for j in range(nf):
        for k in range(j + 1, nf):
            parts.append((log_X[:, j] * log_X[:, k]).reshape(-1, 1))
    Z = np.hstack(parts)
    coef, _, _, _ = np.linalg.lstsq(Z, log_y, rcond=None)

    def pred(Xn):
        log_Xn = np.log(np.maximum(Xn, 1e-10))
        nn = len(Xn)
        ps = [np.ones((nn, 1))]
        for j in range(nf):
            ps.append(log_Xn[:, j : j + 1])
        for j in range(nf):
            ps.append(log_Xn[:, j : j + 1] ** 2)
        for j in range(nf):
            for k in range(j + 1, nf):
                ps.append((log_Xn[:, j] * log_Xn[:, k]).reshape(-1, 1))
        return np.exp(np.hstack(ps) @ coef)

    return pred, coef


def fit_stone_geary(X, y, n_restarts=40, seed=42):
    """Stone-Geary: L = C - A * prod((r_j - gamma_j)^beta_j).  Features: vdom."""
    rng = np.random.default_rng(seed)
    nf = X.shape[1]

    def model(X, p):
        C, log_A = p[0], p[1]
        gammas = np.clip(p[2 : 2 + nf], 0, 0.2)
        betas = np.abs(p[2 + nf : 2 + 2 * nf]) + 1e-6
        shifted = np.maximum(X - gammas, 1e-10)
        log_prod = np.sum(betas * np.log(shifted), axis=1)
        return C - np.exp(log_A + log_prod)

    def loss(p):
        return float(np.sum((model(X, p) - y) ** 2))

    best_l, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = [y.max() + rng.normal(0, 0.1), rng.normal(-1, 1)]
        p0.extend(rng.uniform(0, 0.05, nf))
        p0.extend(rng.uniform(0.1, 2.0, nf))
        bnd = [(None, None), (None, None)]
        bnd += [(0, 0.2)] * nf
        bnd += [(1e-4, None)] * nf
        try:
            res = minimize(loss, np.array(p0), method="L-BFGS-B", bounds=bnd, options={"maxiter": 1000, "ftol": 1e-10})
            if res.fun < best_l:
                best_l, best_p = res.fun, res.x
        except Exception:
            continue
    return lambda Xn: model(Xn, best_p), best_p


def fit_logquad_mix(X, y, **_kwargs):
    """LogQuad(w{-}e): y = exp(quadratic in [p0, p1, log_ep0, log_ep1]).  Features: wt_epoch."""
    log_y = np.log(y)
    n, nf = X.shape
    parts = [np.ones((n, 1))]
    for j in range(nf):
        parts.append(X[:, j : j + 1])
    for j in range(nf):
        parts.append(X[:, j : j + 1] ** 2)
    for j in range(nf):
        for k in range(j + 1, nf):
            parts.append((X[:, j] * X[:, k]).reshape(-1, 1))
    Z = np.hstack(parts)
    coef, _, _, _ = np.linalg.lstsq(Z, log_y, rcond=None)

    def pred(Xn):
        nn = len(Xn)
        ps = [np.ones((nn, 1))]
        for j in range(nf):
            ps.append(Xn[:, j : j + 1])
        for j in range(nf):
            ps.append(Xn[:, j : j + 1] ** 2)
        for j in range(nf):
            for k in range(j + 1, nf):
                ps.append((Xn[:, j] * Xn[:, k]).reshape(-1, 1))
        return np.exp(np.hstack(ps) @ coef)

    return pred, coef


def fit_ridge_translog(X, y, **_kwargs):
    """Ridge Translog: y = exp(quadratic in log(vdom)), L2-regularized.  Features: vdom."""
    from sklearn.linear_model import RidgeCV

    log_X = np.log(np.maximum(X, 1e-10))
    log_y = np.log(y)
    n, nf = X.shape
    parts = [np.ones((n, 1))]
    for j in range(nf):
        parts.append(log_X[:, j : j + 1])
    for j in range(nf):
        parts.append(log_X[:, j : j + 1] ** 2)
    for j in range(nf):
        for k in range(j + 1, nf):
            parts.append((log_X[:, j] * log_X[:, k]).reshape(-1, 1))
    Z = np.hstack(parts)
    # Standardize columns (except intercept) so Ridge penalizes fairly
    mu = Z.mean(axis=0)
    sigma = Z.std(axis=0)
    sigma[0] = 1.0  # don't scale intercept column
    mu[0] = 0.0
    Z_std = (Z - mu) / sigma
    ridge = RidgeCV(alphas=np.logspace(-4, 4, 50), fit_intercept=False)
    ridge.fit(Z_std, log_y)
    # Convert back to original scale: coef_orig = coef_std / sigma
    coef_std = ridge.coef_
    coef = coef_std / sigma
    intercept_adj = -np.sum(coef_std * mu / sigma)
    coef[0] += intercept_adj

    def pred(Xn):
        log_Xn = np.log(np.maximum(Xn, 1e-10))
        nn = len(Xn)
        ps = [np.ones((nn, 1))]
        for j in range(nf):
            ps.append(log_Xn[:, j : j + 1])
        for j in range(nf):
            ps.append(log_Xn[:, j : j + 1] ** 2)
        for j in range(nf):
            for k in range(j + 1, nf):
                ps.append((log_Xn[:, j] * log_Xn[:, k]).reshape(-1, 1))
        return np.exp(np.hstack(ps) @ coef)

    return pred, coef


def fit_scheffe_log(X, y, **_kwargs):
    """Scheffé+log: y = sum b_j r_j + sum b_jk r_j r_k + sum g_j r_j ln(r_j).  Features: vdom."""
    n, nf = X.shape
    safe_X = np.maximum(X, 1e-10)
    parts = []
    for j in range(nf):
        parts.append(X[:, j : j + 1])
    for j in range(nf):
        for k in range(j + 1, nf):
            parts.append((X[:, j] * X[:, k]).reshape(-1, 1))
    for j in range(nf):
        parts.append((safe_X[:, j] * np.log(safe_X[:, j])).reshape(-1, 1))
    Z = np.hstack(parts)
    coef, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)

    def pred(Xn):
        safe_Xn = np.maximum(Xn, 1e-10)
        nn = len(Xn)
        ps = []
        for j in range(nf):
            ps.append(Xn[:, j : j + 1])
        for j in range(nf):
            for k in range(j + 1, nf):
                ps.append((Xn[:, j] * Xn[:, k]).reshape(-1, 1))
        for j in range(nf):
            ps.append((safe_Xn[:, j] * np.log(safe_Xn[:, j])).reshape(-1, 1))
        return np.hstack(ps) @ coef

    return pred, coef


def _ilr_transform(X):
    """ILR transform for d=4 compositional data (sequential binary partition)."""
    r = np.maximum(X, 1e-10)
    z1 = np.sqrt(3.0 / 4) * np.log(r[:, 0] / (r[:, 1] * r[:, 2] * r[:, 3]) ** (1.0 / 3))
    z2 = np.sqrt(2.0 / 3) * np.log(r[:, 1] / (r[:, 2] * r[:, 3]) ** 0.5)
    z3 = np.sqrt(1.0 / 2) * np.log(r[:, 2] / r[:, 3])
    return np.column_stack([z1, z2, z3])


def fit_ilr_quad(X, y, **_kwargs):
    """ILR Quadratic: y = quadratic in ILR(vdom) coordinates.  Features: vdom."""
    Z_ilr = _ilr_transform(X)
    n, nf = Z_ilr.shape
    parts = [np.ones((n, 1))]
    for j in range(nf):
        parts.append(Z_ilr[:, j : j + 1])
    for j in range(nf):
        parts.append(Z_ilr[:, j : j + 1] ** 2)
    for j in range(nf):
        for k in range(j + 1, nf):
            parts.append((Z_ilr[:, j] * Z_ilr[:, k]).reshape(-1, 1))
    D = np.hstack(parts)
    coef, _, _, _ = np.linalg.lstsq(D, y, rcond=None)

    def pred(Xn):
        Z_new = _ilr_transform(Xn)
        nn = len(Z_new)
        ps = [np.ones((nn, 1))]
        for j in range(nf):
            ps.append(Z_new[:, j : j + 1])
        for j in range(nf):
            ps.append(Z_new[:, j : j + 1] ** 2)
        for j in range(nf):
            for k in range(j + 1, nf):
                ps.append((Z_new[:, j] * Z_new[:, k]).reshape(-1, 1))
        return np.hstack(ps) @ coef

    return pred, coef


def fit_ces(X, y, n_restarts=40, seed=42):
    """CES: L = C - A * (sum a_j * r_j^rho)^(1/rho).  Features: vdom."""
    rng = np.random.default_rng(seed)
    nf = X.shape[1]

    def model(X, p):
        C, log_A, rho = p[0], p[1], np.clip(p[2], -10, 0.99)
        log_a = p[3 : 3 + nf]
        a = np.exp(log_a)
        a = a / a.sum()
        inner = np.sum(a * np.power(np.maximum(X, 1e-10), rho), axis=1)
        return C - np.exp(log_A) * np.power(np.maximum(inner, 1e-10), 1.0 / rho)

    def loss(p):
        return float(np.sum((model(X, p) - y) ** 2))

    best_l, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = [y.max() + rng.normal(0, 0.1), rng.normal(-1, 1), rng.uniform(-5, 0.5)]
        p0.extend(rng.normal(0, 0.5, nf))
        try:
            res = minimize(loss, np.array(p0), method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-10})
            if res.fun < best_l:
                best_l, best_p = res.fun, res.x
        except Exception:
            continue
    return lambda Xn: model(Xn, best_p), best_p


def fit_elastic_logquad(X, y, **_kwargs):
    """ElasticLogQuad: y = exp(quadratic in mixed_both), Elastic Net.  Features: wt_epoch."""
    from sklearn.linear_model import ElasticNetCV

    log_y = np.log(y)
    n, nf = X.shape
    parts = [np.ones((n, 1))]
    for j in range(nf):
        parts.append(X[:, j : j + 1])
    for j in range(nf):
        parts.append(X[:, j : j + 1] ** 2)
    for j in range(nf):
        for k in range(j + 1, nf):
            parts.append((X[:, j] * X[:, k]).reshape(-1, 1))
    Z = np.hstack(parts)

    mu = Z.mean(axis=0)
    sigma = Z.std(axis=0)
    mu[0], sigma[0] = 0.0, 1.0
    sigma[sigma < 1e-12] = 1.0
    Z_std = (Z - mu) / sigma

    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
        alphas=np.logspace(-5, 1, 50),
        cv=5,
        fit_intercept=False,
        max_iter=10000,
    )
    enet.fit(Z_std, log_y)

    coef = enet.coef_ / sigma
    coef[0] -= np.sum(enet.coef_ * mu / sigma)

    def pred(Xn):
        nn = len(Xn)
        nf2 = Xn.shape[1]
        ps = [np.ones((nn, 1))]
        for j in range(nf2):
            ps.append(Xn[:, j : j + 1])
        for j in range(nf2):
            ps.append(Xn[:, j : j + 1] ** 2)
        for j in range(nf2):
            for k in range(j + 1, nf2):
                ps.append((Xn[:, j] * Xn[:, k]).reshape(-1, 1))
        return np.exp(np.hstack(ps) @ coef)

    return pred, coef


def fit_logquad_weight(X, y, **_kwargs):
    """LogQuad(weight): y = exp(quadratic in [p0_sc, p1_sc]).  Features: weight."""
    log_y = np.log(y)
    n, nf = X.shape
    parts = [np.ones((n, 1))]
    for j in range(nf):
        parts.append(X[:, j : j + 1])
    for j in range(nf):
        parts.append(X[:, j : j + 1] ** 2)
    for j in range(nf):
        for k in range(j + 1, nf):
            parts.append((X[:, j] * X[:, k]).reshape(-1, 1))
    Z = np.hstack(parts)
    coef, _, _, _ = np.linalg.lstsq(Z, log_y, rcond=None)

    def pred(Xn):
        nn = len(Xn)
        nf2 = Xn.shape[1]
        ps = [np.ones((nn, 1))]
        for j in range(nf2):
            ps.append(Xn[:, j : j + 1])
        for j in range(nf2):
            ps.append(Xn[:, j : j + 1] ** 2)
        for j in range(nf2):
            for k in range(j + 1, nf2):
                ps.append((Xn[:, j] * Xn[:, k]).reshape(-1, 1))
        return np.exp(np.hstack(ps) @ coef)

    return pred, coef


def fit_bayes_logquad(X, y, **_kwargs):
    """BayesLogQuad: y = exp(quadratic in mixed_both), Bayesian Ridge.  Features: wt_epoch."""
    from sklearn.linear_model import BayesianRidge

    log_y = np.log(y)
    n, nf = X.shape
    parts = [np.ones((n, 1))]
    for j in range(nf):
        parts.append(X[:, j : j + 1])
    for j in range(nf):
        parts.append(X[:, j : j + 1] ** 2)
    for j in range(nf):
        for k in range(j + 1, nf):
            parts.append((X[:, j] * X[:, k]).reshape(-1, 1))
    Z = np.hstack(parts)

    mu = Z.mean(axis=0)
    sigma = Z.std(axis=0)
    mu[0], sigma[0] = 0.0, 1.0
    sigma[sigma < 1e-12] = 1.0
    Z_std = (Z - mu) / sigma

    br = BayesianRidge(max_iter=1000, tol=1e-8, fit_intercept=False, compute_score=True)
    br.fit(Z_std, log_y)

    coef = br.coef_ / sigma
    coef[0] -= np.sum(br.coef_ * mu / sigma)

    def pred(Xn):
        nn = len(Xn)
        nf2 = Xn.shape[1]
        ps = [np.ones((nn, 1))]
        for j in range(nf2):
            ps.append(Xn[:, j : j + 1])
        for j in range(nf2):
            ps.append(Xn[:, j : j + 1] ** 2)
        for j in range(nf2):
            for k in range(j + 1, nf2):
                ps.append((Xn[:, j] * Xn[:, k]).reshape(-1, 1))
        return np.exp(np.hstack(ps) @ coef)

    return pred, coef


def fit_huber_logquad(X, y, **_kwargs):
    """HuberLogQuad: y = exp(quadratic in mixed_both), Huber robust loss.  Features: wt_epoch."""
    from sklearn.linear_model import HuberRegressor

    log_y = np.log(y)
    n, nf = X.shape
    parts = [np.ones((n, 1))]
    for j in range(nf):
        parts.append(X[:, j : j + 1])
    for j in range(nf):
        parts.append(X[:, j : j + 1] ** 2)
    for j in range(nf):
        for k in range(j + 1, nf):
            parts.append((X[:, j] * X[:, k]).reshape(-1, 1))
    Z = np.hstack(parts)

    log_y_median = np.median(log_y)
    mad = np.median(np.abs(log_y - log_y_median))
    eps_huber = max(1.35, 1.35 * mad / 0.6745)

    huber = HuberRegressor(epsilon=eps_huber, max_iter=1000, fit_intercept=False, alpha=0.0)
    huber.fit(Z, log_y)
    coef = huber.coef_

    def pred(Xn):
        nn = len(Xn)
        nf2 = Xn.shape[1]
        ps = [np.ones((nn, 1))]
        for j in range(nf2):
            ps.append(Xn[:, j : j + 1])
        for j in range(nf2):
            ps.append(Xn[:, j : j + 1] ** 2)
        for j in range(nf2):
            for k in range(j + 1, nf2):
                ps.append((Xn[:, j] * Xn[:, k]).reshape(-1, 1))
        return np.exp(np.hstack(ps) @ coef)

    return pred, coef


# =========================================================================
# Slice label constructors — LaTeX functional forms on p0_sc=0 slice
# =========================================================================
# Notation: p = p1_starcoder, L = ln(starcoder_epochs_phase1)
# On this slice: p0_sc=0 => phase_0 = 100% nemotron


def _f(v, fmt=".3f"):
    """Format a number for LaTeX."""
    if abs(v) < 0.001:
        return f"{v:.1e}"
    return f"{v:{fmt}}"


def label_linear(params):
    c0, _, c2 = params
    return rf"$y = {_f(c0)} {c2:+.3f}\,p$"


def label_quadratic(params):
    c0, _, c2, _, c4, _ = params
    return rf"$y = {_f(c0)} {c2:+.3f}\,p {c4:+.3f}\,p^2$"


def label_quadratic_4d(params):
    # 15 params for full quadratic on [p0, p1, log_ep0, log_ep1]
    # On slice p0=0, log_ep0=log(EPS)=const: absorb into effective coefficients
    c = params
    le = np.log(EPS)
    const = c[0] + c[3] * le + c[7] * le**2
    b_p = c[2] + c[12] * le
    b_pp = c[6]
    b_L = c[4] + c[14] * le
    b_LL = c[8]
    b_pL = c[13]
    return (
        rf"$y = {_f(const)} {b_p:+.2f}\,p {b_pp:+.2f}\,p^2"
        rf" {b_L:+.2f}\,L {b_LL:+.3f}\,L^2 {b_pL:+.2f}\,pL$"
    )


def label_loglinear(params):
    # params = [c0, c1, c2, c3, c4] for [1, p0, p1, log_ep0, log_ep1]
    # On slice p0=0: log(y) = (c0 + c3*log_eps) + c2*p + c4*L
    c = params
    le = np.log(EPS)
    const = c[0] + c[3] * le
    return rf"$y = \exp({_f(const)} {c[2]:+.3f}\,p {c[4]:+.3f}\,L)$"


def label_powerlaw(params):
    a0, _, b01, g0 = params[0], params[1], params[2], params[3]
    a1, _, b11, g1 = params[4], params[5], params[6], params[7]
    c = params[8]
    return (
        rf"$\mathrm{{SoftPlus}}({_f(a0)}{b01:+.1f}\,p)^{{-{g0:.2f}}}"
        rf" + \mathrm{{SoftPlus}}({_f(a1)}{b11:+.1f}\,p)^{{-{g1:.2f}}}"
        rf" + {_f(c)}$"
    )


def label_dml_m1(params):
    c = params[0]
    k0, t0 = params[1], params[2]
    k1, t1 = params[3], params[4]
    k2, t2 = params[5], params[6]
    k3, t3 = params[7], params[8]
    const = c + k0 * np.exp(np.clip(t0 * 0.5, -20, 20)) + k1
    return (
        rf"${_f(const)} {k2:+.3f}\,e^{{{t2:.1f}(1-p)/2}}"
        rf" {k3:+.3f}\,e^{{{t3:.1f}\,p/2}}$"
    )


def label_slodm(params):
    E = params[0]
    C0, g0 = np.exp(params[1]), np.exp(params[2])
    C2, g2 = np.exp(params[5]), np.exp(params[6])
    C3, g3 = np.exp(params[7]), np.exp(params[8])
    c0_val = C0 * 0.5**g0
    return (
        rf"${_f(E)} + \frac{{1}}{{{_f(c0_val)}"
        rf" + {_f(C2)}\!\left(\frac{{1-p}}{{2}}\right)^{{{g2:.2f}}}"
        rf" + {_f(C3)}\!\left(\frac{{p}}{{2}}\right)^{{{g3:.2f}}}}}$"
    )


def label_bimix(params):
    C = params[0]
    A0, a0 = np.exp(params[1]), np.exp(params[2])
    A1, a1 = np.exp(params[3]), np.exp(params[4])
    A2, a2 = np.exp(params[5]), np.exp(params[6])
    A3, a3 = np.exp(params[7]), np.exp(params[8])
    c_n0 = A0 / (0.5 + 1e-3) ** a0
    c_s0 = A1 / (0 + 1e-3) ** a1
    const = C + c_n0 + c_s0
    return (
        rf"${_f(const)}"
        rf" + \frac{{{_f(A2)}}}{{((1\!-\!p)/2+\varepsilon)^{{{a2:.2f}}}}}"
        rf" + \frac{{{_f(A3)}}}{{(p/2+\varepsilon)^{{{a3:.2f}}}}}$"
    )


def label_cobb_douglas(params):
    C, log_A = params[0], params[1]
    alphas = np.abs(params[2:6]) + 1e-6
    return (
        rf"${_f(C)} - e^{{{_f(log_A)}}}"
        rf"\prod r_j^{{\alpha_j}}$"
        rf" \ $\alpha=[{alphas[0]:.2f},{alphas[1]:.2f},{alphas[2]:.2f},{alphas[3]:.2f}]$"
    )


def label_translog(params):
    # params = [a0, a1..a4, a1^2..a4^2, cross terms] = 15 coefficients
    # On slice: r_nem0=0.5, r_sc0≈0, r_nem1=(1-p)/2, r_sc1=p/2
    # Show general form
    c = params
    return rf"$\exp({_f(c[0])} + \sum a_j \ln r_j + \sum b_{{jk}} \ln r_j \ln r_k)$"


def label_stone_geary(params):
    C, log_A = params[0], params[1]
    gammas = np.clip(params[2:6], 0, 0.2)
    betas = np.abs(params[6:10]) + 1e-6
    return (
        rf"${_f(C)} - e^{{{_f(log_A)}}}"
        rf"\prod (r_j - \gamma_j)^{{\beta_j}}$"
        rf" \ $\gamma=[{gammas[0]:.3f},{gammas[1]:.3f},{gammas[2]:.3f},{gammas[3]:.3f}]$"
    )


def label_ces(params):
    C, log_A, rho = params[0], params[1], np.clip(params[2], -10, 0.99)
    log_a = params[3:7]
    a = np.exp(log_a)
    a = a / a.sum()
    return (
        rf"${_f(C)} - {_f(np.exp(log_A))}"
        rf"(\sum a_j r_j^{{{rho:.2f}}})^{{{1/rho:.2f}}}$"
        rf" \ $a=[{a[0]:.2f},{a[1]:.2f},{a[2]:.2f},{a[3]:.2f}]$"
    )


def label_logquad_mix(params):
    # Same structure as label_quadratic_4d but wrapped in exp()
    c = params
    le = np.log(EPS)
    const = c[0] + c[3] * le + c[7] * le**2
    b_p = c[2] + c[12] * le
    b_pp = c[6]
    b_L = c[4] + c[14] * le
    b_LL = c[8]
    b_pL = c[13]
    return (
        rf"$\exp({_f(const)} {b_p:+.2f}\,p {b_pp:+.2f}\,p^2"
        rf" {b_L:+.2f}\,L {b_LL:+.3f}\,L^2 {b_pL:+.2f}\,pL)$"
    )


def label_ridge_translog(params):
    c = params
    return rf"$\exp({_f(c[0])} + \sum a_j \ln r_j + \sum b_{{jk}} \ln r_j \ln r_k)$ [Ridge]"


def label_scheffe_log(params):
    # params: [b1..b4, b12..b34, g1..g4] = 14 coefficients
    # On slice: r_nem0=0.5, r_sc0≈0, r_nem1=(1-p)/2, r_sc1=p/2
    return rf"$\sum \beta_j r_j + \sum \beta_{{jk}} r_j r_k + \sum \gamma_j r_j \ln r_j$"


def label_ilr_quad(params):
    c = params
    return rf"${_f(c[0])} + \mathrm{{quad}}(\mathrm{{ILR}}(\mathbf{{r}}))$"


def label_elastic_logquad(params):
    c = params
    le = np.log(EPS)
    const = c[0] + c[3] * le + c[7] * le**2
    b_p = c[2] + c[12] * le
    b_pp = c[6]
    b_L = c[4] + c[14] * le
    b_LL = c[8]
    b_pL = c[13]
    return (
        rf"$\exp({_f(const)} {b_p:+.2f}\,p {b_pp:+.2f}\,p^2"
        rf" {b_L:+.2f}\,L {b_LL:+.3f}\,L^2 {b_pL:+.02f}\,pL)$ [EN]"
    )


def label_logquad_weight(params):
    # 6 params: [1, p0, p1, p0², p1², p0*p1]
    # On slice p0=0: log(y) = c[0] + c[2]*p + c[4]*p²
    c = params
    return rf"$\exp({_f(c[0])} {c[2]:+.3f}\,p {c[4]:+.3f}\,p^2)$"


def label_bayes_logquad(params):
    c = params
    le = np.log(EPS)
    const = c[0] + c[3] * le + c[7] * le**2
    b_p = c[2] + c[12] * le
    b_pp = c[6]
    b_L = c[4] + c[14] * le
    b_LL = c[8]
    b_pL = c[13]
    return (
        rf"$\exp({_f(const)} {b_p:+.2f}\,p {b_pp:+.2f}\,p^2"
        rf" {b_L:+.2f}\,L {b_LL:+.003f}\,L^2 {b_pL:+.02f}\,pL)$ [Bayes]"
    )


def label_huber_logquad(params):
    c = params
    le = np.log(EPS)
    const = c[0] + c[3] * le + c[7] * le**2
    b_p = c[2] + c[12] * le
    b_pp = c[6]
    b_L = c[4] + c[14] * le
    b_LL = c[8]
    b_pL = c[13]
    return (
        rf"$\exp({_f(const)} {b_p:+.2f}\,p {b_pp:+.2f}\,p^2"
        rf" {b_L:+.2f}\,L {b_LL:+.003f}\,L^2 {b_pL:+.02f}\,pL)$ [Huber]"
    )


# =========================================================================
# Model registry
# =========================================================================
MODELS: list[ModelSpec] = [
    ModelSpec("Linear", fit_linear, "weight", label_linear, "tab:blue", "--"),
    ModelSpec("Quadratic", fit_quadratic, "weight", label_quadratic, "tab:cyan", "--"),
    ModelSpec("Quadratic(w{-}e)", fit_quadratic_4d, "wt_epoch", label_quadratic_4d, "tab:orange", "--"),
    ModelSpec("LogLinear(w{-}e)", fit_loglinear, "wt_epoch", label_loglinear, "tab:brown", "-"),
    ModelSpec("PowerLaw", fit_powerlaw, "weight", label_powerlaw, "black", "-"),
    ModelSpec("DML M1", fit_dml_m1, "vdom", label_dml_m1, "tab:green", "-"),
    ModelSpec("SLODM", fit_slodm, "vdom", label_slodm, "tab:red", "-"),
    ModelSpec("BiMix", fit_bimix, "vdom", label_bimix, "tab:purple", "-"),
    ModelSpec("Cobb-Douglas", fit_cobb_douglas, "vdom", label_cobb_douglas, "tab:olive", "-"),
    ModelSpec("CES", fit_ces, "vdom", label_ces, "darkgoldenrod", "-"),
    ModelSpec("Translog", fit_translog, "vdom", label_translog, "teal", "--"),
    ModelSpec("Stone-Geary", fit_stone_geary, "vdom", label_stone_geary, "deeppink", "-"),
    ModelSpec("LogQuad(w{-}e)", fit_logquad_mix, "wt_epoch", label_logquad_mix, "navy", "--"),
    ModelSpec("RidgeTranslog", fit_ridge_translog, "vdom", label_ridge_translog, "darkviolet", "--"),
    ModelSpec(r"Scheff\'e+log", fit_scheffe_log, "vdom", label_scheffe_log, "sienna", "-"),
    ModelSpec("ILR Quad", fit_ilr_quad, "vdom", label_ilr_quad, "darkseagreen", "-"),
    ModelSpec("ElasticLogQuad(w{-}e)", fit_elastic_logquad, "wt_epoch", label_elastic_logquad, "crimson", "-."),
    ModelSpec("LogQuad(weight)", fit_logquad_weight, "weight", label_logquad_weight, "gray", ":"),
    ModelSpec("BayesLogQuad(w{-}e)", fit_bayes_logquad, "wt_epoch", label_bayes_logquad, "mediumblue", "-."),
    ModelSpec("HuberLogQuad(w{-}e)", fit_huber_logquad, "wt_epoch", label_huber_logquad, "darkorange", ":"),
]


# =========================================================================
# Slice builders (for 1D and 2D predictions)
# =========================================================================
def make_slice_weight(g: np.ndarray) -> np.ndarray:
    """Build weight features for the p0=0 slice (100% Nemotron phase 0)."""
    return np.column_stack([np.zeros(len(g)), g])


def make_slice_vdom(g: np.ndarray) -> np.ndarray:
    """Build vdom features for the p0=0 slice."""
    n = len(g)
    return np.column_stack([np.full(n, 0.5), np.zeros(n), 0.5 * (1 - g), 0.5 * g])


def make_slice_wt_epoch(g: np.ndarray) -> np.ndarray:
    """Build wt_epoch features for the p0=0 slice."""
    n = len(g)
    return np.column_stack([np.zeros(n), g, np.full(n, np.log(EPS)), np.log(SC_EPOCH_MULT * g + EPS)])


def make_slice_for_kind(kind: FeatureKind, g: np.ndarray) -> np.ndarray:
    """Dispatch to the correct slice builder."""
    if kind == "weight":
        return make_slice_weight(g)
    elif kind == "vdom":
        return make_slice_vdom(g)
    elif kind == "wt_epoch":
        return make_slice_wt_epoch(g)
    else:
        raise ValueError(f"Unknown feature kind: {kind}")


def make_2d_grid(g0: np.ndarray, g1: np.ndarray, kind: FeatureKind = "weight"):
    """Build a 2D feature matrix for a meshgrid of (p0_sc, p1_sc) values.

    Args:
        g0: p0_sc grid (1D array)
        g1: p1_sc grid (1D array)
        kind: Feature type

    Returns:
        (X_grid, P0, P1) where X_grid is (M*N, d) features, P0 and P1 are (M, N) meshgrids
    """
    p0, p1 = np.meshgrid(g0, g1, indexing="ij")
    p0f, p1f = p0.ravel(), p1.ravel()

    if kind == "weight":
        return np.column_stack([p0f, p1f]), p0, p1
    elif kind == "wt_epoch":
        return np.column_stack([
            p0f, p1f,
            np.log(SC_EPOCH_MULT * p0f + EPS),
            np.log(SC_EPOCH_MULT * p1f + EPS),
        ]), p0, p1
    else:  # vdom
        return np.column_stack([
            0.5 * (1 - p0f), 0.5 * p0f,
            0.5 * (1 - p1f), 0.5 * p1f,
        ]), p0, p1
