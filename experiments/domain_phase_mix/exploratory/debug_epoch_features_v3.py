# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy", "scikit-learn", "matplotlib"]
# ///
"""v3: Parametric models only, with explicit functional forms in legends (LaTeX).

Based on v2 and literature_scaling_laws.py findings.
All models return fitted parameters; legends show the explicit
functional form on the p0_sc=0 slice with actual numeric values.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{opensans}\renewcommand{\familydefault}{\sfdefault}"
plt.rcParams["font.family"] = "sans-serif"
script_dir = Path(__file__).parent
_csv_name = os.environ.get("DATA_CSV", "two_phase_starcoder.csv")
df = pd.read_csv(script_dir / _csv_name)
df = df[df["status"] == "completed"].copy()

TARGET = "eval/paloma/dolma_100_programing_languages/bpb"
y = df[TARGET].values
N = len(df)

SC_EPOCH_MULT = 13.2289
NEM_EPOCH_MULT = 0.5

if __name__ == "__main__":
    print(f"N={N}, target={TARGET}")
    print(f"y range: [{y.min():.4f}, {y.max():.4f}], median={np.median(y):.4f}")


# =========================================================================
# Feature construction
# =========================================================================
p0_sc = df["phase_0_starcoder"].values
p1_sc = df["phase_1_starcoder"].values

X_weight = np.column_stack([p0_sc, p1_sc])
X_vdom = np.column_stack(
    [
        0.5 * (1 - p0_sc),
        0.5 * p0_sc,
        0.5 * (1 - p1_sc),
        0.5 * p1_sc,
    ]
)

EPS = 1e-8
sc_ep0 = df["phase_0_starcoder_epochs"].values
sc_ep1 = df["phase_1_starcoder_epochs"].values
# Weight-epoch features: proportions + log-epoch counts
X_wt_epoch = np.column_stack([p0_sc, p1_sc, np.log(sc_ep0 + EPS), np.log(sc_ep1 + EPS)])


# =========================================================================
# Cross-validation
# =========================================================================
def cv_metrics(fit_fn, X, y, n_folds=5, seed=42):
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
# Define all models to test
# =========================================================================
# (name, fit_fn, X_data, label_fn, color, linestyle)
MODELS = [
    ("Linear", fit_linear, X_weight, label_linear, "tab:blue", "--"),
    ("Quadratic", fit_quadratic, X_weight, label_quadratic, "tab:cyan", "--"),
    ("Quadratic(w{-}e)", fit_quadratic_4d, X_wt_epoch, label_quadratic_4d, "tab:orange", "--"),
    ("LogLinear(w{-}e)", fit_loglinear, X_wt_epoch, label_loglinear, "tab:brown", "-"),
    ("PowerLaw", fit_powerlaw, X_weight, label_powerlaw, "black", "-"),
    ("DML M1", fit_dml_m1, X_vdom, label_dml_m1, "tab:green", "-"),
    ("SLODM", fit_slodm, X_vdom, label_slodm, "tab:red", "-"),
    ("BiMix", fit_bimix, X_vdom, label_bimix, "tab:purple", "-"),
    ("Cobb-Douglas", fit_cobb_douglas, X_vdom, label_cobb_douglas, "tab:olive", "-"),
    ("CES", fit_ces, X_vdom, label_ces, "darkgoldenrod", "-"),
    ("Translog", fit_translog, X_vdom, label_translog, "teal", "--"),
    ("Stone-Geary", fit_stone_geary, X_vdom, label_stone_geary, "deeppink", "-"),
    ("LogQuad(w{-}e)", fit_logquad_mix, X_wt_epoch, label_logquad_mix, "navy", "--"),
    ("RidgeTranslog", fit_ridge_translog, X_vdom, label_ridge_translog, "darkviolet", "--"),
    (r"Scheff\'e+log", fit_scheffe_log, X_vdom, label_scheffe_log, "sienna", "-"),
    ("ILR Quad", fit_ilr_quad, X_vdom, label_ilr_quad, "darkseagreen", "-"),
    ("ElasticLogQuad(w{-}e)", fit_elastic_logquad, X_wt_epoch, label_elastic_logquad, "crimson", "-."),
    ("LogQuad(weight)", fit_logquad_weight, X_weight, label_logquad_weight, "gray", ":"),
    ("BayesLogQuad(w{-}e)", fit_bayes_logquad, X_wt_epoch, label_bayes_logquad, "mediumblue", "-."),
    ("HuberLogQuad(w{-}e)", fit_huber_logquad, X_wt_epoch, label_huber_logquad, "darkorange", ":"),
]


# =========================================================================
# Slice builders (module-level for importability)
# =========================================================================
slice_grid = np.linspace(0.002, 0.998, 300)


def make_slice_weight(g):
    return np.column_stack([np.zeros(len(g)), g])


def make_slice_vdom(g):
    n = len(g)
    return np.column_stack([np.full(n, 0.5), np.zeros(n), 0.5 * (1 - g), 0.5 * g])


def make_slice_wt_epoch(g):
    n = len(g)
    return np.column_stack([np.zeros(n), g, np.full(n, np.log(EPS)), np.log(SC_EPOCH_MULT * g + EPS)])


def feature_kind(X_data):
    """Return 'weight', 'wt_epoch', or 'vdom' based on identity with module globals."""
    if X_data is X_weight:
        return "weight"
    elif X_data is X_wt_epoch:
        return "wt_epoch"
    return "vdom"


def make_2d_grid(g0, g1, kind="weight"):
    """Build a 2D feature matrix for a meshgrid of (p0_sc, p1_sc) values."""
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
    else:
        return np.column_stack([
            0.5 * (1 - p0f), 0.5 * p0f,
            0.5 * (1 - p1f), 0.5 * p1f,
        ]), p0, p1


if __name__ == "__main__":
    # =====================================================================
    # Cross-validation
    # =====================================================================
    print("\n" + "=" * 80)
    print(f"{'Model':<14} {'R2':>7} {'RMSE':>7} {'Spearman':>9} {'RMSE_bot':>9}")
    print("=" * 80)

    cv_results = {}
    for name, fit_fn, X_data, _, _, _ in MODELS:
        m = cv_metrics(fit_fn, X_data, y)
        cv_results[name] = m
        print(f"{name:<14} {m['R2']:>7.4f} {m['RMSE']:>7.4f} {m['Spearman']:>9.4f} {m['RMSE_bot']:>9.4f}")

    # =====================================================================
    # Fit on full data, extract params, build labels
    # =====================================================================
    print("\n\nFitting all models on full data...")

    fitted = []  # (name, pred_fn, params, label_str, slice_preds, color, ls, cv_m)

    for name, fit_fn, X_data, label_fn, color, ls in MODELS:
        pred_fn, params = fit_fn(X_data, y)

        if X_data is X_weight:
            Xs = make_slice_weight(slice_grid)
        elif X_data is X_wt_epoch:
            Xs = make_slice_wt_epoch(slice_grid)
        else:
            Xs = make_slice_vdom(slice_grid)

        preds_slice = pred_fn(Xs)
        label = label_fn(params)

        print(f"\n--- {name} ---")
        print(f"  Params: {params}")
        print(f"  Label:  {label}")

        best_i = np.argmin(preds_slice)
        print(f"  Slice optimal: p1_sc={slice_grid[best_i]:.4f}, pred={preds_slice[best_i]:.4f}")

        fitted.append((name, pred_fn, params, label, preds_slice, color, ls, cv_results[name]))

    # =====================================================================
    # Actual data on the slice
    # =====================================================================
    mask = df["phase_0_nemotron_full"].round(4) == 1.0
    df_slice = df[mask].sort_values("phase_1_starcoder")
    x_actual = df_slice["phase_1_starcoder"].values
    y_actual = df_slice[TARGET].values
    actual_best_i = np.argmin(y_actual)

    print(f"\nActual best (slice): p1_sc={x_actual[actual_best_i]:.4f}, bpb={y_actual[actual_best_i]:.4f}")

    # =====================================================================
    # Plot
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(36, 9))

    XLABEL = r"$p$ = StarCoder fraction in the Second Phase"
    YLABEL = r"eval/paloma/dolma\_100\_programing\_languages/bpb"
    NOTATION_LINE = (
        r"\small $p = p_1^{\mathrm{sc}}$,\; $L = \ln(\mathrm{sc\_epochs}_1)$,"
        r"\; (w{-}e) = weight{-}epoch features"
    )
    TOP_MODELS = {"LogQuad(w{-}e)", "Translog", "BayesLogQuad(w{-}e)", "ElasticLogQuad(w{-}e)", "Quadratic(w{-}e)"}

    for panel, (ax, xlim, ylim, title) in enumerate(
        zip(
            axes,
            [(0, 1), (0.1, 0.55), (0.1, 0.55)],
            [(0.85, 1.75), (0.88, 0.97), (0.88, 0.97)],
            ["Full range", "Zoomed: all models", "Zoomed: top 5"],
        )
    ):
        ax.scatter(x_actual, y_actual, s=50, c="black", zorder=10, label="Actual data")

        for name, _, params, label, preds_s, color, ls, cv_m in fitted:
            if panel == 2 and name not in TOP_MODELS:
                continue
            if panel == 0:
                lbl = name
            elif panel == 2:
                lbl = rf"{name}: {label}"
            else:
                lbl = label
            ax.plot(slice_grid, preds_s, label=lbl, linewidth=2.0, color=color, linestyle=ls)

        ax.set_xlabel(XLABEL, fontsize=14)
        ax.set_ylabel(YLABEL, fontsize=14)
        ax.set_title(rf"Slice: 100\% Nemotron in the First Phase --- {title}", fontsize=15)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=12)

        secax = ax.secondary_xaxis("top", functions=(lambda w: w * SC_EPOCH_MULT, lambda e: e / SC_EPOCH_MULT))
        secax.set_xlabel(r"StarCoder epochs in the Second Phase", fontsize=12)
        secax.tick_params(labelsize=10)

        if panel == 0:
            ax.legend(fontsize=9, loc="upper right", ncol=2)
        elif panel == 2:
            ax.plot([], [], " ", label=NOTATION_LINE)
            ax.legend(fontsize=11, loc="upper left", framealpha=0.95)

    fig.tight_layout()
    out_path = script_dir / "debug_epoch_features_v3.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved {out_path}")

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 80)
    print("SUMMARY -- Ranked by RMSE_bot")
    print("=" * 80)
    print(f"{'Model':<14} {'RMSE_bot':>9} {'Spearman':>9} {'R2':>7}  {'Slice opt':>10} {'Pred':>7}")
    print("-" * 65)
    for name, _, params, label, preds_s, _, _, cv_m in sorted(fitted, key=lambda x: x[7]["RMSE_bot"]):
        best_i = np.argmin(preds_s)
        print(
            f"{name:<14} {cv_m['RMSE_bot']:>9.4f} {cv_m['Spearman']:>9.4f} {cv_m['R2']:>7.4f}  "
            f"p1={slice_grid[best_i]:>6.4f} {preds_s[best_i]:>7.4f}"
        )
    print(f"{'Actual':14s} {'':>9} {'':>9} {'':>7}  p1={x_actual[actual_best_i]:>6.4f} {y_actual[actual_best_i]:>7.4f}")
