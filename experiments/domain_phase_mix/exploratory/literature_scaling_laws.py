# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy", "scikit-learn", "matplotlib"]
# ///
"""Literature scaling laws for data mixing, applied to two_phase_starcoder.

Papers:
1. Data Mixing Laws (Ye et al., 2403.16952)
2. Scaling Laws for Optimal Data Mixtures (Liu et al., 2507.09404)
3. BiMix (Park et al., 2405.14908)
4. Aioli (Shin et al., 2411.05735)

For each: implement the functional form, adapt to multi-phase setting,
consider epoch extensions, cross-validate, plot on phase_0=100% nemotron slice.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

script_dir = Path(__file__).parent
results_path = script_dir / "literature_scaling_laws.results"

df = pd.read_csv(script_dir / "two_phase_starcoder.csv")
df = df[df["status"] == "completed"].copy()

TARGET = "eval/paloma/dolma_100_programing_languages/bpb"
y = df[TARGET].values
N = len(df)

NEM_EPOCH_MULT = 0.5
SC_EPOCH_MULT = 13.2289

print(f"N={N}, target={TARGET}")
print(f"y range: [{y.min():.4f}, {y.max():.4f}], median={np.median(y):.4f}")


# =========================================================================
# Feature construction
# =========================================================================
def make_vdom(df):
    """Virtual domain proportions [r_nem0, r_sc0, r_nem1, r_sc1], sum=1."""
    p0 = df["phase_0_starcoder"].values
    p1 = df["phase_1_starcoder"].values
    return np.column_stack([0.5 * (1 - p0), 0.5 * p0, 0.5 * (1 - p1), 0.5 * p1])


def make_epoch(df):
    """Epoch counts [E_nem0, E_sc0, E_nem1, E_sc1]."""
    p0 = df["phase_0_starcoder"].values
    p1 = df["phase_1_starcoder"].values
    return np.column_stack(
        [
            NEM_EPOCH_MULT * (1 - p0),
            SC_EPOCH_MULT * p0,
            NEM_EPOCH_MULT * (1 - p1),
            SC_EPOCH_MULT * p1,
        ]
    )


def make_weight(df):
    """Simple weights [p0_sc, p1_sc]."""
    return df[["phase_0_starcoder", "phase_1_starcoder"]].values


def slice_vdom(p1_grid):
    """Virtual domain features for p0_sc=0 slice."""
    n = len(p1_grid)
    return np.column_stack([np.full(n, 0.5), np.zeros(n), 0.5 * (1 - p1_grid), 0.5 * p1_grid])


def slice_epoch(p1_grid):
    """Epoch features for p0_sc=0 slice."""
    n = len(p1_grid)
    return np.column_stack(
        [np.full(n, NEM_EPOCH_MULT), np.zeros(n), NEM_EPOCH_MULT * (1 - p1_grid), SC_EPOCH_MULT * p1_grid]
    )


def slice_weight(p1_grid):
    """Weight features for p0_sc=0 slice."""
    return np.column_stack([np.zeros(len(p1_grid)), p1_grid])


def search_vdom(p0_rand, p1_rand):
    return np.column_stack([0.5 * (1 - p0_rand), 0.5 * p0_rand, 0.5 * (1 - p1_rand), 0.5 * p1_rand])


def search_epoch(p0_rand, p1_rand):
    return np.column_stack(
        [
            NEM_EPOCH_MULT * (1 - p0_rand),
            SC_EPOCH_MULT * p0_rand,
            NEM_EPOCH_MULT * (1 - p1_rand),
            SC_EPOCH_MULT * p1_rand,
        ]
    )


def search_weight(p0_rand, p1_rand):
    return np.column_stack([p0_rand, p1_rand])


X_vdom = make_vdom(df)
X_epoch = make_epoch(df)
X_weight = make_weight(df)


# =========================================================================
# Cross-validation
# =========================================================================
def cv_metrics(fit_fn, X, y, n_folds=5, seed=42):
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    r2s, rmses, spearmans, rmse_bots = [], [], [], []
    median_y = np.median(y)
    for tr, te in kf.split(X):
        result = fit_fn(X[tr], y[tr])
        pred_fn = result[0] if isinstance(result, tuple) else result
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
# Baselines
# =========================================================================
def _softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def fit_powerlaw(X, y, n_restarts=200, seed=42):
    """PowerLaw: y = Σ_i (alpha_i + beta_i@x)^(-gamma_i) + c"""
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
            res = minimize(loss, np.array(p0), method="L-BFGS-B", bounds=bnd, options={"maxiter": 2000, "ftol": 1e-12})
            if res.fun < best_l:
                best_l, best_p = res.fun, res.x
        except:
            continue
    return lambda Xn: model(Xn, best_p)


def fit_gp(X, y, seed=42):
    """GP(Matern) baseline."""
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    k = ConstantKernel(1.0) * Matern(length_scale=np.ones(X.shape[1]), nu=2.5) + WhiteKernel(0.001)
    gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=10, normalize_y=True)
    gp.fit(Xs, y)
    return lambda Xn: gp.predict(sc.transform(Xn))


# =========================================================================
# Paper 1: Data Mixing Laws (Ye et al., 2403.16952)
#   M4: c + k * exp(Σ t_j * r_j)         — LogLinear (paper's selected)
#   M1: c + Σ_j k_j * exp(t_j * r_j)     — Sum of Exponentials
# =========================================================================
def fit_dml_m4(X, y, n_restarts=150, seed=42):
    """DML M4 (LogLinear): c + k * exp(Σ t_j * r_j)"""
    rng = np.random.default_rng(seed)
    nf = X.shape[1]

    def model(X, p):
        return p[0] + p[1] * np.exp(np.clip(X @ p[2:], -20, 20))

    def loss(p):
        return float(np.sum((model(X, p) - y) ** 2))

    best_l, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = np.array([y.mean(), rng.uniform(0.1, 2), *rng.uniform(-5, 5, nf)])
        try:
            res = minimize(loss, p0, method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-12})
            if res.fun < best_l:
                best_l, best_p = res.fun, res.x
        except:
            continue
    return lambda Xn: model(Xn, best_p)


def fit_dml_m1(X, y, n_restarts=150, seed=42):
    """DML M1 (Sum-Exp): c + Σ_j k_j * exp(t_j * r_j)"""
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
            res = minimize(loss, p0, method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-12})
            if res.fun < best_l:
                best_l, best_p = res.fun, res.x
        except:
            continue
    return lambda Xn: model(Xn, best_p)


# =========================================================================
# Paper 2: Scaling Laws for Optimal Data Mixtures (Liu et al., 2507.09404)
#   Additive: L = E + 1 / Σ_j (C_j * r_j^γ_j)
# =========================================================================
def fit_slodm(X, y, n_restarts=150, seed=42):
    """SLODM Additive: E + 1/Σ(C_j * r_j^γ_j)
    Log-parameterization: C_j = exp(lc_j), γ_j = exp(lg_j) to enforce positivity.
    """
    rng = np.random.default_rng(seed)
    nf = X.shape[1]

    def model(X, p):
        E = p[0]
        denom = np.full(len(X), 1e-10)
        for j in range(nf):
            Cj = np.exp(np.clip(p[1 + 2 * j], -10, 10))
            gj = np.exp(np.clip(p[2 + 2 * j], -3, 3))
            denom += Cj * np.power(np.maximum(X[:, j], 1e-8), gj)
        return E + 1.0 / denom

    def loss(p):
        return float(np.sum((model(X, p) - y) ** 2))

    best_l, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = [y.min() + rng.normal(0, 0.05)]
        for _ in range(nf):
            p0.extend([rng.normal(0, 1.5), rng.normal(0, 0.5)])
        p0 = np.array(p0)
        try:
            res = minimize(loss, p0, method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-12})
            if res.fun < best_l:
                best_l, best_p = res.fun, res.x
        except:
            continue
    return lambda Xn: model(Xn, best_p)


# =========================================================================
# Paper 3: BiMix (Park et al., 2405.14908)
#   Multi-domain: L = Σ_j A_j / (r_j + eps)^α_j + C
# =========================================================================
def fit_bimix(X, y, n_restarts=150, seed=42):
    """BiMix multi-domain: Σ exp(la_j)/(r_j + eps)^exp(lα_j) + C"""
    rng = np.random.default_rng(seed)
    nf = X.shape[1]
    EPS = 1e-3

    def model(X, p):
        r = np.full(len(X), p[0])
        for j in range(nf):
            Aj = np.exp(np.clip(p[1 + 2 * j], -10, 10))
            aj = np.exp(np.clip(p[2 + 2 * j], -3, 3))
            r += Aj / np.power(X[:, j] + EPS, aj)
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
            res = minimize(loss, p0, method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-12})
            if res.fun < best_l:
                best_l, best_p = res.fun, res.x
        except:
            continue
    return lambda Xn: model(Xn, best_p)


# =========================================================================
# Paper 4: Aioli (Shin et al., 2411.05735)
#   Static + interactions: c + b*exp(Σ t_j*r_j + Σ_{j<k} B_jk*r_j*r_k)
# =========================================================================
def fit_aioli(X, y, n_restarts=150, seed=42):
    """Aioli with cross-domain interactions."""
    rng = np.random.default_rng(seed)
    nf = X.shape[1]
    pairs = [(j, k) for j in range(nf) for k in range(j + 1, nf)]
    np_ = len(pairs)

    def model(X, p):
        exp_arg = X @ p[2 : 2 + nf]
        for idx, (j, k) in enumerate(pairs):
            exp_arg += p[2 + nf + idx] * X[:, j] * X[:, k]
        return p[0] + p[1] * np.exp(np.clip(exp_arg, -20, 20))

    def loss(p):
        return float(np.sum((model(X, p) - y) ** 2))

    n_params = 2 + nf + np_
    best_l, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = [y.mean(), rng.uniform(0.1, 2)]
        p0.extend(rng.uniform(-5, 5, nf).tolist())
        p0.extend(rng.uniform(-10, 10, np_).tolist())
        p0 = np.array(p0)
        try:
            res = minimize(loss, p0, method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-12})
            if res.fun < best_l:
                best_l, best_p = res.fun, res.x
        except:
            continue
    return lambda Xn: model(Xn, best_p)


# =========================================================================
# Improved variants: tied params, hybrid models
# =========================================================================
def fit_dml_m1_tied(X, y, n_restarts=200, seed=42):
    """DML M1 with tied domain params across phases (5 params).
    L = c + k_nem*[exp(t_nem*r_nem0) + exp(t_nem*r_nem1)]
          + k_sc*[exp(t_sc*r_sc0) + exp(t_sc*r_sc1)]
    Features must be [r_nem0, r_sc0, r_nem1, r_sc1].
    """
    rng = np.random.default_rng(seed)

    def model(X, p):
        c, k_nem, t_nem, k_sc, t_sc = p
        return (
            c
            + k_nem * np.exp(np.clip(t_nem * X[:, 0], -20, 20))
            + k_nem * np.exp(np.clip(t_nem * X[:, 2], -20, 20))
            + k_sc * np.exp(np.clip(t_sc * X[:, 1], -20, 20))
            + k_sc * np.exp(np.clip(t_sc * X[:, 3], -20, 20))
        )

    def loss(p):
        return float(np.sum((model(X, p) - y) ** 2))

    best_l, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = np.array([y.mean(), rng.uniform(-1, 1), rng.uniform(-8, 8), rng.uniform(-1, 1), rng.uniform(-8, 8)])
        try:
            res = minimize(loss, p0, method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-12})
            if res.fun < best_l:
                best_l, best_p = res.fun, res.x
        except:
            continue
    return lambda Xn: model(Xn, best_p)


def fit_slodm_linear(X, y, n_restarts=200, seed=42):
    """SLODM + linear correction: E + 1/Σ(C_j*r_j^γ_j) + Σ b_j*r_j
    Combines U-shape from reciprocal power with linear tail adjustment.
    """
    rng = np.random.default_rng(seed)
    nf = X.shape[1]

    def model(X, p):
        E = p[0]
        denom = np.full(len(X), 1e-10)
        for j in range(nf):
            Cj = np.exp(np.clip(p[1 + 2 * j], -10, 10))
            gj = np.exp(np.clip(p[2 + 2 * j], -3, 3))
            denom += Cj * np.power(np.maximum(X[:, j], 1e-8), gj)
        linear = X @ p[1 + 2 * nf :]
        return E + 1.0 / denom + linear

    def loss(p):
        return float(np.sum((model(X, p) - y) ** 2))

    best_l, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = [y.min() + rng.normal(0, 0.05)]
        for _ in range(nf):
            p0.extend([rng.normal(0, 1.5), rng.normal(0, 0.5)])
        p0.extend(rng.normal(0, 0.5, nf).tolist())
        p0 = np.array(p0)
        try:
            res = minimize(loss, p0, method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-12})
            if res.fun < best_l:
                best_l, best_p = res.fun, res.x
        except:
            continue
    return lambda Xn: model(Xn, best_p)


def fit_dml_m1_weighted(X, y, n_restarts=200, seed=42):
    """DML M1 with inverse-loss weighting to focus on minimum region."""
    rng = np.random.default_rng(seed)
    nf = X.shape[1]
    # Weight: points with low y get high weight
    w = 1.0 / np.maximum(y - y.min() + 0.01, 0.01)
    w = w / w.sum() * len(y)  # normalize so total weight = N

    def model(X, p):
        r = np.full(len(X), p[0])
        for j in range(nf):
            r += p[1 + 2 * j] * np.exp(np.clip(p[2 + 2 * j] * X[:, j], -20, 20))
        return r

    def loss(p):
        return float(np.sum(w * (model(X, p) - y) ** 2))

    best_l, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = [y.mean()]
        for _ in range(nf):
            p0.extend([rng.uniform(-1, 1), rng.uniform(-8, 8)])
        p0 = np.array(p0)
        try:
            res = minimize(loss, p0, method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-12})
            if res.fun < best_l:
                best_l, best_p = res.fun, res.x
        except:
            continue
    return lambda Xn: model(Xn, best_p)


# =========================================================================
# Run everything paper by paper
# =========================================================================
p1_grid = np.linspace(0.002, 0.998, 300)
rng = np.random.default_rng(123)
p0_rand = rng.uniform(0, 1, 500_000)
p1_rand = rng.uniform(0, 1, 500_000)

# Actual data on slice
mask = df["phase_0_nemotron_full"].round(4) == 1.0
df_slice = df[mask].sort_values("phase_1_starcoder")
x_actual = df_slice["phase_1_starcoder"].values
y_actual = df_slice[TARGET].values
actual_best_slice = x_actual[np.argmin(y_actual)]
actual_best_bpb = y_actual.min()
best_2d_idx = np.argmin(y)
actual_best_2d = X_weight[best_2d_idx]

results_lines = []
all_cv = {}
all_pred_fns = {}  # name -> (pred_fn, slice_fn, search_fn)


def log(msg):
    print(msg)
    results_lines.append(msg)


def flush_results():
    with open(results_path, "w") as f:
        f.write("\n".join(results_lines) + "\n")


def run_paper(paper_name, paper_ref, models):
    """Run CV + full fit for a list of (name, fit_fn, X_train, slice_X, search_fn_name)."""
    log(f"\n{'='*70}")
    log(f"{paper_name}")
    log(f"{paper_ref}")
    log(f"{'='*70}")

    for name, fit_fn, X_train, feat_type in models:
        log(f"\n--- {name} ---")
        # CV
        m = cv_metrics(fit_fn, X_train, y)
        all_cv[name] = m
        log(
            f"  CV: R²={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  "
            f"Spearman={m['Spearman']:.4f}  RMSE_bot={m['RMSE_bot']:.4f}"
        )

        # Full fit
        result = fit_fn(X_train, y)
        pred_fn = result[0] if isinstance(result, tuple) else result

        # Slice prediction
        if feat_type == "vdom":
            Xs = slice_vdom(p1_grid)
            Xr = search_vdom(p0_rand, p1_rand)
        elif feat_type == "epoch":
            Xs = slice_epoch(p1_grid)
            Xr = search_epoch(p0_rand, p1_rand)
        else:
            Xs = slice_weight(p1_grid)
            Xr = search_weight(p0_rand, p1_rand)

        preds_slice = pred_fn(Xs)
        best_i = np.argmin(preds_slice)
        log(f"  Slice optimal: p1_sc={p1_grid[best_i]:.4f}, pred={preds_slice[best_i]:.4f}")

        # 2D search
        preds_2d = pred_fn(Xr)
        top128 = np.argsort(preds_2d)[:128]
        opt_p0 = np.mean(p0_rand[top128])
        opt_p1 = np.mean(p1_rand[top128])
        opt_pred = np.mean(preds_2d[top128])
        log(f"  2D optimal: p0_sc={opt_p0:.4f}, p1_sc={opt_p1:.4f}, pred={opt_pred:.4f}")

        all_pred_fns[name] = (pred_fn, Xs, preds_slice)

    log(f"\n  Actual best (slice): p1_sc={actual_best_slice:.4f}, bpb={actual_best_bpb:.4f}")
    log(f"  Actual best (2D):    p0_sc={actual_best_2d[0]:.4f}, p1_sc={actual_best_2d[1]:.4f}, bpb={y[best_2d_idx]:.4f}")
    flush_results()


# ----- Baselines -----
log("Literature Scaling Laws for Data Mixing — Two-Phase StarCoder Experiment")
log(f"N={N}, target={TARGET}")
log(f"y range: [{y.min():.4f}, {y.max():.4f}], median={np.median(y):.4f}")

run_paper(
    "BASELINES",
    "",
    [
        ("PowerLaw (2-term)", fit_powerlaw, X_weight, "weight"),
        ("GP(Matern)", fit_gp, X_weight, "weight"),
    ],
)

# ----- Paper 1: Data Mixing Laws -----
run_paper(
    "Paper 1: Data Mixing Laws (Ye et al., 2403.16952)",
    "M4: c + k*exp(Σ t_j*r_j)  |  M1: c + Σ k_j*exp(t_j*r_j)",
    [
        ("DML_M4 (vdom)", fit_dml_m4, X_vdom, "vdom"),
        ("DML_M1 (vdom)", fit_dml_m1, X_vdom, "vdom"),
        ("DML_M4 (epoch)", fit_dml_m4, X_epoch, "epoch"),
        ("DML_M1 (epoch)", fit_dml_m1, X_epoch, "epoch"),
    ],
)

# ----- Paper 2: SLODM -----
run_paper(
    "Paper 2: Scaling Laws for Optimal Data Mixtures (Liu et al., 2507.09404)",
    "Additive: E + 1/Σ(C_j * r_j^γ_j)",
    [
        ("SLODM (vdom)", fit_slodm, X_vdom, "vdom"),
        ("SLODM (epoch)", fit_slodm, X_epoch, "epoch"),
    ],
)

# ----- Paper 3: BiMix -----
run_paper(
    "Paper 3: BiMix (Park et al., 2405.14908)",
    "Multi-domain: Σ A_j/(r_j+eps)^α_j + C",
    [
        ("BiMix (vdom)", fit_bimix, X_vdom, "vdom"),
        ("BiMix (epoch)", fit_bimix, X_epoch, "epoch"),
    ],
)

# ----- Paper 4: Aioli -----
run_paper(
    "Paper 4: Aioli (Shin et al., 2411.05735)",
    "Static + interactions: c + b*exp(Σ t_j*r_j + Σ B_jk*r_j*r_k)",
    [
        ("Aioli (vdom)", fit_aioli, X_vdom, "vdom"),
        ("Aioli (epoch)", fit_aioli, X_epoch, "epoch"),
    ],
)

# ----- Improved variants -----
run_paper(
    "IMPROVED VARIANTS",
    "Tied params, hybrid models, weighted loss",
    [
        ("DML_M1_tied (vdom)", fit_dml_m1_tied, X_vdom, "vdom"),
        ("SLODM+linear (vdom)", fit_slodm_linear, X_vdom, "vdom"),
        ("DML_M1_weighted (vdom)", fit_dml_m1_weighted, X_vdom, "vdom"),
    ],
)


# =========================================================================
# Summary comparison
# =========================================================================
log(f"\n{'='*70}")
log("SUMMARY — Ranked by RMSE_bot")
log(f"{'='*70}")
log(f"{'Model':<25} {'R²':>7} {'RMSE':>7} {'Spearman':>9} {'RMSE_bot':>9}")
log("-" * 60)
for name, m in sorted(all_cv.items(), key=lambda x: x[1]["RMSE_bot"]):
    log(f"{name:<25} {m['R2']:>7.4f} {m['RMSE']:>7.4f} {m['Spearman']:>9.4f} {m['RMSE_bot']:>9.4f}")
flush_results()


# =========================================================================
# Visualization: all models on the p0_sc=0 slice
# =========================================================================
colors = {
    "PowerLaw (2-term)": "black",
    "GP(Matern)": "gray",
    "DML_M4 (vdom)": "tab:blue",
    "DML_M1 (vdom)": "tab:cyan",
    "DML_M4 (epoch)": "cornflowerblue",
    "DML_M1 (epoch)": "darkturquoise",
    "SLODM (vdom)": "tab:green",
    "SLODM (epoch)": "limegreen",
    "BiMix (vdom)": "tab:red",
    "BiMix (epoch)": "salmon",
    "Aioli (vdom)": "tab:purple",
    "Aioli (epoch)": "orchid",
    "DML_M1_tied (vdom)": "navy",
    "SLODM+linear (vdom)": "darkgreen",
    "DML_M1_weighted (vdom)": "teal",
}
styles = {
    "PowerLaw (2-term)": "-",
    "GP(Matern)": ":",
    "DML_M4 (vdom)": "--",
    "DML_M1 (vdom)": "-",
    "DML_M4 (epoch)": "--",
    "DML_M1 (epoch)": "-",
    "SLODM (vdom)": "-",
    "SLODM (epoch)": "--",
    "BiMix (vdom)": "-",
    "BiMix (epoch)": "--",
    "Aioli (vdom)": "-",
    "Aioli (epoch)": "--",
    "DML_M1_tied (vdom)": "-.",
    "SLODM+linear (vdom)": "-.",
    "DML_M1_weighted (vdom)": ":",
}

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for panel, (ax, xlim, ylim, title) in enumerate(
    zip(
        axes,
        [(0, 1), (0.1, 0.55)],
        [(0.85, 1.75), (0.88, 0.97)],
        ["Full range", "Zoomed: minimum region"],
    )
):
    ax.scatter(x_actual, y_actual, s=30, c="black", zorder=10, label="Actual data")
    for name, (pred_fn, Xs, preds_s) in all_pred_fns.items():
        rmse_b = all_cv[name]["RMSE_bot"]
        lbl = f"{name} (bot={rmse_b:.4f})" if panel == 1 else name
        ax.plot(
            p1_grid, preds_s, label=lbl, linewidth=1.5, color=colors.get(name, "gray"), linestyle=styles.get(name, "-")
        )
    ax.set_xlabel("phase_1_starcoder")
    ax.set_ylabel(TARGET.split("/")[-1])
    ax.set_title(f"Slice: phase_0 = 100% nemotron — {title}")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(fontsize=6, loc="upper right" if panel == 0 else "upper left")

fig.tight_layout()
out_path = script_dir / "literature_scaling_laws.png"
fig.savefig(out_path, dpi=150)
log(f"\nSaved plot to {out_path.name}")
flush_results()
print(f"\nResults written to {results_path}")
