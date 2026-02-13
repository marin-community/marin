# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
"""v2: Fix plotting, add optimization-relevant error metrics.

R² is misleading because extreme points (p1=0 and p1=1) dominate variance.
Better metrics for our purpose:
1. RMSE in the minimum region (BPB < median)
2. Spearman rank correlation (does the model rank runs correctly?)
3. Predicted optimal location vs actual
4. Predicted BPB at actual optimal point
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
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

script_dir = Path(__file__).parent
df = pd.read_csv(script_dir / "two_phase_starcoder.csv")
df = df[df["status"] == "completed"].copy()

TARGET = "eval/paloma/dolma_100_programing_languages/bpb"
y = df[TARGET].values
N = len(df)

SC_EPOCH_MULT = 13.2289
NEM_EPOCH_MULT = 0.5
EPS = 0.01

print(f"N={N}, target={TARGET}")
print(f"y range: [{y.min():.4f}, {y.max():.4f}], median={np.median(y):.4f}")


# -------------------------------------------------------------------------
# Feature builders
# -------------------------------------------------------------------------
def get_features(df):
    p0_sc = df["phase_0_starcoder"].values
    p1_sc = df["phase_1_starcoder"].values
    sc_ep0 = df["phase_0_starcoder_epochs"].values
    sc_ep1 = df["phase_1_starcoder_epochs"].values
    return {
        "weights": np.column_stack([p0_sc, p1_sc]),
        "mixed": np.column_stack([p0_sc, p1_sc, np.log(sc_ep1 + EPS)]),
        "mixed_both": np.column_stack([p0_sc, p1_sc, np.log(sc_ep0 + EPS), np.log(sc_ep1 + EPS)]),
        "log_all": np.column_stack(
            [
                np.log(sc_ep0 + EPS),
                np.log(sc_ep1 + EPS),
                np.log(df["phase_0_nemotron_epochs"].values + EPS),
                np.log(df["phase_1_nemotron_epochs"].values + EPS),
            ]
        ),
    }


# -------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------
def _softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def fit_powerlaw(X_tr, y_tr):
    """Original PowerLaw: y = sum_i (alpha_i + beta_i @ x)^(-gamma_i) + c"""
    n_feat = X_tr.shape[1]
    n_terms = 2
    rng = np.random.default_rng(42)

    def model(X, params):
        result = np.full(len(X), params[-1])
        for i in range(n_terms):
            b = i * (n_feat + 2)
            raw = params[b] + X @ params[b + 1 : b + 1 + n_feat]
            inner = _softplus(raw) + 0.1
            result += inner ** (-params[b + 1 + n_feat])
        return result

    def loss(params):
        return float(np.sum((model(X_tr, params) - y_tr) ** 2))

    best_loss, best_params = np.inf, None
    for _ in range(200):
        p0 = []
        for _ in range(n_terms):
            p0.extend([rng.uniform(0.5, 3), *rng.uniform(-2, 2, n_feat), rng.uniform(0.1, 1.5)])
        p0.append(y_tr.mean() + rng.normal(0, 0.1))
        bounds = []
        for _ in range(n_terms):
            bounds += [(None, None)] + [(None, None)] * n_feat + [(0.01, 3.0)]
        bounds.append((None, None))
        try:
            res = minimize(
                loss, np.array(p0), method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000, "ftol": 1e-12}
            )
            if res.fun < best_loss:
                best_loss, best_params = res.fun, res.x
        except Exception:
            continue

    return lambda X: model(X, best_params)


def fit_linear(X_tr, y_tr):
    X_aug = np.column_stack([np.ones(len(X_tr)), X_tr])
    coef, _, _, _ = np.linalg.lstsq(X_aug, y_tr, rcond=None)
    return lambda X: np.column_stack([np.ones(len(X)), X]) @ coef


def fit_quadratic(X_tr, y_tr):
    n_feat = X_tr.shape[1]

    def _build(X):
        parts = [np.ones((len(X), 1)), X]
        for i in range(n_feat):
            parts.append(X[:, i : i + 1] ** 2)
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                parts.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.hstack(parts)

    X_aug = _build(X_tr)
    coef, _, _, _ = np.linalg.lstsq(X_aug, y_tr, rcond=None)
    return lambda X: _build(X) @ coef


def fit_gp_matern(X_tr, y_tr):
    scaler = StandardScaler().fit(X_tr)
    X_s = scaler.transform(X_tr)
    k = ConstantKernel(1.0) * Matern(length_scale=np.ones(X_tr.shape[1]), nu=2.5) + WhiteKernel(0.001)
    gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=10, normalize_y=True)
    gp.fit(X_s, y_tr)
    return lambda X: gp.predict(scaler.transform(X)), gp, scaler


def fit_gp_rbf(X_tr, y_tr):
    scaler = StandardScaler().fit(X_tr)
    X_s = scaler.transform(X_tr)
    k = ConstantKernel(1.0) * RBF(length_scale=np.ones(X_tr.shape[1])) + WhiteKernel(0.001)
    gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=10, normalize_y=True)
    gp.fit(X_s, y_tr)
    return lambda X: gp.predict(scaler.transform(X)), gp, scaler


# -------------------------------------------------------------------------
# Cross-validation with multiple metrics
# -------------------------------------------------------------------------
def cv_metrics(fit_fn, X, y, n_folds=5, seed=42):
    """Return dict of metrics: R², RMSE, Spearman, RMSE_bottom (bottom-half only)."""
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    r2s, rmses, spearmans, rmse_bottoms = [], [], [], []
    median_y = np.median(y)

    for train_idx, test_idx in kf.split(X):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        result = fit_fn(X_tr, y_tr)
        pred_fn = result[0] if isinstance(result, tuple) else result
        pred = pred_fn(X_te)

        # R²
        ss_res = np.sum((y_te - pred) ** 2)
        ss_tot = np.sum((y_te - y_te.mean()) ** 2)
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)

        # RMSE
        rmses.append(np.sqrt(np.mean((y_te - pred) ** 2)))

        # Spearman
        if len(y_te) > 2:
            sp, _ = spearmanr(y_te, pred)
            spearmans.append(sp if not np.isnan(sp) else 0.0)

        # RMSE on bottom half (y < median) — the region that matters for optimization
        bottom_mask = y_te < median_y
        if bottom_mask.sum() > 1:
            rmse_bottoms.append(np.sqrt(np.mean((y_te[bottom_mask] - pred[bottom_mask]) ** 2)))

    return {
        "R²": np.mean(r2s),
        "R²_std": np.std(r2s),
        "RMSE": np.mean(rmses),
        "Spearman": np.mean(spearmans) if spearmans else 0.0,
        "RMSE_bottom": np.mean(rmse_bottoms) if rmse_bottoms else np.nan,
    }


# -------------------------------------------------------------------------
# Run all combinations
# -------------------------------------------------------------------------
feature_sets = get_features(df)

# Models: name -> fit function
# fit_fn returns either pred_fn or (pred_fn, gp_model, scaler)
model_defs = {
    "Linear": fit_linear,
    "Quadratic": fit_quadratic,
    "PowerLaw": fit_powerlaw,
    "GP(Matern)": lambda X, y: fit_gp_matern(X, y),
    "GP(RBF)": lambda X, y: fit_gp_rbf(X, y),
}

all_results = {}

print("\n" + "=" * 100)
print(f"{'Features':<25} {'Model':<14} {'R²':>7} {'±':>5}  {'RMSE':>7} {'Spearman':>9} {'RMSE_bot':>9}")
print("=" * 100)

for feat_name, X_feat in feature_sets.items():
    for model_name, fit_fn in model_defs.items():
        # Skip PowerLaw for >2 features (too slow, many params)
        if model_name == "PowerLaw" and X_feat.shape[1] > 2:
            continue

        metrics = cv_metrics(fit_fn, X_feat, y)
        all_results[(feat_name, model_name)] = metrics

        print(
            f"{feat_name:<25} {model_name:<14} {metrics['R²']:>7.4f} {metrics['R²_std']:>5.3f}  "
            f"{metrics['RMSE']:>7.4f} {metrics['Spearman']:>9.4f} {metrics['RMSE_bottom']:>9.4f}"
        )


# -------------------------------------------------------------------------
# Summary: sort by RMSE_bottom (what matters for finding the optimum)
# -------------------------------------------------------------------------
print("\n" + "=" * 100)
print("RANKED BY RMSE_bottom (prediction error in the low-BPB region)")
print("=" * 100)
sorted_by_bottom = sorted(all_results.items(), key=lambda x: x[1]["RMSE_bottom"])
print(f"{'Features':<25} {'Model':<14} {'RMSE_bot':>9} {'Spearman':>9} {'R²':>7} {'RMSE':>7}")
print("-" * 80)
for (feat, model), m in sorted_by_bottom[:12]:
    print(f"{feat:<25} {model:<14} {m['RMSE_bottom']:>9.4f} {m['Spearman']:>9.4f} {m['R²']:>7.4f} {m['RMSE']:>7.4f}")


# -------------------------------------------------------------------------
# Fit best models on full data and plot
# -------------------------------------------------------------------------
print("\n\nFitting on full data for visualization...")

X_weights = feature_sets["weights"]
X_mixed = feature_sets["mixed"]
X_mixed_both = feature_sets["mixed_both"]

pred_pl_w = fit_powerlaw(X_weights, y)
pred_quad_w = fit_quadratic(X_weights, y)
pred_gp_w, _, _ = fit_gp_matern(X_weights, y)
(pred_quad_mixed,) = [fit_quadratic(X_mixed, y)]
pred_quad_mixed_both = fit_quadratic(X_mixed_both, y)
pred_gp_mixed, _, _ = fit_gp_matern(X_mixed, y)
pred_gp_mixed_both, _, _ = fit_gp_matern(X_mixed_both, y)

# Slice at phase_0 = 100% nemotron
mask = df["phase_0_nemotron_full"].round(4) == 1.0
df_slice = df[mask].sort_values("phase_1_starcoder")
x_actual = df_slice["phase_1_starcoder"].values
y_actual = df_slice[TARGET].values

p1_grid = np.linspace(0.001, 0.999, 300)  # avoid exact 0/1 for log
X_w_grid = np.column_stack([np.zeros(300), p1_grid])
X_mixed_grid = np.column_stack(
    [
        np.zeros(300),
        p1_grid,
        np.log(SC_EPOCH_MULT * p1_grid + EPS),
    ]
)
X_mixed_both_grid = np.column_stack(
    [
        np.zeros(300),
        p1_grid,
        np.full(300, np.log(EPS)),
        np.log(SC_EPOCH_MULT * p1_grid + EPS),
    ]
)

# Models to plot
plot_models = [
    ("PowerLaw (weights)", pred_pl_w, X_w_grid, "tab:blue", "-"),
    ("Quad (weights)", pred_quad_w, X_w_grid, "tab:cyan", "--"),
    ("GP(Matern) weights", pred_gp_w, X_w_grid, "tab:gray", ":"),
    ("Quad (mixed)", pred_quad_mixed, X_mixed_grid, "tab:green", "-"),
    ("GP(Matern) mixed", pred_gp_mixed, X_mixed_grid, "tab:orange", "-"),
    ("Quad (mixed_both)", pred_quad_mixed_both, X_mixed_both_grid, "tab:red", "--"),
    ("GP(Matern) mixed_both", pred_gp_mixed_both, X_mixed_both_grid, "tab:purple", ":"),
]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: full range
ax = axes[0]
ax.scatter(x_actual, y_actual, s=30, c="black", zorder=10, label="Actual data")
for name, pred_fn, grid, color, ls in plot_models:
    ax.plot(p1_grid, pred_fn(grid), label=name, linewidth=1.5, color=color, linestyle=ls)
ax.set_xlabel("phase_1_starcoder")
ax.set_ylabel(TARGET)
ax.set_title("Slice: phase_0 = 100% nemotron — full range")
ax.legend(fontsize=7, loc="upper right")
ax.set_ylim(0.85, 1.75)

# Right: zoomed minimum region
ax = axes[1]
ax.scatter(x_actual, y_actual, s=30, c="black", zorder=10, label="Actual data")
for name, pred_fn, grid, color, ls in plot_models:
    preds = pred_fn(grid)
    ax.plot(p1_grid, preds, label=name, linewidth=1.5, color=color, linestyle=ls)
ax.set_xlabel("phase_1_starcoder")
ax.set_ylabel(TARGET)
ax.set_title("Zoomed: minimum region")
ax.legend(fontsize=7, loc="upper left")
ax.set_xlim(0.1, 0.55)
ax.set_ylim(0.88, 0.97)

fig.tight_layout()
out_path = script_dir / "debug_epoch_features_v2.png"
fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")

# -------------------------------------------------------------------------
# Optimal predictions
# -------------------------------------------------------------------------
print("\nPredicted optimal p1_starcoder (at phase_0=100% nemotron):")
for name, pred_fn, grid, _, _ in plot_models:
    preds = pred_fn(grid)
    best_idx = np.argmin(preds)
    sc_ep = SC_EPOCH_MULT * p1_grid[best_idx]
    print(f"  {name:25s}: p1_sc={p1_grid[best_idx]:.4f} (sc_ep={sc_ep:.2f}), pred={preds[best_idx]:.4f}")
print(f"  {'Actual best':25s}: p1_sc={x_actual[np.argmin(y_actual)]:.4f}, bpb={y_actual.min():.4f}")

# Full 2D search
print("\nFull 2D optimal (1M samples):")
rng = np.random.default_rng(123)
X_rand_w = np.column_stack([rng.uniform(0, 1, 500_000), rng.uniform(0, 1, 500_000)])
X_rand_mixed = np.column_stack(
    [
        X_rand_w[:, 0],
        X_rand_w[:, 1],
        np.log(SC_EPOCH_MULT * X_rand_w[:, 1] + EPS),
    ]
)
X_rand_mixed_both = np.column_stack(
    [
        X_rand_w[:, 0],
        X_rand_w[:, 1],
        np.log(SC_EPOCH_MULT * X_rand_w[:, 0] + EPS),
        np.log(SC_EPOCH_MULT * X_rand_w[:, 1] + EPS),
    ]
)
k = 128

for name, pred_fn, X_rand in [
    ("PowerLaw (weights)", pred_pl_w, X_rand_w),
    ("Quad (weights)", pred_quad_w, X_rand_w),
    ("GP(Matern) weights", pred_gp_w, X_rand_w),
    ("Quad (mixed)", pred_quad_mixed, X_rand_mixed),
    ("GP(Matern) mixed", pred_gp_mixed, X_rand_mixed),
    ("Quad (mixed_both)", pred_quad_mixed_both, X_rand_mixed_both),
    ("GP(Matern) mixed_both", pred_gp_mixed_both, X_rand_mixed_both),
]:
    preds = pred_fn(X_rand)
    top_k = np.argsort(preds)[:k]
    opt = np.mean(X_rand_w[top_k], axis=0)  # always report in weight space
    opt_pred = np.mean(preds[top_k])
    sc_ep_total = SC_EPOCH_MULT * (opt[0] + opt[1])
    print(f"  {name:25s}: p0_sc={opt[0]:.4f}, p1_sc={opt[1]:.4f} (sc_total_ep={sc_ep_total:.2f}), pred={opt_pred:.4f}")

best_idx = np.argmin(y)
X_all = feature_sets["weights"]
print(f"  {'Actual best':25s}: p0_sc={X_all[best_idx, 0]:.4f}, p1_sc={X_all[best_idx, 1]:.4f}, bpb={y[best_idx]:.4f}")
