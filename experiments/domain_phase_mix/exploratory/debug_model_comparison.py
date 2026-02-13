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
"""Diagnose why parametric models fail on dolma_100_programing_languages/bpb.

1. Fit each model on full 2D data
2. Plot predicted vs actual on the phase_0=1.0 slice
3. Test quadratic and GP as alternatives
4. Cross-validate all models on the primary target
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

sys.stdout.reconfigure(line_buffering=True)

script_dir = Path(__file__).parent
df = pd.read_csv(script_dir / "two_phase_starcoder.csv")
df = df[df["status"] == "completed"].copy()

FEATURE_COLS = ["phase_0_starcoder", "phase_1_starcoder"]
TARGET = "eval/paloma/dolma_100_programing_languages/bpb"

X = df[FEATURE_COLS].values
y = df[TARGET].values

print(f"N={len(df)}, features={FEATURE_COLS}")
print(f"Target: {TARGET}")
print(f"y range: [{y.min():.4f}, {y.max():.4f}]")


# -------------------------------------------------------------------------
# Model definitions (compact versions)
# -------------------------------------------------------------------------
def _softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def fit_powerlaw(X, y, n_terms=2, n_restarts=200, seed=42):
    """Power law: y = sum_i (alpha_i + beta_i @ x)^(-gamma_i) + c"""
    rng = np.random.default_rng(seed)

    def model(X, params):
        result = np.full(len(X), params[-1])
        for i in range(n_terms):
            b = 4 * i
            raw = params[b] + X @ params[b + 1 : b + 3]
            inner = _softplus(raw) + 0.1
            result += inner ** (-params[b + 3])
        return result

    def loss(params):
        return float(np.sum((model(X, params) - y) ** 2))

    best_loss, best_params = np.inf, None
    for _ in range(n_restarts):
        p0 = []
        for _ in range(n_terms):
            p0.extend([rng.uniform(0.5, 3), *rng.uniform(-2, 2, 2), rng.uniform(0.1, 1.5)])
        p0.append(y.mean() + rng.normal(0, 0.1))
        bounds = []
        for _ in range(n_terms):
            bounds += [(None, None), (None, None), (None, None), (0.01, 3.0)]
        bounds.append((None, None))
        try:
            res = minimize(loss, p0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000, "ftol": 1e-12})
            if res.fun < best_loss:
                best_loss, best_params = res.fun, res.x
        except Exception:
            continue

    return lambda Xnew: model(Xnew, best_params), best_params


def fit_linear(X, y):
    """y = intercept + b @ x"""
    X_aug = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    return lambda Xnew: np.column_stack([np.ones(len(Xnew)), Xnew]) @ coef


def fit_quadratic(X, y):
    """y = intercept + b @ x + x^T A x  (quadratic in features)"""
    x0, x1 = X[:, 0], X[:, 1]
    X_aug = np.column_stack([np.ones(len(X)), x0, x1, x0**2, x1**2, x0 * x1])
    coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

    def predict(Xnew):
        x0n, x1n = Xnew[:, 0], Xnew[:, 1]
        return np.column_stack([np.ones(len(Xnew)), x0n, x1n, x0n**2, x1n**2, x0n * x1n]) @ coef

    return predict, coef


def fit_cubic(X, y):
    """y = full cubic polynomial in features"""
    x0, x1 = X[:, 0], X[:, 1]
    X_aug = np.column_stack(
        [
            np.ones(len(X)),
            x0,
            x1,
            x0**2,
            x1**2,
            x0 * x1,
            x0**3,
            x1**3,
            x0**2 * x1,
            x0 * x1**2,
        ]
    )
    coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

    def predict(Xnew):
        x0n, x1n = Xnew[:, 0], Xnew[:, 1]
        return (
            np.column_stack(
                [
                    np.ones(len(Xnew)),
                    x0n,
                    x1n,
                    x0n**2,
                    x1n**2,
                    x0n * x1n,
                    x0n**3,
                    x1n**3,
                    x0n**2 * x1n,
                    x0n * x1n**2,
                ]
            )
            @ coef
        )

    return predict, coef


def fit_gp(X, y):
    """Gaussian Process with Matern kernel."""
    kernel = ConstantKernel(1.0) * Matern(length_scale=[0.3, 0.3], nu=2.5) + WhiteKernel(0.001)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    gp.fit(X, y)
    return gp.predict, gp


# -------------------------------------------------------------------------
# Cross-validation
# -------------------------------------------------------------------------
def cv_r2(fit_fn, X, y, n_folds=5, seed=42):
    """5-fold CV, return mean R²."""
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    r2s = []
    for train_idx, test_idx in kf.split(X):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        pred_fn = fit_fn(X_tr, y_tr)
        if isinstance(pred_fn, tuple):
            pred_fn = pred_fn[0]
        pred = pred_fn(X_te)
        ss_res = np.sum((y_te - pred) ** 2)
        ss_tot = np.sum((y_te - y_te.mean()) ** 2)
        r2s.append(1 - ss_res / ss_tot)
    return np.mean(r2s), np.std(r2s)


print("\n" + "=" * 60)
print("5-fold CV R² on", TARGET)
print("=" * 60)

models = {
    "PowerLaw(200r)": lambda X, y: (fit_powerlaw(X, y, n_restarts=200)[0],),
    "Linear": lambda X, y: (fit_linear(X, y),),
    "Quadratic": lambda X, y: fit_quadratic(X, y),
    "Cubic": lambda X, y: fit_cubic(X, y),
    "GP(Matern)": lambda X, y: fit_gp(X, y),
}

cv_results = {}
for name, fit_fn in models.items():
    print(f"  {name}...", end=" ", flush=True)
    mean_r2, std_r2 = cv_r2(fit_fn, X, y)
    cv_results[name] = (mean_r2, std_r2)
    print(f"R² = {mean_r2:.4f} ± {std_r2:.4f}")


# -------------------------------------------------------------------------
# Fit on full data and plot slice at phase_0=1.0
# -------------------------------------------------------------------------
print("\nFitting all models on full data...")
pred_powerlaw, pl_params = fit_powerlaw(X, y, n_restarts=200)
pred_linear = fit_linear(X, y)
pred_quad, quad_coef = fit_quadratic(X, y)
pred_cubic, cubic_coef = fit_cubic(X, y)
pred_gp, gp_model = fit_gp(X, y)

# Print power law params for diagnosis
print(f"\nPowerLaw params: {pl_params}")
for i in range(2):
    b = 4 * i
    print(
        f"  Term {i}: alpha={pl_params[b]:.4f}, beta=[{pl_params[b+1]:.4f}, {pl_params[b+2]:.4f}], gamma={pl_params[b+3]:.4f}"
    )
print(f"  c={pl_params[-1]:.4f}")

# Quadratic coefficients
print(
    f"\nQuadratic coefs: intercept={quad_coef[0]:.4f}, b0={quad_coef[1]:.4f}, b1={quad_coef[2]:.4f}, "
    f"x0²={quad_coef[3]:.4f}, x1²={quad_coef[4]:.4f}, x0*x1={quad_coef[5]:.4f}"
)

# Slice: phase_0_starcoder = 0 (nemotron_full = 1.0)
p1_grid = np.linspace(0, 1, 200)
X_slice = np.column_stack([np.zeros(200), p1_grid])

# Actual data points on this slice
mask = df["phase_0_nemotron_full"].round(4) == 1.0
df_slice = df[mask].sort_values("phase_1_starcoder")
x_actual = df_slice["phase_1_starcoder"].values
y_actual = df_slice[TARGET].values

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: all model predictions vs data
ax = axes[0]
ax.scatter(x_actual, y_actual, s=30, c="black", zorder=10, label="Actual data")
ax.plot(p1_grid, pred_powerlaw(X_slice), label="PowerLaw", linewidth=2)
ax.plot(p1_grid, pred_linear(X_slice), label="Linear", linewidth=1.5, linestyle="--")
ax.plot(p1_grid, pred_quad(X_slice), label="Quadratic", linewidth=2)
ax.plot(p1_grid, pred_cubic(X_slice), label="Cubic", linewidth=2, linestyle="-.")
gp_mean, gp_std = gp_model.predict(X_slice, return_std=True)
ax.plot(p1_grid, gp_mean, label="GP(Matern)", linewidth=2, linestyle=":")
ax.fill_between(p1_grid, gp_mean - 2 * gp_std, gp_mean + 2 * gp_std, alpha=0.15, color="purple")
ax.set_xlabel("phase_1_starcoder")
ax.set_ylabel(TARGET)
ax.set_title("Slice: phase_0 = 100% nemotron_full")
ax.legend(fontsize=9)
ax.set_ylim(0.85, 1.75)

# Right: zoomed in on the minimum region
ax = axes[1]
ax.scatter(x_actual, y_actual, s=30, c="black", zorder=10, label="Actual data")
ax.plot(p1_grid, pred_powerlaw(X_slice), label="PowerLaw", linewidth=2)
ax.plot(p1_grid, pred_quad(X_slice), label="Quadratic", linewidth=2)
ax.plot(p1_grid, pred_cubic(X_slice), label="Cubic", linewidth=2, linestyle="-.")
ax.plot(p1_grid, gp_mean, label="GP(Matern)", linewidth=2, linestyle=":")
ax.fill_between(p1_grid, gp_mean - 2 * gp_std, gp_mean + 2 * gp_std, alpha=0.15, color="purple")
ax.set_xlabel("phase_1_starcoder")
ax.set_ylabel(TARGET)
ax.set_title("Zoomed: minimum region")
ax.legend(fontsize=9)
ax.set_xlim(0.1, 0.55)
ax.set_ylim(0.88, 0.97)

fig.tight_layout()
out_path = script_dir / "debug_model_comparison.png"
fig.savefig(out_path, dpi=150)
print(f"\nSaved {out_path}")


# -------------------------------------------------------------------------
# Find optimal p1_starcoder for each model (on slice)
# -------------------------------------------------------------------------
print("\nOptimal phase_1_starcoder (at phase_0=100% nemotron):")
for name, pred_fn in [
    ("PowerLaw", pred_powerlaw),
    ("Linear", pred_linear),
    ("Quadratic", pred_quad),
    ("Cubic", pred_cubic),
]:
    preds = pred_fn(X_slice)
    best_idx = np.argmin(preds)
    print(f"  {name:15s}: p1_sc={p1_grid[best_idx]:.4f}, pred_bpb={preds[best_idx]:.4f}")

gp_preds = gp_model.predict(X_slice)
best_idx = np.argmin(gp_preds)
print(f"  {'GP(Matern)':15s}: p1_sc={p1_grid[best_idx]:.4f}, pred_bpb={gp_preds[best_idx]:.4f}")
print(f"  {'Actual best':15s}: p1_sc={x_actual[np.argmin(y_actual)]:.4f}, bpb={y_actual.min():.4f}")


# -------------------------------------------------------------------------
# Full 2D optimal search
# -------------------------------------------------------------------------
print("\nFull 2D optimal search (1M samples):")
rng = np.random.default_rng(123)
X_rand = np.column_stack([rng.uniform(0, 1, 1_000_000), rng.uniform(0, 1, 1_000_000)])
k = 128

for name, pred_fn in [("PowerLaw", pred_powerlaw), ("Quadratic", pred_quad), ("Cubic", pred_cubic)]:
    preds = pred_fn(X_rand)
    top_k = np.argsort(preds)[:k]
    opt = np.mean(X_rand[top_k], axis=0)
    opt_pred = np.mean(preds[top_k])
    print(f"  {name:15s}: p0_sc={opt[0]:.4f}, p1_sc={opt[1]:.4f}, pred={opt_pred:.4f}")

gp_preds = gp_model.predict(X_rand)
top_k = np.argsort(gp_preds)[:k]
opt = np.mean(X_rand[top_k], axis=0)
opt_pred = np.mean(gp_preds[top_k])
print(f"  {'GP(Matern)':15s}: p0_sc={opt[0]:.4f}, p1_sc={opt[1]:.4f}, pred={opt_pred:.4f}")

best_idx = np.argmin(y)
print(f"  {'Actual best':15s}: p0_sc={X[best_idx, 0]:.4f}, p1_sc={X[best_idx, 1]:.4f}, bpb={y[best_idx]:.4f}")
