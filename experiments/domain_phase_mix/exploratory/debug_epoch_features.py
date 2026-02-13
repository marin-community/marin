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
"""Test whether epoch-based features improve regression for code BPB.

Hypothesis: The U-shape in code BPB vs starcoder weight is driven by epoching.
- Too few starcoder epochs → model hasn't seen enough code
- Too many starcoder epochs → diminishing returns + lost general knowledge
- Scaling laws suggest L(E) ~ E^(-alpha), so log(epochs) should be a good feature

Feature sets tested:
1. weights:       [p0_sc, p1_sc] — baseline (same as before)
2. epochs:        [sc_ep0, sc_ep1, nem_ep0, nem_ep1] — full epoch features
3. log_epochs:    [log(sc_ep0+eps), log(sc_ep1+eps), ...] — log-transformed
4. log_sc_epochs: [log(sc_ep0+eps), log(sc_ep1+eps)] — starcoder-only log epochs
5. mixed:         [p0_sc, p1_sc, log(sc_ep1+eps)] — weight + log epoch hybrid

Models tested for each feature set:
- Linear, Quadratic, GP(Matern), GP(RBF)
- Epoch-power-law: y = A * (sc_total_ep + a)^(-b) + C (1D, fit on total sc epochs)
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern,
    RBF,
    WhiteKernel,
    ConstantKernel,
    RationalQuadratic,
)
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

script_dir = Path(__file__).parent
df = pd.read_csv(script_dir / "two_phase_starcoder.csv")
df = df[df["status"] == "completed"].copy()

TARGET = "eval/paloma/dolma_100_programing_languages/bpb"
y = df[TARGET].values
N = len(df)

# Epoching constants (from add_epoch_columns.py)
SC_EPOCH_MULT = 13.2289  # epochs per unit weight per phase
NEM_EPOCH_MULT = 0.5
EPS = 0.01  # small constant to avoid log(0)

print(f"N={N}, target={TARGET}")
print(f"y range: [{y.min():.4f}, {y.max():.4f}]")


# -------------------------------------------------------------------------
# Feature sets
# -------------------------------------------------------------------------
def make_features(df):
    """Build all feature sets."""
    p0_sc = df["phase_0_starcoder"].values
    p1_sc = df["phase_1_starcoder"].values

    sc_ep0 = df["phase_0_starcoder_epochs"].values
    sc_ep1 = df["phase_1_starcoder_epochs"].values
    nem_ep0 = df["phase_0_nemotron_epochs"].values
    nem_ep1 = df["phase_1_nemotron_epochs"].values
    sc_total = df["total_starcoder_epochs"].values

    feature_sets = {
        "weights [p0_sc, p1_sc]": np.column_stack([p0_sc, p1_sc]),
        "epochs [sc0, sc1, nem0, nem1]": np.column_stack([sc_ep0, sc_ep1, nem_ep0, nem_ep1]),
        "log_epochs [all 4]": np.column_stack(
            [
                np.log(sc_ep0 + EPS),
                np.log(sc_ep1 + EPS),
                np.log(nem_ep0 + EPS),
                np.log(nem_ep1 + EPS),
            ]
        ),
        "log_sc_epochs [sc0, sc1]": np.column_stack(
            [
                np.log(sc_ep0 + EPS),
                np.log(sc_ep1 + EPS),
            ]
        ),
        "log_sc_total [1D]": np.log(sc_total + EPS).reshape(-1, 1),
        "mixed [p0_sc, p1_sc, log_sc1]": np.column_stack(
            [
                p0_sc,
                p1_sc,
                np.log(sc_ep1 + EPS),
            ]
        ),
    }
    return feature_sets


feature_sets = make_features(df)


# -------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------
def fit_linear(X_tr, y_tr):
    X_aug = np.column_stack([np.ones(len(X_tr)), X_tr])
    coef, _, _, _ = np.linalg.lstsq(X_aug, y_tr, rcond=None)
    return lambda X: np.column_stack([np.ones(len(X)), X]) @ coef


def fit_quadratic(X_tr, y_tr):
    n_feat = X_tr.shape[1]
    # Build quadratic features: 1 + linear + squared + cross terms
    parts = [np.ones((len(X_tr), 1)), X_tr]
    for i in range(n_feat):
        parts.append(X_tr[:, i : i + 1] ** 2)
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            parts.append((X_tr[:, i] * X_tr[:, j]).reshape(-1, 1))
    X_aug = np.hstack(parts)
    coef, _, _, _ = np.linalg.lstsq(X_aug, y_tr, rcond=None)

    def predict(X):
        p = [np.ones((len(X), 1)), X]
        for i in range(n_feat):
            p.append(X[:, i : i + 1] ** 2)
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                p.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.hstack(p) @ coef

    return predict


def fit_gp_matern(X_tr, y_tr):
    scaler = StandardScaler().fit(X_tr)
    X_s = scaler.transform(X_tr)
    k = ConstantKernel(1.0) * Matern(length_scale=np.ones(X_tr.shape[1]), nu=2.5) + WhiteKernel(0.001)
    gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=10, normalize_y=True)
    gp.fit(X_s, y_tr)
    return lambda X: gp.predict(scaler.transform(X))


def fit_gp_rbf(X_tr, y_tr):
    scaler = StandardScaler().fit(X_tr)
    X_s = scaler.transform(X_tr)
    k = ConstantKernel(1.0) * RBF(length_scale=np.ones(X_tr.shape[1])) + WhiteKernel(0.001)
    gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=10, normalize_y=True)
    gp.fit(X_s, y_tr)
    return lambda X: gp.predict(scaler.transform(X))


def fit_gp_rq(X_tr, y_tr):
    """GP with Rational Quadratic kernel — handles multiple length scales."""
    scaler = StandardScaler().fit(X_tr)
    X_s = scaler.transform(X_tr)
    k = ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(0.001)
    gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=10, normalize_y=True)
    gp.fit(X_s, y_tr)
    return lambda X: gp.predict(scaler.transform(X))


# -------------------------------------------------------------------------
# Epoch power law: y = A * (sc_total_epochs + a)^(-b) + C
# Directly motivated by scaling law L(E) ~ E^(-alpha) + L_inf
# -------------------------------------------------------------------------
def fit_epoch_powerlaw(X_tr, y_tr):
    """X_tr must be 1D: total_starcoder_epochs or log thereof."""
    sc_total = np.exp(X_tr[:, 0]) - EPS if X_tr.shape[1] == 1 else X_tr[:, 0]

    def model(sc, params):
        A, a, b, C = params
        return A * (sc + np.abs(a) + 0.01) ** (-np.abs(b)) + C

    def loss(params):
        pred = model(sc_total, params)
        return float(np.sum((pred - y_tr) ** 2))

    rng = np.random.default_rng(42)
    best_loss, best_params = np.inf, None
    for _ in range(500):
        p0 = [rng.uniform(0.1, 5), rng.uniform(0.01, 2), rng.uniform(0.1, 2), y_tr.mean() + rng.normal(0, 0.1)]
        try:
            res = minimize(loss, p0, method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-12})
            if res.fun < best_loss:
                best_loss, best_params = res.fun, res.x
        except Exception:
            continue

    A, a, b, C = best_params
    print(f"    EpochPowerLaw params: A={A:.4f}, a={np.abs(a):.4f}, b={np.abs(b):.4f}, C={C:.4f}")

    def predict(X):
        sc = np.exp(X[:, 0]) - EPS if X.shape[1] == 1 else X[:, 0]
        return model(sc, best_params)

    return predict


# -------------------------------------------------------------------------
# Two-term epoch power law: separate terms for starcoder and nemotron
# y = A_sc * (sc_total + a_sc)^(-b_sc) + A_nem * (nem_total + a_nem)^(-b_nem) + C
# -------------------------------------------------------------------------
def fit_epoch_powerlaw_2term(X_tr, y_tr):
    """X_tr has 2 columns: [total_starcoder_epochs, total_nemotron_epochs]."""
    sc = X_tr[:, 0]
    nem = X_tr[:, 1]

    def model(sc, nem, params):
        A_sc, a_sc, b_sc, A_nem, a_nem, b_nem, C = params
        return (
            A_sc * (sc + np.abs(a_sc) + 0.01) ** (-np.abs(b_sc))
            + A_nem * (nem + np.abs(a_nem) + 0.01) ** (-np.abs(b_nem))
            + C
        )

    def loss(params):
        pred = model(sc, nem, params)
        return float(np.sum((pred - y_tr) ** 2))

    rng = np.random.default_rng(42)
    best_loss, best_params = np.inf, None
    for _ in range(500):
        p0 = [
            rng.uniform(0.1, 3),
            rng.uniform(0.01, 1),
            rng.uniform(0.1, 1.5),
            rng.uniform(0.1, 3),
            rng.uniform(0.01, 1),
            rng.uniform(0.1, 1.5),
            y_tr.mean() + rng.normal(0, 0.1),
        ]
        try:
            res = minimize(loss, p0, method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-12})
            if res.fun < best_loss:
                best_loss, best_params = res.fun, res.x
        except Exception:
            continue

    A_sc, a_sc, b_sc, A_nem, a_nem, b_nem, C = best_params
    print(
        f"    2TermPL params: A_sc={A_sc:.4f}, a_sc={np.abs(a_sc):.4f}, b_sc={np.abs(b_sc):.4f}, "
        f"A_nem={A_nem:.4f}, a_nem={np.abs(a_nem):.4f}, b_nem={np.abs(b_nem):.4f}, C={C:.4f}"
    )

    def predict(X):
        return model(X[:, 0], X[:, 1], best_params)

    return predict


# -------------------------------------------------------------------------
# Cross-validation
# -------------------------------------------------------------------------
def cv_r2(fit_fn, X, y, n_folds=5, seed=42):
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    r2s = []
    for train_idx, test_idx in kf.split(X):
        pred_fn = fit_fn(X[train_idx], y[train_idx])
        pred = pred_fn(X[test_idx])
        ss_res = np.sum((y[test_idx] - pred) ** 2)
        ss_tot = np.sum((y[test_idx] - y[test_idx].mean()) ** 2)
        r2s.append(1 - ss_res / ss_tot)
    return np.mean(r2s), np.std(r2s)


# -------------------------------------------------------------------------
# Run all combinations
# -------------------------------------------------------------------------
print("\n" + "=" * 90)
print(f"5-fold CV R² on {TARGET}")
print("=" * 90)

# Standard models for all feature sets
standard_models = {
    "Linear": fit_linear,
    "Quadratic": fit_quadratic,
    "GP(Matern)": fit_gp_matern,
    "GP(RBF)": fit_gp_rbf,
    "GP(RatQuad)": fit_gp_rq,
}

results = {}  # (feature_set, model_name) -> (mean_r2, std_r2)

for feat_name, X_feat in feature_sets.items():
    print(f"\n--- Feature set: {feat_name} (shape: {X_feat.shape}) ---")
    for model_name, fit_fn in standard_models.items():
        # Skip quadratic for 4-feature sets (too many params)
        if model_name == "Quadratic" and X_feat.shape[1] > 3:
            n_quad_params = 1 + X_feat.shape[1] + X_feat.shape[1] + X_feat.shape[1] * (X_feat.shape[1] - 1) // 2
            if n_quad_params > N // 3:
                print(f"  {model_name:15s}: SKIP (too many params: {n_quad_params} for N={N})")
                continue
        print(f"  {model_name:15s}:", end=" ", flush=True)
        mean_r2, std_r2 = cv_r2(fit_fn, X_feat, y)
        results[(feat_name, model_name)] = (mean_r2, std_r2)
        print(f"R² = {mean_r2:.4f} ± {std_r2:.4f}")


# Special models
print("\n--- Special models ---")

# 1. Epoch power law on total starcoder epochs (1D)
sc_total = df["total_starcoder_epochs"].values.reshape(-1, 1)
print("  EpochPowerLaw [sc_total, 1D]:", end=" ", flush=True)
mean_r2, std_r2 = cv_r2(fit_epoch_powerlaw, sc_total, y)
results[("sc_total [1D]", "EpochPowerLaw")] = (mean_r2, std_r2)
print(f"R² = {mean_r2:.4f} ± {std_r2:.4f}")

# 2. Two-term epoch power law on [sc_total, nem_total]
epoch_totals = np.column_stack(
    [
        df["total_starcoder_epochs"].values,
        df["total_nemotron_epochs"].values,
    ]
)
print("  2TermEpochPL [sc_total, nem_total]:", end=" ", flush=True)
mean_r2, std_r2 = cv_r2(fit_epoch_powerlaw_2term, epoch_totals, y)
results[("epoch_totals [2D]", "2TermEpochPL")] = (mean_r2, std_r2)
print(f"R² = {mean_r2:.4f} ± {std_r2:.4f}")

# 3. Epoch power law on per-phase starcoder epochs
sc_per_phase = np.column_stack(
    [
        df["phase_0_starcoder_epochs"].values,
        df["phase_1_starcoder_epochs"].values,
    ]
)
print("  2TermEpochPL [sc_ep0, sc_ep1]:", end=" ", flush=True)
mean_r2, std_r2 = cv_r2(fit_epoch_powerlaw_2term, sc_per_phase, y)
results[("sc_per_phase [2D]", "2TermEpochPL")] = (mean_r2, std_r2)
print(f"R² = {mean_r2:.4f} ± {std_r2:.4f}")


# -------------------------------------------------------------------------
# Summary: top 10 results
# -------------------------------------------------------------------------
print("\n" + "=" * 90)
print("TOP 15 RESULTS (sorted by mean R²)")
print("=" * 90)
sorted_results = sorted(results.items(), key=lambda x: -x[1][0])
print(f"{'Features':<35} {'Model':<16} {'R²':>8}  {'±':>6}")
print("-" * 70)
for (feat, model), (mean_r2, std_r2) in sorted_results[:15]:
    print(f"{feat:<35} {model:<16} {mean_r2:>8.4f}  {std_r2:>6.4f}")


# -------------------------------------------------------------------------
# Visualization: fit on full data, plot slices
# -------------------------------------------------------------------------
print("\n\nFitting best models on full data for visualization...")

# Best models to visualize
p0_sc = df["phase_0_starcoder"].values
p1_sc = df["phase_1_starcoder"].values
X_weights = np.column_stack([p0_sc, p1_sc])
X_log_sc = np.column_stack(
    [np.log(df["phase_0_starcoder_epochs"].values + EPS), np.log(df["phase_1_starcoder_epochs"].values + EPS)]
)

# Fit on full data
pred_gp_weights = fit_gp_matern(X_weights, y)
pred_gp_log_sc = fit_gp_matern(X_log_sc, y)
pred_quad_log_sc = fit_quadratic(X_log_sc, y)
pred_gp_rq_log_sc = fit_gp_rq(X_log_sc, y)

# Also fit the epoch power law on full data
sc_total_full = df["total_starcoder_epochs"].values.reshape(-1, 1)
pred_epoch_pl = fit_epoch_powerlaw(sc_total_full, y)

# Fit 2-term power law on full data
pred_2term = fit_epoch_powerlaw_2term(epoch_totals, y)

# Slice at phase_0 = 100% nemotron (p0_sc = 0)
mask = df["phase_0_nemotron_full"].round(4) == 1.0
df_slice = df[mask].sort_values("phase_1_starcoder")
x_actual = df_slice["phase_1_starcoder"].values
y_actual = df_slice[TARGET].values

# Create prediction grid for slice
p1_grid = np.linspace(0, 1, 200)
X_w_grid = np.column_stack([np.zeros(200), p1_grid])
X_log_sc_grid = np.column_stack(
    [
        np.full(200, np.log(EPS)),  # log(0 + eps)
        np.log(SC_EPOCH_MULT * p1_grid + EPS),
    ]
)
sc_total_grid = (SC_EPOCH_MULT * p1_grid).reshape(-1, 1)
epoch_totals_grid = np.column_stack(
    [
        SC_EPOCH_MULT * p1_grid,
        NEM_EPOCH_MULT * np.ones(200),  # nemotron is 0.5 * 1.0 = 0.5 in phase_0
    ]
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: full view
ax = axes[0]
ax.scatter(x_actual, y_actual, s=30, c="black", zorder=10, label="Actual data")
ax.plot(p1_grid, pred_gp_weights(X_w_grid), label="GP(Matern) weights", linewidth=2, color="tab:blue")
ax.plot(p1_grid, pred_gp_log_sc(X_log_sc_grid), label="GP(Matern) log_sc", linewidth=2, color="tab:orange")
ax.plot(p1_grid, pred_quad_log_sc(X_log_sc_grid), label="Quad log_sc", linewidth=2, linestyle="--", color="tab:green")
ax.plot(p1_grid, pred_epoch_pl(sc_total_grid), label="EpochPowerLaw 1D", linewidth=2, linestyle=":", color="tab:red")
ax.plot(p1_grid, pred_2term(epoch_totals_grid), label="2TermEpochPL", linewidth=2, linestyle="-.", color="tab:purple")
ax.set_xlabel("phase_1_starcoder")
ax.set_ylabel(TARGET)
ax.set_title("Slice: phase_0 = 100% nemotron — full range")
ax.legend(fontsize=8)
ax.set_ylim(0.85, 1.75)

# Right: zoomed minimum region
ax = axes[1]
ax.scatter(x_actual, y_actual, s=30, c="black", zorder=10, label="Actual data")
ax.plot(p1_grid, pred_gp_weights(X_w_grid), label="GP(Matern) weights", linewidth=2, color="tab:blue")
ax.plot(p1_grid, pred_gp_log_sc(X_log_sc_grid), label="GP(Matern) log_sc", linewidth=2, color="tab:orange")
ax.plot(p1_grid, pred_quad_log_sc(X_log_sc_grid), label="Quad log_sc", linewidth=2, linestyle="--", color="tab:green")
ax.plot(p1_grid, pred_epoch_pl(sc_total_grid), label="EpochPowerLaw 1D", linewidth=2, linestyle=":", color="tab:red")
ax.plot(p1_grid, pred_2term(epoch_totals_grid), label="2TermEpochPL", linewidth=2, linestyle="-.", color="tab:purple")
ax.set_xlabel("phase_1_starcoder")
ax.set_ylabel(TARGET)
ax.set_title("Zoomed: minimum region")
ax.legend(fontsize=8)
ax.set_xlim(0.1, 0.55)
ax.set_ylim(0.88, 0.97)

fig.tight_layout()
out_path = script_dir / "debug_epoch_features.png"
fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")


# -------------------------------------------------------------------------
# Optimal predictions from best models
# -------------------------------------------------------------------------
print("\nOptimal phase_1_starcoder (at phase_0=100% nemotron):")
for name, pred_fn, grid in [
    ("GP(Matern) weights", pred_gp_weights, X_w_grid),
    ("GP(Matern) log_sc", pred_gp_log_sc, X_log_sc_grid),
    ("Quad log_sc", pred_quad_log_sc, X_log_sc_grid),
    ("EpochPL 1D", pred_epoch_pl, sc_total_grid),
    ("2TermEpochPL", pred_2term, epoch_totals_grid),
]:
    preds = pred_fn(grid)
    best_idx = np.argmin(preds)
    sc_epochs = SC_EPOCH_MULT * p1_grid[best_idx]
    print(f"  {name:25s}: p1_sc={p1_grid[best_idx]:.4f} (sc_epochs={sc_epochs:.2f}), pred={preds[best_idx]:.4f}")
print(f"  {'Actual best':25s}: p1_sc={x_actual[np.argmin(y_actual)]:.4f}, bpb={y_actual.min():.4f}")
