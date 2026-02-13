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
"""v3: Parametric models only, with explicit functional forms in legends.

Based on v2 and literature_scaling_laws.py findings.
All models return fitted parameters; legends show the explicit
functional form on the p0_sc=0 slice with actual numeric values.
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

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

plt.rcParams["text.usetex"] = False
script_dir = Path(__file__).parent
df = pd.read_csv(script_dir / "two_phase_starcoder.csv")
df = df[df["status"] == "completed"].copy()

TARGET = "eval/paloma/dolma_100_programing_languages/bpb"
y = df[TARGET].values
N = len(df)

SC_EPOCH_MULT = 13.2289
NEM_EPOCH_MULT = 0.5

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
X_mixed_both = np.column_stack([p0_sc, p1_sc, np.log(sc_ep0 + EPS), np.log(sc_ep1 + EPS)])


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
    """y = c₀ + c₁·x₀ + c₂·x₁
    Features: [p0_sc, p1_sc].  Params: [c₀, c₁, c₂]
    """
    X_aug = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    return lambda Xn: np.column_stack([np.ones(len(Xn)), Xn]) @ coef, coef


def fit_quadratic(X, y):
    """y = c₀ + c₁x₀ + c₂x₁ + c₃x₀² + c₄x₁² + c₅x₀x₁
    Features: [p0_sc, p1_sc].  Params: [c₀..c₅]
    """

    def _build(X):
        x0, x1 = X[:, 0], X[:, 1]
        return np.column_stack([np.ones(len(X)), x0, x1, x0**2, x1**2, x0 * x1])

    coef, _, _, _ = np.linalg.lstsq(_build(X), y, rcond=None)
    return lambda Xn: _build(Xn) @ coef, coef


def fit_quadratic_4d(X, y):
    """y = c₀ + Σcᵢxᵢ + Σcᵢᵢxᵢ² + Σcᵢⱼxᵢxⱼ
    Features: [p0_sc, p1_sc, log_sc_ep0, log_sc_ep1].  15 params.
    """

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


def fit_powerlaw(X, y, n_restarts=50, seed=42):
    """y = Σᵢ softplus(αᵢ + βᵢ@x)^(−γᵢ) + c
    Features: [p0_sc, p1_sc].  2 terms.
    Params: [α₀, β₀₀, β₀₁, γ₀, α₁, β₁₀, β₁₁, γ₁, c]
    """
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
    """DML M1 (Sum-Exp): y = c + Σⱼ kⱼ·exp(tⱼ·rⱼ)
    Features: [r_n0, r_s0, r_n1, r_s1] (virtual domain proportions).
    Params: [c, k₀, t₀, k₁, t₁, k₂, t₂, k₃, t₃]
    """
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
    """SLODM: y = E + 1/Σ(Cⱼ·rⱼ^γⱼ)
    Features: [r_n0, r_s0, r_n1, r_s1].
    Params: [E, logC₀, logγ₀, ..., logC₃, logγ₃]  (C=exp(logC), γ=exp(logγ))
    """
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
    """BiMix: y = Σⱼ Aⱼ/(rⱼ+ε)^αⱼ + C
    Features: [r_n0, r_s0, r_n1, r_s1].
    Params: [C, logA₀, logα₀, ..., logA₃, logα₃]  (A=exp(logA), α=exp(logα))
    """
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


# =========================================================================
# Slice label constructors — show functional form on p0_sc=0 slice
# =========================================================================
def _s(v):
    """Format a number: short representation."""
    if abs(v) < 0.001:
        return f"{v:.1e}"
    return f"{v:.4f}"


def label_linear(params):
    # params = [c0, c1, c2]
    # On slice: y = c0 + c2*p1
    c0, c1, c2 = params
    return f"Linear: y = {_s(c0)} + {_s(c2)}·p₁"


def label_quadratic(params):
    # params = [c0, c1, c2, c3, c4, c5]
    # On slice: y = c0 + c2*p1 + c4*p1^2
    c0, _, c2, _, c4, _ = params
    return f"Quad: y = {_s(c0)} {c2:+.4f}·p₁ {c4:+.4f}·p₁²"


def label_quadratic_4d(params):
    # 15 params: [c0, c1..c4, c11..c44, c12,c13,c14,c23,c24,c34]
    # Features: [p0_sc, p1_sc, log_sc_ep0, log_sc_ep1]
    # On slice p0_sc=0: p0_sc=0, log_sc_ep0=log(EPS)≈-18.4
    # x0=0, x1=p1, x2=log(eps), x3=log(SC_EPOCH_MULT*p1+eps)
    # Simplify: terms with x0 vanish. x2=const → absorbed into intercept.
    c = params
    log_eps = np.log(EPS)
    # const = c0 + c3*log_eps + c7*log_eps^2 + cross terms with x2
    const = c[0] + c[3] * log_eps + c[7] * log_eps**2
    # p1 terms: c2 + c13_cross*log_eps (c13 is at index 11: pairs (0,1)=9, (0,2)=10, (0,3)=11, (1,2)=12)
    b_p1 = c[2] + c[12] * log_eps  # c2*p1 + c_12_cross*p1*log_eps
    # p1^2 term
    b_p1sq = c[6]
    # log_sc_ep1 terms: c4 + c_23_cross*log_eps (pair (2,3) = index 14)
    b_lep1 = c[4] + c[14] * log_eps
    # log_sc_ep1^2
    b_lep1sq = c[8]
    # p1*log_sc_ep1 cross: pair (1,3) = index 13
    b_cross = c[13]
    return (
        f"Q4D: {_s(const)} {b_p1:+.3f}p1 {b_p1sq:+.3f}p1^2"
        f" {b_lep1:+.3f}L1 {b_lep1sq:+.3f}L1^2 {b_cross:+.3f}p1*L1"
        f"  [L1=log(sc_ep1)]"
    )


def label_powerlaw(params):
    # params = [α0, β00, β01, γ0, α1, β10, β11, γ1, c]
    # On slice (p0=0): y = sp(α0+β01·p1)^(-γ0) + sp(α1+β11·p1)^(-γ1) + c
    a0, _, b01, g0 = params[0], params[1], params[2], params[3]
    a1, _, b11, g1 = params[4], params[5], params[6], params[7]
    c = params[8]
    return f"PL: sp({_s(a0)}{b01:+.2f}·p₁)^(−{g0:.2f}) + sp({_s(a1)}{b11:+.2f}·p₁)^(−{g1:.2f}) + {_s(c)}"


def label_dml_m1(params):
    # params = [c, k0, t0, k1, t1, k2, t2, k3, t3]
    # Features: [r_n0, r_s0, r_n1, r_s1]
    # On slice (p0=0): r_n0=0.5, r_s0=0, r_n1=(1-p1)/2, r_s1=p1/2
    # y = [c + k0*e^(t0*0.5) + k1*e^0] + k2*e^(t2*(1-p1)/2) + k3*e^(t3*p1/2)
    c = params[0]
    k0, t0 = params[1], params[2]
    k1, t1 = params[3], params[4]
    k2, t2 = params[5], params[6]
    k3, t3 = params[7], params[8]
    const = c + k0 * np.exp(np.clip(t0 * 0.5, -20, 20)) + k1
    return f"M1: {_s(const)} {k2:+.4f}·e^({t2:.1f}·(1−p₁)/2) {k3:+.4f}·e^({t3:.1f}·p₁/2)"


def label_slodm(params):
    # params = [E, lC0, lg0, lC1, lg1, lC2, lg2, lC3, lg3]
    # On slice: r_n0=0.5, r_s0=0, r_n1=(1-p1)/2, r_s1=p1/2
    # r_s0=0 → C1*0^g1 ≈ 0, so that term drops
    E = params[0]
    C0, g0 = np.exp(params[1]), np.exp(params[2])
    C2, g2 = np.exp(params[5]), np.exp(params[6])
    C3, g3 = np.exp(params[7]), np.exp(params[8])
    c0_val = C0 * 0.5**g0  # constant from r_n0=0.5
    return f"SLODM: {_s(E)} + 1/({_s(c0_val)} + {_s(C2)}·((1−p₁)/2)^{g2:.2f} + {_s(C3)}·(p₁/2)^{g3:.2f})"


def label_bimix(params):
    # params = [C, lA0, la0, lA1, la1, lA2, la2, lA3, la3]
    # On slice: r_n0=0.5, r_s0=0, r_n1=(1-p1)/2, r_s1=p1/2
    C = params[0]
    A0, a0 = np.exp(params[1]), np.exp(params[2])
    A1, a1 = np.exp(params[3]), np.exp(params[4])
    A2, a2 = np.exp(params[5]), np.exp(params[6])
    A3, a3 = np.exp(params[7]), np.exp(params[8])
    # r_s0=0 → A1/(0+eps)^a1 is a large constant
    c_n0 = A0 / (0.5 + 1e-3) ** a0
    c_s0 = A1 / (0 + 1e-3) ** a1
    const = C + c_n0 + c_s0
    return f"BiMix: {_s(const)} + {_s(A2)}/((1−p₁)/2+ε)^{a2:.2f} + {_s(A3)}/(p₁/2+ε)^{a3:.2f}"


# =========================================================================
# Define all models to test
# =========================================================================
# (name, fit_fn, X_data, label_fn, color, linestyle)
MODELS = [
    ("Linear", fit_linear, X_weight, label_linear, "tab:blue", "--"),
    ("Quadratic", fit_quadratic, X_weight, label_quadratic, "tab:cyan", "--"),
    ("Quad(mix)", fit_quadratic_4d, X_mixed_both, label_quadratic_4d, "tab:orange", "--"),
    ("PowerLaw", fit_powerlaw, X_weight, label_powerlaw, "black", "-"),
    ("DML_M1", fit_dml_m1, X_vdom, label_dml_m1, "tab:green", "-"),
    ("SLODM", fit_slodm, X_vdom, label_slodm, "tab:red", "-"),
    ("BiMix", fit_bimix, X_vdom, label_bimix, "tab:purple", "-"),
]


# =========================================================================
# Cross-validation
# =========================================================================
print("\n" + "=" * 80)
print(f"{'Model':<14} {'R²':>7} {'RMSE':>7} {'Spearman':>9} {'RMSE_bot':>9}")
print("=" * 80)

cv_results = {}
for name, fit_fn, X_data, _, _, _ in MODELS:
    m = cv_metrics(fit_fn, X_data, y)
    cv_results[name] = m
    print(f"{name:<14} {m['R2']:>7.4f} {m['RMSE']:>7.4f} {m['Spearman']:>9.4f} {m['RMSE_bot']:>9.4f}")


# =========================================================================
# Fit on full data, extract params, build labels
# =========================================================================
print("\n\nFitting all models on full data...")

slice_grid = np.linspace(0.002, 0.998, 300)


# Slice features for each feature type
def make_slice_weight(g):
    return np.column_stack([np.zeros(len(g)), g])


def make_slice_vdom(g):
    n = len(g)
    return np.column_stack([np.full(n, 0.5), np.zeros(n), 0.5 * (1 - g), 0.5 * g])


def make_slice_mixed_both(g):
    n = len(g)
    return np.column_stack([np.zeros(n), g, np.full(n, np.log(EPS)), np.log(SC_EPOCH_MULT * g + EPS)])


fitted = []  # (name, pred_fn, params, label_str, slice_preds, color, ls, cv_m)

for name, fit_fn, X_data, label_fn, color, ls in MODELS:
    pred_fn, params = fit_fn(X_data, y)

    # Construct slice features
    if X_data is X_weight:
        Xs = make_slice_weight(slice_grid)
    elif X_data is X_mixed_both:
        Xs = make_slice_mixed_both(slice_grid)
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


# =========================================================================
# Actual data on the slice
# =========================================================================
mask = df["phase_0_nemotron_full"].round(4) == 1.0
df_slice = df[mask].sort_values("phase_1_starcoder")
x_actual = df_slice["phase_1_starcoder"].values
y_actual = df_slice[TARGET].values
actual_best_i = np.argmin(y_actual)

print(f"\nActual best (slice): p1_sc={x_actual[actual_best_i]:.4f}, bpb={y_actual[actual_best_i]:.4f}")


# =========================================================================
# Plot
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 7))

for panel, (ax, xlim, ylim, title) in enumerate(
    zip(
        axes,
        [(0, 1), (0.1, 0.55)],
        [(0.85, 1.75), (0.88, 0.97)],
        ["Full range", "Zoomed: minimum region"],
    )
):
    ax.scatter(x_actual, y_actual, s=40, c="black", zorder=10, label="Actual data")

    for name, _, params, label, preds_s, color, ls, cv_m in fitted:
        if panel == 0:
            lbl = name
        else:
            lbl = f"{label}\n  RMSE_bot={cv_m['RMSE_bot']:.4f}"
        ax.plot(slice_grid, preds_s, label=lbl, linewidth=1.8, color=color, linestyle=ls)

    ax.set_xlabel("phase_1_starcoder", fontsize=11)
    ax.set_ylabel("dolma_100_programing_languages/bpb", fontsize=11)
    ax.set_title(f"Slice: phase_0 = 100% nemotron — {title}", fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if panel == 0:
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)

fig.tight_layout()
out_path = script_dir / "debug_epoch_features_v3.png"
fig.savefig(out_path, dpi=150)
print(f"\nSaved {out_path}")


# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 80)
print("SUMMARY — Ranked by RMSE_bot")
print("=" * 80)
print(f"{'Model':<14} {'RMSE_bot':>9} {'Spearman':>9} {'R²':>7}  {'Slice opt':>10} {'Pred':>7}")
print("-" * 65)
for name, _, params, label, preds_s, _, _, cv_m in sorted(fitted, key=lambda x: x[7]["RMSE_bot"]):
    best_i = np.argmin(preds_s)
    print(
        f"{name:<14} {cv_m['RMSE_bot']:>9.4f} {cv_m['Spearman']:>9.4f} {cv_m['R2']:>7.4f}  "
        f"p1={slice_grid[best_i]:>6.4f} {preds_s[best_i]:>7.4f}"
    )
print(f"{'Actual':14s} {'':>9} {'':>9} {'':>7}  p1={x_actual[actual_best_i]:>6.4f} {y_actual[actual_best_i]:>7.4f}")
