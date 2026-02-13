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
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "numpy",
#     "scipy",
#     "scikit-learn",
#     "plotly",
#     "kaleido",
# ]
# ///
"""Parametric regression models for the two-phase starcoder experiment.

Implements and compares several parametric regression models for predicting
downstream metrics from mixture weights in a 2-domain (nemotron_full, starcoder)
x 2-phase setup:

1. Power law (adapted from https://arxiv.org/abs/2407.20177):
   y = sum_i (alpha_i + beta_i @ x)^(-gamma_i) + c

2. Linear (from https://github.com/allenai/regmixer):
   y = a + b*x_0 + c*x_1

3. Log-linear (from regmixer):
   y = exp(log_c) + exp(t_0*x_0 + t_1*x_1)

4. Log nonlinear (from regmixer):
   y = exp(log_c) + exp(t_0*x_0 + t_1*x_1) + exp(B*x_0*x_1)

Usage:
    uv run experiments/domain_phase_mix/exploratory/parametric_regression_two_phase_starcoder.py
"""

import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="An input array is constant")
warnings.filterwarnings("ignore", category=RuntimeWarning)

FEATURE_COLS = [
    "phase_0_starcoder",
    "phase_1_starcoder",
]

TARGET_COLS = [
    "eval/paloma/dolma_100_programing_languages/bpb",
    "eval/uncheatable_eval/github_python/bpb",
    "eval/uncheatable_eval/github_cpp/bpb",
    "eval/loss",
    "eval/paloma/c4_en/bpb",
    "eval/paloma/dolma-v1_5/bpb",
    "lm_eval/code2text_python_0shot/smoothed_bleu_4",
    "lm_eval/code2text_java_0shot/smoothed_bleu_4",
    "lm_eval/code2text_go_0shot/smoothed_bleu_4",
    "lm_eval/arc_challenge/acc_norm",
    "lm_eval/hellaswag_0shot/acc_norm",
    "lm_eval/piqa/acc",
    "lm_eval/boolq/acc",
    "lm_eval/averages/macro_avg_acc",
]


def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class ParametricRegressor(ABC):
    """Base class for parametric regression models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ParametricRegressor": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def n_params(self) -> int: ...


# ---------------------------------------------------------------------------
# Power law model (adapted from https://arxiv.org/abs/2407.20177)
#
# The paper models: L_i(w) = (N_0^i + w_i * N)^(-b_i) + c_i
# with total loss L = sum_i L_i.
#
# For our 2-phase 2-domain setup, we generalize each domain's contribution
# to depend on both phase weights via a linear combination:
#
#   y = (alpha_0 + beta_{0,0}*x_0 + beta_{0,1}*x_1)^(-gamma_0)
#     + (alpha_1 + beta_{1,0}*x_0 + beta_{1,1}*x_1)^(-gamma_1) + c
#
# We use softplus on the inner term to guarantee positivity without hard
# clamping, which prevents the inner^(-gamma) explosion that occurs when
# inner approaches 0.
# ---------------------------------------------------------------------------
class PowerLawRegressor(ParametricRegressor):
    """Two-term additive power law: one term per domain."""

    def __init__(self, n_terms: int = 2, n_restarts: int = 50, seed: int = 42):
        self.n_terms = n_terms
        self.n_restarts = n_restarts
        self.seed = seed
        self.params_: np.ndarray | None = None
        self._y_min: float = 0.0
        self._y_max: float = 1.0

    @property
    def name(self) -> str:
        return "PowerLaw"

    @property
    def n_params(self) -> int:
        # Per term: alpha, beta_0, beta_1, gamma  (4 params)
        # Plus one shared constant c
        return self.n_terms * 4 + 1

    @staticmethod
    def _model(X: np.ndarray, params: np.ndarray, n_terms: int) -> np.ndarray:
        """Evaluate the power law model.

        params layout: [alpha_0, beta_{0,0}, beta_{0,1}, gamma_0,
                        alpha_1, beta_{1,0}, beta_{1,1}, gamma_1, ..., c]
        """
        result = np.full(len(X), params[-1])  # constant c
        for i in range(n_terms):
            base = 4 * i
            alpha = params[base]
            beta = params[base + 1 : base + 3]
            gamma = params[base + 3]
            # Use softplus to ensure inner > 0 smoothly; add a floor of 0.1
            # to prevent inner^(-gamma) from exploding.
            raw = alpha + X @ beta
            inner = _softplus(raw) + 0.1
            result = result + inner ** (-gamma)
        return result

    def _loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        pred = self._model(X, params, self.n_terms)
        return float(np.sum((pred - y) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PowerLawRegressor":
        rng = np.random.default_rng(self.seed)
        best_loss = np.inf
        best_params = None

        self._y_min = float(y.min())
        self._y_max = float(y.max())
        y_mean = y.mean()

        for _ in range(self.n_restarts):
            p0 = []
            for _t in range(self.n_terms):
                alpha = rng.uniform(0.5, 3.0)
                beta = rng.uniform(-2.0, 2.0, size=2)
                gamma = rng.uniform(0.1, 1.5)
                p0.extend([alpha, *beta, gamma])
            p0.append(y_mean + rng.normal(0, 0.1))
            p0 = np.array(p0)

            # Bounds: gamma in (0.01, 3], others unconstrained
            bounds = []
            for _t in range(self.n_terms):
                bounds.append((None, None))  # alpha
                bounds.append((None, None))  # beta_0
                bounds.append((None, None))  # beta_1
                bounds.append((0.01, 3.0))  # gamma
            bounds.append((None, None))  # c

            try:
                res = minimize(
                    self._loss,
                    p0,
                    args=(X, y),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 2000, "ftol": 1e-12},
                )
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_params = res.x
            except Exception:
                continue

        self.params_ = best_params
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.params_ is not None, "Model not fitted"
        pred = self._model(X, self.params_, self.n_terms)
        # Clamp to reasonable range based on training data
        y_range = self._y_max - self._y_min
        return np.clip(pred, self._y_min - 2 * y_range, self._y_max + 2 * y_range)


# ---------------------------------------------------------------------------
# Linear model (from regmixer)
#
#   y = a + b*x_0 + c*x_1
#
# Note: regmixer omits intercept because their full domain weight vector
# sums to 1. In our case x_0 and x_1 are from different phases and don't
# sum to 1, so we include an intercept.
# ---------------------------------------------------------------------------
class LinearRegressor(ParametricRegressor):
    """OLS linear regression: y = intercept + beta @ x."""

    def __init__(self):
        self.coef_: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "Linear"

    @property
    def n_params(self) -> int:
        return 3  # intercept + 2 coefficients

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressor":
        X_aug = np.column_stack([np.ones(len(X)), X])
        self.coef_, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.coef_ is not None, "Model not fitted"
        X_aug = np.column_stack([np.ones(len(X)), X])
        return X_aug @ self.coef_


# ---------------------------------------------------------------------------
# Log-linear model (from regmixer)
#
#   y = exp(log_c) + exp(t_0*x_0 + t_1*x_1)
#
# Fitted via L-BFGS with Huber loss, multiple random restarts.
# The init range for log_c is adapted to the target scale.
# ---------------------------------------------------------------------------
class LogLinearRegressor(ParametricRegressor):
    """Log-linear: y = exp(log_c) + exp(t @ x)."""

    def __init__(self, n_restarts: int = 100, seed: int = 42):
        self.n_restarts = n_restarts
        self.seed = seed
        self.params_: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "LogLinear"

    @property
    def n_params(self) -> int:
        return 3  # log_c, t_0, t_1

    @staticmethod
    def _model(X: np.ndarray, params: np.ndarray) -> np.ndarray:
        log_c = params[0]
        t = params[1:]
        return np.exp(np.clip(log_c, -20, 20)) + np.exp(np.clip(X @ t, -20, 20))

    @staticmethod
    def _loss(params: np.ndarray, X: np.ndarray, y: np.ndarray, delta: float) -> float:
        pred = LogLinearRegressor._model(X, params)
        residuals = pred - y
        abs_r = np.abs(residuals)
        loss = np.where(abs_r <= delta, 0.5 * residuals**2, delta * (abs_r - 0.5 * delta))
        return float(np.sum(loss))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogLinearRegressor":
        rng = np.random.default_rng(self.seed)
        best_loss = np.inf
        best_params = None

        # Adapt init range: each exp() term should be ~ y_mean/2
        y_mean = max(float(np.mean(y)), 1e-6)
        log_c_center = np.log(y_mean / 2)
        log_c_range = np.linspace(log_c_center - 2, log_c_center + 1.5, 10)

        # Scale-aware Huber delta
        delta = max(0.02, 0.05 * float(np.std(y)))

        n_per_logc = max(1, self.n_restarts // 10)
        for log_c in log_c_range:
            for _ in range(n_per_logc):
                t = rng.uniform(-2.0, 2.0, size=2)
                p0 = np.array([log_c, *t])

                try:
                    res = minimize(
                        self._loss,
                        p0,
                        args=(X, y, delta),
                        method="L-BFGS-B",
                        options={"maxiter": 2000, "ftol": 1e-12},
                    )
                    if res.fun < best_loss:
                        best_loss = res.fun
                        best_params = res.x
                except Exception:
                    continue

        self.params_ = best_params
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.params_ is not None, "Model not fitted"
        return self._model(X, self.params_)


# ---------------------------------------------------------------------------
# Log nonlinear model (from regmixer)
#
#   y = exp(log_c) + exp(t_0*x_0 + t_1*x_1) + exp(B*x_0*x_1)
#
# Adds a pairwise interaction term inside a third exponential.
# ---------------------------------------------------------------------------
class LogNonLinearRegressor(ParametricRegressor):
    """Log nonlinear: y = exp(log_c) + exp(t @ x) + exp(B * x_0 * x_1)."""

    def __init__(self, n_restarts: int = 100, seed: int = 42):
        self.n_restarts = n_restarts
        self.seed = seed
        self.params_: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "LogNonLinear"

    @property
    def n_params(self) -> int:
        return 4  # log_c, t_0, t_1, B

    @staticmethod
    def _model(X: np.ndarray, params: np.ndarray) -> np.ndarray:
        log_c = params[0]
        t = params[1:3]
        B = params[3]
        interaction = X[:, 0] * X[:, 1]
        return (
            np.exp(np.clip(log_c, -20, 20)) + np.exp(np.clip(X @ t, -20, 20)) + np.exp(np.clip(B * interaction, -20, 20))
        )

    @staticmethod
    def _loss(params: np.ndarray, X: np.ndarray, y: np.ndarray, delta: float) -> float:
        pred = LogNonLinearRegressor._model(X, params)
        residuals = pred - y
        abs_r = np.abs(residuals)
        loss = np.where(abs_r <= delta, 0.5 * residuals**2, delta * (abs_r - 0.5 * delta))
        return float(np.sum(loss))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogNonLinearRegressor":
        rng = np.random.default_rng(self.seed)
        best_loss = np.inf
        best_params = None

        # Adapt init range: each of 3 exp() terms should be ~ y_mean/3
        y_mean = max(float(np.mean(y)), 1e-6)
        log_c_center = np.log(y_mean / 3)
        log_c_range = np.linspace(log_c_center - 2, log_c_center + 1.5, 10)

        delta = max(0.02, 0.05 * float(np.std(y)))

        n_per_logc = max(1, self.n_restarts // 10)
        for log_c in log_c_range:
            for _ in range(n_per_logc):
                t = rng.uniform(-2.0, 2.0, size=2)
                B = rng.uniform(-5.0, 2.0)
                p0 = np.array([log_c, *t, B])

                try:
                    res = minimize(
                        self._loss,
                        p0,
                        args=(X, y, delta),
                        method="L-BFGS-B",
                        options={"maxiter": 2000, "ftol": 1e-12},
                    )
                    if res.fun < best_loss:
                        best_loss = res.fun
                        best_params = res.x
                except Exception:
                    continue

        self.params_ = best_params
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.params_ is not None, "Model not fitted"
        return self._model(X, self.params_)


# ---------------------------------------------------------------------------
# Evaluation and visualization
# ---------------------------------------------------------------------------
def cross_validate(
    model_cls,
    model_kwargs: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """K-fold CV. Returns dict with models and per-fold metrics."""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results: dict[str, list] = {
        "models": [],
        "spearman": [],
        "pearson": [],
        "mse": [],
        "mae": [],
    }

    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_cls(**model_kwargs)
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)

        results["mse"].append(float(np.mean((pred_val - y_val) ** 2)))
        results["mae"].append(float(np.mean(np.abs(pred_val - y_val))))

        if np.std(pred_val) < 1e-12 or np.std(y_val) < 1e-12:
            results["spearman"].append(np.nan)
            results["pearson"].append(np.nan)
        else:
            results["spearman"].append(spearmanr(pred_val, y_val)[0])
            results["pearson"].append(pearsonr(pred_val, y_val)[0])

        results["models"].append(model)

    return results


def fit_full(model_cls, model_kwargs: dict, X: np.ndarray, y: np.ndarray) -> ParametricRegressor:
    """Fit on full dataset for visualization."""
    model = model_cls(**model_kwargs)
    model.fit(X, y)
    return model


def plot_heatmap(
    model: ParametricRegressor,
    target_col: str,
    df_complete: pd.DataFrame,
    output_dir: Path,
    resolution: int = 500,
):
    """Heatmap of predicted metric over weight space, with training runs overlaid."""
    p0 = df_complete["phase_0_starcoder"].values
    p1 = df_complete["phase_1_starcoder"].values
    vals = df_complete[target_col].values
    run_ids = df_complete["run_id"].values

    lower_is_better = "loss" in target_col or "bpb" in target_col

    g0 = np.linspace(0, 1, resolution)
    g1 = np.linspace(0, 1, resolution)
    G0, G1 = np.meshgrid(g0, g1)
    grid_features = np.column_stack([G0.ravel(), G1.ravel()])
    pred = model.predict(grid_features).reshape(resolution, resolution)

    colorscale = "Viridis_r" if lower_is_better else "Viridis"

    opt_flat = int(np.argmin(pred) if lower_is_better else np.argmax(pred))
    opt_p0 = grid_features[opt_flat, 0]
    opt_p1 = grid_features[opt_flat, 1]
    opt_val = pred.ravel()[opt_flat]

    best_idx = int(np.argmin(vals) if lower_is_better else np.argmax(vals))

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=pred,
            x=g0,
            y=g1,
            colorscale=colorscale,
            colorbar=dict(title=dict(text=target_col.replace("/", "/<br>"), font=dict(size=10))),
            hovertemplate="p0_sc=%{x:.3f}<br>p1_sc=%{y:.3f}<br>pred=%{z:.4f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=p0,
            y=p1,
            mode="markers",
            marker=dict(
                size=8,
                color=vals,
                colorscale=colorscale,
                cmin=float(pred.min()),
                cmax=float(pred.max()),
                line=dict(width=1.5, color="white"),
                showscale=False,
            ),
            text=[
                f"run_id={int(rid)}<br>p0_sc={x:.3f}<br>p1_sc={y:.3f}<br>actual={v:.4f}"
                for rid, x, y, v in zip(run_ids, p0, p1, vals)
            ],
            hoverinfo="text",
            name="Training runs",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[opt_p0],
            y=[opt_p1],
            mode="markers",
            marker=dict(size=14, symbol="star", color="red", line=dict(width=1, color="darkred")),
            name=f"Predicted opt: ({opt_p0:.3f}, {opt_p1:.3f}) = {opt_val:.4f}",
            hoverinfo="name",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[p0[best_idx]],
            y=[p1[best_idx]],
            mode="markers",
            marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(width=2.5, color="red")),
            name=f"Best observed: {vals[best_idx]:.4f}",
            hoverinfo="name",
        )
    )

    fig.update_layout(
        title=dict(text=f"{target_col} ({model.name})", font=dict(size=16), x=0.5, xanchor="center"),
        xaxis=dict(title="Phase 0 StarCoder weight", range=[0, 1], constrain="domain"),
        yaxis=dict(title="Phase 1 StarCoder weight", range=[0, 1], scaleanchor="x", scaleratio=1),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
        ),
        width=800,
        height=700,
        margin=dict(l=60, r=20, t=50, b=60),
    )

    safe_name = target_col.replace("/", "_").replace(" ", "_")
    model_tag = model.name.lower()

    html_path = output_dir / f"heatmap_{model_tag}_{safe_name}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"    Saved {html_path.name}")

    png_path = output_dir / f"heatmap_{model_tag}_{safe_name}.png"
    fig.write_image(str(png_path), scale=2)
    print(f"    Saved {png_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    script_dir = Path(__file__).parent
    df = pd.read_csv(script_dir / "two_phase_starcoder.csv")
    df_complete = df[df["status"] == "completed"].copy()
    print(f"Total runs in CSV: {len(df)}")
    print(f"Completed runs: {len(df_complete)}")

    available_targets = [c for c in TARGET_COLS if c in df_complete.columns and df_complete[c].notna().sum() > 0]
    missing = set(TARGET_COLS) - set(available_targets)
    if missing:
        print(f"Missing metrics (skipped): {missing}")

    X = df_complete[FEATURE_COLS].values
    print(f"\nFeature matrix shape: {X.shape}")
    for i, col in enumerate(FEATURE_COLS):
        print(f"  {col}: [{X[:, i].min():.4f}, {X[:, i].max():.4f}]")

    n_folds = 5
    output_dir = script_dir / "two_phase_starcoder_parametric_plots"
    output_dir.mkdir(exist_ok=True)

    heatmap_targets = [
        "eval/paloma/dolma_100_programing_languages/bpb",
        "eval/paloma/c4_en/bpb",
        "eval/uncheatable_eval/github_python/bpb",
        "lm_eval/code2text_python_0shot/smoothed_bleu_4",
        "lm_eval/averages/macro_avg_acc",
    ]

    model_defs: list[tuple[type[ParametricRegressor], dict]] = [
        (PowerLawRegressor, {"n_terms": 2, "n_restarts": 100, "seed": 42}),
        (LinearRegressor, {}),
        (LogLinearRegressor, {"n_restarts": 100, "seed": 42}),
        (LogNonLinearRegressor, {"n_restarts": 100, "seed": 42}),
    ]

    # Collect summary results for the final comparison table
    # Key: (model_name, target_col) -> {"spearman": ..., "pearson": ..., "rmse": ..., "r2": ...}
    summary: dict[tuple[str, str], dict[str, float]] = {}

    for model_cls, model_kwargs in model_defs:
        exemplar = model_cls(**model_kwargs)
        tag = exemplar.name
        print("\n" + "=" * 70)
        print(f"MODEL: {tag} ({exemplar.n_params} parameters)")
        print("=" * 70)

        header = f"{'Metric':<55} {'Spearman':<18} {'Pearson':<18} {'RMSE':<12} {'R²':<12}"
        print(f"\n{header}")
        print("-" * len(header))

        for target_col in available_targets:
            y = df_complete[target_col].values
            cv = cross_validate(model_cls, model_kwargs, X, y, n_folds=n_folds)

            valid_sp = [x for x in cv["spearman"] if not np.isnan(x)]
            valid_pe = [x for x in cv["pearson"] if not np.isnan(x)]
            rmses = [np.sqrt(m) for m in cv["mse"]]

            sp_mean = np.mean(valid_sp) if valid_sp else np.nan
            pe_mean = np.mean(valid_pe) if valid_pe else np.nan
            rmse_mean = np.mean(rmses)

            # R² = 1 - MSE / Var(y) — computed per-fold then averaged
            y_var = float(np.var(y))
            r2 = 1 - np.mean(cv["mse"]) / y_var if y_var > 1e-12 else np.nan

            sp_str = f"{sp_mean:.4f}+-{np.std(valid_sp):.4f}" if valid_sp else "N/A"
            pe_str = f"{pe_mean:.4f}+-{np.std(valid_pe):.4f}" if valid_pe else "N/A"
            rmse_str = f"{rmse_mean:.6f}"
            r2_str = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
            print(f"{target_col:<55} {sp_str:<18} {pe_str:<18} {rmse_str:<12} {r2_str:<12}")

            summary[(tag, target_col)] = {
                "spearman": sp_mean,
                "pearson": pe_mean,
                "rmse": rmse_mean,
                "r2": r2,
            }

        # Heatmaps for key metrics (fit on full data)
        print(f"\n  Generating heatmaps for {tag}...")
        for target_col in heatmap_targets:
            if target_col in available_targets:
                y = df_complete[target_col].values
                full_model = fit_full(model_cls, model_kwargs, X, y)
                print(f"  Plotting: {target_col}")
                plot_heatmap(full_model, target_col, df_complete, output_dir)

    # ========================================================================
    # OPTIMAL MIXTURES (on primary target, fit on full data)
    # ========================================================================
    primary_target = "eval/paloma/dolma_100_programing_languages/bpb"
    assert primary_target in available_targets

    print("\n" + "=" * 70)
    print(f"OPTIMAL MIXTURES FOR {primary_target}")
    print("=" * 70)

    n_grid = 10_000_000
    rng = np.random.default_rng(123)
    grid_samples = np.column_stack([rng.uniform(0, 1, n_grid), rng.uniform(0, 1, n_grid)])
    k = 128

    best_observed_idx = int(np.argmin(df_complete[primary_target].values))
    best_observed_val = float(df_complete[primary_target].values[best_observed_idx])
    best_observed_p0 = float(df_complete["phase_0_starcoder"].values[best_observed_idx])
    best_observed_p1 = float(df_complete["phase_1_starcoder"].values[best_observed_idx])

    # (model_name, p0_sc, p1_sc, pred_val) — collected for results file
    optimal_mixtures: list[tuple[str, float, float, float]] = []

    y_primary = df_complete[primary_target].values
    for model_cls, model_kwargs in model_defs:
        full_model = fit_full(model_cls, model_kwargs, X, y_primary)
        tag = full_model.name

        pred = full_model.predict(grid_samples)
        top_k_idx = np.argsort(pred)[:k]
        opt_mixture = np.mean(grid_samples[top_k_idx], axis=0)
        opt_pred = float(np.mean(pred[top_k_idx]))

        p0_sc = float(opt_mixture[0])
        p1_sc = float(opt_mixture[1])
        optimal_mixtures.append((tag, p0_sc, p1_sc, opt_pred))

        print(f"\n{tag}:")
        print(f"  Predicted optimal BPB: {opt_pred:.4f}")
        print(f"  phase_0: nemotron_full={1 - p0_sc:.4f}  starcoder={p0_sc:.4f}")
        print(f"  phase_1: nemotron_full={1 - p1_sc:.4f}  starcoder={p1_sc:.4f}")
        print(f"  Baseline format: ([{1 - p0_sc:.4f}, {p0_sc:.4f}], [{1 - p1_sc:.4f}, {p1_sc:.4f}])")

    print(f"\nBest observed: {best_observed_val:.4f} at p0_sc={best_observed_p0:.4f}, p1_sc={best_observed_p1:.4f}")

    # ========================================================================
    # COMPARISON SUMMARY
    # ========================================================================
    model_names = [model_cls(**kw).name for model_cls, kw in model_defs]
    key_targets = [
        "eval/paloma/dolma_100_programing_languages/bpb",
        "eval/uncheatable_eval/github_python/bpb",
        "eval/loss",
        "eval/paloma/c4_en/bpb",
        "lm_eval/hellaswag_0shot/acc_norm",
        "lm_eval/averages/macro_avg_acc",
    ]
    key_targets = [t for t in key_targets if t in available_targets]

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY (Spearman r, higher = better)")
    print("=" * 70)
    col_w = 14
    header = f"{'Metric':<55}" + "".join(f"{m:<{col_w}}" for m in model_names)
    print(f"\n{header}")
    print("-" * len(header))
    for t in key_targets:
        row = f"{t:<55}"
        for m in model_names:
            val = summary.get((m, t), {}).get("spearman", np.nan)
            row += f"{val:<{col_w}.4f}" if not np.isnan(val) else f"{'N/A':<{col_w}}"
        print(row)

    print(f"\n{'COMPARISON SUMMARY (R², higher = better)':}")
    print("-" * len(header))
    for t in key_targets:
        row = f"{t:<55}"
        for m in model_names:
            val = summary.get((m, t), {}).get("r2", np.nan)
            row += f"{val:<{col_w}.4f}" if not np.isnan(val) else f"{'N/A':<{col_w}}"
        print(row)

    # ========================================================================
    # WRITE RESULTS FILE
    # ========================================================================
    results_path = script_dir / "parametric_regression_two_phase_starcoder.results"
    lines = []
    lines.append(f"Primary target: {primary_target}")
    lines.append(f"Best observed: {best_observed_val:.4f} at p0_sc={best_observed_p0:.4f}, p1_sc={best_observed_p1:.4f}")
    lines.append("")
    lines.append(f"{'Model':<16} {'p0_starcoder':>12} {'p1_starcoder':>12} {'pred_bpb':>12}")
    lines.append("-" * 56)
    for tag, p0_sc, p1_sc, pred_val in optimal_mixtures:
        lines.append(f"{tag:<16} {p0_sc:>12.4f} {p1_sc:>12.4f} {pred_val:>12.4f}")
    lines.append("")
    lines.append("Cross-validation (5-fold) R² on key metrics:")
    lines.append(f"{'Metric':<55}" + "".join(f"{m:<{col_w}}" for m in model_names))
    lines.append("-" * (55 + col_w * len(model_names)))
    for t in key_targets:
        row = f"{t:<55}"
        for m in model_names:
            val = summary.get((m, t), {}).get("r2", np.nan)
            row += f"{val:<{col_w}.4f}" if not np.isnan(val) else f"{'N/A':<{col_w}}"
        lines.append(row)

    results_path.write_text("\n".join(lines) + "\n")
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
