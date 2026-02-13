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
"""Parametric regression models for the 3-partition x 3-phase experiment.

Implements and compares several parametric regression models for predicting
downstream metrics from mixture weights in a 3-domain (nemotron_full, dolmino,
openthoughts_sft) x 3-phase setup:

1. Power law (adapted from https://arxiv.org/abs/2407.20177):
   y = sum_{d} (alpha_d + beta_d @ x_d)^(-gamma_d) + c
   where x_d = [phase_0_d, phase_1_d, phase_2_d] for each domain d.

2. Linear (from https://github.com/allenai/regmixer):
   y = intercept + beta @ x  (9 features)

3. Log-linear (from regmixer):
   y = exp(log_c) + exp(t @ x)

4. Log nonlinear (from regmixer):
   y = exp(log_c) + exp(t @ x) + sum_{phase p} sum_{pairs (i,j)} exp(B_{p,ij} * x_{p,i} * x_{p,j})

Usage:
    uv run experiments/domain_phase_mix/exploratory/parametric_regression_3_partitions_3_phases.py
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

N_DOMAINS = 3
N_PHASES = 3
N_FEATURES = N_DOMAINS * N_PHASES

DOMAIN_NAMES = ["nemotron_full", "dolmino", "openthoughts_sft"]
PHASE_NAMES = ["phase_0", "phase_1", "phase_2"]

FEATURE_COLS = [f"{phase}_{domain}" for phase in PHASE_NAMES for domain in DOMAIN_NAMES]

TARGET_COLS = [
    "eval/loss",
    "eval/paloma/c4_en/bpb",
    "lm_eval/arc_challenge/acc",
    "lm_eval/arc_challenge/bpb",
    "lm_eval/arc_challenge/choice_logprob",
    "lm_eval/hellaswag_0shot/acc_norm",
    "lm_eval/arc_challenge/acc_norm",
    "lm_eval/piqa/acc",
    "lm_eval/boolq/acc",
    "lm_eval/averages/macro_avg_acc",
]


def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def _extract_domain_features(X: np.ndarray, domain_idx: int) -> np.ndarray:
    """Extract features for one domain across all phases.

    X has columns ordered as [p0_d0, p0_d1, p0_d2, p1_d0, p1_d1, p1_d2, p2_d0, p2_d1, p2_d2].
    For domain d, we extract columns [d, d+3, d+6].
    Returns shape (n_samples, N_PHASES).
    """
    cols = [phase * N_DOMAINS + domain_idx for phase in range(N_PHASES)]
    return X[:, cols]


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
# Domain-specific terms: one per domain, each uses that domain's weights
# across all 3 phases:
#
#   y = sum_{d=0}^{2} (alpha_d + beta_d @ x_d)^(-gamma_d) + c
#
# where x_d = [phase_0_d, phase_1_d, phase_2_d].
# Per term: alpha (1) + beta (3) + gamma (1) = 5 params.
# Total: 3*5 + 1 = 16 params.
# ---------------------------------------------------------------------------
class PowerLawRegressor(ParametricRegressor):
    """Domain-specific additive power law: one term per domain."""

    def __init__(self, n_restarts: int = 50, seed: int = 42):
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
        # Per domain: alpha, beta_0, beta_1, beta_2, gamma = 5
        # Plus one shared constant c
        return N_DOMAINS * (1 + N_PHASES + 1) + 1

    @staticmethod
    def _model(X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Evaluate the power law model.

        params layout: [alpha_0, beta_{0,0..2}, gamma_0,
                        alpha_1, beta_{1,0..2}, gamma_1,
                        alpha_2, beta_{2,0..2}, gamma_2, c]
        """
        params_per_term = 1 + N_PHASES + 1  # alpha + betas + gamma
        result = np.full(len(X), params[-1])  # constant c
        for d in range(N_DOMAINS):
            base = params_per_term * d
            alpha = params[base]
            beta = params[base + 1 : base + 1 + N_PHASES]
            gamma = params[base + 1 + N_PHASES]
            # Extract this domain's features across phases
            X_domain = _extract_domain_features(X, d)
            raw = alpha + X_domain @ beta
            inner = _softplus(raw) + 0.1
            result = result + inner ** (-gamma)
        return result

    def _loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        pred = self._model(X, params)
        return float(np.sum((pred - y) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PowerLawRegressor":
        rng = np.random.default_rng(self.seed)
        best_loss = np.inf
        best_params = None

        self._y_min = float(y.min())
        self._y_max = float(y.max())
        y_mean = y.mean()
        params_per_term = 1 + N_PHASES + 1

        for _ in range(self.n_restarts):
            p0 = []
            for _d in range(N_DOMAINS):
                alpha = rng.uniform(0.5, 3.0)
                beta = rng.uniform(-2.0, 2.0, size=N_PHASES)
                gamma = rng.uniform(0.1, 1.5)
                p0.extend([alpha, *beta, gamma])
            p0.append(y_mean + rng.normal(0, 0.1))
            p0 = np.array(p0)

            bounds = []
            for _d in range(N_DOMAINS):
                bounds.append((None, None))  # alpha
                for _ in range(N_PHASES):
                    bounds.append((None, None))  # beta
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
        pred = self._model(X, self.params_)
        y_range = self._y_max - self._y_min
        return np.clip(pred, self._y_min - 2 * y_range, self._y_max + 2 * y_range)


# ---------------------------------------------------------------------------
# Linear model (from regmixer)
#
#   y = intercept + beta @ x  (9 features + intercept = 10 params)
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
        return N_FEATURES + 1

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
#   y = exp(log_c) + exp(t @ x)  (1 + 9 = 10 params)
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
        return 1 + N_FEATURES

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

        y_mean = max(float(np.mean(y)), 1e-6)
        log_c_center = np.log(y_mean / 2)
        log_c_range = np.linspace(log_c_center - 2, log_c_center + 1.5, 10)
        delta = max(0.02, 0.05 * float(np.std(y)))

        n_per_logc = max(1, self.n_restarts // 10)
        for log_c in log_c_range:
            for _ in range(n_per_logc):
                t = rng.uniform(-2.0, 2.0, size=N_FEATURES)
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
#   y = exp(log_c) + exp(t @ x) + sum_{phase p} sum_{pairs (i,j)} exp(B_{p,ij} * x_{p,i} * x_{p,j})
#
# Within each phase: 3 choose 2 = 3 pairs. 3 phases * 3 pairs = 9 interactions.
# Total: 1 + 9 + 9 = 19 params.
# ---------------------------------------------------------------------------
class LogNonLinearRegressor(ParametricRegressor):
    """Log nonlinear with within-phase pairwise interactions."""

    # Precompute the within-phase interaction column pairs (indices into the 9-feature vector)
    _INTERACTION_PAIRS: list[tuple[int, int]] = []
    for _p in range(N_PHASES):
        _base = _p * N_DOMAINS
        for _i in range(N_DOMAINS):
            for _j in range(_i + 1, N_DOMAINS):
                _INTERACTION_PAIRS.append((_base + _i, _base + _j))
    N_INTERACTIONS = len(_INTERACTION_PAIRS)  # 9

    def __init__(self, n_restarts: int = 100, seed: int = 42):
        self.n_restarts = n_restarts
        self.seed = seed
        self.params_: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "LogNonLinear"

    @property
    def n_params(self) -> int:
        return 1 + N_FEATURES + self.N_INTERACTIONS

    @staticmethod
    def _model(X: np.ndarray, params: np.ndarray) -> np.ndarray:
        log_c = params[0]
        t = params[1 : 1 + N_FEATURES]
        B = params[1 + N_FEATURES :]

        result = np.exp(np.clip(log_c, -20, 20)) + np.exp(np.clip(X @ t, -20, 20))
        for k, (i, j) in enumerate(LogNonLinearRegressor._INTERACTION_PAIRS):
            interaction = X[:, i] * X[:, j]
            result = result + np.exp(np.clip(B[k] * interaction, -20, 20))
        return result

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

        n_interaction = self.N_INTERACTIONS
        y_mean = max(float(np.mean(y)), 1e-6)
        # Each of (2 + n_interaction) exp terms should be ~ y_mean / (2 + n_interaction)
        log_c_center = np.log(y_mean / (2 + n_interaction))
        log_c_range = np.linspace(log_c_center - 2, log_c_center + 1.5, 10)
        delta = max(0.02, 0.05 * float(np.std(y)))

        n_per_logc = max(1, self.n_restarts // 10)
        for log_c in log_c_range:
            for _ in range(n_per_logc):
                t = rng.uniform(-2.0, 2.0, size=N_FEATURES)
                B = rng.uniform(-5.0, 2.0, size=n_interaction)
                p0 = np.array([log_c, *t, *B])

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
# Evaluation
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


# ---------------------------------------------------------------------------
# Sampling (from regmix_regression_kfold.py)
# ---------------------------------------------------------------------------
def sample_mixed_weights(
    rng: np.random.Generator, n_domains: int, vertex_prob: float = 0.3, min_dominant_weight: float = 0.7
) -> np.ndarray:
    """Sample weights using mixed strategy (uniform + vertex-biased)."""
    if rng.random() < vertex_prob:
        dominant = rng.integers(n_domains)
        dominant_weight = rng.uniform(min_dominant_weight, 1.0)
        remaining = 1 - dominant_weight

        weights = np.zeros(n_domains)
        weights[dominant] = dominant_weight

        if n_domains > 1 and remaining > 0:
            other_weights = rng.dirichlet(np.ones(n_domains - 1))
            other_idx = 0
            for i in range(n_domains):
                if i != dominant:
                    weights[i] = remaining * other_weights[other_idx]
                    other_idx += 1
        return weights
    else:
        x = rng.exponential(1.0, n_domains)
        return x / x.sum()


def sample_configs(n_samples: int, seed: int = 42) -> np.ndarray:
    """Sample n_samples mixture configs (sum-to-1 per phase), vectorized."""
    rng = np.random.default_rng(seed)
    result = np.empty((n_samples, N_DOMAINS * N_PHASES))

    for phase in range(N_PHASES):
        col_start = phase * N_DOMAINS

        # Uniform simplex samples for all rows
        exp_samples = rng.exponential(1.0, size=(n_samples, N_DOMAINS))
        row_sums = exp_samples.sum(axis=1, keepdims=True)
        uniform = exp_samples / row_sums

        # Vertex-biased samples: one domain gets high weight
        vertex_mask = rng.random(n_samples) < 0.3
        n_vertex = int(vertex_mask.sum())
        if n_vertex > 0:
            dominant = rng.integers(N_DOMAINS, size=n_vertex)
            dominant_weight = rng.uniform(0.7, 1.0, size=n_vertex)
            remaining = 1.0 - dominant_weight

            vertex_weights = np.zeros((n_vertex, N_DOMAINS))
            # Distribute remaining weight among non-dominant domains
            other_exp = rng.exponential(1.0, size=(n_vertex, N_DOMAINS - 1))
            other_sums = other_exp.sum(axis=1, keepdims=True)
            other_normed = other_exp / other_sums * remaining[:, None]

            for i in range(n_vertex):
                d = dominant[i]
                vertex_weights[i, d] = dominant_weight[i]
                other_idx = 0
                for j in range(N_DOMAINS):
                    if j != d:
                        vertex_weights[i, j] = other_normed[i, other_idx]
                        other_idx += 1

            uniform[vertex_mask] = vertex_weights

        result[:, col_start : col_start + N_DOMAINS] = uniform

    return result


# ---------------------------------------------------------------------------
# Ternary plot visualization
# ---------------------------------------------------------------------------
def simplex_grid(resolution: int = 100) -> np.ndarray:
    """Generate points on the 2-simplex (triangle)."""
    points = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            points.append([i / resolution, j / resolution, k / resolution])
    return np.array(points)


def plot_ternary(
    model: ParametricRegressor,
    target_col: str,
    phase_idx: int,
    optimal_mixture: np.ndarray,
    df_complete: pd.DataFrame,
    output_dir: Path,
    resolution: int = 100,
):
    """Ternary plot for one phase, fixing others at optimal values.

    Args:
        model: Fitted model.
        target_col: Target metric column name.
        phase_idx: Which phase (0, 1, 2) to sweep.
        optimal_mixture: Shape (9,) — the full optimal mixture.
        df_complete: DataFrame with training runs.
        output_dir: Where to save plots.
        resolution: Grid resolution for the simplex.
    """
    lower_is_better = "loss" in target_col or "bpb" in target_col

    # Generate simplex grid for the swept phase
    grid = simplex_grid(resolution)  # (n_points, 3)

    # Build full 9-feature input: fix other phases at optimal, sweep this phase
    n_points = len(grid)
    X_grid = np.tile(optimal_mixture, (n_points, 1))
    phase_start = phase_idx * N_DOMAINS
    X_grid[:, phase_start : phase_start + N_DOMAINS] = grid

    pred = model.predict(X_grid)

    # Training run data for overlay
    run_phase_weights = df_complete[[f"{PHASE_NAMES[phase_idx]}_{d}" for d in DOMAIN_NAMES]].values
    actual_vals = df_complete[target_col].values
    run_ids = df_complete["run_id"].values

    # Optimal point for this phase
    opt_phase = optimal_mixture[phase_start : phase_start + N_DOMAINS]

    colorscale = "Viridis_r" if lower_is_better else "Viridis"

    fig = go.Figure()

    # Contour/scatter over the simplex grid
    fig.add_trace(
        go.Scatterternary(
            a=grid[:, 0],
            b=grid[:, 1],
            c=grid[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color=pred,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text=target_col.replace("/", "/<br>"),
                        font=dict(size=9),
                    ),
                    len=0.8,
                ),
            ),
            text=[f"pred={v:.4f}" for v in pred],
            hovertemplate=(
                f"{DOMAIN_NAMES[0]}=%{{a:.3f}}<br>"
                f"{DOMAIN_NAMES[1]}=%{{b:.3f}}<br>"
                f"{DOMAIN_NAMES[2]}=%{{c:.3f}}<br>"
                "pred=%{text}<extra></extra>"
            ),
            name="Predicted",
            showlegend=False,
        )
    )

    # Overlay training runs
    fig.add_trace(
        go.Scatterternary(
            a=run_phase_weights[:, 0],
            b=run_phase_weights[:, 1],
            c=run_phase_weights[:, 2],
            mode="markers",
            marker=dict(
                size=8,
                color=actual_vals,
                colorscale=colorscale,
                cmin=float(pred.min()),
                cmax=float(pred.max()),
                line=dict(width=1.5, color="white"),
                showscale=False,
            ),
            text=[
                f"run_id={int(rid)}<br>{DOMAIN_NAMES[0]}={w[0]:.3f}<br>"
                f"{DOMAIN_NAMES[1]}={w[1]:.3f}<br>{DOMAIN_NAMES[2]}={w[2]:.3f}<br>"
                f"actual={v:.4f}"
                for rid, w, v in zip(run_ids, run_phase_weights, actual_vals)
            ],
            hoverinfo="text",
            name="Training runs",
        )
    )

    # Mark predicted optimum
    fig.add_trace(
        go.Scatterternary(
            a=[opt_phase[0]],
            b=[opt_phase[1]],
            c=[opt_phase[2]],
            mode="markers",
            marker=dict(size=14, symbol="star", color="red", line=dict(width=1, color="darkred")),
            name=f"Predicted opt: ({opt_phase[0]:.3f}, {opt_phase[1]:.3f}, {opt_phase[2]:.3f})",
            hoverinfo="name",
        )
    )

    # Mark best observed
    best_idx = int(np.argmin(actual_vals) if lower_is_better else np.argmax(actual_vals))
    fig.add_trace(
        go.Scatterternary(
            a=[run_phase_weights[best_idx, 0]],
            b=[run_phase_weights[best_idx, 1]],
            c=[run_phase_weights[best_idx, 2]],
            mode="markers",
            marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(width=2.5, color="red")),
            name=f"Best observed: {actual_vals[best_idx]:.4f}",
            hoverinfo="name",
        )
    )

    # Fixed phases annotation
    fixed_phases_text = []
    for p in range(N_PHASES):
        if p != phase_idx:
            w = optimal_mixture[p * N_DOMAINS : (p + 1) * N_DOMAINS]
            fixed_phases_text.append(f"{PHASE_NAMES[p]}: [{', '.join(f'{v:.3f}' for v in w)}]")
    subtitle = f"Fixed: {'; '.join(fixed_phases_text)}"

    fig.update_layout(
        title=dict(
            text=f"{target_col} ({model.name}) — {PHASE_NAMES[phase_idx]}<br><br>" f"<sub>{subtitle}</sub>",
            font=dict(size=14),
            x=0.5,
            xanchor="center",
            y=0.98,
        ),
        ternary=dict(
            aaxis=dict(title=DOMAIN_NAMES[0], min=0),
            baxis=dict(title=DOMAIN_NAMES[1], min=0),
            caxis=dict(title=DOMAIN_NAMES[2], min=0),
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            font=dict(size=9),
            bgcolor="rgba(255,255,255,0.8)",
        ),
        width=800,
        height=700,
        margin=dict(l=60, r=60, t=100, b=60),
    )

    safe_target = target_col.replace("/", "_").replace(" ", "_")
    model_tag = model.name.lower()

    html_path = output_dir / f"ternary_{model_tag}_{safe_target}_{PHASE_NAMES[phase_idx]}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"      Saved {html_path.name}")

    png_path = output_dir / f"ternary_{model_tag}_{safe_target}_{PHASE_NAMES[phase_idx]}.png"
    fig.write_image(str(png_path), scale=2)
    print(f"      Saved {png_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import sys

    sys.stdout.reconfigure(line_buffering=True)

    script_dir = Path(__file__).parent
    df = pd.read_csv(script_dir / "3_partitions_3_phases_6.csv")

    # Filter out anomalous run 90000
    df = df[df["run_id"] != 90000]
    df_complete = df[df["status"] == "completed"].copy()
    print(f"Total runs (excl. 90000): {len(df)}")
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
    output_dir = script_dir / "3_partitions_3_phases_parametric_plots"
    output_dir.mkdir(exist_ok=True)

    ternary_targets = [
        "eval/paloma/c4_en/bpb",
        "lm_eval/averages/macro_avg_acc",
        "eval/loss",
        "lm_eval/arc_challenge/choice_logprob",
    ]

    # Fewer restarts than the 2-phase script since we have more parameters
    # (16 for PowerLaw, 19 for LogNonLinear) and more features (9 vs 2).
    model_defs: list[tuple[type[ParametricRegressor], dict]] = [
        (PowerLawRegressor, {"n_restarts": 10, "seed": 42}),
        (LinearRegressor, {}),
        (LogLinearRegressor, {"n_restarts": 50, "seed": 42}),
        (LogNonLinearRegressor, {"n_restarts": 30, "seed": 42}),
    ]

    # Collect summary results
    summary: dict[tuple[str, str], dict[str, float]] = {}

    # Collect optimal mixtures for primary target
    primary_target = "eval/paloma/c4_en/bpb"

    # ========================================================================
    # PHASE 1: Cross-validation (fast feedback)
    # ========================================================================
    for model_cls, model_kwargs in model_defs:
        exemplar = model_cls(**model_kwargs)
        tag = exemplar.name
        print("\n" + "=" * 70)
        print(f"MODEL: {tag} ({exemplar.n_params} parameters)")
        print("=" * 70)

        header = f"{'Metric':<50} {'Spearman':<18} {'Pearson':<18} {'RMSE':<12} {'R²':<12}"
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

            y_var = float(np.var(y))
            r2 = 1 - np.mean(cv["mse"]) / y_var if y_var > 1e-12 else np.nan

            sp_str = f"{sp_mean:.4f}+-{np.std(valid_sp):.4f}" if valid_sp else "N/A"
            pe_str = f"{pe_mean:.4f}+-{np.std(valid_pe):.4f}" if valid_pe else "N/A"
            rmse_str = f"{rmse_mean:.6f}"
            r2_str = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
            print(f"{target_col:<50} {sp_str:<18} {pe_str:<18} {rmse_str:<12} {r2_str:<12}")

            summary[(tag, target_col)] = {
                "spearman": sp_mean,
                "pearson": pe_mean,
                "rmse": rmse_mean,
                "r2": r2,
            }

    # ========================================================================
    # PHASE 2: Optimal mixture search + ternary plots
    # ========================================================================
    n_opt_samples = 1_000_000
    k = 128
    print(f"\nSampling {n_opt_samples:,} random mixture configs...")
    opt_samples = sample_configs(n_opt_samples, seed=123)
    print("Done sampling.")

    optimal_mixtures: list[tuple[str, np.ndarray, float]] = []
    y_primary = df_complete[primary_target].values

    for model_cls, model_kwargs in model_defs:
        tag = model_cls(**model_kwargs).name

        # Optimal mixture for primary target
        full_model = fit_full(model_cls, model_kwargs, X, y_primary)
        pred = full_model.predict(opt_samples)
        top_k_idx = np.argsort(pred)[:k]
        opt_mixture = np.mean(opt_samples[top_k_idx], axis=0)
        opt_pred = float(np.mean(pred[top_k_idx]))
        optimal_mixtures.append((tag, opt_mixture, opt_pred))

        print(f"\n  {tag} optimal for {primary_target} (pred={opt_pred:.4f}):")
        for p in range(N_PHASES):
            w = opt_mixture[p * N_DOMAINS : (p + 1) * N_DOMAINS]
            parts = " ".join(f"{DOMAIN_NAMES[d]}={w[d]:.4f}" for d in range(N_DOMAINS))
            print(f"    {PHASE_NAMES[p]}: {parts}")

        # Ternary plots
        print(f"\n  Generating ternary plots for {tag}...")
        for target_col in ternary_targets:
            if target_col not in available_targets:
                continue
            y = df_complete[target_col].values
            full_model_vis = fit_full(model_cls, model_kwargs, X, y)

            # Get optimal mixture for this specific target
            pred_vis = full_model_vis.predict(opt_samples)
            lower_is_better = "loss" in target_col or "bpb" in target_col
            top_k_idx_vis = np.argsort(pred_vis)[:k] if lower_is_better else np.argsort(pred_vis)[-k:]
            opt_mix_vis = np.mean(opt_samples[top_k_idx_vis], axis=0)

            print(f"    Plotting: {target_col}")
            for phase_idx in range(N_PHASES):
                plot_ternary(
                    full_model_vis,
                    target_col,
                    phase_idx,
                    opt_mix_vis,
                    df_complete,
                    output_dir,
                )

    # ========================================================================
    # OPTIMAL MIXTURES SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"OPTIMAL MIXTURES FOR {primary_target}")
    print("=" * 70)

    for tag, opt_mix, opt_pred in optimal_mixtures:
        print(f"\n{tag} (pred={opt_pred:.4f}):")
        for p in range(N_PHASES):
            w = opt_mix[p * N_DOMAINS : (p + 1) * N_DOMAINS]
            parts = " ".join(f"{DOMAIN_NAMES[d]}={w[d]:.4f}" for d in range(N_DOMAINS))
            print(f"  {PHASE_NAMES[p]}: {parts}")

    best_observed_idx = int(np.argmin(y_primary))
    best_observed_val = float(y_primary[best_observed_idx])
    print(f"\nBest observed: {best_observed_val:.4f}")

    # ========================================================================
    # COMPARISON SUMMARY
    # ========================================================================
    model_names = [model_cls(**kw).name for model_cls, kw in model_defs]
    key_targets = [
        t
        for t in available_targets
        if t
        in [
            "eval/loss",
            "eval/paloma/c4_en/bpb",
            "lm_eval/hellaswag_0shot/acc_norm",
            "lm_eval/averages/macro_avg_acc",
            "lm_eval/arc_challenge/acc_norm",
            "lm_eval/piqa/acc",
        ]
    ]

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY (Spearman r, higher = better)")
    print("=" * 70)
    col_w = 14
    header = f"{'Metric':<50}" + "".join(f"{m:<{col_w}}" for m in model_names)
    print(f"\n{header}")
    print("-" * len(header))
    for t in key_targets:
        row = f"{t:<50}"
        for m in model_names:
            val = summary.get((m, t), {}).get("spearman", np.nan)
            row += f"{val:<{col_w}.4f}" if not np.isnan(val) else f"{'N/A':<{col_w}}"
        print(row)

    print(f"\n{'COMPARISON SUMMARY (R², higher = better)':}")
    print("-" * len(header))
    for t in key_targets:
        row = f"{t:<50}"
        for m in model_names:
            val = summary.get((m, t), {}).get("r2", np.nan)
            row += f"{val:<{col_w}.4f}" if not np.isnan(val) else f"{'N/A':<{col_w}}"
        print(row)

    # ========================================================================
    # WRITE RESULTS FILE
    # ========================================================================
    results_path = script_dir / "parametric_regression_3_partitions_3_phases.results"
    lines = []
    lines.append(f"Primary target: {primary_target}")
    lines.append(f"Best observed: {best_observed_val:.4f}")
    lines.append("")

    for tag, opt_mix, pred_val in optimal_mixtures:
        lines.append(f"{tag} (pred={pred_val:.4f}):")
        for p in range(N_PHASES):
            w = opt_mix[p * N_DOMAINS : (p + 1) * N_DOMAINS]
            parts = " ".join(f"{DOMAIN_NAMES[d]}={w[d]:.4f}" for d in range(N_DOMAINS))
            lines.append(f"  {PHASE_NAMES[p]}: {parts}")
        phase_0 = opt_mix[0:3]
        phase_1 = opt_mix[3:6]
        phase_2 = opt_mix[6:9]
        lines.append(
            f"  Baseline format: ([{phase_0[0]:.4f}, {phase_0[1]:.4f}, {phase_0[2]:.4f}], "
            f"[{phase_1[0]:.4f}, {phase_1[1]:.4f}, {phase_1[2]:.4f}], "
            f"[{phase_2[0]:.4f}, {phase_2[1]:.4f}, {phase_2[2]:.4f}])"
        )
        lines.append("")

    lines.append("Cross-validation (5-fold) R² on key metrics:")
    lines.append(f"{'Metric':<50}" + "".join(f"{m:<{col_w}}" for m in model_names))
    lines.append("-" * (50 + col_w * len(model_names)))
    for t in key_targets:
        row = f"{t:<50}"
        for m in model_names:
            val = summary.get((m, t), {}).get("r2", np.nan)
            row += f"{val:<{col_w}.4f}" if not np.isnan(val) else f"{'N/A':<{col_w}}"
        lines.append(row)

    results_path.write_text("\n".join(lines) + "\n")
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
