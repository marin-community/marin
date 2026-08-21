# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Inspect the Power-Ridge single-equation optimum on the many-domain swarm."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    build_two_phase_many_loop_config,
)
from experiments.domain_phase_mix.static_batch_selection import build_dataset_spec_from_frame

SCRIPT_DIR = Path(__file__).resolve().parent
SWARM_CSV = SCRIPT_DIR.parent / "two_phase_many.csv"
TARGET_METRIC = "eval/uncheatable_eval/bpb"
MODEL_NAME = "Power-Ridge"

SUMMARY_JSON = SCRIPT_DIR / "power_ridge_single_optimum_summary.json"
WEIGHTS_CSV = SCRIPT_DIR / "power_ridge_single_optimum_weights.csv"
PLOT_PNG = SCRIPT_DIR / "power_ridge_single_optimum_comparison.png"

PHASE_MIX_ALPHA = 0.37
N_CV_SEEDS = 10
N_FOLDS = 5
N_OPT_STARTS = 64
TOP_K = 12


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _build_spec() -> tuple[pd.DataFrame, object]:
    frame = pd.read_csv(SWARM_CSV)
    if "status" in frame.columns:
        frame = frame[frame["status"] == "completed"].reset_index(drop=True)
    loop = build_two_phase_many_loop_config(objective_metric=TARGET_METRIC, name="power_ridge_single_analysis")
    spec = build_dataset_spec_from_frame(
        frame,
        objective_metric=TARGET_METRIC,
        name="power_ridge_single_analysis",
        loop=loop,
    )
    return frame, spec


def _fit_power_ridge_single(spec) -> tuple[object, dict[str, object]]:
    if spec.N != 2:
        raise ValueError(f"Power-Ridge single requires N=2 phases; got N={spec.N}")

    mixed_weights = PHASE_MIX_ALPHA * spec.weights[:, 0, :] + (1.0 - PHASE_MIX_ALPHA) * spec.weights[:, 1, :]
    x_train = np.column_stack([mixed_weights, np.sqrt(np.clip(mixed_weights, 0.0, None))])
    model = RidgeCV(alphas=np.logspace(-8, 4, 200), fit_intercept=True)
    model.fit(x_train, spec.y)

    def predict(weights_new: np.ndarray) -> np.ndarray:
        weights_new = np.asarray(weights_new, dtype=float)
        if weights_new.ndim == 2:
            weights_new = weights_new[None, :, :]
        if weights_new.shape[1] != 2:
            raise ValueError(f"Power-Ridge single requires N=2 phases; got weights with shape {weights_new.shape}")
        mixed_new = PHASE_MIX_ALPHA * weights_new[:, 0, :] + (1.0 - PHASE_MIX_ALPHA) * weights_new[:, 1, :]
        x_new = np.column_stack([mixed_new, np.sqrt(np.clip(mixed_new, 0.0, None))])
        return model.predict(x_new)

    return predict, {
        "n_params": int(x_train.shape[1] + 1),
        "phase_mix_alpha": PHASE_MIX_ALPHA,
        "ridge_alpha": float(model.alpha_),
        "intercept": float(model.intercept_),
        "coef": np.asarray(model.coef_, dtype=float).copy(),
    }


def _cross_validate(spec) -> dict[str, float]:
    fold_regrets: list[float] = []
    seed_metrics: list[tuple[float, float, float, float]] = []

    for seed in range(N_CV_SEEDS):
        seed_preds = np.full(spec.R, np.nan)
        seed_regrets: list[float] = []
        kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for train_idx, test_idx in kfold.split(spec.weights):
            train_spec = spec.subset(np.asarray(train_idx, dtype=int))
            predict_fn, _ = _fit_power_ridge_single(train_spec)
            fold_preds = np.asarray(predict_fn(spec.weights[test_idx]), dtype=float)
            seed_preds[test_idx] = fold_preds
            y_test = spec.y[test_idx]
            seed_regrets.append(float(y_test[np.argmin(fold_preds)] - np.min(y_test)))

        residuals = spec.y - seed_preds
        sse = float(np.sum(residuals**2))
        sst = float(np.sum((spec.y - np.mean(spec.y)) ** 2))
        seed_r2 = 1.0 - sse / sst
        seed_rmse = float(np.sqrt(np.mean(residuals**2)))
        seed_spearman = float(spearmanr(spec.y, seed_preds).statistic)
        seed_regret = float(np.mean(seed_regrets))
        seed_metrics.append((seed_r2, seed_rmse, seed_spearman, seed_regret))

        if seed == 0:
            fold_regrets = seed_regrets.copy()

    metrics = np.asarray(seed_metrics, dtype=float)
    return {
        "cv_r2_mean": float(metrics[:, 0].mean()),
        "cv_r2_std": float(metrics[:, 0].std(ddof=1)),
        "cv_rmse_mean": float(metrics[:, 1].mean()),
        "cv_rmse_std": float(metrics[:, 1].std(ddof=1)),
        "cv_spearman_mean": float(metrics[:, 2].mean()),
        "cv_spearman_std": float(metrics[:, 2].std(ddof=1)),
        "cv_regret_at_1_mean": float(metrics[:, 3].mean()),
        "cv_regret_at_1_std": float(metrics[:, 3].std(ddof=1)),
        "cv_seed0_regret_at_1": float(np.mean(fold_regrets)),
    }


def _optimize_mixed_simplex(intercept: float, beta_lin: np.ndarray, beta_sqrt: np.ndarray) -> tuple[float, np.ndarray]:
    def objective(logits: np.ndarray) -> float:
        weights = _softmax(logits)
        return float(intercept + beta_lin @ weights + beta_sqrt @ np.sqrt(weights))

    rng = np.random.default_rng(0)
    starts = [np.zeros_like(beta_lin)]
    starts.extend(6.0 * np.eye(beta_lin.size))
    starts.extend(rng.normal(0.0, 1.0, size=(N_OPT_STARTS, beta_lin.size)))

    best_result = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-15},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    if best_result is None:
        raise RuntimeError("Power-Ridge optimum search failed")

    optimum_weights = _softmax(np.asarray(best_result.x, dtype=float))
    return float(best_result.fun), optimum_weights


def _make_plot(
    frame: pd.DataFrame,
    spec,
    *,
    best_idx: int,
    nearest_idx: int,
    optimum_value: float,
    optimum_weights: np.ndarray,
    top_indices: np.ndarray,
) -> None:
    cmap = plt.get_cmap("RdYlGn_r")
    observed_min = float(spec.y.min())
    observed_max = float(spec.y.max())
    norm = colors.Normalize(vmin=min(observed_min, optimum_value), vmax=observed_max)

    mixed_weights = PHASE_MIX_ALPHA * spec.weights[:, 0, :] + (1.0 - PHASE_MIX_ALPHA) * spec.weights[:, 1, :]
    best_weights = mixed_weights[best_idx]
    top_labels = [spec.domain_names[idx] for idx in top_indices]
    y_positions = np.arange(top_indices.size)

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
        constrained_layout=True,
    )

    ax_left.barh(
        y_positions - 0.18,
        best_weights[top_indices],
        height=0.34,
        color=cmap(norm(frame.iloc[best_idx][TARGET_METRIC])),
        alpha=0.7,
        label=f"Best observed: {frame.iloc[best_idx]['run_name']}",
    )
    ax_left.barh(
        y_positions + 0.18,
        optimum_weights[top_indices],
        height=0.34,
        color=cmap(norm(optimum_value)),
        alpha=0.95,
        label="Power-Ridge optimum",
    )
    ax_left.set_yticks(y_positions)
    ax_left.set_yticklabels(top_labels)
    ax_left.invert_yaxis()
    ax_left.set_xlabel("Mixed Weight")
    ax_left.set_title("Top Domains In Predicted Optimum")
    ax_left.legend(loc="lower right")
    ax_left.grid(axis="x", alpha=0.2)

    ax_right.hist(spec.y, bins=24, color="#d9d9d9", edgecolor="white")
    ax_right.axvline(observed_min, color=cmap(norm(observed_min)), linewidth=3, label="Best observed")
    ax_right.axvline(
        float(frame.iloc[nearest_idx][TARGET_METRIC]),
        color=cmap(norm(float(frame.iloc[nearest_idx][TARGET_METRIC]))),
        linewidth=2,
        linestyle="--",
        label=f"Nearest observed ({frame.iloc[nearest_idx]['run_name']})",
    )
    ax_right.axvline(
        optimum_value,
        color=cmap(norm(optimum_value)),
        linewidth=3,
        linestyle="-.",
        label="Predicted optimum",
    )
    ax_right.set_xlabel(TARGET_METRIC)
    ax_right.set_ylabel("Observed Run Count")
    ax_right.set_title("Observed Support Vs Predicted Optimum")
    ax_right.grid(axis="y", alpha=0.2)
    ax_right.legend(loc="upper right")

    fig.suptitle(f"{MODEL_NAME} Single-Equation Optimum Check", fontsize=14)
    fig.savefig(PLOT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frame, spec = _build_spec()
    predict_fn, info = _fit_power_ridge_single(spec)
    predictions = np.asarray(predict_fn(spec.weights), dtype=float)
    residuals = spec.y - predictions
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((spec.y - np.mean(spec.y)) ** 2))

    beta = np.asarray(info["coef"], dtype=float)
    beta_lin = beta[: spec.M]
    beta_sqrt = beta[spec.M :]
    optimum_value, optimum_weights = _optimize_mixed_simplex(
        float(info["intercept"]),
        beta_lin,
        beta_sqrt,
    )

    mixed_weights = PHASE_MIX_ALPHA * spec.weights[:, 0, :] + (1.0 - PHASE_MIX_ALPHA) * spec.weights[:, 1, :]
    total_variation = 0.5 * np.abs(mixed_weights - optimum_weights[None, :]).sum(axis=1)
    nearest_idx = int(np.argmin(total_variation))
    best_idx = int(np.argmin(spec.y))
    chosen_idx = int(np.argmin(predictions))
    top_indices = np.argsort(-optimum_weights)[:TOP_K]

    weight_rows = []
    for rank, domain_idx in enumerate(np.argsort(-optimum_weights), start=1):
        weight_rows.append(
            {
                "rank": rank,
                "domain_name": spec.domain_names[domain_idx],
                "optimum_weight": float(optimum_weights[domain_idx]),
                "best_observed_weight": float(mixed_weights[best_idx, domain_idx]),
                "phase_0_epochs_if_constant_mix": float(
                    optimum_weights[domain_idx] * spec.epoch_multipliers[0, domain_idx]
                ),
                "phase_1_epochs_if_constant_mix": float(
                    optimum_weights[domain_idx] * spec.epoch_multipliers[1, domain_idx]
                ),
                "linear_coef": float(beta_lin[domain_idx]),
                "sqrt_coef": float(beta_sqrt[domain_idx]),
                "corner_value": float(info["intercept"] + beta_lin[domain_idx] + beta_sqrt[domain_idx]),
            }
        )
    pd.DataFrame(weight_rows).to_csv(WEIGHTS_CSV, index=False)

    summary = {
        "model_name": MODEL_NAME,
        "target_metric": TARGET_METRIC,
        "phase_mix_alpha": PHASE_MIX_ALPHA,
        "ridge_alpha": float(info["ridge_alpha"]),
        "n_runs": int(spec.R),
        "n_domains": int(spec.M),
        "train_r2": float(1.0 - sse / sst),
        "train_rmse": float(np.sqrt(np.mean(residuals**2))),
        "train_spearman": float(spearmanr(spec.y, predictions).statistic),
        "train_regret_at_1": float(spec.y[chosen_idx] - spec.y[best_idx]),
        "chosen_observed_run": str(frame.iloc[chosen_idx]["run_name"]),
        "best_observed_run": str(frame.iloc[best_idx]["run_name"]),
        "best_observed_value": float(spec.y[best_idx]),
        "predicted_best_observed_value": float(predictions[best_idx]),
        "predicted_optimum_value": float(optimum_value),
        "predicted_gain_vs_best_observed": float(spec.y[best_idx] - optimum_value),
        "nearest_observed_run": str(frame.iloc[nearest_idx]["run_name"]),
        "nearest_observed_value": float(spec.y[nearest_idx]),
        "nearest_observed_tv_distance": float(total_variation[nearest_idx]),
        "optimum_num_domains_above_1pct": int(np.sum(optimum_weights > 0.01)),
        "optimum_num_domains_above_5pct": int(np.sum(optimum_weights > 0.05)),
        "optimum_top_domains": [
            {
                "domain_name": spec.domain_names[idx],
                "weight": float(optimum_weights[idx]),
                "phase_0_epochs_if_constant_mix": float(optimum_weights[idx] * spec.epoch_multipliers[0, idx]),
                "phase_1_epochs_if_constant_mix": float(optimum_weights[idx] * spec.epoch_multipliers[1, idx]),
            }
            for idx in top_indices
        ],
    }
    summary.update(_cross_validate(spec))

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    _make_plot(
        frame,
        spec,
        best_idx=best_idx,
        nearest_idx=nearest_idx,
        optimum_value=float(optimum_value),
        optimum_weights=optimum_weights,
        top_indices=top_indices[::-1],
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {SUMMARY_JSON}")
    print(f"Saved weights to {WEIGHTS_CSV}")
    print(f"Saved plot to {PLOT_PNG}")


if __name__ == "__main__":
    main()
