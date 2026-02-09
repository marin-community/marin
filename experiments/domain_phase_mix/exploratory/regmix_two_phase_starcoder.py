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
#     "lightgbm",
#     "pandas",
#     "numpy",
#     "scipy",
#     "scikit-learn",
#     "plotly",
#     "kaleido",
# ]
# ///
"""RegMix regression for the two-phase starcoder experiment (2 domains, 2 phases).

Adapted from regmix_regression_kfold.py for the simpler 2-domain (nemotron_full, starcoder)
x 2-phase setup. Because each phase's weights sum to 1 with only 2 domains, the full
feature space is effectively 2D: (phase_0_starcoder, phase_1_starcoder).

This enables clean 2D heatmap visualization of the predicted metric landscape.

Usage:
    uv run experiments/domain_phase_mix/exploratory/regmix_two_phase_starcoder.py
"""

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="An input array is constant")

# Only use the independent features — nemotron_full = 1 - starcoder in each phase,
# so including both creates perfect multicollinearity.
FEATURE_COLS = [
    "phase_0_starcoder",
    "phase_1_starcoder",
]

DOMAIN_NAMES = ["nemotron_full", "starcoder"]
PHASE_NAMES = ["phase_0", "phase_1"]
N_DOMAINS = 2
N_PHASES = 2

# Key metrics to model — mix of code-specific and general
TARGET_COLS = [
    # Code-specific (this is the primary optimization target)
    "eval/paloma/dolma_100_programing_languages/bpb",
    "eval/uncheatable_eval/github_python/bpb",
    "eval/uncheatable_eval/github_cpp/bpb",
    # General
    "eval/loss",
    "eval/paloma/c4_en/bpb",
    "eval/paloma/dolma-v1_5/bpb",
    # Code generation / code2text
    "lm_eval/code2text_python_0shot/smoothed_bleu_4",
    "lm_eval/code2text_java_0shot/smoothed_bleu_4",
    "lm_eval/code2text_go_0shot/smoothed_bleu_4",
    # Standard LM evals
    "lm_eval/arc_challenge/acc_norm",
    "lm_eval/hellaswag_0shot/acc_norm",
    "lm_eval/piqa/acc",
    "lm_eval/boolq/acc",
    "lm_eval/averages/macro_avg_acc",
]


def sample_configs_2d(n_samples: int, seed: int = 42) -> np.ndarray:
    """Sample n_samples mixture configurations for 2 domains x 2 phases.

    Returns array of shape (n_samples, 2) with columns:
    [phase_0_starcoder, phase_1_starcoder]
    """
    rng = np.random.default_rng(seed)
    phase_0_starcoder = rng.uniform(0, 1, n_samples)
    phase_1_starcoder = rng.uniform(0, 1, n_samples)
    return np.column_stack([phase_0_starcoder, phase_1_starcoder])


def plot_scatter(
    fold_models: list,
    target_col: str,
    df_complete: pd.DataFrame,
    output_dir: Path,
    optimal_mixture: np.ndarray | None = None,
    optimal_pred: float | None = None,
):
    """Scatter plot of actual runs: x=phase_0_starcoder, y=phase_1_starcoder, color=metric.

    Args:
        optimal_mixture: Pre-computed optimal [phase_0_starcoder, phase_1_starcoder] from main().
        optimal_pred: Pre-computed predicted value at the optimal mixture.
    """
    p0 = df_complete["phase_0_starcoder"].values
    p1 = df_complete["phase_1_starcoder"].values
    vals = df_complete[target_col].values
    run_ids = df_complete["run_id"].values

    lower_is_better = "loss" in target_col or "bpb" in target_col
    colorscale = "Viridis_r" if lower_is_better else "Viridis"

    # Find the best observed run
    best_idx = int(np.argmin(vals) if lower_is_better else np.argmax(vals))
    best_val = vals[best_idx]

    # Use pre-computed optimal if provided, otherwise compute via grid search
    if optimal_mixture is not None and optimal_pred is not None:
        opt_p0 = optimal_mixture[0]
        opt_p1 = optimal_mixture[1]
        opt_val = optimal_pred
    else:
        resolution = 500
        g0 = np.linspace(0, 1, resolution)
        g1 = np.linspace(0, 1, resolution)
        G0, G1 = np.meshgrid(g0, g1)
        grid_features = np.column_stack([G0.ravel(), G1.ravel()])
        fold_preds = [model.predict(grid_features) for model in fold_models]
        pred = np.mean(fold_preds, axis=0)
        opt_flat = int(np.argmin(pred) if lower_is_better else np.argmax(pred))
        opt_p0 = grid_features[opt_flat, 0]
        opt_p1 = grid_features[opt_flat, 1]
        opt_val = pred[opt_flat]

    # Shorten the colorbar title: break on "/" boundaries
    colorbar_title = target_col.replace("/", "/<br>")

    fig = go.Figure()

    # All training runs as colored dots
    fig.add_trace(go.Scatter(
        x=p0,
        y=p1,
        mode="markers",
        marker=dict(
            size=12,
            color=vals,
            colorscale=colorscale,
            colorbar=dict(title=dict(text=colorbar_title, font=dict(size=10))),
            line=dict(width=0.5, color="#333"),
        ),
        text=[
            f"run_id={int(rid)}<br>p0_sc={x:.3f}<br>p1_sc={y:.3f}<br>{target_col}={v:.4f}"
            for rid, x, y, v in zip(run_ids, p0, p1, vals)
        ],
        hoverinfo="text",
        name="Training runs",
    ))

    # Best observed run — ring marker
    fig.add_trace(go.Scatter(
        x=[p0[best_idx]],
        y=[p1[best_idx]],
        mode="markers",
        marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(width=3, color="red")),
        name=f"Best observed: {best_val:.4f}",
        hoverinfo="name",
    ))

    # Predicted optimum — star marker
    fig.add_trace(go.Scatter(
        x=[opt_p0],
        y=[opt_p1],
        mode="markers",
        marker=dict(size=16, symbol="star", color="red", line=dict(width=1, color="darkred")),
        name=f"Predicted opt: ({opt_p0:.2f}, {opt_p1:.2f}) = {opt_val:.4f}",
        hoverinfo="name",
    ))

    fig.update_layout(
        title=dict(text=target_col, font=dict(size=16), x=0.5, xanchor="center"),
        xaxis=dict(
            title="Phase 0 StarCoder weight",
            range=[-0.02, 1.02],
            showgrid=True, gridwidth=0.5, gridcolor="#e0e0e0",
            showline=True, linewidth=1, linecolor="#666",
            constrain="domain",
        ),
        yaxis=dict(
            title="Phase 1 StarCoder weight",
            range=[-0.02, 1.02],
            showgrid=True, gridwidth=0.5, gridcolor="#e0e0e0",
            showline=True, linewidth=1, linecolor="#666",
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.8)",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=780,
        height=650,
        margin=dict(l=60, r=20, t=50, b=60),
    )

    safe_name = target_col.replace("/", "_").replace(" ", "_")
    html_path = output_dir / f"scatter_{safe_name}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"  Saved {html_path}")

    png_path = output_dir / f"scatter_{safe_name}.png"
    fig.write_image(str(png_path), scale=2)
    print(f"  Saved {png_path}")


def main():
    script_dir = Path(__file__).parent
    df = pd.read_csv(script_dir / "two_phase_starcoder.csv")

    # Filter to completed runs only
    df_complete = df[df["status"] == "completed"].copy()
    print(f"Total runs in CSV: {len(df)}")
    print(f"Completed runs: {len(df_complete)}")

    # Filter target cols to those actually present in the data
    available_targets = [c for c in TARGET_COLS if c in df_complete.columns and df_complete[c].notna().sum() > 0]
    missing = set(TARGET_COLS) - set(available_targets)
    if missing:
        print(f"Missing metrics (skipped): {missing}")

    X = df_complete[FEATURE_COLS].values
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {FEATURE_COLS}")
    print(f"Feature ranges:")
    for i, col in enumerate(FEATURE_COLS):
        print(f"  {col}: [{X[:, i].min():.4f}, {X[:, i].max():.4f}]")

    # K-fold cross-validation
    n_folds = 5
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    print(f"Using {n_folds}-fold cross-validation")

    # With only 50 samples and 2 features, use more conservative hyperparameters:
    # - higher learning rate since we have few samples
    # - more patience for early stopping
    # - fewer leaves to prevent overfitting
    hyper_params = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": ["l1", "l2"],
        "num_iterations": 500,
        "seed": 42,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 5,
        "verbosity": -1,
    }

    # ========================================================================
    # FIT MODELS
    # ========================================================================
    print("\n" + "=" * 70)
    print("FITTING REGRESSION MODELS WITH K-FOLD CV")
    print("=" * 70)

    fold_models: dict[str, list] = {col: [] for col in available_targets}
    fold_spearman: dict[str, list] = {col: [] for col in available_targets}
    fold_pearson: dict[str, list] = {col: [] for col in available_targets}

    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]

        for target_col in available_targets:
            y = df_complete[target_col].values
            y_train, y_val = y[train_idx], y[val_idx]

            gbm = lgb.LGBMRegressor(**hyper_params)
            reg = gbm.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="l2",
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            )

            pred_val = reg.predict(X_val)

            # Guard against constant predictions (model didn't learn) — skip fold for corr
            if np.std(pred_val) < 1e-12 or np.std(y_val) < 1e-12:
                sp_r = np.nan
                pe_r = np.nan
            else:
                sp_r, _ = spearmanr(pred_val, y_val)
                pe_r, _ = pearsonr(pred_val, y_val)

            fold_models[target_col].append(reg)
            fold_spearman[target_col].append(sp_r)
            fold_pearson[target_col].append(pe_r)

    # ========================================================================
    # CROSS-VALIDATION RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<55} {'Spearman r':<20} {'Pearson r':<20}")
    print("-" * 95)
    for target_col in available_targets:
        sp_corrs = [x for x in fold_spearman[target_col] if not np.isnan(x)]
        pe_corrs = [x for x in fold_pearson[target_col] if not np.isnan(x)]
        if sp_corrs:
            sp_mean, sp_std = np.mean(sp_corrs), np.std(sp_corrs)
            sp_str = f"{sp_mean:.4f} +/- {sp_std:.4f}"
        else:
            sp_str = "N/A (constant)"
        if pe_corrs:
            pe_mean, pe_std = np.mean(pe_corrs), np.std(pe_corrs)
            pe_str = f"{pe_mean:.4f} +/- {pe_std:.4f}"
        else:
            pe_str = "N/A (constant)"
        print(f"{target_col:<55} {sp_str:<20} {pe_str:<20}")

    # ========================================================================
    # OPTIMIZE MIXTURE WEIGHTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("OPTIMIZING MIXTURE WEIGHTS")
    print("=" * 70)

    n_samples = 10_000_000
    samples = sample_configs_2d(n_samples, seed=123)
    print(f"\nGenerated {n_samples:,} random mixture samples")

    # Predict with ensemble
    predictions = {}
    for target_col in available_targets:
        fold_preds = [model.predict(samples) for model in fold_models[target_col]]
        predictions[target_col] = np.mean(fold_preds, axis=0)

    k = 128  # Top-k samples to average

    print("\n" + "=" * 70)
    print("OPTIMAL MIXTURES BY METRIC")
    print("=" * 70)

    results = {}
    for target_col in available_targets:
        pred = predictions[target_col]

        lower_is_better = "loss" in target_col or "bpb" in target_col
        if lower_is_better:
            top_k_idx = np.argsort(pred)[:k]
        else:
            top_k_idx = np.argsort(pred)[-k:]

        top_k_samples = samples[top_k_idx]
        optimal_mixture = np.mean(top_k_samples, axis=0)

        results[target_col] = {
            "optimal_mixture": optimal_mixture,
            "pred_mean": np.mean(pred[top_k_idx]),
            "pred_std": np.std(pred[top_k_idx]),
        }

        # optimal_mixture is [phase_0_starcoder, phase_1_starcoder]
        p0_sc = optimal_mixture[0]
        p1_sc = optimal_mixture[1]
        print(f"\n{target_col}:")
        print(f"  Predicted value: {results[target_col]['pred_mean']:.4f} +/- {results[target_col]['pred_std']:.4f}")
        print(f"  phase_0: nemotron_full={1 - p0_sc:.3f}  starcoder={p0_sc:.3f}")
        print(f"  phase_1: nemotron_full={1 - p1_sc:.3f}  starcoder={p1_sc:.3f}")

    # ========================================================================
    # PRIMARY TARGET: dolma_100_programing_languages/bpb
    # ========================================================================
    primary_target = "eval/paloma/dolma_100_programing_languages/bpb"

    print("\n" + "=" * 70)
    print(f"FINAL RECOMMENDATION: OPTIMIZED FOR {primary_target}")
    print("=" * 70)

    optimal = results[primary_target]["optimal_mixture"]
    p0_sc = optimal[0]
    p1_sc = optimal[1]
    print(f"\nOptimal mixture (predicted bpb: {results[primary_target]['pred_mean']:.4f}):\n")
    print("phase_0:")
    print(f"  nemotron_full       : {1 - p0_sc:.4f}")
    print(f"  starcoder           : {p0_sc:.4f}")
    print("phase_1:")
    print(f"  nemotron_full       : {1 - p1_sc:.4f}")
    print(f"  starcoder           : {p1_sc:.4f}")

    # Baseline format
    print("\nBASELINE FORMAT (for two_phase_starcoder_experiment.py):")
    print(f"  ([{1 - p0_sc:.4f}, {p0_sc:.4f}], [{1 - p1_sc:.4f}, {p1_sc:.4f}]),")

    # Best observed
    print(f"\nBest observed {primary_target} in training data: {df_complete[primary_target].min():.4f}")
    print(f"Predicted optimal: {results[primary_target]['pred_mean']:.4f}")

    # ========================================================================
    # CROSS-METRIC: predictions for all key metrics at the code-optimal point
    # ========================================================================
    print("\n" + "=" * 70)
    print("CROSS-METRIC PREDICTIONS AT CODE-OPTIMAL MIXTURE")
    print("=" * 70)

    code_opt = results[primary_target]["optimal_mixture"].reshape(1, -1)
    for metric in available_targets:
        fold_preds = [model.predict(code_opt)[0] for model in fold_models[metric]]
        pred_val = np.mean(fold_preds)
        print(f"  {metric}: {pred_val:.4f}")

    # ========================================================================
    # HEATMAP VISUALIZATIONS
    # ========================================================================
    print("\n" + "=" * 70)
    print("GENERATING HEATMAP VISUALIZATIONS")
    print("=" * 70)

    output_dir = script_dir / "two_phase_starcoder_plots"
    output_dir.mkdir(exist_ok=True)

    # Plot heatmaps for key metrics
    heatmap_targets = [
        primary_target,
        "eval/paloma/c4_en/bpb",
        "eval/uncheatable_eval/github_python/bpb",
        "lm_eval/code2text_python_0shot/smoothed_bleu_4",
        "lm_eval/averages/macro_avg_acc",
    ]

    for target_col in heatmap_targets:
        if target_col in fold_models and fold_models[target_col]:
            print(f"\nPlotting: {target_col}")
            opt_mix = results[target_col]["optimal_mixture"] if target_col in results else None
            opt_pred = results[target_col]["pred_mean"] if target_col in results else None
            plot_scatter(fold_models[target_col], target_col, df_complete, output_dir, opt_mix, opt_pred)


if __name__ == "__main__":
    main()
