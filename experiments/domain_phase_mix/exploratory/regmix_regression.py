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
# ]
# ///
"""RegMix-style regression analysis for three-phase data mixture experiment.

This script implements the RegMix approach (http://arxiv.org/abs/2407.01492) to:
1. Train LightGBM regressors to predict downstream performance from mixture weights
2. Sample many random mixtures using the same sampling strategy as the experiment
3. Use the predictors to find optimal mixture weights

Usage:
    uv run experiments/domain_phase_mix/exploratory/regmix_regression.py
"""

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)


def sample_mixed_weights(
    rng: np.random.Generator, n_domains: int, vertex_prob: float = 0.3, min_dominant_weight: float = 0.7
) -> np.ndarray:
    """Sample weights using mixed strategy (uniform + vertex-biased).

    This matches the MIXED sampling strategy from weight_sampler.py.
    """
    if rng.random() < vertex_prob:
        # Vertex-biased: one domain gets high weight
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
        # Uniform simplex sampling
        x = rng.exponential(1.0, n_domains)
        return x / x.sum()


def sample_configs(n_samples: int, n_domains: int = 3, n_phases: int = 3, seed: int = 42) -> np.ndarray:
    """Sample n_samples mixture configurations using mixed strategy.

    Returns array of shape (n_samples, n_domains * n_phases).
    """
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(n_samples):
        config = []
        for _ in range(n_phases):
            weights = sample_mixed_weights(rng, n_domains)
            config.extend(weights)
        samples.append(config)

    return np.array(samples)


def main():
    rng = np.random.default_rng(42)

    # Load data
    script_dir = Path(__file__).parent
    df = pd.read_csv(script_dir / "3_partitions_3_phases_6.csv")

    # Filter out baseline run 90000 which has anomalous loss (13.47 vs ~3.6 for others)
    # This is the pure SFT-only phase_2 run which diverged
    df = df[df["run_id"] != 90000]

    print(f"Total runs: {len(df)}")
    print(f"Completed runs: {len(df[df['status'] == 'completed'])}")

    # Feature columns (mixture weights for all phases)
    feature_cols = [
        "phase_0_nemotron_full",
        "phase_0_dolmino",
        "phase_0_openthoughts_sft",
        "phase_1_nemotron_full",
        "phase_1_dolmino",
        "phase_1_openthoughts_sft",
        "phase_2_nemotron_full",
        "phase_2_dolmino",
        "phase_2_openthoughts_sft",
    ]

    # Target metrics - we'll focus on key ones
    target_cols = [
        "eval/loss",
        "eval/paloma/c4_en/bpb",
        "lm_eval/hellaswag_0shot/acc_norm",
        "lm_eval/arc_challenge/acc_norm",
        "lm_eval/piqa/acc",
        "lm_eval/boolq/acc",
        "lm_eval/averages/macro_avg_acc",
    ]

    # Filter to completed runs
    df_complete = df[df["status"] == "completed"].copy()

    X = df_complete[feature_cols].values
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {feature_cols}")

    # Split into train/test (80/20)
    n_train = int(0.8 * len(X))
    indices = rng.permutation(len(X))
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # Train LightGBM regressors for each target metric
    hyper_params = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": ["l1", "l2"],
        "num_iterations": 1000,
        "seed": 42,
        "learning_rate": 1e-2,
        "verbosity": -1,
    }

    print("\n" + "=" * 60)
    print("FITTING REGRESSION MODELS")
    print("=" * 60)

    predictors = {}
    for target_col in target_cols:
        y = df_complete[target_col].values
        y_train, y_test = y[train_idx], y[test_idx]

        gbm = lgb.LGBMRegressor(**hyper_params)
        reg = gbm.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(stopping_rounds=3, verbose=False)],
        )

        pred_test = reg.predict(X_test)
        r, _p = spearmanr(pred_test, y_test)

        print(f"{target_col:45s} Spearman r = {r:.4f}")
        predictors[target_col] = reg

    print("\n" + "=" * 60)
    print("OPTIMIZING MIXTURE WEIGHTS")
    print("=" * 60)

    # Sample many random mixtures using the same mixed strategy as the experiment
    n_samples = 100000
    samples = sample_configs(n_samples, n_domains=3, n_phases=3, seed=123)
    print(f"\nGenerated {n_samples} random mixture samples using mixed strategy")

    # Predict for each metric
    print("\nPredicting on random samples...")
    predictions = {}
    for target_col, reg in predictors.items():
        predictions[target_col] = reg.predict(samples)

    # Find optimal mixture for each metric
    print("\n" + "=" * 60)
    print("OPTIMAL MIXTURES BY METRIC")
    print("=" * 60)

    domain_names = ["nemotron_full", "dolmino", "openthoughts_sft"]
    phase_names = ["phase_0", "phase_1", "phase_2"]

    k = 128  # Top-k samples to average

    results = {}
    for target_col in target_cols:
        pred = predictions[target_col]

        # For loss metrics, lower is better; for acc metrics, higher is better
        if "loss" in target_col or "bpb" in target_col:
            top_k_idx = np.argsort(pred)[:k]  # Lowest loss
        else:
            top_k_idx = np.argsort(pred)[-k:]  # Highest accuracy

        top_k_samples = samples[top_k_idx]
        optimal_mixture = np.mean(top_k_samples, axis=0)

        results[target_col] = {
            "optimal_mixture": optimal_mixture,
            "pred_mean": np.mean(pred[top_k_idx]),
            "pred_std": np.std(pred[top_k_idx]),
        }

        print(f"\n{target_col}:")
        print(f"  Predicted value: {results[target_col]['pred_mean']:.4f} ± {results[target_col]['pred_std']:.4f}")

        for i, phase in enumerate(phase_names):
            phase_weights = optimal_mixture[i * 3 : (i + 1) * 3]
            print(f"  {phase}: ", end="")
            for j, domain in enumerate(domain_names):
                print(f"{domain}={phase_weights[j]:.3f} ", end="")
            print()

    # Focus on eval/paloma/c4_en/bpb as the primary optimization target
    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATION: OPTIMIZED FOR eval/paloma/c4_en/bpb")
    print("=" * 60)

    target = "eval/paloma/c4_en/bpb"
    optimal = results[target]["optimal_mixture"]

    print(f"\nOptimal mixture weights (predicted bpb: {results[target]['pred_mean']:.4f}):\n")
    for i, phase in enumerate(phase_names):
        phase_weights = optimal[i * 3 : (i + 1) * 3]
        print(f"{phase}:")
        for j, domain in enumerate(domain_names):
            print(f"  {domain:20s}: {phase_weights[j]:.4f}")
        print()

    # Compare with baseline runs
    print("\n" + "=" * 60)
    print("COMPARISON WITH BASELINE RUNS")
    print("=" * 60)

    baseline_df = df_complete[df_complete["run_id"] >= 90000]
    print("\nBaseline runs (excluding run 90000 which diverged):")
    for _, row in baseline_df.iterrows():
        print(f"  run_id={int(row['run_id'])}: macro_avg_acc={row['lm_eval/averages/macro_avg_acc']:.4f}")

    print(f"\nBest observed macro_avg_acc in training data: {df_complete['lm_eval/averages/macro_avg_acc'].max():.4f}")
    print(f"Predicted optimal macro_avg_acc: {results[target]['pred_mean']:.4f}")


if __name__ == "__main__":
    main()
