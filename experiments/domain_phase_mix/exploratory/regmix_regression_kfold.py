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
"""RegMix-style regression analysis with k-fold cross-validation.

This script implements the RegMix approach (http://arxiv.org/abs/2407.01492) with
k-fold cross-validation for more robust model evaluation and optimal mixture selection.

Key improvements over single train/test split:
1. Uses all data for both training and validation across folds
2. Averages predictions from k models for more stable optimization
3. Reports mean ± std of Spearman correlations across folds

Usage:
    uv run experiments/domain_phase_mix/exploratory/regmix_regression_kfold.py
"""

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)


def sample_mixed_weights(rng: np.random.Generator, n_domains: int, vertex_prob: float = 0.3,
                         min_dominant_weight: float = 0.7) -> np.ndarray:
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


def sample_configs(n_samples: int, n_domains: int = 3, n_phases: int = 3,
                   seed: int = 42) -> np.ndarray:
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
    np.random.seed(42)

    # Load data
    script_dir = Path(__file__).parent
    df = pd.read_csv(script_dir / "3_partitions_3_phases_6.csv")

    # Filter out baseline run 90000 which has anomalous loss (13.47 vs ~3.6 for others)
    # This is the pure SFT-only phase_2 run which diverged
    df = df[df['run_id'] != 90000]

    print(f"Total runs: {len(df)}")
    print(f"Completed runs: {len(df[df['status'] == 'completed'])}")

    # Feature columns (mixture weights for all phases)
    feature_cols = [
        'phase_0_nemotron_full', 'phase_0_dolmino', 'phase_0_openthoughts_sft',
        'phase_1_nemotron_full', 'phase_1_dolmino', 'phase_1_openthoughts_sft',
        'phase_2_nemotron_full', 'phase_2_dolmino', 'phase_2_openthoughts_sft',
    ]

    # Target metrics - we'll focus on key ones
    target_cols = [
        'eval/loss',
        'eval/paloma/c4_en/bpb',
        'lm_eval/arc_challenge/acc',
        'lm_eval/arc_challenge/bpb',
        'lm_eval/arc_challenge/choice_logprob',
        'lm_eval/hellaswag_0shot/acc_norm',
        'lm_eval/arc_challenge/acc_norm',
        'lm_eval/piqa/acc',
        'lm_eval/boolq/acc',
        'lm_eval/averages/macro_avg_acc',
    ]

    # Filter to completed runs
    df_complete = df[df['status'] == 'completed'].copy()

    X = df_complete[feature_cols].values
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {feature_cols}")

    # K-fold cross-validation settings
    n_folds = 5
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    print(f"\nUsing {n_folds}-fold cross-validation")

    # LightGBM hyperparameters
    hyper_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['l1', 'l2'],
        'num_iterations': 1000,
        'seed': 42,
        'learning_rate': 1e-2,
        'verbosity': -1,
    }

    print("\n" + "="*60)
    print("FITTING REGRESSION MODELS WITH K-FOLD CV")
    print("="*60)

    # Store all fold models and correlation scores
    fold_models: dict[str, list] = {col: [] for col in target_cols}
    fold_spearman: dict[str, list] = {col: [] for col in target_cols}
    fold_pearson: dict[str, list] = {col: [] for col in target_cols}

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
        X_train, X_val = X[train_idx], X[val_idx]

        for target_col in target_cols:
            y = df_complete[target_col].values
            y_train, y_val = y[train_idx], y[val_idx]

            gbm = lgb.LGBMRegressor(**hyper_params)
            reg = gbm.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='l2',
                callbacks=[lgb.early_stopping(stopping_rounds=3, verbose=False)]
            )

            pred_val = reg.predict(X_val)
            spearman_r, _ = spearmanr(pred_val, y_val)
            pearson_r, _ = pearsonr(pred_val, y_val)

            fold_models[target_col].append(reg)
            fold_spearman[target_col].append(spearman_r)
            fold_pearson[target_col].append(pearson_r)

    # Report cross-validation results
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)

    print(f"\n{'Metric':<45} {'Spearman r':<20} {'Pearson r':<20}")
    print("-" * 85)
    for target_col in target_cols:
        spearman_corrs = fold_spearman[target_col]
        pearson_corrs = fold_pearson[target_col]
        spearman_mean, spearman_std = np.mean(spearman_corrs), np.std(spearman_corrs)
        pearson_mean, pearson_std = np.mean(pearson_corrs), np.std(pearson_corrs)
        print(f"{target_col:<45} {spearman_mean:.4f} ± {spearman_std:.4f}    {pearson_mean:.4f} ± {pearson_std:.4f}")

    print("\n" + "="*60)
    print("OPTIMIZING MIXTURE WEIGHTS")
    print("="*60)

    # Sample many random mixtures using the same mixed strategy as the experiment
    n_samples = 10_000_000
    samples = sample_configs(n_samples, n_domains=3, n_phases=3, seed=123)
    print(f"\nGenerated {n_samples:,} random mixture samples using mixed strategy")

    # Predict using ensemble of all fold models (average predictions)
    print("\nPredicting on random samples (ensemble of k models)...")
    predictions = {}
    for target_col in target_cols:
        # Average predictions from all fold models
        fold_preds = [model.predict(samples) for model in fold_models[target_col]]
        predictions[target_col] = np.mean(fold_preds, axis=0)

    # Find optimal mixture for each metric
    print("\n" + "="*60)
    print("OPTIMAL MIXTURES BY METRIC")
    print("="*60)

    domain_names = ['nemotron_full', 'dolmino', 'openthoughts_sft']
    phase_names = ['phase_0', 'phase_1', 'phase_2']

    k = 128  # Top-k samples to average

    results = {}
    for target_col in target_cols:
        pred = predictions[target_col]

        # For loss/bpb metrics, lower is better; for acc/logprob metrics, higher is better
        # (choice_logprob is negative, so higher/less negative is better)
        if 'loss' in target_col or 'bpb' in target_col:
            top_k_idx = np.argsort(pred)[:k]  # Lowest loss
        else:
            top_k_idx = np.argsort(pred)[-k:]  # Highest accuracy/logprob

        top_k_samples = samples[top_k_idx]
        optimal_mixture = np.mean(top_k_samples, axis=0)

        results[target_col] = {
            'optimal_mixture': optimal_mixture,
            'pred_mean': np.mean(pred[top_k_idx]),
            'pred_std': np.std(pred[top_k_idx]),
        }

        print(f"\n{target_col}:")
        print(f"  Predicted value: {results[target_col]['pred_mean']:.4f} ± {results[target_col]['pred_std']:.4f}")

        for i, phase in enumerate(phase_names):
            phase_weights = optimal_mixture[i*3:(i+1)*3]
            print(f"  {phase}: ", end="")
            for j, domain in enumerate(domain_names):
                print(f"{domain}={phase_weights[j]:.3f} ", end="")
            print()

    # Focus on eval/paloma/c4_en/bpb as the primary optimization target
    print("\n" + "="*60)
    print("FINAL RECOMMENDATION: OPTIMIZED FOR eval/paloma/c4_en/bpb")
    print("="*60)

    target = 'eval/paloma/c4_en/bpb'
    optimal = results[target]['optimal_mixture']

    print(f"\nOptimal mixture weights (predicted bpb: {results[target]['pred_mean']:.4f}):\n")
    for i, phase in enumerate(phase_names):
        phase_weights = optimal[i*3:(i+1)*3]
        print(f"{phase}:")
        for j, domain in enumerate(domain_names):
            print(f"  {domain:20s}: {phase_weights[j]:.4f}")
        print()

    # Print as baseline format for three_phase_experiment.py
    print("\n" + "="*60)
    print("BASELINE FORMAT (for three_phase_experiment.py)")
    print("="*60)
    phase_0 = optimal[0:3]
    phase_1 = optimal[3:6]
    phase_2 = optimal[6:9]
    print(f"\n# RegMix k-fold CV optimized for eval/paloma/c4_en/bpb")
    print(f"([{phase_0[0]:.4f}, {phase_0[1]:.4f}, {phase_0[2]:.4f}], "
          f"[{phase_1[0]:.4f}, {phase_1[1]:.4f}, {phase_1[2]:.4f}], "
          f"[{phase_2[0]:.4f}, {phase_2[1]:.4f}, {phase_2[2]:.4f}]),")

    # Compare with baseline runs
    print("\n" + "="*60)
    print("COMPARISON WITH BASELINE RUNS")
    print("="*60)

    baseline_df = df_complete[df_complete['run_id'] >= 90000]
    print(f"\nBaseline runs (excluding run 90000 which diverged):")
    for _, row in baseline_df.iterrows():
        bpb = row['eval/paloma/c4_en/bpb']
        acc = row['lm_eval/averages/macro_avg_acc']
        print(f"  run_id={int(row['run_id'])}: c4_bpb={bpb:.4f}, macro_avg_acc={acc:.4f}")

    print(f"\nBest observed c4_bpb in training data: {df_complete['eval/paloma/c4_en/bpb'].min():.4f}")
    print(f"Predicted optimal c4_bpb: {results[target]['pred_mean']:.4f}")

    # Also output optimal mixture for arc_challenge/choice_logprob
    print("\n" + "="*60)
    print("OPTIMIZED FOR lm_eval/arc_challenge/choice_logprob")
    print("="*60)

    target_logprob = 'lm_eval/arc_challenge/choice_logprob'
    optimal_logprob = results[target_logprob]['optimal_mixture']

    print(f"\nOptimal mixture weights (predicted choice_logprob: {results[target_logprob]['pred_mean']:.4f}):\n")
    for i, phase in enumerate(phase_names):
        phase_weights = optimal_logprob[i*3:(i+1)*3]
        print(f"{phase}:")
        for j, domain in enumerate(domain_names):
            print(f"  {domain:20s}: {phase_weights[j]:.4f}")
        print()

    # Print as baseline format for three_phase_experiment.py
    print("BASELINE FORMAT (for three_phase_experiment.py):")
    phase_0 = optimal_logprob[0:3]
    phase_1 = optimal_logprob[3:6]
    phase_2 = optimal_logprob[6:9]
    print(f"\n# RegMix k-fold CV optimized for lm_eval/arc_challenge/choice_logprob")
    print(f"([{phase_0[0]:.4f}, {phase_0[1]:.4f}, {phase_0[2]:.4f}], "
          f"[{phase_1[0]:.4f}, {phase_1[1]:.4f}, {phase_1[2]:.4f}], "
          f"[{phase_2[0]:.4f}, {phase_2[1]:.4f}, {phase_2[2]:.4f}]),")

    # Compare choice_logprob values for baseline runs
    print(f"\nBaseline runs choice_logprob values:")
    for _, row in baseline_df.iterrows():
        logprob = row['lm_eval/arc_challenge/choice_logprob']
        print(f"  run_id={int(row['run_id'])}: choice_logprob={logprob:.4f}")

    print(f"\nBest observed choice_logprob in training data: {df_complete['lm_eval/arc_challenge/choice_logprob'].max():.4f}")
    print(f"Predicted optimal choice_logprob: {results[target_logprob]['pred_mean']:.4f}")

    # Also output optimal mixture for arc_challenge/acc
    print("\n" + "="*60)
    print("OPTIMIZED FOR lm_eval/arc_challenge/acc")
    print("="*60)

    target_acc = 'lm_eval/arc_challenge/acc'
    optimal_acc = results[target_acc]['optimal_mixture']

    print(f"\nOptimal mixture weights (predicted acc: {results[target_acc]['pred_mean']:.4f}):\n")
    for i, phase in enumerate(phase_names):
        phase_weights = optimal_acc[i*3:(i+1)*3]
        print(f"{phase}:")
        for j, domain in enumerate(domain_names):
            print(f"  {domain:20s}: {phase_weights[j]:.4f}")
        print()

    # Print as baseline format for three_phase_experiment.py
    print("BASELINE FORMAT (for three_phase_experiment.py):")
    phase_0 = optimal_acc[0:3]
    phase_1 = optimal_acc[3:6]
    phase_2 = optimal_acc[6:9]
    print(f"\n# RegMix k-fold CV optimized for lm_eval/arc_challenge/acc")
    print(f"([{phase_0[0]:.4f}, {phase_0[1]:.4f}, {phase_0[2]:.4f}], "
          f"[{phase_1[0]:.4f}, {phase_1[1]:.4f}, {phase_1[2]:.4f}], "
          f"[{phase_2[0]:.4f}, {phase_2[1]:.4f}, {phase_2[2]:.4f}]),")

    # Compare acc values for baseline runs
    print(f"\nBaseline runs arc_challenge/acc values:")
    for _, row in baseline_df.iterrows():
        acc = row['lm_eval/arc_challenge/acc']
        print(f"  run_id={int(row['run_id'])}: acc={acc:.4f}")

    print(f"\nBest observed arc_challenge/acc in training data: {df_complete['lm_eval/arc_challenge/acc'].max():.4f}")
    print(f"Predicted optimal arc_challenge/acc: {results[target_acc]['pred_mean']:.4f}")

    # Also output optimal mixture for arc_challenge/bpb
    print("\n" + "="*60)
    print("OPTIMIZED FOR lm_eval/arc_challenge/bpb")
    print("="*60)

    target_arc_bpb = 'lm_eval/arc_challenge/bpb'
    optimal_arc_bpb = results[target_arc_bpb]['optimal_mixture']

    print(f"\nOptimal mixture weights (predicted arc_bpb: {results[target_arc_bpb]['pred_mean']:.4f}):\n")
    for i, phase in enumerate(phase_names):
        phase_weights = optimal_arc_bpb[i*3:(i+1)*3]
        print(f"{phase}:")
        for j, domain in enumerate(domain_names):
            print(f"  {domain:20s}: {phase_weights[j]:.4f}")
        print()

    # Print as baseline format for three_phase_experiment.py
    print("BASELINE FORMAT (for three_phase_experiment.py):")
    phase_0 = optimal_arc_bpb[0:3]
    phase_1 = optimal_arc_bpb[3:6]
    phase_2 = optimal_arc_bpb[6:9]
    print(f"\n# RegMix k-fold CV optimized for lm_eval/arc_challenge/bpb")
    print(f"([{phase_0[0]:.4f}, {phase_0[1]:.4f}, {phase_0[2]:.4f}], "
          f"[{phase_1[0]:.4f}, {phase_1[1]:.4f}, {phase_1[2]:.4f}], "
          f"[{phase_2[0]:.4f}, {phase_2[1]:.4f}, {phase_2[2]:.4f}]),")

    # Compare arc_bpb values for baseline runs
    print(f"\nBaseline runs arc_challenge/bpb values:")
    for _, row in baseline_df.iterrows():
        arc_bpb = row['lm_eval/arc_challenge/bpb']
        print(f"  run_id={int(row['run_id'])}: arc_bpb={arc_bpb:.4f}")

    print(f"\nBest observed arc_challenge/bpb in training data: {df_complete['lm_eval/arc_challenge/bpb'].min():.4f}")
    print(f"Predicted optimal arc_challenge/bpb: {results[target_arc_bpb]['pred_mean']:.4f}")

    # ============================================================================
    # CROSS-METRIC PREDICTIONS FOR VISUALIZATION
    # ============================================================================
    # For each optimized mixture, predict ALL three key metrics
    # This is used for annotating the visualization

    print("\n" + "="*60)
    print("CROSS-METRIC PREDICTIONS FOR VISUALIZATION")
    print("="*60)

    key_metrics = ['eval/paloma/c4_en/bpb', 'lm_eval/arc_challenge/bpb', 'lm_eval/arc_challenge/choice_logprob', 'lm_eval/arc_challenge/acc']
    optimized_mixtures = {
        '90006 (RegMix opt. C4-BPB)': results['eval/paloma/c4_en/bpb']['optimal_mixture'],
        '90007 (RegMix 5-fold opt. C4-BPB)': results['eval/paloma/c4_en/bpb']['optimal_mixture'],
        '90008 (RegMix 5-fold opt. choice_logprob)': results['lm_eval/arc_challenge/choice_logprob']['optimal_mixture'],
        '90009 (RegMix 5-fold opt. arc_bpb)': results['lm_eval/arc_challenge/bpb']['optimal_mixture'],
    }

    print("\nPredicted values for each optimized mixture across all key metrics:")
    print("-" * 80)

    for mixture_name, mixture in optimized_mixtures.items():
        print(f"\n{mixture_name}:")
        mixture_reshaped = mixture.reshape(1, -1)
        for metric in key_metrics:
            # Average predictions from all fold models
            fold_preds = [model.predict(mixture_reshaped)[0] for model in fold_models[metric]]
            pred_mean = np.mean(fold_preds)
            print(f"  {metric}: {pred_mean:.4f}")

    # Print in format for visualize_mixture.py
    print("\n" + "="*60)
    print("REGMIX_PREDICTIONS FOR visualize_mixture.py")
    print("="*60)
    print("\nregmix_predictions = {")

    # Predict for the actual baseline weights used in the experiments
    baseline_weights = {
        90006: np.array([
            0.5608, 0.3962, 0.043, 0.7221, 0.2168, 0.0611, 0.6633, 0.2397, 0.0969
        ]).reshape(1, -1),
        90007: np.array([
            0.4977, 0.4568, 0.0455, 0.6575, 0.314, 0.0285, 0.6756, 0.2445, 0.0799
        ]).reshape(1, -1),
        90008: np.array([
            0.5203, 0.4339, 0.0458, 0.1468, 0.8256, 0.0277, 0.5405, 0.3860, 0.0735
        ]).reshape(1, -1),
    }

    for run_id, weights in baseline_weights.items():
        preds = {}
        for metric in key_metrics:
            fold_preds = [model.predict(weights)[0] for model in fold_models[metric]]
            preds[metric] = np.mean(fold_preds)

        print(f"    {run_id}: {{")
        print(f'        "bpb": {preds["eval/paloma/c4_en/bpb"]:.4f},')
        print(f'        "arc_bpb": {preds["lm_eval/arc_challenge/bpb"]:.4f},')
        print(f'        "choice_logprob": {preds["lm_eval/arc_challenge/choice_logprob"]:.4f},')
        print(f'        "arc_acc": {preds["lm_eval/arc_challenge/acc"]:.4f},')
        print("    },")

    print("}")


if __name__ == "__main__":
    main()
