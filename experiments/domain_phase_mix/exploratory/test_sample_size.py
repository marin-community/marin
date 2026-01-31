# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "lightgbm",
#     "pandas",
#     "numpy",
#     "scikit-learn",
# ]
# ///
"""Test how sample size affects optimal mixture stability."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def sample_mixed_weights(rng, n_domains, vertex_prob=0.3, min_dominant_weight=0.7):
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

def sample_configs(n_samples, n_domains=3, n_phases=3, seed=42):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        config = []
        for _ in range(n_phases):
            weights = sample_mixed_weights(rng, n_domains)
            config.extend(weights)
        samples.append(config)
    return np.array(samples)

# Load data
script_dir = Path(__file__).parent
df = pd.read_csv(script_dir / '3_partitions_3_phases_6.csv')
df = df[df['run_id'] != 90000]
df_complete = df[df['status'] == 'completed'].copy()

feature_cols = [
    'phase_0_nemotron_full', 'phase_0_dolmino', 'phase_0_openthoughts_sft',
    'phase_1_nemotron_full', 'phase_1_dolmino', 'phase_1_openthoughts_sft',
    'phase_2_nemotron_full', 'phase_2_dolmino', 'phase_2_openthoughts_sft',
]
X = df_complete[feature_cols].values
y = df_complete['eval/paloma/c4_en/bpb'].values

# Train k-fold models
print("Training 5-fold models...")
hyper_params = {
    'boosting_type': 'gbdt', 'objective': 'regression',
    'num_iterations': 1000, 'seed': 42, 'learning_rate': 1e-2, 'verbosity': -1,
}
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
models = []
for train_idx, val_idx in kfold.split(X):
    gbm = lgb.LGBMRegressor(**hyper_params)
    gbm.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(stopping_rounds=3, verbose=False)])
    models.append(gbm)

# Test different sample sizes
k = 128
print("\nTesting different sample sizes for optimal mixture stability:\n")
print('n_samples  | phase_0 (nem, dol, sft) | phase_1 (nem, dol, sft) | phase_2 (nem, dol, sft) | pred_bpb')
print('-' * 115)

for n_samples in [10_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]:
    samples = sample_configs(n_samples, seed=123)

    # Ensemble prediction
    preds = np.mean([m.predict(samples) for m in models], axis=0)

    # Top-k
    top_k_idx = np.argsort(preds)[:k]
    optimal = np.mean(samples[top_k_idx], axis=0)
    pred_bpb = np.mean(preds[top_k_idx])

    p0 = optimal[0:3]
    p1 = optimal[3:6]
    p2 = optimal[6:9]
    print(f'{n_samples:>10,} | ({p0[0]:.3f}, {p0[1]:.3f}, {p0[2]:.3f}) | ({p1[0]:.3f}, {p1[1]:.3f}, {p1[2]:.3f}) | ({p2[0]:.3f}, {p2[1]:.3f}, {p2[2]:.3f}) | {pred_bpb:.4f}')

print("\n" + "="*80)
print("Analysis: Look for convergence in the optimal weights as n_samples increases.")
print("If weights stabilize, more samples won't help. If they keep changing, more samples may help.")
