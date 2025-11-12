#!/usr/bin/env python
"""Fit and visualize memorization scaling law: log(P(z)) = c1*log(epochs) + c2*log(seed_set) + c3"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sys

# Check for mode argument
mode = "separate"  # default
if len(sys.argv) > 1:
    mode = sys.argv[1]
    if mode not in ["separate", "total"]:
        print(f"Error: mode must be 'separate' or 'total', got '{mode}'")
        print("Usage: python mem_scaling.py [separate|total]")
        sys.exit(1)

print(f"Running in '{mode}' mode")

# Load data
df = pd.read_csv("comma_150m_runs.csv")

# Filter out duplicates, keep only unique (token_set, epochs) pairs
df_unique = df[df['duplicate'] == 'No'].copy()

print(f"Loaded {len(df)} runs, using {len(df_unique)} unique (token_set, epochs) pairs")

# Check for problematic P(z) values
print(f"\nP(z) statistics:")
print(f"  Min: {df_unique['final_mean_pz'].min()}")
print(f"  Max: {df_unique['final_mean_pz'].max()}")
print(f"  Values <= 0: {(df_unique['final_mean_pz'] <= 0).sum()}")

# For P(z) = 0, use a reasonable epsilon
# Since P(z) values span many orders of magnitude, use a small but reasonable value
# Common practice: use 1e-10 or half the minimum measurable value
zero_mask = df_unique['final_mean_pz'] == 0
if zero_mask.any():
    # Use a reasonable epsilon rather than min_nonzero (which might be too extreme)
    epsilon = 1e-10
    print(f"\nWarning: Found {zero_mask.sum()} runs with P(z) = 0")
    print(f"Replacing with epsilon = {epsilon:.2e}")
    print(df_unique[zero_mask][['token_set', 'epochs', 'final_mean_pz', 'run_name']])
    df_unique.loc[zero_mask, 'final_mean_pz'] = epsilon

# Also check for extremely small values that might be numerical artifacts
tiny_mask = (df_unique['final_mean_pz'] > 0) & (df_unique['final_mean_pz'] < 1e-40)
if tiny_mask.any():
    print(f"\nInfo: Found {tiny_mask.sum()} runs with extremely small P(z) < 1e-40")
    print(df_unique[tiny_mask][['token_set', 'epochs', 'final_mean_pz', 'run_name']])

print(f"\nUsing {len(df_unique)} runs for analysis")

# Parse seed set size (e.g., "1M" -> 1e6)
def parse_token_set(token_set_str):
    """Convert '1M', '10M', '100M' to actual token counts"""
    value = int(token_set_str.rstrip('M'))
    return value * 1_000_000

df_unique['seed_set_tokens'] = df_unique['token_set'].apply(parse_token_set)

# Calculate total tokens
df_unique['total_tokens'] = df_unique['seed_set_tokens'] * df_unique['epochs']

# Prepare data for regression based on mode
if mode == "separate":
    X = np.column_stack([
        np.log(df_unique['epochs']),
        np.log(df_unique['seed_set_tokens'])
    ])
    formula = "log(P(z)) = c₁·log(epochs) + c₂·log(seed_set) + c₃"
    x_labels = ["log(epochs)", "log(seed_set_tokens)"]
elif mode == "total":
    X = np.log(df_unique['total_tokens']).values.reshape(-1, 1)
    formula = "log(P(z)) = c₁·log(total_tokens) + c₂"
    x_labels = ["log(total_tokens)"]

y = np.log(df_unique['final_mean_pz'])

# Fit linear regression in log-space
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# Print results
print("\n" + "="*60)
print(f"MEMORIZATION SCALING LAW ({mode.upper()} MODE)")
print("="*60)

if mode == "separate":
    c1, c2 = model.coef_
    c3 = model.intercept_
    print(f"\nlog(P(z)) = {c1:.4f}·log(epochs) + {c2:.4f}·log(seed_set) + {c3:.4f}")
    print(f"\nR² score: {r2:.4f}")
    print("\nInterpretation:")
    print(f"  - Epochs coefficient (c₁): {c1:.4f}")
    print(f"    → Each 10x increase in epochs multiplies P(z) by {10**c1:.2f}x")
    print(f"  - Seed set coefficient (c₂): {c2:.4f}")
    print(f"    → Each 10x increase in seed set multiplies P(z) by {10**c2:.2f}x")
    print(f"  - Baseline (c₃): {c3:.4f}")
elif mode == "total":
    c1 = model.coef_[0]
    c2 = model.intercept_
    print(f"\nlog(P(z)) = {c1:.4f}·log(total_tokens) + {c2:.4f}")
    print(f"\nR² score: {r2:.4f}")
    print("\nInterpretation:")
    print(f"  - Total tokens coefficient (c₁): {c1:.4f}")
    print(f"    → Each 10x increase in total tokens multiplies P(z) by {10**c1:.2f}x")
    print(f"  - Baseline (c₂): {c2:.4f}")

print("="*60)

# Add predictions to dataframe
df_unique['predicted_mean_pz'] = np.exp(y_pred)
df_unique['residual'] = df_unique['final_mean_pz'] - df_unique['predicted_mean_pz']

# Save results
df_unique.to_csv("comma_150m_scaling_results.csv", index=False)
print(f"\nSaved results to comma_150m_scaling_results.csv")

# Visualization
if mode == "separate":
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
else:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Reshape axes for consistency
    axes = np.array([[axes[0, 0], axes[0, 1], None],
                     [axes[1, 0], axes[1, 1], None]])

# Plot 1: Actual vs Predicted
ax = axes[0, 0]
ax.scatter(df_unique['final_mean_pz'], df_unique['predicted_mean_pz'],
           c=df_unique['seed_set_tokens'], cmap='viridis', s=100, alpha=0.7)
ax.plot([df_unique['final_mean_pz'].min(), df_unique['final_mean_pz'].max()],
        [df_unique['final_mean_pz'].min(), df_unique['final_mean_pz'].max()],
        'k--', lw=2, label='Perfect fit')
ax.set_xlabel('Actual P(z)', fontsize=12)
ax.set_ylabel('Predicted P(z)', fontsize=12)
ax.set_title(f'Actual vs Predicted (R² = {r2:.4f})', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(ax.scatter(df_unique['final_mean_pz'], df_unique['predicted_mean_pz'],
                                c=df_unique['seed_set_tokens'], cmap='viridis', s=100, alpha=0.7), ax=ax)
cbar.set_label('Seed Set Size (tokens)', fontsize=10)

# Plot 2: P(z) vs Epochs or Total Tokens
ax = axes[0, 1]

if mode == "separate":
    # Plot by seed set
    for token_set in sorted(df_unique['token_set'].unique(), key=lambda x: int(x.rstrip('M'))):
        subset = df_unique[df_unique['token_set'] == token_set]
        subset = subset.sort_values('epochs')
        ax.plot(subset['epochs'], subset['final_mean_pz'], 'o-', label=f'{token_set} tokens', markersize=8, linewidth=2)

        # Plot predicted scaling curve
        epochs_range = np.logspace(np.log10(subset['epochs'].min()),
                                   np.log10(subset['epochs'].max()), 100)
        seed_tokens = subset['seed_set_tokens'].iloc[0]
        pred_pz = np.exp(c1 * np.log(epochs_range) + c2 * np.log(seed_tokens) + c3)
        ax.plot(epochs_range, pred_pz, '--', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_title('Memorization Scaling with Epochs', fontsize=14)
    ax.set_ylim([0, 1])  # Set y-axis to [0, 1]

elif mode == "total":
    # Plot by total tokens, colored by seed set
    for token_set in sorted(df_unique['token_set'].unique(), key=lambda x: int(x.rstrip('M'))):
        subset = df_unique[df_unique['token_set'] == token_set]
        subset = subset.sort_values('total_tokens')
        ax.scatter(subset['total_tokens'], subset['final_mean_pz'], label=f'{token_set} seed', s=100, alpha=0.7)

    # Plot predicted scaling curve
    total_range = np.logspace(np.log10(df_unique['total_tokens'].min()),
                              np.log10(df_unique['total_tokens'].max()), 100)
    pred_pz = np.exp(c1 * np.log(total_range) + c2)
    ax.plot(total_range, pred_pz, 'k--', linewidth=2, label='Fit', alpha=0.8)

    ax.set_xlabel('Total Tokens', fontsize=12)
    ax.set_title('Memorization Scaling with Total Tokens', fontsize=14)
    ax.set_ylim([0, 1])  # Set y-axis to [0, 1]

ax.set_xscale('log')
ax.set_ylabel('P(z)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# Plot 3: Residuals vs Epochs
ax = axes[1, 0]
for token_set in sorted(df_unique['token_set'].unique(), key=lambda x: int(x.rstrip('M'))):
    subset = df_unique[df_unique['token_set'] == token_set]
    ax.scatter(subset['epochs'], subset['residual'], label=f'{token_set} tokens', s=100, alpha=0.7)
ax.axhline(0, color='k', linestyle='--', linewidth=2)
ax.set_xscale('log')
ax.set_xlabel('Epochs', fontsize=12)
ax.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
ax.set_title('Residuals Analysis', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Heatmap of P(z) values
ax = axes[1, 1]
pivot = df_unique.pivot_table(values='final_mean_pz', index='token_set', columns='epochs')
# Sort by token set size
pivot = pivot.reindex(sorted(pivot.index, key=lambda x: int(x.rstrip('M'))))
im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f'{int(e)}' for e in pivot.columns], rotation=45, ha='right')
ax.set_xlabel('Epochs', fontsize=12)
ax.set_ylabel('Token Set', fontsize=12)
ax.set_title('P(z) Heatmap', fontsize=14)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('P(z)', fontsize=10)

# Plot 5: Additional 2D plot for separate mode - P(z) vs seed set for different epoch counts
if mode == "separate":
    ax = axes[0, 2]

    # Get unique epoch counts, select a subset for visualization
    all_epochs = sorted(df_unique['epochs'].unique())
    # Select representative epochs (spread across the range)
    if len(all_epochs) > 8:
        epoch_subset = [all_epochs[i] for i in [0, len(all_epochs)//4, len(all_epochs)//2,
                                                  3*len(all_epochs)//4, -1]]
    else:
        epoch_subset = all_epochs

    for epoch_val in epoch_subset:
        subset = df_unique[df_unique['epochs'] == epoch_val]
        if len(subset) >= 2:  # Only plot if we have multiple seed sets
            subset = subset.sort_values('seed_set_tokens')
            ax.plot(subset['seed_set_tokens'], subset['final_mean_pz'],
                   'o-', label=f'{epoch_val} epochs', markersize=8, linewidth=2)

    ax.set_xscale('log')
    ax.set_xlabel('Seed Set Size (tokens)', fontsize=12)
    ax.set_ylabel('P(z)', fontsize=12)
    ax.set_ylim([0, 1])
    ax.set_title('Memorization vs Seed Set Size', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Plot 6: Contour/heatmap of P(z) as function of both epochs and seed set
    ax = axes[1, 2]

    # Create meshgrid for contour plot
    seed_sets_unique = sorted(df_unique['seed_set_tokens'].unique())
    epochs_unique = sorted(df_unique['epochs'].unique())

    # Create pivot table
    pivot = df_unique.pivot_table(values='final_mean_pz',
                                   index='seed_set_tokens',
                                   columns='epochs')

    # Plot as contour
    X_mesh, Y_mesh = np.meshgrid(epochs_unique, seed_sets_unique)
    contour = ax.contourf(X_mesh, Y_mesh, pivot.values, levels=15, cmap='YlOrRd')

    # Add contour lines
    contour_lines = ax.contour(X_mesh, Y_mesh, pivot.values, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Seed Set Size (tokens)', fontsize=12)
    ax.set_title('P(z) Contour Map', fontsize=14)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('P(z)', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
output_filename = f'memorization_scaling_law_{mode}.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nSaved visualization to {output_filename}")
plt.show()
