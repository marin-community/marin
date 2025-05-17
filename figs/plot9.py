import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv('wandb_export_2025-05-13T16_13_44.986-07_00.csv')

# Define steps and loss series
steps = df['Step']
loss_series = {
    'AdamW w/ lr 6e-4': df['sweep-130m-5B-adamw_baseline2e2854lr0.0006-wd0.1-minlr0-warmup20-0ac09e - eval/paloma/c4_en/loss'],
    'Mars': df['sweep-130m-2B-marse45e36lr0.016-wd0.1-minlr0-warmup2000-b10.9-b2-2de81b - eval/paloma/c4_en/loss'],
    'Nesterov AdamW': df['sweep-130m-2B-nadamw96aba0lr0.008-wd0.1-minlr0-warmup2000-b10.95-2ac247 - eval/paloma/c4_en/loss'],
    'AdamW w/ lr 8e-3': df['sweep-130m-2B-adamw0848aelr0.008-wd0.1-minlr0-warmup2000-b10.9-b-413f11 - eval/paloma/c4_en/loss'],
}

color_map = {
    'mars': '#1f77b4',    # blue
    'muon': '#ff7f0e',    # orange
    'lion': '#2ca02c',    # green
    'adamw': '#d62728',   # red
    'nadamw': '#9467bd',  # purple
    'kron': '#8c564b',    # brown
    'scion': '#e377c2',   # pink
    'cautious': '#7f7f7f', # gray
    'soap': '#bcbd22',    # yellow-green
    'sophia': '#17becf',  # cyan
    'mini': '#aec7e8',    # light blue
}


# Extract AdamW 6e-4 values at 5k and 10k
val_5k = df.loc[df['Step'] == 5000, 'sweep-130m-2B-adamw0848aelr0.008-wd0.1-minlr0-warmup2000-b10.9-b-413f11 - eval/paloma/c4_en/loss'].values[0]
val_10k = df.loc[df['Step'] == 10000, 'sweep-130m-5B-adamw_baseline2e2854lr0.0006-wd0.1-minlr0-warmup20-0ac09e - eval/paloma/c4_en/loss'].values[0]

# Plot setup
plt.figure()
for label, series in loss_series.items():
    mask = series.notna()
    if label == 'AdamW w/ lr 8e-3':
        color = color_map["adamw"]
    if label == 'Nesterov AdamW':
        color = color_map["nadamw"]
    if label == 'AdamW w/ lr 6e-4':
        # make it pink
        color = "#e377c2"
    if label == 'Mars':
        color = color_map["mars"]
        
    plt.plot(steps[mask], series[mask], label=label, color=color)

# Zoom in on lower losses
all_losses = pd.concat([s.dropna() for s in loss_series.values()])
plt.ylim(all_losses.min() - 0.05, 4)

# Draw arrow from (5k, val_5k) to (10k, val_10k)
plt.annotate(
    '', 
    xy=(10000, val_10k), 
    xytext=(5000, val_5k), 
    arrowprops=dict(arrowstyle='->', lw=2)
)

# Add text box above the midpoint
mid_x = 2000
mid_y = val_5k + (val_10k - val_5k) / 2 - 0.02
plt.text(
    mid_x, mid_y, 
    'Tuning learning rate\nleads to 2x speedup', 
    fontsize=16, 
    ha='center', va='bottom',
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.8)
)

# Highlight the two endpoints
plt.scatter([5000, 10000], [val_5k, val_10k], zorder=5)

# Formatting
plt.xlabel('Step', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.legend(fontsize=18)
plt.title('Loss on C4/EN on 130M models', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig('loss_curves_zoomed_in.pdf', bbox_inches='tight')
