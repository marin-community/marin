import pandas as pd
import matplotlib.pyplot as plt

# Load the new CSV

color_map = {
    "mars": "#1f77b4",  # blue
    "muon": "#ff7f0e",  # orange
    "lion": "#2ca02c",  # green
    "adamw": "#d62728",  # red
    "nadamw": "#9467bd",  # purple
    "kron": "#8c564b",  # brown
    "scion": "#e377c2",  # pink
    "cautious": "#7f7f7f",  # gray
    "soape": "#bcbd22",  # yellow-green
    "sophia": "#17becf",  # cyan
    "mini": "#aec7e8",  # light blue
}

correct_name = {
    "adamw": "AdamW",
    "lion": "Lion",
    "mini": "Adam-Mini",
    "scion": "Scion",
    "cautious": "Cautious",
    "mars": "Mars",
    "nadamw": "NAdamW",
    "muon": "Muon",
    "soape": "Soap",
    "kron": "Kron",
}


# line_style = {
#     "adamw": "--",
#     "mars": "--",
#     "nadamw": "--",
#     "muon": "-",
#     "soape": "-",
#     "kron": "-",
#     "lion": "--",
#     "mini": "--",
#     "scion": "-",
#     "cautious": "--",
# }

df = pd.read_csv('experiments/optimizer_sweep/Analysis/Meta/fig1.csv')

# Define steps and loss series
steps = df['Step']
loss_series = {
    'AdamW w/ lr 6e-4': df['sweep-130m-5B-adamw_baseline2e2854lr0.0006-wd0.1-minlr0-warmup20-0ac09e - eval/paloma/c4_en/loss'],
    'Mars': df['sweep-130m-2B-marse45e36lr0.016-wd0.1-minlr0-warmup2000-b10.9-b2-2de81b - eval/paloma/c4_en/loss'],
    'Nesterov AdamW': df['sweep-130m-2B-nadamw96aba0lr0.008-wd0.1-minlr0-warmup2000-b10.95-2ac247 - eval/paloma/c4_en/loss'],
    'AdamW w/ lr 8e-3': df['sweep-130m-2B-adamw0848aelr0.008-wd0.1-minlr0-warmup2000-b10.9-b-413f11 - eval/paloma/c4_en/loss'],
}

# Extract AdamW 6e-4 values at 5k and 10k
val_5k = loss_series['AdamW w/ lr 8e-3'][df['Step'] == 5119].values[0]
val_10k = loss_series['AdamW w/ lr 6e-4'][df['Step'] == 10000].values[0]

# Plot setup
plt.figure(figsize=(7, 5))
for label, series in loss_series.items():
    mask = series.notna()
    map_to_key = {
        'AdamW w/ lr 6e-4': 'adamw',
        'AdamW w/ lr 8e-3': 'adamw',
        'Mars': 'mars',
        'Nesterov AdamW': 'nadamw',
    }
    plt.plot(steps[mask], series[mask], label=label, color=color_map[map_to_key[label]])

# Zoom in on lower losses (≤4)
all_losses = pd.concat([s.dropna() for s in loss_series.values()])
plt.ylim(all_losses.min() - 0.05, 4)

# Draw precise arrow from 5k to 10k
plt.annotate(
    '',
    xy=(10000, val_10k),
    xytext=(5000, val_5k),
    arrowprops=dict(arrowstyle='->', lw=2)
)

# Add centered text box above mid-point
mid_x = 2100
mid_y = val_5k + (val_10k - val_5k) / 2 
plt.text(
    mid_x, mid_y,
    'Tuning LR of AdamW \nleads to 2x speedup',
    fontsize=16,
    ha='center', va='bottom',
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.8)
)

# Highlight endpoints
plt.scatter([5000, 10000], [val_5k, val_10k], zorder=5)

# Formatting
plt.xlabel('Step', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.legend(fontsize=18)
# Formatting
plt.title('Loss on C4/EN on 130M models', fontsize=18)
plt.tick_params(axis='both', labelsize=18)
plt.tight_layout()
plt.savefig('experiments/optimizer_sweep/Analysis/Meta/fig1.pdf')
