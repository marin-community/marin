import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Set global font size for plots
plt.rcParams.update({'font.size': 20})

# Load the dataset
df = pd.read_csv('optimizer_loss_scaling_large.csv')


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

correct_name = {
    'adamw': 'AdamW',
    'lion': 'Lion',
    'mini': 'Adam-Mini',
    'scion': 'Scion',
    'cautious': 'Cautious',
    'mars': 'Mars',
    'nadamw': 'NAdamW',
    'muon': 'Muon',
    'soap': 'Soap',
    'kron': 'Kron',
}


line_style = {
    'adamw': '--',
    'mars': '--',
    'nadamw': '--',
    'muon': '-',
    'soap': '-',
    'kron': '-',
    'lion': '--',
    'mini': '--',
    'scion': '-',
    'cautious': '--',
}

print(df)
# Define the scaling model
def scaling_model(D, alpha, B, beta):
    return alpha * D**(-B) + beta

# Fit AdamW baseline parameters for each model size
baseline = df[df['optimizer'] == 'adamw']
params = {}
for model_size, group in baseline.groupby('model_size'):
    D = group['chinchilla'].values
    L = group['loss'].values
    p0 = [L[0] - L[-1], 0.5, L[-1]]
    popt, _ = curve_fit(scaling_model, D, L, p0=p0, maxfev=10000)
    params[model_size] = popt

# Compute effective data budgets and actual budgets
records = []
for _, row in df.iterrows():
    model_size = row['model_size']
    alpha, B, beta = params[model_size]
    D_opt = row['chinchilla']
    L_opt = row['loss']
    D_eff = ((L_opt - beta) / alpha) ** (-1.0 / B)
    records.append({
        'optimizer': row['optimizer'],
        'model_size': model_size,
        'D_opt': D_opt,
        'D_eff': D_eff
    })

eff_df = pd.DataFrame(records)
from matplotlib.patches import Patch

# Plot D_eff vs D_opt with a y=x line for each model size
for model_size in sorted(eff_df['model_size'].unique()):
    sub = eff_df[eff_df['model_size'] == model_size]
    fig, ax = plt.subplots(figsize=(8, 6))
    # Shaded speedup bands with patches for legend
    d_min, d_max = sub['D_opt'].min(), sub['D_opt'].max()
    x_vals = np.array([d_min, d_max])

    grey_patch = Patch(facecolor='bisque', alpha=0.5, label=r'1.0–1.2$\times$')
    blue_patch = Patch(facecolor='lightblue', alpha=0.5, label=r'1.2–1.3$\times$')
    green_patch = Patch(facecolor='lightgreen', alpha=0.5, label=r'1.3–1.4$\times$')
    ax.fill_between(x_vals, x_vals, x_vals * 1.2, color='bisque', alpha=0.5)
    ax.fill_between(x_vals, x_vals * 1.2, x_vals * 1.3, color='lightblue', alpha=0.5)
    ax.fill_between(x_vals, x_vals * 1.3, x_vals * 1.4, color='lightgreen', alpha=0.5)
    # Plot optimizer curves and collect handles
    line_handles, line_labels = [], []
    optimizers = ['nadamw', 'muon']
    for opt in optimizers:
        data = sub[sub['optimizer'] == opt].sort_values('D_opt')
        line, = ax.plot(
            data['D_opt'], data['D_eff'],
            marker='o',
            color=color_map[opt],
            linestyle=line_style[opt]
        )
        line_handles.append(line)
        line_labels.append(correct_name[opt])

    # First legend: optimizer lines (top left)
    legend_opt = ax.legend(
        handles=line_handles,
        labels=line_labels,
        loc='upper left',
        ncol=2,
        # title='Optimizers',
        frameon=True,
        fontsize=18
    )
    ax.add_artist(legend_opt)

    # Second legend: speedup bands (bottom right)
    ax.legend(
        handles=[grey_patch, blue_patch, green_patch],
        loc='lower right',
        title='Speedup',
        frameon=True,
        fontsize=20
    )

    # for opt in ['lion', 'nadamw', 'mars', 'soap', 'kron', 'muon']:
    #     opt_data = sub[sub['optimizer'] == opt]
    #     plt.plot(opt_data['D_opt'], opt_data['D_eff'], label=correct_name[opt], color=color_map[opt])
    # Plot y=x line

    
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xticks([1, 2, 4], ['1', '2', '4'])
    plt.xlabel('Actual Data Budget / Chinchilla ', fontsize=20)
    plt.ylabel('Effective Data Budget / Chinchilla', fontsize=20)
    plt.title(f'$D_{{eff}}$ vs $D_{{opt}}$ (Model Size: {model_size.upper()})', fontsize=20)
    # plt.legend(loc='best', fontsize=16)
    plt.tight_layout()
    # plt.savefig(f'D_eff_vs_D_opt_{model_size}M.pdf', bbox_inches='tight')
    plt.savefig(f'D_eff_vs_D_opt_{model_size}M.png')
