import pandas as pd
import matplotlib.pyplot as plt

# Load the loss data
df_loss = pd.read_csv('optimizer_loss_scaling.csv')

# The same color map from before
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
}

# Generate a plot for each model size
for size in ['520m']:
    data = df_loss[df_loss['model_size'] == size]
    
    plt.figure()
    for optimizer in ['adamw', 'mars', 'nadamw', 'muon', 'soap', 'kron']:
        subset = data[data['optimizer'] == optimizer]
        plt.plot(
            subset['chinchilla'],
            subset['loss'],
            label=correct_name.get(optimizer, optimizer),
            color=color_map.get(optimizer, '#000000'),
            linestyle=line_style.get(optimizer, '-'),
            linewidth=2
        )
    
    # Labels and styling
    plt.xlabel('Chinchilla Ratio', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.title(f'C4/EN Loss Scaling For {size.upper()} Models', fontsize=18)
    plt.legend(fontsize=16, ncol=2)
    plt.xticks([1, 2, 4, 8], ['1', '2', '4', '8'], fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.text(
        5.6, 2.99, 
        'Matrix-based optimizers (solid) \nconsistently outperform \nscalar-based optimizers (dashed)', 
        fontsize=15, 
        ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.8)
    )
    plt.tight_layout()
    plt.savefig(f'optimizer_loss_scaling_{size}_downsampled.pdf', bbox_inches='tight')
