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


# Generate a plot for each model size
for size in df_loss['model_size'].unique():
    data = df_loss[df_loss['model_size'] == size]
    
    plt.figure()
    for optimizer in data['optimizer'].unique():
        subset = data[data['optimizer'] == optimizer]
        plt.plot(
            subset['chinchilla'],
            subset['loss'],
            # label=optimizer,
            color=color_map.get(optimizer, '#000000'),
            linewidth=2,
            label=correct_name.get(optimizer, optimizer),
            linestyle=line_style.get(optimizer, '-')
        )
    
    # Labels and styling
    plt.xticks([1, 2, 4, 8], ['1', '2', '4', '8'])
    plt.xlabel('Chinchilla Ratio', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.title(f'C4/EN Loss for {size.upper()} Model', fontsize=18)
    plt.legend(fontsize=16, ncol=2)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    # plt.savefig(f'optimizer_loss_scaling_{size}.pdf', bbox_inches='tight')
    plt.savefig(f'optimizer_loss_scaling_{size}.png')
