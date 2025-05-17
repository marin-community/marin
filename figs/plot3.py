import pandas as pd
import matplotlib.pyplot as plt

# Load the benchmark data
df = pd.read_csv('optimizer_loss_eval.csv')

# Define the user-provided color map
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

log_scale = {1: 0, 2: 1, 4: 2, 8: 3}
# Generate a plot for each model size
for size in df['model_size'].unique():
    data = df[df['model_size'] == size]
    
    plt.figure()
    for optimizer in ['adamw', 'nadamw', 'mars', 'muon', 'soap', 'kron']:
        subset = data[data['optimizer'] == optimizer]
        plt.plot(
            # [log_scale[x] for x in subset['chinchilla']],
            subset['chinchilla'],
            subset['hellaswag_0shot'],
            # label=optimizer,
            color=color_map.get(optimizer, '#000000'),  # fallback to black if not found
            linewidth=2,
            label=correct_name.get(optimizer, optimizer),
            linestyle=line_style.get(optimizer, '-')
        )
    
    # Set labels, title, and styling as per user preference
    plt.xlabel('Chinchilla Ratio', fontsize=18)
    plt.xticks([1, 2, 4, 8], ['1', '2', '4', '8'])
    plt.ylabel('Accuracy', fontsize=18)
    plt.title(f'HellaSwag Perf for {size.upper()} Model', fontsize=18)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.grid(linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'hellaswag_0shot_{size}.pdf', bbox_inches='tight')
