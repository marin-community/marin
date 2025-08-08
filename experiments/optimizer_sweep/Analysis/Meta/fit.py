import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Re-prepare data for Muon and AdamW
data = {
    'optimizer': ['muon', 'muon', 'muon', 'muon', 'adamw', 'adamw', 'adamw', 'adamw', 
 'soape', 'soape', 'soape', 'soape','nadamw', 'nadamw', 'nadamw', 'nadamw',],
    'model_size': ['130m', '300m', '520m', '1.2b'] * 4,
    'chinchilla': [1]*16,
    'loss': [
3.2946722507476807, 3.079275369644165, 2.9442315101623535, 2.7799859046936035,
3.3216564655303955, 3.094308853149414, 2.958468198776245, 2.7871193885803223,
3.294482469558716, 3.082421064376831, 2.944326400756836, 2.78251,
3.319291353225708, 3.08992862701416, 2.954136371612549, 2.784677267074585,
]
}
df = pd.DataFrame(data)
expected_params = {
    "130m": 134217728,
    "300m": 301989888,
    "520m": 536870912,
    "1.2b": 1207959552,
}
df['params'] = df['model_size'].map(expected_params)
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
    "mini": "#aec7e8",  # light blue
}

# Define power law model
def power_law(N, A, alpha):
    return A * N**(-alpha)

# Fit with linear regression on log-log data
plt.figure(figsize=(8, 6))

for optimizer in ['muon', 'adamw', 'soape', 'nadamw']:
    df_optimizer = df[df['optimizer']==optimizer]
    x_data, y_data = df_optimizer['params'].values, df_optimizer['loss'].values

    # Perform linear regression on log-log data
    log_x = np.log(x_data)
    log_y = np.log(y_data)
    coeffs = np.polyfit(log_x, log_y, 1)
    m, c = coeffs
    
    # Convert back to power law parameters: L = A * N^(-alpha)
    alpha = -m
    A = np.exp(c)

    Ns = np.logspace(np.log10(min(expected_params.values())), np.log10(7e9), 200)
    plt.plot(x_data, y_data, 'o', label=f'{optimizer} data', color=color_map[optimizer])
    plt.plot(Ns, power_law(Ns, A, alpha), '-', label=f'{optimizer} fit (A={A:.3f}, α={alpha:.3f})', color=color_map[optimizer])
    extrap_N = np.array([4e9, 7e9])
    loss_ex = power_law(extrap_N, A, alpha)
    plt.plot(extrap_N, loss_ex, 'x', markersize=8, label=f'{optimizer} @4B/7B', color=color_map[optimizer])




# Plot updated fits
plt.xscale('log')
plt.xlabel('Parameter count (log scale)', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.title('4x Chinchilla', fontsize=18)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('experiments/optimizer_sweep/Analysis/Meta/fit_linear.png')
