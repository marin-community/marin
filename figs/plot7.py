import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv('wandb_export_2025-05-13T09_07_36.509-07_00.csv')

# Define column names for each run
muon_col = 'sweep-520m-85B-muong0ea950lr0.004-wd0.1-minlr0-warmup0-b10.8-b20-9ac240 - eval/paloma/c4_en/loss'
nadamw_col = 'sweep-520m-85B-nadamwcf9d95lr0.004-wd0.1-minlr0-warmup4000-b10.9-d09912 - eval/paloma/c4_en/loss'
adamw_col = 'sweep-520m-85B-adamwf14f39lr0.004-wd0.1-minlr0-warmup1000-b10.9--84afa9 - eval/paloma/c4_en/loss'


# Plotting with zoomed y-axis
plt.figure()

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

plt.plot(df['Step'], df[nadamw_col], label='NAdamW', color=color_map['nadamw'])
plt.plot(df['Step'], df[adamw_col], label='AdamW', color=color_map['adamw'])
plt.plot(df['Step'], df[muon_col], label='Muon', color=color_map['muon'])



plt.text(
    20000, 3.0, 
    'Early loss curve\n may mislead', 
    fontsize=18, 
    ha='center', va='bottom',
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.8)
)

# Styling
plt.xlabel('Step', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.title('520M 8x Chinchilla loss on C4/EN', fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.ylim(2.88, 3.6)  # Zoom in to losses under 3.2
plt.legend(fontsize=18)
# plt.grid(True)
plt.tight_layout()
# plt.savefig('multiple_crossing.pdf', bbox_inches='tight')
plt.savefig('multiple_crossing.png')
