import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv('wandb_export_2025-05-13T09_13_49.333-07_00.csv')

# Define column names for each run

muon_col = "sweep-520m-10B-muonb6462flr0.008-wd0.1-minlr0-warmup0-b10.8-b20.-bb54fc - eval/paloma/c4_en/loss"
nadamw_col = "sweep-520m-10B-muon3e335flr0.008-wd0.2-minlr0-warmup0-b10.8-b20.-b2ac88 - eval/paloma/c4_en/loss"
adamw_col = "sweep-520m-10B-muon9e8901lr0.008-wd0-minlr0-warmup0-b10.8-b20.98-f12151 - eval/paloma/c4_en/loss"

# Plotting with zoomed y-axis
plt.figure()
plt.plot(df['Step'], df[nadamw_col], label='Muon WD=0.2')
plt.plot(df['Step'], df[adamw_col], label='Muon WD=0')
plt.plot(df['Step'], df[muon_col], label='Muon WD=0.1')

# Styling
plt.xlabel('Step', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.title('520M 1x Chinchilla loss on C4/EN', fontsize=18)
plt.yticks(fontsize=18)
plt.xticks([0, 5000, 10000, 15000, 20000], [0, 5000, 10000, 15000, 20000], fontsize=18)
plt.ylim(2.88, 3.6)  # Zoom in to losses under 3.2
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.savefig('multiple_crossing_same_optimizer.pdf', bbox_inches='tight')
