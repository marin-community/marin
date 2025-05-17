import pandas as pd
import matplotlib.pyplot as plt

# Set font size for all elements
plt.rcParams.update({'font.size': 18})

# Load data

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


df_lion = pd.read_csv('wandb_export_2025-05-13T08_58_35.163-07_00.csv')
df_adam = pd.read_csv('wandb_export_2025-05-13T08_57_25.434-07_00.csv')

# Prepare data for Lion
lion_grouped = df_lion.groupby('optimizer.weight_decay')['eval/paloma/c4_en/loss'].min().reset_index()
wd_lion = lion_grouped['optimizer.weight_decay']
loss_lion = lion_grouped['eval/paloma/c4_en/loss']

# Plot for Lion
plt.figure()
plt.plot(wd_lion, loss_lion, label='Lion', color=color_map["lion"])



# Prepare data for Adam
adam_grouped = df_adam.groupby('optimizer.weight_decay')['eval/paloma/c4_en/loss'].mean().reset_index()
wd_adam = adam_grouped['optimizer.weight_decay']
loss_adam = adam_grouped['eval/paloma/c4_en/loss']

opt_wd = 0.6
opt_loss = loss_lion[wd_lion == opt_wd].values[0]

plt.scatter([opt_wd], [opt_loss], marker='*', s=200)
plt.annotate(
    'wd ≈ 0.6 is optimal for Lion',
    xy=(opt_wd, opt_loss),
    xytext=(opt_wd - 0.35, opt_loss + 0.02),
    arrowprops=dict(arrowstyle='->', lw=1.5),
    fontsize=18,
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.8)
)
# Plot for Adam
plt.plot(wd_adam, loss_adam, label='AdamW', color=color_map["adamw"])
plt.xlabel('Weight Decay', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('C4/EN Loss vs Weight Decay', fontsize=18)
plt.tick_params(axis='both', labelsize=18)
plt.legend(fontsize=18)
# plt.savefig('wd_matters.pdf', bbox_inches='tight')
plt.savefig('wd_matters.png')
