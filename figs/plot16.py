import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch

# Load the data
df = pd.read_csv('wandb_export_2025-05-14T21_52_13.808-07_00.csv')
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
# Identify loss columns and map names
loss_cols = [col for col in df.columns if col.endswith('eval/paloma/c4_en/loss')]
mapping = {}
for col in loss_cols:
    lname = col.lower()
    if 'soap' in lname:
        mapping[col] = 'Soap'
    elif 'nadamw' in lname:
        mapping[col] = 'NAdamW'
    elif 'adamw' in lname and 'nadamw' not in lname:
        mapping[col] = 'AdamW'
    elif 'muon' in lname:
        mapping[col] = 'Muon'


# Melt into long format
df_long = (
    df[['Step'] + loss_cols]
    .rename(columns=mapping)
    .melt(id_vars='Step', var_name='Optimizer', value_name='Loss')
)

df_long = df_long.dropna()


# Create main figure and axes
fig, ax = plt.subplots(figsize=(10, 6))
for opt in ['AdamW', 'NAdamW', 'Soap', 'Muon']:
    if opt != 'Muon':
        subset = df_long[df_long['Optimizer'] == opt]
        ax.plot(subset['Step'], subset['Loss'], label=opt, color = color_map[opt.lower()], linestyle = line_style[opt.lower()])
    else:
        subset = df_long[df_long['Optimizer'] == opt]
        ax.plot(subset['Step'] / 2, subset['Loss'], label=opt, color = color_map[opt.lower()], linestyle = line_style[opt.lower()])   

# Main plot styling
ax.set_xlabel('Data', fontsize=18)
ax.set_ylabel('Loss', fontsize=18)
ax.set_ylim(3.18, 3.5)
ax.tick_params(axis='both', labelsize=18)
ax.legend(fontsize=18)
ax.set_title('C4/EN Loss Curve 130M 16x Chinchilla', fontsize=20)

# Define zoom region bounds
max_step = 40959
start_step = 38000
inset_y_min, inset_y_max = 2.98, 3.01

# # Draw rectangle on main axes for zoom region
# rect = Rectangle((start_step, inset_y_min),
#                  max_step - start_step,
#                  inset_y_max - inset_y_min,
#                  fill=False, edgecolor='black', linewidth=1)
# ax.add_patch(rect)

# # Create inset axes for last 5k steps
# axins = ax.inset_axes([0.1, 0.1, 0.35, 0.35])
# for opt in ['AdamW', 'NAdamW', 'Soap', 'Muon']:
#     subset = df_long[(df_long['Optimizer'] == opt) & (df_long['Step'] >= start_step)]
#     axins.plot(subset['Step'], subset['Loss'], label=opt, color = color_map[opt.lower()], linestyle = line_style[opt.lower()])

# # Inset styling
# axins.set_xlim(start_step, max_step)
# axins.set_ylim(inset_y_min, inset_y_max)
# axins.tick_params(labelsize=18)
# # axins.set_title('', fontsize=18)
# for spine in axins.spines.values():
#     spine.set_edgecolor('black')
#     spine.set_linewidth(1)

# # Draw connecting lines from rectangle corners to inset corners
# con1 = ConnectionPatch(
#     xyA=(start_step, inset_y_max), coordsA=ax.transData,
#     xyB=(1, 1), coordsB=axins.transAxes,
#     arrowstyle='-', linewidth=1, color='black'
# )
# con2 = ConnectionPatch(
#     xyA=(max_step, inset_y_min), coordsA=ax.transData,
#     xyB=(1, 0), coordsB=axins.transAxes,
#     arrowstyle='-', linewidth=1, color='black'
# )
# ax.add_artist(con1)
# ax.add_artist(con2)

# # Add explicit box around the inset axes
# pos = axins.get_position()  # Bbox in figure coords
# # fig.patches.append(
# #     Rectangle((pos.x0, pos.y0),
# #               pos.width + 0.02, pos.height + 0.02,
# #               fill=False, edgecolor='black', linewidth=1,
# #               transform=fig.transFigure, zorder=10)
# # )
ax.set_xticks([0, 20000, 40000])
ax.set_xticklabels(['0B', '20B', '40B'])
# axins.set_xticks([])

plt.tight_layout()
plt.savefig('final2.pdf', bbox_inches='tight')
