import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Set global font size for plots
plt.rcParams.update({"font.size": 20})

# Load the dataset
df = pd.read_csv("experiments/optimizer_sweep/Analysis/PhaseIII_1B/loss_to_csv.csv")


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


line_style = {
    "adamw": "--",
    "mars": "--",
    "nadamw": "--",
    "muon": "-",
    "soape": "-",
    "kron": "-",
    "lion": "--",
    "mini": "--",
    "scion": "-",
    "cautious": "--",
}

optimizers = ["nadamw", "muon", "soape"]

print(df)
# map soap to soape

print(df)

# Define the scaling model
def scaling_model(D, alpha, B, beta):
    return alpha * D ** (-B) + beta


# Fit AdamW baseline parameters for each model size
baseline = df[df["optimizer"] == "adamw"]
params = {}
for model_size, group in baseline.groupby("model_size"):
    D = group["chinchilla"].values
    L = group["loss"].values
    p0 = [L[0] - L[-1], 0.5, L[-1]]
    popt, _ = curve_fit(scaling_model, D, L, p0=p0, maxfev=10000)
    params[model_size] = popt

# Compute effective data budgets and actual budgets
records = []
for _, row in df.iterrows():
    model_size = row["model_size"]
    alpha, B, beta = params[model_size]
    D_opt = row["chinchilla"]
    L_opt = row["loss"]
    D_eff = ((L_opt - beta) / alpha) ** (-1.0 / B)
    records.append({"optimizer": row["optimizer"], "model_size": model_size, "D_opt": D_opt, "D_eff": D_eff, "Loss": L_opt})

eff_df = pd.DataFrame(records)
from matplotlib.patches import Patch

# Plot D_eff vs D_opt with a y=x line for each model size
for model_size in sorted(eff_df["model_size"].unique()):
    sub = eff_df[eff_df["model_size"] == model_size]
    fig, ax = plt.subplots(figsize=(8, 6))
    # Shaded speedup bands with patches for legend
    d_min, d_max = sub["D_opt"].min(), sub["D_opt"].max()
    x_vals = np.array([d_min, d_max])

    grey_patch = Patch(facecolor="bisque", alpha=0.5, label=r"1.0–1.2$\times$")
    blue_patch = Patch(facecolor="lightblue", alpha=0.5, label=r"1.2–1.3$\times$")
    green_patch = Patch(facecolor="lightgreen", alpha=0.5, label=r"1.3–1.4$\times$")
    ax.fill_between(x_vals, x_vals, x_vals * 1.2, color="bisque", alpha=0.5)
    ax.fill_between(x_vals, x_vals * 1.2, x_vals * 1.3, color="lightblue", alpha=0.5)
    ax.fill_between(x_vals, x_vals * 1.3, x_vals * 1.4, color="lightgreen", alpha=0.5)
    # Plot optimizer curves and collect handles
    line_handles, line_labels = [], []
    for opt in optimizers:
        data = sub[sub["optimizer"] == opt].sort_values("D_opt")
        (line,) = ax.plot(data["D_opt"], data["D_eff"], marker="o", color=color_map[opt], linestyle=line_style[opt])
        line_handles.append(line)
        line_labels.append(correct_name[opt])

    # First legend: optimizer lines (top left)
    legend_opt = ax.legend(
        handles=line_handles,
        labels=line_labels,
        loc="upper left",
        ncol=2,
        # title='Optimizers',
        frameon=True,
        fontsize=18,
    )
    ax.add_artist(legend_opt)

    # Second legend: speedup bands (bottom right)
    ax.legend(
        handles=[grey_patch, blue_patch, green_patch], loc="lower right", title="Speedup", frameon=True, fontsize=20
    )
    plt.xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    plt.xlabel("Tokens / Chinchilla ", fontsize=20)
    plt.ylabel("Tokens Needed by\n AdamW / Chinchilla", fontsize=20)
    plt.title(f"$D_{{eff}}$ vs $D_{{opt}}$ (Model Size: {model_size.upper()})", fontsize=20)
    # plt.legend(loc='best', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"experiments/optimizer_sweep/Analysis/PhaseIII_1B/figs/D_eff_vs_D_opt_{model_size}M.pdf", bbox_inches="tight")


eff_df['speedup'] = eff_df['D_eff'] / eff_df['D_opt']


# Create a new plot for speedup and loss vs model size
fig, ax1 = plt.subplots(figsize=(10, 7))
another_eff = pd.read_csv("experiments/optimizer_sweep/Analysis/PhaseII/speedup_estimation.csv")
eff_df = pd.concat([eff_df, another_eff])

expected_params = {
    "130m": 134217728,  # 32 * (512*2048*3 + 512*512*4)
    "300m": 301989888,  # 32 * (768*3072*3 + 768*768*4)
    "520m": 536870912,  # 32 * (1024*4096*3 + 1024*1024*4)
    "1.2b": 1207959552,  # 32 * (1536*6144*3 + 1536*1536*4)
}
eff_df['expected_params'] = eff_df['model_size'].map(expected_params)

print(eff_df)

# Create second y-axis for speedup
# ax2 = ax1.twinx()

# Plot loss on left y-axis and speedup on right y-axis
for opt in ['adamw', 'nadamw', 'muon', 'soape']:
    opt_data = eff_df[(eff_df['optimizer'] == opt) & (eff_df['D_opt'] == 8)].sort_values('expected_params')
    print(opt_data)
    if len(opt_data) > 0:
        # # Plot loss on left y-axis (solid lines)
        # line1 = ax1.plot(opt_data['expected_params'], opt_data['Loss'], 
        #                 marker="o", linestyle='-',
        #                 color=color_map[opt], 
        #                 label=f'{correct_name[opt]} (Loss)')
        
        # Plot speedup on right y-axis (dashed lines)
        if opt in ['muon', 'soape', 'nadamw']:
            line2 = ax1.plot(opt_data['expected_params'], opt_data['speedup'], 
                            marker="o", linestyle='--' if opt == 'nadamw' else '-',
                            color=color_map[opt], 
                        label=f'{correct_name[opt]}')
        elif opt == 'adamw':
            line2 = ax1.plot(opt_data['expected_params'], [1.0] * len(opt_data), 
                            marker="s", linestyle='--',
                            color=color_map[opt], 
                            label=f'{correct_name[opt]}')

ax1.set_xscale('log')
ax1.set_xticks(list(expected_params.values()), [x.upper() for x in list(expected_params.keys())])
ax1.tick_params(axis='x', labelsize=24)
# Formatting for left y-axis (Loss)
ax1.set_xlabel('Model Size', fontsize=26)
# Formatting for right y-axis 
ax1.set_ylabel('Speedup w.r.t. AdamW', fontsize=24, color='black')
ax1.tick_params(axis='y', labelcolor='black', labelsize=24)
ax1.set_ylim(0.96, 1.5)

ax1.set_title('Speedup vs Model Size (8x Chinchilla)', fontsize=24)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1, labels1, loc='best', fontsize=22, ncol=2)

ax1.grid(True, alpha=0.3)

# Add text box annotation
plt.text(0.27, 0.76, "Optimizers' speedup w.r.t. AdamW \n decreases with model size",
         transform=ax1.transAxes,
         bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
         verticalalignment='top',
         fontsize=24)

plt.tight_layout()
plt.savefig("experiments/optimizer_sweep/Analysis/PhaseIII_1B/figs/speedup_vs_model_size.pdf", bbox_inches="tight")
plt.show()
