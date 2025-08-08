import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Set global font size for plots
plt.rcParams.update({"font.size": 20})

# Load the dataset
df = pd.read_csv("experiments/optimizer_sweep/Analysis/PhaseII/loss_to_csv.csv")


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

optimizers = ["mini", "nadamw", "cautious", "mars", "lion", "muon", "soape", "kron"]

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
    plt.savefig(f"experiments/optimizer_sweep/Analysis/PhaseII/figs/D_eff_vs_D_opt_{model_size}M.pdf", bbox_inches="tight")

# Add speedup calculation and plot speedup vs model size
eff_df['speedup'] = eff_df['D_eff'] / eff_df['D_opt']


# Create a new plot for speedup vs model size
fig, ax = plt.subplots(figsize=(10, 6))

expected_params = {
    "130m": 134217728,  # 32 * (512*2048*3 + 512*512*4)
    "300m": 301989888,  # 32 * (768*3072*3 + 768*768*4)
    "520m": 536870912,  # 32 * (1024*4096*3 + 1024*1024*4)
    "1.2b": 1207959552,  # 32 * (1536*6144*3 + 1536*1536*4)
}
eff_df['expected_params'] = eff_df['model_size'].map(expected_params)

# Plot speedup for each optimizer across model sizes
for opt in ['muon', 'soape', 'nadamw']:
    opt_data = eff_df[(eff_df['optimizer'] == opt) & (eff_df['D_opt'] == 8)].sort_values('model_size').sort_values('D_opt')
    if len(opt_data) > 0:
        ax.plot(opt_data['expected_params'], opt_data['Loss'], 
                color=color_map[opt], 
                label=correct_name[opt])

# Add horizontal line at speedup = 1.0 for reference
ax.axhline(y=1.0, color='grey', linestyle=':', alpha=0.7, linewidth=1)

# Formatting
ax.set_xlabel('Model Size', fontsize=20)
ax.set_ylabel('Loss', fontsize=20)
ax.set_xscale('log')
ax.set_title('Loss \& Speedup vs Model Size (8x Chinchilla)', fontsize=20)
ax.legend(loc='best', fontsize=16, ncol=2)
ax.grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig("experiments/optimizer_sweep/Analysis/PhaseII/figs/speedup_vs_model_size.pdf", bbox_inches="tight")
plt.show()

eff_df.to_csv("experiments/optimizer_sweep/Analysis/PhaseII/speedup_estimation.csv", index=False)