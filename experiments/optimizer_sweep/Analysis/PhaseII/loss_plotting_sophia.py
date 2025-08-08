import matplotlib.pyplot as plt
import pandas as pd

# Load the loss data
df_loss = pd.read_csv("experiments/optimizer_sweep/Analysis/PhaseII/loss_to_csv.csv")

# The same color map from before
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
    "sophia": "Sophia",
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
    "sophia": "--",
}


# Generate a plot for each model size
for size in df_loss["model_size"].unique():
    data = df_loss[df_loss["model_size"] == size]

    plt.figure()
    for optimizer in ["adamw", "sophia"]:
        subset = data[data["optimizer"] == optimizer]
        plt.plot(
            subset["chinchilla"],
            subset["loss"],
            # label=optimizer,
            color=color_map.get(optimizer, "#000000"),
            linewidth=2,
            label=correct_name.get(optimizer, optimizer),
            linestyle=line_style.get(optimizer, "-"),
        )

    # Labels and styling
    plt.xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    plt.xlabel("Chinchilla Ratio", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.title(f"C4/EN Loss for {size.upper()} Model", fontsize=18)
    plt.legend(fontsize=16, ncol=2)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f"experiments/optimizer_sweep/Analysis/PhaseII/figs/optimizer_loss_scaling_{size}_sophia.pdf", bbox_inches="tight")

# Create a plot comparing model sizes at 1x Chinchilla
plt.figure()
chinchilla_1x = df_loss[df_loss["chinchilla"] == 1]

for optimizer in ["adamw", "sophia"]:
    subset = chinchilla_1x[chinchilla_1x["optimizer"] == optimizer]
    # Sort by model size to ensure correct ordering
    subset = subset.sort_values("model_size")
    
    plt.plot(
        range(len(subset)),
        subset["loss"],
        color=color_map.get(optimizer, "#000000"),
        linewidth=2,
        label=correct_name.get(optimizer, optimizer),
        linestyle=line_style.get(optimizer, "-"),
        marker='o'
    )

# Labels and styling
plt.xticks(range(len(df_loss["model_size"].unique())), 
          [size.upper() for size in sorted(df_loss["model_size"].unique())],
          rotation=0)
plt.xlabel("Model Size", fontsize=18)
plt.ylabel("Loss at 1x Chinchilla", fontsize=18)
plt.title("C4/EN Loss vs Model Size at 1x Chinchilla", fontsize=18)
plt.legend(fontsize=16, ncol=2)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.savefig("experiments/optimizer_sweep/Analysis/PhaseII/figs/optimizer_loss_model_size_sophia.pdf", bbox_inches="tight")
