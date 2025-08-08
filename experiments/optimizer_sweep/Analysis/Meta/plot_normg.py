from matplotlib import pyplot as plt

losses_csv = "wandb_export_2025-07-03T16_03_37.593-07_00.csv"

import pandas as pd

df = pd.read_csv(losses_csv)
cols = [col for col in df.columns if col.endswith("norm/total")]


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
}

for col in df.columns:
    if col.endswith("norm/total"):
        optimizer = col.split("-")[3]
        for i in range(1, 10):
            if optimizer[:i] in color_map:
                optimizer = optimizer[:i]
                break
        plt.plot(df["Step"], df[col], label=correct_name[optimizer], color=color_map[optimizer])
plt.ylim(0, 4)
plt.legend(fontsize=18, ncol=1, loc="upper right")
plt.xticks([0, 50000, 100000, 150000], ["0", "50k", "100k", "150k"], fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Step", fontsize=20)
plt.ylabel("Gradient Norm", fontsize=20)
plt.title("Gradient Norm For 1.2B Model 8x Chinchilla", fontsize=20)
plt.tight_layout()
plt.savefig("normg.pdf", bbox_inches="tight")



