import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("loss_to_csv_mudam.csv")

# Filter for 1x Chinchilla
df_1x = df[df["chinchilla"] == 1]

# Define the target optimizers
optimizers = ['mudam', 'muon', 'adamw', 'soape']
df_1x = df_1x[df_1x["optimizer"].isin(optimizers)]

# Set up visual properties
color_map = {
    "mars": "#1f77b4", "muon": "#ff7f0e", "lion": "#2ca02c", "adamw": "#d62728",
    "nadamw": "#9467bd", "kron": "#8c564b", "scion": "#e377c2", "cautious": "#7f7f7f",
    "soape": "#bcbd22", "mini": "#aec7e8", "muon_adam": "#1f77b4",
}

correct_name = {
    "adamw": "AdamW", "lion": "Lion", "mini": "Adam-Mini", "scion": "Scion",
    "cautious": "Cautious", "mars": "Mars", "nadamw": "NAdamW", "muon": "Muon",
    "soape": "Soap", "kron": "Kron", "mudam": "Shampoo"
}

line_style = {
    "adamw": "--", "mars": "--", "nadamw": "--", "muon": "-", "soape": "-",
    "kron": "-", "lion": "--", "mini": "--", "scion": "-", "cautious": "--", "mudam": "-"
}

# Create the plot
plt.figure(figsize=(8, 6))
for opt in optimizers:
    opt_data = df_1x[df_1x["optimizer"] == opt].copy()
    opt_data["model_size"] = pd.Categorical(opt_data["model_size"], categories=["130m", "300m", "520m"], ordered=True)
    opt_data = opt_data.sort_values("model_size")
    label = correct_name.get(opt, opt)
    color = color_map.get(opt, None)
    linestyle = line_style.get(opt, "-")
    plt.plot(opt_data["model_size"], opt_data["loss"], marker="o", label=label, color=color, linestyle=linestyle)

plt.xlabel("Model Size", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("Scaling of Optimizers at 1x Chinchilla", fontsize=16)
plt.legend(title="Optimizer")
plt.grid(True)
plt.tight_layout()
plt.savefig("figs/loss_plotting_model.png")
