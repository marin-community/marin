"""Estimate effective data speedup relative to AdamW and generate figures.

This module:
- Loads per-optimizer loss results from `Analysis/Results/`.
- Fits an AdamW baseline scaling curve per model size: L(D) = alpha * D^(-B) + beta.
- Computes each optimizer's effective data budget D_eff achieving its observed loss.
- Produces figures per model size comparing D_eff vs. D_opt (the actual data
  budget), with shaded bands indicating 1.0–1.4x effective speedup.
- Also plots speedup at 8x Chinchilla vs. model size for selected optimizers.

Outputs are written to `experiments/optimizer_sweep/Analysis/figs/` as PDFs.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from plotting_config import color_map, correct_name, line_style
from marin.optimizer_sweep.utils_simp import expected_params

# Set global font size for plots
plt.rcParams.update({"font.size": 20})

RESULTS_DIR_DEFAULT = "experiments/optimizer_sweep/Analysis/Results"
FIGURES_DIR = "experiments/optimizer_sweep/Analysis/figs"


def _extract_loss(payload: dict):
    if isinstance(payload, dict):
        if "min_loss" in payload:
            return payload["min_loss"]
        if isinstance(payload.get("result"), dict) and payload["result"]:
            try:
                return float(min(payload["result"].values()))
            except Exception:
                pass
        baseline = payload.get("baseline") or {}
        if isinstance(baseline, dict) and "loss" in baseline:
            return baseline["loss"]
    return None


def load_results_to_df(results_dir: str = RESULTS_DIR_DEFAULT) -> pd.DataFrame:
    records = []
    if not os.path.isdir(results_dir):
        return pd.DataFrame(columns=["optimizer", "model_size", "chinchilla", "loss"])

    for optimizer in os.listdir(results_dir):
        opt_dir = os.path.join(results_dir, optimizer)
        if not os.path.isdir(opt_dir):
            continue
        for model_size in os.listdir(opt_dir):
            model_dir = os.path.join(opt_dir, model_size)
            if not os.path.isdir(model_dir):
                continue
            for chin_ratio in os.listdir(model_dir):
                chin_dir = os.path.join(model_dir, chin_ratio)
                if not os.path.isdir(chin_dir):
                    continue
                result_path = os.path.join(chin_dir, "result.json")
                if not os.path.exists(result_path):
                    continue
                try:
                    with open(result_path, "r") as f:
                        payload = json.load(f)
                except Exception:
                    continue
                loss_val = _extract_loss(payload)
                if loss_val is None:
                    continue
                try:
                    chin_val = int(chin_ratio)
                except Exception:
                    continue
                records.append(
                    {
                        "optimizer": optimizer.lower(),
                        "model_size": model_size.lower(),
                        "chinchilla": chin_val,
                        "loss": float(loss_val),
                    }
                )

    return pd.DataFrame.from_records(records)


# Load the dataset from Results directory
df = load_results_to_df(RESULTS_DIR_DEFAULT)




optimizers = ["mini", "nadamw", "cautious", "mars", "lion", "muon", "soape", "kron"]



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
    plt.savefig(f"{FIGURES_DIR}/D_eff_vs_D_opt_{model_size}.pdf", bbox_inches="tight")

# Add speedup calculation and plot speedup vs model size
eff_df['speedup'] = eff_df['D_eff'] / eff_df['D_opt']


# Create a new plot for speedup vs model size
fig, ax = plt.subplots(figsize=(10, 6))
eff_df['expected_params'] = eff_df['model_size'].map(expected_params)

# Plot speedup for each optimizer across model sizes
for opt in ['muon', 'soape', 'nadamw']:
    opt_data = eff_df[(eff_df['optimizer'] == opt) & (eff_df['D_opt'] == 8)].sort_values('expected_params')
    if len(opt_data) > 0:
        ax.plot(opt_data['expected_params'], opt_data['speedup'], 
                color=color_map[opt], 
                label=correct_name[opt])
# Add horizontal line at speedup = 1.0 for reference
ax.axhline(y=1.0, color='grey', linestyle=':', alpha=0.7, linewidth=1)
ax.set_xticks(list(expected_params.values()))
ax.set_xticklabels(list(expected_params.keys()))
# Formatting
ax.set_xlabel('Model Size', fontsize=20)
ax.set_ylabel('Speedup', fontsize=20)
ax.set_xscale('log')
ax.set_title('Loss \& Speedup vs Model Size (8x Chinchilla)', fontsize=20)
ax.legend(loc='best', fontsize=16, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/speedup_vs_model_size.pdf", bbox_inches="tight")
plt.show()

