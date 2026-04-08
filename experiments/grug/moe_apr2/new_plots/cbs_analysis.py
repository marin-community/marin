# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Critical batch size analysis plots.

Generates two plots:
1. BPB vs batch size for each model dim, with chosen CBS marked
2. CBS vs compute budget with power law fit and projections
"""

import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = "experiments/grug/moe_apr2/new_plots"

# All CBS sweep results: (budget, batch_sizes, bpb_values)
CBS_DATA = {
    "d512": (
        5e17,
        [32, 64, 128, 256, 384, 512, 640, 768, 896],
        [1.107, 1.105, 1.108, 1.120, 1.132, 1.141, 1.160, 1.175, 1.195],
    ),
    "d768": (
        2e18,
        [32, 64, 128, 256, 384, 512, 640, 768, 896],
        [1.033, 1.032, 1.034, 1.038, 1.044, 1.050, 1.058, 1.068, 1.078],
    ),
    "d1024": (
        6.5e18,
        [32, 64, 128, 256, 384, 512, 640, 768, 896],
        [0.973, 0.972, 0.973, 0.975, 0.979, 0.984, 0.985, 0.991, 0.999],
    ),
    "d1280": (
        1.77e19,
        [128, 256, 384, 512, 640, 768, 896, 1024, 1280],
        [0.930, 0.931, 0.933, 0.935, 0.938, 0.941, 0.944, 0.947, 0.954],
    ),
}

# Chosen CBS points per dim
CBS_CHOSEN = {"d512": 256, "d768": 384, "d1024": 512, "d1280": 640}


def main():
    budgets = np.array([CBS_DATA[k][0] for k in ["d512", "d768", "d1024", "d1280"]])
    cbs_points = np.array([CBS_CHOSEN[k] for k in ["d512", "d768", "d1024", "d1280"]])

    # Fit power law: log(CBS) = a * log(C) + b
    a, b = np.polyfit(np.log10(budgets), np.log10(cbs_points), 1)
    coeff = 10**b
    print(f"Fit: CBS = {coeff:.4e} * C^{a:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {"d512": "C0", "d768": "C1", "d1024": "C2", "d1280": "C3"}

    # ---- Left: BPB vs batch size ----
    ax1 = axes[0]
    for name, (budget, bs_list, bpb_list) in CBS_DATA.items():
        color = colors[name]
        ax1.plot(bs_list, bpb_list, "o-", color=color, label=f"{name} ({budget:.0e})", markersize=6)
        cbs_bs = CBS_CHOSEN[name]
        idx = bs_list.index(cbs_bs)
        ax1.plot(cbs_bs, bpb_list[idx], "*", color=color, markersize=15, zorder=5)

    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("eval/paloma/c4_en/bpb")
    ax1.set_title("BPB vs Batch Size (stars = chosen CBS)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- Right: CBS vs compute ----
    ax2 = axes[1]
    ax2.plot(budgets, cbs_points, "o", color="C0", markersize=10, label="Measured CBS")

    C_grid = np.logspace(np.log10(budgets.min()) - 0.3, 23.3, 200)
    cbs_fit = 10 ** (a * np.log10(C_grid) + b)
    ax2.plot(C_grid, cbs_fit, "--", color="C0", alpha=0.5, label=f"CBS = {coeff:.2e} · C^{a:.3f}")

    for c, bs, name in zip(budgets, cbs_points, ["d512", "d768", "d1024", "d1280"]):
        ax2.annotate(f"{name} (bs={bs})", (c, bs), textcoords="offset points", xytext=(8, 5), fontsize=9)

    for C_proj in [1e20, 3e20, 1e21, 1e22, 1e23]:
        proj_cbs = 10 ** (a * np.log10(C_proj) + b)
        ax2.plot(C_proj, proj_cbs, "s", color="C3", markersize=8)
        ax2.annotate(
            f"{C_proj:.0e}: bs={proj_cbs:.0f}",
            (C_proj, proj_cbs),
            textcoords="offset points",
            xytext=(8, -12),
            fontsize=8,
            color="C3",
        )

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Compute Budget (FLOPs)")
    ax2.set_ylabel("Critical Batch Size")
    ax2.set_title("Critical Batch Size vs Compute")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/cbs_full_analysis.png", dpi=150)
    plt.close(fig)
    print("Saved cbs_full_analysis.png")

    # Projections
    for C_proj in [1e20, 3e20, 1e21, 1e22, 1e23]:
        proj = 10 ** (a * np.log10(C_proj) + b)
        print(f"  Projected CBS at {C_proj:.0e}: {proj:.0f}")


if __name__ == "__main__":
    main()
