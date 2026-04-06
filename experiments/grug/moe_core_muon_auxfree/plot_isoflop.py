# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot isoflop curves for the moe-core-isoflop2 sweep.

Plots eval/paloma/c4_en/bpb vs hidden_dim for the two smaller compute budgets (1e18, 3e18).

Usage:
    uv run python experiments/grug/moe_core/plot_isoflop.py
"""

import matplotlib.pyplot as plt

BUDGETS_TO_PLOT = ["1e+18", "3e+18"]

# Hardcoded results from W&B marin-community/dial_moe group=moe-core-isoflop2
# Metric: eval/paloma/c4_en/bpb — all finished runs
DATA = [
    {"budget": "1e+18", "hidden_dim": 512, "bpb": 1.1387},
    {"budget": "1e+18", "hidden_dim": 768, "bpb": 1.1287},
    {"budget": "1e+18", "hidden_dim": 1024, "bpb": 1.1506},
    {"budget": "1e+18", "hidden_dim": 1536, "bpb": 1.2281},
    {"budget": "1e+18", "hidden_dim": 2048, "bpb": 1.3083},
    {"budget": "3e+18", "hidden_dim": 512, "bpb": 1.0892},
    {"budget": "3e+18", "hidden_dim": 768, "bpb": 1.0624},
    {"budget": "3e+18", "hidden_dim": 1024, "bpb": 1.0663},
    {"budget": "3e+18", "hidden_dim": 1536, "bpb": 1.1082},
    {"budget": "3e+18", "hidden_dim": 2048, "bpb": 1.2904},
]


def plot(data):
    fig, ax = plt.subplots(figsize=(8, 5))

    for budget in BUDGETS_TO_PLOT:
        points = sorted(
            [(d["hidden_dim"], d["bpb"]) for d in data if d["budget"] == budget]
        )
        if not points:
            continue
        dims, bpbs = zip(*points)
        ax.plot(dims, bpbs, "o-", label=f"Budget = {budget} FLOPs", markersize=8)

        # Annotate the minimum
        best_idx = bpbs.index(min(bpbs))
        ax.annotate(
            f"d={dims[best_idx]}\n{bpbs[best_idx]:.4f}",
            (dims[best_idx], bpbs[best_idx]),
            textcoords="offset points",
            xytext=(0, -20),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Hidden Dimension", fontsize=12)
    ax.set_ylabel("eval/paloma/c4_en/bpb", fontsize=12)
    ax.set_title("MoE Core Isoflop Curves", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = "experiments/grug/moe_core/isoflop_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    print(f"Plotting {len(DATA)} runs for budgets {BUDGETS_TO_PLOT}")
    for d in sorted(DATA, key=lambda x: (x["budget"], x["hidden_dim"])):
        print(f"  {d['budget']}  d={d['hidden_dim']:>5}  bpb={d['bpb']:.4f}")
    print()
    plot(DATA)
