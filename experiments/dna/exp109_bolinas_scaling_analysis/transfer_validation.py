# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Visualize transfer validation sweep results: eval/loss vs learning rate.

Usage:
    uv run --with pandas --with matplotlib \
        experiments/dna/exp109_bolinas_scaling_analysis/transfer_sweep_validation.py [--refresh]

Caches wandb data locally; pass --refresh to re-fetch.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

VERSION = "v0.14"
WANDB_PROJECT = "eric-czech/marin"
WANDB_RUN_PREFIX = f"dna-bolinas-transfer-{VERSION}"
CACHE_PATH = Path(f"/tmp/transfer_{VERSION}_finished.json")
RESULTS_DIR = Path(f"experiments/dna/exp109_bolinas_scaling_analysis/results/transfer/{VERSION}")
PLOT_PATH = RESULTS_DIR / "transfer_sweep_lr.png"
CSV_PATH = RESULTS_DIR / "transfer_sweep_data.csv"


def fetch_data(project: str, run_prefix: str) -> list[dict]:
    import wandb

    api = wandb.Api()
    runs = api.runs(project, filters={"display_name": {"$regex": f"^{run_prefix}"}})
    data = []
    for r in runs:
        if r.state != "finished":
            continue
        optimizer = r.config.get("optimizer", {})
        # Strip version prefix to get short label (e.g. "positive-control", "learning_rate-2")
        label = r.name
        for prefix in (f"{run_prefix}.", f"{run_prefix}-"):
            if label.startswith(prefix):
                # Handle sub-versions like v0.12.2- by stripping the longest match
                rest = label[len(prefix) :]
                if rest[0].isdigit() and "-" in rest:
                    label = rest.split("-", 1)[1]
                else:
                    label = rest
                break
        data.append(
            {
                "name": r.name,
                "label": label,
                "eval/loss": r.summary.get("eval/loss"),
                "eval/macro_loss": r.summary.get("eval/macro_loss"),
                "learning_rate": optimizer.get("learning_rate"),
                "beta1": optimizer.get("beta1"),
                "beta2": optimizer.get("beta2"),
            }
        )
    return data


def load_data(refresh: bool = False) -> pd.DataFrame:
    if not refresh and CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} runs from cache ({CACHE_PATH})")
    else:
        data = fetch_data(WANDB_PROJECT, WANDB_RUN_PREFIX)
        with open(CACHE_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Fetched {len(data)} finished runs, cached to {CACHE_PATH}")
    df = pd.DataFrame(data)
    df["eval/loss"] = pd.to_numeric(df["eval/loss"], errors="coerce")
    df["eval/macro_loss"] = pd.to_numeric(df["eval/macro_loss"], errors="coerce")
    return df


def export_csv(df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df)} rows to {CSV_PATH}")


def _interpolate_x(lr: float, grid_lrs: list[float], grid_xs: list[float]) -> float:
    """Interpolate an LR value into evenly-spaced x positions using log-scale."""
    import math

    log_lr = math.log(lr)
    log_grid = [math.log(g) for g in grid_lrs]
    # Clamp to grid range
    if log_lr <= log_grid[0]:
        return grid_xs[0]
    if log_lr >= log_grid[-1]:
        return grid_xs[-1]
    for i in range(len(log_grid) - 1):
        if log_grid[i] <= log_lr <= log_grid[i + 1]:
            frac = (log_lr - log_grid[i]) / (log_grid[i + 1] - log_grid[i])
            return grid_xs[i] + frac * (grid_xs[i + 1] - grid_xs[i])
    return grid_xs[-1]


COLOR_LR = "#4C72B0"
COLOR_POSITIVE = "#2CA02C"
COLOR_NEGATIVE = "#D62728"


def plot(df: pd.DataFrame) -> None:
    df = df.dropna(subset=["eval/loss", "learning_rate"]).copy()

    # Combine grid points with positive control (both are "transferred")
    is_transferred = df["label"].str.startswith("learning_rate-") | (df["label"] == "positive-control")
    transferred = df[is_transferred].sort_values("learning_rate")
    negative = df[df["label"] == "negative-control"].iloc[0]

    # Evenly-spaced x positions
    xs = list(range(len(transferred)))
    lrs = transferred["learning_rate"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5))

    def _fmt_lr(lr: float) -> str:
        exp = int(f"{lr:.1e}".split("e")[1])
        mantissa = lr / (10**exp)
        return rf"${mantissa:.1f} \times 10^{{{exp}}}$"

    # Plot transferred series line through all points
    ax.plot(xs, transferred["eval/loss"].values, color=COLOR_LR, linewidth=1, alpha=0.5, zorder=2)

    # Plot grid points (circles) and optimal point (diamond) separately
    is_optimal = transferred["label"] == "positive-control"
    grid_mask = ~is_optimal
    ax.scatter(
        [x for x, m in zip(xs, grid_mask, strict=True) if m],
        transferred.loc[grid_mask, "eval/loss"],
        s=50,
        color=COLOR_LR,
        edgecolors="k",
        linewidths=0.4,
        zorder=3,
        label="LR sweep (transferred)",
    )
    opt_x = next(x for x, m in zip(xs, is_optimal, strict=True) if m)
    opt_row = transferred[is_optimal].iloc[0]
    ax.scatter(
        opt_x,
        opt_row["eval/loss"],
        s=70,
        color=COLOR_LR,
        edgecolors="k",
        linewidths=0.4,
        zorder=4,
        marker="D",
        label="Optimal (transferred)",
    )
    ax.annotate(
        f"LR={_fmt_lr(opt_row['learning_rate'])}",
        (opt_x, opt_row["eval/loss"]),
        textcoords="offset points",
        xytext=(8, -7),
        fontsize=8,
        ha="left",
        color=COLOR_LR,
    )

    # Plot negative control separately
    neg_x = _interpolate_x(negative["learning_rate"], lrs, xs)
    ax.scatter(
        neg_x,
        negative["eval/loss"],
        s=70,
        color=COLOR_NEGATIVE,
        edgecolors="k",
        linewidths=0.4,
        zorder=4,
        marker="D",
        label="Optimal (untransferred)",
    )
    ax.annotate(
        f"LR={_fmt_lr(negative['learning_rate'])}",
        (neg_x, negative["eval/loss"]),
        textcoords="offset points",
        xytext=(8, -4),
        fontsize=8,
        ha="left",
        color=COLOR_NEGATIVE,
    )

    # X-axis: show actual LR values as tick labels
    ax.set_xticks(xs)
    ax.set_xticklabels([_fmt_lr(lr) for lr in lrs], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("eval/loss")
    ax.set_title(
        "Bolinas DNA transfer validation — eval/loss vs learning rate\n"
        "Reference: ~25M params, 2.5B tokens → Transfer: ~1.1B params, 10B tokens"
    )
    handles, labels = ax.get_legend_handles_labels()
    for h in handles:
        if h.get_paths()[0].vertices.shape[0] == 5:  # diamond has 5 vertices
            h.set_sizes([30])
        else:
            h.set_sizes([50])
    ax.legend(handles, labels, fontsize=8, loc="upper left")
    ax.annotate(
        "All configurations equal\nexcept for LR",
        xy=(xs[0], transferred.iloc[0]["eval/loss"]),
        xytext=(40, 30),
        textcoords="offset points",
        fontsize=8,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )
    ax.annotate(
        r"Reference-optimal differs in $\eta_0, \epsilon, \beta_2$",
        xy=(neg_x, negative["eval/loss"]),
        xytext=(25, -35),
        textcoords="offset points",
        fontsize=8,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=300)
    fig.savefig(PLOT_PATH.with_suffix(".pdf"))
    print(f"Saved to {PLOT_PATH}")


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    df = load_data(refresh=refresh)
    export_csv(df)
    plot(df)
