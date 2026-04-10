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

VERSION = "v0.12"
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

    # Separate grid points from controls
    grid = df[df["label"].str.startswith("learning_rate-")].sort_values("learning_rate")
    controls = df[df["label"].isin(("positive-control", "negative-control"))]

    # Evenly-spaced x positions for grid points
    grid_xs = list(range(len(grid)))
    grid_lrs = grid["learning_rate"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot grid points
    ax.plot(grid_xs, grid["eval/loss"].values, color=COLOR_LR, linewidth=1, alpha=0.5, zorder=2)
    ax.scatter(
        grid_xs,
        grid["eval/loss"],
        s=50,
        color=COLOR_LR,
        edgecolors="k",
        linewidths=0.4,
        zorder=3,
        label="LR sweep (transferred)",
    )

    # Plot controls with interpolated x positions — positive first, negative last (for legend order)
    control_order = ["positive-control", "negative-control"]
    control_labels = {"positive-control": "Optimal (transferred)", "negative-control": "Optimal (untransferred)"}
    control_colors = {"positive-control": COLOR_POSITIVE, "negative-control": COLOR_NEGATIVE}
    for ctrl in control_order:
        row = controls[controls["label"] == ctrl].iloc[0]
        x = _interpolate_x(row["learning_rate"], grid_lrs, grid_xs)
        color = control_colors[ctrl]
        ax.scatter(
            x,
            row["eval/loss"],
            s=70,
            color=color,
            edgecolors="k",
            linewidths=0.4,
            zorder=4,
            marker="D",
            label=control_labels[ctrl],
        )
        ax.annotate(
            f"LR={row['learning_rate']:.4g}",
            (x, row["eval/loss"]),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=7,
            ha="left",
            color=color,
        )

    # X-axis: show actual LR values as tick labels
    def _fmt_lr(lr: float) -> str:
        if lr >= 0.001:
            return f"{lr:.4f}".rstrip("0").rstrip(".")
        return f"{lr:.1e}"

    ax.set_xticks(grid_xs)
    ax.set_xticklabels([_fmt_lr(lr) for lr in grid_lrs], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("eval/loss")
    ax.set_title(
        "Bolinas DNA transfer validation — eval/loss vs learning rate\n"
        "Reference: ~25M params, 2.5B tokens → Transfer: ~1.1B params, 10B tokens"
    )
    ax.legend(fontsize=8, loc="upper right")
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
