# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Visualize transfer validation sweep results: eval/loss vs swept hparam.

One plot per swept axis (learning_rate, beta2) alongside positive / negative controls.

Usage:
    uv run --with pandas --with matplotlib \
        experiments/dna/exp109_bolinas_scaling_analysis/transfer_validation.py [--refresh]

Caches wandb data locally; pass --refresh to re-fetch.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

VERSION = "v0.14"
WANDB_PROJECT = "eric-czech/marin"
WANDB_RUN_PREFIX = f"dna-bolinas-transfer-{VERSION}"
CACHE_PATH = Path(f"/tmp/transfer_{VERSION}.json")
RESULTS_DIR = Path(f"experiments/dna/exp109_bolinas_scaling_analysis/results/transfer/{VERSION}")
PLOT_LR_PATH = RESULTS_DIR / "transfer_sweep_lr.png"
PLOT_BETA2_PATH = RESULTS_DIR / "transfer_sweep_beta2.png"
CSV_PATH = RESULTS_DIR / "transfer_sweep_data.csv"

# Runs still "running" in wandb but within this many steps of num_train_steps are
# treated as effectively complete. Covers the common case where the final eval
# step lands just before the run flushes to "finished".
STEP_TOLERANCE = 10


def fetch_data(project: str, run_prefix: str) -> list[dict]:
    import wandb

    api = wandb.Api()
    runs = api.runs(project, filters={"display_name": {"$regex": f"^{run_prefix}"}})
    data = []
    for r in runs:
        optimizer = r.config.get("optimizer", {})
        trainer = r.config.get("trainer", {})
        # Strip version prefix to get short label (e.g. "positive-control", "learning_rate-2", "beta2-0")
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
                "state": r.state,
                "step": r.summary.get("_step"),
                "num_train_steps": trainer.get("num_train_steps"),
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
        print(f"Fetched {len(data)} runs, cached to {CACHE_PATH}")
    df = pd.DataFrame(data)
    df["eval/loss"] = pd.to_numeric(df["eval/loss"], errors="coerce")
    df["eval/macro_loss"] = pd.to_numeric(df["eval/macro_loss"], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["num_train_steps"] = pd.to_numeric(df["num_train_steps"], errors="coerce")
    return df


def filter_completed(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only runs that are state=finished or have reached num_train_steps (within tolerance).

    wandb occasionally leaves a run in 'running' for a short window after the final
    step completes; STEP_TOLERANCE catches those without pulling in genuinely
    in-flight runs.
    """
    is_finished = df["state"] == "finished"
    reached_end = (
        df["step"].notna() & df["num_train_steps"].notna() & (df["step"] >= df["num_train_steps"] - STEP_TOLERANCE)
    )
    keep = is_finished | reached_end
    dropped = df[~keep]
    if not dropped.empty:
        summary = ", ".join(
            f"{row['label']}(state={row['state']}, step={row['step']}/{row['num_train_steps']})"
            for _, row in dropped.iterrows()
        )
        print(f"Dropping {len(dropped)} incomplete run(s): {summary}")
    return df[keep].copy()


def export_csv(df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df)} rows to {CSV_PATH}")


def _interpolate_x(value: float, grid_values: list[float], grid_xs: list[float], log_scale: bool) -> float:
    """Interpolate a hparam value into evenly-spaced x positions; extrapolates outside the grid."""
    if log_scale:
        t = math.log(value)
        grid_t = [math.log(g) for g in grid_values]
    else:
        t = value
        grid_t = list(grid_values)
    if t <= grid_t[0]:
        slope = (grid_xs[1] - grid_xs[0]) / (grid_t[1] - grid_t[0])
        return grid_xs[0] + slope * (t - grid_t[0])
    if t >= grid_t[-1]:
        slope = (grid_xs[-1] - grid_xs[-2]) / (grid_t[-1] - grid_t[-2])
        return grid_xs[-1] + slope * (t - grid_t[-1])
    for i in range(len(grid_t) - 1):
        if grid_t[i] <= t <= grid_t[i + 1]:
            frac = (t - grid_t[i]) / (grid_t[i + 1] - grid_t[i])
            return grid_xs[i] + frac * (grid_xs[i + 1] - grid_xs[i])
    return grid_xs[-1]


COLOR_SWEEP = "#4C72B0"
COLOR_NEGATIVE = "#D62728"


def _fmt_lr(lr: float) -> str:
    exp = int(f"{lr:.1e}".split("e")[1])
    mantissa = lr / (10**exp)
    return rf"${mantissa:.1f} \times 10^{{{exp}}}$"


def _fmt_beta2(b: float) -> str:
    return f"{b:.4f}"


def _plot_axis(
    df: pd.DataFrame,
    *,
    axis_field: str,
    label_prefix: str,
    axis_label: str,
    log_scale: bool,
    value_formatter,
    plot_path: Path,
    equal_except_annotation: str,
    negative_annotation: str,
) -> None:
    """Render a single-axis sweep plot (sweep series + positive + negative control)."""
    df = df.dropna(subset=["eval/loss", axis_field]).copy()

    is_transferred = df["label"].str.startswith(f"{label_prefix}-") | (df["label"] == "positive-control")
    transferred = df[is_transferred].sort_values(axis_field)
    negative = df[df["label"] == "negative-control"].iloc[0]

    xs = list(range(len(transferred)))
    vals = transferred[axis_field].tolist()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(xs, transferred["eval/loss"].values, color=COLOR_SWEEP, linewidth=1, alpha=0.5, zorder=2)

    is_optimal = transferred["label"] == "positive-control"
    grid_mask = ~is_optimal
    ax.scatter(
        [x for x, m in zip(xs, grid_mask, strict=True) if m],
        transferred.loc[grid_mask, "eval/loss"],
        s=50,
        color=COLOR_SWEEP,
        edgecolors="k",
        linewidths=0.4,
        zorder=3,
        label=f"{label_prefix} sweep (transferred)",
    )
    opt_x = next(x for x, m in zip(xs, is_optimal, strict=True) if m)
    opt_row = transferred[is_optimal].iloc[0]
    ax.scatter(
        opt_x,
        opt_row["eval/loss"],
        s=70,
        color=COLOR_SWEEP,
        edgecolors="k",
        linewidths=0.4,
        zorder=4,
        marker="D",
        label="Optimal (transferred)",
    )
    ax.annotate(
        f"{axis_label}={value_formatter(opt_row[axis_field])}",
        (opt_x, opt_row["eval/loss"]),
        textcoords="offset points",
        xytext=(8, -7),
        fontsize=8,
        ha="left",
        color=COLOR_SWEEP,
    )

    neg_x = _interpolate_x(negative[axis_field], vals, xs, log_scale=log_scale)
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
        f"{axis_label}={value_formatter(negative[axis_field])}",
        (neg_x, negative["eval/loss"]),
        textcoords="offset points",
        xytext=(8, -4),
        fontsize=8,
        ha="left",
        color=COLOR_NEGATIVE,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([value_formatter(v) for v in vals], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel(axis_label)
    ax.set_ylabel("eval/loss")
    ax.set_title(
        f"Bolinas DNA transfer validation — eval/loss vs {axis_label}\n"
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
        equal_except_annotation,
        xy=(xs[0], transferred.iloc[0]["eval/loss"]),
        xytext=(40, 30),
        textcoords="offset points",
        fontsize=8,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )
    ax.annotate(
        negative_annotation,
        xy=(neg_x, negative["eval/loss"]),
        xytext=(25, -35),
        textcoords="offset points",
        fontsize=8,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300)
    fig.savefig(plot_path.with_suffix(".pdf"))
    print(f"Saved to {plot_path}")


def plot_lr(df: pd.DataFrame) -> None:
    _plot_axis(
        df,
        axis_field="learning_rate",
        label_prefix="learning_rate",
        axis_label="learning_rate",
        log_scale=True,
        value_formatter=_fmt_lr,
        plot_path=PLOT_LR_PATH,
        equal_except_annotation="All configurations equal\nexcept for LR",
        negative_annotation=r"Reference-optimal differs in $\eta, \eta_0, \epsilon$",
    )


def plot_beta2(df: pd.DataFrame) -> None:
    _plot_axis(
        df,
        axis_field="beta2",
        label_prefix="beta2",
        axis_label=r"$\beta_2$",
        log_scale=False,
        value_formatter=_fmt_beta2,
        plot_path=PLOT_BETA2_PATH,
        equal_except_annotation=r"All configurations equal" "\n" r"except for $\beta_2$",
        negative_annotation=r"Reference-optimal differs in $\eta, \eta_0, \epsilon$",
    )


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    df = load_data(refresh=refresh)
    df = filter_completed(df)
    export_csv(df)
    plot_lr(df)
    plot_beta2(df)
