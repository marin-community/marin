# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Analyze isotoken sweep runs: plot eval_loss vs tokens, grouped by params, faceted by token_fraction."""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)

RUN_PREFIX = "plantcad_width_sweep_v1.0"
RESULT_PATH = "experiments/plantcad/results/isotoken_sweep"

# Plotly colors
PLOTLY_RED = "#EF553B"
PLOTLY_GREEN = "#00CC96"


def fit_changepoint_model(log_x: np.ndarray, y: np.ndarray) -> dict | None:
    """Fit a continuous piecewise linear model with one change point in log-space.

    The model is continuous at the changepoint:
      - For x <= cp: y = y_cp + slope1 * (x - cp)
      - For x > cp:  y = y_cp + slope2 * (x - cp)

    Returns dict with: changepoint, slope1, slope2, y_cp, rss
    """
    n = len(log_x)
    if n < 4:
        # Not enough points for a meaningful changepoint model
        return None

    best_rss = np.inf
    best_result = None

    # Sort by log_x to ensure proper ordering
    order = np.argsort(log_x)
    log_x_sorted = log_x[order]
    y_sorted = y[order]

    # Try each possible changepoint (need at least 2 points on each side)
    for cp_idx in range(2, n - 1):
        cp_log = log_x_sorted[cp_idx]

        # Build design matrix for continuous piecewise linear model:
        # y = y_cp + slope1 * (x - cp) for x <= cp
        # y = y_cp + slope2 * (x - cp) for x > cp
        # Rewrite as: y = y_cp + slope1 * (x - cp) * I(x <= cp) + slope2 * (x - cp) * I(x > cp)
        # Parameters: [y_cp, slope1, slope2]

        x_centered = log_x_sorted - cp_log
        left_mask = log_x_sorted <= cp_log
        right_mask = ~left_mask

        # Design matrix: [1, (x-cp)*I_left, (x-cp)*I_right]
        X = np.column_stack(
            [
                np.ones(n),
                x_centered * left_mask,
                x_centered * right_mask,
            ]
        )

        # Least squares fit
        coeffs, _residuals, _rank, _s = np.linalg.lstsq(X, y_sorted, rcond=None)
        y_cp, slope1, slope2 = coeffs

        # Compute predictions and RSS
        y_pred = X @ coeffs
        rss = np.sum((y_sorted - y_pred) ** 2)

        if rss < best_rss:
            best_rss = rss
            best_result = {
                "changepoint_log": cp_log,
                "changepoint": 10**cp_log,
                "slope1": slope1,
                "slope2": slope2,
                "y_cp": y_cp,
                "rss": rss,
                "cp_idx": cp_idx,
            }

    return best_result


def fetch_runs() -> pd.DataFrame:
    """Fetch isotoken sweep runs from W&B."""
    api = wandb.Api()
    runs = api.runs("marin", filters={"display_name": {"$regex": f"^{RUN_PREFIX}"}})

    data = []
    run_names = []
    for run in runs:
        if run.state != "finished":
            continue
        tags = {k: v for t in run.tags if "=" in t for k, v in [t.split("=", 1)]}
        cfg = run.config
        run_names.append(run.name)
        data.append(
            {
                "eval_loss": run.summary.get("eval/plantcad2/loss"),
                "tokens": float(tags.get("tokens", 0)),
                "params": float(tags.get("params", 0)),
                "token_fraction": float(tags.get("token_fraction", 0)),
                "hidden_size": int(tags.get("hidden_size", 0)),
                "batch_size": cfg.get("trainer", {}).get("train_batch_size"),
                "lr": cfg.get("optimizer", {}).get("learning_rate"),
                "beta2": cfg.get("optimizer", {}).get("beta2"),
                "num_layers": cfg.get("model", {}).get("num_layers"),
                "num_heads": cfg.get("model", {}).get("num_heads"),
            }
        )

    df = pd.DataFrame(data, index=run_names)

    # Log rows with missing essential fields before dropping
    essential = ["eval_loss", "tokens", "params", "token_fraction"]
    missing_mask = df[essential].isna().any(axis=1)
    for run_name in df[missing_mask].index:
        row = df.loc[run_name]
        missing = [col for col in essential if pd.isna(row[col])]
        logger.warning(f"Dropping run '{run_name}': missing {missing}")

    return df.dropna(subset=essential).reset_index(drop=True)


def format_count(value: float, suffix: str = "") -> str:
    """Format large numbers with M/B suffix."""
    if value >= 1e9:
        return f"{value/1e9:.1f}B{suffix}"
    return f"{value/1e6:.1f}M{suffix}"


def format_power_of_2_fraction(frac: float) -> str:
    """Format a fraction as 1/2^n (e.g., 0.25 -> '1/4')."""
    if frac == 1.0:
        return "1"
    denom = round(1 / frac)
    return f"1/{denom}"


def build_fit_annotation(slope: float, rss: float, cp_result: dict | None, cp_label_fmt: callable) -> str:
    """Build annotation text for log-linear and changepoint model fits."""
    lines = ["Log-linear model:", f"  slope = {slope:.4f}", f"  RSS = {rss:.2e}"]
    if cp_result is not None:
        lines.extend(
            [
                "",
                "Changepoint model:",
                f"  slope₁ = {cp_result['slope1']:.4f}",
                f"  slope₂ = {cp_result['slope2']:.4f}",
                f"  changepoint = {cp_label_fmt(cp_result['changepoint'])}",
                f"  RSS = {cp_result['rss']:.2e}",
            ]
        )
    return "\n".join(lines)


def _fmt_int(val) -> str:
    return str(int(val)) if pd.notna(val) else "-"


def _fmt_float(val, fmt: str = ".4f") -> str:
    return f"{val:{fmt}}" if pd.notna(val) else "-"


def print_run_table(df: pd.DataFrame) -> None:
    """Print a rich table summarizing experiment configurations."""
    table = Table(title="Isotoken Sweep Experiments")

    table.add_column("Params", justify="right")
    table.add_column("Data", justify="center")
    table.add_column("Tokens", justify="right")
    table.add_column("Hidden", justify="right")
    table.add_column("Layers", justify="right")
    table.add_column("Heads", justify="right")
    table.add_column("Batch", justify="right")
    table.add_column("LR", justify="right")
    table.add_column("β₂", justify="right")
    table.add_column("Loss", justify="right")

    # Sort by params then by token_fraction descending
    sorted_df = df.sort_values(["params", "token_fraction"], ascending=[True, False])

    for _, row in sorted_df.iterrows():
        table.add_row(
            format_count(row["params"]),
            format_power_of_2_fraction(row["token_fraction"]),
            format_count(row["tokens"]),
            _fmt_int(row["hidden_size"]),
            _fmt_int(row.get("num_layers")),
            _fmt_int(row.get("num_heads")),
            _fmt_int(row.get("batch_size")),
            _fmt_float(row.get("lr")),
            _fmt_float(row.get("beta2")),
            _fmt_float(row["eval_loss"]),
        )

    Console().print(table)


def plot_fits(ax, x: np.ndarray, y: np.ndarray, cp_label_fmt: callable) -> None:
    """Plot log-linear and changepoint model fits with annotations."""
    log_x = np.log10(x)

    # Log-linear fit
    slope, intercept = np.polyfit(log_x, y, 1)
    rss_linear = np.sum((y - (slope * log_x + intercept)) ** 2)

    # Changepoint fit
    cp_result = fit_changepoint_model(log_x, y)

    # Plot fit lines
    x_fit = np.logspace(log_x.min(), log_x.max(), 100)
    ax.plot(x_fit, slope * np.log10(x_fit) + intercept, color="gray", ls="--", lw=1.5, alpha=0.6, label="Log-linear")

    if cp_result is not None:
        cp_log, y_cp = cp_result["changepoint_log"], cp_result["y_cp"]
        for x_seg, mask in [(x_fit[np.log10(x_fit) <= cp_log], True), (x_fit[np.log10(x_fit) >= cp_log], False)]:
            slope_seg = cp_result["slope1"] if mask else cp_result["slope2"]
            label = "Changepoint" if mask else None
            ax.plot(
                x_seg,
                y_cp + slope_seg * (np.log10(x_seg) - cp_log),
                color=PLOTLY_GREEN,
                ls="-",
                lw=2,
                alpha=0.9,
                label=label,
            )
        ax.axvline(cp_result["changepoint"], color=PLOTLY_GREEN, ls=":", alpha=0.5, lw=1)

    # Annotation
    ax.text(
        0.98,
        0.98,
        build_fit_annotation(slope, rss_linear, cp_result, cp_label_fmt),
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )


def save_figure(fig, output_path: str) -> None:
    """Save figure as PNG and PDF."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    print(f"Saved: {out} and {out.with_suffix('.pdf')}")


def plot_loss_by_params(df: pd.DataFrame, output_path: str, max_data_fraction: float = 0.5) -> None:
    """Plot eval_loss vs params, faceted by token_fraction, with log-linear and changepoint fits."""
    df = df[df["token_fraction"] <= max_data_fraction]

    fractions = sorted(df["token_fraction"].unique())
    params_list = sorted(df["params"].unique())

    # Map params to sizes (small range: 30-80)
    sizes = np.linspace(30, 80, len(params_list))
    size_map = {p: s for p, s in zip(params_list, sizes, strict=True)}

    # Shared x-axis limits with padding in log space
    xlim = (params_list[0] * 0.8, params_list[-1] * 1.2)

    n_cols = 3
    n_rows = (len(fractions) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), squeeze=False)

    for i, frac in enumerate(fractions):
        ax = axes[divmod(i, n_cols)]
        data = df[df["token_fraction"] == frac].sort_values("params")
        x, y = data["params"].values, data["eval_loss"].values

        ax.plot(x, y, color="gray", alpha=0.4, lw=1)
        for _, row in data.iterrows():
            ax.scatter(row["params"], row["eval_loss"], color="C0", s=size_map[row["params"]], zorder=5)

        plot_fits(ax, x, y, cp_label_fmt=lambda v: format_count(v))

        total_tokens = data["tokens"].iloc[0]
        ax.set_xlabel("Params")
        ax.set_ylabel("Eval Loss" if i % n_cols == 0 else "")
        ax.set_title(f"Data: {frac:.0%} = {format_power_of_2_fraction(frac)} ({format_count(total_tokens, ' tokens')})")
        ax.set_xscale("log")
        ax.set_xlim(xlim)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)

    for i in range(len(fractions), n_rows * n_cols):
        axes[divmod(i, n_cols)].set_visible(False)

    # Size-based legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="C0", markersize=np.sqrt(size_map[p]))
        for p in params_list
    ]
    fig.legend(
        handles, [format_count(p) for p in params_list], title="Params", loc="center left", bbox_to_anchor=(1, 0.5)
    )

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_loss_by_tokens(df: pd.DataFrame, output_path: str, max_data_fraction: float = 0.5) -> None:
    """Plot eval_loss vs tokens, faceted by params, with log-linear and changepoint fits."""
    df = df[df["token_fraction"] <= max_data_fraction]
    params_list = sorted(df["params"].unique())
    fractions = sorted(df["token_fraction"].unique())

    # Map fractions to sizes (small range: 30-80)
    sizes = np.linspace(30, 80, len(fractions))
    size_map = {f: s for f, s in zip(fractions, sizes, strict=True)}

    n_cols = 3
    n_rows = (len(params_list) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), squeeze=False)

    for i, params in enumerate(params_list):
        ax = axes[divmod(i, n_cols)]
        data = df[df["params"] == params].sort_values("tokens")
        x, y = data["tokens"].values, data["eval_loss"].values

        ax.plot(x, y, color="gray", alpha=0.4, lw=1)
        for _, row in data.iterrows():
            ax.scatter(row["tokens"], row["eval_loss"], color="C0", s=size_map[row["token_fraction"]], zorder=5)

        plot_fits(ax, x, y, cp_label_fmt=lambda v: format_count(v, " tok"))

        ax.set_xlabel("Tokens")
        ax.set_ylabel("Eval Loss" if i % n_cols == 0 else "")
        ax.set_title(f"Params: {format_count(params)}")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)

    for i in range(len(params_list), n_rows * n_cols):
        axes[divmod(i, n_cols)].set_visible(False)

    # Size-based legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="C0", markersize=np.sqrt(size_map[f]))
        for f in fractions
    ]
    fig.legend(
        handles, [f"{f:.0%}" for f in fractions], title="Data Fraction", loc="center left", bbox_to_anchor=(1, 0.5)
    )

    plt.tight_layout()
    save_figure(fig, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-params", default=f"{RESULT_PATH}/isotoken_sweep_loss_by_params.png")
    parser.add_argument("--output-tokens", default=f"{RESULT_PATH}/isotoken_sweep_loss_by_tokens.png")
    parser.add_argument("--csv", default=f"{RESULT_PATH}/isotoken_sweep_runs.csv")
    parser.add_argument("--force", action="store_true", help="Force refetch from W&B")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if csv_path.exists() and not args.force:
        print(f"Loading from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("Fetching from W&B...")
        df = fetch_runs()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} runs to {csv_path}")

    # Ensure numeric types for proper sorting
    for col in ["tokens", "params", "token_fraction"]:
        df[col] = pd.to_numeric(df[col])

    print(f"Loaded {len(df)} runs")

    if df.empty:
        print("No runs found. Check RUN_PREFIX or use --force to refetch.")
    else:
        print_run_table(df)
        plot_loss_by_params(df, args.output_params)
        plot_loss_by_tokens(df, args.output_tokens)
