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

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.interpolate import griddata
import wandb

RUN_VERSION = "1.9"
RUN_PREFIX = f"plantcad_isoflop_v{RUN_VERSION}"
RESULT_PATH = f"experiments/plantcad/results/v{RUN_VERSION}"
EXPORT_DPI = 300
DEFAULT_ARCH = "qwen"

console = Console(record=True)
logger = logging.getLogger(__name__)


def setup_logging(log_path: Path) -> None:
    """Configure logging to both console and file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Clear any existing handlers
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    # File handler
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def filter_to_finished_runs(df: pd.DataFrame, allow_crashed: bool = False) -> pd.DataFrame:
    """Filter dataframe to include finished runs and nearly-complete crashed runs.

    Includes runs where:
    - state == "finished", OR
    - state == "crashed" AND run_progress > 0.999
    """
    is_finished = df["state"] == "finished"
    if allow_crashed:
        is_nearly_complete_crash = (df["state"] == "crashed") & (df["run_progress"] > 0.999)
        return df[is_finished | is_nearly_complete_crash]
    else:
        return df[is_finished]


EXPLODED_RUNS: dict[str, list[str]] = {}
EXPLODED_BUDGETS: dict[str, list[float]] = {
    "1.9": [1.0e16],
}


def filter_exploded_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out runs where training exploded (by run name or budget)."""
    exploded_runs = EXPLODED_RUNS.get(RUN_VERSION, [])
    exploded_budgets = EXPLODED_BUDGETS.get(RUN_VERSION, [])

    # Filter by run name
    run_mask = df["run_name"].isin(exploded_runs)
    for run_name in df.loc[run_mask, "run_name"]:
        logger.warning(f"Filtering exploded run: {run_name}")

    # Filter by budget
    budget_mask = df["flops_budget"].isin(exploded_budgets)
    n_budget_filtered = budget_mask.sum()
    if n_budget_filtered > 0:
        filtered_budgets = df.loc[budget_mask, "flops_budget"].unique()
        for budget in filtered_budgets:
            budget_runs = df.loc[df["flops_budget"] == budget, "run_name"].tolist()
            logger.warning(f"Filtering {len(budget_runs)} runs at exploded budget {budget:.1e}: {budget_runs}")

    return df[~run_mask & ~budget_mask]


def save_figure(fig, output_path: str) -> None:
    """Save figure as both PNG and PDF at EXPORT_DPI resolution."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save PNG
    fig.savefig(output_path, dpi=EXPORT_DPI, bbox_inches="tight")
    logger.info(f"Saved plot to {output_path}")

    # Save PDF
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=EXPORT_DPI, bbox_inches="tight")
    logger.info(f"Saved plot to {pdf_path}")


def log_run_object(run, run_idx):
    """Log a run object as JSON to show available data."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"RUN {run_idx + 1}: {run.name}")
    logger.info(f"{'=' * 80}")
    run_dict = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "tags": run.tags,
        "config": dict(run.config),
        "summary": dict(run.summary),
    }
    logger.info(json.dumps(run_dict, indent=2, default=str))
    logger.info(f"{'=' * 80}\n")


def fetch_plantcad_runs(show_wandb_runs: bool = False):
    """Fetch plantcad isoflop runs and extract metrics/tags into a dataframe."""
    api = wandb.Api()
    # Note: Results from the first run (plantcad_isoflop_01) are available at:
    # https://github.com/marin-community/marin/issues/2101#issuecomment-3581675724
    runs = api.runs(
        "marin",
        filters={"display_name": {"$regex": f"^{RUN_PREFIX}"}},
    )

    data = []
    for idx, run in enumerate(runs):
        # Log first 2 runs in detail
        if show_wandb_runs and idx < 2:
            log_run_object(run, idx)

        # Parse tags like "batch_size=32"
        tags_dict = {}
        for tag in run.tags:
            if "=" in tag:
                key, value = tag.split("=", 1)
                try:
                    # Try to convert to appropriate type
                    if "." in value or "e+" in value or "e-" in value:
                        tags_dict[key] = float(value)
                    else:
                        tags_dict[key] = int(value)
                except ValueError:
                    tags_dict[key] = value

        # Calculate execution time
        start_time = pd.to_datetime(run.created_at) if run.created_at else None
        stop_time = pd.to_datetime(run.summary.get("_timestamp"), unit="s") if run.summary.get("_timestamp") else None

        # Handle timezone differences
        if start_time and stop_time:
            if start_time.tzinfo and not stop_time.tzinfo:
                stop_time = stop_time.tz_localize("UTC")
            elif stop_time.tzinfo and not start_time.tzinfo:
                start_time = start_time.tz_localize("UTC")
            duration = (stop_time - start_time).total_seconds()
        else:
            duration = None

        row = {
            "run_name": run.name,
            "state": run.state,
            "start_time": start_time,
            "stop_time": stop_time,
            "duration_sec": duration,
            # Metrics
            "eval_loss": run.summary.get("eval/plantcad2/loss"),
            "train_loss": run.summary.get("train/loss"),
            "total_gflops": run.summary.get("throughput/total_gflops"),
            "total_tokens": run.summary.get("throughput/total_tokens"),
            "run_progress": run.summary.get("run_progress"),
            # Tags
            "architecture": tags_dict.get("architecture"),
            "batch_size": tags_dict.get("batch_size"),
            "flops_budget": tags_dict.get("flops_budget"),
            "hidden_size": tags_dict.get("hidden_size"),
            "num_layers": tags_dict.get("num_layers"),
            "params": tags_dict.get("params"),
            "steps": tags_dict.get("steps"),
            "tokens": tags_dict.get("tokens"),
            "tpu": tags_dict.get("tpu"),
            "epochs": tags_dict.get("epochs"),
            # Config
            "hf_save_path": run.config.get("hf_save_path"),
        }
        data.append(row)

    return pd.DataFrame(data)


def save_runs(df, output_path=f"{RESULT_PATH}/plantcad_isoflops.csv"):
    """Save dataframe to CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} runs to {output_path}")


def validate_runs(df):
    """Validate that rows are unique by key columns."""
    key_cols = ["architecture", "flops_budget", "tokens", "params", "epochs"]
    duplicates = df[df.duplicated(subset=key_cols, keep=False)]
    if not duplicates.empty:
        logger.warning(f"Found {len(duplicates)} duplicate rows by {key_cols}:")
        logger.warning(duplicates[["run_name", *key_cols]].to_string())
    else:
        logger.info(f"Validation passed: rows are unique by {key_cols}")


def summarize_runs(df):
    """Print formatted summary tables using rich."""
    gflops_to_flops = 1e9

    # Run summary table
    run_summary_cols = [
        "run_name",
        "state",
        "flops_budget",
        "architecture",
        "params",
        "tokens",
        "epochs",
        "eval_loss",
        "run_progress",
    ]
    summary_table = Table(title="Run Summary", show_header=True, header_style="bold cyan")
    for col in run_summary_cols:
        summary_table.add_column(col)
    summary = df[run_summary_cols].copy()
    for _, row in summary.sort_values(["flops_budget", "architecture", "epochs"]).iterrows():
        summary_table.add_row(*[str(v) if pd.notna(v) else "" for v in row])
    console.print(summary_table)

    # Checkpoint summary table - best runs per (flops_budget, architecture, epochs)
    ckpt_cols = ["run_name", "flops_budget", "architecture", "epochs", "eval_loss", "hf_save_path"]
    group_cols = ["flops_budget", "architecture", "epochs"]
    # Find min eval_loss per group and keep all rows matching that min
    df_with_min = df.merge(
        df.groupby(group_cols)["eval_loss"].min().reset_index().rename(columns={"eval_loss": "min_eval_loss"}),
        on=group_cols,
    )
    best_runs = df_with_min[df_with_min["eval_loss"] == df_with_min["min_eval_loss"]][ckpt_cols].copy()
    ckpt_table = Table(
        title="Checkpoint Summary (Best per Budget/Arch/Epochs)", show_header=True, header_style="bold cyan"
    )
    for col in ckpt_cols:
        ckpt_table.add_column(col)
    for _, row in best_runs.sort_values(group_cols).iterrows():
        ckpt_table.add_row(*[str(v) if pd.notna(v) else "" for v in row])
    console.print(ckpt_table)

    # FLOPs summary table
    flops_table = Table(title="FLOPs Summary", show_header=True, header_style="bold cyan")
    flops_table.add_column("Compute Budget", style="bold")
    flops_table.add_column("Runs", justify="right")
    flops_table.add_column("Budget (FLOPs)", justify="right")
    flops_table.add_column("Actual (FLOPs)", justify="right")

    for budget, grp in df.groupby("flops_budget", sort=True):
        flops_table.add_row(
            f"{budget:.1e}",
            str(len(grp)),
            f"{grp['flops_budget'].sum():.3e}",
            f"{grp['total_gflops'].sum() * gflops_to_flops:.3e}",
        )
    flops_table.add_section()
    flops_table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{len(df)}[/bold]",
        f"[bold]{df['flops_budget'].sum():.3e}[/bold]",
        f"[bold]{df['total_gflops'].sum() * gflops_to_flops:.3e}[/bold]",
    )
    console.print(flops_table)


def visualize_loss_by_token_count(df, metric="eval_loss", output_path=f"{RESULT_PATH}/plantcad_loss_by_tokens.png"):
    """Plot loss vs tokens, colored by budget, faceted by architecture (cols) and epochs (rows)."""
    required_cols = [metric, "tokens", "architecture", "flops_budget", "epochs"]
    df_clean = filter_to_finished_runs(df).dropna(subset=required_cols)

    if df_clean.empty:
        logger.warning(f"No finished runs with required columns {required_cols}. Skipping visualization.")
        return

    architectures = sorted(df_clean["architecture"].unique())
    budgets = sorted(df_clean["flops_budget"].unique())
    unique_epochs = sorted(df_clean["epochs"].unique())

    # Create budget colormap
    cmap = plt.get_cmap("viridis")
    budget_colors = {b: cmap(i / max(1, len(budgets) - 1)) for i, b in enumerate(budgets)}

    # Get global x-limits and per-epoch y-limits
    x_min, x_max = df_clean["tokens"].min(), df_clean["tokens"].max()
    epoch_ylims = {}
    for epoch in unique_epochs:
        epoch_data = df_clean[df_clean["epochs"] == epoch]
        y_min, y_max = epoch_data[metric].min(), epoch_data[metric].max()
        y_padding = (y_max - y_min) * 0.1
        epoch_ylims[epoch] = (y_min - y_padding, y_max + y_padding)

    fig, axes = plt.subplots(
        len(unique_epochs), len(architectures), figsize=(5 * len(architectures), 3 * len(unique_epochs)), squeeze=False
    )

    for ei, epoch in enumerate(unique_epochs):
        for ai, arch in enumerate(architectures):
            ax = axes[ei, ai]
            for budget in budgets:
                data = df_clean[
                    (df_clean["architecture"] == arch)
                    & (df_clean["flops_budget"] == budget)
                    & (df_clean["epochs"] == epoch)
                ].sort_values("tokens")
                if data.empty:
                    continue
                color = budget_colors[budget]
                ax.plot(data["tokens"], data[metric], alpha=0.7, linewidth=1.5, color=color)
                ax.scatter(data["tokens"], data[metric], alpha=0.8, color=color, s=30)
            ax.set_xlabel("Token Count")
            ax.set_ylabel("Validation Loss")
            ax.set_title(f"{arch} | {int(epoch)} Ep")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(epoch_ylims[epoch])
            ax.grid(alpha=0.3)

    # Create legend for budget colors
    handles = [
        plt.Line2D([0], [0], color=budget_colors[b], marker="o", linestyle="-", label=f"{b:.1e}") for b in budgets
    ]
    fig.legend(handles, [f"{b:.1e}" for b in budgets], title="Budget", loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    save_figure(fig, output_path)


def visualize_loss_by_param_count(df, metric="eval_loss", output_path=f"{RESULT_PATH}/plantcad_loss_by_params.png"):
    """Plot loss vs params, colored by budget, faceted by architecture (cols) and epochs (rows)."""
    required_cols = [metric, "params", "architecture", "flops_budget", "epochs"]
    df_clean = filter_to_finished_runs(df).dropna(subset=required_cols)

    if df_clean.empty:
        logger.warning(f"No finished runs with required columns {required_cols}. Skipping visualization.")
        return

    architectures = sorted(df_clean["architecture"].unique())
    budgets = sorted(df_clean["flops_budget"].unique())
    unique_epochs = sorted(df_clean["epochs"].unique())

    # Create budget colormap
    cmap = plt.get_cmap("viridis")
    budget_colors = {b: cmap(i / max(1, len(budgets) - 1)) for i, b in enumerate(budgets)}

    # Get global x-limits and per-epoch y-limits
    x_min, x_max = df_clean["params"].min(), df_clean["params"].max()
    epoch_ylims = {}
    for epoch in unique_epochs:
        epoch_data = df_clean[df_clean["epochs"] == epoch]
        y_min, y_max = epoch_data[metric].min(), epoch_data[metric].max()
        y_padding = (y_max - y_min) * 0.1
        epoch_ylims[epoch] = (y_min - y_padding, y_max + y_padding)

    fig, axes = plt.subplots(
        len(unique_epochs), len(architectures), figsize=(5 * len(architectures), 3 * len(unique_epochs)), squeeze=False
    )

    for ei, epoch in enumerate(unique_epochs):
        for ai, arch in enumerate(architectures):
            ax = axes[ei, ai]
            for budget in budgets:
                data = df_clean[
                    (df_clean["architecture"] == arch)
                    & (df_clean["flops_budget"] == budget)
                    & (df_clean["epochs"] == epoch)
                ].sort_values("params")
                if data.empty:
                    continue
                color = budget_colors[budget]
                ax.plot(data["params"], data[metric], alpha=0.7, linewidth=1.5, color=color)
                ax.scatter(data["params"], data[metric], alpha=0.8, color=color, s=30)
            ax.set_xlabel("Param Count")
            ax.set_ylabel("Validation Loss")
            ax.set_title(f"{arch} | {int(epoch)} Ep")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(epoch_ylims[epoch])
            ax.grid(alpha=0.3)

    # Create legend for budget colors
    handles = [
        plt.Line2D([0], [0], color=budget_colors[b], marker="o", linestyle="-", label=f"{b:.1e}") for b in budgets
    ]
    fig.legend(handles, [f"{b:.1e}" for b in budgets], title="Budget", loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    save_figure(fig, output_path)


def get_size_label(rank: int) -> str:
    """Map model size rank to human-readable label."""
    labels = ["XXS", "XS", "S", "M", "L", "XL", "XXL", "XXXL"]
    if rank < len(labels):
        return labels[rank]
    return f"Size-{rank}"


def visualize_loss_by_epochs(
    df,
    metric: str = "eval_loss",
    output_path: str = f"{RESULT_PATH}/plantcad_loss_by_epochs.png",
) -> None:
    """Plot normalized loss vs epochs, faceted by architecture and budget."""
    required_cols = [metric, "tokens", "params", "architecture", "flops_budget", "epochs"]
    df_clean = filter_to_finished_runs(df).dropna(subset=required_cols).copy()

    architectures = sorted(df_clean["architecture"].unique())
    budgets = sorted(df_clean["flops_budget"].unique(), reverse=True)
    if not architectures or not budgets:
        logger.warning("No data to visualize.")
        return

    # Normalize loss to 0-1 per (arch, budget, tokens, params) group
    group_cols = ["architecture", "flops_budget", "tokens", "params"]
    df_clean["loss_norm"] = df_clean.groupby(group_cols)[metric].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
    )

    # Create model size rank based on params (within each budget)
    df_clean["size_rank"] = df_clean.groupby("flops_budget")["params"].transform(
        lambda x: x.rank(method="dense").astype(int) - 1
    )
    unique_ranks = sorted(df_clean["size_rank"].unique())
    n_ranks = len(unique_ranks)
    cmap = plt.get_cmap("tab10" if n_ranks <= 10 else "tab20")
    rank_colors = {r: cmap(r / max(1, n_ranks - 1)) if n_ranks > 1 else cmap(0) for r in unique_ranks}

    fig, axes = plt.subplots(
        len(budgets), len(architectures), figsize=(5 * len(architectures), 2 * len(budgets)), squeeze=False
    )

    for bi, budget in enumerate(budgets):
        for ai, arch in enumerate(architectures):
            ax = axes[bi, ai]
            df_facet = df_clean[(df_clean["architecture"] == arch) & (df_clean["flops_budget"] == budget)]
            if df_facet.empty:
                ax.set_visible(False)
                continue

            # Get unique (tokens, params, size_rank) combos for this facet
            combos = (
                df_facet.groupby(["tokens", "params", "size_rank"])
                .size()
                .reset_index()[["tokens", "params", "size_rank"]]
            )
            combos = combos.sort_values(["size_rank", "tokens"])

            for _, row in combos.iterrows():
                tokens, params, size_rank = row["tokens"], row["params"], row["size_rank"]
                data = df_facet[(df_facet["tokens"] == tokens) & (df_facet["params"] == params)].sort_values("epochs")
                if data.empty:
                    continue
                color = rank_colors[size_rank]
                ax.plot(data["epochs"], data["loss_norm"], color=color, alpha=0.7, linewidth=1.5)
                ax.scatter(data["epochs"], data["loss_norm"], color=color, s=30, zorder=5)

            ax.set_xlabel("Epochs")
            ax.set_ylabel("Normalized Loss (0-1)")
            ax.set_title(f"{arch} | C={budget:.1e}")
            ax.set_xscale("log", base=2)
            ax.set_xlim(df_clean["epochs"].min(), df_clean["epochs"].max())
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)

    # Create legend for model size ranks
    handles = [plt.Line2D([0], [0], color=rank_colors[r], marker="o", linestyle="-") for r in unique_ranks]
    labels = [get_size_label(r) for r in unique_ranks]
    fig.legend(handles, labels, title="Model Size", loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    save_figure(fig, output_path)


def visualize_loss_by_param_and_epoch_count(
    df,
    architecture: str = DEFAULT_ARCH,
    metric: str = "eval_loss",
    clip_percentile: float = 80.0,
    output_path: str = f"{RESULT_PATH}/plantcad_loss_contour.png",
) -> None:
    """2D contour plot of loss vs params (y) and epochs (x), faceted by flops budget."""
    required_cols = [metric, "params", "epochs", "architecture", "flops_budget"]
    df_clean = filter_to_finished_runs(df[df["architecture"] == architecture]).dropna(subset=required_cols).copy()

    if df_clean.empty:
        logger.warning(f"No data for architecture '{architecture}'")
        return

    n_unique_epochs = df_clean["epochs"].nunique()
    if n_unique_epochs < 2:
        logger.warning(f"Cannot create contour plot: need at least 2 unique epoch values, but got {n_unique_epochs}")
        return

    df_clean["log_loss"] = np.log2(df_clean[metric])
    df_clean["log_loss"] = df_clean["log_loss"].clip(upper=df_clean["log_loss"].quantile(clip_percentile / 100))

    budgets = sorted(df_clean["flops_budget"].unique())
    n_budgets = len(budgets)

    # Compute global color scale across all budgets
    global_min = df_clean["log_loss"].min()
    global_max = df_clean["log_loss"].max()
    levels = np.linspace(global_min, global_max, 50)

    fig, axes = plt.subplots(1, n_budgets, figsize=(2.5 * n_budgets, 3.4), squeeze=False)
    axes = axes[0]

    contour = None

    for idx, budget in enumerate(budgets):
        ax = axes[idx]
        df_budget = df_clean[df_clean["flops_budget"] == budget]

        # Interpolate scattered points to a finer grid in log space
        x_data = np.log(df_budget["epochs"].values)
        y_data = np.log(df_budget["params"].values)
        z_data = df_budget["log_loss"].values

        xi = np.geomspace(df_budget["epochs"].min(), df_budget["epochs"].max(), 200)
        yi = np.geomspace(df_budget["params"].min(), df_budget["params"].max(), 200)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        zi = griddata((x_data, y_data), z_data, (np.log(xi_grid), np.log(yi_grid)), method="cubic")

        contour = ax.contourf(xi_grid, yi_grid, zi, levels=levels, cmap="viridis", antialiased=True, extend="both")

        # Only show raw data points for the lowest compute budget
        if idx == 0:
            ax.scatter(df_budget["epochs"], df_budget["params"], c="black", s=22, alpha=0.15, edgecolors="none")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.margins(0)
        ax.set_ylabel("Params" if idx == 0 else "")
        ax.set_title(f"C = {budget:.1e}")
        if idx > 0:
            ax.set_yticklabels([])

    # Add colorbar in dedicated axes on the right, with top/bottom margin for labels
    fig.subplots_adjust(right=0.92, top=0.82, bottom=0.15)

    # Single shared x-axis label
    fig.supxlabel("Epochs")
    fig.suptitle(f"Validation Loss ({architecture})", fontsize=14)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour, cax=cbar_ax, label="log₂(Loss)")
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))

    save_figure(fig, output_path)


# ------------------------------------------------------------
# Isoflop scaling law analysis (from exp2101_plantcad_isoflop_fit.py)
# ------------------------------------------------------------


def fit_quadratic_optimum(x_vals, loss_vals):
    """
    Fits L = a*(ln x)^2 + b*(ln x) + c and returns optimal x.
    Returns (x_opt, fit_coeffs).
    Raises ValueError if the fit is concave (no minimum).
    """
    log_x = np.log(x_vals)
    coeffs = np.polyfit(log_x, loss_vals, 2)  # [a, b, c]
    a, b, _ = coeffs

    if a <= 0:
        raise ValueError(f"Concave fit detected: {coeffs}")

    # Minimum at ln x = -b / (2a)
    ln_x_opt = -b / (2 * a)
    x_opt = np.exp(ln_x_opt)
    return x_opt, coeffs


def analyze_budgets(df):
    """
    Analyzes each budget group to find optimal N and D using independent quadratic fits.
    Returns a DataFrame with columns: budget, opt_N, opt_D, coeffs_N, coeffs_D, group_data.
    """
    budgets = sorted(df["flops_budget"].unique())
    results = []

    for budget in budgets:
        group = df[df["flops_budget"] == budget].sort_values("params")
        if len(group) < 3:
            print(f"Skipping budget {budget}: has fewer than 3 points ({len(group)}), cannot fit.")
            continue

        N = group["params"].values
        D = group["tokens"].values
        L = group["eval_loss"].values

        # Fit independent quadratics
        opt_N, coeffs_N = fit_quadratic_optimum(N, L)
        opt_D, coeffs_D = fit_quadratic_optimum(D, L)

        results.append(
            {
                "budget": budget,
                "opt_N": opt_N,
                "opt_D": opt_D,
                "coeffs_N": coeffs_N,
                "coeffs_D": coeffs_D,
                "group_data": group,
            }
        )

    return pd.DataFrame(results)


def fit_scaling_law(budgets, optimal_vals):
    """
    Fits log(optimal_val) = m * log(budget) + c.
    Returns (m, c, B_smooth, V_smooth) where B_smooth and V_smooth are smoothed
    budget and predicted value arrays for plotting.
    """
    log_B = np.log(budgets)
    log_V = np.log(optimal_vals)

    m, c = np.polyfit(log_B, log_V, 1)

    # Generate smooth line
    B_smooth = np.logspace(np.log10(budgets.min()), np.log10(budgets.max()), 100)
    V_smooth = np.exp(m * np.log(B_smooth) + c)

    return m, c, B_smooth, V_smooth


def plot_isoflop_curves(ax_N, ax_D, analysis_results, colors_N, colors_D):
    """Plots the top row: Loss vs Params/Tokens for each budget."""
    for idx, row in analysis_results.iterrows():
        budget = row["budget"]

        # Determine color based on index
        # Assuming colors_N and colors_D are arrays matching analysis_results length
        color_n = colors_N[idx]
        color_d = colors_D[idx]

        group = row["group_data"]

        # Plot Data - N (Blues)
        ax_N.scatter(group["params"], group["eval_loss"], color=color_n, alpha=0.7, s=20, label=f"{budget:.1e}")

        # Plot Data - D (Greens)
        ax_D.scatter(group["tokens"], group["eval_loss"], color=color_d, alpha=0.7, s=20, label=f"{budget:.1e}")

        # Plot Fits
        N_range = np.logspace(np.log10(group["params"].min() * 0.5), np.log10(group["params"].max() * 2.0), 100)
        D_range = np.logspace(np.log10(group["tokens"].min() * 0.5), np.log10(group["tokens"].max() * 2.0), 100)

        L_pred_N = np.polyval(row["coeffs_N"], np.log(N_range))
        L_pred_D = np.polyval(row["coeffs_D"], np.log(D_range))

        # Lines use the same color scale
        ax_N.plot(N_range, L_pred_N, color=color_n, linestyle="--", alpha=0.5)
        ax_D.plot(D_range, L_pred_D, color=color_d, linestyle="--", alpha=0.5)

        # Plot Optima
        L_min_N = np.polyval(row["coeffs_N"], np.log(row["opt_N"]))
        ax_N.scatter([row["opt_N"]], [L_min_N], color=color_n, marker="s", s=100, edgecolors="black", zorder=10)

        L_min_D = np.polyval(row["coeffs_D"], np.log(row["opt_D"]))
        ax_D.scatter([row["opt_D"]], [L_min_D], color=color_d, marker="s", s=100, edgecolors="black", zorder=10)

    # Configure axes
    for ax, xlabel, title in [(ax_N, "Parameters (N)", "Loss vs Parameters"), (ax_D, "Tokens (D)", "Loss vs Tokens")]:
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Validation Loss")
        ax.set_title(title)
        ax.grid(True, which="both", ls="-", alpha=0.2, axis="x")
        # Don't show legend here anymore, global legend instead


def plot_scaling_laws(ax_N, ax_D, analysis_results, colors_N, colors_D):
    """Plots the bottom row: Optimal Params/Tokens vs FLOPs."""
    valid_data = analysis_results.dropna(subset=["opt_N", "opt_D"])
    valid_indices = valid_data.index

    # Extract colors corresponding to valid data points
    valid_colors_N = colors_N[valid_indices]
    valid_colors_D = colors_D[valid_indices]

    budgets = valid_data["budget"].values
    opt_N = valid_data["opt_N"].values
    opt_D = valid_data["opt_D"].values

    # Fit Scaling Laws
    m_N, c_N, B_smooth_N, N_smooth = fit_scaling_law(budgets, opt_N)
    m_D, c_D, B_smooth_D, D_smooth = fit_scaling_law(budgets, opt_D)

    # Plot Params Scaling
    ax_N.scatter(budgets, opt_N, color=valid_colors_N, marker="s", s=100, edgecolors="black", zorder=5)
    ax_N.plot(B_smooth_N, N_smooth, color="gray", linestyle="--", label=f"$N^* \\propto C^{{{m_N:.3f}}}$")

    # Plot Tokens Scaling
    ax_D.scatter(budgets, opt_D, color=valid_colors_D, marker="s", s=100, edgecolors="black", zorder=5)
    ax_D.plot(B_smooth_D, D_smooth, color="gray", linestyle="--", label=f"$D^* \\propto C^{{{m_D:.3f}}}$")

    # Configure axes
    for ax, ylabel, title in [
        (ax_N, "Optimal Parameters (N*)", "Optimal Parameters vs Compute"),
        (ax_D, "Optimal Tokens (D*)", "Optimal Tokens vs Compute"),
    ]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Compute Budget (FLOPs)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", ls="-", alpha=0.2, axis="x")
        ax.legend()
        # Set explicit tick positions at budget values only
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{b:.1e}" for b in budgets])
        ax.minorticks_off()

    # Calculate Ratio Function
    ratio_coeff = np.exp(c_D - c_N)
    ratio_exp_diff = m_D - m_N  # Simple difference of exponents
    # Optimal ratio exponent: R* = (m_N - m_D) / (m_N + m_D)
    # This is the Chinchilla optimal token/param ratio exponent
    ratio_exp_opt = (m_N - m_D) / (m_N + m_D)

    ratio_str = (
        f"Optimal Ratio: $\\frac{{D_{{opt}}}}{{N_{{opt}}}} = "
        f"{ratio_exp_opt:.4f}$ ($\\frac{{D^*}}{{N^*}} = {ratio_exp_diff:.4f}$)"
    )

    # Return all computed values for summary display
    return {
        "ratio_str": ratio_str,
        "m_N": m_N,
        "c_N": c_N,
        "m_D": m_D,
        "c_D": c_D,
        "ratio_coeff": ratio_coeff,
        "ratio_exp_diff": ratio_exp_diff,
        "ratio_exp_opt": ratio_exp_opt,
    }


class DualColorMarker:
    """
    Custom marker handler for legend that draws two markers side-by-side.
    """

    def __init__(self, color1, color2):
        self.color1 = color1
        self.color2 = color2

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height

        # Increase marker size (taller and wider)
        rect_h = height * 1.4  # Significantly larger
        rect_w = rect_h * 1.2  # Aspect ratio ~ 1.2 (wide)

        # Centered vertically
        y_pos = y0 + (height - rect_h) / 2

        # Gap between markers
        marker_gap = 5  # Fixed pixel gap

        # Total width of the dual marker group
        total_group_width = rect_w + marker_gap + rect_w

        # Align the dual marker group to the RIGHT side of the handlebox area
        # This pushes it closer to the text label which starts immediately after handlebox
        start_x = x0 + width - total_group_width

        # Draw two markers side-by-side
        # Left marker (Params/Blue)
        p1 = plt.Rectangle(
            [start_x, y_pos],
            rect_w,
            rect_h,
            facecolor=self.color1,
            edgecolor="black",
            transform=handlebox.get_transform(),
        )
        # Right marker (Tokens/Green)
        p2 = plt.Rectangle(
            [start_x + rect_w + marker_gap, y_pos],
            rect_w,
            rect_h,
            facecolor=self.color2,
            edgecolor="black",
            transform=handlebox.get_transform(),
        )

        handlebox.add_artist(p1)
        handlebox.add_artist(p2)
        return [p1, p2]


def plot_scaling_extrapolation(analysis_results, scaling_results, out_dir):
    """Creates a figure showing N*, D*, and D*/N* vs compute with extrapolation."""
    valid_data = analysis_results.dropna(subset=["opt_N", "opt_D"])
    budgets = valid_data["budget"].values
    opt_N, opt_D = valid_data["opt_N"].values, valid_data["opt_D"].values

    m_N, c_N = scaling_results["m_N"], scaling_results["c_N"]
    m_D, c_D = scaling_results["m_D"], scaling_results["c_D"]
    ratio_exp_diff = scaling_results["ratio_exp_diff"]  # Simple: m_D - m_N
    ratio_exp_opt = scaling_results["ratio_exp_opt"]  # R*: (m_N - m_D) / (m_N + m_D)

    # Extrapolate from min observed to 1e22 FLOPs
    C_ext = np.logspace(np.log10(budgets.min()), 22, 200)
    N_ext = np.exp(c_N) * C_ext**m_N
    D_ext = np.exp(c_D) * C_ext**m_D
    # Use R* (optimal exponent) for the ratio line
    # D*/N* = ratio_coeff * C^(-R*) since R* = (m_N - m_D)/(m_N + m_D) and we want D*/N*
    ratio_coeff = scaling_results["ratio_coeff"]
    ratio_ext = ratio_coeff * C_ext ** (-ratio_exp_opt)

    _, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    # Plot fits and data
    (ln,) = ax1.plot(C_ext, N_ext, color="tab:blue", lw=2, label=f"N* ∝ C^{m_N:.3f}")
    (ld,) = ax1.plot(C_ext, D_ext, color="tab:green", lw=2, label=f"D* ∝ C^{m_D:.3f}")
    ax1.scatter(budgets, opt_N, color="tab:blue", s=80, edgecolors="black", zorder=5, marker="o")
    ax1.scatter(budgets, opt_D, color="tab:green", s=80, edgecolors="black", zorder=5, marker="s")

    # Extrapolation shading
    ax1.axvspan(budgets.max(), 1e22, alpha=0.1, color="gray")
    ax1.axvline(budgets.max(), color="gray", ls=":", alpha=0.5)
    ax1.set_xlabel("Compute Budget C (FLOPs)", fontsize=12)
    ax1.set_ylabel("Optimal N* (params) / D* (tokens)", fontsize=12)
    ax1.grid(True, which="major", ls="-", alpha=0.2)

    # Right axis: Ratio (set behind ax1 so annotations render on top)
    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    ax2.set_zorder(ax1.get_zorder() - 1)
    ax1.patch.set_visible(False)
    # Label shows both simple and normalized exponents
    (lr,) = ax2.plot(
        C_ext,
        ratio_ext,
        color="gray",
        lw=2,
        ls="--",
        label=f"D*/N* (R*={ratio_exp_opt:.3f}, Δm={ratio_exp_diff:.3f})",
    )
    ax2.set_ylabel("Optimal Ratio D*/N*", fontsize=12)

    # Combined legend
    ax1.legend(handles=[ln, ld, lr], loc="upper left")

    # Reference annotations
    for C_ref in [1e18, 1e20, 1e22]:
        N_ref, D_ref = np.exp(c_N) * C_ref**m_N, np.exp(c_D) * C_ref**m_D
        ax1.axvline(C_ref, color="gray", ls=":", alpha=0.3)
        ax1.annotate(
            f"C={C_ref:.0e}\nN*={N_ref:.1e}\nD*={D_ref:.1e}\nD*/N*={D_ref/N_ref:.0f}",
            xy=(C_ref, N_ref * 1.5),
            fontsize=8,
            ha="center",
            va="bottom",
            zorder=20,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

    plt.title("Scaling Law Extrapolation: Optimal Compute Allocation", fontsize=14)
    plt.tight_layout()

    out_png, out_pdf = out_dir / "plantcad_scaling_extrapolation.png", out_dir / "plantcad_scaling_extrapolation.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    return out_png, out_pdf


def print_summary(df, analysis_results, scaling_results, out_files):
    """Print summary using rich tables."""
    console.print()

    # Optimal allocations table
    opt_table = Table(title="Optimal Allocations")
    opt_table.add_column("Budget (C)", justify="right", style="cyan")
    opt_table.add_column("Opt N*", justify="right")
    opt_table.add_column("Opt D*", justify="right")
    opt_table.add_column("D*/N*", justify="right", style="yellow")

    for _, row in analysis_results.iterrows():
        ratio = row["opt_D"] / row["opt_N"] if row["opt_N"] > 0 else np.nan
        opt_table.add_row(
            f"{row['budget']:.1e}",
            f"{row['opt_N']:.2e}",
            f"{row['opt_D']:.2e}",
            f"{ratio:.1f}",
        )
    console.print(opt_table)
    console.print()

    # Scaling laws
    m_N, c_N = scaling_results["m_N"], scaling_results["c_N"]
    m_D, c_D = scaling_results["m_D"], scaling_results["c_D"]
    ratio_coeff = scaling_results["ratio_coeff"]
    ratio_exp_diff = scaling_results["ratio_exp_diff"]
    ratio_exp_opt = scaling_results["ratio_exp_opt"]

    console.print("[bold]Scaling Laws:[/bold]")
    console.print(f"  N* ∝ C^{m_N:.3f}  [dim](log N* = {m_N:.4f} log C + {c_N:.4f})[/dim]")
    console.print(f"  D* ∝ C^{m_D:.3f}  [dim](log D* = {m_D:.4f} log C + {c_D:.4f})[/dim]")
    console.print(f"  D*/N* = {ratio_coeff:.4e} · C^{ratio_exp_diff:.4f}")
    console.print(f"  R* = (m_N - m_D)/(m_N + m_D) = {ratio_exp_opt:.4f}")
    console.print()

    for f in out_files:
        console.print(f"[green]✓[/green] Saved: {f}")
    console.print()


def run_isoflop_fit_analysis(df: pd.DataFrame, architecture: str = DEFAULT_ARCH) -> None:
    """Run isoflop scaling law fitting and visualization on epoch=1 data for the given architecture."""
    df_fit = filter_to_finished_runs(df)
    df_fit = df_fit[(df_fit["architecture"] == architecture) & (df_fit["epochs"] == 1)].dropna(
        subset=["eval_loss", "tokens", "params", "flops_budget"]
    )
    if df_fit.empty:
        raise ValueError("No valid data found for isoflop fitting")
    analysis_results = analyze_budgets(df_fit)

    # Setup Figure: 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={"height_ratios": [1.5, 1]})

    # Colorscales
    colors_N = plt.cm.Blues(np.linspace(0.4, 1.0, len(analysis_results)))
    colors_D = plt.cm.Greens(np.linspace(0.4, 1.0, len(analysis_results)))

    # Top Row
    plot_isoflop_curves(axes[0, 0], axes[0, 1], analysis_results, colors_N, colors_D)

    # Bottom Row
    scaling_results = plot_scaling_laws(axes[1, 0], axes[1, 1], analysis_results, colors_N, colors_D)

    # Global Legend with dual-color markers
    legend_handles = []
    for _, row in analysis_results.iterrows():
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color="none", label=f"{row['budget']:.1e}"))

    handler_map = {legend_handles[i]: DualColorMarker(colors_N[i], colors_D[i]) for i in range(len(legend_handles))}

    leg = fig.legend(
        handles=legend_handles,
        handler_map=handler_map,
        title="Compute Budget [$C$]\n(FLOPs)",
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        borderaxespad=0.5,
        handlelength=2.5,
        labelspacing=0.6,
        handletextpad=0.8,
    )
    leg.get_title().set_multialignment("center")

    plt.suptitle("PlantCAD Isoflop Analysis: Quadratic Optima & Scaling Laws", y=0.94, fontsize=16)
    plt.figtext(0.5, 0.88, scaling_results["ratio_str"], ha="center", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.85, 0.91])

    out_dir = Path(RESULT_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "plantcad_isoflop_fits.png"
    out_pdf = out_dir / "plantcad_isoflop_fits.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()

    # Create extrapolation figure
    extrap_png, extrap_pdf = plot_scaling_extrapolation(analysis_results, scaling_results, out_dir)

    # Print summary
    out_files = [out_png, out_pdf, extrap_png, extrap_pdf]
    print_summary(df_fit, analysis_results, scaling_results, out_files)


# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and analyze plantcad isoflop runs")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refetch from W&B even if CSV exists",
    )
    parser.add_argument(
        "--output",
        default=f"{RESULT_PATH}/plantcad_isoflops.csv",
        help=f"Output CSV path (default: {RESULT_PATH}/plantcad_isoflops.csv)",
    )
    parser.add_argument(
        "--show-wandb-runs",
        action="store_true",
        help="Log detailed info for first 2 W&B runs",
    )
    args = parser.parse_args()

    # Setup logging to console and file
    log_path = Path(RESULT_PATH) / "plantcad_isoflop_analysis.txt"
    setup_logging(log_path)

    output_path = Path(args.output)

    # Check if CSV exists and load from it unless --force is specified
    if output_path.exists() and not args.force:
        logger.info(f"Loading existing data from {output_path}")
        df = pd.read_csv(output_path)
        logger.info(f"Loaded {len(df)} runs from CSV")
    else:
        logger.info("Fetching runs from W&B...")
        df = fetch_plantcad_runs(show_wandb_runs=args.show_wandb_runs)
        save_runs(df, output_path)

    df = filter_exploded_runs(df)
    validate_runs(df)
    summarize_runs(df)
    visualize_loss_by_token_count(df)
    visualize_loss_by_param_count(df)
    visualize_loss_by_epochs(df)
    visualize_loss_by_param_and_epoch_count(df)
    run_isoflop_fit_analysis(df)

    # Append rich console output to log file
    with open(log_path, "a") as f:
        f.write("\n" + console.export_text())
        f.write(f"\nAnalysis complete. Logs saved to {log_path}\n")
    # Also print to console
    console.print(f"[green]Analysis complete.[/green] Logs saved to {log_path}")
