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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.interpolate import RegularGridInterpolator
import wandb

RUN_PREFIX = "plantcad_isoflop_03"
RESULT_PATH = "experiments/plantcad/results/v2"


def log_run_object(run, run_idx):
    """Log a run object as JSON to show available data."""
    print(f"\n{'=' * 80}")
    print(f"RUN {run_idx + 1}: {run.name}")
    print(f"{'=' * 80}")
    run_dict = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "tags": run.tags,
        "config": dict(run.config),
        "summary": dict(run.summary),
    }
    print(json.dumps(run_dict, indent=2, default=str))
    print(f"{'=' * 80}\n")


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
            "eval_loss": run.summary.get("eval/plantcad_cropped/loss"),
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
        }
        data.append(row)

    return pd.DataFrame(data)


def save_runs(df, output_path=f"{RESULT_PATH}/plantcad_isoflops.csv"):
    """Save dataframe to CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} runs to {output_path}")


def validate_runs(df):
    """Validate that rows are unique by key columns."""
    key_cols = ["architecture", "flops_budget", "tokens", "params", "epochs"]
    duplicates = df[df.duplicated(subset=key_cols, keep=False)]
    if not duplicates.empty:
        print(f"WARNING: Found {len(duplicates)} duplicate rows by {key_cols}:")
        print(duplicates[["run_name", *key_cols]].to_string())
    else:
        print(f"Validation passed: rows are unique by {key_cols}")


def summarize_runs(df):
    """Print formatted summary tables using rich."""
    console = Console()
    gflops_to_flops = 1e9

    # Simplified summary table
    summary_table = Table(title="Run Summary", show_header=True, header_style="bold cyan")
    for col in ["run_name", "state", "flops_budget", "architecture", "epochs", "eval_loss", "run_progress"]:
        summary_table.add_column(col)
    summary = df[["run_name", "state", "flops_budget", "architecture", "epochs", "eval_loss", "run_progress"]].copy()
    for _, row in summary.sort_values(["flops_budget", "architecture", "epochs"]).iterrows():
        summary_table.add_row(*[str(v) if pd.notna(v) else "" for v in row])
    console.print(summary_table)

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
    df_clean = df[df["state"] == "finished"].dropna(subset=required_cols)

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
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def visualize_loss_by_param_count(df, metric="eval_loss", output_path=f"{RESULT_PATH}/plantcad_loss_by_params.png"):
    """Plot loss vs params, colored by budget, faceted by architecture (cols) and epochs (rows)."""
    required_cols = [metric, "params", "architecture", "flops_budget", "epochs"]
    df_clean = df[df["state"] == "finished"].dropna(subset=required_cols)

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
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def format_large_num(n):
    """Simple formatter for large numbers (e.g. 1.5B, 100M)."""
    if n is None:
        return "N/A"
    try:
        n = float(n)
    except (ValueError, TypeError):
        return str(n)
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:.0f}"


HTML_BASE_CSS = """
<style type="text/css">
table.dataframe {
  border-collapse: collapse;
  border: 1px solid #aaa;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 11px;
}
table.dataframe th, table.dataframe td {
  border: 1px solid #aaa;
  padding: 4px 6px;
}
table.dataframe thead tr {
  background-color: #f0f0f0;
}
table.dataframe th.row_heading, table.dataframe th.index_name {
  text-align: left;
  white-space: nowrap;
}
table.dataframe td {
  text-align: right;
}
</style>
"""


def visualize_loss_by_epochs(
    df,
    metric: str = "eval_loss",
    output_path: str = f"{RESULT_PATH}/plantcad_loss_by_epochs.html",
) -> None:
    """Export loss vs. epochs as an HTML table with row-wise color gradients."""
    required_cols = [metric, "tokens", "params", "architecture", "flops_budget", "epochs"]
    df_clean = df[df["state"] == "finished"].dropna(subset=required_cols).copy()

    architectures = sorted(df_clean["architecture"].unique())
    if not architectures:
        print("No data to visualize.")
        return

    # If the user passed a PNG path, switch to HTML next to it.
    out_path = Path(output_path)
    if out_path.suffix.lower() != ".html":
        out_path = out_path.with_suffix(".html")

    html_sections: list[str] = []

    for arch in architectures:
        df_arch = df_clean[df_clean["architecture"] == arch].copy()
        if df_arch.empty:
            continue

        # Format columns for display in the index
        df_arch["budget_str"] = df_arch["flops_budget"].apply(lambda x: f"{x:.1e}")
        df_arch["params_str"] = df_arch["params"].apply(format_large_num)
        df_arch["tokens_str"] = df_arch["tokens"].apply(format_large_num)

        # Pivot: Index=(Budget, Params, Tokens), Columns=Epochs, Values=Loss
        pivot = df_arch.pivot_table(
            index=["flops_budget", "budget_str", "params", "params_str", "tokens", "tokens_str"],
            columns="epochs",
            values=metric,
        )
        if pivot.empty:
            continue
        # Sort rows: high compute first, then params, then tokens
        pivot = pivot.sort_index(level=["flops_budget", "params", "tokens"], ascending=[False, True, True])

        # Replace MultiIndex with a single string index using C, N, D labels
        def _format_index(idx, pivot=pivot) -> str:
            names = pivot.index.names
            c = idx[names.index("budget_str")]
            n = idx[names.index("params_str")]
            d = idx[names.index("tokens_str")]
            return f"C: {c} | N: {n} | D: {d}"

        pivot.index = pd.Index([_format_index(idx) for idx in pivot.index], name=None)

        # Make column labels nicer (e.g., "1 Ep") and ensure no extra header rows for column names
        pivot.columns = [f"{int(col)} Ep" if float(col).is_integer() else str(col) for col in pivot.columns]
        pivot.columns.name = None

        # Style similar to pandas Styler in a Jupyter notebook:
        #  - row-wise background gradient so each configuration uses full cmap span
        #  - minimal extra styling, just number formatting and a caption.
        styled = (
            pivot.style.format("{:.4f}")
            .background_gradient(axis=1, cmap="RdYlGn_r")
            .set_table_attributes('class="dataframe"')
            .set_caption(
                f"Architecture: {arch} "
                "(C = compute budget, N = params, D = tokens; colors are row-wise normalized loss)"
            )
        )

        html_sections.append(styled.to_html())

    if not html_sections:
        print("No tables generated for any architecture.")
        return

    full_html = (
        "<html><head><meta charset='utf-8'>"
        "<title>PlantCAD Loss by Epochs</title>"
        f"{HTML_BASE_CSS}</head><body>\n" + "<br><br>\n".join(html_sections) + "\n</body></html>"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_html, encoding="utf-8")
    print(f"Saved HTML table visualization to {out_path}")


def get_size_label(rank: int) -> str:
    """Map model size rank to human-readable label."""
    labels = ["XXS", "XS", "S", "M", "L", "XL", "XXL", "XXXL"]
    if rank < len(labels):
        return labels[rank]
    return f"Size-{rank}"


def visualize_loss_by_epochs_matplotlib(
    df,
    metric: str = "eval_loss",
    output_path: str = f"{RESULT_PATH}/plantcad_loss_by_epochs.png",
) -> None:
    """Plot normalized loss vs epochs, faceted by architecture and budget."""
    required_cols = [metric, "tokens", "params", "architecture", "flops_budget", "epochs"]
    df_clean = df[df["state"] == "finished"].dropna(subset=required_cols).copy()

    architectures = sorted(df_clean["architecture"].unique())
    budgets = sorted(df_clean["flops_budget"].unique(), reverse=True)
    if not architectures or not budgets:
        print("No data to visualize.")
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
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def visualize_loss_by_param_and_epoch_count(
    df,
    architecture: str = "qwen",
    metric: str = "eval_loss",
    clip_percentile: float = 80.0,
    output_path: str = f"{RESULT_PATH}/plantcad_loss_contour.png",
) -> None:
    """2D contour plot of loss vs params (y) and epochs (x), faceted by flops budget."""
    required_cols = [metric, "params", "epochs", "architecture", "flops_budget"]
    df_clean = df[(df["state"] == "finished") & (df["architecture"] == architecture)].dropna(subset=required_cols).copy()

    if df_clean.empty:
        print(f"No data for architecture '{architecture}'")
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

        # Pivot to 2D grid and interpolate to finer grid in log space
        pivot = df_budget.pivot_table(index="params", columns="epochs", values="log_loss")
        x_orig, y_orig = pivot.columns.values, pivot.index.values

        interp = RegularGridInterpolator((np.log(y_orig), np.log(x_orig)), pivot.values, method="cubic")
        xi = np.geomspace(x_orig.min(), x_orig.max(), 200)
        yi = np.geomspace(y_orig.min(), y_orig.max(), 200)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        zi = interp((np.log(yi_grid), np.log(xi_grid)))

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

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


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

    output_path = Path(args.output)

    # Check if CSV exists and load from it unless --force is specified
    if output_path.exists() and not args.force:
        print(f"Loading existing data from {output_path}")
        df = pd.read_csv(output_path)
        print(f"Loaded {len(df)} runs from CSV")
    else:
        print("Fetching runs from W&B...")
        df = fetch_plantcad_runs(show_wandb_runs=args.show_wandb_runs)
        save_runs(df, output_path)

    validate_runs(df)
    summarize_runs(df)
    visualize_loss_by_token_count(df)
    visualize_loss_by_param_count(df)
    visualize_loss_by_epochs(df)
    visualize_loss_by_epochs_matplotlib(df)
    visualize_loss_by_param_and_epoch_count(df)
