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

"""
One-off visualization script to aggregate data across all plantcad isoflop CSV files
and create a combined isoflop fits figure grouped by compute range.

This aggregates data from v2.x versions matching the sweep configs:
- v2.3: minimal-compute (1x), 2K steps
- v2.6: very-low-compute (2x), 4K steps
- v2.4: low-compute (4x), 8K steps
- v2.5: mid-compute (8x), 16K steps
- v2.2: high-compute (16x), 32K steps

The resulting figure has 4 facets in a 2x2 layout:
- Top row: Loss vs Parameters, Loss vs Tokens (with data/fits from all compute ranges)
- Bottom row: Optimal Params vs FLOPs, Optimal Tokens vs FLOPs (scaling laws per range)
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse existing validation and filtering functions
from experiments.plantcad.exp2101_plantcad_isoflop_analysis import (
    EXPLODED_BUDGETS,
    EXPLODED_RUNS,
    analyze_budgets,
    filter_to_finished_runs,
    fit_scaling_law,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define compute ranges with their metadata
# Format: (version, display_name, steps)
# Note: v2.3 (minimal-compute, 2K steps) excluded from analysis
COMPUTE_RANGES = [
    ("v2.6", "very-low (4K steps)", 4096),
    ("v2.4", "low (8K steps)", 8192),
    ("v2.5", "mid (16K steps)", 16384),
    ("v2.2", "high (32K steps)", 32768),
]

RESULTS_BASE_PATH = Path("experiments/plantcad/results")
OUTPUT_DIR = RESULTS_BASE_PATH / "meta_analysis"
OUTPUT_BASE = "plantcad_isoflop_sweep_bias"
DEFAULT_ARCH = "qwen"
EXPORT_DPI = 300

# Color scheme for compute ranges (distinct colors for each range)
RANGE_COLORS = {
    "v2.3": "#1f77b4",  # Blue - minimal
    "v2.6": "#ff7f0e",  # Orange - very-low
    "v2.4": "#2ca02c",  # Green - low
    "v2.5": "#d62728",  # Red - mid
    "v2.2": "#9467bd",  # Purple - high
}


def load_all_csvs() -> dict[str, pd.DataFrame]:
    """Load all plantcad_isoflops.csv files for v2.x versions."""
    data: dict[str, pd.DataFrame] = {}
    for version, name, steps in COMPUTE_RANGES:
        csv_path = RESULTS_BASE_PATH / version / "plantcad_isoflops.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["version"] = version
            df["compute_range"] = name
            df["range_steps"] = steps
            data[version] = df
            logger.info(f"Loaded {len(df)} runs from {csv_path}")
        else:
            logger.warning(f"CSV not found: {csv_path}")
    return data


def filter_df_by_params(
    df: pd.DataFrame,
    min_params: float | None = None,
    max_params: float | None = None,
) -> pd.DataFrame:
    """Filter a DataFrame to a parameter range inclusively.

    Args:
        df: DataFrame with a 'params' column.
        min_params: Minimum number of parameters (inclusive). None means no lower bound.
        max_params: Maximum number of parameters (inclusive). None means no upper bound.

    Returns:
        Filtered DataFrame.
    """
    mask = pd.Series(True, index=df.index)
    if min_params is not None:
        mask &= df["params"] >= min_params
    if max_params is not None:
        mask &= df["params"] <= max_params
    return df[mask]


def filter_exploded_runs_for_version(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """Filter out runs where training exploded for a specific version."""
    # EXPLODED_BUDGETS/RUNS use version keys without "v" prefix (e.g., "2.2" not "v2.2")
    version_key = version.lstrip("v")
    exploded_runs = EXPLODED_RUNS.get(version_key, [])
    exploded_budgets = EXPLODED_BUDGETS.get(version_key, [])

    run_mask = df["run_name"].isin(exploded_runs)
    budget_mask = df["flops_budget"].isin(exploded_budgets)

    n_filtered = run_mask.sum() + budget_mask.sum()
    if n_filtered > 0:
        logger.warning(f"Filtered {n_filtered} exploded runs/budgets for {version}")

    return df[~run_mask & ~budget_mask]


def print_budget_summary(data: dict[str, pd.DataFrame]) -> None:
    """Print a sorted list of compute range, version, and flop budgets."""
    rows: list[tuple[str, str, float]] = []

    for version, name, _ in COMPUTE_RANGES:
        if version not in data:
            continue

        df = data[version]
        df = filter_exploded_runs_for_version(df, version)
        df = filter_to_finished_runs(df)

        for budget in sorted(df["flops_budget"].unique()):
            rows.append((name, version, budget))

    # Sort by compute range name (which includes step count), then by budget
    rows.sort(key=lambda x: (x[0], x[2]))

    logger.info("\n" + "=" * 60)
    logger.info("Compute Range / Version / FLOPs Budget Summary")
    logger.info("=" * 60)
    logger.info(f"{'Compute Range':<20} {'Version':<10} {'FLOPs Budget':<15}")
    logger.info("-" * 60)
    for name, version, budget in rows:
        logger.info(f"{name:<20} {version:<10} {budget:.2e}")
    logger.info("=" * 60 + "\n")


def prepare_data(
    data: dict[str, pd.DataFrame],
    architecture: str = DEFAULT_ARCH,
    fit_min_params: float | None = None,
    fit_max_params: float | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Prepare and filter data for each compute range, returning data and analysis.

    Args:
        data: Dictionary of version -> DataFrame from load_all_csvs.
        architecture: Architecture to filter by.
        fit_min_params: Optional minimum params for parabola fitting (does not filter scatter data).
        fit_max_params: Optional maximum params for parabola fitting (does not filter scatter data).

    Returns:
        Dictionary of version -> (full_df, analysis_df).
        full_df contains all data points for scatter plots.
        analysis_df contains fits based on optionally filtered data.
    """
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for version, name, _ in COMPUTE_RANGES:
        if version not in data:
            continue

        df = data[version].copy()
        df = filter_exploded_runs_for_version(df, version)
        df = filter_to_finished_runs(df)
        df = df[(df["architecture"] == architecture) & (df["epochs"] == 1)].dropna(
            subset=["eval_loss", "tokens", "params", "flops_budget"]
        )

        if df.empty:
            logger.warning(f"No valid data for {name}")
            continue

        # Apply param filtering only for fitting, not for scatter data
        df_for_fitting = filter_df_by_params(df, fit_min_params, fit_max_params)
        if df_for_fitting.empty:
            logger.warning(f"No data in param range for fitting for {name}")
            analysis = pd.DataFrame()
        else:
            analysis = analyze_budgets(df_for_fitting)

        results[version] = (df, analysis)

    return results


def plot_loss_vs_params(
    ax,
    range_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    fit_min_params: float | None = None,
    fit_max_params: float | None = None,
):
    """Plot Loss vs Parameters with all compute ranges overlaid.

    Args:
        ax: Matplotlib axes to plot on.
        range_data: Dictionary of version -> (full_df, analysis_df).
        fit_min_params: If provided, draw vertical dashed line at this value.
        fit_max_params: If provided, draw vertical dashed line at this value.
    """
    for version, name, _ in COMPUTE_RANGES:
        if version not in range_data:
            continue

        df, analysis = range_data[version]
        color = RANGE_COLORS[version]

        # Plot all data points for this range (more transparent)
        ax.scatter(df["params"], df["eval_loss"], color=color, alpha=0.25, s=15, label=name)

        # Plot parabola fits and optimum points for each budget
        for _, row in analysis.iterrows():
            group = row["group_data"]
            params_min, params_max = group["params"].min(), group["params"].max()

            # Draw parabola fit within actual data range
            N_range = np.logspace(np.log10(params_min), np.log10(params_max), 100)
            L_pred = np.polyval(row["coeffs_N"], np.log(N_range))
            ax.plot(N_range, L_pred, color=color, linestyle="--", alpha=0.5, linewidth=1)

            # Plot optimum point
            if pd.notna(row["opt_N"]):
                L_min = np.polyval(row["coeffs_N"], np.log(row["opt_N"]))
                ax.scatter([row["opt_N"]], [L_min], color=color, marker="s", s=60, edgecolors="black", zorder=10)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (N)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Loss vs Parameters")
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Draw vertical lines for fitting bounds
    if fit_min_params is not None:
        ax.axvline(x=fit_min_params, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
    if fit_max_params is not None:
        ax.axvline(x=fit_max_params, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)


def plot_loss_vs_tokens(
    ax,
    range_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    fit_min_params: float | None = None,
    fit_max_params: float | None = None,
):
    """Plot Loss vs Tokens with all compute ranges overlaid.

    Args:
        ax: Matplotlib axes to plot on.
        range_data: Dictionary of version -> (full_df, analysis_df).
        fit_min_params: If provided, used to indicate fitting was restricted (no vertical line here).
        fit_max_params: If provided, used to indicate fitting was restricted (no vertical line here).

    Note: Vertical lines for param bounds are only shown on the Loss vs Params plot.
    """
    for version, name, _ in COMPUTE_RANGES:
        if version not in range_data:
            continue

        df, analysis = range_data[version]
        color = RANGE_COLORS[version]

        # Plot all data points for this range (more transparent)
        ax.scatter(df["tokens"], df["eval_loss"], color=color, alpha=0.25, s=15, label=name)

        # Plot parabola fits and optimum points for each budget
        for _, row in analysis.iterrows():
            group = row["group_data"]
            tokens_min, tokens_max = group["tokens"].min(), group["tokens"].max()

            # Draw parabola fit within actual data range
            D_range = np.logspace(np.log10(tokens_min), np.log10(tokens_max), 100)
            L_pred = np.polyval(row["coeffs_D"], np.log(D_range))
            ax.plot(D_range, L_pred, color=color, linestyle="--", alpha=0.5, linewidth=1)

            # Plot optimum point
            if pd.notna(row["opt_D"]):
                L_min = np.polyval(row["coeffs_D"], np.log(row["opt_D"]))
                ax.scatter([row["opt_D"]], [L_min], color=color, marker="s", s=60, edgecolors="black", zorder=10)

    ax.set_xscale("log")
    ax.set_xlabel("Tokens (D)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Loss vs Tokens")
    ax.grid(True, which="both", ls="-", alpha=0.2)


def plot_optimal_params_vs_flops(ax, range_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]]):
    """Plot Optimal Parameters vs Compute with scaling laws per range."""
    for version, name, _ in COMPUTE_RANGES:
        if version not in range_data:
            continue

        _, analysis = range_data[version]
        color = RANGE_COLORS[version]

        valid_data = analysis.dropna(subset=["opt_N", "opt_D"])
        if len(valid_data) < 2:
            continue

        budgets = valid_data["budget"].values
        opt_N = valid_data["opt_N"].values

        # Fit and plot scaling law
        m_N, _c_N, B_smooth, N_smooth = fit_scaling_law(budgets, opt_N)

        ax.scatter(budgets, opt_N, color=color, marker="s", s=60, edgecolors="black", zorder=5)
        ax.plot(
            B_smooth, N_smooth, color=color, linestyle="--", alpha=0.7, label=f"{name}: $N^* \\propto C^{{{m_N:.2f}}}$"
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute Budget (FLOPs)")
    ax.set_ylabel("Optimal Parameters (N*)")
    ax.set_title("Optimal Parameters vs Compute")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(fontsize=7, loc="upper left")


def plot_optimal_tokens_vs_flops(ax, range_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]]):
    """Plot Optimal Tokens vs Compute with scaling laws per range."""
    for version, name, _ in COMPUTE_RANGES:
        if version not in range_data:
            continue

        _, analysis = range_data[version]
        color = RANGE_COLORS[version]

        valid_data = analysis.dropna(subset=["opt_N", "opt_D"])
        if len(valid_data) < 2:
            continue

        budgets = valid_data["budget"].values
        opt_D = valid_data["opt_D"].values

        # Fit and plot scaling law
        m_D, _c_D, B_smooth, D_smooth = fit_scaling_law(budgets, opt_D)

        ax.scatter(budgets, opt_D, color=color, marker="s", s=60, edgecolors="black", zorder=5)
        ax.plot(
            B_smooth, D_smooth, color=color, linestyle="--", alpha=0.7, label=f"{name}: $D^* \\propto C^{{{m_D:.2f}}}$"
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute Budget (FLOPs)")
    ax.set_ylabel("Optimal Tokens (D*)")
    ax.set_title("Optimal Tokens vs Compute")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(fontsize=7, loc="upper left")


def create_combined_figure(
    data: dict[str, pd.DataFrame],
    architecture: str = DEFAULT_ARCH,
    fit_min_params: float | None = None,
    fit_max_params: float | None = None,
):
    """Create a 4-facet figure showing isoflop analysis grouped by compute range.

    Layout (2x2):
    - Top-left: Loss vs Parameters
    - Top-right: Loss vs Tokens
    - Bottom-left: Optimal Params vs FLOPs
    - Bottom-right: Optimal Tokens vs FLOPs

    Args:
        data: Dictionary of version -> DataFrame from load_all_csvs.
        architecture: Architecture to filter by.
        fit_min_params: Optional minimum params for parabola fitting.
        fit_max_params: Optional maximum params for parabola fitting.
    """
    range_data = prepare_data(data, architecture, fit_min_params, fit_max_params)

    if not range_data:
        raise ValueError("No valid data found for any compute range")

    logger.info(f"Prepared data for {len(range_data)} compute ranges")

    # Create 2x2 figure with extra space on right for legend
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top row: Loss curves
    plot_loss_vs_params(axes[0, 0], range_data, fit_min_params, fit_max_params)
    plot_loss_vs_tokens(axes[0, 1], range_data, fit_min_params, fit_max_params)

    # Bottom row: Scaling laws
    plot_optimal_params_vs_flops(axes[1, 0], range_data)
    plot_optimal_tokens_vs_flops(axes[1, 1], range_data)

    # Add shared legend for top row to the right of all plots
    # Get handles and labels from the first top-row plot (they're the same for both)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(0.98, 0.75),
        fontsize=9,
        title="Compute Range",
        title_fontsize=10,
    )

    plt.suptitle("PlantCAD Isoflop Analysis: Bias by Compute Range (Step Count)", y=0.98, fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])

    return fig


def save_combined_data(data: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Save all combined data to CSV."""
    all_dfs = []
    for version, _name, _steps in COMPUTE_RANGES:
        if version not in data:
            continue
        df = data[version].copy()
        df = filter_exploded_runs_for_version(df, version)
        df = filter_to_finished_runs(df)
        all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved {len(combined)} rows to {output_path}")


# Optional fitting bounds for parabola fits (set to None to disable)
FIT_MIN_PARAMS: float | None = 10**6.8
FIT_MAX_PARAMS: float | None = 10**8.1


def main():
    """Main entry point."""
    logger.info("Loading CSV files...")
    data = load_all_csvs()

    if not data:
        raise ValueError("No CSV files found")

    # Print sorted budget summary
    print_budget_summary(data)

    logger.info(f"Creating combined figure from {len(data)} compute ranges...")
    if FIT_MIN_PARAMS is not None or FIT_MAX_PARAMS is not None:
        logger.info(f"Fitting restricted to param range [{FIT_MIN_PARAMS:.2e}, {FIT_MAX_PARAMS:.2e}]")
    fig = create_combined_figure(data, fit_min_params=FIT_MIN_PARAMS, fit_max_params=FIT_MAX_PARAMS)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save figure as PNG and PDF
    png_path = OUTPUT_DIR / f"{OUTPUT_BASE}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_BASE}.pdf"
    csv_path = OUTPUT_DIR / f"{OUTPUT_BASE}.csv"

    fig.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight")
    logger.info(f"Saved figure to {png_path}")

    fig.savefig(pdf_path, dpi=EXPORT_DPI, bbox_inches="tight")
    logger.info(f"Saved figure to {pdf_path}")

    plt.close(fig)

    # Save combined data as CSV
    save_combined_data(data, csv_path)


if __name__ == "__main__":
    main()
