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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

RESULT_PATH = "experiments/plantcad/results/v2"

console = Console(record=True)


def load_and_filter_data(csv_path):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_csv(path)

    # Filter
    df = df[(df["architecture"] == "qwen") & (df["state"] == "finished") & (df["epochs"] == 1)].dropna(
        subset=["eval_loss", "tokens", "params", "flops_budget"]
    )

    if df.empty:
        raise ValueError("No valid data found")

    return df


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
            raise ValueError(f"Budget {budget} has fewer than 3 points ({len(group)}), cannot fit.")

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

    # Calculate Ratio Function
    ratio_coeff = np.exp(c_D - c_N)
    ratio_exp = m_D - m_N

    exponent = int(np.floor(np.log10(ratio_coeff)))
    mantissa = ratio_coeff / (10**exponent)
    coeff_str = f"{mantissa:.2f} \\times 10^{{{exponent}}}"

    ratio_str = f"Optimal Token-Parameter Ratio: $D^{{*}}/N^{{*}} = {coeff_str} \\cdot C^{{{ratio_exp:.3f}}}$"

    # Return all computed values for summary display
    return {
        "ratio_str": ratio_str,
        "m_N": m_N,
        "c_N": c_N,
        "m_D": m_D,
        "c_D": c_D,
        "ratio_coeff": ratio_coeff,
        "ratio_exp": ratio_exp,
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
    ratio_exp = scaling_results["ratio_exp"]

    # Extrapolate from min observed to 1e22 FLOPs
    C_ext = np.logspace(np.log10(budgets.min()), 22, 200)
    N_ext = np.exp(c_N) * C_ext**m_N
    D_ext = np.exp(c_D) * C_ext**m_D
    ratio_ext = D_ext / N_ext

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
    (lr,) = ax2.plot(C_ext, ratio_ext, color="gray", lw=2, ls="--", label=f"D*/N* ∝ C^{ratio_exp:.3f}")
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
    console.rule("[bold blue]PlantCAD Isoflop Analysis[/bold blue]")
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
    ratio_exp = scaling_results["ratio_exp"]

    console.print("[bold]Scaling Laws:[/bold]")
    console.print(f"  N* ∝ C^{m_N:.3f}  [dim](log N* = {m_N:.4f} log C + {c_N:.4f})[/dim]")
    console.print(f"  D* ∝ C^{m_D:.3f}  [dim](log D* = {m_D:.4f} log C + {c_D:.4f})[/dim]")
    console.print(f"  D*/N* = {ratio_coeff:.4e} · C^{ratio_exp:.4f}")
    console.print()

    for f in out_files:
        console.print(f"[green]✓[/green] Saved: {f}")
    console.print()


def main():
    csv_path = f"{RESULT_PATH}/plantcad_isoflops.csv"
    df = load_and_filter_data(csv_path)
    analysis_results = analyze_budgets(df)

    # Setup Figure: 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={"height_ratios": [1.5, 1]})

    # Colorscales
    # Params (N) = Blues (0.4-1.0)
    # Tokens (D) = Greens (0.4-1.0)
    colors_N = plt.cm.Blues(np.linspace(0.4, 1.0, len(analysis_results)))
    colors_D = plt.cm.Greens(np.linspace(0.4, 1.0, len(analysis_results)))

    # Top Row
    plot_isoflop_curves(axes[0, 0], axes[0, 1], analysis_results, colors_N, colors_D)

    # Bottom Row
    scaling_results = plot_scaling_laws(axes[1, 0], axes[1, 1], analysis_results, colors_N, colors_D)

    # Global Legend
    # Create dummy handles using our custom artist
    # We use Rectangle((0,0), 1, 1) as a dummy proxy, but map it to our custom handler

    legend_handles = []
    for _, row in analysis_results.iterrows():
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color="none", label=f"{row['budget']:.1e}"))

    # Create a dictionary mapping the dummy handles to the custom handler instances
    handler_map = {legend_handles[i]: DualColorMarker(colors_N[i], colors_D[i]) for i in range(len(legend_handles))}

    leg = fig.legend(
        handles=legend_handles,
        handler_map=handler_map,
        title="Compute Budget [$C$]\n(FLOPs)",
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        borderaxespad=0.5,
        handlelength=2.5,
        labelspacing=0.6,  # Reduced from 1.0 to shrink gap further
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
    print_summary(df, analysis_results, scaling_results, out_files)


if __name__ == "__main__":
    main()
