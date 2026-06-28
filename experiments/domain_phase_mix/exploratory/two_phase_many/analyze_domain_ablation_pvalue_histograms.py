# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "plotly", "scipy"]
# ///
"""P-value histograms for 300M proportional domain-deletion ablations.

The domain-deletion panel is nonlocal: each row deletes one bucket from the
proportional mixture and renormalizes the remaining buckets. This script treats
the proportional repeat panel as a null noise estimate and asks whether the
observed deletion contrast is unusually large for each metric/domain pair.

It produces both raw per-domain p-values and per-benchmark min-p summaries.
The raw min-p summary is intentionally shown as an anti-conservative diagnostic;
the Sidak-corrected min-p is the defensible one-p-value-per-benchmark view.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import t as student_t

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "domain_ablation_pvalue_histograms_20260623"
DOMAIN_COMPARISON = (
    SCRIPT_DIR
    / "reference_outputs"
    / "ppert_bump_vs_log_tilt_comparison_20260614"
    / "domain_ablation_vs_local_gradient_domain_comparison.csv"
)
NOISE_MATRIX = (
    SCRIPT_DIR / "metric_registry" / "raw_metric_matrix_300m" / "raw_metric_matrix_300m_with_proportional_noise.csv"
)

TO_IMAGE_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

SMOOTH_KIND_PRIORITY = {
    "bpb": 0,
    "loss": 1,
    "nll": 2,
    "choice_logprob_norm": 3,
    "choice_logprob": 4,
    "correct_vs_best_incorrect_margin": 5,
    "normalized_correct_vs_best_incorrect_margin": 6,
    "success_macro_bpb": 7,
    "failed_macro_bpb": 8,
    "coderforge_success_macro_bpb": 9,
    "coderforge_failed_macro_bpb": 10,
}


def metric_kind(metric: str) -> str:
    return metric.rsplit("/", maxsplit=1)[-1]


def metric_family(metric: str) -> str:
    return metric.split("/", maxsplit=1)[0]


def lower_is_better(metric: str) -> bool:
    kind = metric_kind(metric).lower()
    return (
        kind in {"bpb", "loss", "nll"}
        or kind.endswith("_bpb")
        or kind.endswith("_loss")
        or kind.endswith("_nll")
        or "perplexity" in kind
    )


def has_ambiguous_direction(metric: str) -> bool:
    kind = metric_kind(metric).lower()
    return any(token in kind for token in ("minus", "diff", "delta", "ratio", "gain", "reduction", "improvement"))


def is_smooth_proxy(metric: str) -> bool:
    if has_ambiguous_direction(metric):
        return False
    kind = metric_kind(metric).lower()
    return kind in SMOOTH_KIND_PRIORITY or kind.endswith("_bpb") or kind.endswith("_loss") or kind.endswith("_nll")


def smooth_priority(metric: str) -> tuple[int, str]:
    kind = metric_kind(metric).lower()
    if kind in SMOOTH_KIND_PRIORITY:
        return SMOOTH_KIND_PRIORITY[kind], metric
    if kind.endswith("_bpb"):
        return 20, metric
    if kind.endswith("_loss"):
        return 30, metric
    if kind.endswith("_nll"):
        return 40, metric
    return 999, metric


def benchmark_key(metric: str) -> str:
    """Return the benchmark identity after stripping the scalar metric suffix."""
    kind = metric_kind(metric).lower()
    if kind in SMOOTH_KIND_PRIORITY:
        return metric.rsplit("/", maxsplit=1)[0]
    if kind.endswith("_bpb") or kind.endswith("_loss") or kind.endswith("_nll"):
        # Macro BPB metrics usually encode the benchmark in the last segment
        # itself, so grouping by the parent would collapse distinct leaves.
        return metric
    return metric.rsplit("/", maxsplit=1)[0]


def utility_values(frame: pd.DataFrame, metric: str) -> pd.Series:
    values = pd.to_numeric(frame[metric], errors="coerce")
    if lower_is_better(metric):
        values = -values
    return values


def utility_reference_values(noise_matrix: pd.DataFrame, metric: str) -> np.ndarray:
    if metric not in noise_matrix.columns:
        return np.asarray([], dtype=float)
    baseline = noise_matrix[noise_matrix["run_name"].eq("baseline_proportional")]
    repeats = noise_matrix[noise_matrix["row_kind"].eq("noise_variable_subset_proportional")]
    reference = pd.concat([baseline, repeats], ignore_index=True)
    return utility_values(reference, metric).dropna().to_numpy(dtype=float)


def select_smooth_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = frame[frame["reportable_metric"].astype(bool) & frame["metric"].map(is_smooth_proxy)].copy()
    candidates["benchmark_key"] = candidates["metric"].map(benchmark_key)
    metric_rows = (
        candidates[["benchmark_key", "metric", "metric_family", "metric_kind", "lower_is_better"]]
        .drop_duplicates()
        .copy()
    )
    metric_rows["_priority"] = metric_rows["metric"].map(smooth_priority)
    metric_rows = metric_rows.sort_values(["benchmark_key", "_priority"])
    selected = metric_rows.groupby("benchmark_key", as_index=False, sort=False).first()
    selected = selected.drop(columns=["_priority"])
    return selected


def summarize_benchmarks(cell: pd.DataFrame) -> pd.DataFrame:
    if cell.empty:
        raise ValueError("No p-values available.")

    summaries = []
    for metric, group in cell.groupby("metric", sort=False):
        n_domains = int(len(group))
        min_harm = float(group["p_harm"].min())
        min_two = float(group["p_two_sided"].min())
        best_harm = group.loc[group["p_harm"].idxmin()]
        best_two = group.loc[group["p_two_sided"].idxmin()]
        summaries.append(
            {
                "benchmark_key": str(best_harm["benchmark_key"]),
                "metric": metric,
                "metric_family": metric_family(metric),
                "metric_kind": metric_kind(metric),
                "n_domains": n_domains,
                "min_p_harm_raw": min_harm,
                "min_p_harm_sidak": float(1.0 - (1.0 - min_harm) ** n_domains),
                "min_p_harm_bonferroni": float(min(1.0, min_harm * n_domains)),
                "min_p_two_sided_raw": min_two,
                "min_p_two_sided_sidak": float(1.0 - (1.0 - min_two) ** n_domains),
                "min_p_two_sided_bonferroni": float(min(1.0, min_two * n_domains)),
                "best_harm_domain": str(best_harm["target_domain"]),
                "best_harm_delta": float(best_harm["domain_deletion_utility_delta"]),
                "best_harm_t": float(best_harm["t_statistic"]),
                "best_two_sided_domain": str(best_two["target_domain"]),
                "best_two_sided_delta": float(best_two["domain_deletion_utility_delta"]),
                "best_two_sided_t": float(best_two["t_statistic"]),
                "median_p_harm": float(group["p_harm"].median()),
                "median_p_two_sided": float(group["p_two_sided"].median()),
                "fraction_harm_p_lt_0p05_raw": float((group["p_harm"] < 0.05).mean()),
                "fraction_two_sided_p_lt_0p05_raw": float((group["p_two_sided"] < 0.05).mean()),
                "mean_deletion_delta": float(group["domain_deletion_utility_delta"].mean()),
                "median_deletion_delta": float(group["domain_deletion_utility_delta"].median()),
                "fraction_domains_harm": float((group["domain_deletion_utility_delta"] < 0.0).mean()),
            }
        )
    benchmark = pd.DataFrame(summaries).sort_values("min_p_harm_sidak")
    return benchmark


def compute_pvalues(comparison: pd.DataFrame, noise_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = select_smooth_metrics(comparison)
    selected_metrics = set(selected["metric"])
    rows: list[dict[str, Any]] = []
    for metric, group in comparison[comparison["metric"].isin(selected_metrics)].groupby("metric", sort=False):
        noise = utility_reference_values(noise_matrix, metric)
        if len(noise) < 3:
            continue
        base_utility = float(np.mean(noise))
        noise_sd = float(np.std(noise, ddof=1))
        if not np.isfinite(noise_sd) or noise_sd <= 0.0:
            continue
        n_noise = int(len(noise))
        df = n_noise - 1
        predictive_sd = noise_sd * math.sqrt(1.0 + 1.0 / n_noise)
        selected_row = selected[selected["metric"].eq(metric)].iloc[0]
        for deletion in group.itertuples(index=False):
            deletion_value = pd.to_numeric(pd.Series([deletion.deletion_metric_value]), errors="coerce").iloc[0]
            if not np.isfinite(deletion_value):
                continue
            deletion_utility = -deletion_value if lower_is_better(metric) else deletion_value
            delta = float(deletion_utility - base_utility)
            if not np.isfinite(delta):
                continue
            t_stat = delta / predictive_sd
            p_harm = float(student_t.cdf(t_stat, df=df))
            p_improve = float(student_t.sf(t_stat, df=df))
            p_two_sided = float(2.0 * min(p_harm, p_improve))
            rows.append(
                {
                    "benchmark_key": selected_row.benchmark_key,
                    "metric": metric,
                    "metric_family": metric_family(metric),
                    "metric_kind": metric_kind(metric),
                    "lower_is_better": lower_is_better(metric),
                    "target_domain": deletion.target_domain,
                    "base_mass": float(deletion.base_mass),
                    "proportional_reference_utility": base_utility,
                    "domain_deletion_utility_delta": delta,
                    "noise_n": n_noise,
                    "noise_sd": noise_sd,
                    "predictive_sd": predictive_sd,
                    "t_statistic": t_stat,
                    "p_harm": min(max(p_harm, 0.0), 1.0),
                    "p_improve": min(max(p_improve, 0.0), 1.0),
                    "p_two_sided": min(max(p_two_sided, 0.0), 1.0),
                }
            )
    cell = pd.DataFrame(rows)
    benchmark = summarize_benchmarks(cell)
    return cell, benchmark


def add_histogram(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    values: pd.Series,
    name: str,
    color: str,
    showlegend: bool = False,
) -> None:
    fig.add_trace(
        go.Histogram(
            x=values.dropna(),
            xbins={"start": 0.0, "end": 1.0, "size": 0.05},
            marker={"color": color, "line": {"color": "white", "width": 1}},
            name=name,
            showlegend=showlegend,
            hovertemplate="p in bin=%{x}<br>count=%{y}<extra></extra>",
        ),
        row=row,
        col=col,
    )


def add_expected_line(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    counts: np.ndarray,
    name: str,
) -> None:
    edges = np.linspace(0.0, 1.0, 21)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=counts,
            mode="lines",
            line={"color": "black", "dash": "dash", "width": 2},
            name=name,
            showlegend=False,
            hovertemplate=f"{name}<br>p bin center=%{{x:.2f}}<br>expected count=%{{y:.3g}}<extra></extra>",
        ),
        row=row,
        col=col,
    )


def plot_histograms(cell: pd.DataFrame, benchmark: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "All selected smooth proxy × deleted-domain cells: one-sided harm p",
            "All selected smooth proxy × deleted-domain cells: two-sided p",
            "One raw min-p per benchmark: anti-conservative if selected post hoc",
            "One Bonferroni-corrected min-p per benchmark: adjusted over 39 deletions",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
    )
    add_histogram(fig, row=1, col=1, values=cell["p_harm"], name="cell harm p", color="#d73027")
    add_histogram(fig, row=1, col=2, values=cell["p_two_sided"], name="cell two-sided p", color="#4575b4")
    add_histogram(fig, row=2, col=1, values=benchmark["min_p_harm_raw"], name="raw min harm p", color="#fc8d59")
    add_histogram(
        fig, row=2, col=2, values=benchmark["min_p_harm_bonferroni"], name="Bonferroni min harm p", color="#91bfdb"
    )
    uniform_cell = np.full(20, len(cell) / 20.0)
    add_expected_line(fig, row=1, col=1, counts=uniform_cell, name="Uniform null")
    add_expected_line(fig, row=1, col=2, counts=uniform_cell, name="Uniform null")
    edges = np.linspace(0.0, 1.0, 21)
    n_domains = int(benchmark["n_domains"].median())
    raw_min_null = len(benchmark) * ((1.0 - edges[:-1]) ** n_domains - (1.0 - edges[1:]) ** n_domains)
    add_expected_line(fig, row=2, col=1, counts=raw_min_null, name=f"Null min of {n_domains}")
    bonf_null = len(benchmark) * (
        (1.0 - edges[:-1] / n_domains) ** n_domains - (1.0 - np.minimum(edges[1:], 1.0) / n_domains) ** n_domains
    )
    # The final bin includes the point mass produced by clipping Bonferroni-adjusted p-values at 1.
    bonf_null[-1] = len(benchmark) * (1.0 - (1.0 - edges[-2] / n_domains) ** n_domains)
    add_expected_line(fig, row=2, col=2, counts=bonf_null, name="Independent-null Bonferroni")
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(range=[0, 1], dtick=0.1, title="p-value", row=row, col=col)
            fig.update_yaxes(title="count", row=row, col=col)
    fig.update_layout(
        title=("300M proportional domain-deletion p-value histograms " "(selected one smooth proxy per benchmark)"),
        bargap=0.03,
        height=900,
        width=1450,
        margin={"l": 80, "r": 40, "t": 110, "b": 80},
    )
    return fig


def write_slide_png(
    benchmark: pd.DataFrame,
    *,
    p_column: str,
    output_name: str,
    callout: str,
    xlabel: str,
) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["text.usetex"] = False
    values = benchmark[p_column].dropna().to_numpy(dtype=float)
    bins = np.linspace(0.0, 1.0, 21)
    counts, _ = np.histogram(values, bins=bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    n_domains = int(benchmark["n_domains"].median())
    expected = len(values) * (
        (1.0 - bins[:-1] / n_domains) ** n_domains - (1.0 - np.minimum(bins[1:], 1.0) / n_domains) ** n_domains
    )
    expected[-1] = len(values) * (1.0 - (1.0 - bins[-2] / n_domains) ** n_domains)

    fig, ax = plt.subplots(figsize=(11.5, 5.7), dpi=240)
    ax.bar(
        centers,
        counts,
        width=0.045,
        color="#2C7BB6",
        edgecolor="white",
        linewidth=1.0,
        label="observed",
    )
    ax.plot(
        centers,
        expected,
        color="#222222",
        linestyle="--",
        linewidth=2.0,
        label="independent-null reference",
    )
    ax.axvline(0.05, color="#D7191C", linestyle="-", linewidth=2.2)
    ax.text(
        0.057,
        max(counts.max(), expected.max()) * 0.92,
        "p = 0.05",
        color="#D7191C",
        fontsize=13,
        va="top",
    )
    significant = int((values <= 0.05).sum())
    ax.text(
        0.63,
        max(counts.max(), expected.max()) * 0.82,
        f"{significant}/{len(values)} smooth proxies\n{callout}",
        fontsize=17,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#C7CDD4", "alpha": 0.95},
    )
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel("benchmark count", fontsize=15)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 11))
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", color="#D9DEE7", linewidth=1.0, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(OUTPUT_DIR / output_name, bbox_inches="tight")
    plt.close(fig)


def write_cell_slide_png(cell: pd.DataFrame) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["text.usetex"] = False
    values = cell["p_harm"].dropna().to_numpy(dtype=float)
    bins = np.linspace(0.0, 1.0, 21)
    counts, _ = np.histogram(values, bins=bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    expected = np.full_like(centers, fill_value=len(values) / (len(bins) - 1), dtype=float)

    fig, ax = plt.subplots(figsize=(11.5, 5.7), dpi=240)
    ax.bar(
        centers,
        counts,
        width=0.045,
        color="#2C7BB6",
        edgecolor="white",
        linewidth=1.0,
        label="observed",
    )
    ax.plot(
        centers,
        expected,
        color="#222222",
        linestyle="--",
        linewidth=2.0,
        label="uniform-null reference",
    )
    ax.axvline(0.05, color="#D7191C", linestyle="-", linewidth=2.2)
    ax.text(
        0.057,
        max(counts.max(), expected.max()) * 0.92,
        "p = 0.05",
        color="#D7191C",
        fontsize=13,
        va="top",
    )
    significant = int((values <= 0.05).sum())
    ax.text(
        0.60,
        max(counts.max(), expected.max()) * 0.82,
        f"{significant}/{len(values)} hypotheses\nraw p <= 0.05",
        fontsize=17,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#C7CDD4", "alpha": 0.95},
    )
    ax.set_xlabel("raw one-sided harm p-value for each benchmark-domain deletion", fontsize=15)
    ax.set_ylabel("hypothesis count", fontsize=15)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 11))
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", color="#D9DEE7", linewidth=1.0, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(OUTPUT_DIR / "domain_ablation_raw_one_sided_pvalue_histogram_slide.png", bbox_inches="tight")
    plt.close(fig)


def plot_top_benchmarks(benchmark: pd.DataFrame) -> go.Figure:
    top = benchmark.sort_values("min_p_harm_sidak").head(40).copy()
    top = top.sort_values("min_p_harm_sidak", ascending=False)
    top["neg_log10_sidak"] = -np.log10(top["min_p_harm_sidak"].clip(lower=1e-300))
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top["neg_log10_sidak"],
            y=top["metric"],
            orientation="h",
            marker={"color": top["fraction_domains_harm"], "colorscale": "RdYlGn_r", "cmin": 0.0, "cmax": 1.0},
            customdata=np.stack(
                [
                    top["best_harm_domain"],
                    top["best_harm_delta"],
                    top["best_harm_t"],
                    top["min_p_harm_raw"],
                    top["min_p_harm_sidak"],
                    top["fraction_domains_harm"],
                ],
                axis=1,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "-log10 Sidak p=%{x:.3g}<br>"
                "best harm domain=%{customdata[0]}<br>"
                "best harm delta=%{customdata[1]:.5g}<br>"
                "best harm t=%{customdata[2]:.3g}<br>"
                "raw min p=%{customdata[3]:.3g}<br>"
                "Sidak p=%{customdata[4]:.3g}<br>"
                "fraction domains harm=%{customdata[5]:.3g}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Strongest benchmark-level deletion harm signals after Sidak correction",
        width=1500,
        height=1050,
        margin={"l": 520, "r": 60, "t": 80, "b": 80},
        coloraxis_colorbar={"title": "fraction domains harm"},
    )
    fig.update_xaxes(title="-log10 adjusted p-value")
    fig.update_yaxes(title="selected smooth proxy")
    return fig


def write_report(cell: pd.DataFrame, benchmark: pd.DataFrame, summary: dict[str, Any]) -> None:
    lines = [
        "# Domain-Ablation P-Value Histograms",
        "",
        "This analysis uses the 300M proportional domain-deletion panel and the pooled proportional reference distribution.",
        "The proportional reference pools the original `baseline_proportional` row with the 10 controlled proportional-repeat rows.",
        "",
        "For a deletion contrast \\(\\Delta U_{m,j}=U_m(w^{\\\\setminus j})-\\bar U_m(p)\\), the test statistic is",
        "",
        "\\[",
        "t_{m,j}=\\frac{\\Delta U_{m,j}}{\\hat\\sigma_m(p)\\sqrt{1+1/n_m}},",
        "\\]",
        "",
        "with \\(n_m-1\\) degrees of freedom. `p_harm` is one-sided for \\(\\Delta U<0\\); `p_two_sided` tests any effect.",
        "",
        "## Summary",
        "",
        f"- Selected smooth proxies: `{summary['selected_smooth_metrics']}`.",
        f"- Per-domain deletion tests: `{summary['domain_cell_tests']}`.",
        f"- Benchmarks with Sidak harm p <= 0.05: `{summary['benchmarks_sidak_harm_p_le_0p05']}`.",
        f"- Benchmarks with raw min harm p <= 0.05: `{summary['benchmarks_raw_min_harm_p_le_0p05']}`.",
        f"- Median benchmark Sidak harm p: `{summary['median_sidak_harm_p']:.4g}`.",
        "",
        "## Artifacts",
        "",
        "- `domain_ablation_cell_pvalues.csv`",
        "- `domain_ablation_benchmark_min_pvalues.csv`",
        "- `domain_ablation_pvalue_histograms.html`",
        "- `domain_ablation_top_adjusted_harm_benchmarks.html`",
        "- `domain_ablation_bonferroni_pvalue_histogram_slide.png`",
        "- `summary.json`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell-pvalues",
        type=Path,
        default=None,
        help="Precomputed domain_ablation_cell_pvalues.csv. If omitted, recompute from comparison/noise inputs.",
    )
    parser.add_argument(
        "--domain-comparison",
        type=Path,
        default=DOMAIN_COMPARISON,
        help="Domain-ablation comparison table used when --cell-pvalues is omitted.",
    )
    parser.add_argument(
        "--noise-matrix",
        type=Path,
        default=NOISE_MATRIX,
        help="Proportional-noise raw metric matrix used when --cell-pvalues is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for histogram artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cell_pvalues is None:
        comparison = pd.read_csv(args.domain_comparison, low_memory=False)
        noise_matrix = pd.read_csv(args.noise_matrix, low_memory=False)
        cell, benchmark = compute_pvalues(comparison, noise_matrix)
    else:
        cell = pd.read_csv(args.cell_pvalues, low_memory=False)
        benchmark = summarize_benchmarks(cell)

    cell.to_csv(OUTPUT_DIR / "domain_ablation_cell_pvalues.csv", index=False)
    benchmark.to_csv(OUTPUT_DIR / "domain_ablation_benchmark_min_pvalues.csv", index=False)
    plot_histograms(cell, benchmark).write_html(
        OUTPUT_DIR / "domain_ablation_pvalue_histograms.html",
        include_plotlyjs="cdn",
        config=TO_IMAGE_CONFIG,
    )
    plot_top_benchmarks(benchmark).write_html(
        OUTPUT_DIR / "domain_ablation_top_adjusted_harm_benchmarks.html",
        include_plotlyjs="cdn",
        config=TO_IMAGE_CONFIG,
    )
    write_slide_png(
        benchmark,
        p_column="min_p_harm_bonferroni",
        output_name="domain_ablation_bonferroni_pvalue_histogram_slide.png",
        callout="Bonferroni p <= 0.05",
        xlabel="Bonferroni-adjusted min p-value over 39 deleted domains",
    )
    write_slide_png(
        benchmark,
        p_column="min_p_two_sided_bonferroni",
        output_name="domain_ablation_bonferroni_two_sided_pvalue_histogram_slide_compact.png",
        callout="two-sided p <= 0.05",
        xlabel="Bonferroni-adjusted min two-sided p-value over 39 deleted domains",
    )
    write_cell_slide_png(cell)
    summary = {
        "cell_pvalues": str(args.cell_pvalues) if args.cell_pvalues is not None else None,
        "domain_comparison": str(args.domain_comparison) if args.cell_pvalues is None else None,
        "noise_matrix": str(args.noise_matrix) if args.cell_pvalues is None else None,
        "selected_smooth_metrics": int(benchmark["metric"].nunique()),
        "domain_cell_tests": int(len(cell)),
        "target_domains_per_metric_min": int(cell.groupby("metric")["target_domain"].nunique().min()),
        "target_domains_per_metric_max": int(cell.groupby("metric")["target_domain"].nunique().max()),
        "benchmarks_raw_min_harm_p_le_0p05": int((benchmark["min_p_harm_raw"] <= 0.05).sum()),
        "benchmarks_sidak_harm_p_le_0p05": int((benchmark["min_p_harm_sidak"] <= 0.05).sum()),
        "benchmarks_bonferroni_harm_p_le_0p05": int((benchmark["min_p_harm_bonferroni"] <= 0.05).sum()),
        "benchmarks_raw_min_two_sided_p_le_0p05": int((benchmark["min_p_two_sided_raw"] <= 0.05).sum()),
        "benchmarks_sidak_two_sided_p_le_0p05": int((benchmark["min_p_two_sided_sidak"] <= 0.05).sum()),
        "median_raw_min_harm_p": float(benchmark["min_p_harm_raw"].median()),
        "median_sidak_harm_p": float(benchmark["min_p_harm_sidak"].median()),
        "median_cell_p_harm": float(cell["p_harm"].median()),
        "median_cell_p_two_sided": float(cell["p_two_sided"].median()),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(cell, benchmark, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
