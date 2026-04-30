#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003, NPY002
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-cluster cross-judge Spearman correlation analysis.

Reads per-statement Spearman results from `judge_spearman.py analyze` and
aggregates them by the semantic clusters in `statement_clusters.py`.
Answers "how does judge agreement vary by rubric domain?"

Outputs:
  1. stdout: per-cluster × per-pair median Spearman table
  2. `plot/output/judge_correlation_by_cluster_heatmap.{pdf,png}` —
     heatmap (clusters × pairs)
  3. `plot/output/judge_correlation_by_cluster_bars.{pdf,png}` —
     grouped bar chart: one cluster per group, one bar per pair

Usage:
    uv run --with matplotlib --with numpy python \\
        experiments/posttrain/plot_judge_correlation_by_cluster.py

See `.agents/logbooks/gpt5_correlation.md` EXP-029b for context.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Import the canonical semantic clusters.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from statement_clusters import SEMANTIC_CLUSTERS

matplotlib.rcParams.update(
    {
        "font.size": 9,
        "font.family": "serif",
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
    }
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEARMAN_JSON = Path.home() / "judge_correlations" / "outputs" / "spearman_per_statement.json"
OUT_DIR = REPO_ROOT / "plot" / "output"

# Pairs to show, in the order they appear in the figure columns.
# Only pairs that involve at least one GPT judge — we don't care about
# Gemini↔Gemini self-agreement for this analysis. Goss pairs also skipped.
PAIRS = [
    ("gpt41_vs_gpt51", "GPT-4.1 ↔ GPT-5.1"),
    ("gpt41_vs_gem3f", "GPT-4.1 ↔ Gem-3-Flash"),
    ("gpt41_vs_gem31p", "GPT-4.1 ↔ Gem-3.1-Pro"),
    ("gpt51_vs_gem3f", "GPT-5.1 ↔ Gem-3-Flash"),
    ("gpt51_vs_gem31p", "GPT-5.1 ↔ Gem-3.1-Pro"),
]

CLUSTER_ORDER = list(SEMANTIC_CLUSTERS.keys())
CLUSTER_LABELS = {
    "safety_and_legality": "Safety & Legality",
    "privacy_and_trust": "Privacy & Trust",
    "politics_and_neutrality": "Politics & Neutrality",
    "epistemics_and_honesty": "Epistemics & Honesty",
    "style_and_tone": "Style & Tone",
    "service_and_execution": "Service & Execution",
}


def load_data() -> dict:
    if not SPEARMAN_JSON.exists():
        raise SystemExit(f"Missing {SPEARMAN_JSON}. Run `judge_spearman.py analyze` first.")
    return json.loads(SPEARMAN_JSON.read_text())


def per_statement_rho(data: dict, pair_key: str) -> dict[str, float]:
    """Return {statement: spearman} for one pair, dropping Nones."""
    return {
        bid: v["spearman"] for bid, v in data["pairs"][pair_key]["per_statement"].items() if v["spearman"] is not None
    }


def cluster_aggregates(
    data: dict,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, list[float]]]]:
    """For each (cluster, pair), compute median ρ AND collect the raw
    per-statement ρ list (used for the bar chart error bars and the
    strip overlay)."""
    median_matrix: dict[str, dict[str, float]] = {}
    raw_matrix: dict[str, dict[str, list[float]]] = {}
    for cluster, stmts in SEMANTIC_CLUSTERS.items():
        median_matrix[cluster] = {}
        raw_matrix[cluster] = {}
        for pair_key, _ in PAIRS:
            rhos = per_statement_rho(data, pair_key)
            vals = [rhos[s] for s in stmts if s in rhos]
            median_matrix[cluster][pair_key] = statistics.median(vals) if vals else float("nan")
            raw_matrix[cluster][pair_key] = vals
    return median_matrix, raw_matrix


def print_table(median_matrix: dict[str, dict[str, float]]) -> None:
    """Print the cluster × pair median Spearman table."""
    # Header
    print()
    print(f"  {'Cluster':<26} {'n':>3}  " + "  ".join(f"{label[:12]:>12}" for _, label in PAIRS))
    print("  " + "-" * (32 + 14 * len(PAIRS)))
    for cluster in CLUSTER_ORDER:
        n = len(SEMANTIC_CLUSTERS[cluster])
        row = " ".join(
            (
                f"{median_matrix[cluster][pk]:>12.3f}"
                if median_matrix[cluster][pk] == median_matrix[cluster][pk]
                else f"{'N/A':>12}"
            )
            for pk, _ in PAIRS
        )
        print(f"  {CLUSTER_LABELS[cluster]:<26} {n:>3}  {row}")
    print()


def plot_heatmap(median_matrix: dict[str, dict[str, float]]) -> None:
    """6 clusters × 6 pairs heatmap of median Spearman."""
    n_rows = len(CLUSTER_ORDER)
    n_cols = len(PAIRS)

    grid = np.full((n_rows, n_cols), np.nan)
    for i, cluster in enumerate(CLUSTER_ORDER):
        for j, (pk, _) in enumerate(PAIRS):
            v = median_matrix[cluster][pk]
            if v == v:  # not nan
                grid[i, j] = v

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0.3, vmax=0.95, aspect="auto")

    # Annotate cells with the numeric value
    for i in range(n_rows):
        for j in range(n_cols):
            val = grid[i, j]
            if val == val:
                # Choose text color for contrast
                text_color = "white" if val < 0.55 or val > 0.85 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                    fontweight="bold",
                )

    # Axis labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([label for _, label in PAIRS], rotation=35, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"{CLUSTER_LABELS[c]}  (n={len(SEMANTIC_CLUSTERS[c])})" for c in CLUSTER_ORDER])

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Median per-statement Spearman ρ", fontsize=9)

    ax.set_title(
        "Cross-judge agreement by semantic cluster\n"
        "Rows = rubric clusters; columns = judge pairs; color + number = median ρ over statements in cluster",
        fontsize=10,
        pad=12,
    )

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "judge_correlation_by_cluster_heatmap.pdf"
    png = OUT_DIR / "judge_correlation_by_cluster_heatmap.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


def plot_bars(
    median_matrix: dict[str, dict[str, float]],
    raw_matrix: dict[str, dict[str, list[float]]],
) -> None:
    """Grouped bar chart: one cluster per group, one bar per pair.
    Overlay individual per-statement ρ as strip points."""
    n_pairs = len(PAIRS)
    bar_width = 0.78 / n_pairs
    x_groups = np.arange(len(CLUSTER_ORDER))

    # Color per pair — roughly grouped by "which GPT" and distinguished
    # by Gemini variant
    pair_colors = {
        "gpt41_vs_gpt51": "#444444",
        "gpt41_vs_gem3f": "#1f77b4",
        "gpt41_vs_gem31p": "#2ca02c",
        "gpt51_vs_gem3f": "#ff7f0e",
        "gpt51_vs_gem31p": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(12, 5.2))

    for i, (pk, label) in enumerate(PAIRS):
        vals = [median_matrix[c][pk] for c in CLUSTER_ORDER]
        offsets = x_groups + (i - (n_pairs - 1) / 2) * bar_width
        ax.bar(
            offsets,
            vals,
            width=bar_width,
            color=pair_colors[pk],
            edgecolor="black",
            linewidth=0.3,
            label=label,
            zorder=2,
        )

        # Strip overlay: individual per-statement ρ as translucent dots
        for j, c in enumerate(CLUSTER_ORDER):
            dots = raw_matrix[c][pk]
            if dots:
                x = offsets[j] + np.random.uniform(-bar_width * 0.15, bar_width * 0.15, size=len(dots))
                ax.scatter(
                    x,
                    dots,
                    s=6,
                    color=pair_colors[pk],
                    edgecolor="black",
                    linewidth=0.2,
                    alpha=0.55,
                    zorder=3,
                )

    # Horizontal reference lines
    for thresh in (0.5, 0.7, 0.9):
        ax.axhline(thresh, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)

    ax.set_xticks(x_groups)
    ax.set_xticklabels(
        [f"{CLUSTER_LABELS[c]}\n(n={len(SEMANTIC_CLUSTERS[c])})" for c in CLUSTER_ORDER],
        fontsize=8.5,
    )
    ax.set_ylabel("Per-statement Spearman ρ  (bar = median; dots = statements)")
    ax.set_ylim(-0.15, 1.0)
    ax.set_xlim(-0.55, len(CLUSTER_ORDER) - 0.45)
    ax.legend(loc="lower right", framealpha=0.9, ncol=2, fontsize=7.5)
    ax.set_title(
        "Per-cluster cross-judge agreement\n"
        "Bars = median ρ within cluster; translucent dots = each rubric's per-statement ρ",
        fontsize=10,
    )
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5, zorder=0)

    plt.tight_layout()
    pdf = OUT_DIR / "judge_correlation_by_cluster_bars.pdf"
    png = OUT_DIR / "judge_correlation_by_cluster_bars.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


def main() -> None:
    np.random.seed(42)  # deterministic strip jitter
    data = load_data()
    median_matrix, raw_matrix = cluster_aggregates(data)

    print("Per-cluster × per-pair median Spearman ρ:")
    print_table(median_matrix)
    plot_heatmap(median_matrix)
    plot_bars(median_matrix, raw_matrix)


if __name__ == "__main__":
    main()
