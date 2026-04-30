#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF003
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot cross-judge Spearman correlation results for GPT-4.1, GPT-5.1, and
Gemini 3 Flash.

Reads `~/judge_correlations/outputs/spearman_per_statement.json` (written
by `judge_spearman.py analyze`) and produces two figures under
`plot/output/`:

  1. `judge_correlation_summary.{pdf,png}` — compact summary. Violin
     distribution of per-statement Spearman for each of the 3 pairs
     among {gpt41, gpt51, gem3f}, with median line and summary stats.
  2. `judge_correlation_per_statement.{pdf,png}` — detailed breakdown.
     Grouped bar chart, one group per statement, sorted descending by
     the gpt41↔gpt51 Spearman.

Usage:
    uv run python experiments/posttrain/plot_judge_correlation.py

See `.agents/logbooks/gpt5_correlation.md` EXP-029 for context.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

# Pairs we care about (user said they don't care about goss)
PAIRS = [
    ("gpt41_vs_gpt51", "GPT-4.1 ↔ GPT-5.1", "#1f77b4"),
    ("gpt41_vs_gem3f", "GPT-4.1 ↔ Gemini 3 Flash", "#2ca02c"),
    ("gpt51_vs_gem3f", "GPT-5.1 ↔ Gemini 3 Flash", "#d62728"),
]


def load_data() -> dict:
    if not SPEARMAN_JSON.exists():
        raise SystemExit(f"Missing {SPEARMAN_JSON}. Run `judge_spearman.py analyze` first.")
    return json.loads(SPEARMAN_JSON.read_text())


def extract_spearmans(data: dict, pair_key: str) -> dict[str, float]:
    """Return {statement_id: spearman} for one pair, dropping Nones."""
    per_stmt = data["pairs"][pair_key]["per_statement"]
    return {bid: v["spearman"] for bid, v in per_stmt.items() if v["spearman"] is not None}


def plot_summary(data: dict) -> None:
    """Violin plot of per-statement Spearman distribution for each pair."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    pair_values = []
    pair_labels = []
    pair_colors = []
    pair_summaries = []
    for pair_key, label, color in PAIRS:
        spearmans = list(extract_spearmans(data, pair_key).values())
        pair_values.append(spearmans)
        pair_labels.append(label)
        pair_colors.append(color)

        summary = data["pairs"][pair_key]["summary"]
        pair_summaries.append(summary)

    positions = list(range(1, len(PAIRS) + 1))

    # Violin bodies
    vp = ax.violinplot(
        pair_values,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.7,
    )
    for body, color in zip(vp["bodies"], pair_colors, strict=True):
        body.set_facecolor(color)
        body.set_alpha(0.35)
        body.set_edgecolor("black")
        body.set_linewidth(0.8)

    # Overlay individual per-statement points (strip)
    rng = np.random.default_rng(42)
    for pos, values, color in zip(positions, pair_values, pair_colors, strict=True):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=14,
            color=color,
            edgecolor="black",
            linewidth=0.3,
            alpha=0.85,
            zorder=3,
        )

    # Median + mean line per pair
    for pos, values in zip(positions, pair_values, strict=True):
        med = float(np.median(values))
        mean = float(np.mean(values))
        ax.hlines(med, pos - 0.35, pos + 0.35, colors="black", linewidth=2.2, zorder=4)
        ax.hlines(
            mean,
            pos - 0.25,
            pos + 0.25,
            colors="white",
            linewidth=1.4,
            linestyles="--",
            zorder=5,
        )

    # Annotate summary stats under each violin
    for pos, summary in zip(positions, pair_summaries, strict=True):
        s = summary["spearman"]
        label = (
            f"median {s['median']:.3f}\n"
            f"mean {s['mean']:.3f}\n"
            f"n={summary['n_statements']} stmts\n"
            f"{s['frac_ge_0.7']:.0%} ≥ 0.7"
        )
        ax.text(
            pos,
            -0.05,
            label,
            ha="center",
            va="top",
            fontsize=7.5,
            transform=ax.get_xaxis_transform(),
        )

    ax.axhline(0.5, color="gray", linewidth=0.6, linestyle=":", alpha=0.7)
    ax.axhline(0.7, color="gray", linewidth=0.6, linestyle=":", alpha=0.7)
    ax.axhline(0.9, color="gray", linewidth=0.6, linestyle=":", alpha=0.7)
    ax.text(
        len(PAIRS) + 0.5,
        0.5,
        "ρ=0.5",
        fontsize=7,
        color="gray",
        va="center",
        ha="left",
    )
    ax.text(
        len(PAIRS) + 0.5,
        0.7,
        "ρ=0.7",
        fontsize=7,
        color="gray",
        va="center",
        ha="left",
    )
    ax.text(
        len(PAIRS) + 0.5,
        0.9,
        "ρ=0.9",
        fontsize=7,
        color="gray",
        va="center",
        ha="left",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(pair_labels)
    ax.set_ylabel("Per-statement Spearman rank correlation")
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlim(0.5, len(PAIRS) + 1.0)
    ax.set_title(
        "Cross-judge correlation on 43 shared statements (4 targets pooled)\n"
        "Black bar = median, dashed white = mean; dots = individual statements",
        fontsize=9,
    )
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)

    plt.tight_layout()
    pdf_path = OUT_DIR / "judge_correlation_summary.pdf"
    png_path = OUT_DIR / "judge_correlation_summary.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


def plot_per_statement(data: dict) -> None:
    """Grouped bar chart: 1 group per statement, 3 bars per group.

    Sorted descending by the gpt41↔gpt51 Spearman so the x-axis shows
    the ordering of hardest→easiest rubrics for the reference pair.
    Missing bars (gem3f-excluded statements) appear as gaps.
    """
    # Use the union of statements across all 3 pairs; sort by gpt41↔gpt51
    # where available, then fall back.
    by_pair = {k: extract_spearmans(data, k) for k, _, _ in PAIRS}
    all_stmts = sorted(set().union(*[set(v.keys()) for v in by_pair.values()]))

    ref = by_pair["gpt41_vs_gpt51"]
    # Sort statements descending by gpt41↔gpt51 ρ; statements missing
    # from the reference pair (shouldn't happen after UNIVERSAL_SKIP)
    # sort last.
    sorted_stmts = sorted(
        all_stmts,
        key=lambda b: (-(ref.get(b, -1.0)), b),
    )

    n = len(sorted_stmts)
    width = 0.26
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(10.5, 0.22 * n + 2), 5.2))

    for i, (pair_key, label, color) in enumerate(PAIRS):
        vals = [by_pair[pair_key].get(bid, np.nan) for bid in sorted_stmts]
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width=width, color=color, label=label, edgecolor="black", linewidth=0.3)

    ax.axhline(0.5, color="gray", linewidth=0.6, linestyle=":", alpha=0.7)
    ax.axhline(0.7, color="gray", linewidth=0.6, linestyle=":", alpha=0.7)
    ax.axhline(0.9, color="gray", linewidth=0.6, linestyle=":", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_stmts, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Spearman ρ")
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_title(
        "Per-statement cross-judge Spearman (sorted by GPT-4.1↔GPT-5.1, 4 targets pooled)\n"
        "Gaps in green/red bars = sexual_content_involving_minors (gem3f-only skip)",
        fontsize=9,
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)

    plt.tight_layout()
    pdf_path = OUT_DIR / "judge_correlation_per_statement.pdf"
    png_path = OUT_DIR / "judge_correlation_per_statement.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


def main() -> None:
    data = load_data()
    plot_summary(data)
    plot_per_statement(data)


if __name__ == "__main__":
    main()
