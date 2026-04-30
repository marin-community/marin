#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-statement Pareto comparison: Gemini 3 Flash vs Gemini 3.1 Pro.

Answers "on which statements is each Gemini a better proxy for each GPT judge?"
by computing the per-statement Spearman difference (gem31p - gem3f) for each
of the two GPT judges (gpt41, gpt51).

Reads `~/judge_correlations/outputs/spearman_per_statement.json` (written by
`judge_spearman.py analyze`) and produces:

  1. `plot/output/gemini_pareto_scatter.{pdf,png}` — scatter of
     gem3f ρ vs gem31p ρ, one point per statement, two panels (vs gpt41
     and vs gpt51). Points above the y=x line = gem31p wins.
  2. `plot/output/gemini_pareto_deltas.{pdf,png}` — per-statement
     bar chart of Δρ = gem31p_ρ − gem3f_ρ, sorted. Two panels.
  3. stdout: full text tables listing every statement where each Gemini wins
     (sorted by absolute delta) for both GPT judges.

Usage:
    uv run --with matplotlib --with numpy python \\
        experiments/posttrain/plot_gemini_pareto.py

See EXP-029b in `.agents/logbooks/gpt5_correlation.md` for context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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

GPT_JUDGES = [
    ("gpt41", "GPT-4.1", "#1f77b4"),
    ("gpt51", "GPT-5.1", "#d62728"),
]

TIE_THRESHOLD = 0.01  # |Δρ| below this is considered a tie


@dataclass(frozen=True)
class StatementDelta:
    statement: str
    flash_rho: float
    pro_rho: float
    delta: float  # pro - flash; positive = gem31p wins


def load_pair(data: dict, pair_key: str) -> dict[str, float]:
    """Return {statement: spearman} for one pair, dropping Nones."""
    return {
        bid: v["spearman"] for bid, v in data["pairs"][pair_key]["per_statement"].items() if v["spearman"] is not None
    }


def compute_deltas(data: dict, gpt_judge: str) -> list[StatementDelta]:
    """Per-statement Δρ = gem31p_ρ − gem3f_ρ for one GPT judge."""
    flash = load_pair(data, f"{gpt_judge}_vs_gem3f")
    pro = load_pair(data, f"{gpt_judge}_vs_gem31p")
    shared = sorted(set(flash.keys()) & set(pro.keys()))
    return [
        StatementDelta(
            statement=s,
            flash_rho=flash[s],
            pro_rho=pro[s],
            delta=pro[s] - flash[s],
        )
        for s in shared
    ]


def plot_scatter(all_deltas: dict[str, list[StatementDelta]]) -> None:
    """Scatter of flash ρ vs pro ρ, one panel per GPT judge."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), sharey=True)

    for ax, (judge_key, judge_label, color) in zip(axes, GPT_JUDGES, strict=True):  # noqa: B007
        deltas = all_deltas[judge_key]
        flash_rhos = np.array([d.flash_rho for d in deltas])
        pro_rhos = np.array([d.pro_rho for d in deltas])

        # Color by which model wins
        colors = [
            "#2ca02c" if d.delta > TIE_THRESHOLD else "#ff7f0e" if d.delta < -TIE_THRESHOLD else "#888888"
            for d in deltas
        ]

        # y=x reference line
        lo, hi = -0.15, 0.95
        ax.plot([lo, hi], [lo, hi], "k:", linewidth=0.7, alpha=0.6, zorder=1)

        # Tie band (|Δ| < TIE_THRESHOLD)
        ax.fill_between(
            [lo, hi],
            [lo - TIE_THRESHOLD, hi - TIE_THRESHOLD],
            [lo + TIE_THRESHOLD, hi + TIE_THRESHOLD],
            color="gray",
            alpha=0.1,
            zorder=0,
        )

        ax.scatter(
            flash_rhos,
            pro_rhos,
            s=35,
            c=colors,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            zorder=2,
        )

        # Label the most extreme points on either side
        extremes = sorted(deltas, key=lambda d: abs(d.delta), reverse=True)[:6]
        for d in extremes:
            ax.annotate(
                d.statement,
                (d.flash_rho, d.pro_rho),
                textcoords="offset points",
                xytext=(5, 5 if d.delta > 0 else -12),
                fontsize=6.5,
                alpha=0.85,
            )

        # Win count annotation
        n_pro = sum(1 for d in deltas if d.delta > TIE_THRESHOLD)
        n_flash = sum(1 for d in deltas if d.delta < -TIE_THRESHOLD)
        n_tied = sum(1 for d in deltas if abs(d.delta) <= TIE_THRESHOLD)

        ax.text(
            0.02,
            0.97,
            f"gem31p wins:  {n_pro}  (green, above y=x)\n"
            f"gem3f  wins:  {n_flash}  (orange, below y=x)\n"
            f"tied:         {n_tied}  (gray, |Δ| < {TIE_THRESHOLD})",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
        )

        ax.set_xlabel(f"Gemini 3 Flash ↔ {judge_label}  (ρ)")
        if ax is axes[0]:
            ax.set_ylabel("Gemini 3.1 Pro ↔ GPT judge  (ρ)")
        ax.set_title(f"vs {judge_label}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_aspect("equal")

    fig.suptitle(
        "Per-statement Spearman: is Gemini 3.1 Pro a Pareto improvement over Gemini 3 Flash?\n"
        "Points above the y=x diagonal = gem31p correlates better with the GPT judge.",
        fontsize=10,
    )
    plt.tight_layout()
    pdf_path = OUT_DIR / "gemini_pareto_scatter.pdf"
    png_path = OUT_DIR / "gemini_pareto_scatter.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


def plot_deltas(all_deltas: dict[str, list[StatementDelta]]) -> None:
    """Per-statement Δρ bar chart, one panel per GPT judge, sorted."""
    n_stmts = len(next(iter(all_deltas.values())))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(11, 0.22 * n_stmts + 2), 7.5),
        sharex=False,
    )

    for ax, (judge_key, judge_label, _color) in zip(axes, GPT_JUDGES, strict=True):
        deltas = sorted(all_deltas[judge_key], key=lambda d: -d.delta)
        xs = np.arange(len(deltas))
        ys = np.array([d.delta for d in deltas])
        colors = [
            "#2ca02c" if d.delta > TIE_THRESHOLD else "#ff7f0e" if d.delta < -TIE_THRESHOLD else "#888888"
            for d in deltas
        ]

        ax.bar(xs, ys, color=colors, edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(TIE_THRESHOLD, color="gray", linewidth=0.4, linestyle=":", alpha=0.6)
        ax.axhline(-TIE_THRESHOLD, color="gray", linewidth=0.4, linestyle=":", alpha=0.6)

        ax.set_xticks(xs)
        ax.set_xticklabels([d.statement for d in deltas], rotation=60, ha="right", fontsize=6.5)
        ax.set_ylabel(r"$\rho_{pro}$ − $\rho_{flash}$")
        ax.set_title(
            f"vs {judge_label}    "
            f"(gem31p wins {sum(1 for d in deltas if d.delta > TIE_THRESHOLD)} / "
            f"gem3f wins {sum(1 for d in deltas if d.delta < -TIE_THRESHOLD)} / "
            f"tied {sum(1 for d in deltas if abs(d.delta) <= TIE_THRESHOLD)})",
            fontsize=9,
        )
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        ax.set_xlim(-0.6, len(deltas) - 0.4)

    fig.suptitle(
        "Per-statement Δρ = Spearman(gem31p) − Spearman(gem3f)\n"
        "Green bars: gem31p wins  |  Orange bars: gem3f wins  |  Gray: tied",
        fontsize=10,
    )
    plt.tight_layout()
    pdf_path = OUT_DIR / "gemini_pareto_deltas.pdf"
    png_path = OUT_DIR / "gemini_pareto_deltas.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


def print_tables(all_deltas: dict[str, list[StatementDelta]]) -> None:
    """Text summary: sorted lists of wins for each Gemini, per GPT judge."""
    for judge_key, judge_label, _ in GPT_JUDGES:
        deltas = all_deltas[judge_key]
        n_pro = sum(1 for d in deltas if d.delta > TIE_THRESHOLD)
        n_flash = sum(1 for d in deltas if d.delta < -TIE_THRESHOLD)
        n_tied = sum(1 for d in deltas if abs(d.delta) <= TIE_THRESHOLD)

        print(f"\n{'=' * 84}")
        print(f"  vs {judge_label}  —  gem31p wins {n_pro} | gem3f wins {n_flash} | tied {n_tied}")
        print("=" * 84)

        pro_wins = sorted(
            [d for d in deltas if d.delta > TIE_THRESHOLD],
            key=lambda d: -d.delta,
        )
        flash_wins = sorted(
            [d for d in deltas if d.delta < -TIE_THRESHOLD],
            key=lambda d: d.delta,
        )

        print(f"\n  Gemini 3.1 Pro wins ({len(pro_wins)} statements, sorted by Δρ):")
        print(f"    {'statement':<42} {'ρ_flash':>8} {'ρ_pro':>8} {'Δρ':>8}")
        for d in pro_wins:
            print(f"    {d.statement:<42} {d.flash_rho:>8.3f} {d.pro_rho:>8.3f} {d.delta:>+8.3f}")

        print(f"\n  Gemini 3 Flash wins ({len(flash_wins)} statements, sorted by |Δρ|):")
        print(f"    {'statement':<42} {'ρ_flash':>8} {'ρ_pro':>8} {'Δρ':>8}")
        for d in flash_wins:
            print(f"    {d.statement:<42} {d.flash_rho:>8.3f} {d.pro_rho:>8.3f} {d.delta:>+8.3f}")


def main() -> None:
    if not SPEARMAN_JSON.exists():
        raise SystemExit(f"Missing {SPEARMAN_JSON}. Run `judge_spearman.py analyze` first.")
    data = json.loads(SPEARMAN_JSON.read_text())

    all_deltas = {judge_key: compute_deltas(data, judge_key) for judge_key, _, _ in GPT_JUDGES}

    # Summary
    for judge_key, judge_label, _ in GPT_JUDGES:
        deltas = all_deltas[judge_key]
        mean_delta = sum(d.delta for d in deltas) / len(deltas)
        median_delta = sorted(d.delta for d in deltas)[len(deltas) // 2]
        print(
            f"{judge_label}: n={len(deltas)} statements, "
            f"mean Δρ = {mean_delta:+.4f}, median Δρ = {median_delta:+.4f}  "
            f"(positive ⇒ gem31p better)"
        )

    print_tables(all_deltas)
    plot_scatter(all_deltas)
    plot_deltas(all_deltas)


if __name__ == "__main__":
    main()
