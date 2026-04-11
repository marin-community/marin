# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas"]
# ///
"""Plot observed mixture weights for a selected pair of observed runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (
    plot_ccpairtotal_vs_ccglobalpremium_vs_best as comparison_plot,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.plot_grp_vs_proportional import (
    TEXT_MUTED_COLOR,
    _cc_topic_and_quality,
    _display_non_cc_label,
    _plot_cc_block,
    _plot_non_cc_block,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    load_two_phase_many_packet,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_PNG = SCRIPT_DIR / "run00125_vs_run00018_weights.png"
WEIGHTS_CSV = SCRIPT_DIR / "run00125_vs_run00018_weights.csv"
BASE_CSV = SCRIPT_DIR / "two_phase_many.csv"
COMPLETED_CSV = SCRIPT_DIR / "qsplit240_300m_6b_completed_vs_60m.csv"
COMPLETED_SUMMARY_JSON = SCRIPT_DIR / "qsplit240_300m_6b_completed_vs_60m_summary.json"

COLOR_MAP = plt.get_cmap("RdYlGn_r")
DEFAULT_RUN_NAMES = ("run_00125", "run_00018")
DEFAULT_RUN_COLORS = {
    "run_00125": COLOR_MAP(0.86),
    "run_00018": COLOR_MAP(0.12),
}


def _display_label(run_name: str) -> str:
    if run_name.startswith("run_"):
        return run_name.replace("run_", "mix_", 1)
    return run_name


@dataclass(frozen=True)
class RunSummary:
    run_name: str
    bpb_60m: float
    rank_60m: int
    bpb_300m_6b: float | None
    rank_300m_6b: int | None


def default_run_color(run_name: str, *, order_index: int, total_runs: int) -> str:
    if run_name in DEFAULT_RUN_COLORS:
        return DEFAULT_RUN_COLORS[run_name]
    if total_runs <= 1:
        return COLOR_MAP(0.5)
    return COLOR_MAP(order_index / (total_runs - 1))


def _domain_order(domain_names: list[str], schedules: list[tuple[str, np.ndarray, str]]) -> tuple[list[int], list[int]]:
    non_cc_indices, cc_indices = comparison_plot._split_domain_blocks(domain_names)
    non_cc_sorted = sorted(
        non_cc_indices,
        key=lambda idx: max(float(weights[0, idx]) for _, weights, _ in schedules),
        reverse=True,
    )

    cc_topics: list[tuple[str, int, int, float]] = []
    for i in range(0, len(cc_indices), 2):
        hi = cc_indices[i]
        lo = cc_indices[i + 1]
        topic, _ = _cc_topic_and_quality(domain_names[hi])
        score = max(
            max(float(weights[0, hi]), float(weights[1, hi]), float(weights[0, lo]), float(weights[1, lo]))
            for _, weights, _ in schedules
        )
        cc_topics.append((topic, hi, lo, score))
    cc_topics.sort(key=lambda item: (-item[3], item[0]))

    cc_sorted: list[int] = []
    for _, hi, lo, _ in cc_topics:
        cc_sorted.extend([hi, lo])
    return non_cc_sorted, cc_sorted


def load_run_summaries(run_names: tuple[str, ...]) -> dict[str, RunSummary]:
    base = (
        pd.read_csv(BASE_CSV)[["run_name", MANY_DOMAIN_TARGET]]
        .dropna()
        .sort_values(MANY_DOMAIN_TARGET, ascending=True)
        .reset_index(drop=True)
    )
    base["rank_60m"] = base.index + 1
    completed = pd.read_csv(COMPLETED_CSV)

    summaries: dict[str, RunSummary] = {}
    for run_name in run_names:
        base_row = base[base["run_name"] == run_name].iloc[0]
        completed_rows = completed[completed["run_name"] == run_name]
        summaries[run_name] = RunSummary(
            run_name=run_name,
            bpb_60m=float(base_row[MANY_DOMAIN_TARGET]),
            rank_60m=int(base_row["rank_60m"]),
            bpb_300m_6b=(None if completed_rows.empty else float(completed_rows.iloc[0]["bpb_300m_6b"])),
            rank_300m_6b=(None if completed_rows.empty else int(completed_rows.iloc[0]["rank_300m_6b"])),
        )
    return summaries


def _format_300m_summary(summary: RunSummary, *, completed_count: int) -> str:
    if summary.bpb_300m_6b is None or summary.rank_300m_6b is None:
        return "300M/6B final pending"
    return f"300M/6B {summary.bpb_300m_6b:.4f} BPB (rank {summary.rank_300m_6b}/{completed_count} completed)"


def build_two_run_weights_plot(
    *,
    run_names: tuple[str, str],
    output_png: Path,
    output_csv: Path,
    run_colors: dict[str, str] | None = None,
) -> None:
    data = load_two_phase_many_packet(target=MANY_DOMAIN_TARGET)
    summaries = load_run_summaries(run_names)
    completed_summary = json.loads(COMPLETED_SUMMARY_JSON.read_text())
    completed_count = int(completed_summary["n_completed"])
    resolved_run_colors = {
        run_name: (
            run_colors[run_name]
            if run_colors is not None and run_name in run_colors
            else default_run_color(run_name, order_index=index, total_runs=len(run_names))
        )
        for index, run_name in enumerate(run_names)
    }

    schedules: list[tuple[str, np.ndarray, str]] = []
    csv_rows: list[dict[str, object]] = []
    for run_name in run_names:
        row_idx = int(data.frame.index[data.frame["run_name"] == run_name][0])
        weights = comparison_plot._row_weights(data.frame, data.domain_names, row_idx)
        schedules.append((_display_label(run_name), weights, resolved_run_colors[run_name]))
        summary = summaries[run_name]
        for phase_idx, phase_name in enumerate(("phase_0", "phase_1")):
            phase_multipliers = data.c0 if phase_idx == 0 else data.c1
            for domain_idx, domain_name in enumerate(data.domain_names):
                csv_rows.append(
                    {
                        "run_name": run_name,
                        "phase": phase_name,
                        "domain_name": domain_name,
                        "weight": float(weights[phase_idx, domain_idx]),
                        "effective_epochs": float(weights[phase_idx, domain_idx] * phase_multipliers[domain_idx]),
                        "bpb_60m": summary.bpb_60m,
                        "rank_60m": summary.rank_60m,
                        "bpb_300m_6b": summary.bpb_300m_6b,
                        "rank_300m_6b_completed": summary.rank_300m_6b,
                    }
                )

    non_cc_indices, cc_indices = _domain_order(data.domain_names, schedules)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(27, 22),
        gridspec_kw={"width_ratios": [1.0, 1.62], "hspace": 0.22, "wspace": 0.31},
        facecolor="white",
    )

    _plot_non_cc_block(
        ax=axes[0, 0],
        indices=non_cc_indices,
        labels=[_display_non_cc_label(data.domain_names[idx]) for idx in non_cc_indices],
        schedules=schedules,
        phase_idx=0,
        multipliers=data.c0,
        title="Phase 0: Non-CC Domains",
        show_legend=True,
    )
    _plot_cc_block(
        ax=axes[0, 1],
        domain_names=data.domain_names,
        indices=cc_indices,
        schedules=schedules,
        phase_idx=0,
        multipliers=data.c0,
        title="Phase 0: CC Domains",
    )
    _plot_non_cc_block(
        ax=axes[1, 0],
        indices=non_cc_indices,
        labels=[_display_non_cc_label(data.domain_names[idx]) for idx in non_cc_indices],
        schedules=schedules,
        phase_idx=1,
        multipliers=data.c1,
        title="Phase 1: Non-CC Domains",
        show_legend=False,
    )
    _plot_cc_block(
        ax=axes[1, 1],
        domain_names=data.domain_names,
        indices=cc_indices,
        schedules=schedules,
        phase_idx=1,
        multipliers=data.c1,
        title="Phase 1: CC Domains",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    left_run_name, right_run_name = run_names
    fig.suptitle(
        f"{_display_label(left_run_name)} vs {_display_label(right_run_name)}",
        fontsize=34,
        y=0.996,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.952,
        "Observed 60M two-phase mixtures, annotated with uncheatable-eval BPB at 60M and 300M/6B",
        ha="center",
        va="center",
        fontsize=20,
        color=TEXT_MUTED_COLOR,
    )
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=18.5, bbox_to_anchor=(0.5, 0.928))

    left_summary = summaries[left_run_name]
    right_summary = summaries[right_run_name]
    fig.text(
        0.5,
        0.073,
        (
            f"{_display_label(left_run_name)}: 60M {left_summary.bpb_60m:.4f} BPB "
            f"(rank {left_summary.rank_60m}/241)   |   "
            f"{_format_300m_summary(left_summary, completed_count=completed_count)}\n"
            f"{_display_label(right_run_name)}: 60M {right_summary.bpb_60m:.4f} BPB "
            f"(rank {right_summary.rank_60m}/241)   |   "
            f"{_format_300m_summary(right_summary, completed_count=completed_count)}"
        ),
        ha="center",
        va="center",
        fontsize=17.5,
        color="#0f172a",
        bbox={
            "boxstyle": "round,pad=0.62,rounding_size=0.18",
            "facecolor": "#f8fafc",
            "edgecolor": "#cbd5e1",
            "alpha": 0.97,
        },
    )
    fig.text(
        0.5,
        0.026,
        (
            "Bar-end labels show effective epochs for that domain in that phase. "
            f"300M/6B ranks are among the {completed_count} terminal-success mixtures only."
        ),
        ha="center",
        va="center",
        fontsize=15,
        color=TEXT_MUTED_COLOR,
    )
    fig.subplots_adjust(top=0.905, left=0.14, right=0.985, bottom=0.13, hspace=0.24, wspace=0.31)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(csv_rows).to_csv(output_csv, index=False)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_csv}")


def main() -> None:
    build_two_run_weights_plot(
        run_names=DEFAULT_RUN_NAMES,
        output_png=PLOT_PNG,
        output_csv=WEIGHTS_CSV,
        run_colors=DEFAULT_RUN_COLORS,
    )


if __name__ == "__main__":
    main()
