# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas"]
# ///
"""Plot completed 300M/6B vs 60M rank and BPB relationships."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

plt.rcParams["text.usetex"] = False

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "qsplit240_300m_6b_completed_vs_60m.csv"
INPUT_SUMMARY_JSON = ROOT / "qsplit240_300m_6b_completed_vs_60m_summary.json"
INPUT_60M_ALL_RUNS_CSV = ROOT / "two_phase_many_all_60m_1p2b.csv"
INPUT_LOGICAL_RUNS_CSV = ROOT / "run_registry" / "logical_runs.csv"
OUTPUT_PNG = ROOT / "qsplit240_300m_6b_completed_vs_60m_rank_shift.png"
OUTPUT_BPB_PNG = ROOT / "qsplit240_300m_6b_completed_vs_60m_bpb_correlation.png"
OUTPUT_SELECTED_CSV = ROOT / "qsplit240_300m_6b_completed_vs_60m_rank_shift_selected.csv"

TOP_MIXTURE_COUNT = 8
BASELINE_LABELS = {
    "baseline_stratified": "Uniform",
    "baseline_unimax": "UniMax",
    "baseline_proportional": "Proportional",
}
EXTRA_REFERENCE_SPECS = (
    {
        "run_name": "baseline_olmix_loglinear_uncheatable_bpb",
        "label": "Olmix",
        "registry_run_name": "baseline_olmix_loglinear_uncheatable_bpb",
        "note": "executor failed; final eval present",
    },
    {
        "run_name": "baseline_genericfamily_power_family_penalty_raw_optimum",
        "label": "GRP (Power-Family Penalty)",
        "registry_run_name": "baseline_genericfamily_power_family_penalty_raw_optimum_300m_6b",
        "note": "executor still running; final eval present",
    },
)
ANNOTATION_OFFSET_POINTS = (4, -4)
POSITIVE_COLOR = "#1a9850"
NEGATIVE_COLOR = "#d73027"
NEUTRAL_COLOR = "#4d4d4d"
EXTRA_POINT_COLOR = "#2b2b2b"
EXTRA_POINT_MARKER = "X"
EXTRA_POINT_SIZE = 88


def _display_label(run_name: str) -> str:
    if run_name in BASELINE_LABELS:
        return BASELINE_LABELS[run_name]
    if run_name.startswith("run_"):
        return run_name.replace("run_", "mix_", 1)
    return run_name


def _extra_rows(frame: pd.DataFrame) -> pd.DataFrame:
    frame_60m = pd.read_csv(INPUT_60M_ALL_RUNS_CSV)
    logical_runs = pd.read_csv(INPUT_LOGICAL_RUNS_CSV)
    rows: list[dict[str, object]] = []
    for spec in EXTRA_REFERENCE_SPECS:
        row_60m = frame_60m.loc[frame_60m["run_name"] == spec["run_name"]]
        if row_60m.empty:
            raise ValueError(f"Missing 60M row for {spec['run_name']}")
        row_300m = logical_runs.loc[logical_runs["run_name"] == spec["registry_run_name"]]
        if row_300m.empty:
            raise ValueError(f"Missing registry row for {spec['registry_run_name']}")
        bpb_60m = float(row_60m.iloc[0]["eval/uncheatable_eval/bpb"])
        bpb_300m = float(row_300m.iloc[0]["objective_metric_value"])
        rank_60m = int((frame["bpb_60m"] < bpb_60m).sum() + 1)
        rank_300m = int((frame["bpb_300m_6b"] < bpb_300m).sum() + 1)
        rows.append(
            {
                "run_name": spec["run_name"],
                "label": spec["label"],
                "group": "Extra 300M final",
                "note": spec["note"],
                "bpb_60m": bpb_60m,
                "bpb_300m_6b": bpb_300m,
                "bpb_delta": bpb_60m - bpb_300m,
                "rank_60m_within_completed": rank_60m,
                "rank_300m_6b": rank_300m,
                "rank_shift": rank_60m - rank_300m,
            }
        )
    return pd.DataFrame(rows)


def _selected_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baselines = frame[frame["run_name"].isin(BASELINE_LABELS)].copy()
    mixtures = frame[frame["run_name"].str.startswith("run_")].copy()
    mixtures = mixtures.sort_values("rank_60m_within_completed").head(TOP_MIXTURE_COUNT).copy()
    selected = pd.concat([baselines, mixtures], ignore_index=True)
    selected["label"] = selected["run_name"].map(_display_label)
    selected["group"] = np.where(selected["run_name"].isin(BASELINE_LABELS), "Baseline", "Best 60M mixture")
    selected = selected.sort_values(["group", "rank_60m_within_completed"]).reset_index(drop=True)
    return selected, _extra_rows(frame)


def _bar_colors(rank_shift: pd.Series) -> list[str]:
    colors: list[str] = []
    for value in rank_shift:
        if value > 0:
            colors.append(POSITIVE_COLOR)
        elif value < 0:
            colors.append(NEGATIVE_COLOR)
        else:
            colors.append(NEUTRAL_COLOR)
    return colors


def _bpb_colors(bpb_values: pd.Series, *, vmin: float, vmax: float) -> list[tuple[float, float, float, float]]:
    cmap = plt.get_cmap("RdYlGn_r")
    norm = Normalize(vmin=vmin, vmax=vmax)
    return [cmap(norm(float(value))) for value in bpb_values]


def _plot_scatter(ax: plt.Axes, frame: pd.DataFrame, selected: pd.DataFrame, extras: pd.DataFrame) -> None:
    cmap = plt.get_cmap("RdYlGn_r")
    norm = Normalize(vmin=frame["bpb_300m_6b"].min(), vmax=frame["bpb_300m_6b"].max())
    scatter = ax.scatter(
        frame["rank_60m_within_completed"],
        frame["rank_300m_6b"],
        c=frame["bpb_300m_6b"],
        cmap=cmap,
        norm=norm,
        s=44,
        alpha=0.9,
        edgecolors="none",
    )
    max_rank = int(frame[["rank_60m_within_completed", "rank_300m_6b"]].to_numpy().max())
    ax.plot([1, max_rank], [1, max_rank], linestyle="--", color="0.45", linewidth=1.2)
    ax.set_xlim(0, max_rank + 3)
    ax.set_ylim(0, max_rank + 3)
    ax.set_xlabel("60M rank within completed set")
    ax.set_ylabel("300M/6B rank within completed set")
    ax.set_title("Completed-mixture rank comparison")
    ax.grid(alpha=0.18, linewidth=0.6)

    annotated = selected.sort_values(["rank_300m_6b", "rank_60m_within_completed"])
    for _, row in annotated.iterrows():
        ax.annotate(
            row["label"],
            (row["rank_60m_within_completed"], row["rank_300m_6b"]),
            xytext=ANNOTATION_OFFSET_POINTS,
            textcoords="offset points",
            fontsize=8,
            alpha=0.92,
        )

    if not extras.empty:
        ax.scatter(
            extras["rank_60m_within_completed"],
            extras["rank_300m_6b"],
            s=EXTRA_POINT_SIZE,
            marker=EXTRA_POINT_MARKER,
            color=EXTRA_POINT_COLOR,
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
        for _, row in extras.iterrows():
            ax.annotate(
                row["label"],
                (row["rank_60m_within_completed"], row["rank_300m_6b"]),
                xytext=ANNOTATION_OFFSET_POINTS,
                textcoords="offset points",
                fontsize=8,
                alpha=0.95,
                fontweight="semibold",
            )

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("300M/6B BPB")


def _plot_selected_bars(ax: plt.Axes, selected: pd.DataFrame, extras: pd.DataFrame) -> None:
    movers = pd.concat([selected, extras], ignore_index=True).sort_values(["group", "rank_shift"]).reset_index(drop=True)
    ax.barh(movers["label"], movers["rank_shift"], color=_bar_colors(movers["rank_shift"]), alpha=0.9)
    ax.axvline(0, color="0.4", linewidth=1.0)
    ax.set_xlabel("Rank shift if included = 60M rank - 300M/6B rank")
    ax.set_title("Baselines + best 60M mixtures")
    ax.grid(axis="x", alpha=0.18, linewidth=0.6)

    xmax = max(abs(float(movers["rank_shift"].min())), abs(float(movers["rank_shift"].max())))
    ax.set_xlim(-xmax - 6, xmax + 14)

    for _, row in movers.iterrows():
        x = float(row["rank_shift"])
        alignment = "left" if x >= 0 else "right"
        x_text = x + 1.0 if x >= 0 else x - 1.0
        ax.text(
            x_text,
            row["label"],
            f"{int(row['rank_300m_6b'])}",
            va="center",
            ha=alignment,
            fontsize=8,
            color="0.25",
        )


def _plot_bpb_scatter(ax: plt.Axes, frame: pd.DataFrame, selected: pd.DataFrame, extras: pd.DataFrame) -> None:
    cmap = plt.get_cmap("RdYlGn_r")
    norm = Normalize(vmin=frame["bpb_300m_6b"].min(), vmax=frame["bpb_300m_6b"].max())
    scatter = ax.scatter(
        frame["bpb_60m"],
        frame["bpb_300m_6b"],
        c=frame["bpb_300m_6b"],
        cmap=cmap,
        norm=norm,
        s=44,
        alpha=0.9,
        edgecolors="none",
    )
    coeffs = np.polyfit(frame["bpb_60m"], frame["bpb_300m_6b"], deg=1)
    x_line = np.linspace(float(frame["bpb_60m"].min()), float(frame["bpb_60m"].max()), 200)
    y_line = coeffs[0] * x_line + coeffs[1]
    ax.plot(x_line, y_line, linestyle="--", color="0.35", linewidth=1.2, label="least-squares fit")
    ax.set_xlabel("60M uncheatable BPB")
    ax.set_ylabel("300M/6B uncheatable BPB")
    ax.set_title("Completed-mixture BPB correlation")
    ax.grid(alpha=0.18, linewidth=0.6)

    annotated = selected.sort_values(["bpb_300m_6b", "bpb_60m"])
    for _, row in annotated.iterrows():
        ax.annotate(
            row["label"],
            (row["bpb_60m"], row["bpb_300m_6b"]),
            xytext=ANNOTATION_OFFSET_POINTS,
            textcoords="offset points",
            fontsize=8,
            alpha=0.92,
        )

    if not extras.empty:
        ax.scatter(
            extras["bpb_60m"],
            extras["bpb_300m_6b"],
            s=EXTRA_POINT_SIZE,
            marker=EXTRA_POINT_MARKER,
            color=EXTRA_POINT_COLOR,
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
        for _, row in extras.iterrows():
            ax.annotate(
                row["label"],
                (row["bpb_60m"], row["bpb_300m_6b"]),
                xytext=ANNOTATION_OFFSET_POINTS,
                textcoords="offset points",
                fontsize=8,
                alpha=0.95,
                fontweight="semibold",
            )

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("300M/6B BPB")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)


def _plot_bpb_deltas(ax: plt.Axes, selected: pd.DataFrame, extras: pd.DataFrame, frame: pd.DataFrame) -> None:
    movers = pd.concat([selected, extras], ignore_index=True).sort_values(["group", "bpb_delta"]).reset_index(drop=True)
    ax.barh(
        movers["label"],
        movers["bpb_delta"],
        color=_bpb_colors(
            movers["bpb_300m_6b"],
            vmin=float(frame["bpb_300m_6b"].min()),
            vmax=float(frame["bpb_300m_6b"].max()),
        ),
        alpha=0.9,
    )
    ax.axvline(0, color="0.4", linewidth=1.0)
    ax.set_xlabel("BPB gain = 60M BPB - 300M/6B BPB")
    ax.set_title("Baselines + best 60M mixtures")
    ax.grid(axis="x", alpha=0.18, linewidth=0.6)

    xmax = float(movers["bpb_delta"].max())
    ax.set_xlim(-0.01, xmax + 0.03)

    for _, row in movers.iterrows():
        ax.text(
            float(row["bpb_delta"]) + 0.002,
            row["label"],
            f"{float(row['bpb_300m_6b']):.3f}",
            va="center",
            ha="left",
            fontsize=8,
            color="0.25",
        )


def main() -> None:
    frame = pd.read_csv(INPUT_CSV)
    summary = json.loads(INPUT_SUMMARY_JSON.read_text())
    selected, extras = _selected_rows(frame)
    selected["bpb_delta"] = selected["bpb_60m"] - selected["bpb_300m_6b"]

    fig, (ax_scatter, ax_movers) = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.8),
        gridspec_kw={"width_ratios": [1.5, 1.0]},
        constrained_layout=True,
    )

    _plot_scatter(ax_scatter, frame, selected, extras)
    _plot_selected_bars(ax_movers, selected, extras)

    fig.suptitle(
        (
            "QSplit240 completed 300M/6B mixtures vs 60M swarm\n"
            f"n={summary['n_completed']} strict-success shared runs, "
            f"Spearman={summary['spearman_rho']:.3f}, Kendall={summary['kendall_tau']:.3f}"
        ),
        fontsize=13,
    )
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig_bpb, (ax_bpb_scatter, ax_bpb_movers) = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.8),
        gridspec_kw={"width_ratios": [1.5, 1.0]},
        constrained_layout=True,
    )

    _plot_bpb_scatter(ax_bpb_scatter, frame, selected, extras)
    _plot_bpb_deltas(ax_bpb_movers, selected, extras, frame)

    fig_bpb.suptitle(
        (
            "QSplit240 completed 300M/6B mixtures vs 60M swarm\n"
            f"n={summary['n_completed']} strict-success shared runs, "
            f"Pearson={summary['pearson_r']:.3f}, Spearman={summary['spearman_rho']:.3f}, "
            f"Kendall={summary['kendall_tau']:.3f}"
        ),
        fontsize=13,
    )
    fig_bpb.savefig(OUTPUT_BPB_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig_bpb)

    selected_out = selected[
        [
            "run_name",
            "label",
            "group",
            "bpb_60m",
            "bpb_300m_6b",
            "bpb_delta",
            "rank_60m_within_completed",
            "rank_300m_6b",
            "rank_shift",
        ]
    ].copy()
    extras_out = extras[
        [
            "run_name",
            "label",
            "group",
            "bpb_60m",
            "bpb_300m_6b",
            "bpb_delta",
            "rank_60m_within_completed",
            "rank_300m_6b",
            "rank_shift",
            "note",
        ]
    ].copy()
    pd.concat([selected_out, extras_out], ignore_index=True).to_csv(OUTPUT_SELECTED_CSV, index=False)
    print(f"Wrote {OUTPUT_PNG}")
    print(f"Wrote {OUTPUT_BPB_PNG}")
    print(f"Wrote {OUTPUT_SELECTED_CSV}")


if __name__ == "__main__":
    main()
