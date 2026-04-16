# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "pandas"]
# ///
"""Plot local RegMix raw-optimum convergence for the many-domain two-phase packet."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

OBJECTIVE_METRIC = "eval/uncheatable_eval/bpb"
BPB_LABEL_DECIMALS = 3
SCRIPT_DIR = Path(__file__).resolve().parent
CURVE_POINTS_CSV = SCRIPT_DIR / "two_phase_many_regmix_raw_curve_points.csv"
PLOT_BPB_TV_PATH = SCRIPT_DIR / "two_phase_many_regmix_raw_bpb_and_tv.png"
PLOT_BPB_PHASE_MOVEMENT_PATH = SCRIPT_DIR / "two_phase_many_regmix_raw_bpb_and_phase_movements.png"
TWO_PHASE_MANY_ALL_CSV = SCRIPT_DIR / "two_phase_many_all_60m_1p2b.csv"
PROPORTIONAL_RUN_NAME = "baseline_proportional"
PROPORTIONAL_COLOR = "#4C78A8"


def _format_bpb_label(value: float) -> str:
    return f"{value:.{BPB_LABEL_DECIMALS}f}"


def _baseline_proportional_bpb() -> float:
    frame = pd.read_csv(TWO_PHASE_MANY_ALL_CSV, usecols=["run_name", OBJECTIVE_METRIC])
    matches = frame.loc[frame["run_name"] == PROPORTIONAL_RUN_NAME, OBJECTIVE_METRIC].dropna()
    if matches.empty:
        raise ValueError(f"Missing proportional BPB in {TWO_PHASE_MANY_ALL_CSV}")
    return float(matches.iloc[-1])


def _plot_bpb_panel(ax_bpb: plt.Axes, frame: pd.DataFrame, *, proportional_bpb: float, cmap) -> None:
    ax_bpb.plot(
        frame["subset_size"],
        frame["predicted_optimum_value"],
        color=cmap(0.18),
        marker="o",
        linewidth=2.2,
        label="Predicted BPB",
    )
    ax_bpb.plot(
        frame["subset_size"],
        frame["subset_best_observed_bpb"],
        color=PROPORTIONAL_COLOR,
        marker="P",
        linewidth=1.8,
        linestyle=":",
        label="Best observed BPB in subset",
    )
    ax_bpb.axhline(
        proportional_bpb,
        color=PROPORTIONAL_COLOR,
        linewidth=1.5,
        linestyle="--",
        alpha=0.95,
        zorder=1,
        label="Proportional BPB",
    )
    ax_bpb.annotate(
        f"Proportional: {_format_bpb_label(proportional_bpb)}",
        (int(frame["subset_size"].max()), proportional_bpb),
        textcoords="offset points",
        xytext=(-6, 8),
        ha="right",
        fontsize=8,
        color=PROPORTIONAL_COLOR,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
    )
    validated = frame[frame["actual_validated_bpb"].notna()].copy()
    if validated.empty:
        return
    ax_bpb.plot(
        validated["subset_size"],
        validated["actual_validated_bpb"],
        color=cmap(0.86),
        marker="X",
        markersize=8,
        linewidth=1.8,
        linestyle="--",
        label="Validated BPB",
    )
    validated_sizes = validated["subset_size"].to_numpy(dtype=float)
    validated_values = validated["actual_validated_bpb"].to_numpy(dtype=float)
    for idx, row in enumerate(validated.itertuples(index=False)):
        y_offset = 9 if idx % 2 == 0 else 15
        x_offset = -8 if idx % 3 == 1 else (8 if idx % 3 == 2 else 0)
        crowded = False
        if idx > 0 and abs(validated_values[idx] - validated_values[idx - 1]) < 0.012:
            crowded = True
        if idx + 1 < len(validated_values) and abs(validated_values[idx] - validated_values[idx + 1]) < 0.012:
            crowded = True
        if crowded:
            y_offset += 4
        if idx > 0 and abs(validated_sizes[idx] - validated_sizes[idx - 1]) <= 40:
            x_offset += 6
        if idx + 1 < len(validated_sizes) and abs(validated_sizes[idx] - validated_sizes[idx + 1]) <= 40:
            x_offset -= 6
        ax_bpb.annotate(
            _format_bpb_label(float(row.actual_validated_bpb)),
            (row.subset_size, row.actual_validated_bpb),
            textcoords="offset points",
            xytext=(x_offset, y_offset),
            ha="center",
            fontsize=8,
            color=cmap(0.88),
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )


def _weights_from_dict(phase_weights: dict[str, dict[str, float]]) -> np.ndarray:
    domain_names = list(phase_weights["phase_0"].keys())
    return np.asarray(
        [
            [float(phase_weights["phase_0"][domain_name]) for domain_name in domain_names],
            [float(phase_weights["phase_1"][domain_name]) for domain_name in domain_names],
        ],
        dtype=float,
    )


def _phase_movement_frame(frame: pd.DataFrame) -> pd.DataFrame:
    movements: list[dict[str, float | int | None]] = []
    previous_weights: np.ndarray | None = None
    for row in frame.sort_values("subset_size").itertuples(index=False):
        weights = _weights_from_dict(row.phase_weights)
        if previous_weights is None:
            phase0_tv = None
            phase1_tv = None
        else:
            phase0_tv = 0.5 * float(np.sum(np.abs(weights[0] - previous_weights[0])))
            phase1_tv = 0.5 * float(np.sum(np.abs(weights[1] - previous_weights[1])))
        movements.append(
            {
                "subset_size": int(row.subset_size),
                "phase0_tv_vs_prev": phase0_tv,
                "phase1_tv_vs_prev": phase1_tv,
            }
        )
        previous_weights = weights
    return pd.DataFrame(movements)


def _plot_bpb_and_phase_movements(frame: pd.DataFrame, *, proportional_bpb: float, cmap) -> None:
    movement_frame = _phase_movement_frame(frame)
    fig, (ax_bpb, ax_move) = plt.subplots(
        2,
        1,
        figsize=(10.2, 7.5),
        dpi=180,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.45, 1.15], "hspace": 0.08},
    )
    _plot_bpb_panel(ax_bpb, frame, proportional_bpb=proportional_bpb, cmap=cmap)
    ax_move.plot(
        movement_frame["subset_size"],
        movement_frame["phase0_tv_vs_prev"],
        color=cmap(0.24),
        marker="o",
        linewidth=2.2,
        label=r"Phase 0 TV movement: $\frac{1}{2}\|w_0^{(k)} - w_0^{(k^-)}\|_1$",
    )
    ax_move.plot(
        movement_frame["subset_size"],
        movement_frame["phase1_tv_vs_prev"],
        color=cmap(0.76),
        marker="D",
        linewidth=2.2,
        label=r"Phase 1 TV movement: $\frac{1}{2}\|w_1^{(k)} - w_1^{(k^-)}\|_1$",
    )
    ax_move.axhline(0.0, color="0.55", linewidth=1.0, linestyle=":")

    ax_bpb.set_title("Two-phase many-domain: RegMix convergence (raw optimum)")
    ax_bpb.set_ylabel("Predicted BPB")
    ax_move.set_ylabel("Phase TV")
    ax_move.set_xlabel("Observed runs used for fitting")
    ax_move.set_xticks(movement_frame["subset_size"].tolist())
    ax_move.set_xlim(int(movement_frame["subset_size"].min()), int(movement_frame["subset_size"].max()))
    ax_move.margins(y=0.08)

    ax_bpb.grid(True, alpha=0.25)
    bpb_handles = ax_bpb.get_lines()
    bpb_labels = [handle.get_label() for handle in bpb_handles if not handle.get_label().startswith("_")]
    if bpb_handles:
        ax_bpb.legend(bpb_handles, bpb_labels, loc="upper right", frameon=True, ncol=2)

    ax_move.grid(True, alpha=0.25)
    move_handles = ax_move.get_lines()
    move_labels = [handle.get_label() for handle in move_handles if not handle.get_label().startswith("_")]
    if move_handles:
        ax_move.legend(
            move_handles,
            move_labels,
            loc="upper right",
            bbox_to_anchor=(0.995, 1.05),
            ncol=1,
            frameon=True,
            alignment="right",
        )

    fig.savefig(PLOT_BPB_PHASE_MOVEMENT_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frame = pd.read_csv(CURVE_POINTS_CSV).sort_values("subset_size").reset_index(drop=True)
    frame["phase_weights"] = frame["phase_weights"].map(
        lambda payload: json.loads(payload) if isinstance(payload, str) and payload else payload
    )
    proportional_bpb = _baseline_proportional_bpb()
    cmap = plt.colormaps["RdYlGn_r"]

    fig, (ax_bpb, ax_move) = plt.subplots(
        2,
        1,
        figsize=(10.2, 5.8),
        dpi=180,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.08},
    )

    _plot_bpb_panel(ax_bpb, frame, proportional_bpb=proportional_bpb, cmap=cmap)
    ax_move.plot(
        frame["subset_size"],
        frame["optimum_move_mean_phase_tv_vs_prev"],
        color=cmap(0.36),
        marker="D",
        linewidth=2.2,
        label="Raw-optimum movement (mean phase TV)",
    )
    ax_move.axhline(0.0, color="0.55", linewidth=1.0, linestyle=":")

    ax_bpb.set_title("Two-phase many-domain: RegMix convergence (raw optimum)")
    ax_bpb.set_ylabel("Predicted BPB")
    ax_move.set_ylabel("Mean phase TV")
    ax_move.set_title(
        r"Movement uses $\mathrm{mean}_{p \in \{0,1\}}\!\left[\frac{1}{2}\,\|w_p^{(k)} - w_p^{(k^-)}\|_1\right]$.",
        fontsize=9,
        pad=6,
    )
    ax_move.set_xlabel("Observed runs used for fitting")
    ax_move.set_xticks(frame["subset_size"].tolist())
    ax_move.set_xlim(int(frame["subset_size"].min()), int(frame["subset_size"].max()))

    for axis in (ax_bpb, ax_move):
        axis.grid(True, alpha=0.25)
        handles = axis.get_lines()
        labels = [handle.get_label() for handle in handles if not handle.get_label().startswith("_")]
        if handles:
            axis.legend(handles, labels, loc="best", frameon=True)

    fig.savefig(PLOT_BPB_TV_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PLOT_BPB_TV_PATH}")
    _plot_bpb_and_phase_movements(frame, proportional_bpb=proportional_bpb, cmap=cmap)
    print(f"Wrote {PLOT_BPB_PHASE_MOVEMENT_PATH}")


if __name__ == "__main__":
    main()
