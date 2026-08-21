# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot GRP-style Starcoder fits on the phase-0 Nemotron slice.

This uses an explicit 2-family GRP parameterization on the Starcoder packet:

- ``broad_text`` = ``nemotron_full``
- ``tech_code`` = ``starcoder``

The model is fit twice, once on the U-shaped subset where
``phase_0_nemotron_full == 1.0`` and once on all 2-phase Starcoder data, and
both fits are evaluated on the same subset slice.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.starcoder_grp import (
    fit_starcoder_grp,
    subset_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    STARCODER_TARGET,
    PacketData,
    load_two_phase_starcoder_packet,
    regression_metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PNG = SCRIPT_DIR / "grp_starcoder_u_shape_fit.png"
OUTPUT_CSV = SCRIPT_DIR / "grp_starcoder_u_shape_fit_subset_predictions.csv"
OUTPUT_SUMMARY_JSON = SCRIPT_DIR / "grp_starcoder_u_shape_fit_summary.json"

FIGSIZE = (15.5, 6.8)
DPI = 300
GRID_SIZE = 1001

TITLE = "GRP on 2-Phase StarCoder: U-Shaped Slice Fit"
SUBTITLE = "Broad-text family = Nemotron; tech-code family = StarCoder; shape hyperparameters tuned separately"
FOOTNOTE = "Black points show observed slice runs. Slice: Nemotron only in first phase. "

ACTUAL_COLOR = "#111827"
ACTUAL_LINE_COLOR = "#9ca3af"
SUBSET_FIT_COLOR = "#1b7837"
ALL_DATA_FIT_COLOR = "#b2182b"
MIN_MARKER_COLOR = "#1f2937"
GRID_COLOR = "#cbd5e1"


def _sort_packet_by_column(packet: PacketData, column: str) -> PacketData:
    """Return a packet sorted by one frame column."""
    order = np.argsort(packet.frame[column].to_numpy(dtype=float))
    return replace(
        packet,
        frame=packet.frame.iloc[order].reset_index(drop=True),
        y=packet.y[order],
        w=packet.w[order],
    )


def _build_slice_weights(starcoder_weights: np.ndarray) -> np.ndarray:
    """Return two-phase weights on the phase-0 Nemotron slice."""
    weights = np.zeros((len(starcoder_weights), 2, 2), dtype=float)
    weights[:, 0, 0] = 1.0
    weights[:, 1, 1] = starcoder_weights
    weights[:, 1, 0] = 1.0 - starcoder_weights
    return weights


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _epochs_from_weight(weight: float, starcoder_phase1_epochs: float) -> float:
    """Return phase-1 StarCoder epochs for a given slice weight."""
    return float(weight * starcoder_phase1_epochs)


def _panel_note(
    *,
    subset_rmse: float,
) -> str:
    """Build a compact per-panel annotation."""
    return f"Subset RMSE: {subset_rmse:.4f}"


def _add_epochs_axis(ax: plt.Axes, starcoder_phase1_epochs: float) -> None:
    """Add a top x-axis in phase-1 StarCoder epochs."""
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda weight: weight * starcoder_phase1_epochs,
            lambda epochs: epochs / starcoder_phase1_epochs,
        ),
    )
    secax.set_xlabel("Phase 1 StarCoder epochs", fontsize=12.5, fontweight="semibold", labelpad=8)
    secax.tick_params(labelsize=11)


def _plot_panel(
    ax: plt.Axes,
    *,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    x_grid: np.ndarray,
    y_fit: np.ndarray,
    fit_color: str,
    fit_label: str,
    panel_title: str,
    panel_note: str,
    starcoder_phase1_epochs: float,
) -> None:
    """Render one slice-fit panel."""
    ax.plot(x_obs, y_obs, color=ACTUAL_LINE_COLOR, linewidth=1.8, linestyle="--", alpha=0.85, zorder=1)
    ax.scatter(
        x_obs,
        y_obs,
        s=54,
        color=ACTUAL_COLOR,
        edgecolors="white",
        linewidths=0.9,
        label="Observed slice runs",
        zorder=4,
    )
    ax.plot(x_grid, y_fit, color=fit_color, linewidth=3.2, label=fit_label, zorder=3)

    min_idx = int(np.argmin(y_fit))
    ax.scatter(
        [x_grid[min_idx]],
        [y_fit[min_idx]],
        s=85,
        color=fit_color,
        edgecolors=MIN_MARKER_COLOR,
        linewidths=1.0,
        zorder=5,
    )
    ax.axvline(x_grid[min_idx], color=fit_color, linewidth=1.2, linestyle=":", alpha=0.8, zorder=2)

    ax.set_title(panel_title, fontsize=18, fontweight="bold", pad=12)
    ax.set_xlabel("Phase 1 StarCoder weight", fontsize=13.5, fontweight="semibold")
    ax.grid(axis="both", color=GRID_COLOR, linewidth=0.85, alpha=0.7)
    ax.tick_params(labelsize=11.5)
    _add_epochs_axis(ax, starcoder_phase1_epochs)

    ax.text(
        0.03,
        0.97,
        panel_note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        fontweight="semibold",
        color="#334155",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "alpha": 0.96},
    )


def main() -> None:
    packet = load_two_phase_starcoder_packet(target=STARCODER_TARGET)
    completed_mask = (
        packet.frame["status"].eq("completed").to_numpy(dtype=bool)
        if "status" in packet.frame.columns
        else np.ones(len(packet.frame), dtype=bool)
    )
    completed_packet = subset_packet(packet, completed_mask)
    slice_mask = completed_packet.frame["phase_0_nemotron_full"].round(4).eq(1.0).to_numpy(dtype=bool)
    subset_packet_data = _sort_packet_by_column(subset_packet(completed_packet, slice_mask), "phase_1_starcoder")

    subset_params, subset_model = fit_starcoder_grp(subset_packet_data, seed=0)
    full_params, full_model = fit_starcoder_grp(completed_packet, seed=0)

    x_obs = subset_packet_data.frame["phase_1_starcoder"].to_numpy(dtype=float)
    y_obs = subset_packet_data.y

    x_grid = np.linspace(0.0, 1.0, GRID_SIZE, dtype=float)
    w_grid = _build_slice_weights(x_grid)
    y_subset_fit = subset_model.predict(w_grid)
    y_full_fit = full_model.predict(w_grid)

    subset_pred_on_subset = subset_model.predict(subset_packet_data.w)
    full_pred_on_subset = full_model.predict(subset_packet_data.w)
    starcoder_phase1_epochs = float(completed_packet.c1[1])

    output_frame = subset_packet_data.frame.copy()
    output_frame["subset_fit_prediction"] = subset_pred_on_subset
    output_frame["all_data_fit_prediction"] = full_pred_on_subset
    output_frame.to_csv(OUTPUT_CSV, index=False)

    summary = {
        "target": STARCODER_TARGET,
        "n_total_runs": len(completed_packet.frame),
        "n_subset_runs": len(subset_packet_data.frame),
        "subset_fit_params": subset_params,
        "all_data_fit_params": full_params,
        "subset_fit_metrics_on_subset": regression_metrics(
            subset_packet_data.frame,
            subset_packet_data.name_col,
            y_obs,
            subset_pred_on_subset,
        ),
        "all_data_fit_metrics_on_subset": regression_metrics(
            subset_packet_data.frame,
            subset_packet_data.name_col,
            y_obs,
            full_pred_on_subset,
        ),
        "observed_subset_min": {
            "phase_1_starcoder": float(x_obs[np.argmin(y_obs)]),
            "bpb": float(np.min(y_obs)),
        },
        "subset_fit_slice_min": {
            "phase_1_starcoder": float(x_grid[np.argmin(y_subset_fit)]),
            "bpb": float(np.min(y_subset_fit)),
            "epochs": _epochs_from_weight(float(x_grid[np.argmin(y_subset_fit)]), starcoder_phase1_epochs),
        },
        "all_data_fit_slice_min": {
            "phase_1_starcoder": float(x_grid[np.argmin(y_full_fit)]),
            "bpb": float(np.min(y_full_fit)),
            "epochs": _epochs_from_weight(float(x_grid[np.argmin(y_full_fit)]), starcoder_phase1_epochs),
        },
        "subset_fit_subset_rmse": _rmse(y_obs, subset_pred_on_subset),
        "all_data_fit_subset_rmse": _rmse(y_obs, full_pred_on_subset),
    }
    OUTPUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True))

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)

    y_min = float(min(np.min(y_obs), np.min(y_subset_fit), np.min(y_full_fit)))
    y_max = float(max(np.max(y_obs), np.max(y_subset_fit), np.max(y_full_fit)))
    y_pad = 0.06 * (y_max - y_min)

    _plot_panel(
        axes[0],
        x_obs=x_obs,
        y_obs=y_obs,
        x_grid=x_grid,
        y_fit=y_subset_fit,
        fit_color=SUBSET_FIT_COLOR,
        fit_label="GRP fit trained on subset",
        panel_title="Fit on U-Shaped Subset",
        panel_note=_panel_note(
            subset_rmse=_rmse(y_obs, subset_pred_on_subset),
        ),
        starcoder_phase1_epochs=starcoder_phase1_epochs,
    )
    _plot_panel(
        axes[1],
        x_obs=x_obs,
        y_obs=y_obs,
        x_grid=x_grid,
        y_fit=y_full_fit,
        fit_color=ALL_DATA_FIT_COLOR,
        fit_label="GRP fit trained on all Starcoder runs",
        panel_title="Fit on All 2-Phase StarCoder Runs",
        panel_note=_panel_note(
            subset_rmse=_rmse(y_obs, full_pred_on_subset),
        ),
        starcoder_phase1_epochs=starcoder_phase1_epochs,
    )

    axes[0].set_ylabel(STARCODER_TARGET, fontsize=13.5, fontweight="semibold")
    axes[0].set_xlim(0.0, 1.0)
    axes[1].set_xlim(0.0, 1.0)
    axes[0].set_ylim(y_min - y_pad, y_max + y_pad)

    fig.suptitle(TITLE, fontsize=24, fontweight="bold", y=0.985)
    fig.text(0.5, 0.905, SUBTITLE, ha="center", fontsize=15.5, color="#475569")
    fig.text(0.5, 0.045, FOOTNOTE, ha="center", fontsize=11.5, color="#64748b")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.76, bottom=0.12, wspace=0.16)
    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUTPUT_PNG}")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
