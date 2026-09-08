# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot linear and Olmix-loglinear fits on the StarCoder U-shaped slice.

This mirrors the old GRP StarCoder U-shape figure, but overlays two lower-
parameter fit families in each panel:

- a linear surrogate fit on flattened phase/domain weights
- an Olmix loglinear surrogate fit on the same flattened weights

Each panel trains on a different dataset:

- the left panel trains on the U-shaped slice only
- the right panel trains on all completed two-phase StarCoder runs

Both fits are evaluated on the same U-shaped slice where:

- phase_0_nemotron_full == 1.0
- phase_0_starcoder == 0.0
- phase_1_starcoder = x
- phase_1_nemotron_full = 1 - x
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.starcoder_grp import (
    subset_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    STARCODER_TARGET,
    PacketData,
    load_two_phase_starcoder_packet,
    regression_metrics,
)
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear_sl_verb import (
    OlmixLoglinearFit,
    fit_olmix_loglinear_model,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PNG = SCRIPT_DIR / "starcoder_u_shape_linear_vs_olmix_fit.png"
OUTPUT_CSV = SCRIPT_DIR / "starcoder_u_shape_linear_vs_olmix_fit_subset_predictions.csv"
OUTPUT_SUMMARY_JSON = SCRIPT_DIR / "starcoder_u_shape_linear_vs_olmix_fit_summary.json"

FIGSIZE = (15.8, 6.9)
DPI = 300
GRID_SIZE = 1001

TITLE = "2-Phase StarCoder: Linear vs Olmix Loglinear on U-Shaped Slice"
SUBTITLE = "Both panels evaluate on the same Nemotron-first slice; only the training data differ"
FOOTNOTE = "Black points show observed slice runs. Slice: Nemotron only in phase 0."

ACTUAL_COLOR = "#111827"
ACTUAL_LINE_COLOR = "#9ca3af"
GRID_COLOR = "#cbd5e1"
COLORMAP = plt.get_cmap("RdYlGn_r")
OLMIX_COLOR = COLORMAP(0.12)
LINEAR_COLOR = COLORMAP(0.88)
MIN_MARKER_COLOR = "#1f2937"


@dataclass(frozen=True)
class LinearFit:
    """Affine fit on flattened phase/domain weights."""

    intercept: float
    coefficients: np.ndarray

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return np.asarray(self.intercept + _flatten_weights(weights) @ self.coefficients, dtype=float)


@dataclass(frozen=True)
class PanelFitSummary:
    """Compact panel metrics for output serialization."""

    linear_formula: str
    olmix_formula: str
    linear_subset_rmse: float
    olmix_subset_rmse: float
    linear_metrics_on_subset: dict[str, float | str]
    olmix_metrics_on_subset: dict[str, float | str]


def _sort_packet_by_column(packet: PacketData, column: str) -> PacketData:
    """Return a packet sorted by one frame column."""
    order = np.argsort(packet.frame[column].to_numpy(dtype=float))
    return PacketData(
        frame=packet.frame.iloc[order].reset_index(drop=True),
        name_col=packet.name_col,
        y=packet.y[order],
        w=packet.w[order],
        m=packet.m,
        c0=packet.c0,
        c1=packet.c1,
        domain_names=packet.domain_names,
    )


def _flatten_weights(weights: np.ndarray) -> np.ndarray:
    """Flatten phase/domain weights into a design matrix."""
    return np.asarray(weights, dtype=float).reshape(len(weights), -1)


def _fit_linear_model(packet: PacketData) -> LinearFit:
    """Fit a minimum-norm linear model on flattened weights."""
    x = _flatten_weights(packet.w)
    x_center = x - x.mean(axis=0, keepdims=True)
    y_center = packet.y - float(packet.y.mean())
    coefficients, *_ = np.linalg.lstsq(x_center, y_center, rcond=None)
    intercept = float(packet.y.mean() - x.mean(axis=0) @ coefficients)
    return LinearFit(intercept=intercept, coefficients=np.asarray(coefficients, dtype=float))


def _build_slice_weights(starcoder_weights: np.ndarray) -> np.ndarray:
    """Return two-phase weights on the phase-0 Nemotron-only slice."""
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


def _format_float(value: float) -> str:
    """Format one float for legend text."""
    return f"{value:.3f}"


def _format_signed_term(value: float, term: str) -> str:
    """Format one signed additive term for legend formulas."""
    sign = "+" if value >= 0 else "-"
    return f" {sign} {_format_float(abs(value))}{term}"


def _format_signed_constant(value: float) -> str:
    """Format one signed additive constant for legend formulas."""
    sign = "+" if value >= 0 else "-"
    return f" {sign} {_format_float(abs(value))}"


def _format_vector(values: tuple[float, ...] | list[float]) -> str:
    """Format one short coefficient vector."""
    entries = ", ".join(_format_float(value) for value in values)
    return f"({entries})"


def _linear_formula(fit: LinearFit) -> str:
    """Return the collapsed linear slice formula."""
    beta_phase0_nem, _beta_phase0_star, beta_phase1_nem, beta_phase1_star = fit.coefficients.tolist()
    constant = fit.intercept + beta_phase0_nem
    return (
        f"Linear: {_format_float(beta_phase1_star)}x"
        f"{_format_signed_term(beta_phase1_nem, '(1-x)')}"
        f"{_format_signed_constant(constant)}"
    )


def _olmix_formula(fit: OlmixLoglinearFit) -> str:
    """Return a compact Olmix legend label in dot-product notation."""
    return (
        rf"Olmix: $e^{{{_format_float(fit.log_c)}}} + e^{{\beta \cdot w}}$"
        "\n"
        rf"$\beta = {_format_vector(fit.coefficients)}$"
    )


def _panel_note(*, linear_rmse: float, olmix_rmse: float) -> str:
    """Build a compact per-panel annotation."""
    return (
        f"Subset RMSE\nLinear {linear_rmse:.4f}\nOlmix {olmix_rmse:.4f}"
        "\n"
        r"$w = (w_{0,\mathrm{Nem}},\, w_{0,\mathrm{SC}},\, w_{1,\mathrm{Nem}},\, w_{1,\mathrm{SC}})$"
    )


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
    y_linear: np.ndarray,
    y_olmix: np.ndarray,
    linear_formula: str,
    olmix_formula: str,
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

    ax.plot(x_grid, y_linear, color=LINEAR_COLOR, linewidth=2.7, label=linear_formula, zorder=3)
    linear_min_idx = int(np.argmin(y_linear))
    ax.scatter(
        [x_grid[linear_min_idx]],
        [y_linear[linear_min_idx]],
        s=72,
        color=LINEAR_COLOR,
        edgecolors=MIN_MARKER_COLOR,
        linewidths=0.9,
        zorder=5,
    )

    ax.plot(x_grid, y_olmix, color=OLMIX_COLOR, linewidth=2.7, label=olmix_formula, zorder=3)
    olmix_min_idx = int(np.argmin(y_olmix))
    ax.scatter(
        [x_grid[olmix_min_idx]],
        [y_olmix[olmix_min_idx]],
        s=72,
        color=OLMIX_COLOR,
        edgecolors=MIN_MARKER_COLOR,
        linewidths=0.9,
        zorder=5,
    )

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
        fontsize=11.2,
        fontweight="semibold",
        color="#334155",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "alpha": 0.96},
    )
    ax.legend(loc="upper right", fontsize=10.4, frameon=True, framealpha=0.97)


def _fit_panel_models(packet: PacketData) -> tuple[LinearFit, OlmixLoglinearFit]:
    """Fit the two overlay models on one packet."""
    linear_fit = _fit_linear_model(packet)
    olmix_fit = fit_olmix_loglinear_model(packet.w, packet.y)
    return linear_fit, olmix_fit


def _panel_summary(
    *,
    packet: PacketData,
    linear_fit: LinearFit,
    olmix_fit: OlmixLoglinearFit,
) -> PanelFitSummary:
    """Build one JSON-serializable panel summary."""
    linear_pred = linear_fit.predict(packet.w)
    olmix_pred = olmix_fit.predict(_flatten_weights(packet.w))
    return PanelFitSummary(
        linear_formula=_linear_formula(linear_fit),
        olmix_formula=_olmix_formula(olmix_fit),
        linear_subset_rmse=_rmse(packet.y, linear_pred),
        olmix_subset_rmse=_rmse(packet.y, olmix_pred),
        linear_metrics_on_subset=regression_metrics(packet.frame, packet.name_col, packet.y, linear_pred),
        olmix_metrics_on_subset=regression_metrics(packet.frame, packet.name_col, packet.y, olmix_pred),
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

    subset_linear_fit, subset_olmix_fit = _fit_panel_models(subset_packet_data)
    full_linear_fit, full_olmix_fit = _fit_panel_models(completed_packet)

    x_obs = subset_packet_data.frame["phase_1_starcoder"].to_numpy(dtype=float)
    y_obs = subset_packet_data.y

    x_grid = np.linspace(0.0, 1.0, GRID_SIZE, dtype=float)
    w_grid = _build_slice_weights(x_grid)
    linear_subset_curve = subset_linear_fit.predict(w_grid)
    olmix_subset_curve = subset_olmix_fit.predict(_flatten_weights(w_grid))
    linear_full_curve = full_linear_fit.predict(w_grid)
    olmix_full_curve = full_olmix_fit.predict(_flatten_weights(w_grid))

    subset_linear_pred = subset_linear_fit.predict(subset_packet_data.w)
    subset_olmix_pred = subset_olmix_fit.predict(_flatten_weights(subset_packet_data.w))
    full_linear_pred = full_linear_fit.predict(subset_packet_data.w)
    full_olmix_pred = full_olmix_fit.predict(_flatten_weights(subset_packet_data.w))

    starcoder_phase1_epochs = float(completed_packet.c1[1])

    output_frame = subset_packet_data.frame.copy()
    output_frame["subset_linear_prediction"] = subset_linear_pred
    output_frame["subset_olmix_prediction"] = subset_olmix_pred
    output_frame["full_linear_prediction"] = full_linear_pred
    output_frame["full_olmix_prediction"] = full_olmix_pred
    output_frame.to_csv(OUTPUT_CSV, index=False)

    summary = {
        "target": STARCODER_TARGET,
        "n_total_runs": len(completed_packet.frame),
        "n_subset_runs": len(subset_packet_data.frame),
        "subset_fit": (
            _panel_summary(
                packet=subset_packet_data,
                linear_fit=subset_linear_fit,
                olmix_fit=subset_olmix_fit,
            ).__dict__
        ),
        "all_data_fit": (
            _panel_summary(
                packet=subset_packet_data,
                linear_fit=full_linear_fit,
                olmix_fit=full_olmix_fit,
            ).__dict__
        ),
        "observed_subset_min": {
            "phase_1_starcoder": float(x_obs[np.argmin(y_obs)]),
            "bpb": float(np.min(y_obs)),
        },
        "subset_linear_slice_min": {
            "phase_1_starcoder": float(x_grid[np.argmin(linear_subset_curve)]),
            "bpb": float(np.min(linear_subset_curve)),
            "epochs": _epochs_from_weight(float(x_grid[np.argmin(linear_subset_curve)]), starcoder_phase1_epochs),
        },
        "subset_olmix_slice_min": {
            "phase_1_starcoder": float(x_grid[np.argmin(olmix_subset_curve)]),
            "bpb": float(np.min(olmix_subset_curve)),
            "epochs": _epochs_from_weight(float(x_grid[np.argmin(olmix_subset_curve)]), starcoder_phase1_epochs),
        },
        "all_data_linear_slice_min": {
            "phase_1_starcoder": float(x_grid[np.argmin(linear_full_curve)]),
            "bpb": float(np.min(linear_full_curve)),
            "epochs": _epochs_from_weight(float(x_grid[np.argmin(linear_full_curve)]), starcoder_phase1_epochs),
        },
        "all_data_olmix_slice_min": {
            "phase_1_starcoder": float(x_grid[np.argmin(olmix_full_curve)]),
            "bpb": float(np.min(olmix_full_curve)),
            "epochs": _epochs_from_weight(float(x_grid[np.argmin(olmix_full_curve)]), starcoder_phase1_epochs),
        },
    }
    OUTPUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True))

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)

    y_min = float(
        min(
            np.min(y_obs),
            np.min(linear_subset_curve),
            np.min(olmix_subset_curve),
            np.min(linear_full_curve),
            np.min(olmix_full_curve),
        )
    )
    y_max = float(
        max(
            np.max(y_obs),
            np.max(linear_subset_curve),
            np.max(olmix_subset_curve),
            np.max(linear_full_curve),
            np.max(olmix_full_curve),
        )
    )
    y_pad = 0.06 * (y_max - y_min)

    _plot_panel(
        axes[0],
        x_obs=x_obs,
        y_obs=y_obs,
        x_grid=x_grid,
        y_linear=linear_subset_curve,
        y_olmix=olmix_subset_curve,
        linear_formula=_linear_formula(subset_linear_fit),
        olmix_formula=_olmix_formula(subset_olmix_fit),
        panel_title="Fit on U-Shaped Subset",
        panel_note=_panel_note(
            linear_rmse=_rmse(y_obs, subset_linear_pred),
            olmix_rmse=_rmse(y_obs, subset_olmix_pred),
        ),
        starcoder_phase1_epochs=starcoder_phase1_epochs,
    )
    _plot_panel(
        axes[1],
        x_obs=x_obs,
        y_obs=y_obs,
        x_grid=x_grid,
        y_linear=linear_full_curve,
        y_olmix=olmix_full_curve,
        linear_formula=_linear_formula(full_linear_fit),
        olmix_formula=_olmix_formula(full_olmix_fit),
        panel_title="Fit on All 2-Phase StarCoder Runs",
        panel_note=_panel_note(
            linear_rmse=_rmse(y_obs, full_linear_pred),
            olmix_rmse=_rmse(y_obs, full_olmix_pred),
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
