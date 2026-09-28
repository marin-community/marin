# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""Compare DSP-family StarCoder fits on the phase-0 Nemotron slice.

This mirrors ``plot_grp_starcoder_u_shape_fit.py``. Separate-heads, canonical
DSP, and effective-exposure DSP are each fit once on the U-shaped slice where
phase 0 is all Nemotron and once on all completed two-phase StarCoder runs. All
fits are evaluated on the same slice.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_table9_phase_split_dsp_300m as phase_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_two_phase_canonical_bowl_candidates_300m as bowl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.plot_lf_sepheads_kl_sweep_300m import (
    SEP_L2,
    _gridmu,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.plot_grp_starcoder_u_shape_fit import (
    FIGSIZE,
    FOOTNOTE,
    GRID_SIZE,
    _build_slice_weights,
    _plot_panel,
    _rmse,
    _sort_packet_by_column,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.starcoder_grp import (
    load_completed_two_phase_starcoder_packet,
    subset_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    STARCODER_TARGET,
    PacketData,
    regression_metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PNG = SCRIPT_DIR / "separate_heads_starcoder_u_shape_fit.png"
OUTPUT_CSV = SCRIPT_DIR / "separate_heads_starcoder_u_shape_fit_subset_predictions.csv"
OUTPUT_SUMMARY_JSON = SCRIPT_DIR / "separate_heads_starcoder_u_shape_fit_summary.json"

DPI = 300
DSP_L2 = 0.01
TITLE = "Two-Phase StarCoder: Separate Heads vs DSP"
SUBTITLE = "Same training rows per panel; separate heads L2 = 0.1, DSP variants L2 = 0.01"

SEPARATE_HEADS_COLOR = "#15803d"
CANONICAL_DSP_COLOR = "#2563eb"
EFFECTIVE_EXPOSURE_DSP_COLOR = "#d97706"


@dataclass(frozen=True)
class SeparateHeadsModel:
    """Two independent phase-specific asymmetric exposure bowls."""

    c0: np.ndarray
    c1: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    intercept: float
    coef: np.ndarray

    def predict(self, weights: np.ndarray) -> np.ndarray:
        """Predict BPB for two-phase mixture weights."""
        zero_c0 = np.zeros_like(self.c0)
        design0 = bowl.abowl_design(weights, self.c0, self.c1, self.mu0, 0.0)
        design1 = bowl.abowl_design(weights, zero_c0, self.c1, self.mu1, 1.0)
        design = np.hstack([design0, design1])
        return np.asarray(self.intercept + design @ self.coef, dtype=float)


def fit_separate_heads(packet: PacketData) -> SeparateHeadsModel:
    """Fit the same separate-heads form used by the validated KL sweep."""
    zero_c0 = np.zeros_like(packet.c0)
    mu0 = _gridmu(packet.w, packet.c0, 0.0, packet.c1, packet.y, SEP_L2)
    mu1 = _gridmu(packet.w, zero_c0, 1.0, packet.c1, packet.y, SEP_L2)
    design0 = bowl.abowl_design(packet.w, packet.c0, packet.c1, mu0, 0.0)
    design1 = bowl.abowl_design(packet.w, zero_c0, packet.c1, mu1, 1.0)
    intercept, coef = bowl.fit_head(np.hstack([design0, design1]), packet.y, SEP_L2)
    return SeparateHeadsModel(
        c0=np.asarray(packet.c0, dtype=float),
        c1=np.asarray(packet.c1, dtype=float),
        mu0=np.asarray(mu0, dtype=float),
        mu1=np.asarray(mu1, dtype=float),
        intercept=float(intercept),
        coef=np.asarray(coef, dtype=float),
    )


def dsp_packet(packet: PacketData) -> dsp.PacketData:
    """Convert the StarCoder packet to the exact standalone DSP packet type."""
    return dsp.PacketData(
        frame=packet.frame.copy(),
        name_col=packet.name_col,
        y=np.asarray(packet.y, dtype=float),
        w=np.asarray(packet.w, dtype=float),
        m=packet.m,
        c0=np.asarray(packet.c0, dtype=float),
        c1=np.asarray(packet.c1, dtype=float),
        domain_names=list(packet.domain_names),
    )


def fit_dsp(packet: PacketData, variant: str) -> dsp.FittedDSPModel:
    """Fit one exact DSP variant with the established comparison settings."""
    model, _tuning = phase_dsp.fit_variant_with_l2(
        dsp_packet(packet),
        variant,
        DSP_L2,
        maxiter=40,
        coarse_top_k=3,
        basin_hopping_iters=0,
    )
    return model


def model_summary(model: SeparateHeadsModel, domain_names: list[str]) -> dict[str, object]:
    """Return labeled model parameters for the JSON artifact."""
    m = len(domain_names)
    return {
        "intercept": model.intercept,
        "l2": SEP_L2,
        "nominal_parameter_count": 4 * m + 3,
        "domains": domain_names,
        "phase_0_preferred_log_exposure": model.mu0.tolist(),
        "phase_1_preferred_log_exposure": model.mu1.tolist(),
        "phase_0_underexposure_coef": model.coef[:m].tolist(),
        "phase_0_overexposure_coef": model.coef[m : 2 * m].tolist(),
        "phase_1_underexposure_coef": model.coef[2 * m : 3 * m].tolist(),
        "phase_1_overexposure_coef": model.coef[3 * m :].tolist(),
    }


def dsp_model_summary(model: dsp.FittedDSPModel) -> dict[str, object]:
    """Return JSON-serializable DSP parameters."""
    params = {
        key: value.tolist() if isinstance(value, np.ndarray) else float(value) for key, value in model.params.items()
    }
    return {
        "variant": model.variant.name,
        "description": model.variant.description,
        "intercept": model.intercept,
        "l2": DSP_L2,
        "total_parameter_count": model.total_param_count,
        "params": params,
        "benefit_coef": model.benefit_coef.tolist(),
        "penalty_coef": model.penalty_coef.tolist(),
    }


def minimum_summary(x_grid: np.ndarray, curve: np.ndarray, starcoder_phase1_epochs: float) -> dict[str, float]:
    """Summarize a fitted curve's minimum on the displayed slice."""
    min_idx = int(np.argmin(curve))
    return {
        "phase_1_starcoder_weight": float(x_grid[min_idx]),
        "phase_1_starcoder_epochs": float(x_grid[min_idx] * starcoder_phase1_epochs),
        "predicted_bpb": float(curve[min_idx]),
    }


def add_dsp_overlays(
    axis: plt.Axes,
    x_grid: np.ndarray,
    canonical_curve: np.ndarray,
    effective_exposure_curve: np.ndarray,
) -> None:
    """Overlay canonical and effective-exposure DSP curves and minima."""
    for curve, color, label, linestyle, marker in (
        (canonical_curve, CANONICAL_DSP_COLOR, "Canonical DSP", "--", "s"),
        (
            effective_exposure_curve,
            EFFECTIVE_EXPOSURE_DSP_COLOR,
            "Effective-exposure DSP",
            "-.",
            "D",
        ),
    ):
        axis.plot(x_grid, curve, color=color, linewidth=2.7, linestyle=linestyle, label=label, zorder=3)
        min_idx = int(np.argmin(curve))
        axis.scatter(
            [x_grid[min_idx]],
            [curve[min_idx]],
            s=58,
            marker=marker,
            facecolors="white",
            edgecolors=color,
            linewidths=1.7,
            zorder=5,
        )


def main() -> None:
    packet = load_completed_two_phase_starcoder_packet(target=STARCODER_TARGET)
    slice_mask = packet.frame["phase_0_nemotron_full"].round(4).eq(1.0).to_numpy(dtype=bool)
    slice_packet = _sort_packet_by_column(subset_packet(packet, slice_mask), "phase_1_starcoder")

    subset_model = fit_separate_heads(slice_packet)
    full_model = fit_separate_heads(packet)
    canonical_subset_model = fit_dsp(slice_packet, "canonical")
    canonical_full_model = fit_dsp(packet, "canonical")
    effective_exposure_subset_model = fit_dsp(slice_packet, "effective_exposure")
    effective_exposure_full_model = fit_dsp(packet, "effective_exposure")

    x_obs = slice_packet.frame["phase_1_starcoder"].to_numpy(dtype=float)
    y_obs = slice_packet.y
    x_grid = np.linspace(0.0, 1.0, GRID_SIZE, dtype=float)
    grid_weights = _build_slice_weights(x_grid)
    subset_curve = subset_model.predict(grid_weights)
    full_curve = full_model.predict(grid_weights)
    subset_prediction = subset_model.predict(slice_packet.w)
    full_prediction = full_model.predict(slice_packet.w)
    canonical_subset_curve = dsp.predict(canonical_subset_model, grid_weights)
    canonical_full_curve = dsp.predict(canonical_full_model, grid_weights)
    canonical_subset_prediction = dsp.predict(canonical_subset_model, slice_packet.w)
    canonical_full_prediction = dsp.predict(canonical_full_model, slice_packet.w)
    effective_exposure_subset_curve = dsp.predict(effective_exposure_subset_model, grid_weights)
    effective_exposure_full_curve = dsp.predict(effective_exposure_full_model, grid_weights)
    effective_exposure_subset_prediction = dsp.predict(effective_exposure_subset_model, slice_packet.w)
    effective_exposure_full_prediction = dsp.predict(effective_exposure_full_model, slice_packet.w)

    output_frame = slice_packet.frame.copy()
    output_frame["separate_heads_subset_fit_prediction"] = subset_prediction
    output_frame["separate_heads_all_data_fit_prediction"] = full_prediction
    output_frame["canonical_dsp_subset_fit_prediction"] = canonical_subset_prediction
    output_frame["canonical_dsp_all_data_fit_prediction"] = canonical_full_prediction
    output_frame["effective_exposure_dsp_subset_fit_prediction"] = effective_exposure_subset_prediction
    output_frame["effective_exposure_dsp_all_data_fit_prediction"] = effective_exposure_full_prediction
    output_frame.to_csv(OUTPUT_CSV, index=False)

    starcoder_phase1_epochs = float(packet.c1[1])
    summary = {
        "target": STARCODER_TARGET,
        "n_total_completed_runs": len(packet.frame),
        "n_slice_runs": len(slice_packet.frame),
        "subset_fit_model": model_summary(subset_model, packet.domain_names),
        "all_data_fit_model": model_summary(full_model, packet.domain_names),
        "canonical_dsp_subset_fit_model": dsp_model_summary(canonical_subset_model),
        "canonical_dsp_all_data_fit_model": dsp_model_summary(canonical_full_model),
        "effective_exposure_dsp_subset_fit_model": dsp_model_summary(effective_exposure_subset_model),
        "effective_exposure_dsp_all_data_fit_model": dsp_model_summary(effective_exposure_full_model),
        "subset_fit_metrics_on_slice": regression_metrics(
            slice_packet.frame,
            slice_packet.name_col,
            y_obs,
            subset_prediction,
        ),
        "all_data_fit_metrics_on_slice": regression_metrics(
            slice_packet.frame,
            slice_packet.name_col,
            y_obs,
            full_prediction,
        ),
        "canonical_dsp_subset_fit_metrics_on_slice": regression_metrics(
            slice_packet.frame,
            slice_packet.name_col,
            y_obs,
            canonical_subset_prediction,
        ),
        "canonical_dsp_all_data_fit_metrics_on_slice": regression_metrics(
            slice_packet.frame,
            slice_packet.name_col,
            y_obs,
            canonical_full_prediction,
        ),
        "effective_exposure_dsp_subset_fit_metrics_on_slice": regression_metrics(
            slice_packet.frame,
            slice_packet.name_col,
            y_obs,
            effective_exposure_subset_prediction,
        ),
        "effective_exposure_dsp_all_data_fit_metrics_on_slice": regression_metrics(
            slice_packet.frame,
            slice_packet.name_col,
            y_obs,
            effective_exposure_full_prediction,
        ),
        "observed_slice_minimum": {
            "phase_1_starcoder_weight": float(x_obs[np.argmin(y_obs)]),
            "bpb": float(np.min(y_obs)),
        },
        "subset_fit_slice_minimum": minimum_summary(x_grid, subset_curve, starcoder_phase1_epochs),
        "all_data_fit_slice_minimum": minimum_summary(x_grid, full_curve, starcoder_phase1_epochs),
        "canonical_dsp_subset_fit_slice_minimum": minimum_summary(
            x_grid, canonical_subset_curve, starcoder_phase1_epochs
        ),
        "canonical_dsp_all_data_fit_slice_minimum": minimum_summary(
            x_grid, canonical_full_curve, starcoder_phase1_epochs
        ),
        "effective_exposure_dsp_subset_fit_slice_minimum": minimum_summary(
            x_grid, effective_exposure_subset_curve, starcoder_phase1_epochs
        ),
        "effective_exposure_dsp_all_data_fit_slice_minimum": minimum_summary(
            x_grid, effective_exposure_full_curve, starcoder_phase1_epochs
        ),
    }
    OUTPUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    all_curves = (
        subset_curve,
        full_curve,
        canonical_subset_curve,
        canonical_full_curve,
        effective_exposure_subset_curve,
        effective_exposure_full_curve,
    )
    y_min = float(min(np.min(y_obs), *(np.min(curve) for curve in all_curves)))
    y_max = float(max(np.max(y_obs), *(np.max(curve) for curve in all_curves)))
    y_pad = 0.06 * (y_max - y_min)

    _plot_panel(
        axes[0],
        x_obs=x_obs,
        y_obs=y_obs,
        x_grid=x_grid,
        y_fit=subset_curve,
        fit_color=SEPARATE_HEADS_COLOR,
        fit_label="Separate heads",
        panel_title="Fit on U-Shaped Subset",
        panel_note=(
            "Slice RMSE\n"
            f"Separate heads  {_rmse(y_obs, subset_prediction):.4f}\n"
            f"Canonical DSP   {_rmse(y_obs, canonical_subset_prediction):.4f}\n"
            f"Eff-exp DSP       {_rmse(y_obs, effective_exposure_subset_prediction):.4f}"
        ),
        starcoder_phase1_epochs=starcoder_phase1_epochs,
    )
    add_dsp_overlays(axes[0], x_grid, canonical_subset_curve, effective_exposure_subset_curve)
    _plot_panel(
        axes[1],
        x_obs=x_obs,
        y_obs=y_obs,
        x_grid=x_grid,
        y_fit=full_curve,
        fit_color=SEPARATE_HEADS_COLOR,
        fit_label="Separate heads",
        panel_title="Fit on All 2-Phase StarCoder Runs",
        panel_note=(
            "Slice RMSE\n"
            f"Separate heads  {_rmse(y_obs, full_prediction):.4f}\n"
            f"Canonical DSP   {_rmse(y_obs, canonical_full_prediction):.4f}\n"
            f"Eff-exp DSP       {_rmse(y_obs, effective_exposure_full_prediction):.4f}"
        ),
        starcoder_phase1_epochs=starcoder_phase1_epochs,
    )
    add_dsp_overlays(axes[1], x_grid, canonical_full_curve, effective_exposure_full_curve)

    axes[0].set_ylabel(STARCODER_TARGET, fontsize=13.5, fontweight="semibold")
    for axis in axes:
        axis.set_xlim(0.0, 1.0)
    axes[0].set_ylim(y_min - y_pad, y_max + y_pad)

    fig.suptitle(TITLE, fontsize=24, fontweight="bold", y=0.985)
    fig.text(0.5, 0.905, SUBTITLE, ha="center", fontsize=15.5, color="#475569")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.855),
        ncol=4,
        frameon=False,
        fontsize=11.5,
    )
    fig.text(0.5, 0.045, FOOTNOTE, ha="center", fontsize=11.5, color="#64748b")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.68, bottom=0.12, wspace=0.16)
    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUTPUT_PNG}")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
