# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, E501

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
#   "tabulate",
# ]
# ///
"""Benchmark a cumulative-learning plus recency-residual DSP on StarCoder.

Separate phase heads model phase 0 and phase 1 as unrelated response surfaces.
This script instead gives every token a cumulative learning/overexposure
effect and gives phase-1 tokens an additional recency/retention residual:

    L(w) = b
         - sum_i a_i S_i(e0_i + e1_i)
         + sum_i p_i H_i(e0_i + e1_i)
         - sum_i r_i S_i^late(e1_i)
         + sum_i q_i H_i^late(e1_i)

where S is a saturating benefit and H is a soft overexposure penalty. The two
channels have exposure-normalized domain scales plus one global scale and
threshold shift each. This keeps the nonlinear parameter count independent of
the number of domains while retaining nonnegative, mechanistic linear heads.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.plot_grp_starcoder_u_shape_fit import (
    _build_slice_weights,
    _sort_packet_by_column,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.plot_separate_heads_starcoder_u_shape_fit import (
    DSP_L2,
    SEP_L2,
    fit_dsp,
    fit_separate_heads,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.starcoder_grp import (
    load_completed_two_phase_starcoder_packet,
    subset_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    STARCODER_TARGET,
    PacketData,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "reference_outputs" / "starcoder_cumulative_recency_dsp_20260710"
OUTPUT_PNG = OUTPUT_DIR / "starcoder_cumulative_recency_fit.png"
OUTPUT_METRICS_CSV = OUTPUT_DIR / "model_comparison.csv"
OUTPUT_SLICE_CSV = OUTPUT_DIR / "slice_predictions.csv"
OUTPUT_GRID_CSV = OUTPUT_DIR / "slice_curve_grid.csv"
OUTPUT_SUMMARY_JSON = OUTPUT_DIR / "fit_summary.json"
OUTPUT_REPORT = OUTPUT_DIR / "report.md"

CV_SEED = 0
N_SPLITS = 5
L2_GRID = (0.0, 1e-5, 1e-4, 3e-4, 1e-3, 1e-2, 1e-1)
GRID_SIZE = 1001
SHAPE_BOUND = 3.0

CUMULATIVE_RECENCY_COLOR = "#7f1d1d"
SEPARATE_HEADS_COLOR = "#15803d"
EFFECTIVE_EXPOSURE_COLOR = "#d97706"
OBSERVED_COLOR = "#111827"


@dataclass(frozen=True)
class ChannelBase:
    """Deterministic domain scales for one exposure channel."""

    rho: np.ndarray
    tau: np.ndarray


@dataclass(frozen=True)
class CumulativeRecencyModel:
    """Cumulative learning plus phase-1 recency residual."""

    cumulative_base: ChannelBase
    recency_base: ChannelBase
    shape_offsets: np.ndarray
    intercept: float
    coef: np.ndarray
    l2: float
    c0: np.ndarray
    c1: np.ndarray

    @property
    def num_domains(self) -> int:
        return len(self.c0)

    @property
    def parameter_count(self) -> int:
        # Four nonnegative amplitudes per domain, four global shape offsets,
        # and one intercept. Per-domain base scales are deterministic.
        return 4 * self.num_domains + 5

    def predict(self, weights: np.ndarray) -> np.ndarray:
        """Predict BPB for two-phase mixture weights."""
        design = design_matrix(
            weights,
            self.c0,
            self.c1,
            self.cumulative_base,
            self.recency_base,
            self.shape_offsets,
        )
        return np.asarray(self.intercept + design @ self.coef, dtype=float)


def softplus(value: np.ndarray) -> np.ndarray:
    """Stable softplus."""
    return np.where(value > 20.0, value, np.log1p(np.exp(np.minimum(value, 20.0))))


def exposures(weights: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative and late-phase materialized exposures."""
    phase_0 = weights[:, 0, :] * c0[None, :]
    phase_1 = weights[:, 1, :] * c1[None, :]
    return phase_0 + phase_1, phase_1


def channel_base(exposure: np.ndarray) -> ChannelBase:
    """Set domain scales from observed positive exposure statistics."""
    positive = np.where(exposure > 1e-8, exposure, np.nan)
    median = np.nanmedian(positive, axis=0)
    fallback = float(np.nanmedian(positive))
    if not np.isfinite(fallback):
        fallback = 1.0
    median = np.where(np.isfinite(median), median, fallback)
    percentile = np.nanpercentile(positive, 75.0, axis=0)
    percentile = np.where(np.isfinite(percentile), percentile, median)
    return ChannelBase(
        rho=np.clip(1.0 / np.maximum(median, 1e-3), 1e-4, 2.0),
        tau=np.clip(np.log1p(percentile), -2.0, 8.0),
    )


def shifted_shape(base: ChannelBase, rho_shift: float, tau_shift: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply global shifts to deterministic domain response scales."""
    rho = np.clip(base.rho * np.exp(rho_shift), 1e-4, 2.0)
    tau = np.clip(base.tau + tau_shift, -2.0, 8.0)
    return rho, tau


def channel_features(exposure: np.ndarray, rho: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return saturating benefit and soft overexposure penalty features."""
    benefit = 1.0 - np.exp(-rho[None, :] * exposure)
    penalty = softplus(np.log1p(exposure) - tau[None, :]) ** 2
    return benefit, penalty


def design_matrix(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    cumulative_base: ChannelBase,
    recency_base: ChannelBase,
    shape_offsets: np.ndarray,
) -> np.ndarray:
    """Build signed columns for a nonnegative linear head."""
    cumulative, recency = exposures(weights, c0, c1)
    cumulative_rho, cumulative_tau = shifted_shape(cumulative_base, shape_offsets[0], shape_offsets[1])
    recency_rho, recency_tau = shifted_shape(recency_base, shape_offsets[2], shape_offsets[3])
    cumulative_benefit, cumulative_penalty = channel_features(cumulative, cumulative_rho, cumulative_tau)
    recency_benefit, recency_penalty = channel_features(recency, recency_rho, recency_tau)
    return np.hstack(
        [
            -cumulative_benefit,
            cumulative_penalty,
            -recency_benefit,
            recency_penalty,
        ]
    )


def fit_head(design: np.ndarray, targets: np.ndarray, l2: float) -> tuple[float, np.ndarray]:
    """Fit the nonnegative amplitudes with centered ridge NNLS."""
    design_mean = design.mean(axis=0, keepdims=True)
    target_mean = float(targets.mean())
    centered_design = design - design_mean
    centered_targets = targets - target_mean
    if l2 > 0.0:
        centered_design = np.vstack([centered_design, np.sqrt(l2) * np.eye(centered_design.shape[1])])
        centered_targets = np.concatenate([centered_targets, np.zeros(design.shape[1], dtype=float)])
    coef, _residual = nnls(centered_design, centered_targets)
    intercept = target_mean - float((design_mean @ coef).item())
    return intercept, np.asarray(coef, dtype=float)


def shape_starts() -> tuple[np.ndarray, ...]:
    """Return a compact deterministic bank of global channel starts."""
    starts = [
        np.asarray([a, b, c, d], dtype=float)
        for a in (-1.0, 0.0)
        for b in (-1.0, 0.0)
        for c in (-1.0, 0.0)
        for d in (-1.0, 0.0)
    ]
    starts.extend(
        [
            np.asarray([-2.0, -2.0, 0.0, -2.0], dtype=float),
            np.asarray([0.0, -2.0, 0.0, -2.0], dtype=float),
            np.asarray([1.0, 0.0, 1.0, 0.0], dtype=float),
        ]
    )
    return tuple(starts)


def fit_cumulative_recency(packet: PacketData, l2: float) -> CumulativeRecencyModel:
    """Fit one cumulative-plus-recency model by variable projection."""
    cumulative, recency = exposures(packet.w, packet.c0, packet.c1)
    cumulative_base = channel_base(cumulative)
    recency_base = channel_base(recency)

    def objective(shape_offsets: np.ndarray) -> float:
        design = design_matrix(
            packet.w,
            packet.c0,
            packet.c1,
            cumulative_base,
            recency_base,
            np.asarray(shape_offsets, dtype=float),
        )
        intercept, coef = fit_head(design, packet.y, l2)
        prediction = intercept + design @ coef
        return float(np.sqrt(np.mean((prediction - packet.y) ** 2)))

    bounds = [(-SHAPE_BOUND, SHAPE_BOUND)] * 4
    results = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-10, "maxls": 30},
        )
        for start in shape_starts()
    ]
    best = min(results, key=lambda result: float(result.fun))
    shape_offsets = np.asarray(best.x, dtype=float)
    design = design_matrix(
        packet.w,
        packet.c0,
        packet.c1,
        cumulative_base,
        recency_base,
        shape_offsets,
    )
    intercept, coef = fit_head(design, packet.y, l2)
    return CumulativeRecencyModel(
        cumulative_base=cumulative_base,
        recency_base=recency_base,
        shape_offsets=shape_offsets,
        intercept=intercept,
        coef=coef,
        l2=l2,
        c0=np.asarray(packet.c0, dtype=float),
        c1=np.asarray(packet.c1, dtype=float),
    )


def packet_rows(packet: PacketData, indices: np.ndarray) -> PacketData:
    """Subset packet rows by positional indices."""
    mask = np.isin(np.arange(len(packet.y)), indices)
    return subset_packet(packet, mask)


def nested_oof_cumulative_recency(packet: PacketData, l2: float) -> np.ndarray:
    """Refit nonlinear and linear parameters inside every fold."""
    prediction = np.empty_like(packet.y, dtype=float)
    folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    for train_indices, test_indices in folds.split(packet.w):
        model = fit_cumulative_recency(packet_rows(packet, train_indices), l2)
        prediction[test_indices] = model.predict(packet.w[test_indices])
    return prediction


def nested_oof_baseline(packet: PacketData, model_name: str) -> np.ndarray:
    """Compute matched full-refit OOF predictions for an existing baseline."""
    prediction = np.empty_like(packet.y, dtype=float)
    folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    for train_indices, test_indices in folds.split(packet.w):
        train_packet = packet_rows(packet, train_indices)
        if model_name == "separate_heads":
            model = fit_separate_heads(train_packet)
            prediction[test_indices] = model.predict(packet.w[test_indices])
        elif model_name == "effective_exposure":
            model = fit_dsp(train_packet, "effective_exposure")
            prediction[test_indices] = dsp.predict(model, packet.w[test_indices])
        else:
            raise ValueError(f"Unknown baseline {model_name!r}")
    return prediction


def metrics(targets: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Return regression metrics used in the comparison."""
    return {
        "rmse": float(np.sqrt(np.mean((prediction - targets) ** 2))),
        "spearman": float(spearmanr(targets, prediction).statistic),
    }


def curve_minimum(x_grid: np.ndarray, prediction: np.ndarray, phase_1_epoch_scale: float) -> dict[str, float]:
    """Return the predicted minimum on the phase-0 Nemotron slice."""
    index = int(np.argmin(prediction))
    return {
        "phase_1_starcoder_weight": float(x_grid[index]),
        "phase_1_starcoder_epochs": float(x_grid[index] * phase_1_epoch_scale),
        "predicted_bpb": float(prediction[index]),
    }


def fit_baseline_models(packet: PacketData) -> tuple[object, dsp.FittedDSPModel]:
    """Fit the two exact comparison baselines."""
    return fit_separate_heads(packet), fit_dsp(packet, "effective_exposure")


def predict_baseline(model: object, weights: np.ndarray, model_name: str) -> np.ndarray:
    """Predict from one exact comparison baseline."""
    if model_name == "separate_heads":
        return np.asarray(model.predict(weights), dtype=float)
    if model_name == "effective_exposure":
        return np.asarray(dsp.predict(model, weights), dtype=float)
    raise ValueError(f"Unknown baseline {model_name!r}")


def plot_curves(
    slice_packet: PacketData,
    x_grid: np.ndarray,
    curves: dict[str, dict[str, np.ndarray]],
    summaries: dict[str, dict[str, object]],
) -> None:
    """Plot slice-only and all-panel fits on the diagnostic slice."""
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), sharex=True, sharey=True)
    labels = {
        "cumulative_recency": "Cumulative + recency DSP",
        "separate_heads": "Separate heads",
        "effective_exposure": "Effective-exposure DSP",
    }
    colors = {
        "cumulative_recency": CUMULATIVE_RECENCY_COLOR,
        "separate_heads": SEPARATE_HEADS_COLOR,
        "effective_exposure": EFFECTIVE_EXPOSURE_COLOR,
    }
    styles = {
        "cumulative_recency": "-",
        "separate_heads": "--",
        "effective_exposure": "-.",
    }
    x_observed = slice_packet.frame["phase_1_starcoder"].to_numpy(dtype=float)
    order = np.argsort(x_observed)
    for axis, fit_scope, title in zip(
        axes,
        ("slice_only", "all_runs"),
        ("Fit on U-shaped slice", "Fit on all two-phase runs"),
        strict=True,
    ):
        axis.scatter(
            x_observed[order],
            slice_packet.y[order],
            color=OBSERVED_COLOR,
            edgecolor="white",
            linewidth=0.7,
            s=39,
            label="Observed runs",
            zorder=5,
        )
        annotation_lines = []
        for model_name in ("cumulative_recency", "separate_heads", "effective_exposure"):
            curve = curves[fit_scope][model_name]
            axis.plot(
                x_grid,
                curve,
                color=colors[model_name],
                linestyle=styles[model_name],
                linewidth=2.6,
                label=labels[model_name],
            )
            minimum_index = int(np.argmin(curve))
            axis.scatter(
                [x_grid[minimum_index]],
                [curve[minimum_index]],
                color=colors[model_name],
                edgecolor="white",
                linewidth=1.0,
                s=70,
                zorder=6,
            )
            rmse = summaries[fit_scope][model_name]["slice_rmse"]
            annotation_lines.append(f"{labels[model_name]}: {rmse:.4f}")
        axis.axvline(
            float(x_observed[np.argmin(slice_packet.y)]),
            color="#64748b",
            linestyle=":",
            linewidth=1.4,
            label="Observed minimum" if fit_scope == "slice_only" else None,
        )
        axis.text(
            0.02,
            0.97,
            "Slice RMSE\n" + "\n".join(annotation_lines),
            transform=axis.transAxes,
            va="top",
            fontsize=9.2,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
        )
        axis.set_title(title, fontsize=15)
        axis.set_xlabel("Phase 1 StarCoder weight")
        axis.grid(True, alpha=0.25, linestyle="--")
    axes[0].set_ylabel("Dolma 100 Programming Languages BPB")
    handles, labels_out = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels_out, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=4, frameon=False)
    figure.suptitle("StarCoder response shape: cumulative learning plus recency residual", fontsize=20, y=0.995)
    figure.text(
        0.5,
        0.015,
        "The cumulative channel sees all exposure; phase 1 contributes an additional retention/recency residual.",
        ha="center",
        color="#475569",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.0, 0.055, 1.0, 0.87))
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(figure)


def report_text(summary: dict[str, object], comparison: pd.DataFrame) -> str:
    """Build a concise Markdown report."""
    best = summary["selected_cumulative_recency"]
    rows = [
        "# Cumulative + recency DSP on the two-phase StarCoder diagnostic",
        "",
        "## Model",
        "",
        r"For domain $i$, let $e_i=e_i^{(0)}+e_i^{(1)}$ and $r_i=e_i^{(1)}$. The fitted form is",
        "",
        r"$$\hat L(w)=\beta_0-\sum_i a_i S_i(e_i)+\sum_i p_i H_i(e_i)-\sum_i a_i^{R}S_i^{R}(r_i)+\sum_i p_i^{R}H_i^{R}(r_i).$$",
        "",
        r"$S$ is a saturating benefit and $H$ is a soft overexposure penalty. Domain scales come from observed exposure statistics; each channel fits only a global saturation-scale shift and threshold shift. The model therefore has $4m+5$ fitted parameters and nests the aggregate-only model when recency coefficients are zero.",
        "",
        "## Results",
        "",
        comparison.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Selected ridge strength by full-refit OOF RMSE: `{best['l2']}`.",
        "",
        "## Interpretation",
        "",
        f"- Slice-only RMSE is `{best['slice_only_rmse']:.6f}` and the predicted minimum is at phase-1 StarCoder weight `{best['slice_only_minimum']['phase_1_starcoder_weight']:.3f}`; the observed minimum is `{summary['observed_minimum']['phase_1_starcoder_weight']:.3f}`.",
        f"- Fitting all 116 runs gives slice RMSE `{best['all_runs_slice_rmse']:.6f}` and full-refit OOF RMSE `{best['nested_oof_rmse']:.6f}`.",
        f"- Holding out the complete U-shaped slice gives RMSE `{best['leave_slice_out_rmse']:.6f}`. This is the relevant check that the response shape is not obtained only by memorizing the dense slice.",
        "- The gain identifies a missing cumulative-learning link in separate heads: phase-1 exposure must affect both total learning and a recency residual. Treating the two phases as unrelated response surfaces discards that shared pathway.",
        "",
        "## Caveats",
        "",
        "- Ridge strength is screened on this small StarCoder panel; the selected value is exploratory until tested on the 300M and production panels.",
        "- The recency penalty is phenomenological. Its coefficients should be ablated before adopting the full form; a benefit-only recency residual is the more parsimonious nested alternative.",
        "- Good interpolation and slice transfer do not establish that an unconstrained high-dimensional optimum will validate. Proposal quality still requires support-aware diagnostics and 3e18 validation.",
        "",
    ]
    return "\n".join(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    packet = load_completed_two_phase_starcoder_packet(target=STARCODER_TARGET)
    slice_mask = packet.frame["phase_0_nemotron_full"].round(4).eq(1.0).to_numpy(dtype=bool)
    slice_packet = _sort_packet_by_column(subset_packet(packet, slice_mask), "phase_1_starcoder")
    non_slice_packet = subset_packet(packet, ~slice_mask)
    x_grid = np.linspace(0.0, 1.0, GRID_SIZE, dtype=float)
    grid_weights = _build_slice_weights(x_grid)
    phase_1_epoch_scale = float(packet.c1[1])

    l2_rows: list[dict[str, float]] = []
    models_by_l2: dict[float, CumulativeRecencyModel] = {}
    for l2 in L2_GRID:
        full_model = fit_cumulative_recency(packet, l2)
        models_by_l2[l2] = full_model
        oof = nested_oof_cumulative_recency(packet, l2)
        holdout_model = fit_cumulative_recency(non_slice_packet, l2)
        holdout_prediction = holdout_model.predict(slice_packet.w)
        full_slice_prediction = full_model.predict(slice_packet.w)
        l2_rows.append(
            {
                "l2": l2,
                "nested_oof_rmse": metrics(packet.y, oof)["rmse"],
                "nested_oof_spearman": metrics(packet.y, oof)["spearman"],
                "leave_slice_out_rmse": metrics(slice_packet.y, holdout_prediction)["rmse"],
                "leave_slice_out_spearman": metrics(slice_packet.y, holdout_prediction)["spearman"],
                "all_runs_slice_rmse": metrics(slice_packet.y, full_slice_prediction)["rmse"],
                "all_runs_slice_spearman": metrics(slice_packet.y, full_slice_prediction)["spearman"],
            }
        )
    l2_frame = pd.DataFrame.from_records(l2_rows)
    selected_l2 = float(l2_frame.sort_values(["nested_oof_rmse", "leave_slice_out_rmse"]).iloc[0]["l2"])
    full_model = models_by_l2[selected_l2]
    slice_model = fit_cumulative_recency(slice_packet, selected_l2)
    holdout_model = fit_cumulative_recency(non_slice_packet, selected_l2)
    cumulative_oof = nested_oof_cumulative_recency(packet, selected_l2)

    separate_slice, effective_slice = fit_baseline_models(slice_packet)
    separate_full, effective_full = fit_baseline_models(packet)
    separate_holdout, effective_holdout = fit_baseline_models(non_slice_packet)
    separate_oof = nested_oof_baseline(packet, "separate_heads")
    effective_oof = nested_oof_baseline(packet, "effective_exposure")

    fitted_models = {
        "slice_only": {
            "cumulative_recency": slice_model,
            "separate_heads": separate_slice,
            "effective_exposure": effective_slice,
        },
        "all_runs": {
            "cumulative_recency": full_model,
            "separate_heads": separate_full,
            "effective_exposure": effective_full,
        },
    }
    curves: dict[str, dict[str, np.ndarray]] = {}
    fit_summaries: dict[str, dict[str, object]] = {}
    for fit_scope, models in fitted_models.items():
        curves[fit_scope] = {}
        fit_summaries[fit_scope] = {}
        for model_name, model in models.items():
            prediction = (
                model.predict(slice_packet.w)
                if model_name in {"cumulative_recency", "separate_heads"}
                else dsp.predict(model, slice_packet.w)
            )
            curve = (
                model.predict(grid_weights)
                if model_name in {"cumulative_recency", "separate_heads"}
                else dsp.predict(model, grid_weights)
            )
            curves[fit_scope][model_name] = np.asarray(curve, dtype=float)
            fit_summaries[fit_scope][model_name] = {
                "slice_rmse": metrics(slice_packet.y, prediction)["rmse"],
                "slice_spearman": metrics(slice_packet.y, prediction)["spearman"],
                "minimum": curve_minimum(x_grid, curve, phase_1_epoch_scale),
            }

    holdout_predictions = {
        "cumulative_recency": holdout_model.predict(slice_packet.w),
        "separate_heads": separate_holdout.predict(slice_packet.w),
        "effective_exposure": dsp.predict(effective_holdout, slice_packet.w),
    }
    oof_predictions = {
        "cumulative_recency": cumulative_oof,
        "separate_heads": separate_oof,
        "effective_exposure": effective_oof,
    }
    comparison_rows = []
    for model_name in ("cumulative_recency", "separate_heads", "effective_exposure"):
        comparison_rows.append(
            {
                "model": model_name,
                "l2": (
                    selected_l2
                    if model_name == "cumulative_recency"
                    else (SEP_L2 if model_name == "separate_heads" else DSP_L2)
                ),
                "parameter_count": (
                    full_model.parameter_count
                    if model_name == "cumulative_recency"
                    else (4 * packet.m + 3 if model_name == "separate_heads" else 4 * packet.m + 2)
                ),
                "slice_only_rmse": fit_summaries["slice_only"][model_name]["slice_rmse"],
                "all_runs_slice_rmse": fit_summaries["all_runs"][model_name]["slice_rmse"],
                "nested_oof_rmse": metrics(packet.y, oof_predictions[model_name])["rmse"],
                "nested_oof_spearman": metrics(packet.y, oof_predictions[model_name])["spearman"],
                "leave_slice_out_rmse": metrics(slice_packet.y, holdout_predictions[model_name])["rmse"],
                "leave_slice_out_spearman": metrics(slice_packet.y, holdout_predictions[model_name])["spearman"],
            }
        )
    comparison = pd.DataFrame.from_records(comparison_rows)
    comparison.to_csv(OUTPUT_METRICS_CSV, index=False)

    slice_output = slice_packet.frame[
        ["run_id", "phase_1_starcoder", "phase_1_starcoder_epochs", STARCODER_TARGET]
    ].copy()
    for model_name, prediction in holdout_predictions.items():
        slice_output[f"{model_name}_leave_slice_out_prediction"] = prediction
    for model_name, model in fitted_models["all_runs"].items():
        slice_output[f"{model_name}_all_runs_prediction"] = (
            predict_baseline(model, slice_packet.w, model_name)
            if model_name != "cumulative_recency"
            else model.predict(slice_packet.w)
        )
    slice_output.to_csv(OUTPUT_SLICE_CSV, index=False)

    grid_output = pd.DataFrame({"phase_1_starcoder_weight": x_grid})
    for fit_scope, model_curves in curves.items():
        for model_name, curve in model_curves.items():
            grid_output[f"{fit_scope}_{model_name}"] = curve
    grid_output.to_csv(OUTPUT_GRID_CSV, index=False)

    selected_row = l2_frame.loc[l2_frame["l2"].eq(selected_l2)].iloc[0]
    summary = {
        "target": STARCODER_TARGET,
        "n_total_completed_runs": len(packet.y),
        "n_slice_runs": len(slice_packet.y),
        "observed_minimum": {
            "phase_1_starcoder_weight": float(
                slice_packet.frame.iloc[int(np.argmin(slice_packet.y))]["phase_1_starcoder"]
            ),
            "bpb": float(np.min(slice_packet.y)),
        },
        "selected_cumulative_recency": {
            "l2": selected_l2,
            "parameter_count": full_model.parameter_count,
            "shape_offsets": full_model.shape_offsets.tolist(),
            "slice_only_rmse": fit_summaries["slice_only"]["cumulative_recency"]["slice_rmse"],
            "slice_only_minimum": fit_summaries["slice_only"]["cumulative_recency"]["minimum"],
            "all_runs_slice_rmse": fit_summaries["all_runs"]["cumulative_recency"]["slice_rmse"],
            "all_runs_minimum": fit_summaries["all_runs"]["cumulative_recency"]["minimum"],
            "nested_oof_rmse": float(selected_row["nested_oof_rmse"]),
            "nested_oof_spearman": float(selected_row["nested_oof_spearman"]),
            "leave_slice_out_rmse": float(selected_row["leave_slice_out_rmse"]),
            "leave_slice_out_spearman": float(selected_row["leave_slice_out_spearman"]),
        },
        "l2_sweep": l2_frame.to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
        "fit_scope_summaries": fit_summaries,
    }
    OUTPUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    OUTPUT_REPORT.write_text(report_text(summary, comparison))
    plot_curves(slice_packet, x_grid, curves, fit_summaries)

    print(comparison.to_string(index=False))
    print(f"Selected cumulative-recency L2: {selected_l2:g}")
    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
