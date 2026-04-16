# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "matplotlib", "numpy", "pandas"]
# ///
"""Plot raw-optimum convergence for the power-family-penalty GRP variant."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import json
from pathlib import Path

import fsspec
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_grp_penalty_calibration_variants import (
    _variant_start_bank,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    load_two_phase_many_candidate_summary_spec,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    load_generic_family_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    build_penalty_calibration_surrogate,
    optimize_penalty_calibration_model,
    penalty_calibration_params_from_metrics,
    tune_penalty_calibration_params,
)
from experiments.domain_phase_mix.static_batch_selection import (
    average_phase_tv_distance,
    retrospective_generic_selection,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
    OBJECTIVE_METRIC,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_penalty_raw_optima_baselines import (
    GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT,
    genericfamily_penalty_raw_optimum_summary,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_power_family_penalty_raw_subset_optima import (
    GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
    GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
    genericfamily_power_family_penalty_raw_subset_optimum_run_name,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    CSV_PATH,
    _mean_phase_tv_distance,
    _phase_weights_from_array,
    _subset_packet,
)

plt.rcParams["text.usetex"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
CURVE_POINTS_CSV = SCRIPT_DIR / "two_phase_many_grp_power_family_penalty_raw_curve_points.csv"
SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_grp_power_family_penalty_raw_summary.json"
PLOT_PATH = SCRIPT_DIR / "two_phase_many_grp_power_family_penalty_raw_convergence_tracks.png"
PLOT_BPB_ONLY_PATH = SCRIPT_DIR / "two_phase_many_grp_power_family_penalty_raw_bpb_only.png"
PLOT_BPB_TV_PATH = SCRIPT_DIR / "two_phase_many_grp_power_family_penalty_raw_bpb_and_tv.png"
PLOT_PHASE_MOVEMENT_PATH = SCRIPT_DIR / "two_phase_many_grp_power_family_penalty_raw_phase_movements.png"
PLOT_BPB_PHASE_MOVEMENT_PATH = SCRIPT_DIR / "two_phase_many_grp_power_family_penalty_raw_bpb_and_phase_movements.png"
PLOT_BPB_PHASE_MOVEMENT_WITH_REGMIX_PATH = (
    SCRIPT_DIR / "two_phase_many_grp_power_family_penalty_raw_bpb_and_phase_movements_with_regmix.png"
)
PLOT_BPB_ONLY_WITH_REGMIX_PATH = SCRIPT_DIR / "two_phase_many_grp_power_family_penalty_raw_bpb_with_regmix.png"
REGMIX_CURVE_POINTS_CSV = SCRIPT_DIR / "two_phase_many_regmix_raw_curve_points.csv"
TWO_PHASE_MANY_ALL_CSV = SCRIPT_DIR / "two_phase_many_all_60m_1p2b.csv"

VARIANT_NAME = "power_family_penalty"
RUN_NAME = "baseline_genericfamily_power_family_penalty_raw_optimum"
METHOD = "Powell"
COARSE_TOP_K = 1
CHECKPOINT_ROOT = "marin-us-east5/checkpoints/" + GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT
SUBSET_CHECKPOINT_ROOT = (
    "marin-us-east5/checkpoints/" + GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_SOURCE_EXPERIMENT
)
MAX_WORKERS = 8
BPB_LABEL_DECIMALS = 3
BASELINE_BPB_LINE_SPECS = (
    ("baseline_olmix_loglinear_uncheatable_bpb", "Olmix", "#B279A2"),
    ("baseline_proportional", "Proportional", "#4C78A8"),
    ("baseline_stratified", "Uniform (stratified)", "#6C6F7D"),
)
PROPORTIONAL_BPB_LINE_SPEC = (BASELINE_BPB_LINE_SPECS[1],)
REGMIX_VALIDATED_COLOR = "#7F3C8D"
REGMIX_PREDICTED_COLOR = "#5B3F95"
BPB_PANEL_TOP_WITH_REGMIX = 1.1
BPB_PANEL_TOP_WITH_REGMIX_PREDICTED = 1.2


def _format_bpb_label(value: float) -> str:
    return f"{value:.{BPB_LABEL_DECIMALS}f}"


def _validated_full_bpb() -> float | None:
    fs = fsspec.filesystem("gs")
    matches = sorted(fs.glob(f"{CHECKPOINT_ROOT}/{RUN_NAME}-*/checkpoints/eval_metrics.jsonl"))
    if not matches:
        return None
    payload: dict[str, float] | None = None
    with fs.open(matches[-1], "r") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
    if payload is None:
        return None
    value = payload.get(OBJECTIVE_METRIC)
    return None if value is None else float(value)


def _validated_subset_bpbs() -> dict[int, float]:
    fs = fsspec.filesystem("gs")
    realized: dict[int, float] = {}
    for subset_size in GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES:
        run_name = genericfamily_power_family_penalty_raw_subset_optimum_run_name(subset_size)
        matches = sorted(fs.glob(f"{SUBSET_CHECKPOINT_ROOT}/{run_name}-*/checkpoints/eval_metrics.jsonl"))
        if not matches:
            continue
        payload: dict[str, float] | None = None
        with fs.open(matches[-1], "r") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
        if payload is None:
            continue
        value = payload.get(OBJECTIVE_METRIC)
        if value is not None:
            realized[subset_size] = float(value)
    return realized


def _apply_realized_validated_bpbs(
    frame: pd.DataFrame, *, full_subset_size: int, best_observed_bpb: float
) -> pd.DataFrame:
    frame = frame.copy()
    realized = _validated_subset_bpbs()
    full_bpb = _validated_full_bpb()
    if full_bpb is not None:
        realized[full_subset_size] = full_bpb
    frame["actual_validated_bpb"] = frame["subset_size"].map(realized)
    frame["validated_prediction_error"] = frame["actual_validated_bpb"] - frame["predicted_optimum_value"]
    frame["validated_regret_at_1"] = frame["actual_validated_bpb"] - best_observed_bpb
    return frame


def _best_observed_in_subset(packet, subset_indices: np.ndarray) -> tuple[str, float]:
    subset_values = packet.base.y[subset_indices]
    best_local_idx = int(np.argmin(subset_values))
    best_idx = int(subset_indices[best_local_idx])
    return (
        str(packet.base.frame.iloc[best_idx][packet.base.name_col]),
        float(packet.base.y[best_idx]),
    )


def _weights_from_dict(
    phase_weights: dict[str, dict[str, float]],
    domain_names: list[str],
) -> np.ndarray:
    return np.asarray(
        [
            [float(phase_weights["phase_0"][domain_name]) for domain_name in domain_names],
            [float(phase_weights["phase_1"][domain_name]) for domain_name in domain_names],
        ],
        dtype=float,
    )


def _fit_subset_point(subset_size: int) -> dict[str, object]:
    _, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many_grp_power_family_penalty_raw",
    )
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    full_summary = genericfamily_penalty_raw_optimum_summary(VARIANT_NAME)
    best_full_observed_bpb = float(np.min(packet.base.y))
    start_bank = _variant_start_bank(VARIANT_NAME)
    if subset_size == len(packet.base.y):
        subset_indices = np.arange(len(packet.base.y), dtype=int)
    else:
        selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=subset_size, seed=0)
        subset_indices = np.asarray(selection.selected_indices, dtype=int)
    train_packet = _subset_packet(packet, subset_indices)
    _, _, tuning_metrics, _ = tune_penalty_calibration_params(
        train_packet,
        variant_name=VARIANT_NAME,
        start_bank=start_bank,
        method=METHOD,
        coarse_top_k=COARSE_TOP_K,
        seed=0,
    )
    params = penalty_calibration_params_from_metrics(tuning_metrics, VARIANT_NAME)
    model = build_penalty_calibration_surrogate(
        train_packet,
        params=params,
        variant_name=VARIANT_NAME,
    ).fit(train_packet.base.w, train_packet.base.y)
    optimizer_result, phase0, phase1 = optimize_penalty_calibration_model(train_packet, model, seed=0)
    deployment = np.stack([phase0, phase1], axis=0)
    fullswarm_predictions = model.predict(packet.base.w)
    chosen_idx = int(np.argmin(fullswarm_predictions))
    distances = average_phase_tv_distance(packet.base.w, deployment[None, :, :])
    nearest_idx = int(np.argmin(distances))
    subset_best_run_name, subset_best_bpb = _best_observed_in_subset(packet, subset_indices)
    phase_weights = _phase_weights_from_array(packet.base.domain_names, deployment)
    actual_validated_bpb = np.nan

    if subset_size == len(packet.base.y):
        for key, expected in full_summary.tuned_params.items():
            actual = float(params[key])
            if not np.isclose(actual, float(expected), atol=1e-8):
                raise ValueError(f"Full-data retune mismatch for {key}: expected {expected}, got {actual}")
        if not np.isclose(
            float(optimizer_result.fun),
            float(full_summary.raw_predicted_optimum_value),
            atol=1e-6,
        ):
            raise ValueError(
                "Full-data raw optimum mismatch: "
                f"expected {full_summary.raw_predicted_optimum_value}, got {float(optimizer_result.fun)}"
            )
        actual_validated_bpb = _validated_full_bpb()

    return {
        "subset_size": subset_size,
        "predicted_optimum_value": float(optimizer_result.fun),
        "subset_best_observed_run_name": subset_best_run_name,
        "subset_best_observed_bpb": subset_best_bpb,
        "fullswarm_chosen_run_name": str(packet.base.frame.iloc[chosen_idx][packet.base.name_col]),
        "fullswarm_chosen_value": float(packet.base.y[chosen_idx]),
        "fullswarm_regret_at_1": float(packet.base.y[chosen_idx] - best_full_observed_bpb),
        "nearest_observed_run_name": str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
        "nearest_observed_value": float(packet.base.y[nearest_idx]),
        "nearest_observed_tv_distance": float(distances[nearest_idx]),
        "tuning_objective": float(tuning_metrics["objective"]),
        "tuning_cv_rmse": float(tuning_metrics["cv_rmse"]),
        "tuning_cv_regret_at_1": float(tuning_metrics["cv_regret_at_1"]),
        "tuning_cv_foldmean_regret_at_1": float(tuning_metrics["cv_foldmean_regret_at_1"]),
        "tuning_lower_tail_optimism": float(tuning_metrics["lower_tail_optimism"]),
        "tuning_cv_depopt_best8": float(tuning_metrics["cv_depopt_best8"]),
        "tuning_cv_rawopt_nearest_tv": float(tuning_metrics["cv_rawopt_nearest_tv"]),
        "phase0_support_below_1e4": int(np.sum(deployment[0] < 1e-4)),
        "phase1_support_below_1e4": int(np.sum(deployment[1] < 1e-4)),
        "actual_validated_bpb": actual_validated_bpb,
        "phase_weights": phase_weights,
    }


def _curve_points() -> pd.DataFrame:
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    subset_sizes = (
        *GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
        len(packet.base.y),
    )
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(subset_sizes))) as executor:
        future_by_size = {executor.submit(_fit_subset_point, subset_size): subset_size for subset_size in subset_sizes}
        for future in as_completed(future_by_size):
            subset_size = future_by_size[future]
            row = future.result()
            rows.append(row)
            print(
                f"Finished subset_size={subset_size} predicted={float(row['predicted_optimum_value']):.6f} "
                f"chosen={float(row['fullswarm_chosen_value']):.6f}",
                flush=True,
            )

    rows.sort(key=lambda row: int(row["subset_size"]))
    previous_weights: np.ndarray | None = None
    for row in rows:
        weights = _weights_from_dict(row["phase_weights"], packet.base.domain_names)
        row["optimum_move_mean_phase_tv_vs_prev"] = (
            None if previous_weights is None else _mean_phase_tv_distance(weights, previous_weights)
        )
        previous_weights = weights

    frame = pd.DataFrame(rows)
    if "actual_validated_bpb" not in frame.columns:
        frame["actual_validated_bpb"] = np.nan
    return frame


def _load_existing_curve_points() -> pd.DataFrame:
    frame = pd.read_csv(CURVE_POINTS_CSV)
    if "phase_weights" in frame.columns:
        frame["phase_weights"] = frame["phase_weights"].map(
            lambda payload: json.loads(payload) if isinstance(payload, str) and payload else payload
        )
    return frame


def _baseline_bpbs() -> dict[str, float]:
    frame = pd.read_csv(TWO_PHASE_MANY_ALL_CSV, usecols=["run_name", OBJECTIVE_METRIC])
    result: dict[str, float] = {}
    for run_name, _, _ in BASELINE_BPB_LINE_SPECS:
        matches = frame.loc[frame["run_name"] == run_name, OBJECTIVE_METRIC].dropna()
        if matches.empty:
            raise ValueError(f"Missing baseline BPB for {run_name} in {TWO_PHASE_MANY_ALL_CSV}")
        result[run_name] = float(matches.iloc[-1])
    return result


def _plot_bpb_panel(
    ax_bpb: plt.Axes,
    frame: pd.DataFrame,
    *,
    cmap,
    baseline_bpbs: dict[str, float] | None = None,
    baseline_line_specs: tuple[tuple[str, str, str], ...] = BASELINE_BPB_LINE_SPECS,
    regmix_validated: pd.DataFrame | None = None,
) -> None:
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
        color="#4C78A8",
        marker="P",
        linewidth=1.8,
        linestyle=":",
        label="Best observed BPB in subset",
    )
    if baseline_bpbs is not None:
        for run_name, label, color in baseline_line_specs:
            ax_bpb.axhline(
                baseline_bpbs[run_name],
                color=color,
                linewidth=1.5,
                linestyle="--",
                alpha=0.95,
                zorder=1,
                label=f"{label} BPB",
            )
    validated = frame[frame["actual_validated_bpb"].notna()].copy()
    if not validated.empty:
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
    if regmix_validated is not None and not regmix_validated.empty:
        ax_bpb.plot(
            regmix_validated["subset_size"],
            regmix_validated["actual_validated_bpb"],
            color=REGMIX_VALIDATED_COLOR,
            marker="s",
            markersize=6,
            linewidth=1.8,
            linestyle="-.",
            label="RegMix validated BPB",
        )


def _plot(frame: pd.DataFrame) -> None:
    cmap = plt.colormaps["RdYlGn_r"]
    fig, (ax_bpb, ax_regret, ax_cvregret, ax_tailopt, ax_move) = plt.subplots(
        5,
        1,
        figsize=(10.2, 11.6),
        dpi=180,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.3, 1.0, 1.0, 1.0, 1.0], "hspace": 0.08},
    )

    _plot_bpb_panel(ax_bpb, frame, cmap=cmap)

    ax_regret.plot(
        frame["subset_size"],
        frame["fullswarm_regret_at_1"],
        color=cmap(0.78),
        marker="s",
        linewidth=2.2,
        label="Retrospective Regret@1",
    )
    ax_cvregret.plot(
        frame["subset_size"],
        frame["tuning_cv_foldmean_regret_at_1"],
        color=cmap(0.66),
        marker="^",
        linewidth=2.2,
        label="CV Mean Regret@1",
    )
    ax_tailopt.plot(
        frame["subset_size"],
        frame["tuning_lower_tail_optimism"],
        color=cmap(0.62),
        marker="v",
        linewidth=2.2,
        label="Tail optimism",
    )
    ax_move.plot(
        frame["subset_size"],
        frame["optimum_move_mean_phase_tv_vs_prev"],
        color=cmap(0.36),
        marker="D",
        linewidth=2.2,
        label="Raw-optimum movement (mean phase TV)",
    )
    ax_move.axhline(0.0, color="0.55", linewidth=1.0, linestyle=":")

    ax_bpb.set_title("Two-phase many-domain: GRP convergence (power-family penalty raw optimum)")
    ax_bpb.set_ylabel("Predicted BPB")
    ax_regret.set_ylabel("Regret@1")
    ax_cvregret.set_ylabel("CV Mean Regret@1")
    ax_tailopt.set_ylabel("Tail optimism")
    ax_move.set_ylabel("Mean phase TV")
    ax_move.set_xlabel("Observed runs used for fitting")
    ax_move.set_xticks(frame["subset_size"].tolist())
    ax_move.set_xlim(int(frame["subset_size"].min()), int(frame["subset_size"].max()))

    for axis in (ax_bpb, ax_regret, ax_cvregret, ax_tailopt, ax_move):
        axis.grid(True, alpha=0.25)
        handles = axis.get_lines()
        labels = [handle.get_label() for handle in handles if not handle.get_label().startswith("_")]
        if handles:
            axis.legend(handles, labels, loc="best", frameon=True)

    fig.savefig(PLOT_PATH, bbox_inches="tight")
    plt.close(fig)


def _plot_bpb_only(frame: pd.DataFrame) -> None:
    cmap = plt.colormaps["RdYlGn_r"]
    fig, ax_bpb = plt.subplots(
        1,
        1,
        figsize=(10.2, 4.4),
        dpi=180,
        constrained_layout=True,
    )
    _plot_bpb_panel(ax_bpb, frame, cmap=cmap)
    ax_bpb.set_title("Two-phase many-domain: GRP convergence (power-family penalty raw optimum)")
    ax_bpb.set_ylabel("Predicted BPB")
    ax_bpb.set_xlabel("Observed runs used for fitting")
    ax_bpb.set_xticks(frame["subset_size"].tolist())
    ax_bpb.set_xlim(int(frame["subset_size"].min()), int(frame["subset_size"].max()))
    ax_bpb.grid(True, alpha=0.25)
    handles = ax_bpb.get_lines()
    labels = [handle.get_label() for handle in handles if not handle.get_label().startswith("_")]
    if handles:
        ax_bpb.legend(handles, labels, loc="best", frameon=True)
    fig.savefig(PLOT_BPB_ONLY_PATH, bbox_inches="tight")
    plt.close(fig)


def _phase_movement_frame(frame: pd.DataFrame) -> pd.DataFrame:
    movements: list[dict[str, float | int | None]] = []
    previous_weights: np.ndarray | None = None
    for row in frame.sort_values("subset_size").itertuples(index=False):
        weights = _weights_from_dict(row.phase_weights, list(row.phase_weights["phase_0"].keys()))
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


def _regmix_validated_frame() -> pd.DataFrame:
    frame = pd.read_csv(
        REGMIX_CURVE_POINTS_CSV,
        usecols=["subset_size", "actual_validated_bpb"],
    )
    return frame.loc[frame["actual_validated_bpb"].notna()].sort_values("subset_size").reset_index(drop=True)


def _regmix_curve_frame() -> pd.DataFrame:
    return (
        pd.read_csv(
            REGMIX_CURVE_POINTS_CSV,
            usecols=["subset_size", "predicted_optimum_value", "actual_validated_bpb"],
        )
        .sort_values("subset_size")
        .reset_index(drop=True)
    )


def _plot_bpb_and_tv(frame: pd.DataFrame) -> None:
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
    _plot_bpb_panel(ax_bpb, frame, cmap=cmap)
    ax_move.plot(
        frame["subset_size"],
        frame["optimum_move_mean_phase_tv_vs_prev"],
        color=cmap(0.36),
        marker="D",
        linewidth=2.2,
        label="Raw-optimum movement (mean phase TV)",
    )
    ax_move.axhline(0.0, color="0.55", linewidth=1.0, linestyle=":")

    ax_bpb.set_title("Two-phase many-domain: GRP convergence (power-family penalty raw optimum)")
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


def _plot_phase_movements(frame: pd.DataFrame) -> None:
    cmap = plt.colormaps["RdYlGn_r"]
    movement_frame = _phase_movement_frame(frame)
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10.2, 4.2),
        dpi=180,
        constrained_layout=True,
    )
    ax.plot(
        movement_frame["subset_size"],
        movement_frame["phase0_tv_vs_prev"],
        color=cmap(0.24),
        marker="o",
        linewidth=2.2,
        label=r"Phase 0 TV movement: $\frac{1}{2}\|w_0^{(k)} - w_0^{(k^-)}\|_1$",
    )
    ax.plot(
        movement_frame["subset_size"],
        movement_frame["phase1_tv_vs_prev"],
        color=cmap(0.76),
        marker="D",
        linewidth=2.2,
        label=r"Phase 1 TV movement: $\frac{1}{2}\|w_1^{(k)} - w_1^{(k^-)}\|_1$",
    )
    ax.axhline(0.0, color="0.55", linewidth=1.0, linestyle=":")
    ax.set_title("Two-phase many-domain: phase-wise raw-optimum movement")
    ax.set_ylabel("Phase TV")
    ax.set_xlabel("Observed runs used for fitting")
    ax.set_xticks(movement_frame["subset_size"].tolist())
    ax.set_xlim(int(movement_frame["subset_size"].min()), int(movement_frame["subset_size"].max()))
    ax.grid(True, alpha=0.25)
    handles = ax.get_lines()
    labels = [handle.get_label() for handle in handles if not handle.get_label().startswith("_")]
    if handles:
        ax.legend(handles, labels, loc="best", frameon=True)
    fig.savefig(PLOT_PHASE_MOVEMENT_PATH, bbox_inches="tight")
    plt.close(fig)


def _plot_bpb_and_phase_movements(frame: pd.DataFrame) -> None:
    cmap = plt.colormaps["RdYlGn_r"]
    movement_frame = _phase_movement_frame(frame)
    baseline_bpbs = _baseline_bpbs()
    fig, (ax_bpb, ax_move) = plt.subplots(
        2,
        1,
        figsize=(10.2, 7.5),
        dpi=180,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.45, 1.15], "hspace": 0.08},
    )
    _plot_bpb_panel(
        ax_bpb,
        frame,
        cmap=cmap,
        baseline_bpbs=baseline_bpbs,
        baseline_line_specs=PROPORTIONAL_BPB_LINE_SPEC,
    )
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

    ax_bpb.set_title("Two-phase many-domain: GRP convergence (power-family penalty raw optimum)")
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

    proportional_bpb = baseline_bpbs["baseline_proportional"]
    ax_bpb.annotate(
        f"Proportional: {_format_bpb_label(proportional_bpb)}",
        (int(frame["subset_size"].max()), proportional_bpb),
        textcoords="offset points",
        xytext=(-6, 8),
        ha="right",
        fontsize=8,
        color="#4C78A8",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
    )

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


def _plot_bpb_and_phase_movements_with_regmix(frame: pd.DataFrame) -> None:
    cmap = plt.colormaps["RdYlGn_r"]
    movement_frame = _phase_movement_frame(frame)
    baseline_bpbs = _baseline_bpbs()
    regmix_validated = _regmix_validated_frame()
    plot_frame = frame.copy()
    off_chart_grp = plot_frame.loc[plot_frame["actual_validated_bpb"] > BPB_PANEL_TOP_WITH_REGMIX].copy()
    if not off_chart_grp.empty:
        plot_frame.loc[off_chart_grp.index, "actual_validated_bpb"] = np.nan
    fig, (ax_bpb, ax_move) = plt.subplots(
        2,
        1,
        figsize=(10.2, 7.5),
        dpi=180,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.45, 1.15], "hspace": 0.08},
    )
    _plot_bpb_panel(
        ax_bpb,
        plot_frame,
        cmap=cmap,
        baseline_bpbs=baseline_bpbs,
        baseline_line_specs=PROPORTIONAL_BPB_LINE_SPEC,
        regmix_validated=regmix_validated,
    )
    ax_bpb.set_ylim(top=BPB_PANEL_TOP_WITH_REGMIX)
    if not off_chart_grp.empty:
        first_row = off_chart_grp.sort_values("subset_size").iloc[0]
        ax_bpb.annotate(
            f"{_format_bpb_label(float(first_row['actual_validated_bpb']))} (off chart)",
            (float(first_row["subset_size"]), BPB_PANEL_TOP_WITH_REGMIX),
            textcoords="offset points",
            xytext=(0, 16),
            ha="center",
            va="bottom",
            fontsize=8,
            color=cmap(0.88),
            clip_on=False,
            arrowprops={
                "arrowstyle": "-|>",
                "color": cmap(0.88),
                "lw": 1.2,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )
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

    ax_bpb.set_title("Two-phase many-domain: GRP convergence (power-family penalty raw optimum)")
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

    proportional_bpb = baseline_bpbs["baseline_proportional"]
    ax_bpb.annotate(
        f"Proportional: {_format_bpb_label(proportional_bpb)}",
        (int(frame["subset_size"].max()), proportional_bpb),
        textcoords="offset points",
        xytext=(-6, 8),
        ha="right",
        fontsize=8,
        color="#4C78A8",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
    )

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

    fig.savefig(PLOT_BPB_PHASE_MOVEMENT_WITH_REGMIX_PATH, bbox_inches="tight")
    plt.close(fig)


def _plot_bpb_with_regmix(frame: pd.DataFrame) -> None:
    cmap = plt.colormaps["RdYlGn_r"]
    baseline_bpbs = _baseline_bpbs()
    regmix_frame = _regmix_curve_frame()
    regmix_validated = regmix_frame.loc[regmix_frame["actual_validated_bpb"].notna()].copy()
    plot_frame = frame.copy()
    off_chart_grp = plot_frame.loc[plot_frame["actual_validated_bpb"] > BPB_PANEL_TOP_WITH_REGMIX_PREDICTED].copy()
    if not off_chart_grp.empty:
        plot_frame.loc[off_chart_grp.index, "actual_validated_bpb"] = np.nan

    fig, ax_bpb = plt.subplots(
        1,
        1,
        figsize=(10.2, 4.7),
        dpi=180,
        constrained_layout=True,
    )
    _plot_bpb_panel(
        ax_bpb,
        plot_frame,
        cmap=cmap,
        baseline_bpbs=baseline_bpbs,
        baseline_line_specs=PROPORTIONAL_BPB_LINE_SPEC,
        regmix_validated=regmix_validated,
    )
    ax_bpb.plot(
        regmix_frame["subset_size"],
        regmix_frame["predicted_optimum_value"],
        color=REGMIX_PREDICTED_COLOR,
        marker="^",
        linewidth=2.0,
        linestyle=":",
        label="RegMix predicted BPB",
    )
    ax_bpb.set_ylim(top=BPB_PANEL_TOP_WITH_REGMIX_PREDICTED)
    if not off_chart_grp.empty:
        first_row = off_chart_grp.sort_values("subset_size").iloc[0]
        ax_bpb.annotate(
            f"{_format_bpb_label(float(first_row['actual_validated_bpb']))} (off chart)",
            (float(first_row["subset_size"]), BPB_PANEL_TOP_WITH_REGMIX_PREDICTED),
            textcoords="offset points",
            xytext=(0, 16),
            ha="center",
            va="bottom",
            fontsize=8,
            color=cmap(0.88),
            clip_on=False,
            arrowprops={
                "arrowstyle": "-|>",
                "color": cmap(0.88),
                "lw": 1.2,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )

    ax_bpb.set_title("Two-phase many-domain: GRP convergence (power-family penalty raw optimum)")
    ax_bpb.set_ylabel("Predicted BPB")
    ax_bpb.set_xlabel("Observed runs used for fitting")
    ax_bpb.set_xticks(frame["subset_size"].tolist())
    ax_bpb.set_xlim(int(frame["subset_size"].min()), int(frame["subset_size"].max()))
    ax_bpb.grid(True, alpha=0.25)

    proportional_bpb = baseline_bpbs["baseline_proportional"]
    ax_bpb.annotate(
        f"Proportional: {_format_bpb_label(proportional_bpb)}",
        (int(frame["subset_size"].max()), proportional_bpb),
        textcoords="offset points",
        xytext=(-6, 8),
        ha="right",
        fontsize=8,
        color="#4C78A8",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
    )

    bpb_handles = ax_bpb.get_lines()
    bpb_labels = [handle.get_label() for handle in bpb_handles if not handle.get_label().startswith("_")]
    if bpb_handles:
        ax_bpb.legend(bpb_handles, bpb_labels, loc="upper right", frameon=True, ncol=2)

    fig.savefig(PLOT_BPB_ONLY_WITH_REGMIX_PATH, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-validated-only",
        action="store_true",
        help="Reuse existing subset-fit curve points and only refresh realized validated BPBs from checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    best_observed_bpb = float(np.min(packet.base.y))
    if args.refresh_validated_only:
        frame = _load_existing_curve_points()
    else:
        frame = _curve_points()
    frame = _apply_realized_validated_bpbs(
        frame,
        full_subset_size=len(packet.base.y),
        best_observed_bpb=best_observed_bpb,
    )
    curve_for_csv = frame.copy()
    curve_for_csv["phase_weights"] = curve_for_csv["phase_weights"].map(json.dumps)
    curve_for_csv.to_csv(CURVE_POINTS_CSV, index=False)
    _plot(frame)
    _plot_bpb_only(frame)
    _plot_bpb_and_tv(frame)
    _plot_phase_movements(frame)
    _plot_bpb_and_phase_movements(frame)
    _plot_bpb_and_phase_movements_with_regmix(frame)
    _plot_bpb_with_regmix(frame)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "objective_metric": OBJECTIVE_METRIC,
                "variant": VARIANT_NAME,
                "curve_points_csv": str(CURVE_POINTS_CSV),
                "plot": str(PLOT_PATH),
                "plot_bpb_only": str(PLOT_BPB_ONLY_PATH),
                "plot_bpb_and_tv": str(PLOT_BPB_TV_PATH),
                "plot_phase_movements": str(PLOT_PHASE_MOVEMENT_PATH),
                "plot_bpb_and_phase_movements": str(PLOT_BPB_PHASE_MOVEMENT_PATH),
                "plot_bpb_and_phase_movements_with_regmix": str(PLOT_BPB_PHASE_MOVEMENT_WITH_REGMIX_PATH),
                "plot_bpb_with_regmix": str(PLOT_BPB_ONLY_WITH_REGMIX_PATH),
                "best_observed_bpb": best_observed_bpb,
                "rows": frame.replace({np.nan: None}).to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
