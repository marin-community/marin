# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "plotly",
#     "scipy",
#     "scikit-learn",
# ]
# ///
"""Fit DSP to the finalized all-22 DCLM Core smooth target.

The post-gapfill DCLM matrix contains hard DCLM Core scores for every component
and at least one smooth scalar for every component. This script constructs a
single higher-is-better smooth utility per DCLM task, z-scores each task, forms
an all-22 macro target, and compares it against the hard centered-accuracy
macro. It then fits DSP to both targets and optimizes the smooth target.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr

from experiments.domain_phase_mix.exploratory.two_phase_many import dclm_matrix_guard
from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dclm_core_dsp_300m import (
    METADATA_CSV,
    RAW_MATRIX_CSV,
    TARGET_COLUMN,
    average_phase_tv,
    phase_stats,
    prediction_metrics,
    weights_with_model_params,
    write_weights_plot,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp

SCRIPT_DIR = Path(__file__).resolve().parent
REGISTRY_DCLM_MATRIX_CSV = (
    SCRIPT_DIR
    / "metric_registry"
    / "300m_dclm_core_completion"
    / "300m_dclm_core_eval_results_full_after_retry8_bigbench_rescored_repeatcopy128.csv"
)
TMP_DCLM_MATRIX_CSV = Path(
    "/tmp/dclm_gapfill_audit/300m_dclm_core_eval_results_full_after_retry8_bigbench_rescored_repeatcopy128.csv"
)
DEFAULT_DCLM_MATRIX_CSV = REGISTRY_DCLM_MATRIX_CSV if REGISTRY_DCLM_MATRIX_CSV.exists() else TMP_DCLM_MATRIX_CSV
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dclm_all22_smooth_dsp_300m_20260614_repeatcopy128"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

DCLM_ALIASES = [
    "agieval_lsat_ar_3shot",
    "arc_easy_10shot",
    "arc_challenge_10shot",
    "bb_qa_wikidata_10shot",
    "bb_dyck_languages_10shot",
    "bb_operators_10shot",
    "bb_repeat_copy_logic_10shot",
    "bb_cs_algorithms_10shot",
    "bb_language_identification_10shot",
    "boolq_10shot",
    "commonsense_qa_10shot",
    "copa_0shot",
    "coqa_0shot",
    "hellaswag_0shot",
    "hellaswag_10shot",
    "jeopardy_10shot",
    "lambada_0shot",
    "openbookqa_0shot",
    "piqa_10shot",
    "squad_10shot",
    "winograd_0shot",
    "winogrande_0shot",
]


@dataclass(frozen=True)
class SmoothComponent:
    """One smooth scalar used for a DCLM Core component."""

    alias: str
    column: str
    metric_kind: str
    utility_transform: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-matrix-csv", type=Path, default=RAW_MATRIX_CSV)
    parser.add_argument("--dclm-matrix-csv", type=Path, default=DEFAULT_DCLM_MATRIX_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--variant", choices=sorted(dsp.VARIANTS), default="effective_exposure")
    parser.add_argument("--component-variant", choices=sorted(dsp.VARIANTS), default="no_penalty")
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--component-maxiter", type=int, default=60)
    parser.add_argument("--coarse-top-k", type=int, default=4)
    parser.add_argument("--component-coarse-top-k", type=int, default=2)
    parser.add_argument("--basin-hopping-iters", type=int, default=0)
    parser.add_argument("--optimum-starts", type=int, default=240)
    parser.add_argument("--max-observed-starts", type=int, default=241)
    return parser.parse_args()


def finite_zscore(values: pd.Series) -> pd.Series:
    """Return z-scored values, preserving missing values."""
    numeric = pd.to_numeric(values, errors="coerce")
    std = float(numeric.std(ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        return numeric * np.nan
    return (numeric - float(numeric.mean())) / std


def safe_corr(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray, method: str) -> float:
    """Return correlation, or NaN if the paired sample is degenerate."""
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    mask = np.isfinite(left_values) & np.isfinite(right_values)
    if int(mask.sum()) < 3:
        return float("nan")
    left_values = left_values[mask]
    right_values = right_values[mask]
    if float(np.std(left_values)) <= 0.0 or float(np.std(right_values)) <= 0.0:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(left_values, right_values).statistic)
    if method == "pearson":
        return float(pearsonr(left_values, right_values).statistic)
    raise ValueError(f"Unknown correlation method {method!r}")


def selected_smooth_component(frame: pd.DataFrame, alias: str) -> SmoothComponent:
    """Select the preferred smooth scalar for one DCLM component."""
    columns = set(frame.columns)
    candidates = [
        (f"lm_eval/{alias}/bpb", "bpb", "negate"),
        (f"lm_eval/{alias}/native_gold_bpb", "native_gold_bpb", "negate"),
        (f"lm_eval/{alias}/native_gold_logprob_per_byte", "native_gold_logprob_per_byte", "identity"),
        (f"lm_eval/{alias}/choice_logprob_norm", "choice_logprob_norm", "identity"),
        (f"lm_eval/{alias}/choice_logprob", "choice_logprob", "identity"),
        (f"lm_eval/{alias}/native_gold_logprob", "native_gold_logprob", "identity"),
        (f"lm_eval/{alias}/perplexity", "perplexity", "neg_log"),
    ]
    proportional = frame.loc[frame["run_name"].eq("baseline_proportional")]
    if not proportional.empty:
        for column, metric_kind, utility_transform in candidates:
            if column in columns and pd.notna(proportional[column].iloc[0]):
                return SmoothComponent(alias, column, metric_kind, utility_transform)
    for column, metric_kind, utility_transform in candidates:
        if column in columns:
            return SmoothComponent(alias, column, metric_kind, utility_transform)
    raise ValueError(f"No smooth scalar found for DCLM alias {alias!r}")


def component_utility(frame: pd.DataFrame, component: SmoothComponent) -> pd.Series:
    """Convert one smooth metric to higher-is-better utility."""
    values = pd.to_numeric(frame[component.column], errors="coerce")
    if component.utility_transform == "identity":
        return values
    if component.utility_transform == "negate":
        return -values
    if component.utility_transform == "neg_log":
        return -np.log(np.clip(values, 1e-12, None))
    raise ValueError(f"Unknown utility transform {component.utility_transform!r}")


def load_joined_frame(raw_matrix_csv: Path, dclm_matrix_csv: Path) -> pd.DataFrame:
    """Join DCLM result columns to the canonical 300M raw metric matrix."""
    raw = pd.read_csv(raw_matrix_csv, low_memory=False)
    dclm = pd.read_csv(dclm_matrix_csv, low_memory=False)
    dclm_matrix_guard.validate_corrected_dclm_matrix(dclm, dclm_matrix_csv)
    metadata_columns = [
        column
        for column in (
            "run_name",
            "registry_run_key",
            "run_id",
            "row_kind",
            "status",
            "scale",
            "cohort",
            "source_experiment",
            "checkpoint_root",
            "is_qsplit240_core",
            "is_baseline_olmix",
            "is_baseline_stratified",
            "is_fit_swarm_60m_default",
        )
        if column in raw.columns
    ]
    weight_columns = [column for column in raw.columns if column.startswith(("phase_0_", "phase_1_"))]
    raw_payload = raw[metadata_columns + weight_columns].copy()
    joined = raw_payload.merge(dclm, on="run_name", how="inner", validate="one_to_one")
    if joined.empty:
        raise ValueError("No rows after joining raw matrix and DCLM matrix on run_name")
    return joined


def add_smooth_targets(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add selected smooth utilities, z-scored components, and the all-22 macro."""
    out = frame.copy()
    component_rows: list[dict[str, Any]] = []
    z_columns: list[str] = []
    for alias in DCLM_ALIASES:
        component = selected_smooth_component(out, alias)
        utility_column = f"dclm_smooth/{alias}/utility"
        z_column = f"dclm_smooth/{alias}/z_utility"
        hard_column = f"lm_eval/dclm_core/{alias}/centered_accuracy"
        out[utility_column] = component_utility(out, component)
        out[z_column] = finite_zscore(out[utility_column])
        z_columns.append(z_column)
        component_rows.append(
            {
                "alias": alias,
                "smooth_column": component.column,
                "metric_kind": component.metric_kind,
                "utility_transform": component.utility_transform,
                "utility_nonnull_count": int(out[utility_column].notna().sum()),
                "utility_std": float(out[utility_column].std(ddof=1)),
                "proportional_available": bool(
                    out.loc[out["run_name"].eq("baseline_proportional"), utility_column].notna().any()
                ),
                "hard_column": hard_column,
                "hard_nonnull_count": int(out[hard_column].notna().sum()) if hard_column in out.columns else 0,
                "utility_vs_hard_spearman": (
                    safe_corr(out[utility_column], out[hard_column], "spearman") if hard_column in out.columns else np.nan
                ),
                "utility_vs_hard_pearson": (
                    safe_corr(out[utility_column], out[hard_column], "pearson") if hard_column in out.columns else np.nan
                ),
            }
        )
    prop_anchor_columns = [
        f"dclm_smooth/{row['alias']}/z_utility" for row in component_rows if bool(row["proportional_available"])
    ]
    out["dclm_smooth/all22_complete_zscore_macro"] = out[z_columns].mean(axis=1, skipna=False)
    out["dclm_smooth/available_zscore_macro"] = out[z_columns].mean(axis=1, skipna=True)
    out["dclm_smooth/proportional_anchor_zscore_macro"] = out[prop_anchor_columns].mean(axis=1, skipna=False)
    out["dclm_smooth/all22_available_count"] = out[z_columns].notna().sum(axis=1)
    out["dclm_smooth/proportional_anchor_component_count"] = out[prop_anchor_columns].notna().sum(axis=1)
    return out, pd.DataFrame.from_records(component_rows)


def fit_frame_for_target(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Build a DSP fit frame for one higher-is-better target."""
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column {target_column!r}")
    mask = frame["row_kind"].eq("signal") & frame["status"].eq("completed") & frame[target_column].notna()
    fit_frame = frame.loc[mask].copy()
    fit_frame["objective_metric"] = -pd.to_numeric(fit_frame[target_column], errors="raise")
    if fit_frame.empty:
        raise ValueError(f"No rows available for target {target_column!r}")
    return fit_frame.reset_index(drop=True)


def predict_frame(packet: dsp.PacketData, model: dsp.FittedDSPModel, target_name: str) -> pd.DataFrame:
    """Return observed-row train and out-of-fold predictions in higher-is-better units."""
    actual = -packet.y
    train = -dsp.predict(model, packet.w)
    oof = -dsp.oof_predictions(packet, model)
    out = packet.frame[["run_name", "registry_run_key"]].copy()
    out["target_name"] = target_name
    out["actual"] = actual
    out["train_pred"] = train
    out["oof_pred"] = oof
    out["train_residual_pred_minus_actual"] = train - actual
    out["oof_residual_pred_minus_actual"] = oof - actual
    out["actual_rank_desc"] = out["actual"].rank(method="min", ascending=False)
    out["oof_rank_desc"] = out["oof_pred"].rank(method="min", ascending=False)
    return out.sort_values("oof_pred", ascending=False).reset_index(drop=True)


def fit_target(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    target_column: str,
    variant: dsp.DSPVariant,
    *,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
    optimize_raw: bool,
    optimum_starts: int,
    max_observed_starts: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Fit DSP to one target and optionally optimize the raw surrogate."""
    fit_frame = fit_frame_for_target(frame, target_column)
    packet = dsp.packet_from_frame(fit_frame, metadata)
    model, tuning = dsp.fit_variant(
        packet,
        variant,
        maxiter=maxiter,
        coarse_top_k=coarse_top_k,
        basin_hopping_iters=basin_hopping_iters,
    )
    actual = -packet.y
    train = -dsp.predict(model, packet.w)
    oof = -dsp.oof_predictions(packet, model)
    prop_mask = packet.frame["run_name"].eq("baseline_proportional").to_numpy()
    prop_idx = int(np.flatnonzero(prop_mask)[0]) if prop_mask.any() else -1
    if prop_idx >= 0:
        proportional_weights = packet.w[prop_idx]
        proportional_actual = float(actual[prop_idx])
    else:
        prop_frame = frame.loc[frame["run_name"].eq("baseline_proportional")].copy()
        if prop_frame.empty:
            raise ValueError("Joined frame does not contain baseline_proportional")
        prop_frame["objective_metric"] = 0.0
        proportional_weights = dsp.packet_from_frame(prop_frame, metadata).w[0]
        proportional_actual = float("nan")
    proportional_pred = float(-dsp.predict(model, proportional_weights[None, :, :])[0])
    best_idx = int(np.argmax(actual))
    summary: dict[str, Any] = {
        "target_name": target_column,
        "variant": variant.name,
        "fit_row_count": int(len(packet.y)),
        "total_param_count": int(model.total_param_count),
        "m_dependent_params_per_domain": int(model.m_dependent_params_per_domain),
        "target_mean": float(np.mean(actual)),
        "target_std": float(np.std(actual, ddof=1)),
        "target_range": float(np.max(actual) - np.min(actual)),
        "best_observed_run_name": str(packet.frame.iloc[best_idx]["run_name"]),
        "best_observed_score": float(actual[best_idx]),
        "proportional_in_fit": bool(prop_idx >= 0),
        "proportional_actual_score": proportional_actual,
        "proportional_pred_score": proportional_pred,
        "best_minus_proportional": float(actual[best_idx] - proportional_actual),
        "best_minus_proportional_pred": float(actual[best_idx] - proportional_pred),
        "fitted_gamma": float(model.params["gamma"]) if "gamma" in model.params else np.nan,
        **prediction_metrics(actual, train, "train"),
        **prediction_metrics(actual, oof, "oof"),
    }
    predictions = predict_frame(packet, model, target_column)
    weights = pd.DataFrame()
    if optimize_raw:
        raw_result, raw_weights = dsp.optimize_raw(
            model,
            num_starts=optimum_starts,
            observed_start_weights=packet.w,
            max_observed_starts=max_observed_starts,
        )
        raw_distances = dsp.average_phase_tv_distance(packet.w, raw_weights[None, :, :])
        nearest_idx = int(np.argmin(raw_distances))
        summary.update(
            {
                "raw_optimization_success": bool(raw_result.success),
                "raw_optimization_message": str(raw_result.message),
                "raw_predicted_score": float(-raw_result.fun),
                "raw_predicted_delta_vs_proportional_actual": float(-raw_result.fun - proportional_actual),
                "raw_predicted_delta_vs_proportional_pred": float(-raw_result.fun - proportional_pred),
                "raw_nearest_observed_run_name": str(packet.frame.iloc[nearest_idx]["run_name"]),
                "raw_nearest_observed_score": float(actual[nearest_idx]),
                "raw_nearest_observed_tv": float(raw_distances[nearest_idx]),
                "raw_tv_to_proportional": average_phase_tv(raw_weights, proportional_weights),
                **phase_stats(raw_weights, prefix="raw"),
            }
        )
        weights = weights_with_model_params(model, raw_weights, proportional_weights)
        weights.insert(0, "target_name", target_column)
    tuning.insert(0, "target_name", target_column)
    return summary, predictions, pd.concat([weights], ignore_index=True) if not weights.empty else weights


def add_hard_coupling(
    summary: dict[str, Any],
    frame: pd.DataFrame,
    target_column: str,
) -> None:
    """Annotate a fit summary with hard DCLM macro coupling diagnostics."""
    hard = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    target = pd.to_numeric(frame[target_column], errors="coerce")
    summary["hard_macro_spearman"] = safe_corr(target, hard, "spearman")
    summary["hard_macro_pearson"] = safe_corr(target, hard, "pearson")
    prop_rows = frame.loc[frame["run_name"].eq("baseline_proportional"), TARGET_COLUMN]
    prop_hard = float(prop_rows.iloc[0]) if len(prop_rows) else np.nan
    for prefix, run_key in (
        ("best_observed", summary.get("best_observed_run_name")),
        ("raw_nearest_observed", summary.get("raw_nearest_observed_run_name")),
    ):
        rows = frame.loc[frame["run_name"].eq(str(run_key)), TARGET_COLUMN] if run_key else pd.Series(dtype=float)
        value = float(rows.iloc[0]) if len(rows) else np.nan
        summary[f"{prefix}_hard_dclm_macro"] = value
        summary[f"{prefix}_hard_minus_proportional"] = value - prop_hard
    summary["proportional_hard_dclm_macro"] = prop_hard


def hard_component_variance_audit(frame: pd.DataFrame, component_map: pd.DataFrame) -> pd.DataFrame:
    """Summarize hard DCLM component variance for interpreting correlations."""
    rows = []
    for alias in component_map["alias"].tolist():
        hard_column = f"lm_eval/dclm_core/{alias}/centered_accuracy"
        values = pd.to_numeric(frame[hard_column], errors="coerce")
        rows.append(
            {
                "alias": alias,
                "hard_count": int(values.notna().sum()),
                "hard_unique": int(values.nunique(dropna=True)),
                "hard_min": float(values.min()) if values.notna().any() else np.nan,
                "hard_max": float(values.max()) if values.notna().any() else np.nan,
                "hard_std": float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(["hard_unique", "hard_std"]).reset_index(drop=True)


def fit_component_targets(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    component_map: pd.DataFrame,
    variant: dsp.DSPVariant,
    *,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit no-raw component DSP diagnostics."""
    rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for index, alias in enumerate(component_map["alias"].tolist(), start=1):
        print(f"[component {index}/{len(component_map)}] fitting {alias}", flush=True)
        target_column = f"dclm_smooth/{alias}/z_utility"
        row, predictions, _weights = fit_target(
            frame,
            metadata,
            target_column,
            variant,
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
            basin_hopping_iters=basin_hopping_iters,
            optimize_raw=False,
            optimum_starts=0,
            max_observed_starts=0,
        )
        row["alias"] = alias
        row["smooth_column"] = str(component_map.loc[component_map["alias"].eq(alias), "smooth_column"].iloc[0])
        rows.append(row)
        predictions["alias"] = alias
        prediction_frames.append(predictions)
    return pd.DataFrame.from_records(rows), pd.concat(prediction_frames, ignore_index=True)


def write_fit_plot(fit_summary: pd.DataFrame, predictions: pd.DataFrame, output_path: Path) -> None:
    """Write aggregate fit diagnostics."""
    targets = fit_summary["target_name"].tolist()
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("aggregate OOF fit", "prediction vs actual", "residual vs rank"),
        horizontal_spacing=0.12,
        column_widths=[0.25, 0.38, 0.37],
    )
    fit_order = fit_summary.sort_values("oof_spearman", ascending=True)
    fig.add_trace(
        go.Bar(
            x=fit_order["oof_spearman"],
            y=fit_order["target_name"],
            orientation="h",
            name="OOF Spearman",
            marker={"color": fit_order["oof_spearman"], "colorscale": "RdYlGn_r", "cmin": -1.0, "cmax": 1.0},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=fit_order["oof_r2"], y=fit_order["target_name"], orientation="h", name="OOF R2"),
        row=1,
        col=1,
    )
    for target in targets:
        subset = predictions.loc[predictions["target_name"].eq(target)]
        fig.add_trace(
            go.Scatter(
                x=subset["actual"],
                y=subset["oof_pred"],
                mode="markers",
                name=target,
                text=subset["run_name"],
                marker={"size": 8},
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=subset["actual_rank_desc"],
                y=subset["oof_residual_pred_minus_actual"],
                mode="markers",
                name=target,
                text=subset["run_name"],
                showlegend=False,
                marker={"size": 8},
            ),
            row=1,
            col=3,
        )
    fig.add_vline(x=0.0, line_dash="dash", line_color="black", row=1, col=1)
    fig.add_hline(y=0.0, line_dash="dash", line_color="black", row=1, col=3)
    fig.update_xaxes(title_text="OOF metric", row=1, col=1)
    fig.update_xaxes(title_text="actual target", row=1, col=2)
    fig.update_yaxes(title_text="OOF predicted target", row=1, col=2)
    fig.update_xaxes(title_text="actual rank, 1 is best", autorange="reversed", row=1, col=3)
    fig.update_yaxes(title_text="OOF prediction - actual", row=1, col=3)
    fig.update_layout(
        template="plotly_white",
        width=1650,
        height=640,
        title={"text": "DCLM all-22 smooth and hard macro DSP fits", "x": 0.5},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.18},
    )
    fig.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_component_plot(component_map: pd.DataFrame, component_fit: pd.DataFrame, output_path: Path) -> None:
    """Write component smooth-hard coupling and DSP predictability diagnostics."""
    merged = component_map.merge(
        component_fit[["alias", "oof_spearman", "oof_r2", "fit_row_count"]],
        on="alias",
        how="left",
        validate="one_to_one",
    )
    order = merged.sort_values("utility_vs_hard_spearman", ascending=True)
    labels = order["alias"] + " (" + order["metric_kind"] + ")"
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("smooth utility vs hard component", "smooth utility DSP fit"),
        horizontal_spacing=0.18,
    )
    fig.add_trace(
        go.Bar(
            x=order["utility_vs_hard_spearman"],
            y=labels,
            orientation="h",
            marker={
                "color": order["utility_vs_hard_spearman"],
                "colorscale": "RdYlGn_r",
                "cmin": -1.0,
                "cmax": 1.0,
            },
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    order_fit = merged.sort_values("oof_spearman", ascending=True)
    fig.add_trace(
        go.Bar(
            x=order_fit["oof_spearman"],
            y=order_fit["alias"] + " (" + order_fit["metric_kind"] + ")",
            orientation="h",
            marker={"color": order_fit["oof_spearman"], "colorscale": "RdYlGn_r", "cmin": -1.0, "cmax": 1.0},
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    for col in (1, 2):
        fig.add_vline(x=0.0, line_dash="dash", line_color="black", row=1, col=col)
    fig.update_xaxes(title_text="Spearman", row=1, col=1)
    fig.update_xaxes(title_text="OOF Spearman", row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        width=1600,
        height=max(780, 30 * len(merged)),
        margin={"l": 360, "r": 40, "t": 85, "b": 90},
        title={"text": "DCLM component smooth-scalar diagnostics", "x": 0.5},
    )
    fig.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_tradeoff_plot(frame: pd.DataFrame, output_path: Path) -> None:
    """Write smooth macro vs hard macro scatter."""
    smooth_column = "dclm_smooth/proportional_anchor_zscore_macro"
    fig = go.Figure()
    plot_frame = frame.loc[
        frame["row_kind"].eq("signal")
        & frame["status"].eq("completed")
        & frame[smooth_column].notna()
        & frame[TARGET_COLUMN].notna()
    ].copy()
    special_names = {"baseline_proportional", "baseline_unimax", "baseline_stratified"}
    fig.add_trace(
        go.Scatter(
            x=plot_frame[smooth_column],
            y=plot_frame[TARGET_COLUMN],
            mode="markers",
            text=plot_frame["run_name"],
            marker={
                "size": np.where(plot_frame["run_name"].isin(special_names), 14, 8),
                "color": plot_frame[TARGET_COLUMN],
                "colorscale": "RdYlGn_r",
                "showscale": True,
                "colorbar": {"title": "hard macro"},
            },
            name="observed",
        )
    )
    for name in sorted(special_names):
        subset = plot_frame.loc[plot_frame["run_name"].eq(name)]
        if subset.empty:
            continue
        fig.add_annotation(
            x=float(subset[smooth_column].iloc[0]),
            y=float(subset[TARGET_COLUMN].iloc[0]),
            text=name,
            showarrow=True,
            arrowhead=2,
            ax=25,
            ay=-25,
        )
    fig.update_layout(
        template="plotly_white",
        width=1050,
        height=720,
        title={"text": "DCLM all-22 smooth macro vs hard centered-accuracy macro", "x": 0.5},
        xaxis_title="all-22 smooth z-score macro, higher is better",
        yaxis_title="hard DCLM Core centered-accuracy macro, higher is better",
    )
    fig.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(
    output_dir: Path,
    fit_summary: pd.DataFrame,
    component_map: pd.DataFrame,
    component_fit: pd.DataFrame,
    hard_audit: pd.DataFrame,
    posthoc_component_map: pd.DataFrame,
) -> None:
    """Write a concise Markdown report."""
    lines = [
        "# DCLM Core v2 All-22 Smooth DSP Analysis",
        "",
        "This target uses one exact DCLM-task smooth scalar per component, z-scores each component, and averages all 22 z-scores. Higher is better.",
        "",
        "The DCLM hard-score matrix used here is the final repeat-copy-aware matrix, with BigBench generation rescores and the 128-token `bb_repeat_copy_logic_10shot` overlay applied.",
        "",
        "OOF fit metrics are optimistic diagnostics: the shared DSP helper refits only the linear head per fold while reusing nonlinear parameters tuned on the full panel.",
        "",
        "## Aggregate Fits",
        "",
        "| target | rows | OOF Spearman | OOF R2 | target-hard Spearman | proportional | best observed | best Δ vs prop | raw predicted | raw nearest observed | raw nearest TV | raw nearest hard Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for _, row in fit_summary.iterrows():
        raw_score = row.get("raw_predicted_score", np.nan)
        raw_run = row.get("raw_nearest_observed_run_name", "")
        raw_tv = row.get("raw_nearest_observed_tv", np.nan)
        raw_hard_delta = row.get("raw_nearest_observed_hard_minus_proportional", np.nan)
        lines.append(
            f"| `{row['target_name']}` | {int(row['fit_row_count'])} | `{row['oof_spearman']:.4f}` | `{row['oof_r2']:.4f}` | `{row['hard_macro_spearman']:.4f}` | `{row['proportional_actual_score']:.6f}` | `{row['best_observed_run_name']}` | `{row['best_minus_proportional']:.6f}` | `{raw_score:.6f}` | `{raw_run}` | `{raw_tv:.4f}` | `{raw_hard_delta:.6f}` |"
        )
    if not posthoc_component_map.empty:
        lines.extend(
            [
                "",
                "The `dclm_smooth/posthoc_hard_signal_positive_zscore_macro` row is diagnostic only: it was selected after looking at hard-component variance and smooth-hard correlation on this same 300M panel.",
                "Its OOF fit metrics are not directly comparable to the pre-registered all-component targets because dropping noisy, constant-hard, or negatively coupled components makes the subset mechanically easier to fit. Borderline positive-correlation inclusions may not replicate on a fresh panel.",
                "",
                "Posthoc included components:",
                "",
            ]
        )
        for alias in posthoc_component_map["alias"].tolist():
            lines.append(f"- `{alias}`.")
    lines.extend(["", "## Smooth Scalar Selection", ""])
    for _, row in component_map.sort_values("alias").iterrows():
        lines.append(
            f"- `{row['alias']}`: `{row['smooth_column']}` as `{row['metric_kind']}`, utility-hard Spearman `{row['utility_vs_hard_spearman']:.3f}`, non-null `{int(row['utility_nonnull_count'])}`."
        )
    weak = component_fit.sort_values("oof_spearman").head(6)
    lines.extend(["", "## Weakest Component DSP Fits", ""])
    for _, row in weak.iterrows():
        lines.append(f"- `{row['alias']}`: OOF Spearman `{row['oof_spearman']:.3f}`, OOF R2 `{row['oof_r2']:.3f}`.")
    constant_hard = hard_audit.loc[hard_audit["hard_unique"].le(1)]
    if not constant_hard.empty:
        lines.extend(["", "## Constant Hard Components", ""])
        for _, row in constant_hard.iterrows():
            note = ""
            if row["alias"] == "bb_repeat_copy_logic_10shot":
                note = " This is a known evaluation artifact pending a 128-token generation rerun, not evidence that the task is intrinsically impossible."
            lines.append(
                f"- `{row['alias']}` has `{int(row['hard_unique'])}` unique hard centered-accuracy value across `{int(row['hard_count'])}` rows.{note}"
            )
    lines.extend(
        [
            "",
            "## Interpretation Caveat",
            "",
            "The smooth all-22 macro is a denoising surrogate, not the official DCLM Core score. It is useful only if it both fits under DSP and couples positively to the hard macro. A raw DSP optimum is an extrapolative model prediction until validated by a trained mixture.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the all-22 smooth DSP analysis."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata_csv)
    frame = load_joined_frame(args.raw_matrix_csv, args.dclm_matrix_csv)
    frame, component_map = add_smooth_targets(frame)
    hard_audit = hard_component_variance_audit(frame, component_map)
    posthoc_component_map = component_map.merge(hard_audit[["alias", "hard_unique"]], on="alias", how="left")
    posthoc_component_map = posthoc_component_map.loc[
        posthoc_component_map["proportional_available"]
        & posthoc_component_map["hard_unique"].gt(1)
        & pd.to_numeric(posthoc_component_map["utility_vs_hard_spearman"], errors="coerce").gt(0.0)
    ].copy()
    posthoc_z_columns = [f"dclm_smooth/{alias}/z_utility" for alias in posthoc_component_map["alias"].tolist()]
    if len(posthoc_z_columns) >= 2:
        frame["dclm_smooth/posthoc_hard_signal_positive_zscore_macro"] = frame[posthoc_z_columns].mean(
            axis=1,
            skipna=False,
        )
    else:
        frame["dclm_smooth/posthoc_hard_signal_positive_zscore_macro"] = np.nan
    component_map.to_csv(args.output_dir / "smooth_component_map.csv", index=False)
    hard_audit.to_csv(args.output_dir / "hard_component_variance_audit.csv", index=False)
    posthoc_component_map.to_csv(args.output_dir / "posthoc_hard_signal_positive_component_map.csv", index=False)
    frame[
        [
            "run_name",
            "dclm_smooth/all22_available_count",
            "dclm_smooth/proportional_anchor_component_count",
            "dclm_smooth/all22_complete_zscore_macro",
            "dclm_smooth/available_zscore_macro",
            "dclm_smooth/proportional_anchor_zscore_macro",
            "dclm_smooth/posthoc_hard_signal_positive_zscore_macro",
            TARGET_COLUMN,
        ]
    ].to_csv(args.output_dir / "smooth_targets.csv", index=False)

    variant = dsp.VARIANTS[args.variant]
    aggregate_rows: list[dict[str, Any]] = []
    aggregate_predictions: list[pd.DataFrame] = []
    aggregate_weights: list[pd.DataFrame] = []
    aggregate_targets = [
        "dclm_smooth/proportional_anchor_zscore_macro",
        "dclm_smooth/all22_complete_zscore_macro",
        TARGET_COLUMN,
    ]
    if len(posthoc_component_map) >= 2:
        aggregate_targets.insert(2, "dclm_smooth/posthoc_hard_signal_positive_zscore_macro")
    for target_column in aggregate_targets:
        print(f"Fitting aggregate target {target_column}", flush=True)
        row, predictions, weights = fit_target(
            frame,
            metadata,
            target_column,
            variant,
            maxiter=args.maxiter,
            coarse_top_k=args.coarse_top_k,
            basin_hopping_iters=args.basin_hopping_iters,
            optimize_raw=target_column.startswith("dclm_smooth/"),
            optimum_starts=args.optimum_starts,
            max_observed_starts=args.max_observed_starts,
        )
        add_hard_coupling(row, frame, target_column)
        aggregate_rows.append(row)
        aggregate_predictions.append(predictions)
        if not weights.empty:
            aggregate_weights.append(weights)

    component_variant = dsp.VARIANTS[args.component_variant]
    component_fit, component_predictions = fit_component_targets(
        frame,
        metadata,
        component_map,
        component_variant,
        maxiter=args.component_maxiter,
        coarse_top_k=args.component_coarse_top_k,
        basin_hopping_iters=args.basin_hopping_iters,
    )

    fit_summary = pd.DataFrame.from_records(aggregate_rows)
    predictions = pd.concat(aggregate_predictions, ignore_index=True)
    raw_weights = pd.concat(aggregate_weights, ignore_index=True) if aggregate_weights else pd.DataFrame()
    fit_summary.to_csv(args.output_dir / "aggregate_fit_summary.csv", index=False)
    predictions.to_csv(args.output_dir / "aggregate_predictions_long.csv", index=False)
    raw_weights.to_csv(args.output_dir / "smooth_raw_optimum_weights.csv", index=False)
    component_fit.to_csv(args.output_dir / "component_fit_summary.csv", index=False)
    component_predictions.to_csv(args.output_dir / "component_predictions_long.csv", index=False)

    if not raw_weights.empty:
        write_weights_plot(raw_weights.drop(columns=["target_name"]), args.output_dir / "smooth_raw_optimum_weights.html")
    write_fit_plot(fit_summary, predictions, args.output_dir / "aggregate_fit_diagnostics.html")
    write_component_plot(component_map, component_fit, args.output_dir / "component_diagnostics.html")
    write_tradeoff_plot(frame, args.output_dir / "smooth_vs_hard_tradeoff.html")
    write_report(args.output_dir, fit_summary, component_map, component_fit, hard_audit, posthoc_component_map)

    summary = {
        "raw_matrix_csv": str(args.raw_matrix_csv),
        "dclm_matrix_csv": str(args.dclm_matrix_csv),
        "output_dir": str(args.output_dir),
        "aggregate_fits": fit_summary.to_dict(orient="records"),
        "component_count": int(len(component_map)),
        "complete_all22_rows": int(frame["dclm_smooth/all22_complete_zscore_macro"].notna().sum()),
        "proportional_anchor_component_count": int(component_map["proportional_available"].sum()),
        "constant_hard_component_count": int(hard_audit["hard_unique"].le(1).sum()),
        "posthoc_hard_signal_positive_component_count": int(len(posthoc_component_map)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
