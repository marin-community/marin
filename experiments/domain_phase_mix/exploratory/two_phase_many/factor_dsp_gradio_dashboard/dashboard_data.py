# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Data loading and recommendation logic for the Gradio factor-DSP dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.factor_dsp_constraint_dashboard_helpers import (
    PROPORTIONAL_RUN_NAME,
    add_materialized_epochs,
    candidate_weight_diagnostics,
    combine_phase_weights,
    epoch_scale_table_from_mixture_weights,
    filter_candidate_summary,
    label_phase_weights_long,
    load_csv,
    load_dashboard_candidate_cache,
    load_parquet,
    load_selected_candidate_weights,
    nearest_observed_tv,
    phase_weight_long_from_wide,
)

TWO_PHASE_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = TWO_PHASE_ROOT / "reference_outputs" / "collaborator_grug_v4_aggregate_repro_20260525"
CURRENT_AGG_DIR = REPRO_ROOT / "sent_raw_metric_matrix_300m_zip"
CURRENT_DSP_DIR = REPRO_ROOT / "canonical_dsp_sent_zip"
CANDIDATE_LIBRARY_DIR = TWO_PHASE_ROOT / "reference_outputs" / "factor_dsp_candidate_library_y_factor_20260526"
ENDPOINT_DISCOVERY_DIR = CANDIDATE_LIBRARY_DIR / "endpoint_discovery"
SENT_MATRIX_CSV = REPRO_ROOT / "sent_zip_input" / "raw_metric_matrix_300m" / "raw_metric_matrix_300m.csv"
NOISE_CSV = REPRO_ROOT / "sent_zip_input" / "raw_metric_matrix_300m" / "noise_baseline_run00097_300m.csv"
TARGET_COLUMN = "y_factor"
DEFAULT_SOURCES = (
    "observed_300m_signal",
    "canonical_dsp_mixture",
    "sobol_logit_trust",
    "canonical_dsp_path",
    "dsp_endpoint_path",
)
PROMISING_MANUAL_CANDIDATES = (
    PROPORTIONAL_RUN_NAME,
    "baseline_unimax",
    "baseline_olmix_loglinear_uncheatable_bpb",
    "dashboard_v4",
    "collaborator_lcb",
    "raw_dsp_optimum",
    "path_dashboard_v4_t1p0",
    "path_dashboard_v4_t0p75",
    "path_dashboard_v4_t0p5",
    "path_collaborator_lcb_t1p0",
    "path_collaborator_lcb_t0p75",
    "path_collaborator_lcb_t0p5",
    "path_raw_dsp_optimum_t0p5",
    "path_raw_dsp_optimum_t0p25",
    "path_endpoint_kl2p0_mw2p0_t1p0",
    "path_endpoint_kl2p0_mw2p0_t0p75",
    "path_endpoint_kl2p0_mw2p0_t0p5",
)


@dataclass(frozen=True)
class DashboardState:
    """Mutable dashboard state stored in Gradio state."""

    candidate: str
    constraints: dict[str, float]
    starting_candidate: str | None = None
    no_feasible: bool = False


def resolved_starting_candidate(state: DashboardState) -> str:
    """Return the user-selected starting candidate for unconstrained slider state."""
    return state.starting_candidate or state.candidate


def starting_candidate_dropdown_value(
    state: DashboardState,
    *,
    choices: list[str],
    fallback: str,
) -> str:
    """Return the dropdown value without coupling it to the recommended candidate."""
    starting_candidate = resolved_starting_candidate(state)
    return starting_candidate if starting_candidate in set(choices) else fallback


@dataclass(frozen=True)
class DashboardData:
    """Loaded data backing the Gradio dashboard."""

    candidate_summary: pd.DataFrame
    task_predictions: pd.DataFrame
    task_prediction_metrics: pd.DataFrame
    selected_tasks: list[str]
    eager_candidate_weights: pd.DataFrame
    epoch_scales: pd.DataFrame
    weight_cache_paths: tuple[Path, ...]


def _signal_frame() -> pd.DataFrame:
    raw_matrix = load_csv(SENT_MATRIX_CSV)
    if "row_kind" in raw_matrix.columns:
        return raw_matrix.loc[raw_matrix["row_kind"].eq("signal")].copy()
    return raw_matrix.copy()


def _candidate_summary(signal_frame: pd.DataFrame, selected_tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_scores = load_csv(CURRENT_AGG_DIR / "aggregate_scores.csv")
    mixture_weights = load_csv(CURRENT_DSP_DIR / "mixture_weights.csv")
    with open(CURRENT_DSP_DIR / "summary.json") as f:
        dsp_summary_json = json.load(f)

    observed_weights = phase_weight_long_from_wide(signal_frame, candidate_column="run_name")
    named_weights = label_phase_weights_long(mixture_weights)
    eager_candidate_weights = combine_phase_weights(observed_weights, named_weights)
    weight_diagnostics = candidate_weight_diagnostics(eager_candidate_weights)
    named_nearest = nearest_observed_tv(named_weights, observed_weights)

    observed_scores = aggregate_scores.loc[:, ["run_name", "run_id", TARGET_COLUMN]].copy()
    proportional_actual = float(
        observed_scores.loc[observed_scores["run_name"].eq(PROPORTIONAL_RUN_NAME), TARGET_COLUMN].iloc[0]
    )
    observed_summary = observed_scores.rename(columns={"run_name": "candidate", TARGET_COLUMN: "target_score"}).copy()
    observed_summary["source"] = "observed_300m_signal"
    observed_summary["score_kind"] = "actual"
    observed_summary["baseline_score"] = proportional_actual
    observed_summary["target_gain"] = observed_summary["target_score"] - proportional_actual
    observed_summary["nearest_observed_run"] = observed_summary["candidate"]
    observed_summary["nearest_observed_tv"] = 0.0
    observed_summary["has_task_deltas"] = True
    observed_summary["target_gain_lcb"] = observed_summary["target_gain"]

    pred_y_by_label = dsp_summary_json["pred_y_by_label"]
    proportional_pred = float(pred_y_by_label["proportional"])
    named_summary = pd.DataFrame(
        [
            {
                "candidate": label,
                "target_score": float(score),
                "source": "canonical_dsp_mixture",
                "score_kind": "dsp_prediction",
                "baseline_score": proportional_pred,
                "target_gain": float(score) - proportional_pred,
                "has_task_deltas": False,
            }
            for label, score in pred_y_by_label.items()
        ]
    )
    named_summary = named_summary.merge(named_nearest, on="candidate", how="left")
    named_summary.loc[named_summary["candidate"].eq("proportional"), "nearest_observed_run"] = PROPORTIONAL_RUN_NAME
    named_summary.loc[named_summary["candidate"].eq("proportional"), "nearest_observed_tv"] = 0.0

    cached_candidate_summary = load_dashboard_candidate_cache(
        candidate_summary_path=CANDIDATE_LIBRARY_DIR / "candidate_summary.parquet",
        endpoint_path_summary_path=ENDPOINT_DISCOVERY_DIR / "endpoint_path_summary.parquet",
    )
    eager_summary = pd.concat([observed_summary, named_summary], ignore_index=True, sort=False)
    eager_summary = eager_summary.merge(weight_diagnostics, on="candidate", how="left")
    eager_summary["deployability_flag"] = (
        eager_summary["target_gain"].ge(0.0)
        & eager_summary["nearest_observed_tv"].fillna(1.0).le(0.45)
        & eager_summary["max_phase_weight"].fillna(1.0).le(0.50)
    )
    summary = pd.concat([eager_summary, cached_candidate_summary], ignore_index=True, sort=False)
    summary["deployability_flag"] = summary["deployability_flag"].fillna(False).astype(bool)
    return summary, eager_candidate_weights


def load_dashboard_data() -> DashboardData:
    """Load all cached data required by the Gradio dashboard."""
    signal_frame = _signal_frame()
    selected_tasks_frame = load_csv(CURRENT_AGG_DIR / "selected_tasks.csv")
    candidate_summary, eager_candidate_weights = _candidate_summary(signal_frame, selected_tasks_frame)
    task_prediction_path = CANDIDATE_LIBRARY_DIR / "task_prediction_wide.parquet"
    task_metrics_path = CANDIDATE_LIBRARY_DIR / "task_prediction_metrics.csv"
    task_predictions = load_parquet(task_prediction_path)
    task_prediction_metrics = load_csv(task_metrics_path) if task_metrics_path.exists() else pd.DataFrame()
    predicted_candidates = set(task_predictions["candidate"].astype(str))
    candidate_summary["has_task_predictions"] = candidate_summary["candidate"].astype(str).isin(predicted_candidates)
    mixture_weights = load_csv(CURRENT_DSP_DIR / "mixture_weights.csv")
    epoch_scales = epoch_scale_table_from_mixture_weights(mixture_weights)
    return DashboardData(
        candidate_summary=candidate_summary,
        task_predictions=task_predictions,
        task_prediction_metrics=task_prediction_metrics,
        selected_tasks=selected_tasks_frame["task"].astype(str).tolist(),
        eager_candidate_weights=eager_candidate_weights,
        epoch_scales=epoch_scales,
        weight_cache_paths=(
            CANDIDATE_LIBRARY_DIR / "candidate_weights_wide.parquet",
            ENDPOINT_DISCOVERY_DIR / "endpoint_path_weights_wide.parquet",
        ),
    )


def selected_candidate_weights(data: DashboardData, candidate: str) -> pd.DataFrame:
    """Return selected candidate weights with materialized epoch diagnostics."""
    selected = load_selected_candidate_weights(
        candidate,
        eager_weights=data.eager_candidate_weights,
        parquet_weight_paths=list(data.weight_cache_paths),
    )
    proportional = data.eager_candidate_weights.loc[
        data.eager_candidate_weights["candidate"].eq(PROPORTIONAL_RUN_NAME)
    ].copy()
    if proportional.empty:
        proportional = data.eager_candidate_weights.loc[
            data.eager_candidate_weights["candidate"].eq("proportional")
        ].copy()
    merged = selected.merge(
        proportional.rename(columns={"weight": "proportional_weight"}).loc[
            :, ["domain", "phase", "proportional_weight"]
        ],
        on=["domain", "phase"],
        how="left",
    )
    merged["weight_delta"] = merged["weight"] - merged["proportional_weight"]
    return add_materialized_epochs(merged, data.epoch_scales)


def select_best_candidate(
    candidate_summary: pd.DataFrame,
    task_predictions: pd.DataFrame,
    *,
    constraints: dict[str, float],
    min_target_gain: float,
    max_nearest_tv: float,
    max_phase_weight: float,
    include_sources: set[str] | None,
    keep_missing: bool,
) -> tuple[str, pd.DataFrame]:
    """Return the best feasible candidate and the filtered candidate table."""
    filtered_base = candidate_summary
    if include_sources is not None:
        filtered_base = filtered_base.loc[filtered_base["source"].astype(str).isin(include_sources)]

    filtered = filter_candidate_summary(
        filtered_base,
        min_target_gain=min_target_gain,
        max_nearest_tv=max_nearest_tv,
        max_phase_weight=max_phase_weight,
    )
    if constraints:
        filtered = _filter_task_constraints_wide(
            filtered,
            task_predictions,
            constraints=constraints,
            keep_missing=keep_missing,
        )
    if filtered.empty:
        fallback = PROPORTIONAL_RUN_NAME
        return fallback, filtered
    score_column = "target_gain_lcb" if "target_gain_lcb" in filtered.columns else "target_gain"
    ranked = filtered.assign(
        _score=pd.to_numeric(filtered[score_column], errors="coerce").fillna(filtered["target_gain"])
    )
    ranked = ranked.sort_values(["_score", "target_gain", "nearest_observed_tv"], ascending=[False, False, True])
    filtered = ranked.drop(columns=["_score"]).reset_index(drop=True)
    return str(filtered.iloc[0]["candidate"]), filtered


def _filter_task_constraints_wide(
    candidate_summary: pd.DataFrame,
    task_predictions: pd.DataFrame,
    *,
    constraints: dict[str, float],
    keep_missing: bool,
) -> pd.DataFrame:
    """Vectorized per-task threshold filtering for the wide prediction cache."""
    active = {task: threshold for task, threshold in constraints.items() if np.isfinite(threshold)}
    if not active or candidate_summary.empty:
        return candidate_summary
    if task_predictions.empty or "candidate" not in task_predictions.columns:
        return candidate_summary if keep_missing else candidate_summary.iloc[0:0].copy()
    if not task_predictions["candidate"].is_unique:
        raise ValueError("task_predictions cache must have unique candidate rows")

    missing_tasks = [task for task in active if task not in task_predictions.columns]
    if missing_tasks and not keep_missing:
        return candidate_summary.iloc[0:0].copy()

    available_tasks = [task for task in active if task in task_predictions.columns]
    if not available_tasks:
        return candidate_summary if keep_missing else candidate_summary.iloc[0:0].copy()

    prediction_cols = ["candidate", *available_tasks]
    merged = candidate_summary.loc[:, ["candidate"]].merge(
        task_predictions.loc[:, prediction_cols],
        on="candidate",
        how="left",
    )
    passes = np.ones(len(merged), dtype=bool)
    for task in available_tasks:
        values = pd.to_numeric(merged[task], errors="coerce")
        task_passes = values.ge(float(active[task]))
        if keep_missing:
            task_passes = task_passes | values.isna()
        else:
            task_passes = task_passes & values.notna()
        passes &= task_passes.to_numpy(dtype=bool)
    return candidate_summary.loc[passes].reset_index(drop=True)


def update_constraint_from_slider(
    state: DashboardState,
    *,
    task_name: str,
    threshold: float,
    candidate_summary: pd.DataFrame,
    task_predictions: pd.DataFrame,
    min_target_gain: float,
    max_nearest_tv: float,
    max_phase_weight: float,
    include_sources: set[str] | None,
    keep_missing: bool,
) -> tuple[DashboardState, pd.DataFrame]:
    """Update one task threshold and optionally switch to the best feasible candidate."""
    constraints = dict(state.constraints)
    constraints[task_name] = float(threshold)
    selected, filtered = select_best_candidate(
        candidate_summary,
        task_predictions,
        constraints=constraints,
        min_target_gain=min_target_gain,
        max_nearest_tv=max_nearest_tv,
        max_phase_weight=max_phase_weight,
        include_sources=include_sources,
        keep_missing=keep_missing,
    )
    no_feasible = bool(filtered.empty and constraints)
    return (
        DashboardState(
            candidate=state.candidate if no_feasible else selected,
            constraints=constraints,
            starting_candidate=resolved_starting_candidate(state),
            no_feasible=no_feasible,
        ),
        filtered,
    )


def update_constraint_from_lock(
    state: DashboardState,
    *,
    task_name: str,
    threshold: float,
    locked: bool,
    candidate_summary: pd.DataFrame,
    task_predictions: pd.DataFrame,
    min_target_gain: float,
    max_nearest_tv: float,
    max_phase_weight: float,
    include_sources: set[str] | None,
    keep_missing: bool,
) -> tuple[DashboardState, pd.DataFrame]:
    """Add or remove one task constraint from a lock checkbox."""
    constraints = dict(state.constraints)
    if locked:
        constraints[task_name] = float(threshold)
    else:
        constraints.pop(task_name, None)
    selected, filtered = select_best_candidate(
        candidate_summary,
        task_predictions,
        constraints=constraints,
        min_target_gain=min_target_gain,
        max_nearest_tv=max_nearest_tv,
        max_phase_weight=max_phase_weight,
        include_sources=include_sources,
        keep_missing=keep_missing,
    )
    no_feasible = bool(filtered.empty and constraints)
    return (
        DashboardState(
            candidate=state.candidate if no_feasible else selected,
            constraints=constraints,
            starting_candidate=resolved_starting_candidate(state),
            no_feasible=no_feasible,
        ),
        filtered,
    )


def clear_constraints(
    state: DashboardState,
    *,
    candidate_summary: pd.DataFrame,
    task_predictions: pd.DataFrame,
    min_target_gain: float,
    max_nearest_tv: float,
    max_phase_weight: float,
    include_sources: set[str] | None,
    keep_missing: bool,
) -> tuple[DashboardState, pd.DataFrame]:
    """Clear all task constraints and return the best global candidate."""
    _, filtered = select_best_candidate(
        candidate_summary,
        task_predictions,
        constraints={},
        min_target_gain=min_target_gain,
        max_nearest_tv=max_nearest_tv,
        max_phase_weight=max_phase_weight,
        include_sources=include_sources,
        keep_missing=keep_missing,
    )
    starting_candidate = resolved_starting_candidate(state)
    return (
        DashboardState(
            candidate=starting_candidate,
            constraints={},
            starting_candidate=starting_candidate,
            no_feasible=False,
        ),
        filtered,
    )


def slider_values_for_candidate(
    task_predictions: pd.DataFrame,
    candidate: str,
    task_names: list[str],
) -> dict[str, float]:
    """Return predicted task slider values for a candidate."""
    rows = task_predictions.loc[task_predictions["candidate"].astype(str).eq(candidate)]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    values: dict[str, float] = {}
    for task in task_names:
        if task in row.index and pd.notna(row[task]) and np.isfinite(float(row[task])):
            values[task] = float(row[task])
    return values


def slider_values_for_state(
    task_predictions: pd.DataFrame,
    state: DashboardState,
    task_names: list[str],
) -> dict[str, float]:
    """Return slider values, preserving locked thresholds in no-feasible states."""
    values = slider_values_for_candidate(task_predictions, state.candidate, task_names)
    if state.no_feasible:
        for task, threshold in state.constraints.items():
            if task in task_names and np.isfinite(threshold):
                values[task] = float(threshold)
    return values


def manual_candidate_choices(candidate_summary: pd.DataFrame) -> list[str]:
    """Return curated candidates worth selecting by hand in the UI."""
    available = set(candidate_summary["candidate"].astype(str))
    return [candidate for candidate in PROMISING_MANUAL_CANDIDATES if candidate in available]


def constraints_table(state: DashboardState, task_predictions: pd.DataFrame) -> pd.DataFrame:
    """Build the locked constraints table for the selected candidate."""
    values = slider_values_for_candidate(task_predictions, state.candidate, list(state.constraints))
    rows = []
    for task, threshold in state.constraints.items():
        prediction = values.get(task, np.nan)
        rows.append(
            {
                "task": task,
                "target_threshold": float(threshold),
                "candidate_prediction": prediction,
                "passes": bool(np.isfinite(prediction) and prediction >= threshold),
            }
        )
    return pd.DataFrame.from_records(rows, columns=["task", "target_threshold", "candidate_prediction", "passes"])


def candidate_row(candidate_summary: pd.DataFrame, candidate: str) -> pd.DataFrame:
    """Return a compact one-row summary for a candidate."""
    display_cols = [
        "candidate",
        "source",
        "score_kind",
        "target_score",
        "target_gain",
        "target_gain_lcb",
        "nearest_observed_tv",
        "nearest_observed_run",
        "max_phase_weight",
        "min_phase_support_gt_1e3",
        "mean_phase_effective_support",
        "has_task_predictions",
        "deployability_flag",
    ]
    rows = candidate_summary.loc[candidate_summary["candidate"].astype(str).eq(candidate)].copy()
    return rows.loc[:, [col for col in display_cols if col in rows.columns]].head(1)
