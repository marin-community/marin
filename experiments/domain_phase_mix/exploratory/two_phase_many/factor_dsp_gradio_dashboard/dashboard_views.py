# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot and table builders for the Gradio factor-DSP dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from experiments.domain_phase_mix.exploratory.two_phase_many.factor_dsp_constraint_dashboard_helpers import (
    sample_frontier_for_plot,
    selected_task_prediction_long,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.factor_dsp_gradio_dashboard.dashboard_data import (
    DashboardData,
    DashboardState,
    constraints_table,
    selected_candidate_weights,
)


def short_task_label(task: str) -> str:
    """Return a compact but still traceable task label."""
    return (
        task.replace("eval/uncheatable_eval/", "uncheatable/")
        .replace("teacher_forced/", "tf/")
        .replace("lm_eval/", "")
        .replace("mcq_smooth/", "smooth/")
        .replace("choice_logprob", "clp")
    )


def empty_figure(title: str) -> go.Figure:
    """Return a stable empty Plotly figure with a visible title."""
    fig = go.Figure()
    fig.update_layout(title=title, height=360)
    fig.add_annotation(
        text="No data for the current selection.",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    return fig


def candidate_frontier_figure(filtered_candidates: pd.DataFrame) -> go.Figure:
    """Build a bounded frontier scatter for the currently feasible candidates."""
    if filtered_candidates.empty:
        return empty_figure("Candidate frontier")
    frame = sample_frontier_for_plot(filtered_candidates, max_rows=20_000, top_rows=2_000)
    hover_cols = [
        col
        for col in [
            "candidate",
            "target_gain_lcb",
            "max_phase_weight",
            "nearest_observed_run",
            "has_task_predictions",
            "deployability_flag",
        ]
        if col in frame.columns
    ]
    fig = px.scatter(
        frame,
        x="nearest_observed_tv",
        y="target_gain",
        color="source",
        symbol="score_kind" if "score_kind" in frame.columns else None,
        hover_data=hover_cols,
        title=f"Candidate frontier ({len(frame):,} plotted / {len(filtered_candidates):,} feasible)",
        labels={
            "nearest_observed_tv": "nearest observed TV",
            "target_gain": "predicted y_factor gain vs proportional",
        },
    )
    fig.add_vline(x=0.45, line_dash="dot", line_color="gray")
    fig.update_layout(height=500, legend_title_text="source")
    return fig


def selected_task_prediction_table(data: DashboardData, state: DashboardState) -> pd.DataFrame:
    """Return candidate-implied task deltas with constraint and quality metadata."""
    table = selected_task_prediction_long(
        data.task_predictions,
        state.candidate,
        thresholds=state.constraints,
    )
    if table.empty:
        return pd.DataFrame(
            columns=[
                "task",
                "predicted_delta",
                "target_threshold",
                "status",
                "train_pearson",
                "train_rmse",
            ]
        )
    table = table.rename(
        columns={
            "task_column": "task",
            "predicted_task_delta_standardized": "predicted_delta",
        }
    )
    table["status"] = np.where(
        table["locked"] & table["meets_target"],
        "locked pass",
        np.where(table["locked"], "locked fail", "unlocked"),
    )
    if not data.task_prediction_metrics.empty:
        quality = data.task_prediction_metrics.rename(columns={"task_column": "task"})
        table = table.merge(
            quality.loc[
                :, [col for col in ["task", "train_pearson", "train_rmse", "train_n"] if col in quality.columns]
            ],
            on="task",
            how="left",
        )
    table["short_task"] = table["task"].map(short_task_label)
    return table


def task_delta_figure(task_table: pd.DataFrame, candidate: str) -> go.Figure:
    """Plot candidate-implied task deltas for all current selected tasks."""
    if task_table.empty:
        return empty_figure("Candidate-implied task deltas")
    plot_frame = task_table.sort_values("predicted_delta")
    fig = px.bar(
        plot_frame,
        x="predicted_delta",
        y="short_task",
        color="status",
        orientation="h",
        color_discrete_map={
            "locked pass": "#2ca25f",
            "locked fail": "#de2d26",
            "unlocked": "#6baed6",
        },
        hover_data=[
            col for col in ["task", "target_threshold", "train_pearson", "train_rmse"] if col in plot_frame.columns
        ],
        title=f"Candidate-implied task deltas: {candidate}",
        labels={"predicted_delta": "standardized oriented delta vs proportional"},
    )
    fig.add_vline(x=0.0, line_dash="dot", line_color="gray")
    fig.update_layout(height=max(560, 18 * len(plot_frame)), yaxis_title="")
    return fig


def mixture_weight_figure(weight_table: pd.DataFrame, candidate: str) -> go.Figure:
    """Plot top phase weights for the selected candidate."""
    if weight_table.empty:
        return empty_figure("Selected mixture weights")
    top_domains = (
        weight_table.groupby("domain", observed=True)["weight"].max().sort_values(ascending=False).head(30).index
    )
    frame = weight_table.loc[weight_table["domain"].isin(top_domains)].copy()
    fig = px.bar(
        frame,
        x="domain",
        y="weight",
        color="phase",
        barmode="group",
        hover_data=["proportional_weight", "weight_delta"],
        title=f"Top materialized weights: {candidate}",
        labels={"weight": "phase weight"},
    )
    fig.update_layout(height=520, xaxis_tickangle=-45)
    return fig


def weight_delta_heatmap(weight_table: pd.DataFrame, candidate: str) -> go.Figure:
    """Plot phase/domain weight deltas against proportional."""
    if weight_table.empty:
        return empty_figure("Weight delta vs proportional")
    heatmap = weight_table.pivot(index="phase", columns="domain", values="weight_delta").fillna(0.0)
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap.to_numpy(),
            x=list(heatmap.columns),
            y=list(heatmap.index),
            colorscale="RdYlGn_r",
            zmid=0.0,
            colorbar={"title": "delta"},
        )
    )
    fig.update_layout(title=f"Weight delta vs proportional: {candidate}", height=420, xaxis_tickangle=-45)
    return fig


def epoch_figure(weight_table: pd.DataFrame, candidate: str) -> go.Figure:
    """Plot largest materialized epochs for the selected candidate."""
    if weight_table.empty:
        return empty_figure("Materialized epochs")
    frame = weight_table.sort_values("materialized_epochs", ascending=False).head(50).copy()
    fig = px.bar(
        frame,
        x="domain",
        y="materialized_epochs",
        color="phase",
        barmode="group",
        hover_data=["proportional_epochs", "epoch_delta_vs_proportional", "weight"],
        title=f"Materialized epochs: {candidate}",
    )
    fig.update_layout(height=520, xaxis_tickangle=-45)
    return fig


def epoch_summary_table(weight_table: pd.DataFrame) -> pd.DataFrame:
    """Return compact materialized-epoch diagnostics."""
    if weight_table.empty:
        return pd.DataFrame()
    return (
        weight_table.sort_values("materialized_epochs", ascending=False)
        .groupby("phase", observed=True)
        .agg(
            max_materialized_epochs=("materialized_epochs", "max"),
            max_epoch_domain=("domain", "first"),
            median_materialized_epochs=("materialized_epochs", "median"),
            mean_materialized_epochs=("materialized_epochs", "mean"),
        )
        .reset_index()
    )


def selected_weight_tables(data: DashboardData, candidate: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load selected weights and derive a compact epoch summary."""
    weights = selected_candidate_weights(data, candidate)
    return weights, epoch_summary_table(weights)


def locked_constraints_table(data: DashboardData, state: DashboardState) -> pd.DataFrame:
    """Return the visible locked-constraints table."""
    return constraints_table(state, data.task_predictions)


def matched_candidate_table(candidate_summary: pd.DataFrame, state: DashboardState) -> pd.DataFrame:
    """Return the selected library row with explicit selection semantics."""
    selection = "no feasible candidate" if state.no_feasible else "matched precomputed candidate"
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
        "has_task_predictions",
        "deployability_flag",
    ]
    rows = candidate_summary.loc[candidate_summary["candidate"].astype(str).eq(state.candidate)].copy()
    if rows.empty:
        return pd.DataFrame.from_records(
            [{"selection": selection, "candidate": state.candidate, "source": "missing from candidate_summary"}]
        )
    row = rows.loc[:, [col for col in display_cols if col in rows.columns]].head(1).copy()
    row.insert(0, "selection", selection)
    return row


READINESS_ORDER = {
    "unknown": 0,
    "weak": 1,
    "caution": 2,
    "usable": 3,
    "ready": 4,
}


def _readiness_from_pearson(value: object) -> str:
    pearson = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(pearson):
        return "unknown"
    if pearson >= 0.70:
        return "ready"
    if pearson >= 0.50:
        return "usable"
    if pearson >= 0.30:
        return "caution"
    return "weak"


def _steering_guidance(readiness: str) -> str:
    return {
        "ready": "reasonable active constraint",
        "usable": "usable with caution",
        "caution": "prefer guardrail, verify manually",
        "weak": "do not trust as steering objective",
        "unknown": "missing quality estimate",
    }[readiness]


def _format_readiness_number(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "n/a"
    return f"{float(numeric):.2f}"


def task_readiness_table_from_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build first-class task-readiness rows from task surrogate fit metrics."""
    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "readiness",
                "readiness_rank",
                "task",
                "short_task",
                "train_pearson",
                "train_rmse",
                "train_n",
                "prediction_source",
                "steering_guidance",
            ]
        )
    frame = metrics.copy()
    if "task_column" in frame.columns:
        frame["task"] = frame["task_column"].astype(str)
    elif "task" not in frame.columns:
        raise ValueError("task readiness metrics must include task_column or task")
    frame["short_task"] = frame["task"].map(short_task_label)
    frame["train_pearson"] = pd.to_numeric(frame.get("train_pearson"), errors="coerce")
    frame["train_rmse"] = pd.to_numeric(frame.get("train_rmse"), errors="coerce")
    frame["readiness"] = frame["train_pearson"].map(_readiness_from_pearson)
    frame["readiness_rank"] = frame["readiness"].map(READINESS_ORDER).astype(int)
    frame["steering_guidance"] = frame["readiness"].map(_steering_guidance)
    display_cols = [
        "readiness",
        "readiness_rank",
        "task",
        "short_task",
        "train_pearson",
        "train_rmse",
        "train_n",
        "prediction_source",
        "alpha",
        "steering_guidance",
    ]
    visible = [col for col in display_cols if col in frame.columns]
    return (
        frame.loc[:, visible]
        .sort_values(
            ["readiness_rank", "train_pearson", "train_rmse", "task"],
            ascending=[True, True, False, True],
            na_position="first",
        )
        .reset_index(drop=True)
    )


def task_slider_metadata_from_metrics(metrics: pd.DataFrame, task_names: list[str]) -> dict[str, dict[str, str]]:
    """Return slider labels and info strings with colocated readiness metadata."""
    readiness = task_readiness_table_from_metrics(metrics).set_index("task", drop=False)
    metadata: dict[str, dict[str, str]] = {}
    for task in task_names:
        if task in readiness.index:
            row = readiness.loc[task]
            readiness_label = str(row["readiness"])
            pearson = _format_readiness_number(row.get("train_pearson"))
            rmse = _format_readiness_number(row.get("train_rmse"))
            train_n = row.get("train_n", "n/a")
            guidance = str(row["steering_guidance"])
        else:
            readiness_label = "unknown"
            pearson = "n/a"
            rmse = "n/a"
            train_n = "n/a"
            guidance = _steering_guidance(readiness_label)
        metadata[task] = {
            "label": f"[{readiness_label} r={pearson}] {task}",
            "info": (
                "0 = proportional; positive = predicted improvement. "
                f"Readiness={readiness_label}; train Pearson={pearson}; RMSE={rmse}; n={train_n}. {guidance}."
            ),
        }
    return metadata


def task_readiness_summary_from_table(readiness_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize task-readiness categories for the visible top-level panel."""
    if readiness_table.empty or "readiness" not in readiness_table.columns:
        return pd.DataFrame(columns=["readiness", "task_count", "median_train_pearson", "guidance"])
    summary = (
        readiness_table.groupby("readiness", observed=True)
        .agg(
            task_count=("task", "count"),
            median_train_pearson=("train_pearson", "median"),
        )
        .reset_index()
    )
    summary["readiness_rank"] = summary["readiness"].map(READINESS_ORDER).astype(int)
    summary["guidance"] = summary["readiness"].map(_steering_guidance)
    return summary.sort_values("readiness_rank").drop(columns=["readiness_rank"]).reset_index(drop=True)


def task_readiness_table(data: DashboardData) -> pd.DataFrame:
    """Return per-task slider readiness, lower-quality tasks first."""
    return task_readiness_table_from_metrics(data.task_prediction_metrics)


def task_readiness_summary(data: DashboardData) -> pd.DataFrame:
    """Return compact counts of task-readiness categories."""
    return task_readiness_summary_from_table(task_readiness_table(data))


def quality_table(data: DashboardData) -> pd.DataFrame:
    """Return task surrogate quality, with lower-quality tasks first."""
    return task_readiness_table(data).head(60)
