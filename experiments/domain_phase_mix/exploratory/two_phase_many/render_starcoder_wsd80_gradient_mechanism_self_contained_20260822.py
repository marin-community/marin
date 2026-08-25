# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.5",
# ]
# ///
"""Render a self-contained presentation revision of the audited gradient-mechanism plots."""

import argparse
import hashlib
import html
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    plot_starcoder_wsd80_gradient_mechanism_repair_20260820 as base,
)

SUPPORT_LABELS = {
    "full": "Full source pool",
    "m100a": "Finite support A (m100a)",
    "m100b": "Finite support B (m100b)",
}
SUPPORT_ORDER = {"full": 0, "m100a": 1, "m100b": 2}
ANALYSIS_LABELS = {
    "h1_trajectory_extension": "H1 descriptive source alignment",
    "h2_primary": "H2 target revaluation",
    "h3_full_support_pair": "H3 full-pool comparator",
    "h3_second_pool_sensitivity": "H3 finite-pool B sensitivity",
    "h5_preregistered_profile": "H5 moved-switch profile",
}
ANALYSIS_ORDER = {key: index for index, key in enumerate(ANALYSIS_LABELS)}
POLICY_LABELS = {
    "common_tied_035": "Tied 35% StarCoder",
    "boundary_beta_0p60": "Data switch at 0.60T",
    "boundary_beta_0p85": "Data switch at 0.85T",
}
POLICY_ORDER = {"common_tied_035": 0, "boundary_beta_0p60": 1, "boundary_beta_0p85": 2}
DISPLAY_REPLACEMENTS = {
    "Post-outcome v10 repair state": "Post-outcome v10 repair",
    "post-outcome v10 state": "Post-outcome v10 repair",
    "Post-outcome plot completion": "Post-outcome plot completion",
    "post-outcome completion": "Post-outcome plot completion",
    "v10 H1 contract state": "Frozen H1",
    "StarCoder support": "StarCoder training-source reference",
    "policy switch": "data switch",
    "Phase 2 begins · 0.60T": "Data switch · 0.60T",
    "Phase 2 begins · 0.80T": "LR decay begins · 0.80T",
    "Phase 2 begins · 0.85T": "Data switch · 0.85T",
    "The vertical bar marks phase 2.": (
        "The dark bar marks the selected policy's data switch, or the 0.80T LR-decay onset for a tied policy."
    ),
    "The dark solid bar marks phase 2; moved-switch policies also show fixed LR-decay onset as a slate dashed bar.": (
        "The dark solid bar marks the data switch for moved-switch policies or the 0.80T LR-decay onset for tied "
        "policies; moved-switch policies also show the fixed LR-decay onset as a slate dashed bar."
    ),
}
FRIENDLY_TOKEN_LABELS = {
    **{key.replace("_", " "): value for key, value in SUPPORT_LABELS.items()},
    **{key.replace("_", " "): value for key, value in ANALYSIS_LABELS.items()},
    **{key.replace("_", " "): value for key, value in POLICY_LABELS.items()},
}
CONTROL_LABELS = frozenset({"Full timeline", "Zoom LR decay"})
LR_DECAY_COLOR = "#6f7f87"
FIXED_TPP_CELL_ORDER = (
    "r0_shared_h0640_s03820",
    "r1_increase_d_h0640_s07320",
    "r2_increase_d_h0640_s14960",
    "r3_increase_d_h0640_s28260",
)
FIXED_TPP_SOURCE_LABELS = {
    "nemotron_aggregate": "Nemotron",
    "starcoder_excluded_global": "StarCoder heldout",
}
FIXED_TPP_SOURCE_COLORS = {"nemotron_aggregate": "#1b7f79", "starcoder_excluded_global": "#d65a31"}
FIXED_TPP_TARGET_LABELS = {
    "paloma_programming_languages": "Programming Languages",
    "paloma_c4_en": "C4",
    "uncheatable_github_python": "GitHub Python",
    "uncheatable_wikipedia_english": "Wikipedia",
}
FIXED_TPP_BOOTSTRAP_DRAWS = 20_000

EXPECTED_SUPPORTS = frozenset(SUPPORT_LABELS)
EXPECTED_ANALYSES = frozenset(ANALYSIS_LABELS)
EXPECTED_POLICIES = frozenset(POLICY_LABELS)

PANEL_GUIDES = {
    "source_source_conflict_trajectory.html": (
        "Source alignment trajectory",
        "Choose a source pool and policy. The teal line is the cosine between StarCoder and Nemotron raw gradients; "
        "the orange line is the cosine between their data-induced optimizer updates. The horizontal axis is the "
        "actual fraction of training updates, so the probes around 0.80T are intentionally close together.",
    ),
    "source_source_conflict_matrix.html": (
        "Source alignment matrix",
        "Choose a source pool and policy. Columns are restored training states; rows distinguish raw gradients from "
        "data-induced optimizer updates. Negative values are direct conflict.",
    ),
    "target_source_utility_trajectories.html": (
        "Target-source utility trajectories",
        "Choose a source pool, study role, and policy. Each subplot is one evaluation target; each colored line is a "
        "candidate source update. The y-axis is cos(-target gradient, source update), so positive values mean the "
        "update locally reduces that target's loss. The horizontal axis is true training progress.",
    ),
    "target_source_utility_matrix.html": (
        "Target-source utility matrix",
        "Choose a source pool, study role, policy, restored state, and evidence provenance. Rows are evaluation "
        "targets and columns are candidate source updates. Positive values mean locally helpful alignment.",
    ),
    "target_source_choice_alignment.html": (
        "Target-conditioned source choice",
        "Choose a source pool, study role, policy, and source contrast. Positive values favor the first source named "
        "in the contrast over the second; negative values favor the second.",
    ),
    "mechanism_effect_forest.html": (
        "Mechanism contrasts",
        "This panel has no selector. It collects the 47 post-outcome H2, H3, and H5 development contrasts. Filled "
        "diamonds survive the global two-sided Holm correction; hollow circles do not.",
    ),
}

SUPPORT_EXPLANATIONS = {
    "Full source pool": (
        "This setting uses the broad StarCoder source pool, so it does not impose finite-support replay."
    ),
    "Finite support A (m100a)": (
        "This setting repeatedly samples the primary fixed StarCoder subset, m100a, creating simulated epoching."
    ),
    "Finite support B (m100b)": (
        "This sensitivity setting repeatedly samples a sequence-disjoint fixed StarCoder subset, m100b. It checks "
        "support dependence rather than serving as an independent population replicate of m100a."
    ),
}
ANALYSIS_EXPLANATIONS = {
    "H1 descriptive source alignment": (
        "This is the descriptive H1 source-alignment view, which asks whether StarCoder and Nemotron become more "
        "directly conflicting through training."
    ),
    "H2 target revaluation": (
        "It is the primary H2 arm, which asks whether the locally useful training source changes as training advances."
    ),
    "H3 full-pool comparator": (
        "It is H3's broad-pool comparator, used to separate generic temporal change from finite-support repetition."
    ),
    "H3 finite-pool B sensitivity": (
        "It is H3's support-B sensitivity check; comparison with m100a tests whether repetition findings depend on "
        "one particular finite subset."
    ),
    "H5 moved-switch profile": (
        "It belongs to H5, which moves the data-switch time while holding aggregate exposure, phase contrast, and "
        "the 0.80T learning-rate schedule fixed."
    ),
}
POLICY_EXPLANATIONS = {
    "Tied 35% StarCoder": (
        "The policy uses 35% StarCoder throughout training; the 0.80T marker is learning-rate decay, not a data switch."
    ),
    "Data switch at 0.60T": (
        "The policy uses about 2% StarCoder before 0.60T and 42% afterward, preserving 18% aggregate StarCoder "
        "exposure; learning-rate decay still begins at 0.80T."
    ),
    "Data switch at 0.85T": (
        "The policy uses about 12% StarCoder before 0.85T and 52% afterward, preserving 18% aggregate StarCoder "
        "exposure; learning-rate decay still begins at 0.80T."
    ),
}
EVIDENCE_EXPLANATIONS = {
    "Frozen H1": "This checkpoint was specified before outcome inspection.",
    "Post-outcome v10 repair": (
        "This post-outcome measurement repaired the numerical mechanism pipeline; it is descriptive and does not "
        "revise the frozen tests."
    ),
    "Post-outcome plot completion": (
        "This checkpoint was recovered post-outcome to complete the trajectory; it is descriptive and does not "
        "revise the frozen tests."
    ),
}
SOURCE_CONTRAST_EXPLANATIONS = {
    "StarCoder heldout - Nemotron": (
        "The plotted contrast is the heldout-StarCoder update minus the Nemotron update; positive values favor "
        "heldout StarCoder for the named target, while negative values favor Nemotron."
    ),
    "StarCoder training-source reference - Nemotron": (
        "The plotted contrast is the selected arm's StarCoder training-source update minus the Nemotron update; "
        "positive values favor that StarCoder source for the named target."
    ),
    "StarCoder training-source reference - StarCoder heldout": (
        "The plotted contrast is the selected arm's StarCoder training-source update minus the global heldout-"
        "StarCoder update; positive values favor the training-source reference."
    ),
}
SOURCE_TIED_ANALYSIS_EXPLANATIONS = {
    "Full source pool": (
        "This descriptive source-alignment view combines preregistered H1 checkpoints with later states registered "
        "under H3's full-pool comparator."
    ),
    "Finite support A (m100a)": (
        "This descriptive source-alignment view combines preregistered H1 checkpoints with later states registered "
        "under the primary H2 target-revaluation arm."
    ),
    "Finite support B (m100b)": (
        "This descriptive source-alignment view combines preregistered H1 checkpoints with later states registered "
        "under H3's finite-support-B sensitivity arm."
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_create_only(path: Path, payload: str) -> None:
    encoded = payload.encode()
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as error:
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Refusing to overwrite a non-identical presentation artifact: {path}") from error


def _bootstrap_mean_interval(values: np.ndarray, *, key: str) -> tuple[float, float, float]:
    if len(values) != 8 or not np.isfinite(values).all():
        raise ValueError(f"Expected eight finite paired-seed values for {key}")
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(values), size=(FIXED_TPP_BOOTSTRAP_DRAWS, len(values)))
    bootstrap = values[sample_indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def _summarize_fixed_tpp_measurements(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_key, group in frame.groupby(group_columns, sort=False, dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row: dict[str, Any] = dict(zip(group_columns, key_values, strict=True))
        for value_column in value_columns:
            mean, low, high = _bootstrap_mean_interval(
                group[value_column].to_numpy(dtype=float),
                key=f"{value_column}:{':'.join(map(str, key_values))}",
            )
            row[f"{value_column}_mean"] = mean
            row[f"{value_column}_ci95_low"] = low
            row[f"{value_column}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _slider_steps(
    *,
    trace_count: int,
    traces_per_cell: int,
    labels: list[str],
    initially_visible_per_cell: int | None = None,
) -> list[dict[str, Any]]:
    if trace_count != traces_per_cell * len(labels):
        raise ValueError("Fixed-TPP slider trace inventory is inconsistent")
    visible_count = traces_per_cell if initially_visible_per_cell is None else initially_visible_per_cell
    if not 0 < visible_count <= traces_per_cell:
        raise ValueError("Initial fixed-TPP trace count must be within one cell's trace inventory")
    steps: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        visible = [False] * trace_count
        start = index * traces_per_cell
        visible[start : start + visible_count] = [True] * visible_count
        steps.append({"method": "update", "label": label, "args": [{"visible": visible}]})
    return steps


def _normalize_fixed_tpp_norms(
    source: pd.DataFrame,
    value_columns: tuple[str, ...],
    *,
    identity: tuple[str, ...] = ("cell_id", "training_seed"),
) -> pd.DataFrame:
    onset = source[source["checkpoint_label"].eq("decay_onset")][[*identity, *value_columns]].copy()
    if onset.duplicated(list(identity)).any() or len(onset) != source[list(identity)].drop_duplicates().shape[0]:
        raise ValueError("Each fixed-TPP seed must have exactly one LR-decay-onset norm reference")
    onset = onset.rename(columns={column: f"{column}_at_decay_onset" for column in value_columns})
    normalized = source.merge(onset, on=identity, how="left", validate="many_to_one")
    for column in value_columns:
        reference_column = f"{column}_at_decay_onset"
        if normalized[reference_column].isna().any() or normalized[reference_column].le(0).any():
            raise ValueError(f"Invalid LR-decay-onset norm reference for {column}")
        normalized[f"{column}_relative"] = normalized[column] / normalized[reference_column]
    return normalized


def fixed_tpp_source_trajectory_figure(source: pd.DataFrame) -> go.Figure:
    norm_columns = (
        "gradient_raw_left_norm",
        "gradient_raw_right_norm",
        "optimizer_update_raw_left_norm",
        "optimizer_update_raw_right_norm",
    )
    source = _normalize_fixed_tpp_norms(source, norm_columns)
    summary = _summarize_fixed_tpp_measurements(
        source,
        group_columns=["cell_id", "checkpoint_label", "normalized_time", "total_parameter_tpp"],
        value_columns=[
            "gradient_raw_cosine",
            "optimizer_update_raw_cosine",
            *norm_columns,
            *(f"{column}_relative" for column in norm_columns),
        ],
    )
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    labels: list[str] = []
    measures = (
        ("gradient_raw_cosine", "Raw gradient", "#1b7f79"),
        ("optimizer_update_raw_cosine", "Optimizer update", "#d65a31"),
    )
    norm_measures = (
        ("gradient_raw_left_norm", "Raw norm · StarCoder", "#d65a31", "dot", "circle-open"),
        ("gradient_raw_right_norm", "Raw norm · Nemotron", "#1b7f79", "dot", "circle-open"),
        ("optimizer_update_raw_left_norm", "Update norm · StarCoder", "#d65a31", "dash", "square-open"),
        ("optimizer_update_raw_right_norm", "Update norm · Nemotron", "#1b7f79", "dash", "square-open"),
    )
    for cell_index, cell_id in enumerate(FIXED_TPP_CELL_ORDER):
        group = summary[summary["cell_id"].eq(cell_id)].sort_values("normalized_time")
        tpp = float(group["total_parameter_tpp"].iloc[0])
        labels.append(f"{tpp:.2f}")
        for value_column, label, color in measures:
            mean = group[f"{value_column}_mean"]
            figure.add_trace(
                go.Scatter(
                    x=group["normalized_time"],
                    y=mean,
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": group[f"{value_column}_ci95_high"] - mean,
                        "arrayminus": mean - group[f"{value_column}_ci95_low"],
                        "thickness": 1.5,
                    },
                    mode="lines+markers",
                    name=label,
                    visible=cell_index == 0,
                    meta={"cell_index": cell_index, "visibility_group": "cosine"},
                    line={"color": color, "width": 3},
                    marker={"size": 8},
                    customdata=group["checkpoint_label"],
                    hovertemplate=(
                        f"{label}<br>state %{{customdata}}<br>time %{{x:.4f}}T" "<br>cosine %{y:.4f}<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )
        for value_column, label, color, dash, symbol in norm_measures:
            relative_column = f"{value_column}_relative"
            mean = group[f"{relative_column}_mean"]
            customdata = np.empty((len(group), 2), dtype=object)
            customdata[:, 0] = group["checkpoint_label"].astype(str)
            customdata[:, 1] = group[f"{value_column}_mean"].to_numpy(dtype=float)
            figure.add_trace(
                go.Scatter(
                    x=group["normalized_time"],
                    y=mean,
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": group[f"{relative_column}_ci95_high"] - mean,
                        "arrayminus": mean - group[f"{relative_column}_ci95_low"],
                        "thickness": 1.2,
                    },
                    mode="lines+markers",
                    name=label,
                    visible=False,
                    meta={
                        "cell_index": cell_index,
                        "visibility_group": "raw_norm" if value_column.startswith("gradient") else "update_norm",
                    },
                    line={"color": color, "width": 2.2, "dash": dash},
                    marker={"size": 7, "symbol": symbol},
                    customdata=customdata,
                    hovertemplate=(
                        f"{label}<br>state %{{customdata[0]}}<br>time %{{x:.4f}}T"
                        "<br>relative to 0.80T %{y:.3f}x"
                        "<br>absolute norm %{customdata[1]:.4f}<extra></extra>"
                    ),
                ),
                secondary_y=True,
            )
    figure.add_vline(x=0.8, line_width=2, line_color="#183149")
    figure.add_annotation(
        x=0.8,
        y=1,
        yref="paper",
        text="LR decay begins",
        showarrow=False,
        textangle=-90,
        xshift=-13,
        yshift=-50,
        font={"size": 12},
    )
    figure.update_xaxes(title_text="Training progress", range=[0.53, 0.92], tickformat=".2f")
    figure.update_yaxes(title_text="StarCoder-Nemotron cosine", range=[0.16, 0.66], secondary_y=False)
    figure.update_yaxes(
        title_text="Vector norm / norm at 0.80T",
        range=[0.25, 1.20],
        showgrid=False,
        visible=False,
        secondary_y=True,
    )
    figure.update_layout(
        template="plotly_white",
        autosize=True,
        height=540,
        margin={"l": 78, "r": 84, "t": 110, "b": 118},
        legend={"orientation": "h", "x": 0, "y": 1.04, "xanchor": "left", "yanchor": "bottom"},
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Total TPP: ", "font": {"size": 15}},
                "pad": {"t": 48},
                "steps": _slider_steps(
                    trace_count=len(figure.data),
                    traces_per_cell=6,
                    initially_visible_per_cell=2,
                    labels=labels,
                ),
            }
        ],
        font={"family": "Avenir Next, sans-serif", "color": "#183149", "size": 14},
        paper_bgcolor="#fffdf7",
        plot_bgcolor="#fffdf7",
    )
    return figure


def fixed_tpp_target_trajectory_figure(
    target_gradients: pd.DataFrame,
    target_utilities: pd.DataFrame,
) -> go.Figure:
    identity = [
        "row_id",
        "cell_id",
        "checkpoint_label",
        "training_seed",
        "normalized_time",
        "total_parameter_tpp",
        "target",
        "source",
    ]
    gradient_columns = [
        "target_source_gradient_raw_cosine",
        "target_gradient_raw_norm",
        "source_gradient_raw_norm",
    ]
    utility_columns = [
        "utility_raw_cosine",
        "utility_target_gradient_raw_norm",
        "source_update_raw_norm",
    ]
    merged = target_gradients[[*identity, *gradient_columns]].merge(
        target_utilities[[*identity, *utility_columns]],
        on=identity,
        validate="one_to_one",
    )
    if not np.allclose(
        merged["target_gradient_raw_norm"],
        merged["utility_target_gradient_raw_norm"],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Target-gradient norms disagree between the gradient and utility tables")
    merged = merged.drop(columns="utility_target_gradient_raw_norm")
    target_identity = [
        "row_id",
        "cell_id",
        "checkpoint_label",
        "training_seed",
        "normalized_time",
        "total_parameter_tpp",
        "target",
    ]
    target_norm_spread = merged.groupby(target_identity, observed=True)["target_gradient_raw_norm"].agg(
        lambda values: float(values.max() - values.min())
    )
    if target_norm_spread.max() > 1e-12:
        raise ValueError("Target-gradient norm changes across candidate sources")
    norm_columns = (
        "target_gradient_raw_norm",
        "source_gradient_raw_norm",
        "source_update_raw_norm",
    )
    merged = _normalize_fixed_tpp_norms(
        merged,
        norm_columns,
        identity=("cell_id", "training_seed", "target", "source"),
    )
    summary = _summarize_fixed_tpp_measurements(
        merged,
        group_columns=["cell_id", "checkpoint_label", "normalized_time", "total_parameter_tpp", "target", "source"],
        value_columns=[
            "target_source_gradient_raw_cosine",
            "utility_raw_cosine",
            *norm_columns,
            *(f"{column}_relative" for column in norm_columns),
        ],
    )
    targets = list(FIXED_TPP_TARGET_LABELS)
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"secondary_y": True}, {"secondary_y": True}], [{"secondary_y": True}, {"secondary_y": True}]],
        subplot_titles=tuple(FIXED_TPP_TARGET_LABELS[target] for target in targets),
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )
    labels: list[str] = []
    cosine_traces_per_cell = len(targets) * len(FIXED_TPP_SOURCE_LABELS) * 2
    raw_norm_traces_per_cell = len(targets) * (1 + len(FIXED_TPP_SOURCE_LABELS))
    update_norm_traces_per_cell = len(targets) * len(FIXED_TPP_SOURCE_LABELS)
    traces_per_cell = cosine_traces_per_cell + raw_norm_traces_per_cell + update_norm_traces_per_cell
    for cell_index, cell_id in enumerate(FIXED_TPP_CELL_ORDER):
        cell = summary[summary["cell_id"].eq(cell_id)]
        tpp = float(cell["total_parameter_tpp"].iloc[0])
        labels.append(f"{tpp:.2f}")
        for target_index, target in enumerate(targets):
            row = target_index // 2 + 1
            col = target_index % 2 + 1
            for source in FIXED_TPP_SOURCE_LABELS:
                group = cell[cell["target"].eq(target) & cell["source"].eq(source)].sort_values("normalized_time")
                for value_column, measure_label, dash, symbol, secondary_y in (
                    ("target_source_gradient_raw_cosine", "raw gradient", "solid", "circle", False),
                    ("utility_raw_cosine", "optimizer utility", "dash", "diamond", True),
                ):
                    mean = group[f"{value_column}_mean"]
                    label = f"{FIXED_TPP_SOURCE_LABELS[source]} · {measure_label}"
                    figure.add_trace(
                        go.Scatter(
                            x=group["normalized_time"],
                            y=mean,
                            error_y={
                                "type": "data",
                                "symmetric": False,
                                "array": group[f"{value_column}_ci95_high"] - mean,
                                "arrayminus": mean - group[f"{value_column}_ci95_low"],
                                "thickness": 1.2,
                            },
                            mode="lines+markers",
                            name=label,
                            legendgroup=f"{source}:{value_column}",
                            showlegend=target_index == 0,
                            visible=cell_index == 0,
                            meta={"cell_index": cell_index, "visibility_group": "cosine"},
                            line={"color": FIXED_TPP_SOURCE_COLORS[source], "width": 2.4, "dash": dash},
                            marker={"size": 7, "symbol": symbol},
                            customdata=group["checkpoint_label"],
                            hovertemplate=(
                                f"{FIXED_TPP_TARGET_LABELS[target]} · {label}<br>state %{{customdata}}"
                                "<br>time %{x:.4f}T<br>cosine %{y:.4f}<extra></extra>"
                            ),
                        ),
                        row=row,
                        col=col,
                        secondary_y=secondary_y,
                    )
        for target_index, target in enumerate(targets):
            row = target_index // 2 + 1
            col = target_index % 2 + 1
            target_group = cell[
                cell["target"].eq(target) & cell["source"].eq(next(iter(FIXED_TPP_SOURCE_LABELS)))
            ].sort_values("normalized_time")
            target_relative_column = "target_gradient_raw_norm_relative"
            target_mean = target_group[f"{target_relative_column}_mean"]
            target_customdata = np.empty((len(target_group), 2), dtype=object)
            target_customdata[:, 0] = target_group["checkpoint_label"].astype(str)
            target_customdata[:, 1] = target_group["target_gradient_raw_norm_mean"].to_numpy(dtype=float)
            figure.add_trace(
                go.Scatter(
                    x=target_group["normalized_time"],
                    y=target_mean,
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": target_group[f"{target_relative_column}_ci95_high"] - target_mean,
                        "arrayminus": target_mean - target_group[f"{target_relative_column}_ci95_low"],
                        "thickness": 1.2,
                    },
                    mode="lines+markers",
                    name="Target gradient norm",
                    legendgroup="target_gradient_norm",
                    showlegend=target_index == 0,
                    visible=False,
                    meta={"cell_index": cell_index, "visibility_group": "raw_norm"},
                    line={"color": "#183149", "width": 2.2, "dash": "dot"},
                    marker={"size": 7, "symbol": "triangle-up-open"},
                    customdata=target_customdata,
                    hovertemplate=(
                        f"{FIXED_TPP_TARGET_LABELS[target]} · target gradient norm"
                        "<br>state %{customdata[0]}<br>time %{x:.4f}T"
                        "<br>relative to 0.80T %{y:.3f}x"
                        "<br>absolute norm %{customdata[1]:.4f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
                secondary_y=False,
            )
            for source in FIXED_TPP_SOURCE_LABELS:
                group = cell[cell["target"].eq(target) & cell["source"].eq(source)].sort_values("normalized_time")
                for value_column, measure_label, dash, symbol, visibility_group in (
                    ("source_gradient_raw_norm", "raw norm", "dot", "circle-open", "raw_norm"),
                    ("source_update_raw_norm", "update norm", "dash", "square-open", "update_norm"),
                ):
                    relative_column = f"{value_column}_relative"
                    mean = group[f"{relative_column}_mean"]
                    label = f"{FIXED_TPP_SOURCE_LABELS[source]} · {measure_label}"
                    customdata = np.empty((len(group), 2), dtype=object)
                    customdata[:, 0] = group["checkpoint_label"].astype(str)
                    customdata[:, 1] = group[f"{value_column}_mean"].to_numpy(dtype=float)
                    figure.add_trace(
                        go.Scatter(
                            x=group["normalized_time"],
                            y=mean,
                            error_y={
                                "type": "data",
                                "symmetric": False,
                                "array": group[f"{relative_column}_ci95_high"] - mean,
                                "arrayminus": mean - group[f"{relative_column}_ci95_low"],
                                "thickness": 1.2,
                            },
                            mode="lines+markers",
                            name=label,
                            legendgroup=f"{source}:{value_column}",
                            showlegend=target_index == 0,
                            visible=False,
                            meta={"cell_index": cell_index, "visibility_group": visibility_group},
                            line={"color": FIXED_TPP_SOURCE_COLORS[source], "width": 2.2, "dash": dash},
                            marker={"size": 7, "symbol": symbol},
                            customdata=customdata,
                            hovertemplate=(
                                f"{FIXED_TPP_TARGET_LABELS[target]} · {label}"
                                "<br>state %{customdata[0]}<br>time %{x:.4f}T"
                                "<br>relative to 0.80T %{y:.3f}x"
                                "<br>absolute norm %{customdata[1]:.4f}<extra></extra>"
                            ),
                        ),
                        row=row,
                        col=col,
                        secondary_y=False,
                    )
    for target_index in range(len(targets)):
        row = target_index // 2 + 1
        col = target_index % 2 + 1
        figure.add_vline(x=0.8, line_width=1.5, line_color="#183149", row=row, col=col)
        figure.update_xaxes(range=[0.53, 0.92], tickformat=".2f", row=row, col=col)
        figure.update_yaxes(range=[0, 1], row=row, col=col, secondary_y=False)
        figure.update_yaxes(range=[0, 0.20], row=row, col=col, secondary_y=True)
    figure.update_xaxes(title_text="Training progress", row=2, col=1)
    figure.update_xaxes(title_text="Training progress", row=2, col=2)
    figure.update_yaxes(title_text="Raw gradient cosine", row=1, col=1, secondary_y=False)
    figure.update_yaxes(title_text="Raw gradient cosine", row=2, col=1, secondary_y=False)
    figure.update_yaxes(title_text="Optimizer utility cosine", row=1, col=2, secondary_y=True)
    figure.update_yaxes(title_text="Optimizer utility cosine", row=2, col=2, secondary_y=True)
    figure.update_layout(
        template="plotly_white",
        autosize=True,
        height=820,
        margin={"l": 78, "r": 82, "t": 112, "b": 118},
        legend={"orientation": "h", "x": 0, "y": 1.08, "xanchor": "left", "yanchor": "bottom"},
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Total TPP: ", "font": {"size": 15}},
                "pad": {"t": 48},
                "steps": _slider_steps(
                    trace_count=len(figure.data),
                    traces_per_cell=traces_per_cell,
                    initially_visible_per_cell=cosine_traces_per_cell,
                    labels=labels,
                ),
            }
        ],
        font={"family": "Avenir Next, sans-serif", "color": "#183149", "size": 13},
        paper_bgcolor="#fffdf7",
        plot_bgcolor="#fffdf7",
    )
    return figure


def _selector_label(raw_label: str) -> str:
    parts = [part.strip() for part in raw_label.split(" | ")]
    supports: list[str] = []
    analyses: list[str] = []
    policies: list[str] = []
    remaining: list[str] = []
    support_tokens = {key.replace("_", " ") for key in SUPPORT_LABELS}
    analysis_tokens = {key.replace("_", " ") for key in ANALYSIS_LABELS}
    policy_tokens = {key.replace("_", " ") for key in POLICY_LABELS}
    for part in parts:
        if part in support_tokens:
            supports.append(FRIENDLY_TOKEN_LABELS[part])
        elif part in analysis_tokens:
            analyses.append(FRIENDLY_TOKEN_LABELS[part])
        elif part in policy_tokens:
            policies.append(FRIENDLY_TOKEN_LABELS[part])
        else:
            remaining.append(part)
    label = " | ".join([*supports, *analyses, *policies, *remaining])
    return _replace_display_text(label)


def _selector_sort_key(raw_label: str, original_index: int) -> tuple[int, int, int, int]:
    parts = {part.strip().replace(" ", "_") for part in raw_label.split(" | ")}
    support = next((value for key, value in SUPPORT_ORDER.items() if key in parts), len(SUPPORT_ORDER))
    analysis = next((value for key, value in ANALYSIS_ORDER.items() if key in parts), len(ANALYSIS_ORDER))
    policy = next((value for key, value in POLICY_ORDER.items() if key in parts), len(POLICY_ORDER))
    return support, analysis, policy, original_index


def _replace_display_text(value: str) -> str:
    revised = value
    for old, new in DISPLAY_REPLACEMENTS.items():
        revised = revised.replace(old, new)
    return revised


def _replace_nested_display_text(value: Any) -> Any:
    if isinstance(value, str):
        return _replace_display_text(value)
    if isinstance(value, dict):
        return {key: _replace_nested_display_text(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_replace_nested_display_text(item) for item in value]
    if hasattr(value, "tolist"):
        return _replace_nested_display_text(value.tolist())
    return value


def _standardize_figure_text(figure: go.Figure) -> go.Figure:
    return go.Figure(_replace_nested_display_text(figure.to_plotly_json()))


def _rewrite_selector_labels(figure: go.Figure) -> go.Figure:
    title = str(figure.layout.title.text or "")
    for menu in figure.layout.updatemenus or ():
        buttons = list(menu.buttons or ())
        if not buttons or all(str(button.label) in CONTROL_LABELS for button in buttons):
            continue
        active_button = buttons[int(menu.active or 0)]
        indexed_buttons: list[tuple[str, Any, int]] = []
        for index, button in enumerate(buttons):
            raw_label = str(button.label)
            revised_label = _selector_label(raw_label)
            button.label = revised_label
            args = list(button.args or ())
            if len(args) > 1 and isinstance(args[1], dict):
                layout = dict(args[1])
                title_key = "title.text"
                if title_key in layout:
                    layout[title_key] = str(layout[title_key]).replace(raw_label, revised_label)
                args[1] = layout
                button.args = args
            title = title.replace(raw_label, revised_label)
            indexed_buttons.append((raw_label, button, index))
        ordered = sorted(indexed_buttons, key=lambda item: _selector_sort_key(item[0], item[2]))
        menu.buttons = [button for _, button, _ in ordered]
        menu.active = next(index for index, (_, button, _) in enumerate(ordered) if button is active_button)
    figure.update_layout(title_text=title)
    return figure


def _cohort_label(support: str, analysis_role: str, policy_role: str) -> str:
    return " | ".join([SUPPORT_LABELS[support], ANALYSIS_LABELS[analysis_role], POLICY_LABELS[policy_role]])


def _lr_decay_shape(*, xref: str, yref: str, show_label: bool) -> dict[str, Any]:
    shape: dict[str, Any] = {
        "type": "line",
        "xref": xref,
        "yref": yref,
        "x0": 0.80,
        "x1": 0.80,
        "y0": 0,
        "y1": 1,
        "line": {"color": LR_DECAY_COLOR, "width": 1.5, "dash": "dash"},
    }
    if show_label:
        shape["label"] = {
            "text": "LR decay begins · 0.80T",
            "textposition": "top left",
            "font": {"color": LR_DECAY_COLOR, "size": 12},
        }
    return shape


def _temporal_shapes(policy_role: str, states: list[str], count: int) -> list[dict[str, Any]]:
    shapes: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        suffix = "" if index == 1 else str(index)
        shapes.append(base._zero_shape(xref=f"x{suffix} domain", yref=f"y{suffix}"))
        shapes.append(
            base._phase_boundary_shape(
                policy_role,
                states,
                xref=f"x{suffix}",
                yref=f"y{suffix} domain",
                show_label=index == 1,
                scaled_time=True,
            )
        )
        if policy_role != "common_tied_035":
            shapes.append(_lr_decay_shape(xref=f"x{suffix}", yref=f"y{suffix} domain", show_label=index == 1))
    return shapes


def _add_source_trajectory_lr_reference(figure: go.Figure) -> go.Figure:
    for menu in figure.layout.updatemenus or ():
        for button in menu.buttons or ():
            if "boundary beta" not in str(button.label):
                continue
            args = list(button.args or ())
            if len(args) < 2 or not isinstance(args[1], dict):
                raise ValueError(f"Moved-switch selector lacks a layout update: {button.label}")
            layout = dict(args[1])
            shapes = list(layout.get("shapes", []))
            shapes.append(_lr_decay_shape(xref="x", yref="paper", show_label=True))
            layout["shapes"] = shapes
            args[1] = layout
            button.args = args
    return figure


def target_utility_trajectory_figure(frame: pd.DataFrame) -> go.Figure:
    selected = frame[frame["geometry"].eq("projected") & frame["component"].eq("trunk")].copy()
    summary = base._summarize_with_intervals(
        selected,
        ["analysis_role", "policy_role", "support_id", "target", "source", "checkpoint_label", "evidence_role"],
        "cosine",
    )
    cohorts = list(
        summary[["analysis_role", "policy_role", "support_id"]]
        .drop_duplicates()
        .sort_values(["support_id", "analysis_role", "policy_role"])
        .itertuples(index=False, name=None)
    )
    target_order = list(base.TARGET_LABELS)
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[base.TARGET_LABELS[target] for target in target_order],
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )
    cohort_indices: list[list[int]] = []
    cohort_orders: list[list[str]] = []
    for cohort_index, (analysis_role, policy_role, support) in enumerate(cohorts):
        indices: list[int] = []
        cohort_frame = summary[
            summary["analysis_role"].eq(analysis_role)
            & summary["policy_role"].eq(policy_role)
            & summary["support_id"].eq(support)
        ]
        state_order = base._checkpoint_order(str(policy_role), set(cohort_frame["checkpoint_label"].astype(str)))
        state_fractions = [base._state_fraction(str(policy_role), state) for state in state_order]
        cohort_orders.append(state_order)
        for target_index, target in enumerate(target_order):
            row = target_index // 2 + 1
            column = target_index % 2 + 1
            for source in base.SOURCE_LABELS:
                source_frame = cohort_frame[cohort_frame["target"].eq(target) & cohort_frame["source"].eq(source)]
                traces = base._band_traces(
                    source_frame,
                    x_order=state_order,
                    x_values=state_fractions,
                    label=base.SOURCE_LABELS[source],
                    color=base.SOURCE_COLORS[source],
                    visible=cohort_index == 0,
                    showlegend=target_index == 0,
                )
                for trace in traces:
                    indices.append(len(figure.data))
                    figure.add_trace(trace, row=row, col=column)
        cohort_indices.append(indices)

    buttons = []
    for (analysis_role, policy_role, support), indices, state_order in zip(
        cohorts, cohort_indices, cohort_orders, strict=True
    ):
        visible = [index in indices for index in range(len(figure.data))]
        label = _cohort_label(str(support), str(analysis_role), str(policy_role))
        layout_update: dict[str, Any] = {
            "title.text": f"Target-source utility through training<br><sup>{label}</sup>",
            "shapes": _temporal_shapes(str(policy_role), state_order, 4),
        }
        for axis_name in ("xaxis", "xaxis2", "xaxis3", "xaxis4"):
            layout_update[f"{axis_name}.type"] = "linear"
            layout_update[f"{axis_name}.range"] = base.FULL_TIMELINE_RANGE
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [{"visible": visible}, layout_update],
            }
        )
    for row in (1, 2):
        for column in (1, 2):
            figure.update_xaxes(
                title_text="Training progress (fraction of updates)",
                row=row,
                col=column,
                type="linear",
                range=base.FULL_TIMELINE_RANGE,
                tickformat=".3~f",
                ticksuffix="T",
            )
            figure.update_yaxes(title_text="Utility cosine", row=row, col=column)
    first_role, first_policy, first_support = cohorts[0]
    first_label = _cohort_label(str(first_support), str(first_role), str(first_policy))
    figure.update_layout(
        title={"text": f"Target-source utility through training<br><sup>{first_label}</sup>", "x": 0.03},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.11,
                "yanchor": "top",
                "showactive": True,
            },
            {
                "buttons": [
                    {
                        "label": "Full timeline",
                        "method": "relayout",
                        "args": [
                            {
                                axis: base.FULL_TIMELINE_RANGE
                                for axis in (
                                    "xaxis.range",
                                    "xaxis2.range",
                                    "xaxis3.range",
                                    "xaxis4.range",
                                )
                            }
                        ],
                    },
                    {
                        "label": "Zoom LR decay",
                        "method": "relayout",
                        "args": [
                            {
                                axis: base.LR_DECAY_ZOOM_RANGE
                                for axis in (
                                    "xaxis.range",
                                    "xaxis2.range",
                                    "xaxis3.range",
                                    "xaxis4.range",
                                )
                            }
                        ],
                    },
                ],
                "type": "buttons",
                "direction": "right",
                "x": 0.68,
                "xanchor": "center",
                "y": 1.11,
                "yanchor": "top",
                "showactive": True,
            },
        ],
        height=1080,
        margin={"l": 85, "r": 45, "t": 165, "b": 245},
        paper_bgcolor="#fbf8ef",
        plot_bgcolor="#fbf8ef",
        font={"family": "Avenir Next, Avenir, sans-serif", "color": "#183149"},
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.05, "yanchor": "bottom"},
        shapes=_temporal_shapes(str(first_policy), cohort_orders[0], 4),
        annotations=[
            *figure.layout.annotations,
            base._footnote(
                (
                    "Horizontal position is the actual fraction of training updates; use Zoom LR decay to separate "
                    "the closely spaced probes. Positive cosine means the counterfactual source update locally "
                    "reduces target loss. Bands are pointwise seed-bootstrap 95% intervals. The dark solid bar "
                    "marks the data switch for moved-switch policies or the 0.80T LR-decay onset for tied policies; "
                    "moved-switch policies also show the fixed LR-decay onset as a slate dashed bar. "
                    "Final target-update cosine is undefined because LR=0. "
                    f"{base.PRIMARY_PATH_NOTE}"
                ),
                y=-0.31,
            ),
        ],
    )
    return figure


def _state_explanation(part: str) -> str | None:
    if re.fullmatch(r"0\.\d+T", part):
        percent = float(part.removesuffix("T")) * 100
        return f"The restored checkpoint is at {percent:g}% of the training updates."
    if part == "LR decay onset":
        return "The restored checkpoint is exactly at the fixed 0.80T learning-rate-decay onset."
    match = re.fullmatch(r"LR decay ([+-]) (\d+)", part)
    if match:
        relation = "after" if match.group(1) == "+" else "before"
        return f"The restored checkpoint is {match.group(2)} updates {relation} the 0.80T decay onset."
    if part == "data switch":
        return "The restored checkpoint is exactly at this policy's data switch."
    match = re.fullmatch(r"data switch ([+-]) (\d+)", part)
    if match:
        relation = "after" if match.group(1) == "+" else "before"
        return f"The restored checkpoint is {match.group(2)} updates {relation} this policy's data switch."
    return None


def _selected_setting_explanation(filename: str, label: str) -> str:
    explanations: list[str] = []
    recognized: set[str] = set()
    parts = [part.strip() for part in label.split(" | ")]
    if filename.startswith("source_source_"):
        if "Tied 35% StarCoder" in parts:
            support = next(part for part in parts if part in SOURCE_TIED_ANALYSIS_EXPLANATIONS)
            explanations.append(SOURCE_TIED_ANALYSIS_EXPLANATIONS[support])
        else:
            explanations.append(ANALYSIS_EXPLANATIONS["H5 moved-switch profile"])
    for part in parts:
        explanation = SUPPORT_EXPLANATIONS.get(part)
        if explanation is not None:
            explanations.append(explanation)
            recognized.add(part)
            continue
        explanation = ANALYSIS_EXPLANATIONS.get(part)
        if explanation is not None:
            explanations.append(explanation)
            recognized.add(part)
            continue
        explanation = POLICY_EXPLANATIONS.get(part)
        if explanation is not None:
            explanations.append(explanation)
            recognized.add(part)
            continue
        explanation = _state_explanation(part)
        if explanation is not None:
            explanations.append(explanation)
            recognized.add(part)
            continue
        explanation = EVIDENCE_EXPLANATIONS.get(part)
        if explanation is not None:
            explanations.append(explanation)
            recognized.add(part)
            continue
        explanation = SOURCE_CONTRAST_EXPLANATIONS.get(part)
        if explanation is not None:
            explanations.append(explanation)
            recognized.add(part)
    unrecognized = [part for part in parts if part not in recognized]
    if unrecognized:
        raise ValueError(f"No contextual explanation for selector components {unrecognized} in {filename}")
    return " ".join(explanations)


def _selector_explanations(filename: str, figure: go.Figure) -> tuple[str, dict[str, str]] | None:
    selector_menus = [
        menu
        for menu in figure.layout.updatemenus or ()
        if menu.buttons and not all(str(button.label) in CONTROL_LABELS for button in menu.buttons)
    ]
    if not selector_menus:
        return None
    if len(selector_menus) != 1:
        raise ValueError(f"Expected one scientific selector in {filename}, found {len(selector_menus)}")
    menu = selector_menus[0]
    buttons = list(menu.buttons or ())
    explanations = {str(button.label): _selected_setting_explanation(filename, str(button.label)) for button in buttons}
    active_label = str(buttons[int(menu.active or 0)].label)
    return active_label, explanations


def _panel_document(
    filename: str,
    figure: go.Figure,
    supplement: tuple[str, str, go.Figure, str] | None = None,
) -> str:
    title, explanation = PANEL_GUIDES[filename]
    div_id = Path(filename).stem
    fragment = pio.to_html(
        figure,
        include_plotlyjs=True,
        full_html=False,
        config=base.PLOT_CONFIG,
        div_id=div_id,
    )
    selection = _selector_explanations(filename, figure)
    selection_context = ""
    selection_script = ""
    supplement_html = ""
    supplement_script = ""
    if selection is not None:
        active_label, explanations = selection
        context_id = f"{div_id}-selection-context"
        selection_context = (
            '<aside class="selection-context"><span>Selected setting</span>'
            f'<p id="{context_id}">{html.escape(explanations[active_label])}</p></aside>'
        )
        selection_script = f"""<script>
const plot = document.getElementById({json.dumps(div_id)});
const context = document.getElementById({json.dumps(context_id)});
const explanations = {json.dumps(explanations, sort_keys=True)};
plot.on("plotly_buttonclicked", event => {{
  const label = event && event.button ? String(event.button.label) : "";
  if (Object.hasOwn(explanations, label)) context.textContent = explanations[label];
}});
</script>"""
    if supplement is not None:
        supplement_title, supplement_description, supplement_figure, norm_axis_mode = supplement
        supplement_div_id = f"{div_id}-fixed-n-tpp"
        supplement_fragment = pio.to_html(
            supplement_figure,
            include_plotlyjs=False,
            full_html=False,
            config=base.PLOT_CONFIG,
            div_id=supplement_div_id,
        )
        if norm_axis_mode not in {"primary", "secondary"}:
            raise ValueError(f"Unknown norm-axis mode {norm_axis_mode}")
        raw_toggle_id = f"{supplement_div_id}-raw-norm-toggle"
        update_toggle_id = f"{supplement_div_id}-update-norm-toggle"
        slider_labels = [str(step.label) for step in supplement_figure.layout.sliders[0].steps]
        supplement_controls = (
            '<div class="supplement-controls"><label class="norm-toggle">'
            f'<input id="{raw_toggle_id}" type="checkbox">'
            "<span>Show raw-gradient norms</span></label>"
            '<label class="norm-toggle">'
            f'<input id="{update_toggle_id}" type="checkbox">'
            "<span>Show optimizer-update norms</span></label></div>"
        )
        if norm_axis_mode == "secondary":
            norm_axis_script = 'Plotly.relayout(normPlot, {"yaxis2.visible": anyNormVisible});'
        else:
            norm_axis_script = """const primaryRange = anyNormVisible ? [0, 2] : [0, 1];
  const primaryTitle = anyNormVisible ? "Raw cosine / norm relative to 0.80T" : "Raw gradient cosine";
  Plotly.relayout(normPlot, {
    "yaxis.range": primaryRange,
    "yaxis3.range": primaryRange,
    "yaxis5.range": primaryRange,
    "yaxis7.range": primaryRange,
    "yaxis.title.text": primaryTitle,
    "yaxis5.title.text": primaryTitle
  });"""
        supplement_script = f"""<script>
const normPlot = document.getElementById({json.dumps(supplement_div_id)});
const rawNormToggle = document.getElementById({json.dumps(raw_toggle_id)});
const updateNormToggle = document.getElementById({json.dumps(update_toggle_id)});
const normSliderLabels = {json.dumps(slider_labels)};
let activeNormCell = 0;
function applyNormVisibility() {{
  const visible = normPlot.data.map(trace => {{
    const metadata = trace.meta || {{}};
    if (Number(metadata.cell_index) !== activeNormCell) return false;
    if (metadata.visibility_group === "cosine") return true;
    if (metadata.visibility_group === "raw_norm") return rawNormToggle.checked;
    if (metadata.visibility_group === "update_norm") return updateNormToggle.checked;
    return false;
  }});
  Plotly.restyle(normPlot, "visible", visible);
  const anyNormVisible = rawNormToggle.checked || updateNormToggle.checked;
  {norm_axis_script}
}}
rawNormToggle.addEventListener("change", applyNormVisibility);
updateNormToggle.addEventListener("change", applyNormVisibility);
normPlot.on("plotly_sliderchange", event => {{
  const label = event && event.step ? String(event.step.label) : "";
  const index = normSliderLabels.indexOf(label);
  if (index >= 0) activeNormCell = index;
  window.setTimeout(applyNormVisibility, 0);
}});
</script>"""
        supplement_html = (
            '<section class="supplement"><div class="supplement-head">'
            "<span>Fixed-N TPP extension</span>"
            f"<h2>{html.escape(supplement_title)}</h2>"
            f"<p>{html.escape(supplement_description)}</p>{supplement_controls}</div>"
            f'<div class="supplement-plot">{supplement_fragment}</div></section>'
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root{{--ink:#183149;--paper:#fbf8ef;--line:#d8cfbd;--teal:#1b7f79;--orange:#d65a31}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:"Avenir Next",Avenir,sans-serif}}
.guide{{max-width:1500px;margin:0 auto;padding:32px 42px 0}}
.guide a{{color:var(--teal);font-weight:700;text-decoration:none}}
.guide h1{{font-family:Georgia,serif;font-size:38px;margin:12px 0 8px}}
.guide>p{{font-size:17px;line-height:1.55;max-width:1100px}}
.selection-context{{
  margin-top:20px;max-width:1180px;border-left:5px solid var(--orange);padding:14px 18px;background:#fffdf7
}}
.selection-context span{{
  display:block;color:#a44729;font-size:12px;font-weight:800;letter-spacing:.12em;text-transform:uppercase
}}
.selection-context p{{font-size:17px;line-height:1.5;margin:5px 0 0}}
.supplement{{max-width:1500px;margin:34px auto 64px;background:#fffdf7;border:1px solid var(--line)}}
.supplement-head{{padding:28px 34px 0}}
.supplement-head span{{color:#a44729;font-size:12px;font-weight:800;letter-spacing:.12em;text-transform:uppercase}}
.supplement-head h2{{font-family:Georgia,serif;font-size:31px;margin:7px 0 8px}}
.supplement-head p{{font-size:16px;line-height:1.55;max-width:1120px;color:#536674}}
.supplement-controls{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:18px 0 2px}}
.norm-toggle{{
  display:inline-flex;align-items:center;gap:10px;padding:9px 13px;border:1px solid var(--line);
  background:#fbf8ef;font-weight:700;cursor:pointer
}}
.norm-toggle input{{width:18px;height:18px;accent-color:var(--teal)}}
.supplement-plot{{width:100%;min-width:0;overflow:hidden}}
.supplement-plot .js-plotly-plot,.supplement-plot .plot-container,.supplement-plot .svg-container{{width:100%!important}}
@media(max-width:780px){{.guide{{padding:24px 18px 0}}}}
</style></head><body><header class="guide"><a href="index.html">Back to report</a><h1>{html.escape(title)}</h1>
<p>{html.escape(explanation)}</p>{selection_context}</header>{fragment}{supplement_html}{selection_script}{supplement_script}</body></html>"""


def _revised_index(
    base_index: str,
    fixed_summary: Mapping[str, Any],
    lr_onset_summary: Mapping[str, Any],
) -> str:
    old_support = (
        '<article class="explain"><h3>Two support regimes</h3><p><strong>Full</strong> uses the broad source pool.\n'
        "\t\t<strong>m100a</strong> repeatedly draws from a fixed finite support, "
        "creating simulated repetition.</p></article>"
    )
    new_support = (
        '<article class="explain"><h3>Three support regimes</h3><p><strong>Full</strong> uses the broad source pool. '
        "<strong>m100a</strong> is the primary finite repeated support. <strong>m100b</strong> is a disjoint finite "
        "support used only as a sensitivity check.</p></article>"
    )
    if old_support not in base_index:
        raise ValueError("Base index support explanation changed unexpectedly")
    revised = base_index.replace(old_support, new_support)
    old_verdict = (
        "Two-phase scheduling has real endpoint effects in this setting, but the\n"
        "\tdata do not support broad, increasing gradient conflict as their cause. During every decision-relevant "
        "checkpoint,\n\tStarCoder and Nemotron updates remain positively aligned and become slightly more aligned. "
        "Target preferences are\n\tstable, the registered repetition interaction is null, and the endpoint arms lack "
        "mediator measurements. Gradient\n\talignment remains a candidate local feature, not an established mechanism."
    )
    new_verdict = (
        "Two-phase scheduling has real endpoint effects in this setting, but the evidence does not establish broad "
        "gradient conflict as their cause. A randomized LR-schedule intervention shows that LR decay causes the late "
        "raw-gradient divergence: no decay remains positively aligned while earlier decay drives StarCoder and "
        "Nemotron gradients toward anti-alignment as their norms collapse. Optimizer-update alignment does not show "
        "the same conflict while the update is "
        "nonzero. Gradient alignment therefore remains a descriptive state feature, not an established endpoint "
        "mediator or optimizer target."
    )
    if old_verdict not in revised:
        raise ValueError("Base index verdict changed unexpectedly")
    revised = revised.replace(old_verdict, new_verdict)
    old_h1 = (
        '<tr><th scope="row"><span class="tag">H1</span></th><td>Do StarCoder and Nemotron updates become '
        "increasingly conflicting late?</td><td><strong>No actionable late conflict trend (descriptive, tied-0.35 "
        "subset)</strong></td><td>On the 56 tied-0.35 trajectories with H1 rows, full-support projected-trunk raw "
        "cosine is 0.471 at 0.10T and 0.546 at decay onset, then -0.301 only at the terminal zero-LR state. "
        "Optimizer-update cosine rises from 0.446 to 0.471 and is undefined when LR is zero.</td></tr>"
    )
    primary = fixed_summary["primary"]
    decline_range = (min(row["decline_mean"] for row in primary), max(row["decline_mean"] for row in primary))
    new_h1 = (
        '<tr><th scope="row"><span class="tag">H1</span></th><td>Do StarCoder and Nemotron updates become '
        "increasingly conflicting late?</td><td><strong>LR decay causes raw-gradient divergence, not demonstrated "
        "optimizer-update conflict</strong></td><td>The fixed-N extension finds a 0.55T/0.70T-to-0.90T finite-batch "
        f"cosine decline of {decline_range[0]:.3f} to {decline_range[1]:.3f} in all 32 seed-rung pairs. The causal "
        "intervention then moves LR-decay onset across 0.60T, 0.80T, 0.90T, or never: the raw-gradient decline follows "
        "the schedule, while nonzero optimizer updates remain positively aligned. This rules out TPP as the sole "
        "clock but does not connect the rotation to endpoint gain.</td></tr>"
    )
    if old_h1 not in revised:
        raise ValueError("Base index H1 row changed unexpectedly")
    revised = revised.replace(old_h1, new_h1)
    old_role = (
        "Broad, growing StarCoder-versus-Nemotron conflict is not supported as the driver of the two-phase gain. A "
        "local, target- and state-specific source-selection signal remains possible, but was not validated here."
    )
    new_role = (
        "LR decay causally changes raw source-gradient geometry as source norms collapse, but the corresponding "
        "nonzero optimizer updates stay "
        "positively aligned. Without endpoint-arm mediation, that rotation does not explain the two-phase gain."
    )
    if old_role not in revised:
        raise ValueError("Base index interpretation changed unexpectedly")
    revised = revised.replace(old_role, new_role)
    replacements = {
        "A dark vertical bar marks the beginning\n\tof phase 2 in every panel whose horizontal axis is training time.": (
            "For tied policies, the dark 0.80T bar marks LR-decay onset; the data mixture does not change. For "
            "moved-switch policies, the dark bar marks the data switch and a slate dashed bar separately marks the "
            "fixed 0.80T LR-decay onset."
        ),
        "with pointwise intervals and the phase boundary.": (
            "with pointwise intervals and policy-specific data-switch or LR-decay markers."
        ),
        "over training, with the phase boundary.": (
            "over training, with policy-specific data-switch or LR-decay markers."
        ),
        "through training; the bar marks phase 2.": (
            "through training, with policy-specific data-switch and LR-decay markers."
        ),
        "through time, with the phase boundary.": "through time, with policy-specific data-switch or LR-decay markers.",
    }
    for old, new in replacements.items():
        if old not in revised:
            raise ValueError(f"Base index presentation text changed unexpectedly: {old}")
        revised = revised.replace(old, new)
    diagnostics = fixed_summary["trend_diagnostics"]
    fixed_section = f"""
\t<section class="section"><div class="section-head"><span class="section-number">6 / Fixed-N extension</span><div>
\t<h2>Higher TPP does not identify the divergence clock</h2><p class="section-intro">A separate four-rung ladder holds
\tmodel size fixed while increasing training tokens. The same frozen panels measure both unprojected raw gradients and
\toptimizer-aware updates at eight restored states.</p></div></div>
\t<div class="endpoint-grid"><article class="endpoint"><h3>Measured late decline</h3>
\t<div class="value">32 / 32 seeds</div>
\t<p>Every seed-rung pair has lower finite-batch source-gradient cosine at 0.90T than its 0.55T/0.70T
\tplateau.</p></article>
\t<article class="endpoint"><h3>Highest minus lowest TPP</h3>
\t<div class="value">{diagnostics['r3_minus_r0_decline_mean']:+.4f}</div>
\t<p>95% CI [{diagnostics['r3_minus_r0_decline_bootstrap_ci95_low']:+.3f},
\t{diagnostics['r3_minus_r0_decline_bootstrap_ci95_high']:+.3f}]. The panel is too coarse and underpowered
\tto identify a TPP-controlled onset.</p></article>
\t<article class="endpoint"><h3>Measurement boundary</h3><div class="value">39&ndash;53%</div>
\t<p>Mean-gradient norms fall over the same interval. The high-TPP result survives doubling probe precision,
\tbut lower-rung attenuation remains unresolved.</p></article></div>
\t<div class="equation"><a href="fixed_n_tpp_gradient_onset.html">Open the responsive fixed-N analysis</a>.
\tThe source and target trajectory pages also include TPP sliders, raw-gradient cosines, optimizer-update
\tgeometry, and pointwise intervals.</div></section>
"""
    lr_primary = lr_onset_summary["primary"]
    lr_raw = lr_onset_summary["primary_sensitivities"]["uncorrected_projected_gradient_cosine"]
    lr_reliability = lr_onset_summary["reliability"]
    lr_section = f"""
\t<section class="section"><div class="section-head"><span class="section-number">7 / LR intervention</span><div>
\t<h2>The decline follows learning-rate decay</h2><p class="section-intro">A 32-trajectory randomized intervention
\tholds model, data mixture, token horizon, optimizer, and eight paired seeds fixed while moving cosine-decay onset
\tto 0.60T, 0.80T, 0.90T, or removing decay.</p></div></div>
\t<div class="endpoint-grid"><article class="endpoint"><h3>Uncorrected final contrast</h3>
\t<div class="value">{lr_raw['mean_difference']:+.3f}</div>
\t<p>No-decay minus 0.60T-decay raw-gradient cosine; 95% CI
\t[{lr_raw['bootstrap_ci95_low']:+.3f}, {lr_raw['bootstrap_ci95_high']:+.3f}], with
\t{lr_raw['positive_pairs']}/8 paired effects positive.</p></article>
\t<article class="endpoint"><h3>Frozen primary</h3><div class="value">{lr_primary['mean_difference']:+.3f}</div>
\t<p>The split-half noise-corrected contrast is larger, but
\t{lr_reliability['primary_decay_rows_below_threshold']}/8 decay-arm
\tendpoints have sub-threshold Nemotron reliability, so its confirmatory status is formally inconclusive.</p></article>
\t<article class="endpoint"><h3>Mechanistic boundary</h3><div class="value">Raw, not update</div>
\t<p>The divergence follows LR decay, but optimizer-update cosine stays positive while defined. No endpoint benchmark
\twas read, so this does not establish a two-phase performance mechanism.</p></article></div>
\t<div class="equation"><a href="lr_onset_gradient_causality.html">Open the LR-onset intervention</a>.
\tThis separates the frozen split-half noise-corrected primary, its uncorrected sensitivity, and the
\toptimizer-update secondary
\tfamily.</div></section>
"""
    evidence_marker = (
        '<section class="section"><div class="section-head"><span class="section-number">6 / Evidence</span>'
    )
    if evidence_marker not in revised:
        raise ValueError("Base index evidence section changed unexpectedly")
    revised = revised.replace(
        evidence_marker,
        fixed_section
        + lr_section
        + '\n\t<section class="section"><div class="section-head"><span class="section-number">8 / Evidence</span>',
    )
    scientific_marker = '<span class="section-number">7 / Scientific status</span>'
    if scientific_marker not in revised:
        raise ValueError("Base index scientific-status section changed unexpectedly")
    revised = revised.replace(scientific_marker, '<span class="section-number">9 / Scientific status</span>')
    mechanism_card = (
        '<a class="card" href="mechanism_effect_forest.html"><h2>Mechanism contrasts</h2><p>The 47 post-outcome '
        "H2/H3/H5 development contrasts and multiplicity audit.</p></a>"
    )
    fixed_card = (
        '<a class="card" href="fixed_n_tpp_gradient_onset.html"><h2>Fixed-N TPP extension</h2><p>Responsive scaling '
        "summary with raw-gradient, optimizer-update, target-utility, precision, and onset diagnostics.</p></a>"
    )
    lr_card = (
        '<a class="card" href="lr_onset_gradient_causality.html"><h2>LR-onset intervention</h2><p>Paired causal '
        "test of whether late raw-gradient rotation follows learning-rate decay.</p></a>"
    )
    if mechanism_card not in revised:
        raise ValueError("Base index evidence-card inventory changed unexpectedly")
    revised = revised.replace(mechanism_card, fixed_card + lr_card + mechanism_card)
    return revised


def _assert_selector_inventory(source: pd.DataFrame, utilities: pd.DataFrame, alignment: pd.DataFrame) -> None:
    combined = pd.concat(
        [
            source[["support_id", "analysis_role", "policy_role"]],
            utilities[["support_id", "analysis_role", "policy_role"]],
            alignment[["support_id", "analysis_role", "policy_role"]],
        ],
        ignore_index=True,
    )
    if frozenset(combined["support_id"]) != EXPECTED_SUPPORTS:
        raise ValueError("Support selector inventory changed")
    if frozenset(combined["analysis_role"]) != EXPECTED_ANALYSES:
        raise ValueError("Analysis-role selector inventory changed")
    if frozenset(combined["policy_role"]) != EXPECTED_POLICIES:
        raise ValueError("Policy selector inventory changed")


def render(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    release = json.loads(args.release.read_text())
    plot_module = Path(base.__file__).resolve()
    plot_key = str(release["materialization"]["plot_module"])
    if release["implementation_files"][plot_key] != _sha256(plot_module):
        raise ValueError("Audited base plot module drifted from the completion release")

    source = pd.read_csv(args.source_geometry, low_memory=False)
    frozen_source = pd.read_csv(args.input_dir / "source_source_geometry.csv", low_memory=False)
    utilities = pd.read_csv(args.input_dir / "target_source_utilities_visualization_only.csv", low_memory=False)
    alignment = pd.read_csv(args.input_dir / "target_source_choice_alignment_visualization_only.csv", low_memory=False)
    summaries = [
        pd.read_csv(args.input_dir / name, low_memory=False)
        for name in ("h2_h3_summary.csv", "h3_repetition_mechanism_summary.csv", "h5_profile_summary.csv")
    ]
    multiplicity = pd.read_csv(args.multiplicity_audit, low_memory=False)
    fixed_summary = json.loads((args.fixed_n_tpp_results_dir / "analysis_summary.json").read_text())
    fixed_source = pd.read_csv(args.fixed_n_tpp_results_dir / "source_measurements.csv", low_memory=False)
    fixed_target_gradients = pd.read_csv(
        args.fixed_n_tpp_results_dir / "target_source_gradient_measurements.csv",
        low_memory=False,
    )
    fixed_target_utilities = pd.read_csv(
        args.fixed_n_tpp_results_dir / "target_source_utility_measurements.csv",
        low_memory=False,
    )
    lr_onset_summary = json.loads((args.lr_onset_results_dir / "analysis_summary.json").read_text())
    if fixed_summary["row_count"] != 256 or fixed_summary["utility_row_count"] != 2_048:
        raise ValueError("Fixed-N TPP analysis inventory changed")
    if tuple(fixed_summary["primary"][index]["cell_id"] for index in range(4)) != FIXED_TPP_CELL_ORDER:
        raise ValueError("Fixed-N TPP cell order changed")
    if len(fixed_source) != 256 or len(fixed_target_gradients) != 2_048 or len(fixed_target_utilities) != 2_048:
        raise ValueError("Fixed-N TPP measurement tables are incomplete")
    if lr_onset_summary["row_count"] != 192 or lr_onset_summary["endpoint_metrics_read"] is not False:
        raise ValueError("LR-onset analysis inventory or endpoint-blindness marker changed")
    _assert_selector_inventory(source, utilities, alignment)

    raw_figures = {
        "source_source_conflict_trajectory.html": _rewrite_selector_labels(
            _add_source_trajectory_lr_reference(base.source_conflict_trajectory_figure(source, frozen_source))
        ),
        "source_source_conflict_matrix.html": _rewrite_selector_labels(
            base.source_conflict_figure(source, frozen_source)
        ),
        "target_source_utility_trajectories.html": target_utility_trajectory_figure(utilities),
        "target_source_utility_matrix.html": _rewrite_selector_labels(base.target_utility_figure(utilities)),
        "target_source_choice_alignment.html": _rewrite_selector_labels(base.source_choice_figure(alignment)),
        "mechanism_effect_forest.html": base.mechanism_forest_figure(summaries, multiplicity),
    }
    figures = {filename: _standardize_figure_text(figure) for filename, figure in raw_figures.items()}
    supplements = {
        "source_source_conflict_trajectory.html": (
            "Compare source alignment across TPP",
            "Move the slider across the four fixed-model-size token horizons. The solid teal line is the unprojected "
            "raw-gradient cosine; orange is the exact optimizer-aware update cosine from the same frozen panels. "
            "The optional norm overlay uses each seed's 0.80T value as 1x so raw gradients and optimizer updates can "
            "share a readable scale; the two vector families have independent controls and hover reports the "
            "absolute norm.",
            fixed_tpp_source_trajectory_figure(fixed_source),
            "secondary",
        ),
        "target_source_utility_trajectories.html": (
            "Compare target alignment across TPP",
            "Move the slider across token horizons. Solid circle traces use unprojected target-source raw-gradient "
            "cosine on the left axes; dashed diamond traces use optimizer-aware utility cosine on the right axes. "
            "Independent norm controls add the target gradient, both source gradients, or both optimizer updates; "
            "each norm is paired within seed and shown relative to its own 0.80T value.",
            fixed_tpp_target_trajectory_figure(fixed_target_gradients, fixed_target_utilities),
            "primary",
        ),
    }
    target_trajectory = figures["target_source_utility_trajectories.html"]
    for axis_name in ("xaxis", "xaxis2", "xaxis3", "xaxis4"):
        axis = getattr(target_trajectory.layout, axis_name)
        if axis.type != "linear" or list(axis.range) != base.FULL_TIMELINE_RANGE:
            raise ValueError(f"Target-source trajectory axis {axis_name} is not on true training time")

    for filename, figure in figures.items():
        document = _panel_document(filename, figure, supplements.get(filename))
        if filename != "mechanism_effect_forest.html" and "Selected setting" not in document:
            raise ValueError(f"Contextual selector explanation missing from {filename}")
        _write_create_only(args.output_dir / filename, document)

    base_index_path = args.base_plots_dir / "index.html"
    index = _revised_index(base_index_path.read_text(), fixed_summary, lr_onset_summary)
    _write_create_only(args.output_dir / "index.html", index)
    fixed_page = (args.fixed_n_tpp_results_dir / "fixed_n_tpp_gradient_onset.html").read_text()
    back_placeholder = "<!-- STUDY_BACK_LINK -->"
    if fixed_page.count(back_placeholder) != 1:
        raise ValueError("Fixed-N TPP dashboard back-link placeholder changed")
    fixed_page = fixed_page.replace(back_placeholder, '<a href="index.html">Back to mechanistic study</a>')
    _write_create_only(args.output_dir / "fixed_n_tpp_gradient_onset.html", fixed_page)
    lr_page = (args.lr_onset_results_dir / "lr_onset_gradient_causality.html").read_text()
    if lr_page.count(back_placeholder) != 1:
        raise ValueError("LR-onset dashboard back-link placeholder changed")
    lr_page = lr_page.replace(back_placeholder, '<a href="index.html">Back to mechanistic study</a>')
    _write_create_only(args.output_dir / "lr_onset_gradient_causality.html", lr_page)

    expected = {"index.html", "fixed_n_tpp_gradient_onset.html", "lr_onset_gradient_causality.html", *PANEL_GUIDES}
    observed = {path.name for path in args.output_dir.glob("*.html")}
    if observed != expected:
        raise ValueError(f"Rendered HTML inventory drifted: {sorted(observed)} != {sorted(expected)}")
    input_paths = {
        "release": args.release,
        "source_geometry": args.source_geometry,
        "multiplicity_audit": args.multiplicity_audit,
        "base_index": base_index_path,
        **{
            f"fixed_n_tpp:{name}": args.fixed_n_tpp_results_dir / name
            for name in (
                "analysis_summary.json",
                "fixed_n_tpp_gradient_onset.html",
                "report.md",
                "source_measurements.csv",
                "target_source_gradient_measurements.csv",
                "target_source_utility_measurements.csv",
            )
        },
        **{
            f"lr_onset:{name}": args.lr_onset_results_dir / name
            for name in (
                "analysis_summary.json",
                "lr_onset_gradient_causality.html",
                "report.md",
                "trajectory_summary.csv",
                "primary_paired_effects.csv",
                "frozen_optimizer_update_secondary.csv",
                "postfreeze_raw_gradient_time_axis_sensitivity.csv",
            )
        },
        **{f"table:{path.name}": path for path in sorted(args.input_dir.glob("*.csv"))},
    }
    manifest = {
        "presentation_revision": "2026-08-23-lr-onset-causal-extension-v23",
        "render_command": (
            "uv run python experiments/domain_phase_mix/exploratory/two_phase_many/"
            "render_starcoder_wsd80_gradient_mechanism_self_contained_20260822.py "
            f"--input-dir {args.input_dir} --source-geometry {args.source_geometry} "
            f"--multiplicity-audit {args.multiplicity_audit} --release {args.release} "
            f"--base-plots-dir {args.base_plots_dir} --fixed-n-tpp-results-dir {args.fixed_n_tpp_results_dir} "
            f"--lr-onset-results-dir {args.lr_onset_results_dir} "
            f"--output-dir {args.output_dir}"
        ),
        "scientific_release_sha256": release["release_sha256"],
        "base_plot_module_sha256": _sha256(plot_module),
        "presentation_renderer_sha256": _sha256(Path(__file__)),
        "inputs": {name: _sha256(path) for name, path in sorted(input_paths.items())},
        "files": {name: _sha256(args.output_dir / name) for name in sorted(expected)},
        "scientific_values_changed": False,
        "scientific_extension": {
            "fixed_n_tpp_release_sha256": fixed_summary["release_sha256"],
            "fixed_n_tpp_audit_sha256": fixed_summary["audit_sha256"],
            "fixed_n_tpp_row_count": fixed_summary["row_count"],
            "lr_onset_release_sha256": lr_onset_summary["release_sha256"],
            "lr_onset_row_count": lr_onset_summary["row_count"],
        },
        "presentation_changes": [
            "contextual explanation for every selectable scientific setting",
            "human-readable selector labels",
            "selectors are grouped by source pool and ordered tied before moved-switch policies",
            "target-source trajectories use actual training fractions",
            "moved-switch trajectories distinguish data-switch and LR-decay reference bars",
            "responsive fixed-N TPP analysis integrated into the report",
            "TPP sliders added to source and target trajectory pages",
            "fixed-N trajectory supplements expose unprojected raw-gradient cosines",
            "fixed-N source and target trajectories add independently toggled raw-gradient and optimizer-update "
            "norm overlays normalized within seed at LR-decay onset",
            "paired LR-decay-onset causal intervention integrated with confirmatory reliability caveat",
        ],
    }
    _write_create_only(args.output_dir / "render_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-geometry", type=Path, required=True)
    parser.add_argument("--multiplicity-audit", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--base-plots-dir", type=Path, required=True)
    parser.add_argument("--fixed-n-tpp-results-dir", type=Path, required=True)
    parser.add_argument("--lr-onset-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
