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
"""Render the frozen gradient-mechanism repair tables without changing their estimands."""

import argparse
import hashlib
import html
import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.basedatatypes import BaseTraceType
from plotly.subplots import make_subplots

PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}
BOOTSTRAP_DRAWS = 50_000
BOOTSTRAP_SEED = 2_026_082_101
COMMON_TOTAL_UPDATES = 28_260
H5_TOTAL_UPDATES = 28_160
FULL_TIMELINE_RANGE = [0.075, 1.025]
LR_DECAY_ZOOM_RANGE = [0.785, 0.815]
H1_STATE_ORDER = ["fraction_0p10", "fraction_0p25", "fraction_0p70", "decay_onset", "final"]
COMMON_STATE_ORDER = [
    "fraction_0p10",
    "fraction_0p25",
    "fraction_0p40",
    "fraction_0p55",
    "fraction_0p70",
    "decay_minus_256",
    "decay_minus_64",
    "decay_onset",
    "decay_plus_64",
    "decay_plus_256",
    "fraction_0p90",
    "final",
]
H5_STATE_ORDER = {
    "boundary_beta_0p60": [
        "fraction_0p40",
        "fraction_0p55",
        "data_switch_minus_64",
        "data_switch",
        "data_switch_plus_64",
        "optimizer_decay_minus_256",
        "optimizer_decay_minus_64",
        "optimizer_decay_onset",
        "optimizer_decay_plus_64",
        "fraction_0p90",
        "final",
    ],
    "boundary_beta_0p85": [
        "fraction_0p40",
        "fraction_0p55",
        "optimizer_decay_minus_256",
        "optimizer_decay_minus_64",
        "optimizer_decay_onset",
        "optimizer_decay_plus_64",
        "data_switch_minus_64",
        "data_switch",
        "data_switch_plus_64",
        "fraction_0p90",
        "final",
    ],
}
STATE_LABELS = {
    "fraction_0p10": "0.10T",
    "fraction_0p25": "0.25T",
    "fraction_0p40": "0.40T",
    "fraction_0p55": "0.55T",
    "fraction_0p70": "0.70T",
    "fraction_0p90": "0.90T",
    "decay_minus_256": "LR decay - 256",
    "decay_minus_64": "LR decay - 64",
    "decay_onset": "LR decay onset",
    "decay_plus_64": "LR decay + 64",
    "decay_plus_256": "LR decay + 256",
    "optimizer_decay_minus_256": "LR decay - 256",
    "optimizer_decay_minus_64": "LR decay - 64",
    "optimizer_decay_onset": "LR decay onset",
    "optimizer_decay_plus_64": "LR decay + 64",
    "data_switch_minus_64": "policy switch - 64",
    "data_switch": "policy switch",
    "data_switch_plus_64": "policy switch + 64",
    "final": "final",
}
PRIMARY_PATH_NOTE = (
    "Projected trunk was named primary in the design prose, but the numerical repair contract omitted the "
    "geometry/component fields; component-path sensitivity is descriptive."
)
TARGET_LABELS = {
    "paloma_programming_languages": "Programming Languages",
    "paloma_c4_en": "C4",
    "uncheatable_github_python": "GitHub Python",
    "uncheatable_wikipedia_english": "Wikipedia",
}
SOURCE_LABELS = {
    "starcoder_excluded_global": "StarCoder heldout",
    "starcoder_support_reference": "StarCoder support",
    "nemotron_aggregate": "Nemotron",
}
STATISTIC_LABELS = {"gradient": "Raw gradient", "optimizer_update": "Optimizer update"}
SOURCE_COLORS = {
    "starcoder_excluded_global": "#d65a31",
    "starcoder_support_reference": "#d9a21b",
    "nemotron_aggregate": "#1b7f79",
}
STATISTIC_COLORS = {"gradient": "#1b7f79", "optimizer_update": "#d65a31"}
PHASE_BOUNDARY_COLOR = "#102f43"
EVIDENCE_LABELS = {
    "v10 H1 contract state": "v10 H1 contract state",
    "Post-outcome v10 repair state": "post-outcome v10 state",
    "Post-outcome plot completion": "post-outcome completion",
}
REQUIRED_FILES = (
    "source_source_geometry.csv",
    "target_source_utilities_visualization_only.csv",
    "target_source_choice_alignment_visualization_only.csv",
    "h2_h3_summary.csv",
    "h3_repetition_mechanism_summary.csv",
    "h5_profile_summary.csv",
)


def _require_columns(frame: pd.DataFrame, columns: set[str], *, source: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _read_table(input_dir: Path, name: str, columns: set[str]) -> pd.DataFrame:
    path = input_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing analyzer output: {path}")
    frame = pd.read_csv(path)
    _require_columns(frame, columns, source=path)
    return frame


def _friendly(value: Any, labels: dict[str, str]) -> str:
    text = str(value)
    return labels.get(text, text.replace("_", " "))


def _registered_checkpoint_order(policy_role: str) -> list[str]:
    if policy_role == "common_tied_035":
        return COMMON_STATE_ORDER
    if policy_role in H5_STATE_ORDER:
        return H5_STATE_ORDER[policy_role]
    raise ValueError(f"No temporal order registered for policy role {policy_role!r}")


def _checkpoint_order(policy_role: str, observed_states: set[str]) -> list[str]:
    order = _registered_checkpoint_order(policy_role)
    unexpected = sorted(observed_states - set(order))
    if unexpected:
        raise ValueError(f"Unexpected checkpoint labels for {policy_role}: {unexpected}")
    return [state for state in order if state in observed_states]


def _phase_boundary_fraction(policy_role: str) -> tuple[float, str]:
    if policy_role == "common_tied_035":
        return 0.80, "Phase 2 begins · 0.80T"
    if policy_role == "boundary_beta_0p60":
        return 0.60, "Phase 2 begins · 0.60T"
    if policy_role == "boundary_beta_0p85":
        return 0.85, "Phase 2 begins · 0.85T"
    raise ValueError(f"No phase boundary registered for policy role {policy_role!r}")


def _state_fraction(policy_role: str, state: str) -> float:
    if state.startswith("fraction_0p"):
        return float(f"0.{state.removeprefix('fraction_0p')}")
    if state == "final":
        return 1.0

    if policy_role == "common_tied_035":
        offsets = {
            "decay_minus_256": -256,
            "decay_minus_64": -64,
            "decay_onset": 0,
            "decay_plus_64": 64,
            "decay_plus_256": 256,
        }
        if state in offsets:
            return 0.80 + offsets[state] / COMMON_TOTAL_UPDATES

    if policy_role in H5_STATE_ORDER:
        decay_offsets = {
            "optimizer_decay_minus_256": -256,
            "optimizer_decay_minus_64": -64,
            "optimizer_decay_onset": 0,
            "optimizer_decay_plus_64": 64,
        }
        if state in decay_offsets:
            return 0.80 + decay_offsets[state] / H5_TOTAL_UPDATES
        switch_offsets = {
            "data_switch_minus_64": -64,
            "data_switch": 0,
            "data_switch_plus_64": 64,
        }
        if state in switch_offsets:
            boundary, _ = _phase_boundary_fraction(policy_role)
            return boundary + switch_offsets[state] / H5_TOTAL_UPDATES

    raise ValueError(f"No training fraction registered for {policy_role}/{state}")


def _phase_boundary_position(policy_role: str, state_order: list[str]) -> tuple[float, str]:
    _, label = _phase_boundary_fraction(policy_role)
    if policy_role == "common_tied_035":
        if "decay_onset" in state_order:
            return float(state_order.index("decay_onset")), label
        left = state_order.index("decay_minus_64")
        right = state_order.index("decay_plus_64")
        return (left + right) / 2, label
    if policy_role == "boundary_beta_0p60":
        return float(state_order.index("data_switch")), label
    if policy_role == "boundary_beta_0p85":
        return float(state_order.index("data_switch")), label
    raise ValueError(f"No phase boundary registered for policy role {policy_role!r}")


def _phase_boundary_shape(
    policy_role: str,
    state_order: list[str],
    *,
    xref: str = "x",
    yref: str = "paper",
    show_label: bool = True,
    scaled_time: bool = False,
) -> dict[str, Any]:
    position, label = (
        _phase_boundary_fraction(policy_role) if scaled_time else _phase_boundary_position(policy_role, state_order)
    )
    shape: dict[str, Any] = {
        "type": "line",
        "xref": xref,
        "yref": yref,
        "x0": position,
        "x1": position,
        "y0": 0,
        "y1": 1,
        "line": {"color": PHASE_BOUNDARY_COLOR, "width": 2},
    }
    if show_label:
        shape["label"] = {
            "text": label,
            "textposition": "top right",
            "font": {"color": PHASE_BOUNDARY_COLOR, "size": 12},
        }
    return shape


def _zero_shape(*, xref: str = "paper", yref: str = "y") -> dict[str, Any]:
    return {
        "type": "line",
        "xref": xref,
        "yref": yref,
        "x0": 0,
        "x1": 1,
        "y0": 0,
        "y1": 0,
        "line": {"color": "#183149", "width": 1},
    }


def _subplot_temporal_shapes(policy_role: str, state_order: list[str], *, count: int) -> list[dict[str, Any]]:
    shapes: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        suffix = "" if index == 1 else str(index)
        shapes.append(_zero_shape(xref=f"x{suffix} domain", yref=f"y{suffix}"))
        shapes.append(
            _phase_boundary_shape(
                policy_role,
                state_order,
                xref=f"x{suffix}",
                yref=f"y{suffix} domain",
                show_label=index == 1,
            )
        )
    return shapes


def _contrast_label(contrast: str) -> str:
    left, separator, right = contrast.partition("__minus__")
    if not separator:
        return _friendly(contrast, {})
    return f"{_friendly(left, SOURCE_LABELS)} - {_friendly(right, SOURCE_LABELS)}"


def _bootstrap_interval(values: pd.Series, identity: str) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if len(array) == 0:
        return np.nan, np.nan
    if len(array) == 1:
        return float(array[0]), float(array[0])
    identity_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
    rng = np.random.default_rng(BOOTSTRAP_SEED ^ identity_seed)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _summarize_with_intervals(frame: pd.DataFrame, groups: list[str], value: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for identity, group in frame.groupby(groups, sort=True, dropna=False):
        identity_tuple = identity if isinstance(identity, tuple) else (identity,)
        values = group[value].dropna()
        low, high = _bootstrap_interval(values, "/".join(map(str, identity_tuple)))
        rows.append(
            {
                **dict(zip(groups, identity_tuple, strict=True)),
                "mean": float(values.mean()) if len(values) else np.nan,
                "ci_low": low,
                "ci_high": high,
                "seed_sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "n": len(values),
            }
        )
    return pd.DataFrame(rows)


def _rgba(color: str, alpha: float) -> str:
    red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def _footnote(text: str, *, y: float, width: int = 105) -> dict[str, Any]:
    wrapped = "<br>".join(textwrap.wrap(text, width=width, break_long_words=False))
    return {
        "text": wrapped,
        "xref": "paper",
        "yref": "paper",
        "x": 0,
        "y": y,
        "showarrow": False,
        "xanchor": "left",
        "align": "left",
    }


def _band_traces(
    summary: pd.DataFrame,
    *,
    x_order: list[str],
    x_values: list[float] | None = None,
    label: str,
    color: str,
    visible: bool,
    showlegend: bool,
) -> list[go.Scatter]:
    indexed = summary.set_index("checkpoint_label").reindex(x_order)
    state_labels = [STATE_LABELS[state] for state in x_order]
    if x_values is not None and len(x_values) != len(x_order):
        raise ValueError("Numeric checkpoint positions do not match the checkpoint inventory")
    x = state_labels if x_values is None else x_values
    state_details = (
        state_labels
        if x_values is None
        else [f"{state_label} · {fraction:.5f}T" for state_label, fraction in zip(state_labels, x_values, strict=True)]
    )
    lower = indexed["ci_low"].to_numpy(dtype=float)
    upper = indexed["ci_high"].to_numpy(dtype=float)
    mean = indexed["mean"].to_numpy(dtype=float)
    distinguishes_evidence = "evidence_role" in indexed
    evidence_role = (
        indexed["evidence_role"].fillna("No measurement").to_numpy(dtype=object)
        if distinguishes_evidence
        else np.full(len(indexed), "Measured state", dtype=object)
    )
    custom = np.column_stack(
        [
            lower,
            upper,
            indexed["seed_sd"].to_numpy(dtype=float),
            indexed["n"].fillna(0).to_numpy(dtype=int),
            evidence_role,
            state_details,
        ]
    )
    evidence_symbols = {
        "v10 H1 contract state": "circle",
        "Post-outcome v10 repair state": "circle-open",
        "Post-outcome plot completion": "diamond-open",
    }
    marker_symbols = (
        [evidence_symbols.get(str(role), "x-open") for role in evidence_role]
        if distinguishes_evidence
        else ["circle"] * len(evidence_role)
    )
    return [
        go.Scatter(
            x=x,
            y=lower,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
            visible=visible,
        ),
        go.Scatter(
            x=x,
            y=upper,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=_rgba(color, 0.14),
            hoverinfo="skip",
            showlegend=False,
            visible=visible,
        ),
        go.Scatter(
            x=x,
            y=mean,
            mode="lines+markers",
            name=label,
            line={"color": color, "width": 3},
            connectgaps=True,
            marker={
                "color": color,
                "size": 8,
                "symbol": marker_symbols,
                "line": {"color": color, "width": 1.4},
            },
            customdata=custom,
            hovertemplate=(
                f"{label}<br>%{{customdata[5]}}<br>mean=%{{y:+.4f}}"
                "<br>bootstrap 95% CI=[%{customdata[0]:+.4f}, %{customdata[1]:+.4f}]"
                "<br>seed SD=%{customdata[2]:.4f}<br>n=%{customdata[3]}"
                "<br>%{customdata[4]}<extra></extra>"
            ),
            showlegend=showlegend,
            visible=visible,
        ),
    ]


def _cohort_label(row: pd.Series, *, include_checkpoint: bool) -> str:
    parts = [
        _friendly(row["analysis_role"], {}),
        _friendly(row["policy_role"], {}),
        _friendly(row["support_id"], {}),
    ]
    if include_checkpoint:
        parts.append(_friendly(row["checkpoint_label"], STATE_LABELS))
    if "evidence_role" in row.index:
        parts.append(EVIDENCE_LABELS.get(str(row["evidence_role"]), str(row["evidence_role"])))
    return " | ".join(parts)


def _dropdown_figure(
    traces: list[BaseTraceType],
    labels: list[str],
    *,
    title: str,
    layout_updates: list[dict[str, Any]] | None = None,
) -> go.Figure:
    if not traces:
        raise ValueError(f"No traces available for {title}")
    if layout_updates is not None and len(layout_updates) != len(traces):
        raise ValueError(f"Layout updates do not match traces for {title}")
    for index, trace in enumerate(traces):
        trace.visible = index == 0
    buttons = []
    for index, label in enumerate(labels):
        visibility = [position == index for position in range(len(traces))]
        layout_update = {"title.text": f"{title}<br><sup>{label}</sup>"}
        if layout_updates is not None:
            layout_update.update(layout_updates[index])
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [{"visible": visibility}, layout_update],
            }
        )
    figure = go.Figure(traces)
    initial_layout = layout_updates[0] if layout_updates is not None else {}
    figure.update_layout(
        title={"text": f"{title}<br><sup>{labels[0]}</sup>", "x": 0.03},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
                "showactive": True,
            }
        ],
        margin={"l": 130, "r": 50, "t": 145, "b": 90},
        paper_bgcolor="#fbf8ef",
        plot_bgcolor="#fbf8ef",
        font={"family": "Avenir Next, Avenir, sans-serif", "color": "#183149"},
    )
    if initial_layout:
        figure.update_layout(initial_layout)
    return figure


def source_conflict_figure(frame: pd.DataFrame, frozen_h1: pd.DataFrame) -> go.Figure:
    selected = frame[frame["geometry"].eq("projected") & frame["component"].eq("trunk")].copy()
    fallback_roles = np.where(
        selected["row_id"].isin(set(frozen_h1["row_id"])),
        "v10 H1 contract state",
        "Post-outcome v10 repair state",
    )
    if "evidence_role" in selected:
        selected["evidence_role"] = selected["evidence_role"].fillna(pd.Series(fallback_roles, index=selected.index))
    else:
        selected["evidence_role"] = fallback_roles
    summary = selected.groupby(
        ["policy_role", "support_id", "checkpoint_label", "statistic"], as_index=False, dropna=False
    ).agg(
        mean=("cosine", "mean"),
        std=("cosine", "std"),
        count=("cosine", "count"),
        evidence_role=("evidence_role", "first"),
        evidence_role_count=("evidence_role", "nunique"),
    )
    if not summary["evidence_role_count"].eq(1).all():
        raise ValueError("A source-geometry checkpoint mixes frozen and additional evidence roles")
    cohorts = summary[["policy_role", "support_id"]].drop_duplicates().sort_values(["support_id", "policy_role"])
    traces: list[BaseTraceType] = []
    labels: list[str] = []
    layout_updates: list[dict[str, Any]] = []
    for cohort in cohorts.itertuples(index=False):
        group = summary[summary["policy_role"].eq(cohort.policy_role) & summary["support_id"].eq(cohort.support_id)]
        state_order = _checkpoint_order(str(cohort.policy_role), set(group["checkpoint_label"].astype(str)))
        mean = group.pivot(index="statistic", columns="checkpoint_label", values="mean").reindex(
            index=["gradient", "optimizer_update"], columns=state_order
        )
        std = group.pivot(index="statistic", columns="checkpoint_label", values="std").reindex(
            index=mean.index, columns=mean.columns
        )
        count = group.pivot(index="statistic", columns="checkpoint_label", values="count").reindex(
            index=mean.index, columns=mean.columns
        )
        evidence = group.pivot(index="statistic", columns="checkpoint_label", values="evidence_role").reindex(
            index=mean.index, columns=mean.columns
        )
        text = np.empty(mean.shape, dtype=object)
        custom = np.empty((*mean.shape, 4), dtype=object)
        for row_index in range(mean.shape[0]):
            for column_index in range(mean.shape[1]):
                value = mean.iat[row_index, column_index]
                if pd.isna(value):
                    text[row_index, column_index] = "N/A"
                    custom[row_index, column_index] = [np.nan, np.nan, 0, evidence.iat[row_index, column_index]]
                else:
                    sd = std.iat[row_index, column_index]
                    n = int(count.iat[row_index, column_index])
                    text[row_index, column_index] = f"{value:+.3f}"
                    custom[row_index, column_index] = [value, sd, n, evidence.iat[row_index, column_index]]
        traces.append(
            go.Heatmap(
                z=mean.to_numpy(dtype=float),
                x=[STATE_LABELS[state] for state in mean.columns],
                y=[STATISTIC_LABELS[statistic] for statistic in mean.index],
                text=text,
                texttemplate="%{text}",
                customdata=custom,
                colorscale="RdYlGn",
                zmin=-1,
                zmax=1,
                colorbar={"title": "Source cosine"},
                hovertemplate=(
                    "%{y} at %{x}<br>mean cosine=%{customdata[0]:+.4f}"
                    "<br>seed SD=%{customdata[1]:.4f}<br>defined n=%{customdata[2]}"
                    "<br>%{customdata[3]}<extra></extra>"
                ),
            )
        )
        labels.append(f"{cohort.support_id} | {_friendly(cohort.policy_role, {})}")
        category_labels = [STATE_LABELS[state] for state in state_order]
        layout_updates.append(
            {
                "xaxis": {"categoryorder": "array", "categoryarray": category_labels},
                "shapes": [_phase_boundary_shape(str(cohort.policy_role), state_order)],
            }
        )
    figure = _dropdown_figure(
        traces,
        labels,
        title="StarCoder-Nemotron source conflict over training",
        layout_updates=layout_updates,
    )
    figure.update_layout(
        height=640,
        margin={"l": 130, "r": 50, "t": 145, "b": 175},
        xaxis_title="Restored training state",
        yaxis_title="Measured vector",
        annotations=[
            _footnote(
                (
                    "Negative/red means direct source conflict; declining positive values mean increasing "
                    "disagreement. Every measured checkpoint is shown. Final optimizer update is undefined because "
                    "LR=0; it is not imputed. Hover identifies frozen H1, post-outcome v10, and post-outcome "
                    "plot-completion measurements. "
                    f"{PRIMARY_PATH_NOTE}"
                ),
                y=-0.34,
            )
        ],
    )
    figure.update_xaxes(tickangle=-28)
    figure.update_yaxes(autorange="reversed")
    return figure


def source_conflict_trajectory_figure(frame: pd.DataFrame, frozen_h1: pd.DataFrame) -> go.Figure:
    selected = frame[frame["geometry"].eq("projected") & frame["component"].eq("trunk")].copy()
    fallback_roles = np.where(
        selected["row_id"].isin(set(frozen_h1["row_id"])),
        "v10 H1 contract state",
        "Post-outcome v10 repair state",
    )
    if "evidence_role" in selected:
        selected["evidence_role"] = selected["evidence_role"].fillna(pd.Series(fallback_roles, index=selected.index))
    else:
        selected["evidence_role"] = fallback_roles
    summary = _summarize_with_intervals(
        selected,
        ["policy_role", "support_id", "checkpoint_label", "statistic"],
        "cosine",
    )
    evidence = selected.groupby(["policy_role", "support_id", "checkpoint_label", "statistic"], as_index=False).agg(
        evidence_role=("evidence_role", "first"),
        evidence_role_count=("evidence_role", "nunique"),
    )
    if not evidence["evidence_role_count"].eq(1).all():
        raise ValueError("A source-geometry checkpoint mixes frozen and additional evidence roles")
    summary = summary.merge(
        evidence.drop(columns="evidence_role_count"),
        on=["policy_role", "support_id", "checkpoint_label", "statistic"],
        how="left",
        validate="one_to_one",
    )
    cohorts = (
        summary[["policy_role", "support_id"]]
        .drop_duplicates()
        .sort_values(["support_id", "policy_role"])
        .itertuples(index=False, name=None)
    )
    cohorts = list(cohorts)
    figure = go.Figure()
    cohort_indices: list[list[int]] = []
    cohort_orders: list[list[str]] = []
    for cohort_index, (policy_role, support) in enumerate(cohorts):
        indices: list[int] = []
        support_frame = summary[summary["policy_role"].eq(policy_role) & summary["support_id"].eq(support)]
        state_order = _checkpoint_order(str(policy_role), set(support_frame["checkpoint_label"].astype(str)))
        cohort_orders.append(state_order)
        for statistic in ("gradient", "optimizer_update"):
            statistic_frame = support_frame[support_frame["statistic"].eq(statistic)]
            traces = _band_traces(
                statistic_frame,
                x_order=state_order,
                x_values=[_state_fraction(str(policy_role), state) for state in state_order],
                label=STATISTIC_LABELS[statistic],
                color=STATISTIC_COLORS[statistic],
                visible=cohort_index == 0,
                showlegend=True,
            )
            for trace in traces:
                indices.append(len(figure.data))
                figure.add_trace(trace)
        cohort_indices.append(indices)

    buttons = []
    for (policy_role, support), indices, state_order in zip(cohorts, cohort_indices, cohort_orders, strict=True):
        visible = [index in indices for index in range(len(figure.data))]
        label = f"{support} | {_friendly(policy_role, {})}"
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title.text": "StarCoder-Nemotron alignment over training",
                        "xaxis.type": "linear",
                        "xaxis.range": FULL_TIMELINE_RANGE,
                        "shapes": [
                            _zero_shape(),
                            _phase_boundary_shape(str(policy_role), state_order, scaled_time=True),
                        ],
                    },
                ],
            }
        )
    first_policy, _ = cohorts[0]
    first_order = cohort_orders[0]
    figure.update_layout(
        title={
            "text": "StarCoder-Nemotron alignment over training",
            "x": 0.03,
        },
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.16,
                "yanchor": "top",
                "showactive": True,
            },
            {
                "buttons": [
                    {
                        "label": "Full timeline",
                        "method": "relayout",
                        "args": [{"xaxis.range": FULL_TIMELINE_RANGE}],
                    },
                    {
                        "label": "Zoom LR decay",
                        "method": "relayout",
                        "args": [{"xaxis.range": LR_DECAY_ZOOM_RANGE}],
                    },
                ],
                "type": "buttons",
                "direction": "right",
                "x": 0.52,
                "xanchor": "center",
                "y": 1.16,
                "yanchor": "top",
                "showactive": True,
            },
        ],
        xaxis_title="Training progress (fraction of updates)",
        xaxis={
            "type": "linear",
            "range": FULL_TIMELINE_RANGE,
            "tickformat": ".3~f",
            "ticksuffix": "T",
        },
        yaxis_title="StarCoder-Nemotron cosine (negative = direct conflict)",
        yaxis={"range": [-1.0, 1.0]},
        shapes=[_zero_shape(), _phase_boundary_shape(str(first_policy), first_order, scaled_time=True)],
        height=820,
        margin={"l": 100, "r": 45, "t": 145, "b": 260},
        paper_bgcolor="#fbf8ef",
        plot_bgcolor="#fbf8ef",
        font={"family": "Avenir Next, Avenir, sans-serif", "color": "#183149"},
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.08, "yanchor": "bottom"},
        annotations=[
            _footnote(
                (
                    "Horizontal position is the actual fraction of training updates; use Zoom LR decay to separate "
                    "the ±64/±256 probes. Filled markers are frozen H1 states and open markers are post-outcome v10 "
                    "or plot-completion measurements. Bands are pointwise seed-bootstrap 95% intervals. Negative "
                    "values are direct conflict; the final optimizer update is undefined because LR=0. Projected "
                    "trunk is shown."
                ),
                y=-0.43,
                width=80,
            )
        ],
    )
    return figure


def target_utility_figure(frame: pd.DataFrame) -> go.Figure:
    selected = frame[frame["geometry"].eq("projected") & frame["component"].eq("trunk")].copy()
    cohort_columns = ["analysis_role", "policy_role", "support_id", "checkpoint_label", "evidence_role"]
    summary = selected.groupby([*cohort_columns, "target", "source"], as_index=False)["cosine"].agg(
        mean="mean", std="std", count="count"
    )
    cohorts = summary[cohort_columns].drop_duplicates().copy()
    cohorts["state_rank"] = cohorts.apply(
        lambda row: _registered_checkpoint_order(str(row["policy_role"])).index(str(row["checkpoint_label"])),
        axis=1,
    )
    cohorts = cohorts.sort_values(["support_id", "analysis_role", "policy_role", "state_rank"])
    target_order = list(TARGET_LABELS)
    source_order = list(SOURCE_LABELS)
    color_limit = float(summary["mean"].abs().max())
    traces: list[BaseTraceType] = []
    labels: list[str] = []
    for cohort in cohorts.itertuples(index=False):
        group = summary[
            summary["analysis_role"].eq(cohort.analysis_role)
            & summary["policy_role"].eq(cohort.policy_role)
            & summary["support_id"].eq(cohort.support_id)
            & summary["checkpoint_label"].eq(cohort.checkpoint_label)
            & summary["evidence_role"].eq(cohort.evidence_role)
        ]
        mean = group.pivot(index="target", columns="source", values="mean").reindex(
            index=target_order, columns=source_order
        )
        std = group.pivot(index="target", columns="source", values="std").reindex(index=mean.index, columns=mean.columns)
        count = group.pivot(index="target", columns="source", values="count").reindex(
            index=mean.index, columns=mean.columns
        )
        text = mean.map(lambda value: "" if pd.isna(value) else f"{value:+.3f}").to_numpy()
        evidence = np.full(mean.shape, EVIDENCE_LABELS.get(str(cohort.evidence_role), str(cohort.evidence_role)))
        custom = np.stack(
            [mean.to_numpy(dtype=float), std.to_numpy(dtype=float), count.fillna(0).to_numpy(dtype=int), evidence],
            axis=-1,
        )
        traces.append(
            go.Heatmap(
                z=mean.to_numpy(dtype=float),
                x=[SOURCE_LABELS[source] for source in mean.columns],
                y=[TARGET_LABELS[target] for target in mean.index],
                text=text,
                texttemplate="%{text}",
                customdata=custom,
                colorscale="RdYlGn",
                zmin=-color_limit,
                zmax=color_limit,
                colorbar={"title": "Utility cosine"},
                hovertemplate=(
                    "%{y} target / %{x} update<br>mean utility cosine=%{customdata[0]:+.4f}"
                    "<br>seed SD=%{customdata[1]:.4f}<br>n=%{customdata[2]}"
                    "<br>%{customdata[3]}<extra></extra>"
                ),
            )
        )
        labels.append(_cohort_label(pd.Series(cohort._asdict()), include_checkpoint=True))
    figure = _dropdown_figure(traces, labels, title="Target-source optimizer-update alignment matrix")
    figure.update_layout(
        height=610,
        xaxis_title="Counterfactual source update",
        yaxis_title="Evaluation target gradient",
        annotations=[
            _footnote(
                (
                    "Cell labels and colors are utility cosines: positive/green means the source update locally "
                    "reduces target loss. Dropdown states are temporally ordered within each policy and explicitly "
                    "identify post-outcome completion states. "
                    f"{PRIMARY_PATH_NOTE}"
                ),
                y=-0.18,
            )
        ],
    )
    figure.update_yaxes(autorange="reversed")
    return figure


def target_utility_trajectory_figure(frame: pd.DataFrame) -> go.Figure:
    selected = frame[frame["geometry"].eq("projected") & frame["component"].eq("trunk")].copy()
    summary = _summarize_with_intervals(
        selected,
        ["analysis_role", "policy_role", "support_id", "target", "source", "checkpoint_label", "evidence_role"],
        "cosine",
    )
    cohorts = (
        summary[["analysis_role", "policy_role", "support_id"]]
        .drop_duplicates()
        .sort_values(["support_id", "analysis_role", "policy_role"])
        .itertuples(index=False, name=None)
    )
    cohorts = list(cohorts)
    target_order = list(TARGET_LABELS)
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[TARGET_LABELS[target] for target in target_order],
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
        state_order = _checkpoint_order(str(policy_role), set(cohort_frame["checkpoint_label"].astype(str)))
        cohort_orders.append(state_order)
        for target_index, target in enumerate(target_order):
            row = target_index // 2 + 1
            column = target_index % 2 + 1
            for source in SOURCE_LABELS:
                source_frame = cohort_frame[cohort_frame["target"].eq(target) & cohort_frame["source"].eq(source)]
                traces = _band_traces(
                    source_frame,
                    x_order=state_order,
                    label=SOURCE_LABELS[source],
                    color=SOURCE_COLORS[source],
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
        label = f"{support} | {_friendly(analysis_role, {})} | {_friendly(policy_role, {})}"
        category_labels = [STATE_LABELS[state] for state in state_order]
        layout_update: dict[str, Any] = {
            "title.text": f"Target-source utility through training<br><sup>{label}</sup>",
            "shapes": _subplot_temporal_shapes(str(policy_role), state_order, count=4),
        }
        for axis_name in ("xaxis", "xaxis2", "xaxis3", "xaxis4"):
            layout_update[f"{axis_name}.categoryorder"] = "array"
            layout_update[f"{axis_name}.categoryarray"] = category_labels
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"visible": visible},
                    layout_update,
                ],
            }
        )
    for row in (1, 2):
        for column in (1, 2):
            figure.update_xaxes(
                title_text="Training state",
                row=row,
                col=column,
                tickangle=-28,
                categoryorder="array",
                categoryarray=[STATE_LABELS[state] for state in cohort_orders[0]],
            )
            figure.update_yaxes(title_text="Utility cosine", row=row, col=column)
    first_role, first_policy, first_support = cohorts[0]
    figure.update_layout(
        title={
            "text": (
                "Target-source utility through training"
                f"<br><sup>{first_support} | {_friendly(first_role, {})} | {_friendly(first_policy, {})}</sup>"
            ),
            "x": 0.03,
        },
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.10,
                "yanchor": "top",
                "showactive": True,
            }
        ],
        height=1080,
        margin={"l": 85, "r": 45, "t": 155, "b": 245},
        paper_bgcolor="#fbf8ef",
        plot_bgcolor="#fbf8ef",
        font={"family": "Avenir Next, Avenir, sans-serif", "color": "#183149"},
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.05, "yanchor": "bottom"},
        shapes=_subplot_temporal_shapes(str(first_policy), cohort_orders[0], count=4),
        annotations=[
            *figure.layout.annotations,
            _footnote(
                (
                    "Positive cosine means the counterfactual source update locally reduces target loss. "
                    "Bands are pointwise seed-bootstrap 95% intervals, not simultaneous bands; this is local, "
                    "off-policy evidence. H5 policies are shown in their actual event order; their pre-to-mid shift "
                    "is confounded by cumulative exposure. The vertical bar marks the policy's phase-2 boundary. "
                    "Open markers are post-outcome v10 or plot-completion measurements and do not revise frozen tests. "
                    f"{PRIMARY_PATH_NOTE}"
                ),
                y=-0.31,
            ),
        ],
    )
    return figure


def source_choice_figure(frame: pd.DataFrame) -> go.Figure:
    selected = frame[frame["geometry"].eq("projected") & frame["component"].eq("trunk")].copy()
    cohort_columns = ["analysis_role", "policy_role", "support_id", "contrast"]
    summary = selected.groupby([*cohort_columns, "target", "checkpoint_label", "evidence_role"], as_index=False)[
        "A_y"
    ].agg(mean="mean", std="std", count="count")
    cohorts = summary[cohort_columns].drop_duplicates().sort_values(cohort_columns)
    target_order = list(TARGET_LABELS)
    color_limit = float(summary["mean"].abs().max())
    traces: list[BaseTraceType] = []
    labels: list[str] = []
    layout_updates: list[dict[str, Any]] = []
    for cohort in cohorts.itertuples(index=False):
        group = summary[
            summary["analysis_role"].eq(cohort.analysis_role)
            & summary["policy_role"].eq(cohort.policy_role)
            & summary["support_id"].eq(cohort.support_id)
            & summary["contrast"].eq(cohort.contrast)
        ]
        observed_states = _checkpoint_order(str(cohort.policy_role), set(group["checkpoint_label"].astype(str)))
        state_evidence = (
            group.groupby("checkpoint_label")["evidence_role"]
            .agg(lambda values: values.iloc[0] if values.nunique() == 1 else "mixed provenance")
            .reindex(observed_states)
        )
        state_labels = [
            f"{STATE_LABELS.get(state, state)} {'●' if str(state_evidence[state]) == 'v10 H1 contract state' else '○'}"
            for state in observed_states
        ]
        mean = group.pivot(index="target", columns="checkpoint_label", values="mean").reindex(
            index=target_order, columns=observed_states
        )
        std = group.pivot(index="target", columns="checkpoint_label", values="std").reindex(
            index=mean.index, columns=mean.columns
        )
        count = group.pivot(index="target", columns="checkpoint_label", values="count").reindex(
            index=mean.index, columns=mean.columns
        )
        text = mean.map(lambda value: "" if pd.isna(value) else f"{value:+.3f}").to_numpy()
        evidence = np.tile(
            np.array(
                [
                    EVIDENCE_LABELS.get(str(state_evidence[state]), str(state_evidence[state]))
                    for state in observed_states
                ],
                dtype=object,
            ),
            (len(mean.index), 1),
        )
        custom = np.stack(
            [mean.to_numpy(dtype=float), std.to_numpy(dtype=float), count.fillna(0).to_numpy(dtype=int), evidence],
            axis=-1,
        )
        traces.append(
            go.Heatmap(
                z=mean.to_numpy(dtype=float),
                x=state_labels,
                y=[TARGET_LABELS[target] for target in mean.index],
                text=text,
                texttemplate="%{text}",
                customdata=custom,
                colorscale="RdYlGn",
                zmin=-color_limit,
                zmax=color_limit,
                colorbar={"title": "Choice alignment<br>A_y"},
                hovertemplate=(
                    "%{y} at %{x}<br>mean A_y=%{customdata[0]:+.4f}"
                    "<br>seed SD=%{customdata[1]:.4f}<br>n=%{customdata[2]}"
                    "<br>%{customdata[3]}<extra></extra>"
                ),
            )
        )
        labels.append(
            " | ".join(
                [
                    _friendly(cohort.analysis_role, {}),
                    _friendly(cohort.policy_role, {}),
                    _friendly(cohort.support_id, {}),
                    _contrast_label(str(cohort.contrast)),
                ]
            )
        )
        layout_updates.append(
            {
                "xaxis": {
                    "categoryorder": "array",
                    "categoryarray": state_labels,
                },
                "shapes": [_phase_boundary_shape(str(cohort.policy_role), observed_states)],
            }
        )
    figure = _dropdown_figure(
        traces,
        labels,
        title="Target-conditioned source-choice alignment",
        layout_updates=layout_updates,
    )
    figure.update_layout(
        height=700,
        margin={"l": 130, "r": 50, "t": 145, "b": 185},
        xaxis_title="Restored training state",
        yaxis_title="Evaluation target",
        annotations=[
            _footnote(
                (
                    "Positive/green A_y favors the first-named source in the dropdown contrast. H5 switch-relative "
                    "states are ordered separately for each policy because the 0.60 switch precedes LR decay and the "
                    "0.85 switch follows it. The vertical bar marks phase 2. All dots in this target-conditioned "
                    "panel are open because these are post-outcome v10 or plot-completion measurements; they do not "
                    "revise frozen tests. "
                    f"{PRIMARY_PATH_NOTE}"
                ),
                y=-0.34,
            )
        ],
    )
    figure.update_xaxes(tickangle=-28)
    figure.update_yaxes(autorange="reversed")
    return figure


def mechanism_forest_figure(frames: list[pd.DataFrame], multiplicity: pd.DataFrame) -> go.Figure:
    combined = pd.concat(frames, ignore_index=True)
    _require_columns(
        combined,
        {
            "contrast",
            "mean",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
            "n_paired_seeds",
            "evidence_role",
            "alternative",
            "exact_sign_flip_p_unadjusted",
        },
        source=Path("summary tables"),
    )
    _require_columns(
        multiplicity,
        {"contrast", "exact_two_sided_sign_flip_p", "holm_p_across_47"},
        source=Path("multiplicity audit"),
    )
    combined = combined.merge(
        multiplicity[["contrast", "exact_two_sided_sign_flip_p", "holm_p_across_47"]],
        on="contrast",
        how="left",
        validate="one_to_one",
    )
    if combined[["exact_two_sided_sign_flip_p", "holm_p_across_47"]].isna().any().any():
        raise ValueError("Multiplicity audit does not cover all frozen contrasts")
    combined["family"] = combined["contrast"].str.extract(r"^(H[235])", expand=False)
    if combined["family"].isna().any():
        raise ValueError("Every frozen contrast must start with H2, H3, or H5")
    combined["holm_survivor"] = combined["holm_p_across_47"].le(0.05)

    family_order = ["H2", "H3", "H5"]
    family_colors = {"H2": "#1b7f79", "H3": "#d65a31", "H5": "#d9a21b"}
    family_counts = [int(combined["family"].eq(family).sum()) for family in family_order]
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.055,
        row_heights=family_counts,
        subplot_titles=[
            f"{family}: {count} frozen contrasts" for family, count in zip(family_order, family_counts, strict=True)
        ],
    )
    for row, family in enumerate(family_order, start=1):
        group = combined[combined["family"].eq(family)].copy()
        custom = group[
            [
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "n_paired_seeds",
                "evidence_role",
                "alternative",
                "exact_sign_flip_p_unadjusted",
                "exact_two_sided_sign_flip_p",
                "holm_p_across_47",
            ]
        ].to_numpy()
        figure.add_trace(
            go.Scatter(
                x=group["mean"],
                y=group["contrast"],
                mode="markers",
                marker={
                    "size": np.where(group["holm_survivor"], 13, 10),
                    "symbol": np.where(group["holm_survivor"], "diamond", "circle-open"),
                    "color": family_colors[family],
                    "line": {"color": "#183149", "width": 1.2},
                },
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": group["bootstrap_ci95_high"] - group["mean"],
                    "arrayminus": group["mean"] - group["bootstrap_ci95_low"],
                    "color": "#183149",
                    "thickness": 1.4,
                },
                customdata=custom,
                hovertemplate=(
                    "%{y}<br>mean=%{x:+.5f}"
                    "<br>bootstrap 95% CI=[%{customdata[0]:+.5f}, %{customdata[1]:+.5f}]"
                    "<br>paired seeds=%{customdata[2]}<br>role=%{customdata[3]}"
                    "<br>preregistered alternative=%{customdata[4]}"
                    "<br>p for registered alternative=%{customdata[5]:.4g}"
                    "<br>post-review two-sided p=%{customdata[6]:.4g}"
                    "<br>Holm p across 47=%{customdata[7]:.4g}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        figure.add_vline(x=0, line_color="#6f7f87", line_dash="dash", row=row, col=1)
        figure.update_xaxes(title_text="Post-outcome repair contrast estimate", row=row, col=1)
        figure.update_yaxes(autorange="reversed", row=row, col=1)

    survivor_count = int(combined["holm_survivor"].sum())
    figure.update_layout(
        title={"text": "H2/H3/H5 post-outcome development contrasts and multiplicity audit", "x": 0.03},
        height=max(1280, 31 * len(combined) + 430),
        margin={"l": 470, "r": 45, "t": 120, "b": 135},
        paper_bgcolor="#fbf8ef",
        plot_bgcolor="#fbf8ef",
        font={"family": "Avenir Next, Avenir, sans-serif", "color": "#183149"},
        annotations=[
            *figure.layout.annotations,
            _footnote(
                (
                    f"Hollow circles are not global discoveries; filled diamonds survive two-sided Holm across all "
                    f"47 tests ({survivor_count}/47, all H5 pre-minus-mid). Intervals are two-sided seed-bootstrap "
                    "intervals. For the 38 contrasts with a one-sided registered alternative, a CI excluding zero is "
                    "not by itself evidence for that alternative; no contrast counts as a global discovery unless it "
                    "is a filled diamond."
                ),
                y=-0.075,
            ),
        ],
    )
    return figure


def _write_figure(figure: go.Figure, path: Path) -> None:
    _write_create_only_text(
        path,
        pio.to_html(figure, include_plotlyjs=True, full_html=True, config=PLOT_CONFIG, div_id=path.stem),
    )


def _write_create_only_text(path: Path, text: str) -> None:
    payload = text.encode()
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite a non-identical rendered artifact: {path}") from error


def _write_index(output_dir: Path) -> None:
    policy_rows = [
        (
            "P1",
            "Tied probe spine",
            "(0.35, 0.35)",
            "Restored at multiple times to measure gradients, optimizer updates, and short rollouts.",
        ),
        (
            "P2",
            "Selected tied comparator",
            "(0.70, 0.70)",
            "The best constant-mixture comparator selected before this mechanism study.",
        ),
        (
            "P3",
            "Selected two-phase policy",
            "(0.02, 0.82)",
            "Almost no StarCoder early and 82% StarCoder during the final 20% of training.",
        ),
        (
            "P4",
            "Aggregate-matched tied control",
            "(0.180, 0.180)",
            "Matches P3's total StarCoder exposure, isolating schedule at this low-code aggregate.",
        ),
        (
            "B",
            "Data-switch ladder",
            "switch at 0.60T to 0.90T",
            "Moves the data switch while optimizer decay remains fixed at 0.80T; aggregate and contrast are fixed.",
        ),
    ]
    policy_html = "".join(
        "<tr>"
        f'<th scope="row"><span class="tag">{html.escape(label)}</span></th>'
        f"<td><strong>{html.escape(name)}</strong></td>"
        f"<td><code>{html.escape(schedule)}</code></td>"
        f"<td>{html.escape(purpose)}</td>"
        "</tr>"
        for label, name, schedule, purpose in policy_rows
    )
    verdict_rows = [
        (
            "Endpoint control",
            "Does phase scheduling matter at fixed aggregate?",
            "Yes, at the low 0.18 aggregate inherited from P3",
            "P3 beats aggregate-matched P4 by 0.10837 BPB on full support and 0.08020 on m100a. "
            "This establishes trajectory dependence, not superiority to the best tied policy. No gradient rows exist "
            "on P2, P3, or P4, so it cannot establish mediation.",
        ),
        (
            "H1",
            "Do StarCoder and Nemotron updates become increasingly conflicting late?",
            "No actionable late conflict trend (descriptive, tied-0.35 subset)",
            "On the 56 tied-0.35 trajectories with H1 rows, full-support projected-trunk raw cosine is 0.471 at 0.10T "
            "and 0.546 at decay onset, then -0.301 only at the terminal zero-LR state. Optimizer-update cosine rises "
            "from 0.446 to 0.471 and is undefined when LR is zero.",
        ),
        (
            "H2",
            "Does the target-relative value of StarCoder rise late versus C4?",
            "Null",
            "The primary m100a effect is +0.000794 (95% CI -0.003956 to +0.005639; one-sided p=0.376), and the required "
            "GitHub-Python sign check fails. Target preferences are structured but temporally stable.",
        ),
        (
            "H3",
            "Does finite-support repetition amplify the temporal conflict signal?",
            "Null for the preregistered mechanism",
            "The m100a-minus-full interaction is 0.00191 (95% CI -0.00342 to 0.00705; two-sided p=0.489), and 0/32 "
            "scale-free signatures survive Holm. The two smallest-p signatures are Programming Languages "
            "support-separation effects in the direction opposite the registered prediction.",
        ),
        (
            "H4",
            "Can optimizer-aware local utility predict the effect of nearby mixtures?",
            "Predictive post hoc, but policy selection is unidentified",
            "The unfrozen mapping tracks held-out 512-update BPB (R2 0.825; Spearman 0.93), but selection is "
            "degenerate: predicted and observed optima both hit the registered boundary q=0.55, while every curve "
            "keeps improving through q=1.0. RMSE 0.0381 BPB is 29 times the H5 endpoint effect.",
        ),
        (
            "H5",
            "Does moving the data switch change state and endpoint performance?",
            "Endpoint effect supported; mechanism unidentified",
            "Moving the switch from 0.60T to 0.85T changes BPB by -0.001324 (95% CI -0.002008 to -0.000631; "
            "p=0.00284; lower is better), but the interval crosses the 0.001 practical margin. The cumulative-exposure "
            "gap falls from 0.1001 at mid-training to 0.0022 before decay; the alignment shift remains negative, so "
            "exposure alone does not fully account for it.",
        ),
    ]
    verdict_html = "".join(
        "<tr>"
        f'<th scope="row"><span class="tag">{html.escape(label)}</span></th>'
        f"<td>{html.escape(question)}</td>"
        f"<td><strong>{html.escape(verdict)}</strong></td>"
        f"<td>{html.escape(evidence)}</td>"
        "</tr>"
        for label, question, verdict, evidence in verdict_rows
    )
    interpretation_rows = [
        (
            "Role of gradient conflict",
            "Broad, growing StarCoder-versus-Nemotron conflict is not supported as the driver of the two-phase gain. "
            "A local, target- and state-specific source-selection signal remains possible, but was not validated here.",
        ),
        (
            "State for the surrogate",
            "Cumulative per-source exposure is the first trajectory state justified by this panel. Gradient descriptors "
            "should be added only if they improve held-out branch prediction beyond exposure.",
        ),
        (
            "Strongest repetition clue",
            "The registered interaction is null. The two smallest-p scale-free signatures point opposite the "
            "prediction, and none of 32 signatures survives multiplicity correction.",
        ),
        (
            "What would identify causality",
            "Measure the mediator on endpoint arms, then cross tied versus two-phase schedules with standard versus "
            "norm-matched conflict-neutralized updates. The interaction, not another cosine, is the causal test.",
        ),
    ]
    interpretation_html = "".join(
        f'<article class="answer"><h3>{html.escape(question)}</h3><p>{html.escape(answer)}</p></article>'
        for question, answer in interpretation_rows
    )
    cards = [
        (
            "Endpoint interventions",
            "../starcoder_wsd80_gradient_probe_full_results_20260818/endpoint_interventions.html",
            "Aggregate-matched, selected-policy, and moved-switch endpoint effects.",
        ),
        (
            "Local utility rollouts",
            "../starcoder_wsd80_gradient_probe_full_results_20260818/h4_rollout_validation.html",
            "Post-hoc calibration from optimizer-aware utility to 512-update BPB changes.",
        ),
        (
            "Source alignment trajectory",
            "source_source_conflict_trajectory.html",
            "Every restored StarCoder-Nemotron raw-gradient and optimizer-update cosine, with pointwise intervals "
            "and the phase boundary.",
        ),
        (
            "Source alignment matrix",
            "source_source_conflict_matrix.html",
            "Every restored StarCoder-Nemotron gradient and optimizer-update cosine over training, with the phase "
            "boundary.",
        ),
        (
            "Target-source trajectories",
            "target_source_utility_trajectories.html",
            "How each source update aligns with four evaluation targets through training; the bar marks phase 2.",
        ),
        (
            "Target-source utility",
            "target_source_utility_matrix.html",
            "Which source update locally helps each evaluation target.",
        ),
        (
            "Source-choice alignment",
            "target_source_choice_alignment.html",
            "Target-specific preference for StarCoder versus Nemotron updates through time, with the phase boundary.",
        ),
        (
            "Mechanism contrasts",
            "mechanism_effect_forest.html",
            "The 47 post-outcome H2/H3/H5 development contrasts and multiplicity audit.",
        ),
    ]
    card_html = "".join(
        f'<a class="card" href="{html.escape(href)}"><h2>{html.escape(title)}</h2><p>{html.escape(body)}</p></a>'
        for title, href, body in cards
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Does gradient conflict explain two-phase WSD80 gains?</title>
<style>
:root{{--ink:#183149;--paper:#fbf8ef;--teal:#1b7f79;--orange:#d65a31;--line:#d8cfbd}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:"Avenir Next",Avenir,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:72px 32px 96px}}
.eyebrow{{text-transform:uppercase;letter-spacing:.16em;color:var(--orange);font-weight:700}}
h1{{font-family:Georgia,serif;font-size:clamp(42px,7vw,78px);line-height:.98;margin:12px 0 24px;max-width:980px}}
.deck{{font-size:20px;line-height:1.55;max-width:920px;margin:0}}
.facts{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line);
border:1px solid var(--line);margin:38px 0}}
.fact{{background:#fffdf7;padding:20px}}
.fact b{{display:block;font-family:Georgia,serif;font-size:23px;margin-bottom:5px}}
.fact span{{font-size:14px;line-height:1.4;color:#536674}}
.section{{margin-top:64px}}
.section-head{{display:grid;grid-template-columns:180px 1fr;gap:28px;align-items:start;margin-bottom:24px}}
.section-number{{color:var(--orange);font-size:13px;font-weight:800;letter-spacing:.12em;text-transform:uppercase}}
.section h2{{font-family:Georgia,serif;font-size:clamp(31px,4vw,46px);line-height:1.05;margin:0 0 9px}}
.section-intro{{line-height:1.55;color:#536674;max-width:820px;margin:0}}
.verdict{{margin:42px 0 24px;padding:26px 28px;background:#15394a;color:#fffdf7;border-left:8px solid var(--orange)}}
.verdict h2{{font-family:Georgia,serif;font-size:30px;margin:0 0 10px}}
.verdict p{{font-size:18px;line-height:1.55;margin:0;max-width:980px}}
.explain-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}}
.explain{{padding:22px 24px;background:#fffdf7;border-top:4px solid var(--orange)}}
.explain h3{{font-family:Georgia,serif;font-size:23px;margin:0 0 9px}}
.explain p{{line-height:1.5;margin:0}}
.equation{{margin-top:16px;padding:20px 24px;background:#edf4f1;border-left:5px solid var(--teal);line-height:1.55}}
.equation code{{font-size:15px;white-space:normal}}
.table-wrap{{overflow-x:auto;margin-top:32px;border:1px solid var(--line);background:#fffdf7}}
table{{width:100%;border-collapse:collapse;min-width:940px}}
caption{{font-family:Georgia,serif;font-size:30px;text-align:left;padding:24px 24px 16px}}
th,td{{padding:17px 18px;text-align:left;vertical-align:top;border-top:1px solid var(--line);line-height:1.45}}
thead th{{font-size:12px;text-transform:uppercase;letter-spacing:.10em;color:#536674;background:#f4efe3}}
tbody th{{width:90px}}
.tag{{display:inline-block;padding:5px 9px;background:#e9f3ef;color:#155b58;font-size:12px;
letter-spacing:.08em;font-weight:800}}
.answers{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:24px}}
.answer{{padding:22px 24px;border-top:4px solid var(--teal);background:#fffdf7}}
.answer h3{{font-family:Georgia,serif;font-size:23px;line-height:1.15;margin:0 0 10px}}
.answer p{{line-height:1.5;margin:0}}
.endpoint-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}}
.endpoint{{padding:24px;background:#fffdf7;border:1px solid var(--line)}}
.endpoint .value{{font-family:Georgia,serif;font-size:34px;color:var(--teal);margin:8px 0}}
.endpoint h3{{font-size:14px;text-transform:uppercase;letter-spacing:.08em;margin:0;color:#536674}}
.endpoint p{{line-height:1.45;margin:0}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:24px}}
.card{{display:block;color:inherit;text-decoration:none;border:1px solid var(--line);background:#fffdf7;
padding:28px;min-height:170px;transition:transform .16s ease,border-color .16s ease}}
.card:hover{{transform:translateY(-4px);border-color:var(--teal)}}
.card h2{{font-family:Georgia,serif;font-size:28px;margin:0 0 12px}}
.card p{{line-height:1.55;margin:0}}
footer{{margin-top:48px;padding-top:20px;border-top:1px solid var(--line);line-height:1.5;color:#536674}}
@media(max-width:820px){{.facts{{grid-template-columns:repeat(2,minmax(0,1fr))}}.section-head{{display:block}}
.section-number{{display:block;margin-bottom:10px}}.explain-grid,.endpoint-grid{{grid-template-columns:1fr}}}}
@media(max-width:720px){{main{{padding:44px 18px 70px}}.grid,.answers{{grid-template-columns:1fr}}}}
</style></head><body><main>
	<p class="eyebrow">StarCoder WSD80 mechanistic study</p>
	<h1>Does gradient conflict explain why two-phase data mixtures help?</h1>
	<p class="deck">We train one model on StarCoder code and Nemotron text, and vary the StarCoder share between the
	first 80% and final 20% of training. Optimizer decay starts at 80% of training, hence "WSD80". The goal is to learn
	whether source-gradient conflict explains why a time-varying mixture can outperform a constant one.</p>
	<section class="facts">
		<div class="fact"><b>210M</b><span>total model parameters</span></div>
		<div class="fact"><b>7.408B</b><span>training tokens</span></div>
		<div class="fact"><b>80 / 20</b><span>stable-learning-rate then cosine-decay schedule</span></div>
		<div class="fact"><b>BPB</b><span>evaluation loss; lower is better</span></div>
	</section>
	<section class="verdict"><h2>Answer</h2><p>Two-phase scheduling has real endpoint effects in this setting, but the
	data do not support broad, increasing gradient conflict as their cause. During every decision-relevant checkpoint,
	StarCoder and Nemotron updates remain positively aligned and become slightly more aligned. Target preferences are
	stable, the registered repetition interaction is null, and the endpoint arms lack mediator measurements. Gradient
	alignment remains a candidate local feature, not an established mechanism.</p></section>

	<section class="section"><div class="section-head"><span class="section-number">1 / Setup</span><div>
	<h2>Policy and data</h2><p class="section-intro">A policy is written <code>(p0, p1)</code>, where each value is the
	StarCoder fraction in that phase; Nemotron receives the remainder. A tied or single-phase policy has
	<code>p0 = p1</code>. The primary target is Programming Languages BPB.</p></div></div>
	<div class="explain-grid">
		<article class="explain"><h3>Two support regimes</h3><p><strong>Full</strong> uses the broad source pool.
		<strong>m100a</strong> repeatedly draws from a fixed finite support, creating simulated repetition.</p></article>
		<article class="explain"><h3>Four targets</h3><p>Programming Languages and GitHub Python are code-facing;
		C4 English and Wikipedia English are natural-language references.</p></article>
		<article class="explain"><h3>Inference</h3><p>The training seed is the inferential unit. Intervals are paired
		seed-bootstrap intervals; repeated probe blocks measure precision but are not extra samples.</p></article>
	</div>
	<div class="table-wrap"><table><caption>Policy arms</caption><thead><tr><th>ID</th><th>Role</th>
	<th>StarCoder share</th>
	<th>Why it exists</th></tr></thead><tbody>{policy_html}</tbody></table></div></section>

	<section class="section"><div class="section-head"><span class="section-number">2 / Measurement</span><div>
	<h2>What "gradient conflict" means here</h2><p class="section-intro">At restored checkpoints we compute gradients of
	next-token loss with respect to model parameters, then apply the exact saved optimizer state counterfactually to a
	StarCoder or Nemotron batch.</p></div></div>
	<div class="explain-grid">
		<article class="explain"><h3>Source-source cosine</h3><p>Cosine between StarCoder and Nemotron gradients
		or optimizer updates. Negative means direct conflict; a declining positive value means more disagreement.</p>
		</article>
		<article class="explain"><h3>Target-source alignment</h3><p>Whether replacing a Nemotron update with a StarCoder
		update locally improves a chosen evaluation target. Positive favors StarCoder for that target.</p></article>
		<article class="explain"><h3>Behavioral checks</h3><p>Exact 512-update continuations test local
		predictions; full training endpoints test final BPB. Local validity does not by itself establish endpoint
		mediation.</p></article>
	</div>
	<div class="equation"><code>g_q(t) = gradient of source-q loss at state t;&nbsp;&nbsp;
	A_y(t) = -&lt;g_y, Delta_StarCoder - Delta_Nemotron&gt; / normalized magnitudes.</code><br>
	The optimizer-aware <code>A_y</code> statistic asks which source update is locally better aligned with target
	<code>y</code>.</div></section>

	<section class="section"><div class="section-head"><span class="section-number">3 / Endpoint evidence</span><div>
	<h2>The trajectory matters, but the mechanism is not identified</h2><p class="section-intro">These endpoint controls
	establish schedule and support dependence. None measures a gradient mediator on the policies being compared.</p>
	</div></div><div class="endpoint-grid">
		<article class="endpoint"><h3>Fixed aggregate: P3 over P4</h3><div class="value">+0.1084 BPB</div>
		<p>Full support; +0.0802 on m100a. P3 is better, but P4 is a poor low-code tied policy, not the global tied
		optimum.</p>
		</article>
		<article class="endpoint"><h3>Selected policies: P3 over P2</h3><div class="value">support-dependent</div>
		<p>P3 is 0.00339 BPB worse on full support and 0.00744 better on m100a. This is a joint policy contrast, not pure
		ordering.</p></article>
		<article class="endpoint"><h3>Later versus earlier switch</h3><div class="value">0.00132 BPB</div>
		<p>Moving the switch from 60% to 85% of training improves BPB, but the interval crosses the 0.001 practical
		margin and switch time is confounded with phase weights.</p></article>
	</div></section>

	<section class="section"><div class="section-head"><span class="section-number">4 / Hypotheses</span><div>
	<h2>Answers to the preregistered questions</h2><p class="section-intro">H1-H5 are internal labels for five planned
	mechanism checks. The table gives the effect, uncertainty, and interpretation boundary for each.</p></div></div>
	<div class="table-wrap"><table><thead><tr><th>Test</th><th>Question</th><th>Verdict</th>
	<th>Decisive evidence and scope</th></tr></thead><tbody>{verdict_html}</tbody></table></div></section>

	<section class="section"><div class="section-head"><span class="section-number">5 / Interpretation</span><div>
	<h2>What changes in our model of two-phase training</h2><p class="section-intro">The experiment narrows the mechanism
	space. It does not yet supply a reliable feature for solving the global two-phase optimum.</p></div></div>
	<section class="answers">{interpretation_html}</section></section>

	<section class="section"><div class="section-head"><span class="section-number">6 / Evidence</span><div>
	<h2>Inspect the measurements</h2><p class="section-intro">Open any panel for checkpoint trajectories,
	uncertainty, cohort selectors, exact contrasts, and endpoint interventions. A dark vertical bar marks the beginning
	of phase 2 in every panel whose horizontal axis is training time.</p></div></div>
	<section class="grid">{card_html}</section>
	</section>

	<section class="section"><div class="section-head"><span class="section-number">7 / Scientific status</span><div>
	<h2>Development evidence, not untouched confirmation</h2><p class="section-intro">Five confirmatory families were
	planned, but the familywise verdict could not be computed without post-hoc choices. The v10 repair was frozen only
	after outcomes had been inspected.</p></div></div>
	<div class="equation">The post-review audit contains 47 H2/H3/H5 development contrasts. Four survive two-sided Holm;
	all are H5 pre-minus-mid shifts carrying a cumulative-exposure confound, although the alignment shift remains after
	the gap nearly closes. H4's mapping was not frozen and its selection score is degenerate. Projected trunk was primary
	in prose, but geometry/component fields were absent from the numerical repair contract. Final optimizer-update cosine
	is undefined at zero learning rate.</div></section>

	<footer>A direct causal test must measure gradient conflict on the endpoint arms and intervene on it. The clean next
	step is a fresh-seed tied-versus-two-phase by standard-versus-conflict-neutralized, norm-matched factorial, plus the
	missing tied/no-switch control for the switch-timing study.</footer>
</main></body></html>"""
    _write_create_only_text(output_dir / "index.html", document)


def render(
    input_dir: Path,
    output_dir: Path,
    multiplicity_audit_path: Path,
    source_geometry_all_states_path: Path,
    release_path: Path,
) -> None:
    missing = [name for name in REQUIRED_FILES if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Analyzer output is incomplete; missing {missing} in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    release = json.loads(release_path.read_text())
    plot_module = str(release["materialization"]["plot_module"])
    actual_plot_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if release["implementation_files"][plot_module] != actual_plot_sha256:
        raise ValueError("Plot module drifted from the frozen completion release")
    source_geometry = _read_table(
        source_geometry_all_states_path.parent,
        source_geometry_all_states_path.name,
        {
            "row_id",
            "policy_role",
            "support_id",
            "checkpoint_label",
            "statistic",
            "geometry",
            "component",
            "cosine",
        },
    )
    frozen_h1 = _read_table(
        input_dir,
        "source_source_geometry.csv",
        {
            "row_id",
            "analysis_role",
            "policy_role",
            "support_id",
            "checkpoint_label",
            "statistic",
            "geometry",
            "component",
            "cosine",
        },
    )
    if set(frozen_h1["checkpoint_label"]) != set(H1_STATE_ORDER):
        raise ValueError("v10 H1 checkpoint inventory drifted from the five-state display contract")
    if not set(frozen_h1["row_id"]).issubset(set(source_geometry["row_id"])):
        raise ValueError("All-state source geometry does not contain every frozen H1 row")
    utilities = _read_table(
        input_dir,
        "target_source_utilities_visualization_only.csv",
        {
            "analysis_role",
            "policy_role",
            "support_id",
            "checkpoint_label",
            "target",
            "source",
            "geometry",
            "component",
            "cosine",
            "evidence_role",
        },
    )
    alignment = _read_table(
        input_dir,
        "target_source_choice_alignment_visualization_only.csv",
        {
            "analysis_role",
            "policy_role",
            "support_id",
            "checkpoint_label",
            "target",
            "contrast",
            "geometry",
            "component",
            "A_y",
            "evidence_role",
        },
    )
    summaries = [
        _read_table(
            input_dir,
            name,
            {
                "contrast",
                "mean",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "n_paired_seeds",
                "evidence_role",
                "alternative",
                "exact_sign_flip_p_unadjusted",
            },
        )
        for name in ("h2_h3_summary.csv", "h3_repetition_mechanism_summary.csv", "h5_profile_summary.csv")
    ]
    multiplicity = _read_table(
        multiplicity_audit_path.parent,
        multiplicity_audit_path.name,
        {"contrast", "exact_two_sided_sign_flip_p", "holm_p_across_47"},
    )
    _write_figure(
        source_conflict_trajectory_figure(source_geometry, frozen_h1),
        output_dir / "source_source_conflict_trajectory.html",
    )
    _write_figure(
        source_conflict_figure(source_geometry, frozen_h1),
        output_dir / "source_source_conflict_matrix.html",
    )
    _write_figure(target_utility_trajectory_figure(utilities), output_dir / "target_source_utility_trajectories.html")
    _write_figure(target_utility_figure(utilities), output_dir / "target_source_utility_matrix.html")
    _write_figure(source_choice_figure(alignment), output_dir / "target_source_choice_alignment.html")
    _write_figure(mechanism_forest_figure(summaries, multiplicity), output_dir / "mechanism_effect_forest.html")
    _write_index(output_dir)
    expected_html = {
        "index.html",
        "mechanism_effect_forest.html",
        "source_source_conflict_matrix.html",
        "source_source_conflict_trajectory.html",
        "target_source_choice_alignment.html",
        "target_source_utility_matrix.html",
        "target_source_utility_trajectories.html",
    }
    observed_html = {path.name for path in output_dir.glob("*.html")}
    if observed_html != expected_html:
        raise ValueError(f"Rendered HTML inventory drifted: {sorted(observed_html)} != {sorted(expected_html)}")
    render_manifest = {
        "release_sha256": release["release_sha256"],
        "plot_module_sha256": actual_plot_sha256,
        "files": {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in sorted(expected_html)},
    }
    _write_create_only_text(
        output_dir / "render_manifest.json",
        json.dumps(render_manifest, indent=2, sort_keys=True) + "\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory produced by the frozen analyzer")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination for self-contained HTML plots")
    parser.add_argument(
        "--multiplicity-audit",
        type=Path,
        required=True,
        help="Post-review recomputation table carrying two-sided global Holm p-values",
    )
    parser.add_argument(
        "--source-geometry-all-states",
        type=Path,
        required=True,
        help="Post-outcome descriptive source-source geometry at every repaired checkpoint",
    )
    parser.add_argument("--release-path", type=Path, required=True, help="Frozen plot-completion release")
    args = parser.parse_args()
    render(args.input_dir, args.output_dir, args.multiplicity_audit, args.source_geometry_all_states, args.release_path)


if __name__ == "__main__":
    main()
