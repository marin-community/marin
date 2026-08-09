# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kaleido==0.2.1",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "tabulate",
# ]
# ///

"""Test and plot a pooled TPP collapse of the matched-N,D phase optima."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plot_starcoder_wsd80_matched_nd_optimum_scaling_20260802 import (
    CONFIRMATION_COLOR,
    DEFAULT_CONFIRMATION_SUMMARY,
    DEFAULT_DISCOVERY_SUMMARY,
    DEFAULT_FITTED_CANDIDATES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DESIGN,
    DEFAULT_STAGE1_SUMMARY,
    GRID_COLOR,
    PAPER_BACKGROUND,
    PAPER_TEXT,
    PLOT_BACKGROUND,
    TRACK_LABELS,
    TRACK_ORDER,
    _confirmation_summary,
    _tracks_for_cell,
    discovered_optima,
)
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from scipy.optimize import least_squares
from scipy.stats import spearmanr

GAIN_METRIC = "policy_class_gap_bpb"
DISTANCE_METRIC = "optimum_l2_distance"
MODEL_ORDER = ("constant", "log_linear", "power", "hill_asymptote")
MODEL_PARAMETER_COUNTS = {
    "constant": 1,
    "log_linear": 2,
    "power": 2,
    "hill_asymptote": 3,
}
TRACK_COLORS = dict(zip(TRACK_ORDER, sample_colorscale("RdYlGn_r", (0.88, 0.12, 0.50)), strict=True))
TRACK_SYMBOLS = {
    "increase_d": "square",
    "increase_n": "circle",
    "increase_nd": "diamond",
    "shared": "x",
}
EXPORT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "starcoder_wsd80_matched_nd_tpp_scaling",
        "height": 900,
        "width": 1700,
        "scale": 4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-summary", type=Path, default=DEFAULT_DISCOVERY_SUMMARY)
    parser.add_argument("--fitted-candidates", type=Path, default=DEFAULT_FITTED_CANDIDATES)
    parser.add_argument("--source-design", type=Path, default=DEFAULT_SOURCE_DESIGN)
    parser.add_argument("--stage1-summary", type=Path, default=DEFAULT_STAGE1_SUMMARY)
    parser.add_argument("--confirmation-summary", type=Path, default=DEFAULT_CONFIRMATION_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _exclusive_track(row: pd.Series) -> str:
    tracks = _tracks_for_cell(row)
    if len(tracks) == 1:
        return tracks[0]
    if set(tracks) == set(TRACK_ORDER):
        return "shared"
    raise ValueError(f"{row['cell_id']}: expected one branch or the shared base cell, got {tracks}")


def _hill_prediction(parameters: np.ndarray, x: np.ndarray) -> np.ndarray:
    asymptote, half_tpp, exponent = parameters
    log_ratio = exponent * (np.log(half_tpp) - np.log(x))
    return asymptote / (1.0 + np.exp(np.clip(log_ratio, -700.0, 700.0)))


def _power_prediction(parameters: np.ndarray, x: np.ndarray) -> np.ndarray:
    amplitude, exponent = parameters
    return np.exp(np.clip(np.log(amplitude) + exponent * np.log(x), -700.0, 700.0))


def _fit_hill(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y_max = max(float(y.max()), 1e-12)
    lower = np.log([y_max, float(x.min()) / 100.0, 0.05])
    upper = np.log([y_max * 1000.0, float(x.max()) * 100.0, 8.0])
    scale = max(float(y.std()), 1e-8)
    starts = [
        np.log([y_max * asymptote_scale, float(np.median(x)) * half_scale, exponent])
        for asymptote_scale in (1.001, 1.2, 2.0, 10.0)
        for half_scale in (0.1, 0.3, 1.0, 3.0, 10.0)
        for exponent in (0.2, 0.5, 1.0, 2.0, 4.0)
    ]
    best: tuple[float, np.ndarray] | None = None
    for start in starts:
        result = least_squares(
            lambda theta: (_hill_prediction(np.exp(theta), x) - y) / scale,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            max_nfev=20_000,
        )
        parameters = np.exp(result.x)
        residual_sum = float(np.sum((_hill_prediction(parameters, x) - y) ** 2))
        if best is None or residual_sum < best[0]:
            best = residual_sum, parameters
    if best is None:
        raise RuntimeError("Hill fit produced no candidate")
    return best[1]


def _fit_power(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    scale = max(float(y.std()), 1e-8)
    lower = np.array([-30.0, -4.0])
    upper = np.array([10.0, 4.0])
    starts = [
        np.array([np.log(max(float(np.median(y)), 1e-12)) - exponent * np.log(np.median(x)), exponent])
        for exponent in (-1.0, -0.2, 0.2, 0.5, 1.0, 2.0)
    ]
    best: tuple[float, np.ndarray] | None = None
    for start in starts:
        result = least_squares(
            lambda theta: (_power_prediction(np.array([np.exp(theta[0]), theta[1]]), x) - y) / scale,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            max_nfev=20_000,
        )
        parameters = np.array([np.exp(result.x[0]), result.x[1]])
        residual_sum = float(np.sum((_power_prediction(parameters, x) - y) ** 2))
        if best is None or residual_sum < best[0]:
            best = residual_sum, parameters
    if best is None:
        raise RuntimeError("Power fit produced no candidate")
    return best[1]


def _fit_model(model: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if model == "constant":
        return np.array([float(y.mean())])
    if model == "log_linear":
        design = np.column_stack([np.ones(len(x)), np.log(x)])
        return np.linalg.lstsq(design, y, rcond=None)[0]
    if model == "power":
        return _fit_power(x, y)
    if model == "hill_asymptote":
        return _fit_hill(x, y)
    raise ValueError(f"Unknown scaling model: {model}")


def _predict_model(model: str, parameters: np.ndarray, x: np.ndarray) -> np.ndarray:
    if model == "constant":
        return np.full_like(x, parameters[0], dtype=float)
    if model == "log_linear":
        return parameters[0] + parameters[1] * np.log(x)
    if model == "power":
        return _power_prediction(parameters, x)
    if model == "hill_asymptote":
        return _hill_prediction(parameters, x)
    raise ValueError(f"Unknown scaling model: {model}")


def _aicc(residual_sum: float, point_count: int, parameter_count: int) -> float:
    aic = point_count * np.log(max(residual_sum / point_count, 1e-300)) + 2.0 * parameter_count
    correction = 2.0 * parameter_count * (parameter_count + 1) / (point_count - parameter_count - 1)
    return float(aic + correction)


def _leave_one_out_rmse(model: str, x: np.ndarray, y: np.ndarray) -> float:
    predictions = np.empty(len(y), dtype=float)
    for held_out in range(len(y)):
        train = np.arange(len(y)) != held_out
        parameters = _fit_model(model, x[train], y[train])
        predictions[held_out] = _predict_model(model, parameters, x[[held_out]])[0]
    return float(np.sqrt(np.mean((predictions - y) ** 2)))


def _track_holdout_diagnostics(
    model: str,
    x: np.ndarray,
    y: np.ndarray,
    branches: np.ndarray,
) -> tuple[float, float]:
    squared_errors: list[float] = []
    fixed_n_bias = np.nan
    for branch in TRACK_ORDER:
        train = branches != branch
        test = branches == branch
        parameters = _fit_model(model, x[train], y[train])
        residual = _predict_model(model, parameters, x[test]) - y[test]
        squared_errors.extend((residual**2).tolist())
        if branch == "increase_d":
            fixed_n_bias = float(residual.mean())
    return float(np.sqrt(np.mean(squared_errors))), fixed_n_bias


def pooled_tpp_fits(optima: pd.DataFrame) -> pd.DataFrame:
    """Fit simple candidate collapses and retain diagnostics for every model."""
    frame = optima.copy()
    frame["exclusive_track"] = frame.apply(_exclusive_track, axis=1)
    x = frame["total_parameter_tpp"].to_numpy(dtype=float)
    branches = frame["exclusive_track"].to_numpy(dtype=str)
    rows: list[dict[str, object]] = []
    for metric in (GAIN_METRIC, DISTANCE_METRIC):
        y = frame[metric].to_numpy(dtype=float)
        rho, rho_p = spearmanr(x, y)
        total_sum = float(np.sum((y - y.mean()) ** 2))
        for model in MODEL_ORDER:
            parameters = _fit_model(model, x, y)
            fitted = _predict_model(model, parameters, x)
            residual_sum = float(np.sum((fitted - y) ** 2))
            track_rmse, fixed_n_bias = _track_holdout_diagnostics(model, x, y, branches)
            row: dict[str, object] = {
                "metric": metric,
                "model": model,
                "point_count": len(y),
                "parameter_count": MODEL_PARAMETER_COUNTS[model],
                "rmse": float(np.sqrt(residual_sum / len(y))),
                "leave_one_cell_out_rmse": _leave_one_out_rmse(model, x, y),
                "leave_one_track_out_rmse": track_rmse,
                "fixed_n_increase_d_holdout_bias": fixed_n_bias,
                "r_squared": 1.0 - residual_sum / total_sum,
                "aicc": _aicc(residual_sum, len(y), MODEL_PARAMETER_COUNTS[model]),
                "spearman_rho": float(rho),
                "spearman_p": float(rho_p),
                "amplitude": np.nan,
                "exponent": np.nan,
                "asymptote": np.nan,
                "half_tpp": np.nan,
                "asymptote_jackknife_min": np.nan,
                "asymptote_jackknife_max": np.nan,
                "asymptote_identified": False,
                "displayed": False,
            }
            if model == "constant":
                row["amplitude"] = parameters[0]
            elif model == "log_linear":
                row["amplitude"] = parameters[0]
                row["exponent"] = parameters[1]
            elif model == "power":
                row["amplitude"], row["exponent"] = parameters
            else:
                row["asymptote"], row["half_tpp"], row["exponent"] = parameters
                jackknife = []
                for held_out in range(len(y)):
                    train = np.arange(len(y)) != held_out
                    jackknife.append(_fit_hill(x[train], y[train])[0])
                row["asymptote_jackknife_min"] = float(np.min(jackknife))
                row["asymptote_jackknife_max"] = float(np.max(jackknife))
                row["asymptote_identified"] = bool(
                    parameters[0] <= 5.0 * float(y.max())
                    and parameters[1] <= 5.0 * float(x.max())
                    and max(jackknife) <= 5.0 * min(jackknife)
                )
            rows.append(row)
    fits = pd.DataFrame(rows)
    fits["delta_aicc"] = fits["aicc"] - fits.groupby("metric")["aicc"].transform("min")
    gain_hill = fits.loc[fits["metric"].eq(GAIN_METRIC) & fits["model"].eq("hill_asymptote")].iloc[0]
    monotone_rungs = []
    for rung in (1, 2, 3):
        rung_frame = frame.loc[frame["rung"].eq(rung)].sort_values("total_parameter_tpp")
        monotone_rungs.append(bool(np.all(np.diff(rung_frame[GAIN_METRIC].to_numpy(dtype=float)) > 0.0)))
    if gain_hill["spearman_rho"] < 0.8 or not all(monotone_rungs):
        raise ValueError("The pooled TPP ordering failed its descriptive plotting gate")
    return fits


def _fit_row(fits: pd.DataFrame, metric: str, model: str) -> pd.Series:
    selected = fits.loc[fits["metric"].eq(metric) & fits["model"].eq(model)]
    if len(selected) != 1:
        raise ValueError(f"Expected one pooled TPP fit for {metric}/{model}, found {len(selected)}")
    return selected.iloc[0]


def _point_custom_data(group: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            group["cell_id"],
            group["exclusive_track"].map({**TRACK_LABELS, "shared": "Shared base cell"}),
            group["rung"],
            group["total_parameters"] / 1e6,
            group["materialized_tokens"] / 1e9,
            group["compute_flops"] / 1e18,
            group["best_tied_p0"],
            group["best_tied_bpb"],
            group["two_phase_p0"],
            group["two_phase_p1"],
            group["aggregate_two_phase_starcoder"],
            group["two_phase_p1"] - group["two_phase_p0"],
            group["two_phase_bpb"],
            group[GAIN_METRIC],
            group[DISTANCE_METRIC],
            group["non_embedding_tpp"],
            group["bootstrap_positive_gain_probability"],
            group["confirmation_eligible"],
        ]
    )


def _point_hover_template(value_label: str) -> str:
    return (
        "<b>%{customdata[0]}</b><br>"
        "%{customdata[1]} · rung %{customdata[2]:.0f}<br>"
        "N: %{customdata[3]:.1f}M · D: %{customdata[4]:.3f}B<br>"
        "Compute: %{customdata[5]:.3f}e18 FLOPs<br>"
        "Total/non-embedding TPP: %{x:.3f} / %{customdata[15]:.3f}<br><br>"
        "Tied optimum p: %{customdata[6]:.3f} · BPB: %{customdata[7]:.6f}<br>"
        "Two-phase p0/p1: %{customdata[8]:.3f} / %{customdata[9]:.3f}<br>"
        "Aggregate: %{customdata[10]:.3f} · contrast: %{customdata[11]:+.3f}<br>"
        "Two-phase BPB: %{customdata[12]:.6f}<br>"
        "Gain: %{customdata[13]:.6f} BPB · L2: %{customdata[14]:.6f}<br>"
        "Model-bootstrap P(gain&gt;0): %{customdata[16]:.3f}<br>"
        "Passes frozen discovery gate: %{customdata[17]}<br>"
        f"<b>{value_label}: %{{y:.6f}}</b><extra></extra>"
    )


def build_figure(optima: pd.DataFrame, fits: pd.DataFrame, confirmation: pd.DataFrame) -> go.Figure:
    """Build a TPP ordering plot without asserting a universal pooled law."""
    frame = optima.copy()
    frame["exclusive_track"] = frame.apply(_exclusive_track, axis=1)
    figure = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.13,
        subplot_titles=("Two-phase performance gain", "Where the fitted optima allocate StarCoder"),
    )

    for rung in (1, 2, 3):
        rung_frame = frame.loc[frame["rung"].eq(rung)].sort_values("total_parameter_tpp")
        compute = float(rung_frame["compute_flops"].mean()) / 1e18
        figure.add_trace(
            go.Scatter(
                x=rung_frame["total_parameter_tpp"],
                y=rung_frame[GAIN_METRIC],
                mode="lines",
                name="Matched-compute rung",
                legendgroup="matched-compute",
                showlegend=rung == 1,
                line={"color": GRID_COLOR, "width": 2.0},
                hovertemplate=(
                    f"<b>Matched-compute rung {rung}</b><br>"
                    f"Compute: {compute:.3f}e18 FLOPs<br>"
                    "The connector orders the three N,D interventions by total-parameter TPP."
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    legend_order = ("increase_n", "increase_nd", "increase_d", "shared")
    for track in legend_order:
        group = frame.loc[frame["exclusive_track"].eq(track)].sort_values("total_parameter_tpp")
        if group.empty:
            continue
        color = PAPER_TEXT if track == "shared" else TRACK_COLORS[track]
        label = "Shared base cell" if track == "shared" else TRACK_LABELS[track]
        figure.add_trace(
            go.Scatter(
                x=group["total_parameter_tpp"],
                y=group[GAIN_METRIC],
                mode="markers",
                name=label,
                legendgroup=track,
                marker={
                    "color": color,
                    "size": 14 if track == "shared" else 11,
                    "symbol": TRACK_SYMBOLS[track],
                    "line": {"color": PAPER_TEXT, "width": 1.2},
                },
                customdata=_point_custom_data(group),
                hovertemplate=_point_hover_template("Smooth two-phase gain"),
            ),
            row=1,
            col=1,
        )

    shared = frame.loc[frame["exclusive_track"].eq("shared")]
    for track in TRACK_ORDER:
        branch = frame.loc[frame["exclusive_track"].eq(track)]
        trajectory = pd.concat([shared, branch]).sort_values("total_parameter_tpp")
        color = TRACK_COLORS[track]
        for metric, dash in (("best_tied_p0", "solid"), ("aggregate_two_phase_starcoder", "dot")):
            figure.add_trace(
                go.Scatter(
                    x=trajectory["total_parameter_tpp"],
                    y=trajectory[metric],
                    mode="lines",
                    legendgroup=f"policy-trajectory-{track}",
                    showlegend=False,
                    line={"color": color, "width": 2.0, "dash": dash},
                    hoverinfo="skip",
                ),
                row=1,
                col=2,
            )

    segment_x: list[float | None] = []
    segment_y: list[float | None] = []
    for _, row in frame.iterrows():
        segment_x.extend([float(row["total_parameter_tpp"]), float(row["total_parameter_tpp"]), None])
        segment_y.extend([float(row["best_tied_p0"]), float(row["aggregate_two_phase_starcoder"]), None])
    figure.add_trace(
        go.Scatter(
            x=segment_x,
            y=segment_y,
            mode="lines",
            showlegend=False,
            line={"color": GRID_COLOR, "width": 1.4},
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    for track in legend_order:
        group = frame.loc[frame["exclusive_track"].eq(track)].sort_values("total_parameter_tpp")
        if group.empty:
            continue
        color = PAPER_TEXT if track == "shared" else TRACK_COLORS[track]
        for metric, symbol, value_label in (
            ("best_tied_p0", "circle-open", "Tied optimum StarCoder fraction"),
            ("aggregate_two_phase_starcoder", "diamond", "Two-phase optimum aggregate StarCoder fraction"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=group["total_parameter_tpp"],
                    y=group[metric],
                    mode="markers",
                    showlegend=False,
                    marker={
                        "color": color,
                        "size": 11,
                        "symbol": symbol,
                        "line": {"color": PAPER_TEXT, "width": 1.2},
                    },
                    customdata=_point_custom_data(group),
                    hovertemplate=_point_hover_template(value_label),
                ),
                row=1,
                col=2,
            )
    for name, symbol in (("Tied optimum", "circle-open"), ("Two-phase optimum aggregate", "diamond")):
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=name,
                legendgroup=f"policy-class-{name}",
                marker={"color": PAPER_TEXT, "size": 11, "symbol": symbol},
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )

    if not confirmation.empty:
        merged = confirmation.merge(frame, on="cell_id", how="left", validate="one_to_one")
        mean = merged["mean_gain_bpb"].to_numpy(dtype=float)
        figure.add_trace(
            go.Scatter(
                x=merged["total_parameter_tpp"],
                y=mean,
                mode="markers",
                name="Fresh-seed confirmation",
                marker={
                    "color": CONFIRMATION_COLOR,
                    "size": 17,
                    "symbol": "star",
                    "line": {"color": PLOT_BACKGROUND, "width": 1.4},
                },
                error_y={
                    "type": "data",
                    "array": merged["ci95_high"].to_numpy(dtype=float) - mean,
                    "arrayminus": mean - merged["ci95_low"].to_numpy(dtype=float),
                    "color": CONFIRMATION_COLOR,
                    "thickness": 1.6,
                },
                customdata=np.column_stack(
                    [merged["cell_id"], merged["ci95_low"], merged["ci95_high"], merged["confirmed"]]
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Total-parameter TPP: %{x:.3f}<br>"
                    "Paired mean gain: %{y:.6f} BPB<br>"
                    "95% CI: [%{customdata[1]:.6f}, %{customdata[2]:.6f}]<br>"
                    "Frozen confirmation passed: %{customdata[3]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    gain_upper = max(
        float(frame[GAIN_METRIC].max()),
        float(confirmation["ci95_high"].max()) if not confirmation.empty else 0.0,
    )
    for column in (1, 2):
        figure.update_xaxes(
            type="log",
            title_text="Total-parameter tokens per parameter (TPP)",
            gridcolor=GRID_COLOR,
            showline=True,
            linecolor=PAPER_TEXT,
            ticks="outside",
            row=1,
            col=column,
        )
    figure.update_yaxes(
        title_text="Fitted tied - two-phase optimum BPB<br><sup>larger is better</sup>",
        range=[0.0, 1.25 * gain_upper],
        tickformat=".4f",
        gridcolor=GRID_COLOR,
        zeroline=True,
        zerolinecolor=PAPER_TEXT,
        showline=True,
        linecolor=PAPER_TEXT,
        ticks="outside",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text="Optimal aggregate StarCoder fraction",
        range=[0.0, 0.82],
        tickformat=".2f",
        gridcolor=GRID_COLOR,
        zeroline=True,
        zerolinecolor=PAPER_TEXT,
        showline=True,
        linecolor=PAPER_TEXT,
        ticks="outside",
        row=1,
        col=2,
    )
    gain_hill = _fit_row(fits, GAIN_METRIC, "hill_asymptote")
    gain_power = _fit_row(fits, GAIN_METRIC, "power")
    non_embedding_rho = spearmanr(frame["non_embedding_tpp"], frame[GAIN_METRIC]).statistic
    figure.update_layout(
        title={
            "text": (
                "StarCoder WSD80 phase optima ordered by TPP"
                "<br><sup>10 unique N,D cells · faint connectors join each matched-compute N,D triplet</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
            "font": {"family": "Georgia, serif", "size": 28, "color": PAPER_TEXT},
        },
        width=1700,
        height=900,
        paper_bgcolor=PAPER_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font={"family": "Avenir Next, Helvetica Neue, sans-serif", "size": 14, "color": PAPER_TEXT},
        hoverlabel={"font": {"family": "Avenir Next, Helvetica Neue, sans-serif", "size": 13}},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.02,
            "yanchor": "bottom",
            "bgcolor": "rgba(255,255,255,0.78)",
            "bordercolor": GRID_COLOR,
            "borderwidth": 1,
        },
        margin={"l": 115, "r": 65, "t": 180, "b": 180},
        annotations=[
            *figure.layout.annotations,
            {
                "text": (
                    "Within every matched-compute rung, discovered gain rises monotonically with total-parameter TPP. "
                    f"Pooled Spearman rho={gain_hill['spearman_rho']:.3f}; "
                    f"non-embedding TPP gives rho={non_embedding_rho:.3f}.<br>"
                    "No pooled law is drawn: TPP is confounded with the N,D intervention, and the Hill asymptote is "
                    f"not distinguished from a power curve "
                    f"(|Delta AICc|={abs(gain_hill['aicc'] - gain_power['aicc']):.2f}). "
                    "Only the highest-TPP cell has fresh-seed confirmation."
                ),
                "x": 0.5,
                "xref": "paper",
                "y": -0.22,
                "yref": "paper",
                "showarrow": False,
                "xanchor": "center",
                "align": "center",
                "font": {"size": 13, "color": PAPER_TEXT},
            },
        ],
    )
    return figure


def write_report(
    output_dir: Path,
    optima: pd.DataFrame,
    fits: pd.DataFrame,
    confirmation: pd.DataFrame,
) -> None:
    gain_hill = _fit_row(fits, GAIN_METRIC, "hill_asymptote")
    gain_power = _fit_row(fits, GAIN_METRIC, "power")
    distance_hill = _fit_row(fits, DISTANCE_METRIC, "hill_asymptote")
    distance_power = _fit_row(fits, DISTANCE_METRIC, "power")
    table = optima.copy()
    table["track"] = table.apply(_exclusive_track, axis=1)
    table = table.sort_values("total_parameter_tpp")
    non_embedding_rho = spearmanr(table["non_embedding_tpp"], table[GAIN_METRIC]).statistic
    triplet_rows = []
    for rung in (1, 2, 3):
        triplet = table.loc[table["rung"].eq(rung)].sort_values("total_parameter_tpp")
        triplet_rows.append(
            {
                "rung": rung,
                "compute_e18": float(triplet["compute_flops"].mean()) / 1e18,
                "total_tpp_order": " < ".join(f"{value:.2f}" for value in triplet["total_parameter_tpp"]),
                "gain_order_bpb": " < ".join(f"{value:.6f}" for value in triplet[GAIN_METRIC]),
                "strictly_monotone": bool(np.all(np.diff(triplet[GAIN_METRIC].to_numpy(dtype=float)) > 0.0)),
            }
        )
    triplets = pd.DataFrame(triplet_rows)
    confirmed_mean = float(confirmation["mean_gain_bpb"].iloc[0]) if not confirmation.empty else np.nan
    confirmed_high = float(confirmation["ci95_high"].iloc[0]) if not confirmation.empty else np.nan
    lines = [
        "# StarCoder WSD80 pooled TPP scaling audit",
        "",
        "- Unit of analysis: ten unique matched-N,D cells. The shared base cell is not duplicated across tracks.",
        "- The strongest design-supported result is within-rung ordering: all three matched-compute triplets have "
        "strictly increasing discovered gain as total-parameter TPP increases. The nominal permutation probability "
        "of three monotone triplets is (1/6)^3 = 0.00463, but the repeated track structure means the triplets are not "
        "independent draws.",
        "- Pooled total-parameter TPP has Spearman rho "
        f"{gain_hill['spearman_rho']:.4f} with smooth gain. Non-embedding TPP gives rho {non_embedding_rho:.4f}; "
        "the TPP definition therefore matters.",
        "- No universal pooled curve is displayed. The design has no interior TPP overlap between different N,D "
        "tracks, so a TPP-only law is confounded with model size and token horizon.",
        "- Sensitivity only: the positive Hill fit g(TPP) = g_inf / [1 + (TPP_50 / TPP)^alpha] gives "
        f"g_inf = {gain_hill['asymptote']:.6f} BPB, TPP_50 = {gain_hill['half_tpp']:.3f}, "
        f"alpha = {gain_hill['exponent']:.3f}, R2 = {gain_hill['r_squared']:.3f}, and "
        f"leave-one-cell-out RMSE = {gain_hill['leave_one_cell_out_rmse']:.6f} BPB.",
        "- The asymptote is not established. Its AICc advantage over the two-parameter power curve is only "
        f"{gain_power['aicc'] - gain_hill['aicc']:.2f}; the jackknife asymptote spans "
        f"[{gain_hill['asymptote_jackknife_min']:.6f}, {gain_hill['asymptote_jackknife_max']:.6f}] BPB. The sole "
        f"fresh-seed mean is {confirmed_mean:.6f}, with upper CI {confirmed_high:.6f}, already at or above the "
        "fitted ceiling.",
        "- TPP is not sufficient by itself. Leaving out the fixed-N/increasing-D branch causes a gain bias of "
        f"{gain_hill['fixed_n_increase_d_holdout_bias']:+.6f} BPB on that branch.",
        "- L2 separation is omitted from the pooled figure because it has exactly the same rank order as gain and is "
        "already near its geometric ceiling in the highest-TPP cell. A finite L2 asymptote is not identified "
        f"(asymptote_identified={distance_hill['asymptote_identified']}). The linear-space NLS sensitivity fit is "
        f"d(TPP) = {distance_power['amplitude']:.6f} TPP^{distance_power['exponent']:.3f}, "
        f"R2 = {distance_power['r_squared']:.3f}, leave-one-cell-out RMSE = "
        f"{distance_power['leave_one_cell_out_rmse']:.6f}.",
        "- Because the untied class nests the tied class and both optima are selected on fitted surfaces, small "
        "positive discovery gains carry winner's-curse bias. The model-bootstrap positive-gain probability remains "
        "in hover text.",
        "- Only the high-TPP fixed-N endpoint has fresh-seed confirmation. All other points are discovery-surface "
        "estimates from one shared reference seed.",
        "",
        "## Matched-compute triplets",
        "",
        triplets.to_markdown(index=False, floatfmt=".7f"),
        "",
        "## Cells",
        "",
        table[
            [
                "cell_id",
                "track",
                "total_parameters",
                "materialized_tokens",
                "compute_flops",
                "total_parameter_tpp",
                "non_embedding_tpp",
                GAIN_METRIC,
                DISTANCE_METRIC,
                "best_tied_p0",
                "aggregate_two_phase_starcoder",
            ]
        ].to_markdown(index=False, floatfmt=".7f"),
        "",
        "## Candidate fits",
        "",
        fits.to_markdown(index=False, floatfmt=".7f"),
        "",
    ]
    (output_dir / "pooled_tpp_scaling_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    optima = discovered_optima(
        args.discovery_summary,
        args.fitted_candidates,
        args.source_design,
        args.stage1_summary,
    )
    fits = pooled_tpp_fits(optima)
    confirmation = _confirmation_summary(args.confirmation_summary)
    figure = build_figure(optima, fits, confirmation)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fits.to_csv(args.output_dir / "pooled_tpp_scaling_fits.csv", index=False)
    (args.output_dir / "pooled_tpp_scaling_fit_manifest.json").write_text(
        json.dumps(
            {
                "cell_count": len(optima),
                "displayed_scaling_model": None,
                "sensitivity_models": {"gain": "hill_asymptote", "distance": "power"},
                "claim_boundary": "matched-compute TPP ordering; no universal pooled scaling law identified",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    figure.write_html(
        args.output_dir / "starcoder_wsd80_matched_nd_tpp_scaling.html",
        include_plotlyjs=True,
        config=EXPORT_CONFIG,
    )
    figure.write_image(args.output_dir / "starcoder_wsd80_matched_nd_tpp_scaling.png", scale=2)
    write_report(args.output_dir, optima, fits, confirmation)


if __name__ == "__main__":
    main()
