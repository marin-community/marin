# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kaleido==0.2.1",
#   "numpy",
#   "pandas",
#   "plotly",
#   "tabulate",
# ]
# ///

"""Visualize the unweighted quartic-ridge WSD80 horizon-by-replay surfaces."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plot_starcoder_wsd80_dense_horizon_replay_gain_20260811 as raw_plot
import plot_starcoder_wsd80_dense_horizon_replay_surface_sensitivity_20260811 as sensitivity
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_dense_horizon_replay_surface_explorer_20260811"

DISPLAY_GRID_SIZE = 81
DISPLAY_DELTA_CAP = 0.15
TRACES_PER_BLOCK = 10

OBSERVED_COLOR = "#17324D"
RAW_TIED_COLOR = "#F4C542"
RAW_UNTIED_COLOR = "#D84A3A"
FITTED_TIED_COLOR = "#70C1B3"
FITTED_UNTIED_COLOR = "#145A76"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-policies", type=Path, default=raw_plot.DEFAULT_SELECTED_POLICIES)
    parser.add_argument("--coverage-observations", type=Path, default=raw_plot.DEFAULT_COVERAGE_OBSERVATIONS)
    parser.add_argument("--design", type=Path, default=raw_plot.DEFAULT_DESIGN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _display_surface(
    group: pd.DataFrame,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates = group[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
    target = group["bpb"].to_numpy(dtype=float)
    features = sensitivity._coordinates_to_features(coordinates)
    coefficients = sensitivity._ridge_operator(features, ridge) @ target

    axis = np.linspace(0.0, 1.0, DISPLAY_GRID_SIZE)
    p0, p1 = np.meshgrid(axis, axis, indexing="xy")
    candidates = np.column_stack((p0.ravel(), p1.ravel()))
    hull = sensitivity._convex_hull(coordinates)
    inside = sensitivity._inside_convex_hull(candidates, hull).reshape(p0.shape)
    prediction = (sensitivity._coordinates_to_features(candidates) @ coefficients).reshape(p0.shape)
    prediction[~inside] = np.nan
    fitted_at_observations = features @ coefficients
    return p0, p1, prediction, fitted_at_observations, target - fitted_at_observations


def _block_label(row: pd.Series) -> str:
    return (
        f"D={row['materialized_tokens_b']:.2f}B · "
        f"{raw_plot.SUPPORT_MARKER_LABELS[str(row['support_id'])]} StarCoder repetition"
    )


def _block_traces(
    group: pd.DataFrame,
    row: pd.Series,
    visible: bool,
) -> list[go.BaseTraceType]:
    p0, p1, prediction, fitted_observed, residual = _display_surface(group, float(row["selected_ridge"]))
    fitted_minimum = float(np.nanmin(prediction))
    display_delta = np.minimum(prediction - fitted_minimum, DISPLAY_DELTA_CAP)
    observed_delta = np.minimum(group["bpb"].to_numpy(dtype=float) - fitted_minimum, DISPLAY_DELTA_CAP)
    observed_custom = np.column_stack(
        (
            group["coordinate_id"],
            group["bpb"],
            fitted_observed,
            residual,
            group["aggregate_starcoder"],
            group["phase_contrast"],
        )
    )
    raw_p0 = np.asarray([row["tied_p0"], row["untied_p0"]], dtype=float)
    raw_p1 = np.asarray([row["tied_p1"], row["untied_p1"]], dtype=float)
    raw_bpb = np.asarray([row["tied_bpb"], row["untied_bpb"]], dtype=float)
    fitted_p0 = np.asarray([row["surface_tied_p"], row["surface_untied_p0"]], dtype=float)
    fitted_p1 = np.asarray([row["surface_tied_p"], row["surface_untied_p1"]], dtype=float)
    fitted_bpb = np.asarray([row["surface_tied_bpb"], row["surface_untied_bpb"]], dtype=float)

    contour = go.Contour(
        x=p0[0],
        y=p1[:, 0],
        z=display_delta,
        coloraxis="coloraxis",
        contours={"start": 0.0, "end": DISPLAY_DELTA_CAP, "size": 0.01, "coloring": "heatmap"},
        customdata=prediction,
        hovertemplate=("p0=%{x:.3f}<br>p1=%{y:.3f}<br>Fitted BPB=%{customdata:.6f}<extra></extra>"),
        showscale=False,
        visible=visible,
        showlegend=False,
    )
    observed_2d = go.Scatter(
        x=group["phase_0_starcoder"],
        y=group["phase_1_starcoder"],
        mode="markers",
        name="Observed coordinate",
        marker={
            "size": 7,
            "color": observed_delta,
            "coloraxis": "coloraxis",
            "line": {"color": OBSERVED_COLOR, "width": 0.8},
        },
        customdata=observed_custom,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>p0=%{x:.3f}; p1=%{y:.3f}<br>"
            "Observed BPB=%{customdata[1]:.6f}<br>Fitted BPB=%{customdata[2]:.6f}<br>"
            "Residual=%{customdata[3]:+.6f}<br>Aggregate=%{customdata[4]:.4f}; "
            "contrast=%{customdata[5]:+.4f}<extra></extra>"
        ),
        visible=visible,
        showlegend=True,
    )
    raw_tied_2d = go.Scatter(
        x=[raw_p0[0]],
        y=[raw_p1[0]],
        mode="markers",
        name="Raw tied minimum",
        marker={"size": 16, "symbol": "x", "color": RAW_TIED_COLOR, "line": {"width": 3}},
        hovertemplate=f"Raw tied minimum<br>p={raw_p0[0]:.4f}<br>{raw_bpb[0]:.6f} BPB<extra></extra>",
        visible=visible,
        showlegend=True,
    )
    raw_untied_2d = go.Scatter(
        x=[raw_p0[1]],
        y=[raw_p1[1]],
        mode="markers",
        name="Raw untied minimum",
        marker={"size": 17, "symbol": "star", "color": RAW_UNTIED_COLOR, "line": {"color": "white", "width": 1}},
        hovertemplate=(
            f"Raw untied minimum<br>p0={raw_p0[1]:.4f}; p1={raw_p1[1]:.4f}<br>" f"{raw_bpb[1]:.6f} BPB<extra></extra>"
        ),
        visible=visible,
        showlegend=True,
    )
    fitted_tied_2d = go.Scatter(
        x=[fitted_p0[0]],
        y=[fitted_p1[0]],
        mode="markers",
        name="Fitted tied minimum",
        marker={
            "size": 15,
            "symbol": "diamond-open",
            "color": FITTED_TIED_COLOR,
            "line": {"color": FITTED_TIED_COLOR, "width": 3},
        },
        hovertemplate=(f"Fitted tied minimum<br>p={fitted_p0[0]:.4f}<br>{fitted_bpb[0]:.6f} BPB<extra></extra>"),
        visible=visible,
        showlegend=True,
    )
    fitted_untied_2d = go.Scatter(
        x=[fitted_p0[1]],
        y=[fitted_p1[1]],
        mode="markers",
        name="Fitted untied minimum",
        marker={
            "size": 17,
            "symbol": "star-open",
            "color": FITTED_UNTIED_COLOR,
            "line": {"color": FITTED_UNTIED_COLOR, "width": 3},
        },
        hovertemplate=(
            f"Fitted untied minimum<br>p0={fitted_p0[1]:.4f}; p1={fitted_p1[1]:.4f}<br>"
            f"{fitted_bpb[1]:.6f} BPB<extra></extra>"
        ),
        visible=visible,
        showlegend=True,
    )
    surface = go.Surface(
        x=p0,
        y=p1,
        z=prediction,
        surfacecolor=display_delta,
        coloraxis="coloraxis",
        hovertemplate="p0=%{x:.3f}<br>p1=%{y:.3f}<br>Fitted BPB=%{z:.6f}<extra></extra>",
        opacity=0.83,
        visible=visible,
        showscale=False,
        showlegend=False,
    )
    observed_3d = go.Scatter3d(
        x=group["phase_0_starcoder"],
        y=group["phase_1_starcoder"],
        z=group["bpb"],
        mode="markers",
        name="Observed coordinates",
        marker={"size": 3.8, "color": OBSERVED_COLOR, "opacity": 0.88},
        customdata=observed_custom,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>p0=%{x:.3f}; p1=%{y:.3f}<br>"
            "Observed BPB=%{z:.6f}<br>Fitted BPB=%{customdata[2]:.6f}<br>"
            "Residual=%{customdata[3]:+.6f}<extra></extra>"
        ),
        visible=visible,
        showlegend=False,
    )
    raw_3d = go.Scatter3d(
        x=raw_p0,
        y=raw_p1,
        z=raw_bpb,
        mode="markers",
        name="Raw minima",
        marker={"size": 8, "symbol": "diamond", "color": [RAW_TIED_COLOR, RAW_UNTIED_COLOR]},
        text=["Raw tied minimum", "Raw untied minimum"],
        hovertemplate="%{text}<br>p0=%{x:.4f}; p1=%{y:.4f}<br>%{z:.6f} BPB<extra></extra>",
        visible=visible,
        showlegend=False,
    )
    fitted_3d = go.Scatter3d(
        x=fitted_p0,
        y=fitted_p1,
        z=fitted_bpb,
        mode="markers",
        name="Fitted minima",
        marker={"size": 9, "symbol": "diamond-open", "color": [FITTED_TIED_COLOR, FITTED_UNTIED_COLOR]},
        text=["Fitted tied minimum", "Fitted untied minimum"],
        hovertemplate="%{text}<br>p0=%{x:.4f}; p1=%{y:.4f}<br>%{z:.6f} BPB<extra></extra>",
        visible=visible,
        showlegend=False,
    )
    return [
        contour,
        observed_2d,
        raw_tied_2d,
        raw_untied_2d,
        fitted_tied_2d,
        fitted_untied_2d,
        surface,
        observed_3d,
        raw_3d,
        fitted_3d,
    ]


def build_figure(comparison: pd.DataFrame, coverage: pd.DataFrame) -> go.Figure:
    """Build a dropdown explorer over all horizon-by-repetition blocks."""
    ordered = comparison.sort_values(["rung", "support_order"]).reset_index(drop=True)
    default_index = int(ordered["surface_minus_raw_gain_bpb"].idxmax())
    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        column_widths=[0.46, 0.54],
        horizontal_spacing=0.035,
        subplot_titles=("<b>Top-down fitted surface</b>", "<b>Rotatable fitted surface and observations</b>"),
    )

    for block_index, row in ordered.iterrows():
        group = coverage.loc[
            coverage["cell_id"].eq(row["cell_id"]) & coverage["support_id"].eq(row["support_id"])
        ].sort_values("coordinate_id")
        traces = _block_traces(group, row, block_index == default_index)
        if len(traces) != TRACES_PER_BLOCK:
            raise AssertionError("Unexpected block trace count")
        for trace_index, trace in enumerate(traces):
            figure.add_trace(trace, row=1, col=1 if trace_index < 6 else 2)

    buttons = []
    total_traces = len(ordered) * TRACES_PER_BLOCK
    for block_index, row in ordered.iterrows():
        visible = [False] * total_traces
        start = block_index * TRACES_PER_BLOCK
        visible[start : start + TRACES_PER_BLOCK] = [True] * TRACES_PER_BLOCK
        buttons.append(
            {
                "label": _block_label(row),
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title.text": (
                            "<b>Failed unweighted quartic-ridge surface diagnostic</b><br>"
                            f"<sup>{_block_label(row)} · raw gain {row['raw_two_phase_gain_bpb']:+.6f} BPB · "
                            f"smooth gain {row['surface_global_two_phase_gain_bpb']:+.6f} BPB · "
                            f"spatial-CV RMSE {row['unweighted_spatial_cv_rmse']:.6f} BPB</sup>"
                        )
                    },
                ],
            }
        )

    default = ordered.loc[default_index]
    figure.update_layout(
        title={
            "text": (
                "<b>Failed unweighted quartic-ridge surface diagnostic</b><br>"
                f"<sup>{_block_label(default)} · raw gain {default['raw_two_phase_gain_bpb']:+.6f} BPB · "
                f"smooth gain {default['surface_global_two_phase_gain_bpb']:+.6f} BPB · "
                f"spatial-CV RMSE {default['unweighted_spatial_cv_rmse']:.6f} BPB</sup>"
            ),
            "x": 0.04,
            "xanchor": "left",
            "font": {"size": 28, "family": "Georgia, Times New Roman, serif", "color": raw_plot.PAPER_TEXT},
        },
        width=1900,
        height=1080,
        paper_bgcolor=raw_plot.PAPER_BACKGROUND,
        plot_bgcolor=raw_plot.PLOT_BACKGROUND,
        font={"family": "Avenir Next, Source Sans Pro, sans-serif", "size": 14, "color": raw_plot.PAPER_TEXT},
        margin={"l": 85, "r": 330, "t": 175, "b": 145},
        coloraxis={
            "colorscale": "RdYlGn_r",
            "cmin": 0.0,
            "cmax": DISPLAY_DELTA_CAP,
            "colorbar": {
                "title": {"text": "BPB above fitted<br>surface minimum<br>(clipped at 0.15)"},
                "x": 1.035,
                "len": 0.28,
                "y": 0.49,
            },
        },
        updatemenus=[
            {
                "buttons": buttons,
                "active": default_index,
                "direction": "down",
                "showactive": True,
                "x": 1.02,
                "xanchor": "left",
                "y": 0.24,
                "yanchor": "top",
                "bgcolor": raw_plot.PLOT_BACKGROUND,
                "bordercolor": raw_plot.GRID_COLOR,
                "font": {"size": 12},
            }
        ],
        legend={
            "x": 1.02,
            "xanchor": "left",
            "y": 0.99,
            "yanchor": "top",
            "bgcolor": "rgba(255,253,248,0.96)",
            "bordercolor": raw_plot.GRID_COLOR,
            "borderwidth": 1,
            "font": {"size": 12},
        },
        annotations=[
            *figure.layout.annotations,
            {
                "text": (
                    "Surface color shows fitted BPB above that block's fitted minimum; values above 0.15 are "
                    "clipped only for color readability. Observed points retain their actual BPB on the 3D axis. "
                    "This unweighted surface fails predictive sanity checks and is shown only to diagnose its "
                    "spurious basins."
                ),
                "x": 0.5,
                "xref": "paper",
                "y": -0.11,
                "yref": "paper",
                "showarrow": False,
                "xanchor": "center",
                "align": "center",
                "font": {"size": 12, "color": raw_plot.PAPER_TEXT},
            },
        ],
    )
    figure.update_xaxes(
        title_text="Phase 0 StarCoder weight",
        range=[0.0, 1.0],
        gridcolor=raw_plot.GRID_COLOR,
        scaleanchor="y",
        scaleratio=1,
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text="Phase 1 StarCoder weight",
        range=[0.0, 1.0],
        gridcolor=raw_plot.GRID_COLOR,
        row=1,
        col=1,
    )
    figure.update_scenes(
        xaxis_title="Phase 0 StarCoder weight",
        yaxis_title="Phase 1 StarCoder weight",
        zaxis_title="Programming Languages BPB",
        xaxis={"range": [0.0, 1.0], "backgroundcolor": raw_plot.PLOT_BACKGROUND},
        yaxis={"range": [0.0, 1.0], "backgroundcolor": raw_plot.PLOT_BACKGROUND},
        zaxis={"backgroundcolor": raw_plot.PLOT_BACKGROUND},
        camera={"eye": {"x": 1.45, "y": 1.45, "z": 0.95}},
        aspectmode="manual",
        aspectratio={"x": 1.0, "y": 1.0, "z": 0.75},
    )
    return figure


def write_report(output_dir: Path, comparison: pd.DataFrame) -> None:
    worst = comparison.loc[comparison["surface_minus_raw_gain_bpb"].idxmax()]
    lines = [
        "# StarCoder WSD80 unweighted surface explorer",
        "",
        "This artifact visualizes all 28 unweighted quartic-ridge fits as a failure diagnostic.",
        "",
        "- The fitted surfaces are not accepted estimates of the expected response.",
        "- Select any of four token horizons and seven StarCoder repetition regimes from the dropdown.",
        "- Each view overlays the 125 observations, raw grid minima, and fitted continuous minima.",
        (
            "- The default block has the largest surface-minus-raw discrepancy: "
            f"`{worst['cell_id']}/{worst['support_id']}`, {worst['surface_minus_raw_gain_bpb']:+.7f} BPB."
        ),
        (
            "- Generic Bayesian optimization should not target these fitted minima. First materialize the frozen "
            "calibration repeats, fit the preregistered heteroskedastic estimator, and verify that its spatial-CV "
            "error and optimum stability are commensurate with the effect size."
        ),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    comparison = sensitivity.load_comparison(
        args.selected_policies,
        args.coverage_observations,
        args.design,
    )
    coverage = pd.read_csv(args.coverage_observations)
    figure = build_figure(comparison, coverage)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        args.output_dir / "starcoder_wsd80_dense_horizon_replay_surface_explorer.html",
        include_plotlyjs=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "starcoder_wsd80_dense_horizon_replay_surface_explorer",
                "height": 2160,
                "width": 3800,
                "scale": 4,
            },
        },
    )
    figure.write_image(
        args.output_dir / "starcoder_wsd80_dense_horizon_replay_surface_explorer.png",
        width=1900,
        height=1080,
        scale=2,
    )
    write_report(args.output_dir, comparison)


if __name__ == "__main__":
    main()
