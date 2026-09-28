# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
# ]
# ///
"""Render completed Stage-3 StarCoder matched-N,D surfaces and scaling diagnostics."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull, Delaunay

SCRIPT_DIR = Path(__file__).resolve().parent
PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_matched_nd_stage1_20260731"
RESULTS_DIR = PANEL_DIR / "stage3_dense_surface_results_20260802"
DEFAULT_OBSERVATIONS = RESULTS_DIR / "combined_discovery_observations.csv"
DEFAULT_CANDIDATES = RESULTS_DIR / "fitted_surface_candidates.csv"
DEFAULT_SUMMARY = RESULTS_DIR / "cell_discovery_summary.csv"
DEFAULT_SOURCE_DESIGN = PANEL_DIR / "stage2_results_20260801" / "source_design.json"
DEFAULT_CONFIRMATION = PANEL_DIR / "confirmation_results_20260801" / "cell_confirmation_summary.csv"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "visualizations_20260803"

TRACK_ORDER = ("increase_d", "increase_n", "increase_nd")
TRACK_LABELS = {
    "increase_d": "Fixed N, increase D",
    "increase_n": "Fixed D, increase N",
    "increase_nd": "Increase N and D",
}
TRACK_COLORS = {
    "increase_d": "#177E89",
    "increase_n": "#D95F32",
    "increase_nd": "#7A5195",
}
PAPER = "#F7F3E8"
PANEL = "#FFFDF8"
INK = "#17324D"
MUTED = "#617386"
GRID = "#D8D1C2"
ORANGE = "#D95F32"
GREEN = "#177E89"
PHASE_0_FRACTION = 0.8
GAIN_GATE = 0.005
HEADLINE_CELL = "r3_increase_d_h0640_s28260"
EXPORT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--source-design", type=Path, default=DEFAULT_SOURCE_DESIGN)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _track_memberships(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    if not isinstance(value, str):
        raise ValueError(f"Invalid track memberships: {value!r}")
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Invalid track memberships: {value!r}")
    return tuple(str(item) for item in parsed)


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observations = pd.read_csv(args.observations)
    candidates = pd.read_csv(args.candidates)
    summary = pd.read_csv(args.summary)
    confirmation = pd.read_csv(args.confirmation)
    source = json.loads(args.source_design.read_text(encoding="utf-8"))
    cells = pd.DataFrame(source.get("source_cells", source.get("cells")))

    if len(observations) != 714 or observations["cell_id"].nunique() != 10:
        raise ValueError("Expected the complete 714-row, 10-cell discovery panel")
    if len(candidates) != 10 or len(summary) != 10:
        raise ValueError("Expected one candidate and one summary row per cell")
    if len(confirmation) != 1 or not bool(confirmation.iloc[0]["confirmed"]):
        raise ValueError("Expected the completed one-cell fresh-seed confirmation")
    if not np.isfinite(observations["starcoder_bpb"]).all():
        raise ValueError("Observation BPB contains non-finite values")
    if set(observations["cell_id"]) != set(candidates["cell_id"]) or set(candidates["cell_id"]) != set(cells["cell_id"]):
        raise ValueError("Cell identities disagree across inputs")

    cells["track_memberships"] = cells["track_memberships"].map(_track_memberships)
    cells["total_parameter_tpp"] = cells["materialized_tokens"] / cells["total_parameters"]
    merged = cells.merge(candidates, on="cell_id", validate="one_to_one").merge(
        summary.drop(columns=["rung", "hidden_size", "materialized_tokens", "total_parameters", "total_parameter_tpp"]),
        on="cell_id",
        validate="one_to_one",
    )
    return observations, merged, confirmation, cells


def _cell_order(cells: pd.DataFrame) -> list[str]:
    order: list[str] = []
    for track in TRACK_ORDER:
        subset = cells.loc[
            cells["track_memberships"].map(lambda tracks, selected=track: selected in tracks)
        ].sort_values("rung")
        for cell_id in subset["cell_id"].astype(str):
            if cell_id not in order:
                order.append(cell_id)
    if len(order) != 10:
        raise ValueError(f"Expected 10 cells, found {len(order)}")
    order.remove(HEADLINE_CELL)
    return [HEADLINE_CELL, *order]


def _hover(frame: pd.DataFrame) -> list[str]:
    return [
        "<br>".join(
            [
                f"{row.source_stage} · {row.selection_label}",
                f"phase 0: {row.phase_0_starcoder:.5f}",
                f"phase 1: {row.phase_1_starcoder:.5f}",
                f"80/20 aggregate: {row.aggregate_starcoder:.5f}",
                f"contrast p1-p0: {row.phase_contrast:+.5f}",
                f"BPB: {row.starcoder_bpb:.6f}",
                f"run: {row.run_name}",
            ]
        )
        for row in frame.itertuples(index=False)
    ]


def _surface_traces(frame: pd.DataFrame, result: pd.Series, visible: bool) -> list[go.BaseTraceType]:
    frame = frame.reset_index(drop=True)
    points = frame[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
    triangles = Delaunay(points).simplices
    hull = ConvexHull(points)
    z = frame["starcoder_bpb"].to_numpy(dtype=float)
    color_max = min(float(np.quantile(z, 0.82)), float(z.min() + 0.05))
    color_max = max(color_max, float(z.min() + 0.005))

    tied = frame.loc[frame["policy_class"].eq("tied")].sort_values("phase_0_starcoder")
    primary = frame.loc[frame["selection_label"].isin(["primary_fiber", "primary_tied_anchor"])].sort_values(
        "phase_contrast"
    )
    secondary = frame.loc[frame["selection_label"].isin(["secondary_fiber", "secondary_tied_anchor"])].sort_values(
        "phase_contrast"
    )
    raw_best = frame.loc[frame["starcoder_bpb"].idxmin()]
    fitted = pd.DataFrame(
        [
            {
                "label": "smooth untied candidate",
                "p0": result["fitted_untied_p0"],
                "p1": result["fitted_untied_p1"],
                "bpb": result["fitted_untied_bpb"],
            },
            {
                "label": "smooth tied comparator",
                "p0": result["fitted_tied_weight"],
                "p1": result["fitted_tied_weight"],
                "bpb": result["fitted_tied_bpb"],
            },
        ]
    )
    hull_path = [*hull.vertices.tolist(), int(hull.vertices[0])]

    traces: list[go.BaseTraceType] = [
        go.Mesh3d(
            x=frame["phase_0_starcoder"],
            y=frame["phase_1_starcoder"],
            z=z,
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            intensity=z,
            colorscale="RdYlGn_r",
            cmin=float(z.min()),
            cmax=color_max,
            opacity=0.58,
            flatshading=False,
            name="observed triangulation",
            hoverinfo="skip",
            showscale=False,
            visible=visible,
            legendgroup="surface",
        ),
        go.Scatter3d(
            x=frame["phase_0_starcoder"],
            y=frame["phase_1_starcoder"],
            z=z,
            mode="markers",
            marker={
                "size": np.where(frame["source_stage"].eq("stage3"), 4.8, 3.5),
                "color": z,
                "colorscale": "RdYlGn_r",
                "cmin": float(z.min()),
                "cmax": color_max,
                "line": {"width": 0.4, "color": PANEL},
            },
            text=_hover(frame),
            hovertemplate="%{text}<extra></extra>",
            name="observed checkpoints",
            visible=visible,
            legendgroup="observed",
        ),
        go.Scatter3d(
            x=tied["phase_0_starcoder"],
            y=tied["phase_1_starcoder"],
            z=tied["starcoder_bpb"],
            mode="lines",
            line={"color": INK, "width": 6},
            name="tied diagonal",
            hoverinfo="skip",
            visible=visible,
            legendgroup="tied",
        ),
        go.Scatter3d(
            x=primary["phase_0_starcoder"],
            y=primary["phase_1_starcoder"],
            z=primary["starcoder_bpb"],
            mode="lines+markers",
            line={"color": "#4C78A8", "width": 7},
            marker={"size": 3},
            name="primary one-sided fiber",
            hoverinfo="skip",
            visible=visible,
            legendgroup="primary",
        ),
        go.Scatter3d(
            x=secondary["phase_0_starcoder"],
            y=secondary["phase_1_starcoder"],
            z=secondary["starcoder_bpb"],
            mode="lines+markers",
            line={"color": "#F2B134", "width": 6},
            marker={"size": 3},
            name="secondary one-sided fiber",
            hoverinfo="skip",
            visible=visible,
            legendgroup="secondary",
        ),
        go.Scatter3d(
            x=fitted["p0"],
            y=fitted["p1"],
            z=fitted["bpb"],
            mode="markers",
            marker={"size": 8, "symbol": "diamond", "color": [ORANGE, INK], "line": {"width": 2, "color": PANEL}},
            text=fitted["label"],
            hovertemplate="%{text}<br>p0=%{x:.4f}<br>p1=%{y:.4f}<br>fitted BPB=%{z:.6f}<extra></extra>",
            name="smooth fitted optima",
            visible=visible,
            legendgroup="fitted",
        ),
        go.Scatter3d(
            x=[raw_best["phase_0_starcoder"]],
            y=[raw_best["phase_1_starcoder"]],
            z=[raw_best["starcoder_bpb"]],
            mode="markers",
            marker={"size": 8, "symbol": "x", "color": "#111827"},
            text=["raw observed minimum; selection-biased"],
            hovertemplate="%{text}<br>p0=%{x:.4f}<br>p1=%{y:.4f}<br>BPB=%{z:.6f}<extra></extra>",
            name="raw observed minimum",
            visible=visible,
            legendgroup="raw",
        ),
        go.Scatter(
            x=frame["phase_0_starcoder"],
            y=frame["phase_1_starcoder"],
            mode="markers",
            marker={
                "size": np.where(frame["source_stage"].eq("stage3"), 8, 6),
                "color": z,
                "colorscale": "RdYlGn_r",
                "cmin": float(z.min()),
                "cmax": color_max,
                "colorbar": {"title": "BPB", "len": 0.72, "x": 1.01},
                "line": {"width": 0.7, "color": PANEL},
            },
            text=_hover(frame),
            hovertemplate="%{text}<extra></extra>",
            name="observed checkpoints",
            showlegend=False,
            visible=visible,
            legendgroup="observed",
        ),
        go.Scatter(
            x=points[hull_path, 0],
            y=points[hull_path, 1],
            mode="lines",
            line={"color": MUTED, "width": 1.2, "dash": "dot"},
            name="empirical support hull",
            hoverinfo="skip",
            showlegend=False,
            visible=visible,
        ),
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            line={"color": INK, "width": 1.5},
            name="tied diagonal",
            hoverinfo="skip",
            showlegend=False,
            visible=visible,
            legendgroup="tied",
        ),
        go.Scatter(
            x=fitted["p0"],
            y=fitted["p1"],
            mode="markers",
            marker={
                "size": 13,
                "symbol": ["diamond", "square"],
                "color": [ORANGE, INK],
                "line": {"width": 2, "color": PANEL},
            },
            text=fitted["label"],
            hovertemplate="%{text}<br>p0=%{x:.4f}<br>p1=%{y:.4f}<extra></extra>",
            name="smooth fitted optima",
            showlegend=False,
            visible=visible,
            legendgroup="fitted",
        ),
    ]
    return traces


def render_surface_explorer(
    observations: pd.DataFrame, results: pd.DataFrame, cells: pd.DataFrame, output: Path
) -> None:
    order = _cell_order(cells)
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.62, 0.38],
        horizontal_spacing=0.035,
    )
    traces_per_cell = 11
    titles: list[str] = []
    for index, cell_id in enumerate(order):
        result = results.loc[results["cell_id"].eq(cell_id)].iloc[0]
        frame = observations.loc[observations["cell_id"].eq(cell_id)]
        traces = _surface_traces(frame, result, visible=index == 0)
        for trace_index, trace in enumerate(traces):
            fig.add_trace(trace, row=1, col=1 if trace_index < 7 else 2)
        eligibility = "passes frozen gain gate" if bool(result["confirmation_eligible"]) else "does not pass gate"
        titles.append(
            f"{cell_id} · N={int(result['total_parameters']) / 1e6:.1f}M · "
            f"D={int(result['materialized_tokens']) / 1e9:.2f}B · TPP={result['total_parameter_tpp']:.2f}<br>"
            f"<sup>smooth gain {result['fitted_gain_tied_minus_untied_bpb']:.4f} BPB · {eligibility} · "
            f"candidate-location bootstrap p90 L2={result['bootstrap_candidate_l2_p90']:.3f}</sup>"
        )

    buttons = []
    for cell_index, cell_id in enumerate(order):
        visible = [False] * len(fig.data)
        start = cell_index * traces_per_cell
        visible[start : start + traces_per_cell] = [True] * traces_per_cell
        buttons.append(
            {
                "label": cell_id,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": {"text": titles[cell_index], "x": 0.5, "xanchor": "center"}},
                ],
            }
        )

    fig.update_layout(
        title={"text": titles[0], "x": 0.5, "xanchor": "center"},
        paper_bgcolor=PAPER,
        plot_bgcolor=PANEL,
        font={"family": "Avenir Next, Avenir, sans-serif", "color": INK, "size": 13},
        height=900,
        margin={"l": 18, "r": 58, "t": 120, "b": 70},
        legend={"orientation": "h", "y": -0.06, "x": 0.0, "bgcolor": "rgba(255,253,248,.8)"},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.11,
                "yanchor": "top",
                "showactive": True,
                "bgcolor": PANEL,
                "bordercolor": GRID,
            }
        ],
        annotations=[
            {
                "text": (
                    "All 714 discovery observations are shown. The mesh is linear triangulation of measured BPB, not a "
                    "surrogate. Colors saturate 0.05 BPB above each cell minimum so the optimum region remains visible. "
                    "Raw minima are selection-biased; diamonds/squares are the preregistered smooth fit."
                ),
                "x": 0.5,
                "y": -0.12,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 12, "color": MUTED},
                "align": "left",
            }
        ],
    )
    fig.update_scenes(
        xaxis={"title": "Phase 0 StarCoder", "range": [0, 1], "backgroundcolor": PANEL, "gridcolor": GRID},
        yaxis={"title": "Phase 1 StarCoder", "range": [0, 1], "backgroundcolor": PANEL, "gridcolor": GRID},
        zaxis={"title": "Programming BPB", "backgroundcolor": PANEL, "gridcolor": GRID},
        aspectmode="manual",
        aspectratio={"x": 1.0, "y": 1.0, "z": 0.75},
        camera={"eye": {"x": 1.45, "y": -1.65, "z": 1.0}},
    )
    fig.update_xaxes(title="Phase 0 StarCoder", range=[0, 1], gridcolor=GRID, zeroline=False, row=1, col=2)
    fig.update_yaxes(title="Phase 1 StarCoder", range=[0, 1], gridcolor=GRID, zeroline=False, row=1, col=2)
    pio.write_html(fig, output, include_plotlyjs=True, full_html=True, config=EXPORT_CONFIG)


def _track_cells(cells: pd.DataFrame, track: str) -> pd.DataFrame:
    return cells.loc[cells["track_memberships"].map(lambda tracks: track in tracks)].sort_values("rung")


def render_scaling(
    results: pd.DataFrame,
    confirmation: pd.DataFrame,
    cells: pd.DataFrame,
    output: Path,
) -> None:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Fitted two-phase gain", "Fitted optimum coordinates", "Candidate-location instability"),
        horizontal_spacing=0.08,
    )
    confirmed_cell = str(confirmation.iloc[0]["cell_id"])
    for track in TRACK_ORDER:
        selected_cells = _track_cells(cells, track)
        frame = selected_cells[["cell_id", "rung", "total_parameter_tpp"]].merge(
            results, on=["cell_id", "rung", "total_parameter_tpp"], validate="one_to_one"
        )
        color = TRACK_COLORS[track]
        hover = [
            "<br>".join(
                [
                    f"{row['cell_id']}",
                    f"N={float(row['total_parameters']) / 1e6:.1f}M",
                    f"D={float(row['materialized_tokens']) / 1e9:.3f}B",
                    f"TPP={float(row['total_parameter_tpp']):.3f}",
                    f"smooth gain={float(row['fitted_gain_tied_minus_untied_bpb']):.6f}",
                    f"bootstrap P(gain>0)={float(row['bootstrap_positive_gain_probability']):.3f}",
                ]
            )
            for row in frame.to_dict(orient="records")
        ]
        fig.add_trace(
            go.Scatter(
                x=frame["total_parameter_tpp"],
                y=frame["fitted_gain_tied_minus_untied_bpb"],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": frame["bootstrap_gain_p95"] - frame["fitted_gain_tied_minus_untied_bpb"],
                    "arrayminus": frame["fitted_gain_tied_minus_untied_bpb"] - frame["bootstrap_gain_p05"],
                    "color": color,
                    "thickness": 1.2,
                },
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={
                    "size": np.where(frame["confirmation_eligible"], 13, 9),
                    "color": color,
                    "symbol": np.where(frame["confirmation_eligible"], "diamond", "circle"),
                    "line": {"width": 1.5, "color": PANEL},
                },
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                name=TRACK_LABELS[track],
                legendgroup=track,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=frame["total_parameter_tpp"],
                y=frame["discovery_gain_tied_minus_untied_bpb"],
                mode="markers",
                marker={"size": 8, "symbol": "circle-open", "color": color, "opacity": 0.6},
                text=frame["cell_id"],
                hovertemplate="%{text}<br>raw min gain=%{y:.6f}<extra>selection-biased</extra>",
                name=f"{TRACK_LABELS[track]} raw minima",
                legendgroup=track,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        for field, label, symbol in (
            ("fitted_untied_p0", "phase 0", "triangle-down"),
            ("fitted_untied_p1", "phase 1", "triangle-up"),
            ("fitted_tied_weight", "tied", "square-open"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=frame["total_parameter_tpp"],
                    y=frame[field],
                    mode="lines+markers",
                    line={"color": color, "width": 1.5, "dash": "solid" if label == "tied" else "dot"},
                    marker={"size": 8, "symbol": symbol, "color": color},
                    text=frame["cell_id"],
                    hovertemplate=f"%{{text}}<br>{label}=%{{y:.4f}}<extra>{TRACK_LABELS[track]}</extra>",
                    name=f"{TRACK_LABELS[track]} · {label}",
                    legendgroup=track,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
        fig.add_trace(
            go.Scatter(
                x=frame["total_parameter_tpp"],
                y=frame["bootstrap_candidate_l2_p90"],
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"size": 9, "color": color},
                text=frame["cell_id"],
                hovertemplate="%{text}<br>bootstrap candidate L2 p90=%{y:.4f}<extra></extra>",
                name=TRACK_LABELS[track],
                legendgroup=track,
                showlegend=False,
            ),
            row=1,
            col=3,
        )

    confirmed_result = results.loc[results["cell_id"].eq(confirmed_cell)].iloc[0]
    confirmed = confirmation.iloc[0]
    fig.add_trace(
        go.Scatter(
            x=[confirmed_result["total_parameter_tpp"]],
            y=[confirmed["mean_gain_bpb"]],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [confirmed["ci95_high"] - confirmed["mean_gain_bpb"]],
                "arrayminus": [confirmed["mean_gain_bpb"] - confirmed["ci95_low"]],
                "color": "#111827",
                "thickness": 2,
            },
            mode="markers",
            marker={"size": 15, "symbol": "star", "color": "#111827", "line": {"width": 1.5, "color": PANEL}},
            text=[f"fresh confirmation · {confirmed_cell} · 8/8 paired wins"],
            hovertemplate="%{text}<br>gain=%{y:.6f}<extra></extra>",
            name="Fresh-seed confirmed gain",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=GAIN_GATE, line={"color": ORANGE, "dash": "dash", "width": 1.5}, row=1, col=1)
    fig.add_annotation(
        x=0.99,
        y=GAIN_GATE,
        xref="x domain",
        yref="y",
        text="frozen 0.005-BPB gain gate",
        showarrow=False,
        xanchor="right",
        yshift=10,
        font={"size": 11, "color": ORANGE},
    )
    fig.update_xaxes(
        type="log",
        title="Total-parameter tokens per parameter",
        tickmode="array",
        tickvals=[1, 2, 5, 10, 20, 40],
        ticktext=["1", "2", "5", "10", "20", "40"],
        gridcolor=GRID,
    )
    fig.update_yaxes(title="Tied minus untied BPB", zeroline=True, zerolinecolor=INK, gridcolor=GRID, row=1, col=1)
    fig.update_yaxes(title="StarCoder mixture weight", range=[0, 1], gridcolor=GRID, row=1, col=2)
    fig.update_yaxes(title="90th percentile L2 displacement", type="log", gridcolor=GRID, row=1, col=3)
    fig.update_layout(
        title={
            "text": (
                "StarCoder WSD80 matched-N,D phase advantage after dense sampling<br>"
                "<sup>Only 1/10 cells clears the frozen smooth-gain gate; exact optimum location remains unstable</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        paper_bgcolor=PAPER,
        plot_bgcolor=PANEL,
        font={"family": "Avenir Next, Avenir, sans-serif", "color": INK, "size": 13},
        height=730,
        margin={"l": 75, "r": 35, "t": 115, "b": 120},
        legend={"orientation": "h", "x": 0.0, "y": -0.17, "bgcolor": "rgba(255,253,248,.8)"},
        annotations=[
            *fig.layout.annotations,
            {
                "text": (
                    "Solid points: preregistered smooth-surface gain. Open points: descriptive raw min-vs-min gain. "
                    "Error bars are the frozen leverage-corrected residual bootstrap with ridge held fixed.<br>"
                    "The confirmed star is the prior 8-seed comparison at (0.02, 0.82) versus tied (0.70, 0.70). "
                    "Track curves are descriptive;<br>ten cells do not support precise scaling-law standard errors."
                ),
                "x": 0.5,
                "y": -0.27,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 12, "color": MUTED},
                "align": "left",
            },
        ],
    )
    pio.write_html(fig, output, include_plotlyjs=True, full_html=True, config=EXPORT_CONFIG)


def main() -> None:
    args = parse_args()
    observations, results, confirmation, cells = load_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_surface_explorer(observations, results, cells, args.output_dir / "dense_surface_explorer.html")
    render_scaling(results, confirmation, cells, args.output_dir / "phase_gain_scaling.html")


if __name__ == "__main__":
    main()
