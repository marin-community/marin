# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["kaleido==0.2.1", "matplotlib", "numpy", "pandas", "plotly"]
# ///
"""Render a Plotly prototype of the two-phase StarCoder figure.

This is intentionally separate from `starcoder_two_phase_nonmonotonicity.py` so
we can compare Plotly against the current Matplotlib artifact before replacing
anything in the manuscript.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import get_colorscale

from experiments.domain_phase_mix.exploratory.paper_plots.paper_plot_style import (
    GRP_COLOR,
    PAPER_AXIS,
    PAPER_BACKGROUND,
    PAPER_GRID,
    PAPER_TEXT,
    UNIFORM_COLOR,
)

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR / "img"
SOURCE_CSV = SCRIPT_DIR / "data" / "two_phase_starcoder_combined_143_from_wandb.csv"

TARGET = "eval/paloma/dolma_100_programing_languages/bpb"
LANDSCAPE_OUTPUT_STEM = IMG_DIR / "starcoder_two_phase_landscape_plotly"
SLICE_OUTPUT_STEM = IMG_DIR / "starcoder_two_phase_slice_plotly"

LANDSCAPE_WIDTH = 1100
LANDSCAPE_HEIGHT = 860
SLICE_WIDTH = 1100
SLICE_HEIGHT = 700
RDYLGN_R = get_colorscale("RdYlGn_r")
MONO_FONT = "IBM Plex Mono, Menlo, DejaVu Sans Mono, monospace"
SERIF_FONT = "Times New Roman, Times, serif"


def _completed_frame() -> pd.DataFrame:
    """Load completed StarCoder two-phase runs with the target metric present."""
    frame = pd.read_csv(SOURCE_CSV)
    frame = frame[frame["status"].eq("completed") & frame[TARGET].notna()].copy()
    if frame.empty:
        raise ValueError(f"No completed rows with {TARGET} in {SOURCE_CSV}")
    return frame


def _phase_1_starcoder_epoch_multiplier(frame: pd.DataFrame) -> float:
    """Return StarCoder phase-1 epochs per unit mixture weight."""
    nonzero = frame["phase_1_starcoder"].to_numpy(dtype=float) > 0
    ratios = frame.loc[nonzero, "phase_1_starcoder_epochs"].to_numpy(dtype=float) / frame.loc[
        nonzero, "phase_1_starcoder"
    ].to_numpy(dtype=float)
    if ratios.size == 0:
        raise ValueError("Cannot infer StarCoder epoch multiplier from zero-only phase-1 weights")
    return float(np.median(ratios))


def _slice_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the Nemotron-only first-phase slice used in the background figure."""
    slice_rows = frame[frame["phase_0_nemotron_full"].round(4).eq(1.0)].copy()
    if slice_rows.empty:
        raise ValueError("No phase_0_nemotron_full == 1.0 rows found")
    return slice_rows.sort_values("phase_1_starcoder")


def _hover_text(frame: pd.DataFrame) -> list[str]:
    """Return concise 3D hover labels for StarCoder runs."""
    return [
        "<br>".join(
            [
                f"p0 StarCoder: {row['phase_0_starcoder']:.3f}",
                f"p1 StarCoder: {row['phase_1_starcoder']:.3f}",
                f"Code BPB: {row[TARGET]:.4f}",
                f"Status: {row['status']}",
            ]
        )
        for _, row in frame.iterrows()
    ]


def _triangulation_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Plotly mesh triangle indices from the irregular p0/p1 sample."""
    triangulation = mtri.Triangulation(
        frame["phase_0_starcoder"].to_numpy(dtype=float),
        frame["phase_1_starcoder"].to_numpy(dtype=float),
    )
    triangles = triangulation.triangles
    return triangles[:, 0], triangles[:, 1], triangles[:, 2]


def _write_plotly_outputs(fig: go.Figure, output_stem: Path) -> None:
    """Write interactive HTML plus static PNG/PDF for a Plotly figure."""
    fig.write_html(output_stem.with_suffix(".html"), include_plotlyjs="cdn", include_mathjax="cdn")
    fig.write_image(output_stem.with_suffix(".png"), scale=2)
    fig.write_image(output_stem.with_suffix(".pdf"))


def render_landscape(frame: pd.DataFrame, slice_rows: pd.DataFrame) -> go.Figure:
    """Render the interactive 3D loss landscape."""
    x = frame["phase_0_starcoder"].to_numpy(dtype=float)
    y = frame["phase_1_starcoder"].to_numpy(dtype=float)
    z = frame[TARGET].to_numpy(dtype=float)
    mesh_i, mesh_j, mesh_k = _triangulation_indices(frame)
    global_min = frame.loc[frame[TARGET].idxmin()]
    slice_min = slice_rows.loc[slice_rows[TARGET].idxmin()]

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=mesh_i,
            j=mesh_j,
            k=mesh_k,
            intensity=z,
            colorscale=RDYLGN_R,
            cmin=float(z.min()),
            cmax=float(np.percentile(z, 96)),
            opacity=0.36,
            showscale=False,
            hoverinfo="skip",
            name="interpolated surface",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker={
                "size": 4.2,
                "color": z,
                "colorscale": RDYLGN_R,
                "cmin": float(z.min()),
                "cmax": float(np.percentile(z, 96)),
                "line": {"color": "white", "width": 0.8},
                "showscale": False,
            },
            text=_hover_text(frame),
            hoverinfo="text",
            name="observed runs",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=slice_rows["phase_0_starcoder"],
            y=slice_rows["phase_1_starcoder"],
            z=slice_rows[TARGET],
            mode="lines",
            line={"color": UNIFORM_COLOR, "width": 5},
            hoverinfo="skip",
            name="p<sub>0</sub>=0 slice",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[global_min["phase_0_starcoder"]],
            y=[global_min["phase_1_starcoder"]],
            z=[global_min[TARGET]],
            mode="markers",
            marker={
                "symbol": "diamond",
                "size": 6,
                "color": UNIFORM_COLOR,
                "line": {"color": "white", "width": 1.4},
            },
            text=[
                "<br>".join(
                    [
                        "global min",
                        f"p0: {global_min['phase_0_starcoder']:.3f}",
                        f"p1: {global_min['phase_1_starcoder']:.3f}",
                        f"Code BPB: {global_min[TARGET]:.4f}",
                    ]
                )
            ],
            hoverinfo="text",
            name="global min: p<sub>0</sub>=0.23, p<sub>1</sub>=0.25",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[slice_min["phase_0_starcoder"]],
            y=[slice_min["phase_1_starcoder"]],
            z=[slice_min[TARGET]],
            mode="markers",
            marker={
                "symbol": "x",
                "size": 7,
                "color": UNIFORM_COLOR,
                "line": {"color": "white", "width": 2.0},
            },
            text=[
                "<br>".join(
                    [
                        "p0=0 slice min",
                        f"p1: {slice_min['phase_1_starcoder']:.3f}",
                        f"Code BPB: {slice_min[TARGET]:.4f}",
                    ]
                )
            ],
            hoverinfo="text",
            name="p<sub>0</sub>=0 slice min: p<sub>1</sub>=0.28",
        )
    )
    fig.update_layout(
        template="plotly_white",
        width=LANDSCAPE_WIDTH,
        height=LANDSCAPE_HEIGHT,
        paper_bgcolor=PAPER_BACKGROUND,
        font={"family": SERIF_FONT, "size": 17, "color": PAPER_TEXT},
        title={
            "text": (
                "Two-phase loss landscape"
                f"<br><span style='font-family:{MONO_FONT};font-size:12px'>"
                "loss: paloma/dolma_100_programing_languages/bpb</span>"
            ),
            "x": 0.53,
            "xanchor": "center",
            "y": 0.98,
            "yanchor": "top",
            "font": {"family": SERIF_FONT, "size": 24, "color": PAPER_TEXT},
        },
        legend={
            "x": 0.015,
            "y": 0.84,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.88)",
            "bordercolor": PAPER_AXIS,
            "borderwidth": 1,
            "font": {"size": 14, "family": SERIF_FONT, "color": PAPER_TEXT},
        },
        scene={
            "domain": {"x": [0.0, 1.0], "y": [0.0, 0.82]},
            "xaxis": {
                "title": "Phase 0 StarCoder (p<sub>0</sub>)",
                "range": [0, 1],
                "gridcolor": "white",
                "backgroundcolor": "#E8EEF6",
                "showbackground": True,
                "zeroline": False,
            },
            "yaxis": {
                "title": "Phase 1 StarCoder (p<sub>1</sub>)",
                "range": [0, 1],
                "gridcolor": "white",
                "backgroundcolor": "#E8EEF6",
                "showbackground": True,
                "zeroline": False,
            },
            "zaxis": {
                "title": "Code BPB",
                "range": [float(z.min()) - 0.04, float(z.max()) + 0.08],
                "gridcolor": "white",
                "backgroundcolor": "#E8EEF6",
                "showbackground": True,
                "zeroline": False,
            },
            "camera": {
                "eye": {"x": -1.55, "y": -1.55, "z": 1.25},
                "center": {"x": 0.0, "y": 0.0, "z": -0.08},
                "projection": {"type": "orthographic"},
            },
            "aspectmode": "cube",
            "aspectratio": {"x": 1.0, "y": 1.0, "z": 1.0},
        },
        margin={"l": 0, "r": 0, "t": 104, "b": 0},
    )
    return fig


def render_slice(slice_rows: pd.DataFrame, starcoder_epoch_multiplier: float) -> go.Figure:
    """Render the interactive Nemotron-first slice."""
    x = slice_rows["phase_1_starcoder"].to_numpy(dtype=float)
    y = slice_rows[TARGET].to_numpy(dtype=float)
    slice_min_idx = int(np.argmin(y))
    slice_min_x = float(x[slice_min_idx])
    slice_min_y = float(y[slice_min_idx])
    slice_min_epochs = slice_min_x * starcoder_epoch_multiplier

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line={"color": PAPER_AXIS, "dash": "dash", "width": 3},
            marker={"color": GRP_COLOR, "size": 8, "line": {"color": "white", "width": 1}},
            name="p<sub>0</sub>=0 slice",
            hovertemplate="p1=%{x:.3f}<br>Code BPB=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[slice_min_x],
            y=[slice_min_y],
            mode="markers",
            marker={"symbol": "star", "color": UNIFORM_COLOR, "size": 14, "line": {"color": "white", "width": 1}},
            name=f"slice min: p<sub>1</sub>={slice_min_x:.2f}",
            hovertemplate=f"slice minimum<br>p1={slice_min_x:.3f}<br>epochs={slice_min_epochs:.1f}"
            f"<br>Code BPB={slice_min_y:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=slice_min_x, line={"color": UNIFORM_COLOR, "dash": "dot", "width": 2})
    fig.add_annotation(
        x=slice_min_x,
        y=slice_min_y,
        ax=110,
        ay=-145,
        text=f"slice minimum at p<sub>1</sub>={slice_min_x:.2f}<br>({slice_min_epochs:.1f} StarCoder epochs)",
        showarrow=True,
        arrowcolor=UNIFORM_COLOR,
        arrowwidth=2,
        bgcolor="rgba(255,255,255,0.86)",
        bordercolor="rgba(0,0,0,0)",
        font={"family": SERIF_FONT, "size": 17, "color": PAPER_TEXT},
    )
    fig.update_layout(
        template="plotly_white",
        width=SLICE_WIDTH,
        height=SLICE_HEIGHT,
        paper_bgcolor=PAPER_BACKGROUND,
        plot_bgcolor=PAPER_BACKGROUND,
        font={"family": SERIF_FONT, "size": 18, "color": PAPER_TEXT},
        title={
            "text": "More code eventually hurts code loss",
            "x": 0.5,
            "xanchor": "center",
            "font": {"family": SERIF_FONT, "size": 24, "color": PAPER_TEXT},
        },
        xaxis={
            "title": "Phase 1 StarCoder weight (p<sub>1</sub>)",
            "range": [-0.02, 1.02],
            "showline": True,
            "linecolor": PAPER_AXIS,
            "gridcolor": PAPER_GRID,
            "zeroline": False,
        },
        yaxis={
            "title": "Code BPB",
            "range": [float(y.min()) - 0.05, float(y.max()) + 0.06],
            "showline": True,
            "linecolor": PAPER_AXIS,
            "gridcolor": PAPER_GRID,
            "zeroline": False,
        },
        legend={
            "x": 0.02,
            "y": 0.98,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.88)",
            "bordercolor": PAPER_AXIS,
            "borderwidth": 1,
        },
        margin={"l": 80, "r": 40, "t": 90, "b": 80},
    )
    return fig


def main() -> None:
    """Render Plotly prototype artifacts."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    frame = _completed_frame()
    slice_rows = _slice_frame(frame)
    starcoder_epoch_multiplier = _phase_1_starcoder_epoch_multiplier(frame)

    landscape = render_landscape(frame, slice_rows)
    _write_plotly_outputs(landscape, LANDSCAPE_OUTPUT_STEM)
    slice_figure = render_slice(slice_rows, starcoder_epoch_multiplier)
    _write_plotly_outputs(slice_figure, SLICE_OUTPUT_STEM)

    print(f"Wrote {LANDSCAPE_OUTPUT_STEM.with_suffix('.html')}")
    print(f"Wrote {LANDSCAPE_OUTPUT_STEM.with_suffix('.png')}")
    print(f"Wrote {LANDSCAPE_OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"Wrote {SLICE_OUTPUT_STEM.with_suffix('.html')}")
    print(f"Wrote {SLICE_OUTPUT_STEM.with_suffix('.png')}")
    print(f"Wrote {SLICE_OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"Completed rows: {len(frame)}; slice rows: {len(slice_rows)}")


if __name__ == "__main__":
    main()
