# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["kaleido==0.2.1", "numpy", "pandas", "plotly"]
# ///
"""Draw why a fixed-aggregate phase-order experiment can miss the two-phase optimum.

Left panel: the two measured fibers, as gain over each fiber's own tied policy. The fiber anchored at
the one-phase optimum never rises above zero at any of nine contrasts. The fiber anchored twelve
aggregate points below it rises to a clear interior peak. Shading marks the region where a contrast
has a feasible mirror image, which is the only region where the odd/even decomposition is defined;
the peak of the lower fiber sits outside it.

Right panel: the one-phase response against the two-phase envelope, both as functions of the token
aggregate. The vertical gap between them is the phase gain, and it closes exactly at the one-phase
optimum. That is the whole explanation for the misplaced optimum: the two curves have different
minimizers because the gap is not flat.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
SOURCE_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_surface_refined_20260714"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "wsd80_two_fiber_decomposition_20260728"

PHASE_0_FRACTION = 0.8
PHASE_1_FRACTION = 0.2
REFERENCE_SEED = 20260711
AGGREGATE_BIN_WIDTH = 0.02
ENVELOPE_AGGREGATE_LIMIT = 0.42
COMPARISON_AGGREGATE_LOW = 0.10
COMPARISON_AGGREGATE_HIGH = 0.30
ONE_PHASE_COLOR = "#2E5D50"
TWO_PHASE_COLOR = "#A9421F"
NULL_FIBER_COLOR = "#6E6A62"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def paired_region_bound(aggregate: float) -> float:
    """Largest phase contrast whose mirror image is still a valid pair of mixtures."""
    return aggregate / PHASE_0_FRACTION


def two_phase_envelope(surface: pd.DataFrame) -> pd.DataFrame:
    """Best measured policy in each aggregate bin, regardless of its phase contrast."""
    binned = surface.assign(bin=(surface["aggregate"] / AGGREGATE_BIN_WIDTH).round() * AGGREGATE_BIN_WIDTH)
    rows = []
    for value, block in binned.groupby("bin"):
        if value > ENVELOPE_AGGREGATE_LIMIT:
            continue
        best = block.loc[block["wsd80_bpb"].idxmin()]
        rows.append(
            {
                "aggregate": float(value),
                "bpb": float(best["wsd80_bpb"]),
                "contrast": float(best["contrast"]),
                "policies": len(block),
            }
        )
    return pd.DataFrame(rows).sort_values("aggregate")


def build_figure(fibers: pd.DataFrame, diagonal: pd.DataFrame, envelope: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.09,
        subplot_titles=(
            "Two fixed-aggregate fibers, gain over each fiber's own tied policy",
            "One-phase response vs two-phase envelope",
        ),
    )

    for aggregate, color, name in (
        (0.30, NULL_FIBER_COLOR, "fiber at aggregate 0.30 (the one-phase optimum)"),
        (0.18, TWO_PHASE_COLOR, "fiber at aggregate 0.18 (through the global optimum)"),
    ):
        block = fibers[np.isclose(fibers["aggregate"], aggregate)].sort_values("contrast")
        tied = float(block.loc[np.isclose(block["contrast"], 0.0), "wsd80_bpb"].iloc[0])
        bound = paired_region_bound(aggregate)
        figure.add_vrect(
            x0=-bound,
            x1=bound,
            fillcolor=color,
            opacity=0.07,
            line_width=0,
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=block["contrast"],
                y=tied - block["wsd80_bpb"],
                mode="lines+markers",
                line={"color": color, "width": 2.2},
                marker={"size": 5},
                name=name,
                hovertemplate="contrast %{x:+.3f}<br>gain %{y:+.4f} BPB<extra></extra>",
            ),
            row=1,
            col=1,
        )
        best = block.loc[block["wsd80_bpb"].idxmin()]
        if abs(float(best["contrast"])) > 1e-9:
            figure.add_trace(
                go.Scatter(
                    x=[best["contrast"]],
                    y=[tied - best["wsd80_bpb"]],
                    mode="markers+text",
                    marker={"size": 13, "color": color, "symbol": "star"},
                    text=[f"  p0={best['phase_0_starcoder']:.2f}, p1={best['phase_1_starcoder']:.2f}"],
                    textposition="middle right",
                    textfont={"size": 11},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
    figure.add_hline(y=0.0, line={"color": "#444", "width": 1.2, "dash": "dot"}, row=1, col=1)
    figure.add_annotation(
        x=-0.40,
        y=0.024,
        text=(
            "shaded: contrasts whose mirror image exists.<br>"
            "Only there are the ordering effect o and the<br>"
            "asymmetry cost c both defined."
        ),
        showarrow=False,
        align="left",
        font={"size": 10, "color": "#555"},
        xanchor="left",
        row=1,
        col=1,
    )
    figure.add_annotation(
        x=0.42,
        y=0.0128,
        text="the two-phase optimum sits<br>outside the paired region",
        showarrow=True,
        arrowhead=2,
        arrowsize=0.8,
        arrowwidth=1.1,
        arrowcolor="#888",
        ax=48,
        ay=42,
        align="left",
        font={"size": 10, "color": "#555"},
        xanchor="left",
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=diagonal["aggregate"],
            y=diagonal["wsd80_bpb"],
            mode="lines+markers",
            line={"color": ONE_PHASE_COLOR, "width": 2.4},
            marker={"size": 6},
            name="one-phase policies (tied mixtures)",
            hovertemplate="aggregate %{x:.3f}<br>%{y:.4f} BPB<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=envelope["aggregate"],
            y=envelope["bpb"],
            mode="lines+markers",
            line={"color": TWO_PHASE_COLOR, "width": 2.4},
            marker={"size": 6},
            name="best two-phase policy at that aggregate",
            hovertemplate="aggregate %{x:.3f}<br>%{y:.4f} BPB<br>contrast %{customdata:+.3f}<extra></extra>",
            customdata=envelope["contrast"],
        ),
        row=1,
        col=2,
    )
    best_tied = diagonal.loc[diagonal["wsd80_bpb"].idxmin()]
    best_two = envelope.loc[envelope["bpb"].idxmin()]
    for x, y, label, color, position in (
        (best_tied["aggregate"], best_tied["wsd80_bpb"], "one-phase optimum", ONE_PHASE_COLOR, "top center"),
        (best_two["aggregate"], best_two["bpb"], "two-phase optimum", TWO_PHASE_COLOR, "bottom center"),
    ):
        figure.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker={"size": 14, "color": color, "symbol": "star"},
                text=[label],
                textposition=position,
                textfont={"size": 11, "color": color},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
    figure.add_annotation(
        x=0.245,
        y=0.9760,
        text="the vertical gap is the phase gain;<br>it closes exactly at the one-phase optimum",
        showarrow=False,
        align="left",
        font={"size": 10, "color": "#555"},
        xanchor="left",
        row=1,
        col=2,
    )

    figure.update_xaxes(title_text="phase contrast  d = p1 - p0", range=[-0.42, 0.94], row=1, col=1)
    figure.update_yaxes(
        title_text="BPB gain over own tied policy",
        # The feasibility endpoints of both fibers lose 0.15 to 0.24 BPB, which flattens everything
        # else if it is left in view. The decision-relevant band is an order of magnitude smaller.
        range=[-0.055, 0.028],
        row=1,
        col=1,
    )
    figure.update_xaxes(title_text="token aggregate StarCoder share", range=[0.055, 0.335], row=1, col=2)
    figure.update_yaxes(title_text="code BPB", range=[0.928, 1.045], row=1, col=2)
    figure.update_layout(
        template="simple_white",
        height=520,
        width=1380,
        title={
            "text": (
                "The 80/20 WSD two-phase optimum is not on the fiber through the best aggregate"
                "<br><sub>60M parameters, 1B tokens, StarCoder against Nemotron, target "
                "dolma_100_programming_languages BPB. Reference seed 20260711.</sub>"
            )
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.26, "xanchor": "center", "x": 0.5},
        margin={"t": 100, "b": 120},
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fibers = pd.read_csv(args.source_dir / "wsd80_measured_fiber_observations.csv")
    fibers = fibers[fibers["data_seed"] == REFERENCE_SEED].copy()
    fibers["contrast"] = fibers["phase_1_starcoder"] - fibers["phase_0_starcoder"]
    fibers["aggregate"] = fibers["aggregate_starcoder_share_80_20"]

    surface = pd.read_csv(args.source_dir / "wsd80_observed_metrics.csv").dropna(subset=["wsd80_bpb"])
    surface["contrast"] = surface["phase_1_starcoder"] - surface["phase_0_starcoder"]
    surface["aggregate"] = (
        PHASE_0_FRACTION * surface["phase_0_starcoder"] + PHASE_1_FRACTION * surface["phase_1_starcoder"]
    )
    diagonal = surface[np.isclose(surface["contrast"], 0.0)].sort_values("aggregate")
    diagonal = diagonal[diagonal["aggregate"] <= ENVELOPE_AGGREGATE_LIMIT]
    envelope = two_phase_envelope(surface)

    print("two-phase envelope against the one-phase response:")
    print(f"{'aggregate':>10} {'one-phase':>10} {'two-phase':>10} {'gain':>10} {'contrast':>9} {'n':>4}")
    for _, row in envelope.iterrows():
        tied = float(np.interp(row["aggregate"], diagonal["aggregate"], diagonal["wsd80_bpb"]))
        print(
            f"{row['aggregate']:10.2f} {tied:10.6f} {row['bpb']:10.6f} "
            f"{tied - row['bpb']:+10.6f} {row['contrast']:+9.3f} {int(row['policies']):4d}"
        )
    # The envelope is only a tight estimate of the profile where the bin was actually sampled; bins
    # holding a handful of points understate the achievable gain, and a bin holding only tied policies
    # reports zero gain by construction. Quote the flattening over the densely sampled interior.
    window = (envelope["aggregate"] >= COMPARISON_AGGREGATE_LOW) & (envelope["aggregate"] <= COMPARISON_AGGREGATE_HIGH)
    interior = envelope[window]
    tied_window = diagonal[
        (diagonal["aggregate"] >= COMPARISON_AGGREGATE_LOW) & (diagonal["aggregate"] <= COMPARISON_AGGREGATE_HIGH)
    ]
    one_phase_range = tied_window["wsd80_bpb"].max() - tied_window["wsd80_bpb"].min()
    two_phase_range = interior["bpb"].max() - interior["bpb"].min()
    print(
        f"\nover aggregate {COMPARISON_AGGREGATE_LOW:.2f} to {COMPARISON_AGGREGATE_HIGH:.2f} the one-phase"
        f" response spans {one_phase_range:.4f} BPB and the two-phase envelope spans {two_phase_range:.4f} BPB"
    )
    print(f"phase freedom flattens the aggregate response by {one_phase_range / two_phase_range:.1f}x")
    print(
        "bins above aggregate 0.30 hold one to three policies each, so their envelope is a weak lower"
        " bound and their apparent negative gain is a sampling artifact, not a measurement"
    )

    figure = build_figure(fibers, diagonal, envelope)
    figure.write_html(args.output_dir / "wsd80_profile_versus_fiber.html", include_plotlyjs="cdn", config=PLOT_CONFIG)
    figure.write_image(args.output_dir / "wsd80_profile_versus_fiber.png", scale=3)
    envelope.to_csv(args.output_dir / "two_phase_envelope.csv", index=False)
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
