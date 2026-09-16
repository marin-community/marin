# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["kaleido==0.2.1", "numpy", "pandas", "plotly"]
# ///
"""Draw the phase-order cancellation from the 3e18 ladder, on both contrast axes.

The claim the figure has to carry is that the ordering benefit and the cost of making the phases
unequal grow together and cancel. Plotting the two decomposed terms against each other would show
that, but it asks the audience to accept an algebraic step before they can read the plot. Plotting the
raw runs does not.

So both orientations are drawn against their own same-seed tied control, which puts the control on
zero by construction. The two branches then carry the whole argument geometrically: their half
separation is the ordering effect, their midpoint is the asymmetry cost, and the lower branch is the
best a two-phase policy can do at that contrast magnitude. A lower branch that stays inside the noise
band is the cancellation, with no decomposition required to see it.

Two x-axes are produced because they suit different audiences. Total variation is what the panel
varied and needs no setup. Phase information is the natural unit -- it is what the epsilon-budget work
constrained, and it makes the quadratic growth of the cost legible, since along a fixed direction the
two are related by information = 0.33 * TV^2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
PANEL_DIR = REFERENCE_OUTPUTS / "delphi_3e18_uncheatable_phase_tv_ladder_20260727"
RESULTS_DIR = REFERENCE_OUTPUTS / "delphi_3e18_uncheatable_phase_tv_ladder_results_20260727"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "phase_order_cancellation_figure_20260727"

ALPHA_0 = 0.7981376787495837
ALPHA_1 = 1.0 - ALPHA_0
RUN_SIGMA = {"uncheatable": 0.000913, "table9": 0.003772}
PANEL_TITLES = {"uncheatable": "Uncheatable", "table9": "Table-9 macro"}
# The contrast direction moves a technical-specialization group between phases. Naming the branches by
# what they do to the data reads better in a talk than naming them by the sign of the contrast.
BRANCH_LABEL = {"plus": "technical group late", "minus": "technical group early"}
BRANCH_COLOR = {"plus": "#1A6FB5", "minus": "#C1443C"}
NOISE_BAND_COLOR = "rgba(120,120,120,0.16)"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def phase_information_by_level(panel: pd.DataFrame, domains: tuple[str, ...]) -> dict[float, float]:
    """Average KL of each phase mixture from the aggregate, in nats, per contrast level."""

    def divergence(left: np.ndarray, right: np.ndarray) -> float:
        left = np.maximum(left, 1e-12)
        right = np.maximum(right, 1e-12)
        return float(np.sum(left * np.log(left / right)))

    values: dict[float, float] = {}
    for _, row in panel[panel["sign"] == "plus"].iterrows():
        phase_0 = np.array([row[f"phase_0_{name}"] for name in domains], dtype=float)
        phase_1 = np.array([row[f"phase_1_{name}"] for name in domains], dtype=float)
        aggregate = ALPHA_0 * phase_0 + ALPHA_1 * phase_1
        values[float(row["phase_tv"])] = ALPHA_0 * divergence(phase_0, aggregate) + ALPHA_1 * divergence(
            phase_1, aggregate
        )
    return values


def build_figure(levels: pd.DataFrame, x_column: str, x_title: str, x_tickformat: str) -> go.Figure:
    """One panel per objective, both orientations drawn against their own tied control."""
    targets = [target for target in ("uncheatable", "table9") if target in set(levels["target"])]
    figure = make_subplots(
        rows=1,
        cols=len(targets),
        subplot_titles=[PANEL_TITLES[target] for target in targets],
        horizontal_spacing=0.10,
    )
    for column, target in enumerate(targets, start=1):
        sigma = RUN_SIGMA[target]
        block = levels[levels["target"] == target]
        span = [0.0, float(block[x_column].max()) * 1.06]
        # Run noise around the tied control, so "cancels" is anchored to something measured rather
        # than to the eye.
        figure.add_trace(
            go.Scatter(
                x=span + span[::-1],
                y=[1, 1, -1, -1],
                mode="lines",
                fill="toself",
                fillcolor=NOISE_BAND_COLOR,
                line={"width": 0},
                hoverinfo="skip",
                showlegend=column == 1,
                name="±1 run sigma",
            ),
            row=1,
            col=column,
        )
        figure.add_hline(y=0.0, line={"color": "#444", "width": 1.2, "dash": "dot"}, row=1, col=column)

        for sign in ("minus", "plus"):
            gains = block.assign(gain=(block[sign] - block["tied"]) / sigma)
            figure.add_trace(
                go.Scatter(
                    x=gains[x_column],
                    y=gains["gain"],
                    mode="markers",
                    marker={"color": BRANCH_COLOR[sign], "size": 6, "opacity": 0.35},
                    hovertemplate=f"{BRANCH_LABEL[sign]}<br>%{{x}}<br>%{{y:+.2f}} sigma<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
            mean = gains.groupby(x_column)["gain"].mean().reset_index()
            figure.add_trace(
                go.Scatter(
                    x=mean[x_column],
                    y=mean["gain"],
                    mode="lines+markers",
                    line={"color": BRANCH_COLOR[sign], "width": 2.6},
                    marker={"color": BRANCH_COLOR[sign], "size": 9},
                    name=BRANCH_LABEL[sign],
                    showlegend=column == 1,
                    hovertemplate=f"{BRANCH_LABEL[sign]} mean<br>%{{x}}<br>%{{y:+.2f}} sigma<extra></extra>",
                ),
                row=1,
                col=column,
            )
        figure.update_xaxes(title_text=x_title, range=span, tickformat=x_tickformat, row=1, col=column)
        figure.update_yaxes(
            title_text="BPB vs same-seed tied control (run sigma)" if column == 1 else None,
            zeroline=False,
            row=1,
            col=column,
        )

    figure.update_layout(
        template="simple_white",
        height=440,
        width=980,
        title={
            "text": (
                "Two-phase policies against their tied control at 3e18<br>"
                "<sub>Half the separation between branches is the ordering effect; their midpoint is the "
                "asymmetry cost. The lower branch is the best a two-phase policy achieves.</sub>"
            )
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.32, "xanchor": "center", "x": 0.5},
        margin={"t": 96, "b": 96},
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    levels = pd.read_csv(RESULTS_DIR / "level_decomposition.csv")
    panel_files = sorted(PANEL_DIR.glob("ladder_panel-*.csv"))
    assert len(panel_files) == 1, f"expected one ladder panel, found {panel_files}"
    panel = pd.read_csv(panel_files[0])
    domains = tuple(column[len("phase_0_") :] for column in panel.columns if column.startswith("phase_0_"))
    information = phase_information_by_level(panel, domains)
    levels["phase_information"] = levels["phase_tv"].map(information)
    assert levels["phase_information"].notna().all(), "a contrast level has no phase information"

    ratios = {tv: value / tv**2 for tv, value in sorted(information.items())}
    print("phase information by contrast level (nats), and information / TV^2:")
    for tv, value in sorted(information.items()):
        print(f"  TV {tv:.2f}   {value:.6f} nats   ratio {ratios[tv]:.4f}")

    for x_column, x_title, tickformat, stem in (
        ("phase_tv", "phase contrast, total variation", ".2f", "cancellation_by_total_variation"),
        ("phase_information", "phase contrast, information (nats)", ".3f", "cancellation_by_phase_information"),
    ):
        figure = build_figure(levels, x_column, x_title, tickformat)
        figure.write_html(args.output_dir / f"{stem}.html", include_plotlyjs="cdn", config=PLOT_CONFIG)
        figure.write_image(args.output_dir / f"{stem}.png", scale=4)
        print(f"wrote {args.output_dir / stem}.html and .png")

    print("\nlower-branch mean (best two-phase at each level), in run sigma:")
    for target in sorted(set(levels["target"])):
        sigma = RUN_SIGMA[target]
        block = levels[levels["target"] == target]
        best = block.assign(gain=(block[["plus", "minus"]].min(axis=1) - block["tied"]) / sigma)
        rendered = "  ".join(
            f"TV {tv:.2f}: {value:+.2f}" for tv, value in best.groupby("phase_tv")["gain"].mean().items()
        )
        print(f"  {PANEL_TITLES[target]:<16}{rendered}")


if __name__ == "__main__":
    main()
