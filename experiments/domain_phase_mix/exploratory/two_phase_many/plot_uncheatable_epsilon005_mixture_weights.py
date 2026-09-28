# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kaleido==0.2.1",
#   "numpy",
#   "pandas",
#   "plotly",
# ]
# ///
"""Plot four-column policy diagnostics for Uncheatable epsilon=0.005 mixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "decoupled_phase_information_validation_panel_20260712"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "reference_outputs" / "decoupled_phase_information_validation_results_20260712"
DEFAULT_OBSERVED_RESULTS = DEFAULT_RESULTS_DIR / "observed_results.csv"
EXPORT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

CANDIDATES = {
    "Canonical DSP": "dphase_unch05_can_e0p005",
    "Effective-exposure DSP": "dphase_unch05_eff_e0p005",
    "Effective-exposure + geometry": "dphase_unch05_geo_e0p005",
    "Separate heads": "dphase_unch05_sep_e0p005",
}
PROPORTIONAL_COLOR = "#748797"
POLICY_COLOR = "#E36F2C"
PHASE_FRACTIONS = np.array([0.8, 0.2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--observed-results", type=Path, default=DEFAULT_OBSERVED_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def clean_domain(domain: str) -> str:
    cleaned = domain
    for prefix in ("dolma3_", "dolmino_"):
        if cleaned.startswith(prefix):
            cleaned = cleaned.removeprefix(prefix)
            break
    return cleaned.replace("cc/", "CC: ").replace("_", " ")


def load_policy(panel_dir: Path, candidate: str) -> pd.DataFrame:
    frame = pd.read_csv(panel_dir / "mixtures" / f"{candidate}.csv")
    required = {
        "domain",
        "proportional",
        "phase_0_weight",
        "phase_1_weight",
        "aggregate_weight",
        "simulated_epochs",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{candidate} is missing columns: {sorted(missing)}")
    if not np.isclose(frame["phase_0_weight"].sum(), 1.0) or not np.isclose(frame["phase_1_weight"].sum(), 1.0):
        raise ValueError(f"{candidate} phase weights do not sum to one")
    reconstructed = PHASE_FRACTIONS[0] * frame["phase_0_weight"] + PHASE_FRACTIONS[1] * frame["phase_1_weight"]
    if not np.allclose(reconstructed, frame["aggregate_weight"], atol=1e-12):
        raise ValueError(f"{candidate} aggregate weights do not match the 80/20 phase average")

    frame["proportional_simulated_epochs"] = (
        frame["simulated_epochs"] * frame["proportional"] / frame["aggregate_weight"]
    )
    frame["domain_short"] = frame["domain"].astype(str).map(clean_domain)
    return frame


def add_panel(
    figure: go.Figure,
    frame: pd.DataFrame,
    row: int,
    col: int,
    policy_column: str,
    proportional_column: str,
    axis_title: str,
) -> None:
    policy = frame[policy_column]
    proportional = frame[proportional_column]
    customdata = np.stack([frame["domain"], policy - proportional, policy, proportional], axis=-1)
    for values, label, color, opacity, legendgroup in (
        (proportional, "Proportional", PROPORTIONAL_COLOR, 0.84, "proportional"),
        (policy, "Model policy", POLICY_COLOR, 0.92, "policy"),
    ):
        figure.add_trace(
            go.Bar(
                x=values,
                y=frame["domain_short"],
                orientation="h",
                name=label,
                legendgroup=legendgroup,
                showlegend=row == 1 and col == 1,
                marker_color=color,
                opacity=opacity,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{axis_title}: %{{x:.6f}}<br>"
                    "model - proportional: %{customdata[1]:+.6f}<br>"
                    "model value: %{customdata[2]:.6f}<br>"
                    "proportional value: %{customdata[3]:.6f}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
    figure.update_xaxes(title_text=axis_title, row=row, col=col)
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=frame["domain_short"].tolist(),
        tickfont={"size": 10},
        row=row,
        col=col,
    )


def render(panel_dir: Path, observed_results: Path, output_dir: Path) -> None:
    observed = pd.read_csv(observed_results).set_index("candidate")
    policies = {model: load_policy(panel_dir, candidate) for model, candidate in CANDIDATES.items()}
    reference = next(iter(policies.values()))
    for model, frame in policies.items():
        if not np.allclose(frame["aggregate_weight"], reference["aggregate_weight"], atol=1e-12):
            raise ValueError(f"{model} does not preserve the common aggregate weights")

    aggregate_delta = reference["aggregate_weight"] - reference["proportional"]
    domain_order = (
        reference.assign(aggregate_delta=aggregate_delta)
        .sort_values(["aggregate_delta", "domain"], ascending=[True, True])["domain"]
        .tolist()
    )
    for model, frame in policies.items():
        policies[model] = frame.set_index("domain").loc[domain_order].reset_index()

    row_titles = []
    for model, candidate in CANDIDATES.items():
        result = observed.loc[candidate]
        row_titles.append(f"{model}<br>TV={result['phase_tv']:.3f}<br>BPB={result['observed_uncheatable_bpb']:.6f}")
    subplot_titles = ["Phase 0 weights", "Phase 1 weights", "Aggregate weights", "Aggregate exposure"] + [""] * 12
    figure = make_subplots(
        rows=4,
        cols=4,
        subplot_titles=subplot_titles,
        row_titles=row_titles,
        shared_yaxes="rows",
        horizontal_spacing=0.035,
        vertical_spacing=0.055,
    )
    panels = [
        ("phase_0_weight", "proportional", "mixture weight"),
        ("phase_1_weight", "proportional", "mixture weight"),
        ("aggregate_weight", "proportional", "mixture weight"),
        ("simulated_epochs", "proportional_simulated_epochs", "realized simulated epochs"),
    ]
    for row, frame in enumerate(policies.values(), start=1):
        for col, (policy_column, proportional_column, axis_title) in enumerate(panels, start=1):
            add_panel(
                figure,
                frame,
                row,
                col,
                policy_column,
                proportional_column,
                axis_title,
            )

    figure.update_layout(
        title={
            "text": "Uncheatable epsilon_phase=0.005 policies versus proportional",
            "x": 0.5,
            "xanchor": "center",
        },
        barmode="group",
        template="plotly_white",
        width=2600,
        height=3900,
        margin={"l": 250, "r": 260, "t": 180, "b": 115},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.025,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.94)",
            "bordercolor": "#d9e0ea",
            "borderwidth": 1,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    for annotation in figure.layout.annotations:
        if annotation.text in row_titles:
            annotation.update(textangle=0, x=1.01, xanchor="left", align="left", font={"size": 13})
    figure.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.045,
        showarrow=False,
        text=(
            "Each model is compared with proportional in both phases. Aggregate weights use 0.8 phase 0 + 0.2 "
            "phase 1 and are identical across model rows by construction."
        ),
        font={"size": 15, "color": "#44546a"},
    )
    output_html = output_dir / "uncheatable_epsilon005_four_column_mixtures.html"
    figure.write_html(output_html, include_plotlyjs=True, config=EXPORT_CONFIG)
    figure.write_image(output_dir / "uncheatable_epsilon005_four_column_mixtures.png", scale=2)
    print(output_html)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render(args.panel_dir, args.observed_results, args.output_dir)


if __name__ == "__main__":
    main()
