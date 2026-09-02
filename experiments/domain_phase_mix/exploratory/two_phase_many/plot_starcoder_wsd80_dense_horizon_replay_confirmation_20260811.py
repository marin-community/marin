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

"""Plot fresh confirmation of dense WSD80 horizon-by-replay selected policies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plot_starcoder_wsd80_dense_horizon_replay_gain_20260811 import (
    DEFAULT_COVERAGE_OBSERVATIONS,
    DEFAULT_DESIGN,
    DEFAULT_SELECTED_POLICIES,
    EXPECTED_BLOCKS,
    EXPECTED_CELLS,
    GRID_COLOR,
    PAPER_BACKGROUND,
    PAPER_TEXT,
    PLOT_BACKGROUND,
    SUPPORT_COLORS,
    SUPPORT_LABELS,
    SUPPORT_MARKER_LABELS,
    SUPPORT_ORDER,
    ZERO_COLOR,
    load_summary,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_CONFIRMATION_SUMMARY = (
    REFERENCE_OUTPUTS / "starcoder_wsd80_dense_support_empirical_optimum_confirmation_results_20260811/block_summary.csv"
)
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_dense_horizon_replay_confirmation_scaling_20260811"

EXPECTED_PAIRS_PER_BLOCK = 5
DISCOVERY_TRACE_COLOR = "#748492"
HOLM_COLOR = "#F0B429"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-policies", type=Path, default=DEFAULT_SELECTED_POLICIES)
    parser.add_argument("--coverage-observations", type=Path, default=DEFAULT_COVERAGE_OBSERVATIONS)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--confirmation-summary", type=Path, default=DEFAULT_CONFIRMATION_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_confirmation_summary(
    selected_path: Path,
    coverage_path: Path,
    design_path: Path,
    confirmation_path: Path,
) -> pd.DataFrame:
    """Join the discovery minima to their fresh paired confirmation summaries."""
    discovery = load_summary(selected_path, coverage_path, design_path)
    confirmation = pd.read_csv(confirmation_path)
    required = {
        "cell_id",
        "support_id",
        "pair_count",
        "untied_win_count",
        "mean_gain_bpb",
        "ci95_low",
        "ci95_high",
        "paired_t_holm_p",
        "holm_positive",
        "fresh_tied_mean_bpb",
        "fresh_untied_mean_bpb",
        "tied_coordinate_id",
        "untied_coordinate_id",
    }
    missing = required - set(confirmation)
    if missing:
        raise ValueError(f"Confirmation summary is missing fields: {sorted(missing)}")
    if len(confirmation) != EXPECTED_BLOCKS or confirmation[["cell_id", "support_id"]].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_BLOCKS} unique confirmation blocks")
    if not confirmation["pair_count"].eq(EXPECTED_PAIRS_PER_BLOCK).all():
        raise ValueError("Every confirmation block must contain five paired seeds")
    confirmation["holm_positive"] = confirmation["holm_positive"].astype(str).str.lower().eq("true")

    summary = discovery.merge(
        confirmation[
            [
                "cell_id",
                "support_id",
                "pair_count",
                "untied_win_count",
                "mean_gain_bpb",
                "ci95_low",
                "ci95_high",
                "paired_t_one_sided_p",
                "paired_t_holm_p",
                "holm_positive",
                "fresh_tied_mean_bpb",
                "fresh_untied_mean_bpb",
                "tied_coordinate_id",
                "untied_coordinate_id",
            ]
        ],
        on=["cell_id", "support_id"],
        suffixes=("", "_confirmed"),
        validate="one_to_one",
    )
    if len(summary) != EXPECTED_BLOCKS:
        raise ValueError(f"Expected {EXPECTED_BLOCKS} joined confirmation rows")
    for policy_class in ("tied", "untied"):
        expected = summary[f"{policy_class}_coordinate_id"].astype(str)
        observed = summary[f"{policy_class}_coordinate_id_confirmed"].astype(str)
        if not expected.eq(observed).all():
            raise ValueError(f"Fresh confirmation selected different {policy_class} coordinates")
    if not np.allclose(
        summary["raw_two_phase_gain_bpb"],
        summary["tied_bpb"] - summary["untied_bpb"],
        atol=1e-12,
    ):
        raise ValueError("Discovery gain no longer matches its selected policies")
    return summary.sort_values(["support_order", "rung"]).reset_index(drop=True)


def _fresh_custom_data(group: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            group["cell_id"],
            group["support_id"].map(SUPPORT_LABELS),
            group["raw_two_phase_gain_bpb"],
            group["fresh_tied_mean_bpb"],
            group["fresh_untied_mean_bpb"],
            group["untied_win_count"],
            group["paired_t_one_sided_p"],
            group["paired_t_holm_p"],
            group["holm_positive"].map({True: "yes", False: "no"}),
            group["tied_coordinate_id"],
            group["untied_coordinate_id"],
            group["untied_p0"],
            group["untied_p1"],
        ]
    )


def build_figure(summary: pd.DataFrame) -> go.Figure:
    """Overlay fresh selected-policy inference on the raw discovery gain tracks."""
    figure = go.Figure()
    for support_id in SUPPORT_ORDER:
        group = summary.loc[summary["support_id"].eq(support_id)].sort_values("rung")
        if len(group) != EXPECTED_CELLS:
            raise ValueError(f"{support_id}: expected {EXPECTED_CELLS} token horizons")
        figure.add_trace(
            go.Scatter(
                x=group["materialized_tokens_b"],
                y=group["raw_two_phase_gain_bpb"],
                mode="lines+markers",
                line={"color": SUPPORT_COLORS[support_id], "width": 1.4, "dash": "dot"},
                marker={
                    "color": PLOT_BACKGROUND,
                    "size": 11,
                    "symbol": "circle",
                    "line": {"color": SUPPORT_COLORS[support_id], "width": 1.8},
                },
                opacity=0.47,
                name=f"{SUPPORT_LABELS[support_id]} discovery",
                legendgroup=support_id,
                showlegend=False,
                hovertemplate=(
                    f"<b>{SUPPORT_LABELS[support_id]}</b><br>"
                    "Materialized tokens: %{x:.3f}B<br>"
                    "Discovery raw sampled-grid gain: %{y:+.6f} BPB<br>"
                    "One selecting seed; hollow dotted trace<extra></extra>"
                ),
            )
        )
        line_widths = np.where(group["holm_positive"], 5.0, 2.3)
        line_colors = np.where(group["holm_positive"], HOLM_COLOR, PAPER_TEXT)
        figure.add_trace(
            go.Scatter(
                x=group["materialized_tokens_b"],
                y=group["mean_gain_bpb"],
                mode="lines+markers+text",
                line={"color": SUPPORT_COLORS[support_id], "width": 3.0},
                marker={
                    "color": SUPPORT_COLORS[support_id],
                    "size": 35,
                    "symbol": "circle",
                    "line": {"color": line_colors, "width": line_widths},
                },
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": group["ci95_high"] - group["mean_gain_bpb"],
                    "arrayminus": group["mean_gain_bpb"] - group["ci95_low"],
                    "color": SUPPORT_COLORS[support_id],
                    "thickness": 2.0,
                    "width": 5,
                },
                text=[SUPPORT_MARKER_LABELS[support_id]] * len(group),
                textposition="middle center",
                textfont={
                    "color": PLOT_BACKGROUND,
                    "family": "Avenir Next Condensed, Arial Narrow, sans-serif",
                    "size": 10,
                },
                customdata=_fresh_custom_data(group),
                name=SUPPORT_LABELS[support_id],
                legendgroup=support_id,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[1]}<br>"
                    "Materialized tokens: %{x:.3f}B<br><br>"
                    "<b>Fresh confirmation of discovery-selected policies</b><br>"
                    "Paired selected-policy gain: %{y:+.6f} BPB<br>"
                    "95% paired-t interval shown by whisker<br>"
                    "Tied mean: %{customdata[3]:.6f} BPB<br>"
                    "Untied mean: %{customdata[4]:.6f} BPB<br>"
                    "Untied wins: %{customdata[5]:.0f}/5<br>"
                    "One-sided p: %{customdata[6]:.4g}<br>"
                    "Holm p: %{customdata[7]:.4g}; significant: %{customdata[8]}<br><br>"
                    "Discovery raw sampled-grid gain: %{customdata[2]:+.6f} BPB<br>"
                    "Selected tied/untied: %{customdata[9]} / %{customdata[10]}<br>"
                    "Untied weights: p0=%{customdata[11]:.4f}, p1=%{customdata[12]:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines+markers",
            line={"color": DISCOVERY_TRACE_COLOR, "width": 1.5, "dash": "dot"},
            marker={"color": PLOT_BACKGROUND, "line": {"color": DISCOVERY_TRACE_COLOR, "width": 2}},
            name="Discovery raw sampled-grid gain",
            legendgroup="evidence",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines+markers",
            line={"color": PAPER_TEXT, "width": 3},
            marker={"color": PAPER_TEXT, "line": {"color": PAPER_TEXT, "width": 2}},
            name="Fresh paired mean ± ordinary 95% CI",
            legendgroup="evidence",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"color": PAPER_TEXT, "size": 14, "line": {"color": HOLM_COLOR, "width": 5}},
            name="Holm-positive over 28 tests",
            legendgroup="evidence",
        )
    )

    y_min = min(float(summary["ci95_low"].min()), float(summary["raw_two_phase_gain_bpb"].min()))
    y_max = max(float(summary["ci95_high"].max()), float(summary["raw_two_phase_gain_bpb"].max()))
    padding = 0.12 * (y_max - y_min)
    figure.add_hrect(y0=0.0, y1=y_max + padding, fillcolor="#DFF1E5", opacity=0.24, line_width=0)
    figure.add_hrect(y0=y_min - padding, y1=0.0, fillcolor="#F5DED8", opacity=0.20, line_width=0)
    figure.add_hline(y=0.0, line={"color": ZERO_COLOR, "width": 2.0})
    figure.update_layout(
        title={
            "text": (
                "<b>Fresh confirmation of discovery-selected StarCoder WSD80 policies</b><br>"
                "<sup>Fixed N=210M · five paired fresh seeds per block · 28-test Holm family</sup>"
            ),
            "x": 0.045,
            "xanchor": "left",
            "font": {"size": 28, "color": PAPER_TEXT, "family": "Georgia, Times New Roman, serif"},
        },
        width=1500,
        height=1250,
        paper_bgcolor=PAPER_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font={"family": "Avenir Next, Source Sans Pro, sans-serif", "size": 15, "color": PAPER_TEXT},
        margin={"l": 130, "r": 420, "t": 145, "b": 190},
        hoverlabel={"bgcolor": PLOT_BACKGROUND, "font": {"size": 13, "color": PAPER_TEXT}},
        legend={
            "title": {"text": "<b>Simulated-epoching repetition multiplier</b>"},
            "x": 1.025,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "bgcolor": "rgba(255,253,248,0.97)",
            "bordercolor": GRID_COLOR,
            "borderwidth": 1.5,
            "font": {"size": 13},
            "itemsizing": "constant",
        },
        annotations=[
            {
                "text": (
                    "Foreground: fresh mean BPB of the two coordinates selected on the discovery surface; "
                    "whiskers are ordinary paired-t intervals.<br>"
                    "Background dotted traces: one-seed raw sampled-grid minima. Gold outlines mark the three "
                    "blocks passing Holm correction.<br>"
                    "This confirms selected discrete policies, not either continuous global policy-class optimum."
                ),
                "x": 0.5,
                "xref": "paper",
                "y": -0.13,
                "yref": "paper",
                "showarrow": False,
                "xanchor": "center",
                "align": "center",
                "font": {"size": 13, "color": PAPER_TEXT},
            }
        ],
    )
    ticks = summary.drop_duplicates("rung").sort_values("rung")["materialized_tokens_b"]
    figure.update_xaxes(
        type="log",
        title_text="Materialized training tokens D",
        tickmode="array",
        tickvals=ticks,
        ticktext=[f"{value:.2f}B" for value in ticks],
        gridcolor=GRID_COLOR,
        zeroline=False,
        showline=True,
        linecolor=PAPER_TEXT,
        linewidth=1.2,
        ticks="outside",
    )
    figure.update_yaxes(
        title_text=("Selected-policy two-phase gain (BPB)<br><sup>tied minus untied; higher favors two-phase</sup>"),
        range=[y_min - padding, y_max + padding],
        tickformat="+.3f",
        gridcolor=GRID_COLOR,
        zeroline=False,
        showline=True,
        linecolor=PAPER_TEXT,
        linewidth=1.2,
        ticks="outside",
    )
    return figure


def write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    matrix = summary.pivot(index="support_id", columns="materialized_tokens_b", values="mean_gain_bpb")
    matrix = matrix.reindex(SUPPORT_ORDER)
    matrix.index = [SUPPORT_LABELS[value] for value in matrix.index]
    matrix.columns = [f"{value:.2f}B" for value in matrix.columns]
    holm = summary.loc[summary["holm_positive"]]
    lines = [
        "# Fresh confirmation of dense horizon-by-repetition selected policies",
        "",
        "- Scope: fixed 210M-parameter model, four token horizons, and seven StarCoder support regimes.",
        "- Each block confirms the tied and eligible-untied coordinates selected on the one-seed 125-policy grid.",
        "- Fresh estimates use five matched seeds; the discovery seed is not pooled.",
        "- Whiskers are ordinary 95% paired-t intervals. Positive significance uses Holm correction over 28 blocks.",
        "- These are selected discrete-policy effects, not continuous global policy-class optimum estimates.",
        "",
        "## Fresh selected-policy gain matrix",
        "",
        matrix.to_markdown(floatfmt="+.6f"),
        "",
        "## Holm-positive blocks",
        "",
        holm[
            [
                "materialized_tokens_b",
                "support_id",
                "mean_gain_bpb",
                "ci95_low",
                "ci95_high",
                "untied_win_count",
                "paired_t_holm_p",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "At 7.41B tokens, fresh gain is strictly increasing across all six finite replay regimes. None of the four "
        "full-pool selected untied policies has a positive fresh mean.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = load_confirmation_summary(
        args.selected_policies,
        args.coverage_observations,
        args.design,
        args.confirmation_summary,
    )
    figure = build_figure(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "confirmed_selected_policy_gain_by_horizon_and_replay.csv", index=False)
    figure.write_html(
        args.output_dir / "starcoder_wsd80_dense_horizon_replay_confirmation_scaling.html",
        include_plotlyjs=True,
        full_html=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "starcoder_wsd80_dense_horizon_replay_confirmation_scaling",
                "height": 2500,
                "width": 3000,
                "scale": 4,
            },
        },
    )
    figure.write_image(
        args.output_dir / "starcoder_wsd80_dense_horizon_replay_confirmation_scaling.png",
        width=1500,
        height=1250,
        scale=2,
    )
    write_report(args.output_dir, summary)


if __name__ == "__main__":
    main()
