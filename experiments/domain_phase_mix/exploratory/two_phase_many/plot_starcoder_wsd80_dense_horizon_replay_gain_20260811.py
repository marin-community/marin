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

"""Plot raw global two-phase gain across the dense WSD80 horizon-by-replay panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
SOURCE_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_dense_support_empirical_optimum_confirmation_design_20260811"
DEFAULT_SELECTED_POLICIES = SOURCE_DIR / "selected_policies.csv"
DEFAULT_COVERAGE_OBSERVATIONS = SOURCE_DIR / "coverage_observations.csv"
DEFAULT_DESIGN = SCRIPT_DIR.parents[1] / "starcoder_wsd80_dense_support_surface_design_20260808.json"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_dense_horizon_replay_gain_scaling_20260811"

EXPECTED_CELLS = 4
EXPECTED_SUPPORTS = 7
EXPECTED_BLOCKS = EXPECTED_CELLS * EXPECTED_SUPPORTS
EXPECTED_POLICY_ROWS = EXPECTED_BLOCKS * 2

SUPPORT_ORDER = ("full", "m0125", "m025", "m050", "m100", "m200", "m400")
SUPPORT_LABELS = {
    "full": "Full StarCoder pool (no forced replay)",
    "m0125": "0.125x StarCoder repetition target",
    "m025": "0.25x StarCoder repetition target",
    "m050": "0.5x StarCoder repetition target",
    "m100": "1x StarCoder repetition target",
    "m200": "2x StarCoder repetition target",
    "m400": "4x StarCoder repetition target",
}
SUPPORT_MARKER_LABELS = {
    "full": "full",
    "m0125": ".125x",
    "m025": ".25x",
    "m050": ".5x",
    "m100": "1x",
    "m200": "2x",
    "m400": "4x",
}
SUPPORT_COLORS = dict(
    zip(SUPPORT_ORDER, sample_colorscale("RdYlGn_r", np.linspace(0.05, 0.95, len(SUPPORT_ORDER))), strict=True)
)
SUPPORT_COLORS["m050"] = "#9B7800"

PAPER_BACKGROUND = "#F7F3E8"
PLOT_BACKGROUND = "#FFFDF8"
PAPER_TEXT = "#17324D"
GRID_COLOR = "#D8D1C2"
ZERO_COLOR = "#17324D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-policies", type=Path, default=DEFAULT_SELECTED_POLICIES)
    parser.add_argument("--coverage-observations", type=Path, default=DEFAULT_COVERAGE_OBSERVATIONS)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _design_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = pd.DataFrame(payload["cells"])
    supports = pd.DataFrame(payload["supports"])
    if len(cells) != EXPECTED_CELLS or cells["cell_id"].nunique() != EXPECTED_CELLS:
        raise ValueError(f"Expected {EXPECTED_CELLS} token-horizon cells")
    if len(supports) != EXPECTED_BLOCKS or supports[["cell_id", "support_id"]].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_BLOCKS} unique horizon-support blocks")
    return cells, supports


def load_summary(
    selected_path: Path,
    coverage_path: Path,
    design_path: Path,
) -> pd.DataFrame:
    """Return one observed tied-versus-untied minimum comparison per block."""
    selected = pd.read_csv(selected_path)
    required = {
        "cell_id",
        "support_id",
        "policy_class",
        "coordinate_id",
        "phase_0_starcoder",
        "phase_1_starcoder",
        "discovery_bpb",
        "discovery_run_name",
    }
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"Selected-policy table is missing fields: {sorted(missing)}")
    if len(selected) != EXPECTED_POLICY_ROWS:
        raise ValueError(f"Expected {EXPECTED_POLICY_ROWS} selected-policy rows, got {len(selected)}")
    counts = selected.groupby(["cell_id", "support_id", "policy_class"]).size()
    if not counts.eq(1).all() or set(selected["policy_class"]) != {"tied", "untied"}:
        raise ValueError("Every block must contain exactly one tied and one untied selected policy")

    coverage = pd.read_csv(coverage_path)
    coverage_fields = {
        "run_name",
        "starcoder_phase_0_epochs",
        "starcoder_phase_1_epochs",
        "starcoder_total_sequences",
        "starcoder_realized_support_tokens",
    }
    missing = coverage_fields - set(coverage.columns)
    if missing:
        raise ValueError(f"Coverage table is missing fields: {sorted(missing)}")
    coverage = coverage[list(coverage_fields)].drop_duplicates("run_name")
    selected = selected.merge(
        coverage,
        left_on="discovery_run_name",
        right_on="run_name",
        how="left",
        validate="many_to_one",
    )
    if selected["run_name"].isna().any():
        raise ValueError("Coverage observations do not contain every selected discovery run")

    key = ["cell_id", "support_id"]
    tied = selected.loc[selected["policy_class"].eq("tied")].rename(
        columns={
            "coordinate_id": "tied_coordinate_id",
            "phase_0_starcoder": "tied_p0",
            "phase_1_starcoder": "tied_p1",
            "discovery_bpb": "tied_bpb",
            "starcoder_phase_0_epochs": "tied_phase_0_epochs",
            "starcoder_phase_1_epochs": "tied_phase_1_epochs",
        }
    )
    untied = selected.loc[selected["policy_class"].eq("untied")].rename(
        columns={
            "coordinate_id": "untied_coordinate_id",
            "phase_0_starcoder": "untied_p0",
            "phase_1_starcoder": "untied_p1",
            "discovery_bpb": "untied_bpb",
            "starcoder_phase_0_epochs": "untied_phase_0_epochs",
            "starcoder_phase_1_epochs": "untied_phase_1_epochs",
        }
    )
    summary = tied[
        [
            *key,
            "tied_coordinate_id",
            "tied_p0",
            "tied_p1",
            "tied_bpb",
            "tied_phase_0_epochs",
            "tied_phase_1_epochs",
        ]
    ].merge(
        untied[
            [
                *key,
                "untied_coordinate_id",
                "untied_p0",
                "untied_p1",
                "untied_bpb",
                "untied_phase_0_epochs",
                "untied_phase_1_epochs",
            ]
        ],
        on=key,
        validate="one_to_one",
    )

    cells, supports = _design_tables(design_path)
    summary = summary.merge(
        cells[
            [
                "cell_id",
                "rung",
                "materialized_tokens",
                "total_parameters",
                "non_embedding_parameters",
                "total_steps",
            ]
        ],
        on="cell_id",
        validate="many_to_one",
    )
    summary = summary.merge(
        supports[
            [
                "cell_id",
                "support_id",
                "epoch_multiplier",
                "starcoder_realized_support_tokens",
                "starcoder_support_fraction",
            ]
        ],
        on=key,
        validate="one_to_one",
    )
    if len(summary) != EXPECTED_BLOCKS:
        raise ValueError(f"Expected {EXPECTED_BLOCKS} summary rows, got {len(summary)}")

    summary["raw_two_phase_gain_bpb"] = summary["tied_bpb"] - summary["untied_bpb"]
    summary["materialized_tokens_b"] = summary["materialized_tokens"] / 1e9
    summary["total_parameter_tpp"] = summary["materialized_tokens"] / summary["total_parameters"]
    summary["non_embedding_tpp"] = summary["materialized_tokens"] / summary["non_embedding_parameters"]
    summary["starcoder_support_b"] = summary["starcoder_realized_support_tokens"] / 1e9
    summary["support_order"] = summary["support_id"].map({value: index for index, value in enumerate(SUPPORT_ORDER)})
    if summary["support_order"].isna().any():
        raise ValueError("Unknown support regime")
    return summary.sort_values(["support_order", "rung"]).reset_index(drop=True)


def _custom_data(group: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            group["cell_id"],
            group["support_id"].map(SUPPORT_LABELS),
            group["total_parameter_tpp"],
            group["non_embedding_tpp"],
            group["starcoder_support_b"],
            group["tied_p0"],
            group["tied_bpb"],
            group["untied_p0"],
            group["untied_p1"],
            group["untied_bpb"],
            group["tied_phase_0_epochs"],
            group["tied_phase_1_epochs"],
            group["untied_phase_0_epochs"],
            group["untied_phase_1_epochs"],
            group["tied_coordinate_id"],
            group["untied_coordinate_id"],
            group["raw_two_phase_gain_bpb"].map(lambda value: f"{float(value):+.6f}"),
        ]
    )


def build_figure(summary: pd.DataFrame) -> go.Figure:
    """Build the fixed-N horizon scaling figure."""
    figure = go.Figure()
    for support_id in SUPPORT_ORDER:
        group = summary.loc[summary["support_id"].eq(support_id)].sort_values("rung")
        if len(group) != EXPECTED_CELLS:
            raise ValueError(f"{support_id}: expected {EXPECTED_CELLS} horizon rows")
        figure.add_trace(
            go.Scatter(
                x=group["materialized_tokens_b"],
                y=group["raw_two_phase_gain_bpb"],
                mode="lines+markers+text",
                name=SUPPORT_LABELS[support_id],
                line={
                    "color": SUPPORT_COLORS[support_id],
                    "width": 3.2 if support_id in {"full", "m100", "m400"} else 2.0,
                },
                marker={
                    "color": PLOT_BACKGROUND,
                    "size": 34,
                    "symbol": "circle",
                    "line": {"color": SUPPORT_COLORS[support_id], "width": 4.0},
                },
                text=[SUPPORT_MARKER_LABELS[support_id]] * len(group),
                textposition="middle center",
                textfont={
                    "color": SUPPORT_COLORS[support_id],
                    "family": "Avenir Next Condensed, Arial Narrow, sans-serif",
                    "size": 10,
                },
                customdata=_custom_data(group),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[1]}<br>"
                    "Materialized tokens: %{x:.3f}B<br>"
                    "Total/non-embedding TPP: %{customdata[2]:.2f} / %{customdata[3]:.2f}<br>"
                    "Unique StarCoder support: %{customdata[4]:.4f}B tokens<br><br>"
                    "<b>Raw selected grid minima</b><br>"
                    "Tied %{customdata[14]}: p=%{customdata[5]:.4f}, %{customdata[6]:.6f} BPB<br>"
                    "Tied StarCoder epochs: %{customdata[10]:.3f} early + %{customdata[11]:.3f} late<br>"
                    "Untied %{customdata[15]}: p0=%{customdata[7]:.4f}, p1=%{customdata[8]:.4f}, "
                    "%{customdata[9]:.6f} BPB<br>"
                    "Untied StarCoder epochs: %{customdata[12]:.3f} early + %{customdata[13]:.3f} late<br>"
                    "Raw global two-phase gain: %{customdata[16]} BPB<br>"
                    "One discovery seed; selected minima on common 125-coordinate grid"
                    "<extra></extra>"
                ),
            )
        )

    y_min = float(summary["raw_two_phase_gain_bpb"].min())
    y_max = float(summary["raw_two_phase_gain_bpb"].max())
    y_padding = 0.16 * (y_max - y_min)
    figure.add_hline(y=0.0, line={"color": ZERO_COLOR, "width": 2.0})
    figure.add_annotation(
        x=0.015,
        y=0.97,
        xref="paper",
        yref="paper",
        text="<b>Global two-phase policy wins</b>",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font={"size": 14, "color": "#287A4D"},
    )
    figure.add_annotation(
        x=0.015,
        y=0.035,
        xref="paper",
        yref="paper",
        text="<b>Tied policy wins</b>",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"size": 14, "color": "#B54132"},
    )
    figure.update_layout(
        title={
            "text": (
                "<b>StarCoder WSD80 global two-phase gain across horizon and StarCoder repetition</b><br>"
                "<sup>Fixed N = 210M parameters · raw one-seed minima on the same 125 policy coordinates per block</sup>"
            ),
            "x": 0.045,
            "xanchor": "left",
            "font": {"size": 29, "color": PAPER_TEXT, "family": "Georgia, Times New Roman, serif"},
        },
        width=1450,
        height=1240,
        paper_bgcolor=PAPER_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font={"family": "Avenir Next, Source Sans Pro, sans-serif", "size": 15, "color": PAPER_TEXT},
        margin={"l": 115, "r": 390, "t": 145, "b": 180},
        hoverlabel={"bgcolor": PLOT_BACKGROUND, "font": {"size": 13, "color": PAPER_TEXT}},
        legend={
            "title": {"text": "<b>StarCoder simulated-epoching<br>repetition multiplier</b>"},
            "x": 1.025,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "bgcolor": "rgba(255,253,248,0.96)",
            "bordercolor": GRID_COLOR,
            "borderwidth": 1.5,
            "font": {"size": 14},
            "itemsizing": "constant",
            "traceorder": "normal",
        },
        annotations=[
            *figure.layout.annotations,
            {
                "text": (
                    "Global two-phase gain = lowest observed tied BPB - lowest observed untied BPB; "
                    "positive is better for two-phase.<br>"
                    "Descriptive, selection-biased points: one discovery seed per block; no continuous-surface "
                    "optimum or fresh-seed confirmation is shown."
                ),
                "x": 0.5,
                "xref": "paper",
                "y": -0.12,
                "yref": "paper",
                "showarrow": False,
                "xanchor": "center",
                "align": "center",
                "font": {"size": 13, "color": PAPER_TEXT},
            },
        ],
    )
    figure.update_xaxes(
        type="log",
        title_text="Materialized training tokens D",
        tickmode="array",
        tickvals=summary.drop_duplicates("rung").sort_values("rung")["materialized_tokens_b"],
        ticktext=[
            f"{value:.2f}B" for value in summary.drop_duplicates("rung").sort_values("rung")["materialized_tokens_b"]
        ],
        gridcolor=GRID_COLOR,
        zeroline=False,
        showline=True,
        linecolor=PAPER_TEXT,
        linewidth=1.2,
        ticks="outside",
    )
    figure.update_yaxes(
        title_text=(
            "Raw global two-phase gain (BPB)<br>"
            "<sup>sampled-grid tied minimum - untied minimum; higher is better</sup>"
        ),
        range=[y_min - y_padding, y_max + y_padding],
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
    matrix = summary.pivot(index="support_id", columns="materialized_tokens_b", values="raw_two_phase_gain_bpb")
    matrix = matrix.reindex(SUPPORT_ORDER)
    matrix.index = [SUPPORT_LABELS[value] for value in matrix.index]
    matrix.columns = [f"{value:.2f}B" for value in matrix.columns]
    lines = [
        "# StarCoder WSD80 dense horizon-by-repetition global two-phase gain scaling",
        "",
        "- Scope: fixed 210M-parameter model; four token horizons and seven StarCoder support regimes.",
        "- Every block uses the same 125-coordinate policy grid.",
        (
            "- Global two-phase gain is the lowest observed tied BPB minus the lowest eligible untied BPB "
            "within the sampled grid; positive favors two-phase."
        ),
        (
            "- These are one-seed, selection-biased grid minima. They are not continuous-surface estimates or "
            "confirmations."
        ),
        (
            "- The finite-support multiplier sets the StarCoder repetition target relative to historical simulated "
            "epoching; it is not a fixed epoch count across policy coordinates. Nemotron always uses its full "
            "physical caches."
        ),
        "",
        "## Raw global two-phase gain matrix",
        "",
        matrix.to_markdown(floatfmt="+.6f"),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = load_summary(args.selected_policies, args.coverage_observations, args.design)
    figure = build_figure(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "raw_policy_class_gain_by_horizon_and_replay.csv", index=False)
    figure.write_html(
        args.output_dir / "starcoder_wsd80_dense_horizon_replay_gain_scaling.html",
        include_plotlyjs=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "starcoder_wsd80_dense_horizon_replay_gain_scaling",
                "height": 2480,
                "width": 2900,
                "scale": 4,
            },
        },
    )
    figure.write_image(
        args.output_dir / "starcoder_wsd80_dense_horizon_replay_gain_scaling.png",
        width=1450,
        height=1240,
        scale=2,
    )
    write_report(args.output_dir, summary)


if __name__ == "__main__":
    main()
