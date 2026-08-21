# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kaleido>=1.0",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
# ]
# ///
# ruff: noqa: E501
"""Build the July 14-21 PI-meeting data-mixing progress packet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = BASE_DIR / "reference_outputs"
OUTPUT_DIR = REFERENCE_DIR / "pi_meeting_weekly_progress_20260721"
SOURCE_DIR = OUTPUT_DIR / "source_data"

SAMPLE_EFFICIENCY_DIR = REFERENCE_DIR / "delphi_phase_policy_sample_efficiency_20260721"
RAW_OPTIMUM_DIR = REFERENCE_DIR / "delphi_expanded_fit_raw_optimum_model_comparison_20260721"
HELDOUT_REGISTRY_PATH = REFERENCE_DIR / "delphi_3e18_append_only_heldouts_20260714" / "heldout_current.csv"

TWO_RUN_DIFFERENCE_SD = {
    "Uncheatable": 0.001291,
    "Table-9": 0.005334,
}

PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}

NAVY = "#183447"
BLUE = "#2E6F9E"
LIGHT_BLUE = "#78A9C8"
ORANGE = "#E8753D"
GOLD = "#D8A62A"
GREEN = "#3D8B6D"
RED = "#C6493B"
GRAY = "#72818C"
LIGHT_GRAY = "#D7DEE2"
PAPER = "#FBFAF6"
CARD = "#F1EEE5"
GRID = "#D8D3C8"

WIDTH = 1600
HEIGHT = 1000

MODEL_LABELS = {
    "bucket_family_grp": "Bucket-family GRP",
    "compact_retained_state": "Compact retained state",
    "effective_exposure": "Effective-exposure DSP",
    "hierarchical_phase_bucket_replay": "Hierarchical phase replay",
    "separate_heads": "Separate heads",
}
SHORT_MODEL_LABELS = {
    "bucket_family_grp": "Bucket GRP",
    "compact_retained_state": "Compact",
    "effective_exposure": "Eff-exp DSP",
    "hierarchical_phase_bucket_replay": "HPR",
    "separate_heads": "Sep. heads",
}


def add_fact_card(figure: go.Figure, headline: str, evidence: str, scope: str) -> None:
    """Add a presentation-readable fact card below the plotting area."""
    figure.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0.0,
        x1=1.0,
        y0=-0.41,
        y1=-0.13,
        fillcolor=CARD,
        line={"color": "#B9B2A4", "width": 1},
        layer="above",
    )
    figure.add_annotation(
        x=0.018,
        y=-0.155,
        xref="paper",
        yref="paper",
        text="<b>WHAT WE LEARNED</b>",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font={"family": "Arial, sans-serif", "size": 15, "color": ORANGE},
    )
    figure.add_annotation(
        x=0.018,
        y=-0.21,
        xref="paper",
        yref="paper",
        text=f"<b>{headline}</b>",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font={"family": "Arial, sans-serif", "size": 18, "color": NAVY},
    )
    figure.add_annotation(
        x=0.018,
        y=-0.275,
        xref="paper",
        yref="paper",
        text=f"<b>Evidence</b>  {evidence}",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font={"family": "Arial, sans-serif", "size": 13, "color": NAVY},
    )
    figure.add_annotation(
        x=0.018,
        y=-0.34,
        xref="paper",
        yref="paper",
        text=f"<b>Interpretation boundary</b>  {scope}",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font={"family": "Arial, sans-serif", "size": 12, "color": GRAY},
    )


def style_figure(figure: go.Figure, title: str) -> None:
    """Apply the common presentation style."""
    figure.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 30, "color": NAVY}},
        template="plotly_white",
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        width=WIDTH,
        height=HEIGHT,
        font={"family": "Arial, sans-serif", "size": 15, "color": NAVY},
        margin={"l": 105, "r": 55, "t": 115, "b": 305},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 13},
        },
        hoverlabel={"font": {"family": "Arial, sans-serif", "size": 13}},
    )
    figure.update_xaxes(gridcolor=GRID, zerolinecolor=GRAY, linecolor=GRAY)
    figure.update_yaxes(gridcolor=GRID, zerolinecolor=GRAY, linecolor=GRAY)


def write_figure(figure: go.Figure, stem: str) -> tuple[Path, Path]:
    """Write one interactive and one high-resolution static figure."""
    html_path = OUTPUT_DIR / f"{stem}.html"
    png_path = OUTPUT_DIR / f"{stem}.png"
    figure.write_html(
        html_path,
        include_plotlyjs="cdn",
        include_mathjax="cdn",
        config=PLOT_CONFIG,
        full_html=True,
    )
    figure.write_image(png_path, width=WIDTH, height=HEIGHT, scale=2)
    return html_path, png_path


def evidence_base_frame() -> pd.DataFrame:
    """Return physical checkpoint counts from the completed weekly panels."""
    return pd.DataFrame(
        [
            ("Two-phase fit swarm", 280, "fit"),
            ("One-phase fit swarm (new)", 238, "fit"),
            ("Adversarial stress", 120, "development"),
            ("Frontier phase fiber", 200, "development"),
            ("HPR source-scale", 124, "development"),
            ("Random phase population", 296, "development"),
            ("Hybrid phase ordering", 176, "development"),
        ],
        columns=["panel", "checkpoints", "role"],
    )


def plot_evidence_base() -> tuple[go.Figure, pd.DataFrame]:
    frame = evidence_base_frame()
    figure = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.58, 0.42],
        subplot_titles=["Completed physical checkpoints since July 14", "Append-only 3e18 archive"],
        horizontal_spacing=0.12,
    )
    ordered = frame.iloc[::-1]
    colors = [BLUE if role == "fit" else ORANGE for role in ordered["role"]]
    figure.add_trace(
        go.Bar(
            x=ordered["checkpoints"],
            y=ordered["panel"],
            orientation="h",
            marker={"color": colors},
            text=ordered["checkpoints"].map(lambda value: f"{value:,}"),
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:,} checkpoints<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=["July 14", "July 21"],
            y=[356, 1518],
            mode="lines+markers+text",
            line={"color": NAVY, "width": 5},
            marker={"size": [19, 25], "color": [LIGHT_BLUE, ORANGE], "line": {"color": NAVY, "width": 2}},
            text=["356 observations", "1,518 observations<br>1,472 unique policies"],
            textposition=["top right", "top center"],
            textfont={"size": 16, "color": NAVY},
            hovertemplate="%{x}<br>%{text}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.add_annotation(
        x="July 21",
        y=900,
        xref="x2",
        yref="y2",
        text="<b>4.3x more observations</b>",
        showarrow=False,
        font={"size": 18, "color": ORANGE},
    )
    figure.update_xaxes(title_text="Physical checkpoints", range=[0, 335], row=1, col=1)
    figure.update_xaxes(title_text="Snapshot", row=1, col=2)
    figure.update_yaxes(title_text="", row=1, col=1)
    figure.update_yaxes(title_text="Complete observations", range=[0, 1725], row=1, col=2)
    style_figure(figure, "The deployment-scale archive grew 4.3x and now includes matched fit swarms")
    add_fact_card(
        figure,
        "We moved from ad hoc validation to two matched fit swarms and a large intervention archive.",
        "1,434 physical checkpoints completed across the seven new panels shown; after the hybrid panel, "
        "the append-only archive contains 1,518 observations over 1,472 unique policies, all with both targets.",
        "Archive rows are heterogeneous interventions, not IID test samples. The logical one-phase fit panel has "
        "280 rows because it reuses 42 phase-tied coordinates and trains 238 new checkpoints.",
    )
    return figure, frame


def hpr_frame() -> pd.DataFrame:
    """Return source-scale HPR results recorded in Fieldbook."""
    rows = [
        ("Uncheatable", "300M fit", "One phase", 0.990980, 0.985120),
        ("Uncheatable", "300M fit", "Two phase", 0.989819, 0.985120),
        ("Uncheatable", "3e18 fit", "One phase", 0.989192, 0.985120),
        ("Uncheatable", "3e18 fit", "Two phase", 0.987639, 0.985120),
        ("Table-9", "300M fit", "One phase", 1.066050, 1.057530),
        ("Table-9", "300M fit", "Two phase", 1.065807, 1.057530),
        ("Table-9", "3e18 fit", "One phase", 1.071131, 1.057530),
        ("Table-9", "3e18 fit", "Two phase", 1.059499, 1.057530),
    ]
    frame = pd.DataFrame(rows, columns=["target", "fit_source", "policy_class", "observed_bpb", "reference_bpb"])
    frame["excess_bpb"] = frame["observed_bpb"] - frame["reference_bpb"]
    return frame


def plot_hpr_source_scale() -> tuple[go.Figure, pd.DataFrame]:
    frame = hpr_frame()
    figure = make_subplots(rows=1, cols=2, subplot_titles=["Uncheatable", "Table-9 macro"], horizontal_spacing=0.12)
    for column, target in enumerate(["Uncheatable", "Table-9"], start=1):
        subset = frame[frame["target"] == target]
        for source, color in [("300M fit", GRAY), ("3e18 fit", BLUE)]:
            source_rows = subset[subset["fit_source"] == source]
            figure.add_trace(
                go.Bar(
                    x=source_rows["policy_class"],
                    y=source_rows["excess_bpb"],
                    name=source,
                    marker={"color": color},
                    text=source_rows["excess_bpb"].map(lambda value: f"+{value:.4f}"),
                    textposition="outside",
                    hovertemplate=(
                        f"{source}<br>%{{x}}<br>Observed: %{{customdata[0]:.6f}}"
                        "<br>Excess over observed reference: %{y:.6f}<extra></extra>"
                    ),
                    customdata=source_rows[["observed_bpb"]].to_numpy(),
                    legendgroup=source,
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        figure.add_hline(y=0, line={"color": GREEN, "width": 2}, row=1, col=column)
        figure.update_yaxes(title_text="Excess BPB over observed reference", rangemode="tozero", row=1, col=column)
        figure.update_xaxes(title_text="Policy class", row=1, col=column)
    style_figure(figure, "3e18 refitting improves HPR two-phase proposals, but does not solve selection")
    add_fact_card(
        figure,
        "Scale-matched fitting helps the two-phase HPR proposals; the effect is not uniform across policy classes.",
        "For two-phase HPR, refitting on 3e18 reduces excess over the observed reference from 0.00470 to 0.00252 "
        "on Uncheatable and from 0.00828 to 0.00197 on Table-9. The one-phase Table-9 proposal becomes worse.",
        "The 1.057530 Table-9 reference is one unreplicated low draw. Bars are best observed points from frozen "
        "candidate sweeps, so percentages describe this panel rather than an estimator's sampling uncertainty.",
    )
    return figure, frame


def random_phase_frame() -> pd.DataFrame:
    """Return random-policy effects matched to same-anchor, same-seed tied controls."""
    registry = pd.read_csv(HELDOUT_REGISTRY_PATH, low_memory=False)
    panel = registry[registry["panel_tag"].eq("delphi-3e18-frontier-random-phase-population")].copy()
    controls = panel[panel["candidate_kind"].eq("center_control")].copy()
    random = panel[panel["candidate_kind"].eq("random_isotropic")].copy()
    key = ["proposal_target", "anchor_id", "seed_block"]
    if controls.duplicated(key).any():
        raise ValueError("Random phase panel has duplicate tied controls")

    control_metrics = controls[[*key, "uncheatable_bpb", "table9_macro_bpb"]].rename(
        columns={
            "uncheatable_bpb": "control_uncheatable_bpb",
            "table9_macro_bpb": "control_table9_bpb",
        }
    )
    frame = random.merge(control_metrics, on=key, how="left", validate="many_to_one")
    frame["target"] = frame["proposal_target"].map({"uncheatable": "Uncheatable", "table9": "Table-9"})
    frame["observed_bpb"] = np.where(
        frame["proposal_target"].eq("uncheatable"), frame["uncheatable_bpb"], frame["table9_macro_bpb"]
    )
    frame["tied_control_bpb"] = np.where(
        frame["proposal_target"].eq("uncheatable"),
        frame["control_uncheatable_bpb"],
        frame["control_table9_bpb"],
    )
    frame["delta_bpb"] = frame["observed_bpb"] - frame["tied_control_bpb"]
    frame["two_run_difference_sd"] = frame["target"].map(TWO_RUN_DIFFERENCE_SD)
    frame["delta_in_noise_sd"] = frame["delta_bpb"] / frame["two_run_difference_sd"]
    frame["improved"] = frame["delta_bpb"] < 0
    frame["radius_label"] = frame["radius_fraction"].map(lambda value: f"{value:.2f}")
    columns = [
        "candidate_id",
        "target",
        "anchor_id",
        "direction_id",
        "radius_fraction",
        "radius_label",
        "seed_block",
        "observed_bpb",
        "tied_control_bpb",
        "delta_bpb",
        "two_run_difference_sd",
        "delta_in_noise_sd",
        "improved",
    ]
    result = frame[columns].sort_values(["target", "radius_fraction", "direction_id"]).reset_index(drop=True)
    if len(result) != 288 or result[["observed_bpb", "tied_control_bpb"]].isna().any().any():
        raise ValueError(f"Expected 288 complete random phase policies, found {len(result)}")
    return result


def plot_random_phase_population() -> tuple[go.Figure, pd.DataFrame]:
    frame = random_phase_frame()
    figure = make_subplots(
        rows=1, cols=2, subplot_titles=["Uncheatable anchor", "Table-9 anchor"], horizontal_spacing=0.12
    )
    for column, target in enumerate(["Uncheatable", "Table-9"], start=1):
        subset = frame[frame["target"] == target]
        figure.add_hrect(
            y0=-1,
            y1=1,
            fillcolor=LIGHT_GRAY,
            opacity=0.45,
            line_width=0,
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Box(
                x=subset["radius_label"],
                y=subset["delta_in_noise_sd"],
                boxpoints="all",
                jitter=0.38,
                pointpos=0,
                fillcolor="rgba(46,111,158,0.16)" if target == "Uncheatable" else "rgba(232,117,61,0.16)",
                line={"color": BLUE if target == "Uncheatable" else ORANGE, "width": 2},
                marker={"size": 5, "opacity": 0.58, "color": BLUE if target == "Uncheatable" else ORANGE},
                customdata=subset[["candidate_id", "delta_bpb", "observed_bpb", "tied_control_bpb"]].to_numpy(),
                hovertemplate=(
                    "%{customdata[0]}<br>Radius: %{x}<br>Random - tied: %{customdata[1]:+.6f} BPB"
                    "<br>Effect / two-run difference SD: %{y:+.3f}"
                    "<br>Random: %{customdata[2]:.6f}<br>Tied: %{customdata[3]:.6f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        summary = (
            subset.groupby("radius_label", sort=False)
            .agg(mean_effect=("delta_in_noise_sd", "mean"), fraction_better=("improved", "mean"))
            .reset_index()
        )
        figure.add_trace(
            go.Scatter(
                x=summary["radius_label"],
                y=summary["mean_effect"],
                mode="markers+text",
                marker={"symbol": "diamond", "size": 13, "color": GOLD, "line": {"color": NAVY, "width": 1.5}},
                text=summary["fraction_better"].map(lambda value: f"{value:.0%} better"),
                textposition="top center",
                textfont={"size": 12, "color": NAVY},
                hovertemplate="Radius: %{x}<br>Mean effect / two-run difference SD: %{y:+.3f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.add_hline(y=0, line={"color": NAVY, "width": 1.5}, row=1, col=column)
        figure.add_hline(
            y=-1,
            line={"color": GRAY, "width": 1, "dash": "dot"},
            row=1,
            col=column,
        )
        figure.add_hline(
            y=1,
            line={"color": GRAY, "width": 1, "dash": "dot"},
            row=1,
            col=column,
        )
        figure.update_xaxes(
            title_text="Fraction of feasible phase-contrast radius",
            categoryorder="array",
            categoryarray=["0.25", "0.50", "0.75"],
            row=1,
            col=column,
        )
        figure.update_yaxes(
            title_text="Random - tied BPB / two-run difference SD",
            tickformat="+.1f",
            range=[-3.2, 3.2],
            row=1,
            col=column,
        )
    style_figure(figure, "Under the preregistered random sampler, phase asymmetry is centered near zero")
    add_fact_card(
        figure,
        "Generic aggregate-preserving phase perturbations contain wins and losses, with little average target-matched value.",
        "Each of 288 policies is compared with its same-anchor, same-seed tied control. Across six anchor-radius cells, "
        "mean effects range from -0.21 to +0.03 two-run difference SDs and 38%-60% of policies improve.",
        "This characterizes one preregistered isotropic sampler, not the optimal phase schedule. Points share four tied "
        "controls per anchor; boxes are descriptive distributions and the gray band is a noise scale, not a confidence interval.",
    )
    return figure, frame


def hybrid_phase_frame() -> pd.DataFrame:
    """Return fixed-aggregate phase-ordering effects matched by aggregate KL."""
    registry = pd.read_csv(HELDOUT_REGISTRY_PATH, low_memory=False)
    panel = registry[registry["panel_tag"].eq("delphi-3e18-hybrid-phase-ordering-validation")].copy()
    controls = panel[panel["candidate_kind"].eq("tied_separate_heads_anchor")].copy()
    candidates = panel[panel["candidate_kind"].str.startswith("fixed_aggregate_", na=False)].copy()
    key = ["proposal_target", "aggregate_kl_coefficient"]
    if controls.duplicated(key).any():
        raise ValueError("Hybrid panel has duplicate tied controls")
    control_metrics = controls[[*key, "uncheatable_bpb", "table9_macro_bpb"]].rename(
        columns={
            "uncheatable_bpb": "control_uncheatable_bpb",
            "table9_macro_bpb": "control_table9_bpb",
        }
    )
    frame = candidates.merge(control_metrics, on=key, how="left", validate="many_to_one")
    frame["target"] = frame["proposal_target"].map({"uncheatable": "Uncheatable", "table9": "Table-9"})
    frame["phase_head"] = frame["candidate_kind"].map(
        {
            "fixed_aggregate_bucket_family_grp": "Bucket GRP",
            "fixed_aggregate_compact_retained_state": "Compact",
            "fixed_aggregate_effective_exposure": "Eff-exp DSP",
            "fixed_aggregate_hierarchical_phase_replay": "HPR",
        }
    )
    frame["observed_bpb"] = np.where(
        frame["proposal_target"].eq("uncheatable"), frame["uncheatable_bpb"], frame["table9_macro_bpb"]
    )
    frame["tied_control_bpb"] = np.where(
        frame["proposal_target"].eq("uncheatable"),
        frame["control_uncheatable_bpb"],
        frame["control_table9_bpb"],
    )
    frame["delta_bpb"] = frame["observed_bpb"] - frame["tied_control_bpb"]
    frame["delta_millibpb"] = 1000 * frame["delta_bpb"]
    frame["is_ex_post_best"] = frame["candidate_id"].eq("hyb3_unch_eff_akl0p025_eps0p0025")
    frame["is_model_selected"] = frame["candidate_id"].eq("hyb3_unch_eff_akl0p025_eps0p025")
    columns = [
        "candidate_id",
        "target",
        "phase_head",
        "aggregate_kl_coefficient",
        "phase_information_budget",
        "observed_bpb",
        "tied_control_bpb",
        "delta_bpb",
        "delta_millibpb",
        "is_ex_post_best",
        "is_model_selected",
    ]
    result = frame[columns].sort_values(["target", "phase_head", "aggregate_kl_coefficient", "phase_information_budget"])
    if len(result) != 160 or result[["phase_head", "observed_bpb", "tied_control_bpb"]].isna().any().any():
        raise ValueError(f"Expected 160 complete fixed-aggregate hybrid policies, found {len(result)}")
    return result.reset_index(drop=True)


def plot_hybrid_phase_ordering() -> tuple[go.Figure, pd.DataFrame]:
    frame = hybrid_phase_frame()
    figure = make_subplots(rows=1, cols=2, subplot_titles=["Uncheatable", "Table-9 macro"], horizontal_spacing=0.12)
    for column, target in enumerate(["Uncheatable", "Table-9"], start=1):
        subset = frame[frame["target"] == target]
        noise = 1000 * TWO_RUN_DIFFERENCE_SD[target]
        figure.add_hrect(
            y0=-noise,
            y1=noise,
            fillcolor=LIGHT_GRAY,
            opacity=0.45,
            line_width=0,
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Box(
                x=subset["phase_head"],
                y=subset["delta_millibpb"],
                boxpoints="all",
                jitter=0.35,
                pointpos=0,
                fillcolor="rgba(46,111,158,0.16)" if target == "Uncheatable" else "rgba(232,117,61,0.16)",
                line={"color": BLUE if target == "Uncheatable" else ORANGE, "width": 2},
                marker={"size": 5, "opacity": 0.55, "color": BLUE if target == "Uncheatable" else ORANGE},
                customdata=subset[
                    [
                        "candidate_id",
                        "observed_bpb",
                        "tied_control_bpb",
                        "aggregate_kl_coefficient",
                        "phase_information_budget",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "%{customdata[0]}<br>%{x}<br>Candidate - tied: %{y:+.3f} milli-BPB"
                    "<br>Candidate: %{customdata[1]:.6f}<br>Tied: %{customdata[2]:.6f}"
                    "<br>Aggregate KL: %{customdata[3]:.3f}<br>Phase budget: %{customdata[4]:.4f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        summary = (
            subset.groupby("phase_head", sort=False)
            .agg(
                mean_delta=("delta_millibpb", "mean"), fraction_better=("delta_bpb", lambda values: (values < 0).mean())
            )
            .reset_index()
        )
        figure.add_trace(
            go.Scatter(
                x=summary["phase_head"],
                y=summary["mean_delta"],
                mode="markers+text",
                marker={"symbol": "diamond", "size": 13, "color": GOLD, "line": {"color": NAVY, "width": 1.5}},
                text=summary["fraction_better"].map(lambda value: f"{value:.0%} better"),
                textposition="top center",
                textfont={"size": 12, "color": NAVY},
                hovertemplate="%{x}<br>Mean candidate - tied: %{y:+.3f} milli-BPB<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        if target == "Uncheatable":
            for flag, symbol, color, name in [
                ("is_ex_post_best", "diamond-open", GREEN, "Ex-post low draw"),
                ("is_model_selected", "star", GOLD, "Model-selected"),
            ]:
                highlighted = subset[subset[flag]]
                figure.add_trace(
                    go.Scatter(
                        x=highlighted["phase_head"],
                        y=highlighted["delta_millibpb"],
                        mode="markers",
                        marker={"symbol": symbol, "size": 21, "color": color, "line": {"color": NAVY, "width": 2}},
                        customdata=highlighted[["observed_bpb", "tied_control_bpb"]].to_numpy(),
                        hovertemplate=(
                            f"{name}<br>Candidate - tied: %{{y:+.3f}} milli-BPB"
                            "<br>Candidate: %{customdata[0]:.6f}<br>Tied: %{customdata[1]:.6f}<extra></extra>"
                        ),
                        name=name,
                        showlegend=True,
                    ),
                    row=1,
                    col=column,
                )
        figure.add_hline(y=0, line={"color": NAVY, "width": 1.5}, row=1, col=column)
        figure.update_xaxes(
            title_text="Phase-ordering head",
            categoryorder="array",
            categoryarray=["Eff-exp DSP", "Bucket GRP", "Compact", "HPR"],
            tickangle=0,
            row=1,
            col=column,
        )
        figure.update_yaxes(title_text="Candidate - aggregate-matched tied control (milli-BPB)", row=1, col=column)
    style_figure(figure, "Model-directed phase ordering shifts Uncheatable, not Table-9")
    figure.update_layout(legend={"y": 1.10, "x": 1.0})
    add_fact_card(
        figure,
        "Effective-exposure ordering is consistently favorable for Uncheatable; no phase head transfers that effect to Table-9.",
        "Relative to the same aggregate-KL tied control, effective-exposure has mean delta -2.41 milli-BPB and 19/20 "
        "Uncheatable candidates improve. Its ex-post low draw is 0.984045, while its model-selected candidate is 0.985503.",
        "The 20 points per head are correlated hyperparameter-policy sweep points sharing four tied controls, not 20 "
        "independent trials. Boxes are descriptive; gray bands show +/- one same-configuration two-run difference SD.",
    )
    return figure, frame


def load_endpoint_changes() -> pd.DataFrame:
    path = SAMPLE_EFFICIENCY_DIR / "endpoint_comparison.csv"
    frame = pd.read_csv(path)
    frame["model_label"] = frame["model"].map(MODEL_LABELS)
    frame["short_label"] = frame["model"].map(SHORT_MODEL_LABELS)
    frame["target_label"] = frame["target"].map({"uncheatable": "Uncheatable", "table9": "Table-9"})
    frame["design_label"] = frame["design"].map(
        {"two_phase_only": "Two-phase only", "tied_spine_plus_two_phase": "Tied spine + two-phase"}
    )
    return frame


def plot_fit_vs_selection() -> tuple[go.Figure, pd.DataFrame]:
    frame = load_endpoint_changes()
    figure = go.Figure()
    for target, color in [("Uncheatable", BLUE), ("Table-9", ORANGE)]:
        for design, symbol in [("Two-phase only", "circle"), ("Tied spine + two-phase", "diamond")]:
            subset = frame[(frame["target_label"] == target) & (frame["design_label"] == design)]
            labels = []
            for row in subset.itertuples():
                if abs(row.regret1_change) > 0.003 or row.rmse_change > 0:
                    labels.append(row.short_label)
                else:
                    labels.append("")
            figure.add_trace(
                go.Scatter(
                    x=subset["rmse_change"],
                    y=subset["regret1_change"],
                    mode="markers+text",
                    marker={"size": 14, "color": color, "symbol": symbol, "line": {"color": NAVY, "width": 1}},
                    text=labels,
                    textposition="top center",
                    textfont={"size": 11},
                    customdata=subset[
                        ["model_label", "rmse_start", "rmse_end", "regret1_start", "regret1_end"]
                    ].to_numpy(),
                    hovertemplate=(
                        "%{customdata[0]}<br>RMSE: %{customdata[1]:.5f} -> %{customdata[2]:.5f}"
                        "<br>Regret@1: %{customdata[3]:.5f} -> %{customdata[4]:.5f}"
                        "<br>Delta RMSE: %{x:+.5f}<br>Delta Regret@1: %{y:+.5f}<extra></extra>"
                    ),
                    name=f"{target} / {design}",
                )
            )
    figure.add_shape(
        type="rect",
        x0=-0.04,
        x1=0,
        y0=-0.08,
        y1=0,
        fillcolor="#DCEDE5",
        opacity=0.55,
        line_width=0,
        layer="below",
    )
    figure.add_vline(x=0, line={"color": NAVY, "width": 1.5})
    figure.add_hline(y=0, line={"color": NAVY, "width": 1.5})
    figure.add_annotation(
        x=-0.032,
        y=-0.071,
        text="Both improve",
        showarrow=False,
        font={"size": 16, "color": GREEN},
    )
    figure.update_xaxes(title_text="Change in heldout RMSE, 760 rows - 280 rows (negative is better)")
    figure.update_yaxes(title_text="Change in heldout Regret@1 (negative is better)")
    style_figure(figure, "More phase evidence lowers broad prediction error, but not selected-policy regret reliably")
    add_fact_card(
        figure,
        "More rows help estimate the broad response surface; they do not reliably identify a better policy.",
        "Adding 480 phase-varying rows improves heldout RMSE in 18/20 model-target-design curves, while Regret@1 "
        "improves in only 6/20, is unchanged in 11/20, and worsens in 3/20.",
        "The 20 curves are model-target-design comparisons, not exchangeable replications. Regret is a step function on "
        "a fixed 461-policy development archive; the tied-spine design also uses 238 extra tied checkpoints.",
    )
    return figure, frame


def load_pairwise_tv() -> pd.DataFrame:
    path = RAW_OPTIMUM_DIR / "pairwise_raw_optimum_policy_tv.csv"
    frame = pd.read_csv(path)
    required = {"target", "left_model", "right_model", "weighted_policy_tv"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing raw-optimum pairwise columns: {sorted(missing)}")
    return frame


def pairwise_matrix(frame: pd.DataFrame, target: str) -> tuple[list[str], np.ndarray]:
    models = list(MODEL_LABELS)
    index = {model: position for position, model in enumerate(models)}
    matrix = np.full((len(models), len(models)), np.nan)
    np.fill_diagonal(matrix, 0.0)
    for row in frame[frame["target"] == target].itertuples():
        i = index[row.left_model]
        j = index[row.right_model]
        matrix[i, j] = row.weighted_policy_tv
        matrix[j, i] = row.weighted_policy_tv
    if np.isnan(matrix).any():
        raise ValueError(f"Incomplete pairwise matrix for {target}")
    return [SHORT_MODEL_LABELS[model] for model in models], matrix


def plot_raw_optimum_disagreement() -> tuple[go.Figure, pd.DataFrame]:
    frame = load_pairwise_tv()
    figure = make_subplots(
        rows=1, cols=2, subplot_titles=["Uncheatable raw optima", "Table-9 raw optima"], horizontal_spacing=0.12
    )
    for column, target in enumerate(["uncheatable", "table9"], start=1):
        labels, matrix = pairwise_matrix(frame, target)
        text = np.vectorize(lambda value: f"{value:.2f}")(matrix)
        figure.add_trace(
            go.Heatmap(
                z=matrix,
                x=labels,
                y=labels,
                zmin=0,
                zmax=0.9,
                colorscale="RdYlGn_r",
                text=text,
                texttemplate="%{text}",
                textfont={"size": 13},
                colorbar={"title": "Weighted<br>policy TV", "x": 1.03} if column == 2 else None,
                showscale=column == 2,
                hovertemplate="%{y} vs %{x}<br>Weighted policy TV: %{z:.4f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(tickangle=-28, row=1, col=column)
        figure.update_yaxes(autorange="reversed", row=1, col=column)
    style_figure(figure, "Raw global optima remain model-dependent on the same 3e18 evidence")
    add_fact_card(
        figure,
        "The global optimum is not identified: different plausible models optimize to materially different policies.",
        "Median pairwise weighted-policy TV is 0.4166 for Uncheatable and 0.4239 for Table-9; even the closest "
        "distinct pairs are 0.2144 and 0.2365. Effective-exposure DSP reaches one-bucket corners on both targets.",
        "All models use the same 998-policy 3e18 evidence and optimization protocol. An independent 16-start seed "
        "reproduced the candidates, so disagreement is primarily model-form uncertainty, not optimizer randomness.",
    )
    return figure, frame


def write_source_table(frame: pd.DataFrame, name: str) -> Path:
    path = SOURCE_DIR / name
    frame.to_csv(path, index=False)
    return path


def write_report(manifest: pd.DataFrame) -> Path:
    report = """# Data-mixture progress, July 14-21, 2026

## Executive readout

We now have deployment-scale evidence rather than only 300M proxy fits: matched 280-policy one-phase and two-phase Delphi 3e18 fit panels, plus an append-only archive with 1,518 observations over 1,472 unique phase-weight coordinates. Every completed observation has Uncheatable and native Table-9 BPB.

The main conclusion is a boundary, not a model win: **the data support a local, target-specific two-phase ordering effect, but they do not identify a globally reliable two-phase optimum.** Scale-matched fitting improves two-phase HPR proposals. Random aggregate-preserving phase perturbations are centered near zero under the tested sampler. Effective-exposure ordering shifts Uncheatable favorably relative to aggregate-matched tied controls, but Table-9 does not show the same effect and the model does not select the ex-post best draw. More two-phase examples improve heldout RMSE much more reliably than post-selection regret, while five plausible surrogates still optimize to substantially different raw policies.

**Decision:** do not present or scale a raw global optimum as solved. Treat aggregate selection and local phase ordering as separate estimation problems, and confirm the latter with a fixed candidate-versus-tied-control experiment using fresh paired seeds.

## Experimental setup and estimands

| Item | Definition |
|---|---|
| Training system | Delphi 3e18 configuration: 358,306,688 parameters, about 1.4B training tokens over 3,007 steps, nominal 80%/20% WSD phases |
| Mixture space | 39 buckets; one-phase policies have 38 simplex degrees of freedom, two-phase policies have 76 |
| One-phase restriction | Phase-tied weights: the same mixture is used throughout training |
| Two-phase policy | Independent simplex weights before and after the WSD phase boundary, at the same model and token budget |
| Primary targets | Uncheatable BPB and OLMoBaseEval Table-9 macro BPB; lower is better |
| Regret@1 | Observed BPB of the model-selected heldout policy minus the best observed BPB in the fixed heldout candidate set |
| Weighted-policy TV | Phase-duration-weighted total-variation distance between two two-phase policies; 0 means identical |
| HPR | Hierarchical Phase Replay, a retained-state surrogate with family-level phase replay structure |
| Evidence status | All panels in this report are fit or development evidence. No result here is an untouched final confirmation |

## Questions answered this week

| Question | Current answer | Evidence | Strength and boundary |
|---|---|---|---|
| Does fitting at deployment scale solve 300M-to-3e18 transfer? | It helps two-phase HPR proposals, but does not solve policy selection. | Frozen 124-candidate source-scale comparison | Moderate; best-of-sweep comparisons, not repeated estimators |
| Does arbitrary phase asymmetry help near tied anchors? | Not on average under the preregistered isotropic sampler. | 288 random policies matched to same-anchor, same-seed tied controls | Strong for this sampler; does not rule out structured orderings |
| Can model-directed ordering find local gains? | Effective-exposure ordering shifts Uncheatable by -0.00241 BPB on average; no comparable Table-9 shift. | 20 correlated policy points per phase head, each matched by aggregate KL | Suggestive; shared controls and hyperparameter correlation preclude treating 20 points as independent replications |
| Does more phase evidence solve selection? | It improves broad prediction error more consistently than the selected policy. | Fixed 461-policy archive; 20 model-target-design learning curves | Moderate development evidence; comparisons are heterogeneous, not IID replicates |
| Is the global two-phase optimum identified? | No. | Five models fit to the same 998 policies and optimized with matched multistart procedures | Strong evidence of model-form uncertainty; not evidence that every model is wrong |

## Meeting narrative

### 1. We built the deployment-scale evidence base

[Figure 1](01_evidence_base_expansion.png) is the operational result. The project now has matched fit swarms and targeted intervention panels at the actual Delphi 3e18 training configuration. This turns previously invisible out-of-support errors into measurable failures and permits policy-class, source-scale, and matched phase-ordering comparisons.

### 2. Scale mismatch mattered, but was not the whole failure

[Figure 2](02_scale_matched_hpr_fit.png) applies the same HPR procedure after fitting on either 300M or 3e18 evidence, then evaluates every candidate at 3e18. Refitting at 3e18 reduces the two-phase excess over the observed Uncheatable reference from 0.00470 to 0.00252 BPB and the Table-9 excess from 0.00828 to 0.00197 BPB. It does not improve every policy class: the one-phase Table-9 proposal becomes worse. The result supports scale-matched fitting, but rejects “scale mismatch alone” as an explanation.

### 3. Unlocking phase degrees of freedom is not automatically useful

[Figure 3](03_random_phase_population.png) shows all 288 preregistered random phase policies, not only their means. Each point is an aggregate-preserving phase perturbation minus its same-anchor, same-seed tied control. The six anchor-radius cell means range from -0.21 to +0.03 same-configuration two-run difference SDs, and 38%-60% of points improve. The tested phase fiber contains both wins and losses, but random search has little average target-matched value.

### 4. Structured ordering exposes a local Uncheatable effect

[Figure 4](04_hybrid_phase_ordering.png) holds the aggregate mixture fixed and compares four models' phase orderings with an aggregate-matched tied control. Effective-exposure has mean Uncheatable delta -0.00241 BPB and 19/20 policies improve; the other heads are weaker, and none produces a comparable Table-9 shift. This is the most direct evidence that ordering can matter locally. It is not a frontier claim: the sweep points are correlated, share four controls, and the model-selected effective-exposure candidate scores 0.985503 rather than the ex-post 0.984045 low draw.

### 5. Prediction is improving faster than decision quality

[Figure 5](05_more_rows_fit_vs_selection.png) expands each two-phase fit from 280 to 760 phase-varying policies while holding a 461-policy development archive fixed. Heldout RMSE improves in 18/20 model-target-design comparisons. Regret@1 improves in only 6/20, is unchanged in 11/20, and worsens in 3/20. The bottleneck is therefore not only sample count: support and model-form error are amplified by optimization.

### 6. Global optimization remains unidentified

[Figure 6](06_raw_optimum_disagreement.png) fits five model families to the same 998-policy evidence set and applies the same multistart optimizer. Median pairwise weighted-policy TV among raw optima is 0.4166 for Uncheatable and 0.4239 for Table-9. An independent 16-start seed reproduces the candidates, so the disagreement is primarily model-form uncertainty rather than optimizer randomness. Effective-exposure still reaches one-bucket corners; other models remain outside empirical support. A plausible predicted value is not yet sufficient evidence for a policy.

## Decision and next confirmation

The defensible paper claim today is: **two-phase policies offer local value, but the globally optimal ordering is harder to identify than the aggregate mixture under a fixed swarm budget.** The current evidence does not support declaring one surrogate's raw optimum final.

The next untouched confirmation should isolate one claim:

1. Freeze one model-selected Uncheatable two-phase policy and its exactly aggregate-matched tied control before observing any new outcomes.
2. Run eight fresh paired seed blocks at 3e18. The primary estimand is the mean paired Uncheatable BPB difference; success requires its upper 95% confidence bound to be below zero.
3. Treat Table-9 as a preregistered safety endpoint with a non-inferiority margin, not as another opportunity to select a winner.
4. Do not use the ex-post minimum across a sweep as the estimate of phase-ordering value.

If this test succeeds, the next modeling step is to learn phase ordering locally conditional on a separately estimated aggregate. If it fails, the 0.00241-BPB development shift should be treated as selection or shared-control optimism rather than a transferable curriculum effect.

## Statistical cautions

- The Table-9 value 1.057530 is one unreplicated low draw. It is an observed reference, not a certified frontier.
- Hybrid sweep points are correlated policy/hyperparameter choices and reuse four tied controls; `19/20` is descriptive, not a binomial experiment.
- Random phase-population points also share tied controls within seed blocks; boxes show empirical distributions, not confidence intervals.
- The append-only archive is deliberately heterogeneous and is not an IID test sample. Proposal-source and policy-class stratification remain necessary.
- Compact retained state is a plausible geometry/heldout-fit compromise, not a validated winner; its predicted raw gain is still outside support.

## Figure index

"""
    for row in manifest.itertuples(index=False):
        report += f"- [{row.title}]({row.png}) ([interactive HTML]({row.html}))\n"
    report += """

## Reproduction

Run:

```bash
uv run experiments/domain_phase_mix/exploratory/two_phase_many/build_pi_meeting_weekly_progress_20260721.py
```

Frozen source tables used by the packet are under `source_data/`. Figures 3 and 4 are rebuilt from the append-only heldout registry rather than copied summary statistics. Fieldbook provenance is listed in [the statistical review memo](pi_statistical_review.md).
"""
    path = OUTPUT_DIR / "report.md"
    path.write_text(report)
    return path


def write_review_memo() -> Path:
    """Write the adversarial statistical review that motivated the packet revision."""
    review = """# Statistical review memo for the PI meeting packet

## Review standard

The packet was reviewed as a statistical-learning progress report: every headline should name its estimand, distinguish a fitted or selected quantity from an observed one, state the intervention or sampling design, and bound what can be inferred. This is an internal adversarial review; an independent Claude Code review was not available in this side-conversation run.

## Material issues found and resolved

1. **Ex-post minima were called frontiers.** The revised packet uses "observed reference" for 0.985120 and 1.057530 and explicitly states that the Table-9 reference is an unreplicated low draw.
2. **Random phase effects were reduced to connected means.** Figure 3 now displays all 288 matched random-minus-tied effects. Means are diamonds, not a fitted radius trend, and effects are normalized by the measured same-configuration two-run difference SD.
3. **Hybrid ordering used best-in-sweep values.** Figure 4 now shows all 20 matched candidate-minus-tied effects for each head and target. The model-selected and ex-post lowest effective-exposure policies are marked separately.
4. **Correlated sweep points looked like replications.** The report now states that hybrid points share four controls and are correlated hyperparameter-policy choices. No binomial or independent-sample interpretation is attached to `19/20`.
5. **Setup and decision metrics were implicit.** The report now defines the model/token configuration, policy dimensions, targets, HPR, Regret@1, and weighted-policy TV.
6. **Fit improvement was conflated with selection improvement.** Figure 5 and the report explicitly separate heldout RMSE from post-selection regret and identify the fixed candidate archive.
7. **The conclusion lacked a decision gate.** The report now recommends a fixed two-phase policy versus its exact aggregate-matched tied control under eight fresh paired seed blocks, with a preregistered mean-difference endpoint.

## Remaining strongest objections

- The eight-pair recommendation uses the current repeat-noise scale and should be recalculated if the paired design changes the variance.
- The effective-exposure local effect may be anchor-specific; one confirmed coordinate would not establish a global phase-ordering law.
- The archive's proposal mechanisms are heterogeneous. Pooled heldout metrics can obscure source-specific failure and should not be interpreted as population risk.
- Five-model disagreement proves non-identification under the tested forms and evidence, not that no adequate mechanistic surrogate exists.
- No model recommendation in this packet is confirmatory until a new outcome-sealed experiment succeeds.

## Fieldbook provenance

- `exp_01kxhhy6k73fwz111ab42q5ssq`: Delphi 3e18 two-phase fit swarm.
- `exp_01kxm37amarnnd1wey37nj8ed5`: matched one-phase fit swarm.
- `exp_01kxm37abk8afjmewxhczz2wyv`: adversarial stress panel.
- `exp_01kxwpm5236b4ej68ph3xvra2t`: frontier phase-fiber design.
- `exp_01kxz0nrznpqm9mk08zepcxqxe`: HPR source-scale comparison.
- `exp_01kxz7jz8z240zgytg06m7ez2d`: random phase population.
- `exp_01kxzpcwssap936vs2mn6a9bny`: hybrid phase-ordering panel.
- `exp_01kxtds5p37dk3pxkhfkd5426c`: sample-efficiency and heterogeneous-design audit.
- `exp_01kxwtkg4wbwj2m784t003ymrt`: raw-optimum model comparison.
"""
    path = OUTPUT_DIR / "pi_statistical_review.md"
    path.write_text(review)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    builders = [
        ("01_evidence_base_expansion", "Evidence-base expansion", plot_evidence_base),
        ("02_scale_matched_hpr_fit", "Scale-matched HPR fitting", plot_hpr_source_scale),
        ("03_random_phase_population", "Random phase-population DOE", plot_random_phase_population),
        ("04_hybrid_phase_ordering", "Hybrid phase-ordering validation", plot_hybrid_phase_ordering),
        ("05_more_rows_fit_vs_selection", "More evidence: fit versus selection", plot_fit_vs_selection),
        ("06_raw_optimum_disagreement", "Raw-optimum disagreement", plot_raw_optimum_disagreement),
    ]
    manifest_rows = []
    for stem, title, builder in builders:
        figure, source = builder()
        html_path, png_path = write_figure(figure, stem)
        source_path = write_source_table(source, f"{stem}.csv")
        manifest_rows.append(
            {
                "order": len(manifest_rows) + 1,
                "title": title,
                "html": html_path.name,
                "png": png_path.name,
                "source_data": str(source_path.relative_to(OUTPUT_DIR)),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUTPUT_DIR / "plot_manifest.csv", index=False)
    report_path = write_report(manifest)
    review_path = write_review_memo()
    metadata = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "period_start": "2026-07-14",
        "period_end": "2026-07-21",
        "figure_count": len(manifest),
        "report": report_path.name,
        "statistical_review": review_path.name,
        "archive_observations_after_hybrid": 1518,
        "archive_unique_policies_after_hybrid": 1472,
        "note": "Hybrid rows are complete and coordinate-disjoint but were not yet present in the July-20 registry materialization.",
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(manifest)} figures and report to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
