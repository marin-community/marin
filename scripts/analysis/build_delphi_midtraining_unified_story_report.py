# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Build one unified Plotly HTML report for the Delphi midtraining contamination story.

The report is intentionally local-data-only: it reads the CSV/JSON artifacts
produced by the earlier analysis scripts and the contamination worktree, then
writes a single HTML document with narrative sections, a glossary, a table of
contents, and interactive Plotly figures.

Run:
    uv run --with plotly --with pandas --with numpy \\
      python scripts/analysis/build_delphi_midtraining_unified_story_report.py
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from marin.scaling_laws.scaling_plots import MARKERS, PALETTE
from plotly.subplots import make_subplots

pio.templates.default = "plotly_white"

DEFAULT_OUTPUT = Path("sk_midtrain_analysis_fable/delphi_midtraining_unified_story_report.html")
DEFAULT_CONTAMINATION_ROOT = Path.cwd().parent / "nemotron_contam"

SCALE_ORDER = ["3e18", "9e18", "2e19", "3e19", "9e19", "2e20", "3e20", "1e21", "1e22"]
SCALE_FLOPS = {
    "3e18": 3e18,
    "9e18": 9e18,
    "2e19": 2e19,
    "3e19": 3e19,
    "9e19": 9e19,
    "2e20": 2e20,
    "3e20": 3e20,
    "1e21": 1e21,
    "1e22": 1e22,
}
FIT_CUTOFF_SCALE = "3e20"
HELDOUT_SCALES = {"1e21", "1e22"}

OLD_ERROR_TARGETS = [
    ("p33m67 lr0.33", 0.572544, 0.681570, 19.04, 0.109026),
    ("p33m67 lr0.50", 0.561019, 0.665204, 18.57, 0.104185),
    ("p33m67 lr0.67", 0.559539, 0.661742, 18.27, 0.102203),
    ("p33m67 lr0.83", 0.563027, 0.663669, 17.88, 0.100642),
]

BASE_STEP0_ERRORS = [
    ("base step-0 math loss", 0.7, 2.4),
    ("best-LR endpoint p33m67", 2.9, 18.6),
    ("best-LR endpoint p50m50", 2.4, 16.4),
    ("best-LR endpoint p67m33", 1.7, 12.9),
]

EXPOSURE_BY_SCALE = [
    ("3e18", 0.635),
    ("3e20", 5.755),
    ("1e21", 10.282),
    ("1e22", 20.165),
]

EXPOSURE_BY_MIX = [
    ("p33m67", 20.165),
    ("p50m50", 17.281),
    ("p67m33", 14.015),
]


@dataclass(frozen=True)
class Paths:
    output: Path
    contamination_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contamination-root", type=Path, default=DEFAULT_CONTAMINATION_ROOT)
    return parser.parse_args()


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_json(path: str | Path) -> Any:
    with open(path) as handle:
        return json.load(handle)


def scale_label(value: Any) -> str:
    if isinstance(value, str) and value in SCALE_FLOPS:
        return value
    numeric = float(value)
    for label, flops in SCALE_FLOPS.items():
        if math.isclose(numeric, flops, rel_tol=1e-9):
            return label
    raise ValueError(f"Unknown scale value: {value!r}")


def scale_sort_key(value: Any) -> int:
    return SCALE_ORDER.index(scale_label(value))


def normalized_compute(scale_flops: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(scale_flops, dtype=float) / 1e18


def floor_power(scale_flops: np.ndarray, floor: float, amplitude: float, exponent: float) -> np.ndarray:
    return floor + amplitude * np.power(normalized_compute(scale_flops), -exponent)


def figure_html(fig: go.Figure, div_id: str) -> str:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=70, r=30, t=72, b=58),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left", x=0),
    )
    return pio.to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        div_id=div_id,
        config={"responsive": True, "displaylogo": False},
    )


def table_html(frame: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    out = frame.copy()
    if columns is not None:
        out = out[columns]
    if max_rows is not None:
        out = out.head(max_rows)
    return out.to_html(index=False, escape=False, classes="data-table", border=0)


def metric_card(label: str, value: str, note: str) -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-label">{html.escape(label)}</div>
      <div class="metric-value">{html.escape(value)}</div>
      <div class="metric-note">{html.escape(note)}</div>
    </div>
    """


def section(section_id: str, eyebrow: str, title: str, body: str) -> str:
    return f"""
    <section id="{html.escape(section_id)}" class="report-section">
      <div class="section-eyebrow">{html.escape(eyebrow)}</div>
      <h2>{html.escape(title)}</h2>
      {body}
    </section>
    """


def make_original_error_figure() -> go.Figure:
    names = [row[0] for row in OLD_ERROR_TARGETS]
    actual = [row[1] for row in OLD_ERROR_TARGETS]
    predicted = [row[2] for row in OLD_ERROR_TARGETS]
    pct_error = [row[3] for row in OLD_ERROR_TARGETS]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("1e22 old 4plus loss: actual vs prediction", "Prediction error at 1e22"),
    )
    fig.add_trace(go.Bar(x=names, y=actual, name="actual", marker_color=PALETTE[0]), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=predicted, name="predicted", marker_color=PALETTE[1]), row=1, col=1)
    fig.add_trace(
        go.Bar(
            x=names,
            y=pct_error,
            name="prediction error",
            marker_color=PALETTE[3],
            hovertemplate="%{x}<br>prediction error=%{y:.2f}%<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="loss", row=1, col=1)
    fig.update_yaxes(title_text="prediction minus actual (%)", row=1, col=2)
    fig.update_layout(title="Original K=0.20 p33m67 miss on the old validation target", barmode="group", height=460)
    return fig


def make_original_curve_figure() -> go.Figure:
    endpoints = read_csv("midtrain_analysis_outputs/small_final_loss_scaling/endpoints.csv")
    fits = read_csv("midtrain_analysis_outputs/small_final_loss_scaling/fit_summary.csv")
    predictions = read_csv("midtrain_analysis_outputs/small_final_loss_scaling/extrapolation_predictions.csv")

    points = endpoints[
        endpoints["metric_label"].eq("math_val_loss")
        & endpoints["mix"].eq("p33m67")
        & endpoints["lr"].astype(str).eq("50")
    ].copy()
    points["scale_label"] = points["scale_flops"].map(scale_label)
    points["scale_order"] = points["scale_label"].map(SCALE_ORDER.index)
    points = points.sort_values("scale_order")

    fit = fits[
        fits["metric_label"].eq("math_val_loss")
        & fits["mix"].eq("p33m67")
        & fits["lr"].astype(str).eq("50")
        & fits["fit_kind"].eq("floor_plus_power")
    ].iloc[0]
    xs = np.logspace(math.log10(SCALE_FLOPS["3e18"]), math.log10(SCALE_FLOPS["1e22"]), 300)
    ys = floor_power(xs, float(fit["floor"]), float(fit["amplitude"]), float(fit["exponent"]))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="fit through 3e20",
            line=dict(color=PALETTE[1], width=3),
            hovertemplate="scale=%{x:.2e}<br>fit=%{y:.4f}<extra></extra>",
        )
    )
    train_points = points[~points["scale_label"].isin(HELDOUT_SCALES)]
    fig.add_trace(
        go.Scatter(
            x=train_points["scale_flops"],
            y=train_points["value"],
            mode="markers",
            name="actual fit",
            marker=dict(color=PALETTE[0], symbol="circle", size=11, line=dict(color="white", width=1)),
            text=train_points["scale_label"],
            hovertemplate="%{text}<br>actual=%{y:.4f}<extra></extra>",
        )
    )

    heldout_pred = predictions[
        predictions["metric_label"].eq("math_val_loss")
        & predictions["mix"].eq("p33m67")
        & predictions["lr"].astype(str).eq("50")
        & predictions["fit_kind"].eq("floor_plus_power")
    ].copy()
    heldout_pred["target_label"] = heldout_pred["target_scale_flops"].map(scale_label)
    fig.add_trace(
        go.Scatter(
            x=heldout_pred["target_scale_flops"],
            y=heldout_pred["observed"],
            mode="markers",
            name="actual heldout",
            marker=dict(color=PALETTE[3], symbol="triangle-up", size=11, line=dict(color="white", width=1)),
            text=heldout_pred["target_label"],
            hovertemplate="%{text}<br>actual=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=heldout_pred["target_scale_flops"],
            y=heldout_pred["predicted"],
            mode="markers",
            name="heldout prediction",
            marker=dict(color=PALETTE[1], symbol="x", size=14, line=dict(width=3)),
            text=heldout_pred["target_label"],
            customdata=np.stack([heldout_pred["observed"], heldout_pred["predicted"]], axis=-1),
            hovertemplate="%{text}<br>actual=%{customdata[0]:.4f}<br>predicted=%{customdata[1]:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=SCALE_FLOPS[FIT_CUTOFF_SCALE], line_dash="dash", line_color="#94a3b8")
    fig.update_xaxes(type="log", title_text="base pretraining compute (FLOPs)")
    fig.update_yaxes(title_text="old 4plus math validation loss")
    fig.update_layout(title="Old target: 1e21 was close, 1e22 was far below the fitted curve", height=560)
    return fig


def make_endpoint_form_figure() -> go.Figure:
    summary = read_csv("midtrain_analysis_outputs/small_final_loss_scaling/endpoint_form_comparison_summary.csv")
    summary["scale_label"] = summary["target_scale"].map(scale_label)
    order = summary[summary["scale_label"].eq("1e22")].sort_values("mean_abs_pct_error")["form"].tolist()
    fig = go.Figure()
    for scale, color in [("1e21", PALETTE[0]), ("1e22", PALETTE[3])]:
        sub = summary[summary["scale_label"].eq(scale)].set_index("form").loc[order].reset_index()
        fig.add_trace(
            go.Bar(
                x=sub["form_label"],
                y=sub["mean_abs_pct_error"],
                name=scale,
                marker_color=color,
                hovertemplate="%{x}<br>mean abs error=%{y:.2f}%<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="heldout mean absolute percent error")
    fig.update_xaxes(tickangle=25)
    fig.update_layout(title="The original target stayed hard across several endpoint forms", barmode="group", height=520)
    return fig


def make_fit_family_figure() -> go.Figure:
    old = read_csv("sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report_summary.csv").iloc[0]
    iso = read_csv("sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report_isotoken_only_summary.csv").iloc[0]
    clean = read_csv("sk_midtrain_analysis_fable/delphi_k020_clean_seen_fit_family_report_summary.csv").iloc[0]
    rows = pd.DataFrame(
        [
            {
                "report": "old target, all series",
                "best_model": old["model_label"],
                "heldout_mae_pct": old["heldout_mae_pct"],
                "focus_error_pct": old["1e22_focus_error_pct"],
            },
            {
                "report": "old target, iso-token only",
                "best_model": iso["model_label"],
                "heldout_mae_pct": iso["heldout_mae_pct"],
                "focus_error_pct": iso["1e22_focus_error_pct"],
            },
            {
                "report": "clean-seen K=0.20",
                "best_model": clean["model_label"],
                "heldout_mae_pct": clean["heldout_mae_pct"],
                "focus_error_pct": clean["1e22_050_error_pct"],
            },
        ]
    )
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Heldout endpoint MAE", "Focus 1e22 error"))
    fig.add_trace(
        go.Bar(x=rows["report"], y=rows["heldout_mae_pct"], name="heldout MAE", marker_color=PALETTE[0]),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=rows["report"], y=rows["focus_error_pct"], name="focus error", marker_color=PALETTE[3]),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="percent", row=1, col=1)
    fig.update_yaxes(title_text="percent", row=1, col=2)
    fig.update_xaxes(tickangle=20)
    fig.update_layout(title="The same fit families work on cleaner/fixed-token views", showlegend=False, height=480)
    return fig


def make_base_step0_figure() -> go.Figure:
    rows = pd.DataFrame(BASE_STEP0_ERRORS, columns=["series", "heldout_1e21_pct", "heldout_1e22_pct"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=rows["series"], y=rows["heldout_1e21_pct"], name="1e21", marker_color=PALETTE[0]))
    fig.add_trace(go.Bar(x=rows["series"], y=rows["heldout_1e22_pct"], name="1e22", marker_color=PALETTE[1]))
    fig.update_yaxes(title_text="Chinchilla holdout error (%)")
    fig.update_xaxes(tickangle=20)
    fig.update_layout(title="Base step-0 math loss was smooth; endpoint loss was not", barmode="group", height=460)
    return fig


def fit_curve_from_row(row: pd.Series, xs: np.ndarray, prefix: str = "fp_") -> np.ndarray:
    if prefix:
        floor = float(row[f"{prefix}floor"])
        amplitude = float(row[f"{prefix}amplitude"])
        alpha = float(row[f"{prefix}alpha"])
    else:
        floor = float(row["floor"])
        amplitude = float(row["amplitude"])
        alpha = float(row["alpha"])
    return floor_power(xs, floor, amplitude, alpha)


def make_old_isotoken_figure() -> go.Figure:
    isotoken = read_csv("sk_midtrain_analysis_fable/isotoken_endpoints.csv")
    k020 = read_csv("sk_midtrain_analysis_fable/isoflop_k020_endpoints.csv")
    points = pd.concat([isotoken, k020], ignore_index=True)
    fits = read_csv("sk_midtrain_analysis_fable/isotoken_scaling_fits.csv")
    xs = np.logspace(math.log10(SCALE_FLOPS["3e18"]), math.log10(SCALE_FLOPS["1e22"]), 300)
    series_order = ["tok500m", "tok1b", "tok2b", "tok4b", "tok8b", "k0p20"]
    labels = {
        "tok500m": "iso-token 500M",
        "tok1b": "iso-token 1B",
        "tok2b": "iso-token 2B",
        "tok4b": "iso-token 4B",
        "tok8b": "iso-token 8B",
        "k0p20": "K=0.20",
    }
    fig = go.Figure()
    for index, series in enumerate(series_order):
        sub = points[points["series"].eq(series)].copy()
        if sub.empty:
            continue
        sub["scale_name"] = sub["scale_flops"].map(scale_label)
        color = "#D62728" if series == "k0p20" else PALETTE[index % len(PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=sub["scale_flops"],
                y=sub["value"],
                mode="markers+lines",
                name=labels[series],
                marker=dict(color=color, symbol=MARKERS[index % len(MARKERS)], size=8),
                line=dict(color=color, dash="dash" if series == "k0p20" else "solid", width=2),
                text=sub["scale_name"],
                hovertemplate="%{fullData.name}<br>%{text}<br>old loss=%{y:.4f}<extra></extra>",
            )
        )
        fit = fits[fits["series"].eq(series)]
        if not fit.empty:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=fit_curve_from_row(fit.iloc[0], xs, prefix="fp_"),
                    mode="lines",
                    name=f"{labels[series]} fit",
                    showlegend=False,
                    line=dict(color=color, dash="dot", width=2),
                    hovertemplate="%{fullData.name}<br>scale=%{x:.2e}<br>fit=%{y:.4f}<extra></extra>",
                )
            )
    fig.add_vline(x=SCALE_FLOPS[FIT_CUTOFF_SCALE], line_dash="dash", line_color="#94a3b8")
    fig.update_xaxes(type="log", title_text="base pretraining compute (FLOPs)")
    fig.update_yaxes(title_text="old 4plus math validation loss")
    fig.update_layout(title="Old target: fixed-token ladders were smooth, K=0.20 was not", height=620)
    return fig


def make_clean_unified_figure() -> go.Figure:
    predictions = read_csv("sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_unified_report_predictions.csv")
    series_order = ["tok1b", "tok2b", "tok4b", "tok8b", "k0p20"]
    target_order = [("old_4plus", "old 4plus validation"), ("clean_seen", "new clean-seen validation")]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=[label for _, label in target_order])
    for row_index, (target_key, _) in enumerate(target_order, start=1):
        target = predictions[predictions["target_key"].eq(target_key)].copy()
        for index, series in enumerate(series_order):
            sub = target[target["series"].eq(series)].sort_values("scale_order")
            if sub.empty:
                continue
            color = PALETTE[index % len(PALETTE)] if series != "k0p20" else "#D62728"
            name = str(sub["series_label"].iloc[0])
            fig.add_trace(
                go.Scatter(
                    x=sub["scale_flops"],
                    y=sub["actual"],
                    mode="markers+lines",
                    name=name if row_index == 1 else None,
                    showlegend=row_index == 1,
                    legendgroup=series,
                    marker=dict(color=color, symbol=MARKERS[index % len(MARKERS)], size=8),
                    line=dict(color=color, width=2),
                    text=sub["scale"].map(scale_label),
                    hovertemplate="%{fullData.name}<br>%{text}<br>actual=%{y:.4f}<extra></extra>",
                ),
                row=row_index,
                col=1,
            )
            heldout = sub[sub["scale"].map(scale_label).isin(HELDOUT_SCALES)]
            fig.add_trace(
                go.Scatter(
                    x=heldout["scale_flops"],
                    y=heldout["prediction"],
                    mode="markers",
                    name=f"{name} heldout prediction" if row_index == 1 else None,
                    showlegend=False,
                    legendgroup=series,
                    marker=dict(color=color, symbol="x", size=12, line=dict(width=3)),
                    text=heldout["scale"].map(scale_label),
                    customdata=np.stack([heldout["actual"], heldout["error_pct"]], axis=-1),
                    hovertemplate="%{fullData.name}<br>%{text}<br>actual=%{customdata[0]:.4f}<br>prediction=%{y:.4f}<br>error=%{customdata[1]:+.2f}%<extra></extra>",
                ),
                row=row_index,
                col=1,
            )
    fig.add_vline(x=SCALE_FLOPS[FIT_CUTOFF_SCALE], line_dash="dash", line_color="#94a3b8")
    fig.update_xaxes(type="log", title_text="base pretraining compute (FLOPs)", row=2, col=1)
    fig.update_yaxes(title_text="loss", row=1, col=1)
    fig.update_yaxes(title_text="loss", row=2, col=1)
    fig.update_layout(
        title="Clean-seen target removes the K=0.20 outlier while preserving smooth iso-token fits", height=820
    )
    return fig


def make_seen_partition_figure() -> go.Figure:
    points = read_csv("sk_midtrain_analysis_fable/delphi_k020_seen_partition_scaling_points.csv")
    summary = read_csv("sk_midtrain_analysis_fable/delphi_k020_seen_partition_scaling_fit_summary.csv")
    targets = [
        ("old_full", "old full 4plus", "old_4plus_loss", "#64748b", "circle"),
        ("clean_retained", "retained clean", "clean_seen_loss", PALETTE[0], "circle"),
        ("dropped_seen", "dropped contaminated", "dropped_seen_loss", "#D62728", "diamond"),
    ]
    xs = np.logspace(math.log10(SCALE_FLOPS["3e18"]), math.log10(SCALE_FLOPS["1e22"]), 300)
    fig = go.Figure()
    for target_key, label, column, color, marker in targets:
        sub_summary = summary[summary["target_key"].eq(target_key)].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=fit_curve_from_row(sub_summary, xs, prefix=""),
                mode="lines",
                name=f"{label} fit",
                line=dict(color=color, width=3, dash="dot" if target_key == "old_full" else "solid"),
                hovertemplate="%{fullData.name}<br>scale=%{x:.2e}<br>fit=%{y:.4f}<extra></extra>",
            )
        )
        for split, symbol_suffix in [("fit", marker), ("heldout", "triangle-up")]:
            sub = points[points["split"].eq(split)].copy()
            fig.add_trace(
                go.Scatter(
                    x=sub["scale_flops"],
                    y=sub[column],
                    mode="markers",
                    name=f"{label} actual {split}",
                    showlegend=split == "fit",
                    legendgroup=target_key,
                    marker=dict(color=color, symbol=symbol_suffix, size=10, line=dict(color="white", width=1)),
                    text=sub["scale"].map(scale_label),
                    hovertemplate="%{fullData.name}<br>%{text}<br>actual=%{y:.4f}<extra></extra>",
                )
            )
        heldout_rows = [
            (
                "1e21",
                float(sub_summary["pred_1e21"]),
                float(sub_summary["actual_1e21"]),
                float(sub_summary["error_1e21_pct"]),
            ),
            (
                "1e22",
                float(sub_summary["pred_1e22"]),
                float(sub_summary["actual_1e22"]),
                float(sub_summary["error_1e22_pct"]),
            ),
        ]
        fig.add_trace(
            go.Scatter(
                x=[SCALE_FLOPS[label_] for label_, _, _, _ in heldout_rows],
                y=[pred for _, pred, _, _ in heldout_rows],
                mode="markers",
                name=f"{label} heldout prediction",
                showlegend=False,
                legendgroup=target_key,
                marker=dict(color=color, symbol="x", size=13, line=dict(width=3)),
                text=[label_ for label_, _, _, _ in heldout_rows],
                customdata=np.array([[actual, error_pct] for _, _, actual, error_pct in heldout_rows]),
                hovertemplate="%{fullData.name}<br>%{text}<br>actual=%{customdata[0]:.4f}<br>prediction=%{y:.4f}<br>error=%{customdata[1]:+.2f}%<extra></extra>",
            )
        )
    fig.add_vline(x=SCALE_FLOPS[FIT_CUTOFF_SCALE], line_dash="dash", line_color="#94a3b8")
    fig.update_xaxes(type="log", title_text="base pretraining compute (FLOPs)")
    fig.update_yaxes(title_text="loss")
    fig.update_layout(title="Seen partition: clean retained fixes the miss; dropped contaminated keeps it", height=640)
    return fig


def make_final_error_figure() -> go.Figure:
    seen = read_csv("sk_midtrain_analysis_fable/delphi_k020_seen_partition_scaling_fit_summary.csv")
    iso = read_csv("sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_unified_report_fit_summary.csv")
    seen_rows = seen[["target_label", "error_1e22_pct", "abs_error_1e22"]].copy()
    clean_iso = iso[iso["target_key"].eq("clean_seen")][["series_label", "error_1e22_pct", "abs_error_1e22_pct"]].copy()
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("K=0.20 seen-partition 1e22 error", "Clean-seen 1e22 error by series")
    )
    fig.add_trace(
        go.Bar(
            x=seen_rows["target_label"],
            y=seen_rows["error_1e22_pct"],
            marker_color=["#64748b", PALETTE[0], "#D62728"],
            name="K=0.20",
            hovertemplate="%{x}<br>error=%{y:+.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=clean_iso["series_label"],
            y=clean_iso["error_1e22_pct"],
            marker_color=PALETTE[1],
            name="clean-seen",
            hovertemplate="%{x}<br>error=%{y:+.2f}%<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0, line_color="#94a3b8", line_width=1)
    fig.update_yaxes(title_text="prediction minus actual (%)")
    fig.update_xaxes(tickangle=15)
    fig.update_layout(
        title="Final 1e22 errors after separating target and token-budget confounds", showlegend=False, height=480
    )
    return fig


def make_jaccard_histogram_figure(paths: Paths) -> go.Figure:
    val_hist_path = paths.contamination_root / "plots/4plus_jaccard_val_doc_max_histogram_0p05.csv"
    pair_hist_path = paths.contamination_root / "plots/4plus_jaccard_pair_histogram_0p05.csv"
    val_hist = read_csv(val_hist_path)
    pair_hist = read_csv(pair_hist_path)
    val_hist["bin"] = val_hist.apply(lambda row: f"{row['bin_start']:.2f}-{row['bin_end']:.2f}", axis=1)
    pair_hist["bin"] = pair_hist.apply(lambda row: f"{row['bin_start']:.2f}-{row['bin_end']:.2f}", axis=1)
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("validation docs by max train Jaccard", "verified near-duplicate pairs")
    )
    fig.add_trace(
        go.Bar(x=val_hist["bin"], y=val_hist["val_doc_count"], name="validation docs", marker_color=PALETTE[0]),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=pair_hist["bin"], y=pair_hist["pair_count"], name="pairs", marker_color=PALETTE[1]),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Jaccard bin", tickangle=35)
    fig.update_yaxes(title_text="count")
    fig.update_layout(title="Fuzzy contamination was invisible to exact dedup but visible by Jaccard", height=500)
    return fig


def make_exposure_figure() -> go.Figure:
    by_scale = pd.DataFrame(EXPOSURE_BY_SCALE, columns=["scale", "tokens_m"])
    by_mix = pd.DataFrame(EXPOSURE_BY_MIX, columns=["mix", "tokens_m"])
    fig = make_subplots(rows=1, cols=2, subplot_titles=("p33m67 K=0.20 exposure by scale", "1e22 exposure by mix"))
    fig.add_trace(
        go.Scatter(
            x=[SCALE_FLOPS[s] for s in by_scale["scale"]],
            y=by_scale["tokens_m"],
            mode="markers+lines",
            name="p33m67",
            marker=dict(color=PALETTE[3], size=10),
            line=dict(color=PALETTE[3], width=3),
            text=by_scale["scale"],
            hovertemplate="%{text}<br>exposed val tokens=%{y:.3f}M<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=by_mix["mix"],
            y=by_mix["tokens_m"],
            name="1e22",
            marker_color=[PALETTE[3], PALETTE[1], PALETTE[0]],
            hovertemplate="%{x}<br>exposed val tokens=%{y:.3f}M<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(type="log", title_text="base pretraining compute (FLOPs)", row=1, col=1)
    fig.update_xaxes(title_text="mix", row=1, col=2)
    fig.update_yaxes(title_text="combined exposed validation tokens (M)")
    fig.update_layout(title="Actual exposure grew with scale and math fraction", height=500, showlegend=False)
    return fig


def make_ppl_gap_figure(paths: Paths) -> go.Figure:
    data = read_json(paths.contamination_root / "plots/ppl_gap_report_data.json")
    scales = [scale for scale in data["scales"] if scale in SCALE_FLOPS]
    docs = pd.DataFrame(data["docs"])
    band_order = ["clean", "j050", "j060", "j075", "j088"]
    band_colors = {
        "clean": PALETTE[0],
        "j050": "#4393c3",
        "j060": "#92c5de",
        "j075": "#f4a582",
        "j088": "#b2182b",
        "train_twin": "#111827",
    }
    fig = go.Figure()
    for band in band_order:
        sub = docs[(docs["band"].eq(band)) & (docs["role"].eq("val"))]
        if sub.empty:
            continue
        means = []
        for scale in scales:
            means.append(float(np.mean([row.get(scale, np.nan) for row in sub["loss"]])))
        fig.add_trace(
            go.Scatter(
                x=[SCALE_FLOPS[scale] for scale in scales],
                y=means,
                mode="markers+lines",
                name=band,
                marker=dict(color=band_colors[band], size=9),
                line=dict(color=band_colors[band], width=3),
                text=scales,
                hovertemplate="%{fullData.name}<br>%{text}<br>mean loss=%{y:.4f}<extra></extra>",
            )
        )
    twin = docs[(docs["band"].eq("j088")) & (docs["role"].eq("train_twin"))]
    if not twin.empty:
        means = []
        for scale in scales:
            means.append(float(np.mean([row.get(scale, np.nan) for row in twin["loss"]])))
        fig.add_trace(
            go.Scatter(
                x=[SCALE_FLOPS[scale] for scale in scales],
                y=means,
                mode="markers+lines",
                name="j088 train twins",
                marker=dict(color=band_colors["train_twin"], symbol="x", size=10),
                line=dict(color=band_colors["train_twin"], width=2, dash="dash"),
                text=scales,
                hovertemplate="%{fullData.name}<br>%{text}<br>mean loss=%{y:.4f}<extra></extra>",
            )
        )
    fig.update_xaxes(type="log", title_text="base pretraining compute (FLOPs)")
    fig.update_yaxes(title_text="curated document mean loss")
    fig.update_layout(title="Mechanism check: high-J documents improve much faster at 1e22", height=560)
    return fig


def make_artifact_table() -> str:
    rows = pd.DataFrame(
        [
            {
                "artifact": "Original public report",
                "location": (
                    '<a href="https://ahmeda14960.github.io/delphi-midtraining/?v=c86be93c">delphi-midtraining public dashboard</a>'
                ),
                "status": "public",
            },
            {
                "artifact": "GitHub tracking issue",
                "location": (
                    '<a href="https://github.com/marin-community/marin/issues/6742">marin-community/marin#6742</a>'
                ),
                "status": "public",
            },
            {
                "artifact": "Contamination branch",
                "location": '<a href="https://github.com/marin-community/marin/tree/deconamint">deconamint</a>',
                "status": "public branch",
            },
            {
                "artifact": "Final retrospective",
                "location": "<code>.agents/logbooks/midtraining_prediction_final.md</code>",
                "status": "local",
            },
            {
                "artifact": "Clean-seen K=0.20 summary",
                "location": (
                    "<code>gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_k020/summary_p33m67_clean_seen_1e22_k020.csv</code>"
                ),
                "status": "GCS",
            },
            {
                "artifact": "Clean-seen iso-token summary",
                "location": (
                    "<code>gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_isotoken_p33m67_lr50/summary_p33m67_isotoken_clean_seen_1e22.csv</code>"
                ),
                "status": "GCS",
            },
            {
                "artifact": "Seen-partition output root",
                "location": (
                    "<code>gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_seen_partition_1e22_k020_lr50</code>"
                ),
                "status": "GCS",
            },
        ]
    )
    return table_html(rows)


def glossary_html() -> str:
    terms = pd.DataFrame(
        [
            {
                "term": "old 4plus validation",
                "meaning": (
                    "The original Nemotron-CC-Math 4plus math validation anchor used by the early Delphi midtraining scaling reports."
                ),
            },
            {
                "term": "clean-seen validation",
                "meaning": (
                    "A validation set decontaminated against documents actually seen by the 1e22 p33m67 K=0.20 math midtraining run."
                ),
            },
            {
                "term": "dropped contaminated",
                "meaning": (
                    "The complement of the clean-seen set: validation documents removed because the seen training stream contained same-source or near-duplicate evidence."
                ),
            },
            {
                "term": "K=0.20",
                "meaning": (
                    "A midtraining budget equal to 20% of the base model pretraining token budget. It is not a fixed-token condition."
                ),
            },
            {
                "term": "iso-token",
                "meaning": (
                    "A control ladder where every base scale gets the same total midtraining token budget, such as 1B, 2B, 4B, or 8B tokens."
                ),
            },
            {
                "term": "p33m67",
                "meaning": "The midtraining mix with about 33% pretraining-like data and 67% math data.",
            },
            {
                "term": "heldout endpoint",
                "meaning": "The 1e21 and 1e22 points, excluded from fits trained through 3e20.",
            },
            {
                "term": "Jaccard near-duplicate",
                "meaning": (
                    "A fuzzy overlap measure over normalized 5-character shingles. High values mean two extracted documents share substantial text."
                ),
            },
            {
                "term": "same-source/window leakage",
                "meaning": (
                    "The split excluded validation windows, but other windows from the same source document could still appear in the training stream."
                ),
            },
            {
                "term": "prediction error",
                "meaning": (
                    "Prediction minus actual. Positive error means the fit predicted too high a loss; the model did better than expected."
                ),
            },
        ]
    )
    return table_html(terms)


def render_report(paths: Paths) -> str:
    original_error = make_original_error_figure()
    original_curve = make_original_curve_figure()
    endpoint_forms = make_endpoint_form_figure()
    base_step0 = make_base_step0_figure()
    fit_family = make_fit_family_figure()
    old_isotoken = make_old_isotoken_figure()
    jaccard = make_jaccard_histogram_figure(paths)
    exposure = make_exposure_figure()
    ppl_gap = make_ppl_gap_figure(paths)
    clean_unified = make_clean_unified_figure()
    seen_partition = make_seen_partition_figure()
    final_error = make_final_error_figure()

    original_table = pd.DataFrame(
        OLD_ERROR_TARGETS,
        columns=["series", "old_1e22_actual", "prediction", "prediction_error_pct", "loss_error"],
    )
    seen_summary = read_csv("sk_midtrain_analysis_fable/delphi_k020_seen_partition_scaling_fit_summary.csv")
    iso_summary = read_csv("sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_unified_report_fit_summary.csv")
    fit_rows = pd.DataFrame(
        [
            {"fact": "old K=0.20 lr0.50 1e22 error", "value": "+18.56%", "source": "old 4plus target"},
            {"fact": "clean-seen K=0.20 lr0.50 1e22 error", "value": "+2.83%", "source": "clean-seen target"},
            {
                "fact": "dropped contaminated 1e22 absolute loss error",
                "value": "+0.0999",
                "source": "seen-partition complement",
            },
            {"fact": "retained clean 1e22 absolute loss error", "value": "+0.0233", "source": "seen partition"},
            {
                "fact": "iso-token clean-seen 1e22 errors",
                "value": "-2.31% to -2.82%",
                "source": "1B/2B/4B/8B fixed-token ladders",
            },
        ]
    )

    cards = "\n".join(
        [
            metric_card("+18.6%", "old K=0.20 1e22 miss", "p33m67 lr0.50 on old 4plus"),
            metric_card("+2.83%", "clean-seen K=0.20 miss", "same lr0.50 fit target"),
            metric_card("+0.0999", "dropped-set absolute miss", "1e22 loss error on contaminated complement"),
            metric_card("-2.3% to -2.8%", "clean iso-token errors", "1e22 fixed-token clean-seen series"),
        ]
    )

    sections = [
        section(
            "original",
            "1. Original Symptom",
            "The old 4plus target made 1e22 look too good",
            f"""
            <p>The frozen original report fit endpoint laws through 3e20 and held out 1e21/1e22. The p33m67 K=0.20 ladder was close at 1e21 but badly high at 1e22: the fit predicted loss around 0.665 for lr0.50 while the old target measured 0.561.</p>
            <p>The old target was <code>eval/nemotron_cc_math_v1/4plus/loss_anchor</code>. The sign convention in this report is prediction minus actual, so positive error means the model did better than the fit expected.</p>
            {figure_html(original_error, "original-error")}
            {figure_html(original_curve, "original-curve")}
            <h3>Original p33m67 K=0.20 old-target 1e22 numbers</h3>
            {table_html(original_table)}
            """,
        ),
        section(
            "base-and-fits",
            "2. Failed Explanations",
            "The base models were smooth, and extra fit forms did not fix the old target",
            f"""
            <p>The step-0 base loss did not show the same failure. A Chinchilla-style fit through 3e20 predicted base step-0 math loss at 1e22 within about +2.4%, while the endpoint p33m67 old-target fit missed by +18.6%.</p>
            {figure_html(base_step0, "base-step0")}
            <p>We then tried per-recipe power laws, Chinchilla floor-plus-power fits, pooled LR-aware fits, log-log fits, parameter/data axes, base rows at D=0, and separate base/improvement components. These fits described fixed-token series, but the old K=0.20 target remained an outlier.</p>
            {figure_html(endpoint_forms, "endpoint-forms")}
            {figure_html(fit_family, "fit-family")}
            """,
        ),
        section(
            "token-budget",
            "3. Token-Budget Confound",
            "K=0.20 was never a fixed-token ladder",
            f"""
            <p>K=0.20 spends 20% of the base model's pretraining token budget on midtraining. In p33m67 this means the total midtraining budget grows from about 0.245B tokens at 3e18 to about 32B tokens at 1e22, and about 67% of that budget is math.</p>
            <p>The iso-token controls held the midtraining token budget fixed while sweeping base scale. On the old target, fixed-token ladders had small 1e22 errors around -3% to -4%; K=0.20 had the large positive error.</p>
            {figure_html(old_isotoken, "old-isotoken")}
            """,
        ),
        section(
            "contamination",
            "4. Validation Contamination",
            "The old validation split had fuzzy and same-source leakage",
            f"""
            <p>The exact duplicate scan found zero duplicate document hashes across the 45.1M-doc corpus. That result was not enough. Fuzzy MinHash/LSH plus exact 5-character-shingle Jaccard verification found substantial near-duplicate overlap between train and validation documents.</p>
            <p>At verified Jaccard >= 0.75, 9,757 / 57,243 validation docs were implicated, touching 6,839 / 12,500 validation windows and 9.53M / 51.20M validation tokens.</p>
            {figure_html(jaccard, "jaccard-hist")}
            <p>The actual exposure replay made the mechanism scale-dependent. For p33m67 K=0.20, combined exposed validation tokens grew from 0.635M at 3e18 to 20.165M at 1e22. At 1e22 the exposure also tracked math fraction across mixes.</p>
            {figure_html(exposure, "exposure")}
            <p>The curated perplexity-gap study showed the same mechanism at document level. High-J documents improved far more at 1e22 than clean documents, consistent with memorization or near-twin exposure rather than a generic base-scaling effect.</p>
            {figure_html(ppl_gap, "ppl-gap")}
            """,
        ),
        section(
            "clean-seen",
            "5. Clean-Seen Re-Evals",
            "The endpoint fits became smooth on the actual-seen clean target",
            f"""
            <p>The final clean-seen set was built against documents actually seen by the 1e22 p33m67 K=0.20 math midtraining stream. It kept 3,367 docs, 2,265,243 tokens, and 553 eval sequences.</p>
            <p>The K=0.20 lr0.50 1e22 error moved from +18.56% on old 4plus to +2.83% on clean-seen. The dropped contaminated complement retained a large absolute miss: +0.0999 loss at 1e22, nearly the old target's +0.1042.</p>
            {figure_html(clean_unified, "clean-unified")}
            {figure_html(seen_partition, "seen-partition")}
            {figure_html(final_error, "final-error")}
            <h3>Final compact facts</h3>
            {table_html(fit_rows)}
            <h3>Seen-partition summary</h3>
            {table_html(seen_summary[["target_label", "actual_1e22", "pred_1e22", "error_1e22_pct", "abs_error_1e22", "heldout_mae_pct"]])}
            <h3>Clean-seen iso-token summary</h3>
            {table_html(iso_summary[iso_summary["target_key"].eq("clean_seen")][["series_label", "actual_1e22", "pred_1e22", "error_1e22_pct", "heldout_mae_pct"]])}
            """,
        ),
        section(
            "conclusion",
            "6. Current Interpretation",
            "The best explanation is an eval-target confound plus token-budget confound",
            f"""
            <p>The current evidence supports a validation/measurement confound rather than a broken law of midtraining. The old K=0.20 target mixed base scale, midtraining token budget, math exposure, and near-duplicate validation exposure. Once we fixed token budget or moved to actual-seen clean validation, the 1e22 endpoint errors returned to low single digits.</p>
            <p>The result is not a claim that every old-target artifact is fully explained. The clean-seen target is built against the 1e22 p33m67 K=0.20 seen set. Per-mix actual-seen clean sets would be the stricter follow-up for p50m50 and p67m33.</p>
            <h3>Artifact links</h3>
            {make_artifact_table()}
            """,
        ),
    ]

    nav_items = [
        ("glossary", "Glossary"),
        ("original", "Original"),
        ("base-and-fits", "Fits"),
        ("token-budget", "Token Budget"),
        ("contamination", "Contamination"),
        ("clean-seen", "Clean-Seen"),
        ("conclusion", "Conclusion"),
    ]
    nav = "\n".join([f'<a href="#{section_id}">{html.escape(label)}</a>' for section_id, label in nav_items])

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Delphi Midtraining Prediction Retrospective</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --ink: #172033;
      --muted: #5b667a;
      --line: #d8e0eb;
      --soft: #f6f8fb;
      --soft-2: #eef3f8;
      --blue: #1877F2;
      --red: #d62728;
      --orange: #f0701a;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: #ffffff;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    a {{ color: #0b63ce; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{
      background: #eef3f8;
      border-radius: 5px;
      padding: 0.08rem 0.28rem;
      font-size: 0.92em;
    }}
    .page {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 34px 28px 72px;
    }}
    .hero {{
      border-bottom: 1px solid var(--line);
      padding-bottom: 26px;
      margin-bottom: 22px;
    }}
    .eyebrow, .section-eyebrow {{
      color: var(--blue);
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    h1 {{
      font-size: clamp(2rem, 4vw, 4rem);
      line-height: 1.04;
      letter-spacing: 0;
      margin: 10px 0 14px;
      max-width: 980px;
    }}
    h2 {{
      font-size: 2rem;
      margin: 6px 0 12px;
      letter-spacing: 0;
    }}
    h3 {{
      font-size: 1.15rem;
      margin: 24px 0 10px;
      letter-spacing: 0;
    }}
    p {{ max-width: 960px; }}
    .lead {{
      font-size: 1.08rem;
      color: var(--muted);
      max-width: 980px;
    }}
    .nav {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      background: rgba(255, 255, 255, 0.94);
      backdrop-filter: blur(8px);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      margin: 18px 0 22px;
    }}
    .nav a {{
      display: inline-flex;
      align-items: center;
      min-height: 34px;
      padding: 6px 10px;
      border-radius: 6px;
      background: var(--soft);
      color: #213047;
      font-weight: 650;
      font-size: 0.92rem;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 22px 0 10px;
    }}
    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      background: var(--soft);
      min-height: 118px;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 0.82rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .metric-value {{
      margin-top: 6px;
      font-size: 1.65rem;
      font-weight: 750;
    }}
    .metric-note {{
      color: var(--muted);
      margin-top: 4px;
      font-size: 0.9rem;
    }}
    .glossary {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      padding: 18px;
      margin: 20px 0 28px;
    }}
    .report-section {{
      padding: 34px 0 38px;
      border-top: 1px solid var(--line);
    }}
    .plotly-graph-div {{
      width: 100% !important;
      margin: 18px 0 24px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 22px;
      font-size: 0.92rem;
    }}
    .data-table th {{
      text-align: left;
      background: var(--soft-2);
      color: #334155;
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      vertical-align: bottom;
    }}
    .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      vertical-align: top;
    }}
    .data-table tr:nth-child(even) td {{ background: #fbfcfe; }}
    .note {{
      color: var(--muted);
      font-size: 0.92rem;
    }}
    @media (max-width: 860px) {{
      .page {{ padding: 22px 14px 58px; }}
      .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      h1 {{ font-size: 2.3rem; }}
      h2 {{ font-size: 1.55rem; }}
    }}
    @media (max-width: 560px) {{
      .metric-grid {{ grid-template-columns: 1fr; }}
      .nav {{ position: static; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <div class="eyebrow">Delphi midtraining prediction retrospective</div>
      <h1>Why the old K=0.20 math-validation scaling law broke at 1e22</h1>
      <p class="lead">The old 4plus validation target combined a growing midtraining token budget with scale-dependent exposure to near-duplicate math documents. Fixed-token controls and actual-seen clean validation make the endpoint fits smooth again.</p>
      <div class="metric-grid">{cards}</div>
    </header>
    <nav class="nav" aria-label="Report sections">{nav}</nav>
    <section id="glossary" class="glossary">
      <div class="section-eyebrow">Glossary</div>
      <h2>Terms used in the report</h2>
      {glossary_html()}
    </section>
    {''.join(sections)}
    <p class="note">Generated by <code>scripts/analysis/build_delphi_midtraining_unified_story_report.py</code>. Plotly is loaded from the public CDN; the report data is embedded into this HTML at generation time.</p>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    paths = Paths(output=args.output, contamination_root=args.contamination_root)
    if not paths.contamination_root.exists():
        raise FileNotFoundError(f"Contamination worktree not found: {paths.contamination_root}")
    paths.output.parent.mkdir(parents=True, exist_ok=True)
    paths.output.write_text(render_report(paths), encoding="utf-8")
    print(paths.output)


if __name__ == "__main__":
    main()
