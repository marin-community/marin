# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
# ]
# ///
"""Render diagnostics for the frozen phase-blind RPL tied-spine audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "reference_outputs" / "phase_blind_rpl_tied_spine_20260731"
OUTPUT_NAME = "tied_spine_diagnostics.html"
TARGETS = ("uncheatable", "table9")
TARGET_LABELS = {
    "uncheatable": "Uncheatable",
    "table9": "OLMoBaseEval Table-9",
}
COLORS = {
    "observed": "#16384a",
    "predicted": "#dc5a34",
    "bootstrap": "#e2b437",
    "fold": "#2c8b74",
    "grid": "#d8d1c3",
    "paper": "#f5f1e8",
}


def plotly_fragment(figure: go.Figure, include_plotlyjs: bool) -> str:
    return pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {"format": "png", "scale": 4},
        },
    )


def calibration_figure(input_dir: Path) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[TARGET_LABELS[target] for target in TARGETS],
        horizontal_spacing=0.12,
    )
    for column, target in enumerate(TARGETS, start=1):
        predictions = pd.read_csv(input_dir / "cells" / target / "predictions.csv")
        low = float(min(predictions["observed"].min(), predictions["predicted"].min()))
        high = float(max(predictions["observed"].max(), predictions["predicted"].max()))
        padding = 0.04 * (high - low)
        figure.add_trace(
            go.Scatter(
                x=predictions["observed"],
                y=predictions["predicted"],
                mode="markers",
                marker={
                    "color": predictions["residual"],
                    "colorscale": "RdYlGn_r",
                    "cmin": -float(np.max(np.abs(predictions["residual"]))),
                    "cmax": float(np.max(np.abs(predictions["residual"]))),
                    "line": {"color": COLORS["observed"], "width": 0.7},
                    "size": 8,
                    "showscale": column == 2,
                    "colorbar": {
                        "title": "predicted -<br>observed",
                        "thickness": 14,
                    },
                },
                customdata=np.column_stack(
                    [
                        predictions["run_name"],
                        predictions["policy_family"],
                        predictions["outer_fold"],
                        predictions["residual"],
                    ]
                ),
                hovertemplate=(
                    "%{customdata[0]}<br>family=%{customdata[1]}"
                    "<br>fold=%{customdata[2]}<br>observed=%{x:.6f}"
                    "<br>predicted=%{y:.6f}<br>residual=%{customdata[3]:+.6f}"
                    "<extra></extra>"
                ),
                name=TARGET_LABELS[target],
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=[low - padding, high + padding],
                y=[low - padding, high + padding],
                mode="lines",
                line={"color": "#7c8990", "dash": "dash", "width": 1.5},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(
            title_text="Observed BPB",
            range=[low - padding, high + padding],
            row=1,
            col=column,
        )
        figure.update_yaxes(
            title_text="OOF predicted BPB",
            range=[low - padding, high + padding],
            row=1,
            col=column,
        )
    figure.update_layout(
        title={"text": "Physically tied OOF calibration", "x": 0.02},
        height=560,
        margin={"l": 70, "r": 80, "t": 85, "b": 65},
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["paper"],
        font={"family": "Avenir Next, sans-serif", "color": COLORS["observed"]},
    )
    figure.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    figure.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return figure


def optimum_figure(input_dir: Path, target: str) -> go.Figure:
    policy = pd.read_csv(input_dir / "cells" / target / "raw_optimum_policy.csv")
    policy["absolute_difference"] = np.abs(policy["weight"] - policy["observed_best_weight"])
    policy = policy.sort_values("absolute_difference", ascending=True)
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            y=policy["domain"],
            x=policy["observed_best_weight"],
            orientation="h",
            name="Best observed tied",
            marker_color=COLORS["observed"],
            opacity=0.8,
            customdata=policy[["materialized_epochs"]],
            hovertemplate=(
                "%{y}<br>weight=%{x:.5f}<br>RPL-optimum epochs=%{customdata[0]:.2f}" "<extra>best observed tied</extra>"
            ),
        )
    )
    figure.add_trace(
        go.Bar(
            y=policy["domain"],
            x=policy["weight"],
            orientation="h",
            name="Raw RPL optimum",
            marker_color=COLORS["predicted"],
            opacity=0.78,
            customdata=policy[["materialized_epochs"]],
            hovertemplate=("%{y}<br>weight=%{x:.5f}<br>epochs=%{customdata[0]:.2f}" "<extra>raw RPL optimum</extra>"),
        )
    )
    figure.update_layout(
        title={
            "text": f"{TARGET_LABELS[target]} raw optimum versus observed tied frontier",
            "x": 0.02,
        },
        barmode="overlay",
        height=980,
        margin={"l": 230, "r": 35, "t": 85, "b": 55},
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["paper"],
        font={"family": "Avenir Next, sans-serif", "color": COLORS["observed"]},
        xaxis_title="Tied mixture weight",
        legend={"orientation": "h", "y": 1.03, "x": 0.0},
    )
    figure.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    return figure


def stability_figure(input_dir: Path) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[TARGET_LABELS[target] for target in TARGETS],
        horizontal_spacing=0.12,
    )
    for column, target in enumerate(TARGETS, start=1):
        cell = input_dir / "cells" / target
        fold = pd.read_csv(cell / "fold_optima.csv")
        bootstrap = pd.read_csv(cell / "bootstrap_optima.csv")
        for label, frame, color in (
            ("Nested outer-fold refits", fold, COLORS["fold"]),
            ("Conditional group bootstraps", bootstrap, COLORS["bootstrap"]),
        ):
            figure.add_trace(
                go.Box(
                    y=frame["l1_to_full_optimum"],
                    name=label,
                    boxpoints="all",
                    jitter=0.25,
                    pointpos=0,
                    marker={"color": color, "size": 7},
                    line={"color": color},
                    showlegend=column == 1,
                    hovertemplate="L1 to full optimum=%{y:.4f}<extra>%{fullData.name}</extra>",
                ),
                row=1,
                col=column,
            )
        figure.update_yaxes(
            title_text="L1 distance to full-data raw optimum",
            rangemode="tozero",
            row=1,
            col=column,
        )
    figure.update_layout(
        title={"text": "Raw-optimum instability", "x": 0.02},
        boxmode="group",
        height=560,
        margin={"l": 70, "r": 35, "t": 85, "b": 65},
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["paper"],
        font={"family": "Avenir Next, sans-serif", "color": COLORS["observed"]},
        legend={"orientation": "h", "y": 1.04, "x": 0.4},
    )
    figure.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return figure


def metric_cards(input_dir: Path) -> str:
    metrics = pd.read_csv(input_dir / "metrics.csv").set_index("target")
    optimum = pd.read_csv(input_dir / "optimum_diagnostics.csv").set_index("target")
    cards = []
    for target in TARGETS:
        row = metrics.loc[target]
        opt = optimum.loc[target]
        cards.append(
            f"""
            <article class="card">
              <p class="eyebrow">{TARGET_LABELS[target]}</p>
              <h2>{row['rmse']:.6f} OOF RMSE</h2>
              <p><strong>{100.0 * row['relative_rmse_to_reference']:+.1f}%</strong> versus the frozen tied reference.</p>
              <dl>
                <div><dt>Calibration slope</dt><dd>{row['observed_on_predicted_slope']:.3f}</dd></div>
                <div><dt>Raw predicted BPB</dt><dd>{opt['predicted_bpb']:.6f}</dd></div>
                <div><dt>Observed tied frontier</dt><dd>{opt['observed_best_tied_bpb']:.6f}</dd></div>
                <div><dt>Nearest support TV</dt><dd>{opt['nearest_policy_tv']:.3f}</dd></div>
                <div><dt>Near-zero buckets</dt><dd>{int(opt['near_zero_bucket_count'])}</dd></div>
              </dl>
            </article>
            """
        )
    return "\n".join(cards)


def main() -> None:
    input_dir = DEFAULT_INPUT_DIR
    protocol = json.loads((input_dir / "protocol.json").read_text())
    figures = [
        calibration_figure(input_dir),
        optimum_figure(input_dir, "uncheatable"),
        optimum_figure(input_dir, "table9"),
        stability_figure(input_dir),
    ]
    fragments = [plotly_fragment(figure, include_plotlyjs=index == 0) for index, figure in enumerate(figures)]
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase-blind RPL tied-spine audit</title>
  <style>
    :root {{ --ink: {COLORS['observed']}; --paper: {COLORS['paper']}; --accent: {COLORS['predicted']}; --line: #cbc2b2; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--paper); color: var(--ink); font-family: "Avenir Next", sans-serif; }}
    header {{ padding: 58px max(5vw, 28px) 38px; background: var(--ink); color: var(--paper); border-bottom: 8px solid var(--accent); }}
    header p {{ max-width: 900px; font-size: 18px; line-height: 1.55; color: #d8e1df; }}
    h1 {{ margin: 0; max-width: 980px; font-family: Georgia, serif; font-size: clamp(36px, 5vw, 68px); line-height: 1.02; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 30px 24px 80px; }}
    .cards {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; margin: 0 0 30px; }}
    .card {{ border: 1px solid var(--line); background: #fffdf8; padding: 24px; box-shadow: 8px 8px 0 #ded6c8; }}
    .card h2 {{ margin: 4px 0 8px; font-family: Georgia, serif; font-size: 30px; }}
    .eyebrow {{ margin: 0; color: var(--accent); font-size: 12px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }}
    dl {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px 22px; }}
    dl div {{ border-top: 1px solid var(--line); padding-top: 8px; }}
    dt {{ color: #60737b; font-size: 12px; text-transform: uppercase; }}
    dd {{ margin: 3px 0 0; font-family: Georgia, serif; font-size: 21px; }}
    section {{ margin-top: 24px; border: 1px solid var(--line); background: #fffdf8; }}
    footer {{ color: #63777d; font-size: 13px; margin-top: 24px; }}
    @media (max-width: 760px) {{ .cards {{ grid-template-columns: 1fr; }} dl {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <p class="eyebrow">Aggregate-spine falsification</p>
    <h1>Phase-blind RPL does not survive the tied-policy gate.</h1>
    <p>The model was fit only to 282 physically tied policies. It misses the preregistered OOF tolerance on both targets, then optimizes into sparse policies far outside observed support. Epoch overload prevents a one-bucket dose singularity but does not constrain compositional extrapolation.</p>
  </header>
  <main>
    <div class="cards">{metric_cards(input_dir)}</div>
    <section>{fragments[0]}</section>
    <section>{fragments[1]}</section>
    <section>{fragments[2]}</section>
    <section>{fragments[3]}</section>
    <footer>Protocol <code>{protocol['protocol_hash']}</code>. No asymmetric endpoint, deployment regularizer, trust region, or output calibration entered this audit.</footer>
  </main>
</body>
</html>
"""
    output = input_dir / OUTPUT_NAME
    output.write_text(html, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
