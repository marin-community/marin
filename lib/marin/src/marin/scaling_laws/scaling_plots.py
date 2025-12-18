# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualization functions for scaling ladder analysis.

This module provides plotting utilities for isoflop analysis results.
All plotly-related code is contained here to keep the core scaling_ladder
module free of visualization dependencies.
"""

import logging
import os

import fsspec
import jax.numpy as jnp
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


logger = logging.getLogger(__name__)

# ---------------- Theme ----------------
pio.templates.default = "plotly_white"

# ---------------- Visual Constants ----------------
PALETTE = [
    "#1877F2",
    "#F0701A",
    "#5A24C7",
    "#E42C97",
    "#00487C",
    "#0EAC96",
    "#AB76FF",
    "#B50550",
    "#0099E6",
    "#22085F",
    "#783301",
]

MARKERS = [
    "circle",
    "square",
    "cross",
    "x",
    "triangle-up",
    "triangle-down",
    "triangle-left",
    "triangle-right",
    "pentagon",
    "hexagon",
    "hexagon2",
    "star",
    "star-triangle-up",
    "star-triangle-down",
    "star-square",
    "star-diamond",
    "hourglass",
    "bowtie",
]

DASHES = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]

_MIN_MARKER = dict(symbol="diamond", size=10, color="#000000")
_SCALE_MARKER = dict(symbol="circle", size=9, color=PALETTE[0])
_SCALE_LINE = dict(dash="dot", width=2, color=PALETTE[0])


def create_isoflop_plot(
    df: pd.DataFrame,
    minima_records: list[dict],
    fit_curves: dict[tuple[str, float], tuple[float, float, float]],
) -> go.Figure:
    """Create the IsoFLOP plot showing loss vs tokens for each compute budget.

    Args:
        df: DataFrame with columns: tokens, loss, flops, params, name, label
        minima_records: List of dicts with optimal config info per (label, flops)
        fit_curves: Dict of {(label, flops): (a, b, c)} quadratic fit coefficients

    Returns:
        Plotly Figure with the isoflop visualization
    """
    if df.empty:
        return go.Figure()

    datasets = list(dict.fromkeys(df["label"].tolist()))

    buckets = sorted(df.flops.unique())
    bucket_color = {C: PALETTE[i % len(PALETTE)] for i, C in enumerate(buckets)}
    ds_marker = {lab: MARKERS[i % len(MARKERS)] for i, lab in enumerate(datasets)}

    fig = go.Figure()

    # Build lookup for minima
    minima_lookup = {(rec["label"], rec["flops"]): rec for rec in minima_records}

    for lab in datasets:
        for C in buckets:
            sub = df[(df.flops == C) & (df.label == lab)].sort_values("tokens")
            if sub.empty:
                continue

            # Scatter points
            fig.add_trace(
                go.Scatter(
                    x=sub.tokens,
                    y=sub.loss,
                    mode="markers",
                    marker=dict(symbol=ds_marker[lab], color=bucket_color[C], size=8),
                    name=f"{lab}, {C:.2e} FLOPs",
                    legendgroup=f"{lab}, {C:.2e}",
                    hovertemplate=(
                        "C=%{text:.2e} FLOPs<br>tokens=%{x:.3e}<br>"
                        "loss=%{y:.4f}<br>params=%{customdata:.3e}<extra></extra>"
                    ),
                    text=[C] * len(sub),
                    customdata=sub.params.values,
                )
            )

            # Draw fit curve if available
            key = (lab, C)
            if key in fit_curves:
                a, b, c = fit_curves[key]
                if a != 0:
                    Ls = jnp.linspace(jnp.log10(sub.tokens.min()), jnp.log10(sub.tokens.max()), 200)
                    fig.add_trace(
                        go.Scatter(
                            x=10**Ls,
                            y=a * Ls**2 + b * Ls + c,
                            mode="lines",
                            line=dict(color=bucket_color[C], dash="dash", width=2),
                            showlegend=False,
                            legendgroup=f"{lab}, {C:.2e}",
                        )
                    )

            # Draw minimum marker
            if key in minima_lookup:
                rec = minima_lookup[key]
                fig.add_trace(
                    go.Scatter(
                        x=[rec["optimal_tokens"]],
                        y=[rec["loss_at_optimal"]],
                        mode="markers",
                        marker=_MIN_MARKER,
                        showlegend=False,
                        legendgroup=f"{lab}, {C:.2e}",
                        hovertemplate=(
                            "<b>Compute-optimal</b><br>"
                            "C=%{text:.2e} FLOPs<br>tokens=%{x:.3e}<br>"
                            "loss=%{y:.4f}<br>params=%{customdata:.3e}<extra></extra>"
                        ),
                        text=[C],
                        customdata=[rec["optimal_params"]],
                    )
                )

    fig.update_layout(
        template="plotly_white",
        xaxis_type="log",
        xaxis_title="Tokens (log scale)",
        yaxis_title="Bits Per Byte Validation",
        title="Marin IsoFLOP Suite",
        width=1000,
        height=600,
    )

    return fig


def create_scaling_plot(
    minima_records: list[dict],
    scaling_fits: dict[str, tuple[float, float]],
) -> go.Figure:
    """Create the scaling law fit plot showing N* vs compute budget.

    Args:
        minima_records: List of dicts with optimal config info per (label, flops)
        scaling_fits: Dict of {label: (alpha, A)} for N* ~ A * C^alpha

    Returns:
        Plotly Figure with the scaling fit visualization
    """
    if not minima_records:
        return go.Figure()

    # Group by label
    by_lab = {}
    for rec in minima_records:
        by_lab.setdefault(rec["label"], []).append(rec)

    datasets = list(by_lab.keys())

    fig = go.Figure()

    for i, lab in enumerate(datasets):
        recs = by_lab.get(lab, [])
        if not recs:
            continue

        recs = sorted(recs, key=lambda r: r["flops"])
        Cs = jnp.array([r["flops"] for r in recs])
        Ns = jnp.array([r["optimal_tokens"] for r in recs])

        color = PALETTE[i % len(PALETTE)]
        dash = DASHES[i % len(DASHES)]

        # Plot minima points
        fig.add_trace(
            go.Scatter(
                x=list(map(float, Cs)),
                y=list(map(float, Ns)),
                mode="markers",
                marker=dict(symbol=_SCALE_MARKER["symbol"], size=_SCALE_MARKER["size"], color=color),
                name=f"{lab} minima",
                legendgroup=lab,
            )
        )

        # Plot fit line if available
        if lab in scaling_fits:
            alpha, A = scaling_fits[lab]
            Cmin, Cmax = float(Cs.min()), float(Cs.max())
            C_fit = jnp.logspace(jnp.log10(Cmin) - 0.1, jnp.log10(Cmax) + 0.1, 400)
            N_fit = A * (C_fit**alpha)

            fig.add_trace(
                go.Scatter(
                    x=list(map(float, C_fit)),
                    y=list(map(float, N_fit)),
                    mode="lines",
                    line=dict(color=color, dash=dash, width=_SCALE_LINE["width"]),
                    name=f"{lab} fit (α={alpha:.3f})",
                    legendgroup=lab,
                )
            )

    fig.update_layout(
        template="plotly_white",
        xaxis_type="log",
        yaxis_type="log",
        xaxis_title="Compute budget C (FLOPs, log)",
        yaxis_title="Optimal tokens N* (log)",
        title="Scaling fits per dataset",
    )

    return fig


def save_plots(
    fig_isoflop: go.Figure,
    fig_scaling: go.Figure,
    output_path: str,
) -> None:
    """Save isoflop and scaling plots to HTML files.

    Args:
        fig_isoflop: IsoFLOP plot figure
        fig_scaling: Scaling fit plot figure
        output_path: Directory path to save plots
    """
    fs, _, _ = fsspec.get_fs_token_paths(output_path)
    fs.makedirs(output_path, exist_ok=True)

    iso_path = os.path.join(output_path, "isoflop_plot.html")
    scaling_path = os.path.join(output_path, "scaling_plot.html")

    with fs.open(iso_path, "w") as f:
        f.write(fig_isoflop.to_html())
    logger.info(f"Wrote isoflop plot to {iso_path}")

    with fs.open(scaling_path, "w") as f:
        f.write(fig_scaling.to_html())
    logger.info(f"Wrote scaling plot to {scaling_path}")


def upload_plots_to_wandb(
    fig_isoflop: go.Figure,
    fig_scaling: go.Figure,
    entity: str = "marin-community",
    project: str = "marin-analysis",
    run_name: str = "scaling-ladder-analysis",
) -> None:
    """Upload plots to Weights & Biases.

    Args:
        fig_isoflop: IsoFLOP plot figure
        fig_scaling: Scaling fit plot figure
        entity: WandB entity
        project: WandB project
        run_name: Name for the WandB run
    """
    if not WANDB_AVAILABLE:
        logger.warning("wandb not available, cannot upload plots")
        return

    wandb.login()
    run = wandb.init(
        entity=entity,
        project=project,
        job_type="scaling-ladder",
        name=run_name,
        resume="allow",
    )
    wandb.log(
        {
            "isoFLOP_plot": wandb.Plotly(fig_isoflop),
            "scaling_plot": wandb.Plotly(fig_scaling),
        }
    )
    run.finish()
    logger.info("Uploaded plots to WandB")
