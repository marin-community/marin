# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Analysis: Exp87 scaling ladder for repeat downweighting.

Fetches W&B metrics and plots LL(functional) - LL(non-functional) vs FLOPs
for the two weighting strategies, with power-law curve fits.

Usage:
    uv run experiments/dna/exp87_scaling_ladder_repeat_downweighting_analysis.py
"""

import numpy as np
import plotly.graph_objects as go
import wandb
from scipy.optimize import curve_fit

WANDB_ENTITY = "gonzalobenegas"
WANDB_PROJECT = "marin"
WANDB_GROUP = "exp87-scaling-ladder-repeat-downweighting"

FUNCTIONAL_LOSS_KEY = "eval/val_functional/loss"
NONFUNCTIONAL_LOSS_KEY = "eval/val_nonfunctional/loss"


def fetch_exp87_metrics(
    entity: str = WANDB_ENTITY,
    project: str = WANDB_PROJECT,
) -> list[dict]:
    """Fetch final validation losses for all exp87 runs from W&B."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"group": WANDB_GROUP})

    records = []
    for run in runs:
        func_loss = run.summary.get(FUNCTIONAL_LOSS_KEY)
        nonfunc_loss = run.summary.get(NONFUNCTIONAL_LOSS_KEY)
        if func_loss is None or nonfunc_loss is None:
            continue

        # Extract flops and weight name from tags (e.g. "flops=1e+17", "rw0.01")
        flops = None
        weight_name = None
        for tag in run.tags:
            if tag.startswith("flops="):
                flops = float(tag.split("=")[1])
            elif tag.startswith("rw"):
                weight_name = tag

        if flops is None or weight_name is None:
            continue

        records.append(
            {
                "run_name": run.name,
                "weight_name": weight_name,
                "flops": flops,
                "functional_loss": float(func_loss),
                "nonfunctional_loss": float(nonfunc_loss),
                "loss_gap": float(nonfunc_loss) - float(func_loss),
            }
        )

    return records


def power_law(x, a, b):
    return a * np.power(x, b)


def plot_loss_gap_vs_flops(records: list[dict], output_path: str = "exp87_loss_gap_vs_flops.html"):
    """Plot LL(functional) - LL(non-functional) vs FLOPs with power-law fits."""
    fig = go.Figure()

    colors = {"rw0.01": "#636EFA", "rw1.0": "#EF553B"}
    labels = {"rw0.01": "Downweighted (0.01)", "rw1.0": "Uniform (1.0)"}

    for wn in ["rw0.01", "rw1.0"]:
        subset = sorted([r for r in records if r["weight_name"] == wn], key=lambda r: r["flops"])
        if not subset:
            continue

        flops = np.array([r["flops"] for r in subset])
        gaps = np.array([r["loss_gap"] for r in subset])

        fig.add_trace(
            go.Scatter(
                x=flops,
                y=gaps,
                mode="markers",
                name=labels[wn],
                marker=dict(size=10, color=colors[wn]),
            )
        )

        # Power-law fit: gap = a * FLOPs^b
        if len(flops) >= 2:
            try:
                popt, _ = curve_fit(power_law, flops, gaps, p0=[1.0, -0.1], maxfev=10000)
                flops_fit = np.logspace(
                    np.log10(flops.min() * 0.8),
                    np.log10(flops.max() * 1.2),
                    100,
                )
                fig.add_trace(
                    go.Scatter(
                        x=flops_fit,
                        y=power_law(flops_fit, *popt),
                        mode="lines",
                        name=f"{labels[wn]} fit: {popt[0]:.2e} · F^{popt[1]:.3f}",
                        line=dict(color=colors[wn], dash="dash"),
                    )
                )
            except RuntimeError:
                pass

    fig.update_layout(
        title="Exp87: LL(functional) - LL(non-functional) vs FLOPs",
        xaxis_title="FLOPs",
        yaxis_title="LL(functional) - LL(non-functional)",
        xaxis_type="log",
        template="plotly_white",
        width=600,
        height=400,
    )

    fig.write_html(output_path)
    fig.show()


if __name__ == "__main__":
    records = fetch_exp87_metrics()
    plot_loss_gap_vs_flops(records)
