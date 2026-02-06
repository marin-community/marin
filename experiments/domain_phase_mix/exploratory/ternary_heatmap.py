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

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "plotly",
#     "kaleido",
#     "pandas",
#     "numpy",
# ]
# ///
"""Ternary heatmap visualization of metrics across mixture weight space.

Creates ternary plots showing how metrics (e.g., choice_logprob) vary across
the simplex of mixture weights for each training phase.

Usage:
    uv run experiments/domain_phase_mix/exploratory/ternary_heatmap.py
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Domain labels for display
DOMAIN_LABELS = {
    "nemotron_full": "Nemotron",
    "dolmino": "Dolmino",
    "openthoughts_sft": "SFT",
}


def load_experiment_data(csv_path: str) -> pd.DataFrame:
    """Load experiment data from CSV file.

    Args:
        csv_path: Path to the CSV file with experiment results.

    Returns:
        DataFrame with experiment data, excluding diverged runs.
    """
    df = pd.read_csv(csv_path)

    # Exclude diverged runs (BPB > 2.0 indicates divergence)
    df = df[df["eval/paloma/c4_en/bpb"] < 2.0].copy()

    return df


def create_ternary_heatmap(
    df: pd.DataFrame,
    metric: str,
    title: str = "Metric Heatmap on Mixture Simplex",
    colorscale: str = "Viridis",
    reverse_colorscale: bool = False,
) -> go.Figure:
    """Create a ternary heatmap showing metric values across mixture weights.

    Creates 3 subplots (one per phase) showing how the metric varies
    across the simplex of mixture weights.

    Args:
        df: DataFrame with experiment data.
        metric: Column name of the metric to visualize.
        title: Figure title.
        colorscale: Plotly colorscale name.
        reverse_colorscale: Whether to reverse the colorscale.

    Returns:
        Plotly Figure object.
    """
    phases = ["phase_0", "phase_1", "phase_2"]
    phase_labels = ["Phase 1", "Phase 2", "Phase 3"]
    domains = ["nemotron_full", "dolmino", "openthoughts_sft"]

    # Create subplots with ternary axes
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=phase_labels,
        specs=[[{"type": "ternary"}, {"type": "ternary"}, {"type": "ternary"}]],
        horizontal_spacing=0.05,
    )

    # Get metric values for color scaling
    metric_values = df[metric].values
    vmin, vmax = metric_values.min(), metric_values.max()

    # Determine if higher is better (for choice_logprob, higher/less negative is better)
    if "logprob" in metric.lower() or "acc" in metric.lower():
        # Higher is better - use reversed colorscale so red=bad, blue=good
        colorscale_to_use = colorscale + "_r" if not reverse_colorscale else colorscale
    else:
        # Lower is better (e.g., BPB) - normal colorscale
        colorscale_to_use = colorscale if not reverse_colorscale else colorscale + "_r"

    for col_idx, phase in enumerate(phases, 1):
        # Extract weights for this phase (convert to percentages for ternary)
        a = df[f"{phase}_{domains[0]}"].values * 100  # nemotron
        b = df[f"{phase}_{domains[1]}"].values * 100  # dolmino
        c = df[f"{phase}_{domains[2]}"].values * 100  # sft

        # Create hover text with details
        hover_text = [
            f"Run {int(row['run_id'])}<br>"
            f"{DOMAIN_LABELS[domains[0]]}: {row[f'{phase}_{domains[0]}']:.1%}<br>"
            f"{DOMAIN_LABELS[domains[1]]}: {row[f'{phase}_{domains[1]}']:.1%}<br>"
            f"{DOMAIN_LABELS[domains[2]]}: {row[f'{phase}_{domains[2]}']:.1%}<br>"
            f"{metric}: {row[metric]:.4f}"
            for _, row in df.iterrows()
        ]

        fig.add_trace(
            go.Scatterternary(
                a=a,
                b=b,
                c=c,
                mode="markers",
                marker=dict(
                    size=12,
                    color=metric_values,
                    colorscale=colorscale_to_use,
                    cmin=vmin,
                    cmax=vmax,
                    showscale=(col_idx == 3),  # Only show colorbar on last subplot
                    colorbar=dict(
                        title=dict(text=metric.split("/")[-1], font=dict(size=12)),
                        x=1.02,
                        len=0.8,
                    ),
                    line=dict(width=1, color="white"),
                ),
                text=hover_text,
                hoverinfo="text",
                showlegend=False,
                cliponaxis=False,  # Prevent markers from being clipped at edges
            ),
            row=1,
            col=col_idx,
        )

    # Update ternary axes for each subplot
    for col_idx in range(1, 4):
        ternary_key = f"ternary{col_idx}" if col_idx > 1 else "ternary"
        fig.update_layout(
            **{
                ternary_key: dict(
                    aaxis=dict(
                        title=dict(text=DOMAIN_LABELS[domains[0]], font=dict(size=11)),
                        linewidth=0,
                        showline=False,
                        tickfont=dict(size=9),
                    ),
                    baxis=dict(
                        title=dict(text=DOMAIN_LABELS[domains[1]], font=dict(size=11)),
                        linewidth=0,
                        showline=False,
                        tickfont=dict(size=9),
                    ),
                    caxis=dict(
                        title=dict(text=DOMAIN_LABELS[domains[2]], font=dict(size=11)),
                        linewidth=0,
                        showline=False,
                        tickfont=dict(size=9),
                    ),
                    bgcolor="rgba(250,250,250,1)",
                    sum=100,
                )
            }
        )

    # Update overall layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, family="Arial"),
            x=0.5,
            xanchor="center",
        ),
        height=550,
        width=1500,
        margin=dict(l=50, r=80, t=80, b=30),
        paper_bgcolor="white",
        font=dict(family="Arial"),
    )

    # Update subplot titles
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=14, family="Arial")

    return fig


def create_multi_metric_heatmap(
    df: pd.DataFrame,
    metrics: list[str],
    title: str = "Metric Heatmaps on Mixture Simplex",
) -> go.Figure:
    """Create a grid of ternary heatmaps for multiple metrics.

    Creates a grid with metrics as rows and phases as columns.

    Args:
        df: DataFrame with experiment data.
        metrics: List of metric column names to visualize.
        title: Figure title.

    Returns:
        Plotly Figure object.
    """
    phases = ["phase_0", "phase_1", "phase_2"]
    phase_labels = ["Phase 1", "Phase 2", "Phase 3"]
    domains = ["nemotron_full", "dolmino", "openthoughts_sft"]

    n_metrics = len(metrics)
    n_phases = len(phases)

    # Create subplot titles
    subplot_titles = []
    for metric in metrics:
        metric_name = metric.split("/")[-1]
        for phase_label in phase_labels:
            subplot_titles.append(f"{metric_name} - {phase_label}")

    # Create specs for ternary subplots
    specs = [[{"type": "ternary"} for _ in range(n_phases)] for _ in range(n_metrics)]

    fig = make_subplots(
        rows=n_metrics,
        cols=n_phases,
        subplot_titles=subplot_titles,
        specs=specs,
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    for row_idx, metric in enumerate(metrics, 1):
        metric_values = df[metric].values
        vmin, vmax = metric_values.min(), metric_values.max()

        # Determine colorscale direction
        if "logprob" in metric.lower() or "acc" in metric.lower():
            colorscale = "RdYlBu"  # Red=bad, Blue=good (higher is better)
        else:
            colorscale = "RdYlBu_r"  # Blue=good, Red=bad (lower is better)

        for col_idx, phase in enumerate(phases, 1):
            a = df[f"{phase}_{domains[0]}"].values * 100
            b = df[f"{phase}_{domains[1]}"].values * 100
            c = df[f"{phase}_{domains[2]}"].values * 100

            hover_text = [
                f"Run {int(row['run_id'])}<br>"
                f"{DOMAIN_LABELS[domains[0]]}: {row[f'{phase}_{domains[0]}']:.1%}<br>"
                f"{DOMAIN_LABELS[domains[1]]}: {row[f'{phase}_{domains[1]}']:.1%}<br>"
                f"{DOMAIN_LABELS[domains[2]]}: {row[f'{phase}_{domains[2]}']:.1%}<br>"
                f"{metric}: {row[metric]:.4f}"
                for _, row in df.iterrows()
            ]

            # Show colorbar only on rightmost column
            show_colorbar = col_idx == n_phases

            fig.add_trace(
                go.Scatterternary(
                    a=a,
                    b=b,
                    c=c,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=metric_values,
                        colorscale=colorscale,
                        cmin=vmin,
                        cmax=vmax,
                        showscale=show_colorbar,
                        colorbar=(
                            dict(
                                title=dict(
                                    text=metric.split("/")[-1],
                                    font=dict(size=10),
                                ),
                                x=1.02,
                                y=1 - (row_idx - 0.5) / n_metrics,
                                len=0.8 / n_metrics,
                                thickness=15,
                            )
                            if show_colorbar
                            else None
                        ),
                        line=dict(width=0.5, color="white"),
                    ),
                    text=hover_text,
                    hoverinfo="text",
                    showlegend=False,
                    cliponaxis=False,  # Prevent markers from being clipped at edges
                ),
                row=row_idx,
                col=col_idx,
            )

    # Update ternary axes for each subplot
    for row_idx in range(1, n_metrics + 1):
        for col_idx in range(1, n_phases + 1):
            subplot_num = (row_idx - 1) * n_phases + col_idx
            ternary_key = f"ternary{subplot_num}" if subplot_num > 1 else "ternary"

            fig.update_layout(
                **{
                    ternary_key: dict(
                        aaxis=dict(
                            title=dict(text=DOMAIN_LABELS[domains[0]], font=dict(size=9)),
                            linewidth=0,
                            showline=False,
                            tickfont=dict(size=8),
                        ),
                        baxis=dict(
                            title=dict(text=DOMAIN_LABELS[domains[1]], font=dict(size=9)),
                            linewidth=0,
                            showline=False,
                            tickfont=dict(size=8),
                        ),
                        caxis=dict(
                            title=dict(text=DOMAIN_LABELS[domains[2]], font=dict(size=9)),
                            linewidth=0,
                            showline=False,
                            tickfont=dict(size=8),
                        ),
                        bgcolor="rgba(250,250,250,1)",
                        sum=100,
                    )
                }
            )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, family="Arial"),
            x=0.5,
            xanchor="center",
        ),
        height=380 * n_metrics,
        width=1400,
        margin=dict(l=30, r=100, t=80, b=30),
        paper_bgcolor="white",
        font=dict(family="Arial"),
    )

    # Update subplot titles
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=11, family="Arial")

    return fig


def main():
    from pathlib import Path

    output_dir = Path(__file__).parent
    csv_path = output_dir / "3_partitions_3_phases_6.csv"

    # Load data
    df = load_experiment_data(str(csv_path))
    print(f"Loaded {len(df)} runs from {csv_path}")

    # Create single metric heatmap for choice_logprob
    fig_logprob = create_ternary_heatmap(
        df,
        metric="lm_eval/arc_challenge/choice_logprob",
        title="Arc Challenge choice_logprob by Per-Phase Mixture Weights",
        colorscale="RdYlBu",
    )

    html_path = output_dir / "ternary_choice_logprob.html"
    fig_logprob.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"Saved: {html_path}")

    png_path = output_dir / "ternary_choice_logprob.png"
    fig_logprob.write_image(str(png_path), scale=2)
    print(f"Saved: {png_path}")

    # Create multi-metric heatmap
    metrics = [
        "eval/paloma/c4_en/bpb",
        "lm_eval/arc_challenge/acc",
        "lm_eval/arc_challenge/choice_logprob",
    ]

    fig_multi = create_multi_metric_heatmap(
        df,
        metrics=metrics,
        title="Key Metrics by Mixture Weights (per Phase)",
    )

    html_path_multi = output_dir / "ternary_multi_metric.html"
    fig_multi.write_html(str(html_path_multi), include_plotlyjs="cdn")
    print(f"Saved: {html_path_multi}")

    png_path_multi = output_dir / "ternary_multi_metric.png"
    fig_multi.write_image(str(png_path_multi), scale=2)
    print(f"Saved: {png_path_multi}")


if __name__ == "__main__":
    main()
