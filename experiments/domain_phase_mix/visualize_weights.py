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

"""Visualization script for mixture weight distributions.

Generates plotly visualizations of mixture weight distributions across phases
from a weight_configs.json file (either GCS or local path).

Usage:
    python -m experiments.domain_phase_mix.visualize_weights \
        gs://marin-us-central1/path/to/weight_configs.json \
        --output weights_viz.png

    # Or with local file:
    python -m experiments.domain_phase_mix.visualize_weights \
        /path/to/weight_configs.json \
        --output weights_viz.png
"""

import argparse
import json

import fsspec
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_weight_configs(path: str) -> dict:
    """Load weight configurations from a JSON file (GCS or local).

    Args:
        path: Path to the weight_configs.json file (supports gs:// URLs).

    Returns:
        Dictionary containing experiment metadata and configs.
    """
    with fsspec.open(path) as f:
        return json.load(f)


def extract_weights_by_phase(
    configs: list[dict],
    phases: list[str],
    domains: list[str],
) -> dict[str, dict[str, list[float]]]:
    """Extract weight arrays organized by phase and domain.

    Args:
        configs: List of weight configuration dictionaries.
        phases: List of phase names.
        domains: List of domain names.

    Returns:
        Nested dict: {phase_name: {domain_name: [weights...]}}
    """
    result: dict[str, dict[str, list[float]]] = {phase: {domain: [] for domain in domains} for phase in phases}

    for config in configs:
        phase_weights = config.get("phase_weights", {})
        for phase in phases:
            weights = phase_weights.get(phase, {})
            for domain in domains:
                result[phase][domain].append(weights.get(domain, 0.0))

    return result


def create_weight_distribution_figure(
    weights_by_phase: dict[str, dict[str, list[float]]],
    phases: list[str],
    domains: list[str],
    experiment_name: str,
) -> go.Figure:
    """Create a plotly figure showing weight distributions per phase.

    Creates a grid of subplots with:
    - Rows: phases
    - Each row shows histograms for all domains overlaid

    Args:
        weights_by_phase: Nested dict from extract_weights_by_phase.
        phases: List of phase names.
        domains: List of domain names.
        experiment_name: Experiment name for the title.

    Returns:
        Plotly Figure object.
    """
    n_phases = len(phases)

    # Create subplots: one row per phase
    fig = make_subplots(
        rows=n_phases,
        cols=1,
        subplot_titles=[f"{phase}" for phase in phases],
        vertical_spacing=0.12,
    )

    # Color palette for domains
    colors = [
        "rgba(31, 119, 180, 0.7)",  # Blue
        "rgba(255, 127, 14, 0.7)",  # Orange
        "rgba(44, 160, 44, 0.7)",  # Green
        "rgba(214, 39, 40, 0.7)",  # Red
        "rgba(148, 103, 189, 0.7)",  # Purple
        "rgba(140, 86, 75, 0.7)",  # Brown
    ]

    # Add histograms for each phase and domain
    for row_idx, phase in enumerate(phases):
        for domain_idx, domain in enumerate(domains):
            weights = weights_by_phase[phase][domain]
            color = colors[domain_idx % len(colors)]

            fig.add_trace(
                go.Histogram(
                    x=weights,
                    name=domain,
                    xbins=dict(start=0, end=1, size=0.05),
                    marker_color=color,
                    opacity=0.7,
                    showlegend=(row_idx == 0),  # Only show legend once
                    legendgroup=domain,
                ),
                row=row_idx + 1,
                col=1,
            )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Mixture Weight Distributions: {experiment_name}",
            font=dict(size=16),
        ),
        barmode="overlay",
        height=300 * n_phases,
        width=900,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        font=dict(size=12),
    )

    # Update x-axes
    for row_idx in range(n_phases):
        fig.update_xaxes(
            title_text="Weight",
            range=[0, 1],
            row=row_idx + 1,
            col=1,
        )
        fig.update_yaxes(
            title_text="Count",
            row=row_idx + 1,
            col=1,
        )

    return fig


def create_box_plot_figure(
    weights_by_phase: dict[str, dict[str, list[float]]],
    phases: list[str],
    domains: list[str],
    experiment_name: str,
) -> go.Figure:
    """Create a box plot showing weight distributions across phases.

    Creates grouped box plots with phases on x-axis and domains as colors.

    Args:
        weights_by_phase: Nested dict from extract_weights_by_phase.
        phases: List of phase names.
        domains: List of domain names.
        experiment_name: Experiment name for the title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    # Color palette for domains
    colors = [
        "rgb(31, 119, 180)",  # Blue
        "rgb(255, 127, 14)",  # Orange
        "rgb(44, 160, 44)",  # Green
        "rgb(214, 39, 40)",  # Red
        "rgb(148, 103, 189)",  # Purple
        "rgb(140, 86, 75)",  # Brown
    ]

    for domain_idx, domain in enumerate(domains):
        color = colors[domain_idx % len(colors)]

        for phase in phases:
            weights = weights_by_phase[phase][domain]
            fig.add_trace(
                go.Box(
                    y=weights,
                    name=domain,
                    x=[phase] * len(weights),
                    marker_color=color,
                    legendgroup=domain,
                    showlegend=(phase == phases[0]),
                    boxpoints="outliers",
                )
            )

    fig.update_layout(
        title=dict(
            text=f"Weight Distribution by Phase: {experiment_name}",
            font=dict(size=16),
        ),
        boxmode="group",
        height=500,
        width=900,
        yaxis=dict(title="Weight", range=[0, 1.05]),
        xaxis=dict(title="Phase"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        font=dict(size=12),
    )

    return fig


def create_ternary_plot_figure(
    weights_by_phase: dict[str, dict[str, list[float]]],
    phases: list[str],
    domains: list[str],
    experiment_name: str,
) -> go.Figure:
    """Create a ternary plot showing weight distributions (for 3 domains only).

    Each point represents one run's weights for a single phase.
    Different phases are shown with different colors/shapes.

    Args:
        weights_by_phase: Nested dict from extract_weights_by_phase.
        phases: List of phase names.
        domains: List of domain names (must be exactly 3).
        experiment_name: Experiment name for the title.

    Returns:
        Plotly Figure object.
    """
    if len(domains) != 3:
        raise ValueError(f"Ternary plot requires exactly 3 domains, got {len(domains)}")

    fig = go.Figure()

    # Colors for phases
    phase_colors = [
        "rgb(31, 119, 180)",  # Blue
        "rgb(255, 127, 14)",  # Orange
        "rgb(44, 160, 44)",  # Green
    ]

    for phase_idx, phase in enumerate(phases):
        # Convert to percentage (0-100) for proper ternary display
        a = [w * 100 for w in weights_by_phase[phase][domains[0]]]
        b = [w * 100 for w in weights_by_phase[phase][domains[1]]]
        c = [w * 100 for w in weights_by_phase[phase][domains[2]]]

        fig.add_trace(
            go.Scatterternary(
                a=a,
                b=b,
                c=c,
                mode="markers",
                name=phase,
                marker=dict(
                    size=8,
                    color=phase_colors[phase_idx % len(phase_colors)],
                    opacity=0.7,
                    line=dict(width=1, color="white"),
                ),
                cliponaxis=False,  # Prevent markers from being clipped at edges
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Ternary Weight Distribution: {experiment_name}",
            font=dict(size=16),
        ),
        ternary=dict(
            # Use sum=100 for percentage display (0-100%)
            aaxis=dict(
                title=dict(text=domains[0]),
                linewidth=0,  # Remove axis line
                showline=False,
            ),
            baxis=dict(
                title=dict(text=domains[1]),
                linewidth=0,
                showline=False,
            ),
            caxis=dict(
                title=dict(text=domains[2]),
                linewidth=0,
                showline=False,
            ),
            bgcolor="rgba(0,0,0,0)",  # Transparent background
            sum=100,  # Coordinates sum to 100 (percentages)
        ),
        height=800,
        width=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        font=dict(size=12),
    )

    return fig


def visualize_weights(
    weight_configs_path: str,
    output_path: str | None = None,
    plot_type: str = "all",
    dpi: int = 300,
) -> dict[str, go.Figure]:
    """Generate visualizations of mixture weight distributions.

    Args:
        weight_configs_path: Path to weight_configs.json (GCS or local).
        output_path: Output path for saving images (without extension).
            If None, figures are not saved.
        plot_type: Type of plot to generate: "histogram", "box", "ternary", or "all".
        dpi: DPI for saved images (default 300).

    Returns:
        Dictionary mapping plot names to Figure objects.
    """
    # Load weight configurations
    data = load_weight_configs(weight_configs_path)

    experiment_name = data.get("experiment_name", "Unknown Experiment")
    domains = data["domains"]
    phases = data["phases"]
    configs = data["configs"]

    print(f"Loaded {len(configs)} configurations")
    print(f"Domains: {domains}")
    print(f"Phases: {phases}")

    # Extract weights by phase
    weights_by_phase = extract_weights_by_phase(configs, phases, domains)

    figures: dict[str, go.Figure] = {}

    # Generate requested plots
    if plot_type in ("histogram", "all"):
        fig_hist = create_weight_distribution_figure(weights_by_phase, phases, domains, experiment_name)
        figures["histogram"] = fig_hist

    if plot_type in ("box", "all"):
        fig_box = create_box_plot_figure(weights_by_phase, phases, domains, experiment_name)
        figures["box"] = fig_box

    if plot_type in ("ternary", "all") and len(domains) == 3:
        fig_ternary = create_ternary_plot_figure(weights_by_phase, phases, domains, experiment_name)
        figures["ternary"] = fig_ternary

    # Save figures if output path provided
    if output_path:
        # Calculate scale factor for target DPI (plotly default is ~72 DPI)
        scale = dpi / 72.0

        for name, fig in figures.items():
            # Determine output filename
            if len(figures) == 1:
                out_file = output_path
            else:
                # Add suffix for multiple plots
                base, ext = output_path.rsplit(".", 1) if "." in output_path else (output_path, "png")
                out_file = f"{base}_{name}.{ext}"

            # Ensure .png extension for images
            if not out_file.endswith((".png", ".pdf", ".svg", ".jpeg", ".webp", ".html")):
                out_file = f"{out_file}.png"

            print(f"Saving {name} plot to {out_file} (DPI={dpi})")

            # Try to save as image, fall back to HTML if kaleido not available
            if out_file.endswith(".html"):
                fig.write_html(out_file)
            else:
                try:
                    fig.write_image(out_file, scale=scale)
                except ValueError as e:
                    if "kaleido" in str(e).lower():
                        # Fall back to HTML
                        html_file = out_file.rsplit(".", 1)[0] + ".html"
                        print(f"  Kaleido not installed, saving as HTML: {html_file}")
                        print("  Install kaleido for PNG export: pip install kaleido")
                        fig.write_html(html_file)
                    else:
                        raise

    return figures


def main():
    parser = argparse.ArgumentParser(
        description="Visualize mixture weight distributions from weight_configs.json"
    )
    parser.add_argument(
        "weight_configs_path",
        type=str,
        help="Path to weight_configs.json (supports gs:// URLs)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for saving images (without extension for multiple plots)",
    )
    parser.add_argument(
        "--plot-type",
        "-t",
        type=str,
        choices=["histogram", "box", "ternary", "all"],
        default="all",
        help="Type of plot to generate (default: all)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved images (default: 300)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (requires browser)",
    )

    args = parser.parse_args()

    figures = visualize_weights(
        weight_configs_path=args.weight_configs_path,
        output_path=args.output,
        plot_type=args.plot_type,
        dpi=args.dpi,
    )

    if args.show:
        for fig in figures.values():
            fig.show()


if __name__ == "__main__":
    main()
