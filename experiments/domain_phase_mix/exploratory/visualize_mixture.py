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
# ]
# ///
"""Visualize multi-phase mixture weights as stacked bar charts.

Creates publication-quality visualizations of data mixture weights across
training phases using Plotly.

Usage:
    uv run experiments/domain_phase_mix/exploratory/visualize_mixture.py
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Publication-quality color palette (colorblind-friendly)
COLORS = {
    "nemotron_full": "#2E86AB",  # Steel blue
    "dolmino": "#A23B72",  # Deep rose
    "openthoughts_sft": "#F18F01",  # Amber
}

# Readable labels for domains
DOMAIN_LABELS = {
    "nemotron_full": "Pretrain (Nemotron)",
    "dolmino": "Midtrain (Dolmino)",
    "openthoughts_sft": "SFT (OpenThoughts)",
}

PHASE_LABELS = ["Phase 1", "Phase 2", "Phase 3"]


def create_mixture_chart(
    weights: list[tuple[float, float, float]],
    title: str = "",
    domain_names: list[str] | None = None,
    phase_labels: list[str] | None = None,
    show_legend: bool = True,
    width: int = 600,
    height: int = 400,
) -> go.Figure:
    """Create a stacked bar chart for mixture weights across phases.

    Args:
        weights: List of (nemotron, dolmino, sft) weight tuples, one per phase.
        title: Chart title.
        domain_names: Domain names in order. Defaults to standard three domains.
        phase_labels: Labels for each phase. Defaults to Phase 1, 2, 3.
        show_legend: Whether to show the legend.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        Plotly Figure object.
    """
    domain_names = domain_names or ["nemotron_full", "dolmino", "openthoughts_sft"]
    phase_labels = phase_labels or PHASE_LABELS

    fig = go.Figure()

    # Add bars for each domain (stacked)
    for i, domain in enumerate(domain_names):
        domain_weights = [w[i] for w in weights]
        fig.add_trace(
            go.Bar(
                name=DOMAIN_LABELS.get(domain, domain),
                x=phase_labels,
                y=domain_weights,
                marker_color=COLORS.get(domain, f"hsl({i * 120}, 70%, 50%)"),
                marker_line_width=0,
                text=[f"{w:.1%}" for w in domain_weights],
                textposition="inside",
                textfont=dict(size=13, color="white", family="Arial"),
                insidetextanchor="middle",
                hovertemplate="%{x}<br>%{fullData.name}: %{y:.1%}<extra></extra>",
            )
        )

    # Update layout for publication quality
    fig.update_layout(
        barmode="stack",
        title=dict(
            text=title,
            font=dict(size=18, family="Arial", color="#1a1a1a"),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title=None,
            tickfont=dict(size=14, family="Arial", color="#333"),
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor="#666",
        ),
        yaxis=dict(
            title=dict(text="Weight", font=dict(size=14, family="Arial", color="#333")),
            tickfont=dict(size=12, family="Arial", color="#333"),
            tickformat=".0%",
            range=[0, 1.02],
            showgrid=True,
            gridwidth=0.5,
            gridcolor="#e0e0e0",
            showline=True,
            linewidth=1,
            linecolor="#666",
        ),
        legend=(
            dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=12, family="Arial"),
                bgcolor="rgba(255,255,255,0.8)",
            )
            if show_legend
            else dict(visible=False)
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=20, t=10, b=40),
        width=width,
        height=height,
        bargap=0,  # No gap between bars in different phases
        bargroupgap=0,  # No gap between stacked bars
    )

    return fig


def create_comparison_chart(
    mixtures: dict[str, list[tuple[float, float, float]]],
    title: str = "Data Mixture Comparison",
    width: int = 900,
    height: int = 450,
) -> go.Figure:
    """Create a comparison chart showing multiple mixtures side by side.

    Args:
        mixtures: Dict mapping mixture names to weight lists.
        title: Chart title.
        width: Figure width.
        height: Figure height.

    Returns:
        Plotly Figure with subplots.
    """
    n_mixtures = len(mixtures)
    fig = make_subplots(
        rows=1,
        cols=n_mixtures,
        subplot_titles=list(mixtures.keys()),
        horizontal_spacing=0.08,
    )

    domain_names = ["nemotron_full", "dolmino", "openthoughts_sft"]

    for col, (_name, weights) in enumerate(mixtures.items(), 1):
        for i, domain in enumerate(domain_names):
            domain_weights = [w[i] for w in weights]
            show_legend = col == 1  # Only show legend for first subplot

            fig.add_trace(
                go.Bar(
                    name=DOMAIN_LABELS.get(domain, domain),
                    x=PHASE_LABELS,
                    y=domain_weights,
                    marker_color=COLORS.get(domain),
                    marker_line_width=0,
                    text=[f"{w:.0%}" for w in domain_weights],
                    textposition="inside",
                    textfont=dict(size=11, color="white", family="Arial"),
                    insidetextanchor="middle",
                    showlegend=show_legend,
                    legendgroup=domain,
                ),
                row=1,
                col=col,
            )

    fig.update_layout(
        barmode="stack",
        title=dict(
            text=title,
            font=dict(size=20, family="Arial", color="#1a1a1a"),
            x=0.5,
            xanchor="center",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
            font=dict(size=12, family="Arial"),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=20, t=120, b=40),
        width=width,
        height=height,
        bargap=0,
        bargroupgap=0,
    )

    # Update all y-axes
    for i in range(1, n_mixtures + 1):
        fig.update_yaxes(
            tickformat=".0%",
            range=[0, 1.02],
            showgrid=True,
            gridwidth=0.5,
            gridcolor="#e0e0e0",
            showline=True,
            linewidth=1,
            linecolor="#666",
            row=1,
            col=i,
        )
        fig.update_xaxes(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor="#666",
            tickfont=dict(size=12, family="Arial"),
            row=1,
            col=i,
        )

    # Only show y-axis title on first subplot
    fig.update_yaxes(title=dict(text="Weight", font=dict(size=13, family="Arial")), row=1, col=1)

    return fig


def create_all_baselines_chart(
    baselines: dict[str, dict],
    title: str = "All Baseline Mixtures with C4 BPB",
    width: int = 1400,
    height: int = 500,
) -> go.Figure:
    """Create a chart showing all baseline mixtures with their BPB, arc_challenge acc, and choice_logprob scores.

    Args:
        baselines: Dict mapping names to {"weights": [...], "bpb": float,
            "arc_acc": float, "arc_bpb": float, "choice_logprob": float}.
        title: Chart title.
        width: Figure width.
        height: Figure height.

    Returns:
        Plotly Figure with all baselines.
    """
    n_baselines = len(baselines)

    # Build subplot titles with metrics
    subplot_titles = []
    for name, data in baselines.items():
        predicted = data.get("predicted", {})

        # BPB with optional predicted value
        bpb = data["bpb"]
        if "bpb" in predicted:
            bpb_str = f"C4-EN/BPB: {bpb:.4f} (pred: {predicted['bpb']:.4f})"
        else:
            bpb_str = f"C4-EN/BPB: {bpb:.4f}"

        # Arc challenge accuracy with optional predicted value
        arc_acc = data.get("arc_acc")
        if arc_acc is not None:
            if "arc_acc" in predicted:
                arc_acc_str = f"Arc Acc: {arc_acc:.2%} (pred: {predicted['arc_acc']:.2%})"
            else:
                arc_acc_str = f"Arc Challenge Acc: {arc_acc:.2%}"
        else:
            arc_acc_str = ""

        # Arc challenge BPB with optional predicted value
        arc_bpb = data.get("arc_bpb")
        if arc_bpb is not None:
            if "arc_bpb" in predicted:
                arc_bpb_str = f"Arc BPB: {arc_bpb:.4f} (pred: {predicted['arc_bpb']:.4f})"
            else:
                arc_bpb_str = f"Arc BPB: {arc_bpb:.4f}"
        else:
            arc_bpb_str = ""

        # Choice logprob with optional predicted value
        logprob = data.get("choice_logprob")
        if logprob is not None:
            if "choice_logprob" in predicted:
                logprob_str = f"choice_logprob: {logprob:.4f} (pred: {predicted['choice_logprob']:.4f})"
            else:
                logprob_str = f"choice_logprob: {logprob:.4f}"
        else:
            logprob_str = ""

        lines = [name, f"<b>{bpb_str}</b>"]
        if arc_acc_str:
            lines.append(f"<b>{arc_acc_str}</b>")
        if arc_bpb_str:
            lines.append(f"<b>{arc_bpb_str}</b>")
        if logprob_str:
            lines.append(f"<b>{logprob_str}</b>")
        subplot_titles.append("<br>".join(lines))

    fig = make_subplots(
        rows=1,
        cols=n_baselines,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
    )

    domain_names = ["nemotron_full", "dolmino", "openthoughts_sft"]

    for col, (_name, data) in enumerate(baselines.items(), 1):
        weights = data["weights"]
        for i, domain in enumerate(domain_names):
            domain_weights = [w[i] for w in weights]
            show_legend = col == 1

            # For small weights, show text outside; for larger weights, show inside
            text_positions = ["inside" if w >= 0.01 else "none" for w in domain_weights]
            fig.add_trace(
                go.Bar(
                    name=DOMAIN_LABELS.get(domain, domain),
                    x=PHASE_LABELS,
                    y=domain_weights,
                    marker_color=COLORS.get(domain),
                    marker_line_width=0,
                    text=[f"{w:.0%}" for w in domain_weights],
                    textposition=text_positions,
                    textfont=dict(size=10, color="white", family="Arial"),
                    insidetextanchor="middle",
                    showlegend=show_legend,
                    legendgroup=domain,
                ),
                row=1,
                col=col,
            )

    fig.update_layout(
        barmode="stack",
        title=dict(
            text=title,
            font=dict(size=20, family="Arial", color="#1a1a1a"),
            x=0.5,
            xanchor="center",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="center",
            x=0.5,
            font=dict(size=12, family="Arial"),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=50, t=220, b=40),
        width=width,
        height=height,
        bargap=0,
        bargroupgap=0,
    )

    # Update all axes
    for i in range(1, n_baselines + 1):
        fig.update_yaxes(
            tickformat=".0%",
            range=[0, 1.02],
            showgrid=True,
            gridwidth=0.5,
            gridcolor="#e0e0e0",
            showline=True,
            linewidth=1,
            linecolor="#666",
            tickfont=dict(size=10, family="Arial"),
            row=1,
            col=i,
        )
        fig.update_xaxes(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor="#666",
            tickfont=dict(size=10, family="Arial"),
            row=1,
            col=i,
        )

    fig.update_yaxes(title=dict(text="Weight", font=dict(size=12, family="Arial")), row=1, col=1)

    # Update subplot title font
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=11, family="Arial")

    return fig


def load_baselines_from_csv(csv_path: str) -> dict[str, dict]:
    """Load baseline run data from CSV file.

    Args:
        csv_path: Path to the results CSV file.

    Returns:
        Dict mapping display names to baseline data with weights, bpb, and choice_logprob.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Filter to baseline runs (run_id >= 90000)
    baseline_df = df[df["run_id"] >= 90000].copy()

    # Human-readable labels for each baseline
    # 90006 and 90007 optimized for C4-BPB, 90008 optimized for choice_logprob
    baseline_labels = {
        90000: "90000: nem→dol→sft\n(diverged)",
        90001: "90001: nem→dol→\nnem+sft",
        90002: "90002: nem→dol→\nbalanced",
        90003: "90003: nem→dol→\nnem+dol",
        90004: "90004: pure\nnemotron",
        90005: "90005: nem→nem→\ndol",
        90006: "90006: RegMix\n(opt. C4-BPB)",
        90007: "90007: RegMix 5-fold\n(opt. C4-BPB)",
        90008: "90008: RegMix 5-fold\n(opt. choice_logprob)",
        90009: "90009: RegMix 5-fold\n(opt. arc_challenge/bpb)",
    }

    # Predicted values from RegMix regression analysis (regmix_regression_kfold.results.txt)
    # These are the predicted metric values for the optimized mixtures
    regmix_predictions = {
        90006: {
            "bpb": 1.1422,
            "arc_bpb": 1.3601,
            "choice_logprob": -5.6879,
            "arc_acc": 0.1824,
        },
        90007: {
            "bpb": 1.1422,
            "arc_bpb": 1.3481,
            "choice_logprob": -5.6666,
            "arc_acc": 0.1842,
        },
        90008: {
            "bpb": 1.1506,
            "arc_bpb": 1.3379,
            "choice_logprob": -5.6642,
            "arc_acc": 0.1859,
        },
        90009: {
            "bpb": 1.1590,
            "arc_bpb": 1.3331,
            "choice_logprob": -5.6707,
            "arc_acc": 0.1861,
        },
    }

    baselines = {}
    for _, row in baseline_df.iterrows():
        run_id = int(row["run_id"])
        label = baseline_labels.get(run_id, f"{run_id}")

        # Extract phase weights
        weights = [
            (row["phase_0_nemotron_full"], row["phase_0_dolmino"], row["phase_0_openthoughts_sft"]),
            (row["phase_1_nemotron_full"], row["phase_1_dolmino"], row["phase_1_openthoughts_sft"]),
            (row["phase_2_nemotron_full"], row["phase_2_dolmino"], row["phase_2_openthoughts_sft"]),
        ]

        baseline_data = {
            "weights": weights,
            "bpb": row["eval/paloma/c4_en/bpb"],
            "arc_acc": row["lm_eval/arc_challenge/acc"],
            "arc_bpb": row["lm_eval/arc_challenge/bpb"],
            "choice_logprob": row["lm_eval/arc_challenge/choice_logprob"],
        }

        # Add predicted values for RegMix runs
        if run_id in regmix_predictions:
            baseline_data["predicted"] = regmix_predictions[run_id]

        baselines[label] = baseline_data

    return baselines


def main():
    from pathlib import Path

    output_dir = Path(__file__).parent
    csv_path = output_dir / "3_partitions_3_phases_6.csv"

    # Load all baselines from CSV
    all_baselines = load_baselines_from_csv(str(csv_path))
    print(f"Loaded {len(all_baselines)} baselines from {csv_path}")

    # Include all baselines (including diverged) for the main comparison
    completed_baselines = {k: v for k, v in all_baselines.items() if v["bpb"] is not None}

    # Create chart with all baselines
    fig_all = create_all_baselines_chart(
        baselines=completed_baselines,
        title="Baseline & RegMix-Optimized Data Mixtures",
        width=2400,
        height=700,
    )

    html_path = output_dir / "all_baselines.html"
    fig_all.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"Saved: {html_path}")

    png_path = output_dir / "all_baselines.png"
    fig_all.write_image(str(png_path), scale=2)
    print(f"Saved: {png_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY (sorted by C4 BPB)")
    print("=" * 60)

    sorted_baselines = sorted(
        [(k, v) for k, v in all_baselines.items() if v["bpb"] is not None], key=lambda x: x[1]["bpb"]
    )

    for name, data in sorted_baselines:
        bpb = data["bpb"]
        logprob = data.get("choice_logprob")
        weights = data["weights"]
        status = " (DIVERGED)" if bpb > 2.0 else ""
        print(f"\n{name.replace(chr(10), ' ')}{status}")
        logprob_str = f", choice_logprob: {logprob:.4f}" if logprob is not None else ""
        print(f"  C4-BPB: {bpb:.4f}{logprob_str}")
        for i, (nem, dol, sft) in enumerate(weights):
            print(f"  Phase {i+1}: nem={nem:.0%}, dol={dol:.0%}, sft={sft:.0%}")

    # Best observed (by BPB, excluding diverged)
    best = min(sorted_baselines, key=lambda x: x[1]["bpb"] if x[1]["bpb"] < 2.0 else float("inf"))
    print(f"\n{'='*60}")
    print(f"BEST BASELINE (by C4-BPB): {best[0].replace(chr(10), ' ')}")
    print(f"BPB: {best[1]['bpb']:.4f}")
    print("=" * 60)

    # Best by choice_logprob (higher/less negative is better)
    valid_logprobs = [(k, v) for k, v in sorted_baselines if v.get("choice_logprob") is not None and v["bpb"] < 2.0]
    if valid_logprobs:
        best_logprob = max(valid_logprobs, key=lambda x: x[1]["choice_logprob"])
        print(f"\nBEST BASELINE (by choice_logprob): {best_logprob[0].replace(chr(10), ' ')}")
        print(f"choice_logprob: {best_logprob[1]['choice_logprob']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
