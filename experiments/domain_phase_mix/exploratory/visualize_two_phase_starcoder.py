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
"""Visualize two-phase starcoder baselines as stacked bar charts.

Shows all baselines with their code BPB metrics. For the RegMix-optimized
baseline (90007), shows both predicted and actual metric values.

Usage:
    uv run experiments/domain_phase_mix/exploratory/visualize_two_phase_starcoder.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "nemotron_full": "#2E86AB",  # Steel blue
    "starcoder": "#F18F01",  # Amber
}

DOMAIN_LABELS = {
    "nemotron_full": "Nemotron (web)",
    "starcoder": "StarCoder (code)",
}

PHASE_LABELS = ["Phase 0", "Phase 1"]

BASELINE_LABELS = {
    90000: "Nem → SC",
    90001: "Balanced",
    90002: "99/1 → 20/80",
    90003: "95/5 → 20/80",
    90004: "99/1 → 50/50",
    90005: "Nemotron only",
    90006: "StarCoder only",
    90007: "RegMix optimized",
}

# Predicted values for base_90007 from regmix_two_phase_starcoder.result
REGMIX_PREDICTIONS = {
    90007: {
        "eval/paloma/dolma_100_programing_languages/bpb": 0.9425,
        "eval/uncheatable_eval/github_python/bpb": 0.8560,
    },
}

METRICS = [
    "eval/paloma/dolma_100_programing_languages/bpb",
    "eval/uncheatable_eval/github_python/bpb",
]

METRIC_SHORT = {
    "eval/paloma/dolma_100_programing_languages/bpb": "Dolma Code BPB",
    "eval/uncheatable_eval/github_python/bpb": "GitHub Python BPB",
}


def create_baselines_chart(baselines: list[dict], width: int = 1800, height: int = 550) -> go.Figure:
    """Create a comparison chart showing all baselines as stacked bar charts with metrics."""
    n = len(baselines)
    domain_names = ["nemotron_full", "starcoder"]

    subplot_titles = []
    for b in baselines:
        rid = b["run_id"]
        label = BASELINE_LABELS.get(rid, str(rid))
        lines = [f"<b>{label}</b> ({rid})"]

        for metric in METRICS:
            short = METRIC_SHORT[metric]
            val = b.get(metric)
            pred = REGMIX_PREDICTIONS.get(rid, {}).get(metric)
            if val is not None:
                line = f"{short}: {val:.4f}"
                if pred is not None:
                    line += f" (pred: {pred:.4f})"
                lines.append(line)

        subplot_titles.append("<br>".join(lines))

    fig = make_subplots(
        rows=1,
        cols=n,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.03,
    )

    for col, b in enumerate(baselines, 1):
        weights = b["weights"]  # list of (nem, sc) per phase
        for i, domain in enumerate(domain_names):
            domain_weights = [w[i] for w in weights]
            show_legend = col == 1

            text_positions = ["inside" if w >= 0.05 else "none" for w in domain_weights]
            fig.add_trace(
                go.Bar(
                    name=DOMAIN_LABELS[domain],
                    x=PHASE_LABELS,
                    y=domain_weights,
                    marker_color=COLORS[domain],
                    marker_line_width=0,
                    text=[f"{w:.0%}" for w in domain_weights],
                    textposition=text_positions,
                    textfont=dict(size=12, color="white", family="Arial"),
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
            text="Two-Phase StarCoder Baselines & RegMix-Optimized Mixture",
            font=dict(size=20, family="Arial", color="#1a1a1a"),
            x=0.5,
            xanchor="center",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5,
            font=dict(size=13, family="Arial"),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=30, t=200, b=40),
        width=width,
        height=height,
        bargap=0,
        bargroupgap=0,
    )

    for i in range(1, n + 1):
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
            tickfont=dict(size=11, family="Arial"),
            row=1,
            col=i,
        )

    fig.update_yaxes(title=dict(text="Weight", font=dict(size=13, family="Arial")), row=1, col=1)

    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=11, family="Arial")

    return fig


def main():
    script_dir = Path(__file__).parent
    csv_path = script_dir / "two_phase_starcoder.csv"

    df = pd.read_csv(csv_path)
    baseline_df = df[df["run_id"] >= 90000].sort_values("run_id")

    baselines = []
    for _, row in baseline_df.iterrows():
        rid = int(row["run_id"])
        b = {
            "run_id": rid,
            "weights": [
                (row["phase_0_nemotron_full"], row["phase_0_starcoder"]),
                (row["phase_1_nemotron_full"], row["phase_1_starcoder"]),
            ],
        }
        for metric in METRICS:
            b[metric] = row.get(metric)
        baselines.append(b)

    print(f"Loaded {len(baselines)} baselines from {csv_path}")

    fig = create_baselines_chart(baselines)

    html_path = script_dir / "two_phase_starcoder_baselines.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"Saved: {html_path}")

    png_path = script_dir / "two_phase_starcoder_baselines.png"
    fig.write_image(str(png_path), scale=2)
    print(f"Saved: {png_path}")

    # Print summary sorted by code BPB
    code_metric = "eval/paloma/dolma_100_programing_languages/bpb"
    print(f"\n{'='*60}")
    print("BASELINES SORTED BY CODE BPB (lower is better)")
    print(f"{'='*60}")
    for b in sorted(baselines, key=lambda x: x.get(code_metric, float("inf"))):
        rid = b["run_id"]
        label = BASELINE_LABELS.get(rid, str(rid))
        code_bpb = b.get(code_metric, float("nan"))
        gh_bpb = b.get("eval/uncheatable_eval/github_python/bpb", float("nan"))
        pred = REGMIX_PREDICTIONS.get(rid, {})
        pred_str = ""
        if pred:
            pred_str = f"  (pred: code={pred.get(code_metric, 0):.4f}, gh={pred.get('eval/uncheatable_eval/github_python/bpb', 0):.4f})"
        print(f"  {rid} {label:20s}  code_bpb={code_bpb:.4f}  github_bpb={gh_bpb:.4f}{pred_str}")


if __name__ == "__main__":
    main()
