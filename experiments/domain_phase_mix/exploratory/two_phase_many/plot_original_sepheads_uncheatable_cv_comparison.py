# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "plotly"]
# ///
"""Compare the validated and CV-tuned Uncheatable separate-heads mixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
OUTPUT_DIR = REFERENCE_OUTPUTS / "original_separate_heads_policy_ablation_20260712"
ORIGINAL_PATH = (
    REFERENCE_OUTPUTS / "sep_lf_kl_sweep_panel_20260706" / "seplf_unch_sep_kl0p1" / "proposed_mixture_weights.csv"
)
CV_TUNED_PATH = OUTPUT_DIR / "original_sep_uncheatable_cv_selected_2p_kl0p1.csv"
OUTPUT_HTML = OUTPUT_DIR / "uncheatable_original_vs_cv_tuned_mixture.html"
OUTPUT_CSV = OUTPUT_DIR / "uncheatable_original_vs_cv_tuned_mixture.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "uncheatable_original_vs_cv_tuned_summary.json"

ORIGINAL_LABEL = "Original: L2=0.1, KL=0.1 (observed 3e18 BPB 0.988712)"
CV_TUNED_LABEL = "CV-tuned: L2=1.0, KL=0.1 (unvalidated)"
PHASE_0_FRACTION = 0.8
PHASE_1_FRACTION = 0.2
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def clean_domain(domain: str) -> str:
    prefixes = ("dolma3_", "dolmino_")
    cleaned = domain
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned.removeprefix(prefix)
            break
    return cleaned.replace("cc/", "CC: ").replace("_", " ")


def load_comparison() -> pd.DataFrame:
    original = pd.read_csv(ORIGINAL_PATH)
    tuned = pd.read_csv(CV_TUNED_PATH)
    required = {
        "domain",
        "phase_0_weight",
        "phase_1_weight",
        "aggregate_weight",
        "simulated_epochs",
    }
    for label, frame in (("original", original), ("CV-tuned", tuned)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} mixture is missing columns: {sorted(missing)}")
        if abs(frame["phase_0_weight"].sum() - 1.0) > 1e-8:
            raise ValueError(f"{label} phase 0 weights do not sum to one")
        if abs(frame["phase_1_weight"].sum() - 1.0) > 1e-8:
            raise ValueError(f"{label} phase 1 weights do not sum to one")

    merged = original[list(required)].merge(
        tuned[list(required)],
        on="domain",
        how="inner",
        suffixes=("_original", "_cv_tuned"),
        validate="one_to_one",
    )
    if len(merged) != len(original) or len(merged) != len(tuned):
        raise ValueError("Mixture domain sets differ")
    for quantity in ("phase_0_weight", "phase_1_weight", "aggregate_weight", "simulated_epochs"):
        merged[f"{quantity}_delta"] = merged[f"{quantity}_cv_tuned"] - merged[f"{quantity}_original"]
    merged["domain_short"] = merged["domain"].map(clean_domain)
    return merged.sort_values(["aggregate_weight_delta", "domain"]).reset_index(drop=True)


def tv(left: pd.Series, right: pd.Series) -> float:
    return float(0.5 * (left - right).abs().sum())


def plot(frame: pd.DataFrame, summary: dict[str, float]) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=["Phase 0 weights", "Phase 1 weights", "Aggregate weights", "Aggregate exposure"],
        shared_yaxes=True,
        horizontal_spacing=0.03,
    )
    panels = [
        ("phase_0_weight", "mixture weight"),
        ("phase_1_weight", "mixture weight"),
        ("aggregate_weight", "mixture weight"),
        ("simulated_epochs", "realized simulated epochs"),
    ]
    candidates = [
        ("original", ORIGINAL_LABEL, "#748797", 0.84),
        ("cv_tuned", CV_TUNED_LABEL, "#E36F2C", 0.92),
    ]
    y_order = frame["domain_short"].tolist()
    for column, (quantity, axis_title) in enumerate(panels, start=1):
        for suffix, label, color, opacity in candidates:
            figure.add_trace(
                go.Bar(
                    x=frame[f"{quantity}_{suffix}"],
                    y=frame["domain_short"],
                    orientation="h",
                    name=label,
                    legendgroup=suffix,
                    showlegend=column == 1,
                    marker_color=color,
                    opacity=opacity,
                    customdata=frame[["domain", f"{quantity}_delta"]],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"{axis_title}: %{{x:.6f}}<br>"
                        "CV-tuned - original: %{customdata[1]:+.6f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        figure.update_xaxes(title_text=axis_title, row=1, col=column)

    figure.update_yaxes(categoryorder="array", categoryarray=y_order)
    figure.update_layout(
        title={
            "text": "Uncheatable separate-heads mixture: original versus CV-tuned ridge",
            "x": 0.5,
            "xanchor": "center",
        },
        barmode="group",
        template="plotly_white",
        height=1220,
        width=2200,
        margin={"l": 230, "r": 40, "t": 145, "b": 115},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.06,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.94)",
            "bordercolor": "#d9e0ea",
            "borderwidth": 1,
        },
    )
    figure.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.065,
        showarrow=False,
        text=(
            "Both use the original 280-row separate-heads fit and deployment KL=0.1. "
            f"TV(original, tuned): phase 0={summary['phase_0_tv']:.3f}, "
            f"phase 1={summary['phase_1_tv']:.3f}, aggregate={summary['aggregate_tv']:.3f}."
        ),
        font={"size": 15, "color": "#44546a"},
    )
    figure.add_annotation(
        xref="paper",
        yref="paper",
        x=1,
        y=-0.115,
        showarrow=False,
        xanchor="right",
        text=(
            "Aggregate weights use 0.8 phase 0 + 0.2 phase 1. " "Only the original mixture has a 3e18 observed result."
        ),
        font={"size": 13, "color": "#64748b"},
    )
    return figure


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_comparison()
    summary = {
        "phase_0_tv": tv(frame["phase_0_weight_original"], frame["phase_0_weight_cv_tuned"]),
        "phase_1_tv": tv(frame["phase_1_weight_original"], frame["phase_1_weight_cv_tuned"]),
        "aggregate_tv": tv(frame["aggregate_weight_original"], frame["aggregate_weight_cv_tuned"]),
        "original_predicted_bpb_300m": 0.880611,
        "cv_tuned_predicted_bpb_300m": 0.879587,
        "original_observed_bpb_3e18": 0.988712,
    }
    frame.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    figure = plot(frame, summary)
    figure.write_html(OUTPUT_HTML, include_plotlyjs=True, config=PLOT_CONFIG)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
