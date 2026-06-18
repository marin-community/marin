# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "plotly",
# ]
# ///
"""Plot HellaSwag quality-swap effects and exposure changes.

This is a preliminary diagnostic for the proportional perturbation experiment.
It compares 300M HellaSwag 0-shot and 5-shot hard/smooth metric changes for
the 50% low-to-high Common Crawl quality swaps, and shows the corresponding
materialized epochs for each high/low split.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (
    TARGET_BUDGET_DOLMA3_COMMON_CRAWL,
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
)


TWO_PHASE_ROOT = Path(__file__).resolve().parent
PPERT_DIR = TWO_PHASE_ROOT / "metric_registry" / "proportional_perturbation_scale_transfer"
NOISE_DIR = TWO_PHASE_ROOT / "metric_registry" / "raw_metric_matrix_300m"
OUTPUT_DIR = TWO_PHASE_ROOT / "reference_outputs" / "ppert_mcq_smooth_controllability_20260518"
IMG_DIR = OUTPUT_DIR / "img"
MANIFEST_CSV = (
    TWO_PHASE_ROOT
    / "reference_outputs"
    / "proportional_perturbation_scale_transfer_20260507"
    / "intervention_manifest.csv"
)
ENGLISH_LITE_CSV = PPERT_DIR / "ppert_english_lite_eval_results.csv"
NOISE_PARITY_CSV = PPERT_DIR / "ppert_noise_parity_eval_results.csv"
CANDIDATES_CSV = PPERT_DIR / "proportional_perturbation_eval_candidates.csv"
PROPORTIONAL_NOISE_CSV = NOISE_DIR / "noise_baseline_proportional_variable_subset_300m.csv"

OUTPUT_HTML = IMG_DIR / "hellaswag_quality_swap_0shot_5shot_metrics_epochs.html"
EFFECTS_CSV = OUTPUT_DIR / "hellaswag_quality_swap_metric_effects.csv"
EPOCHS_CSV = OUTPUT_DIR / "hellaswag_quality_swap_materialized_epochs.csv"

SCALE = "300m_6b"
BASELINE_PANEL = "proportional_baseline_anchor_300m_6b"
PERTURBATION_PANEL = "proportional_perturbation_300m_6b"
EXPECTED_QUALITY_SWAPS = 13
PHASE_FRACTIONS = {
    "phase_0": 0.8,
    "phase_1": 0.2,
}
PLOTLY_CONFIG = {
    "displaylogo": False,
    # Plotly controls PNG export resolution by scaling CSS pixels. scale=4 is
    # roughly 384 DPI if interpreted at the browser default 96 CSS px/in.
    "toImageButtonOptions": {
        "format": "png",
        "scale": 4,
    },
}

METRIC_LEAVES = [
    "acc",
    "acc_norm",
    "bpb",
    "choice_logprob",
    "choice_logprob_norm",
]
SHOT_SPECS = [
    ("0shot", NOISE_PARITY_CSV),
    ("5shot", ENGLISH_LITE_CSV),
]


def metric_column(shot: str, leaf: str) -> str:
    return f"lm_eval/hellaswag_{shot}/{leaf}"


def metric_kind(leaf: str) -> str:
    return "hard" if leaf in {"acc", "acc_norm"} else "smooth"


def orientation(leaf: str) -> float:
    return -1.0 if leaf == "bpb" else 1.0


def topic_label(topic: str) -> str:
    return topic.replace("_and_", " + ").replace("_", " ")


def read_results_with_candidate_metadata(path: Path) -> pd.DataFrame:
    results = pd.read_csv(path, low_memory=False)
    candidates = pd.read_csv(CANDIDATES_CSV, low_memory=False)
    metadata_cols = [
        "panel",
        "scale",
        "registry_key",
        "source_experiment",
        "cohort",
        "checkpoint_root",
        "expected_checkpoint_step",
        "intervention_id",
        "intervention_type",
        "target_unit",
        "target_domain",
        "target_family",
        "tv_distance",
    ]
    candidate_by_key = candidates.set_index("registry_key", drop=False)
    enriched = results.copy()
    for col in metadata_cols:
        if col not in enriched.columns:
            enriched[col] = pd.NA
    for idx, row in enriched.iterrows():
        key = row.get("registry_key")
        if pd.isna(key) or key not in candidate_by_key.index:
            continue
        candidate_row = candidate_by_key.loc[key]
        for col in metadata_cols:
            if pd.isna(enriched.at[idx, col]) or str(enriched.at[idx, col]) == "":
                enriched.at[idx, col] = candidate_row[col]
    return enriched


def build_metric_effects() -> pd.DataFrame:
    noise = pd.read_csv(PROPORTIONAL_NOISE_CSV, low_memory=False)
    rows: list[dict[str, object]] = []
    for shot, path in SHOT_SPECS:
        results = read_results_with_candidate_metadata(path)
        metric_cols = [metric_column(shot, leaf) for leaf in METRIC_LEAVES]
        missing_cols = [col for col in metric_cols if col not in results.columns or col not in noise.columns]
        if missing_cols:
            raise ValueError(f"Missing HellaSwag {shot} metric/noise columns: {missing_cols}")

        baseline = results[results["panel"].eq(BASELINE_PANEL)]
        if len(baseline) != 1:
            raise ValueError(f"Expected one 300M baseline row in {path.name}, found {len(baseline)}")
        baseline_row = baseline.iloc[0]

        qswaps = results[
            results["panel"].eq(PERTURBATION_PANEL)
            & results["scale"].eq(SCALE)
            & results["intervention_type"].eq("quality_swap")
        ].copy()
        if len(qswaps) != EXPECTED_QUALITY_SWAPS:
            raise ValueError(f"Expected {EXPECTED_QUALITY_SWAPS} quality swaps in {path.name}, found {len(qswaps)}")
        if qswaps[metric_cols].isna().any().any():
            missing = qswaps.loc[qswaps[metric_cols].isna().any(axis=1), ["run_name", "registry_key"]]
            raise ValueError(f"Quality-swap rows have missing HellaSwag {shot} metrics:\n{missing}")

        noise_std = noise[metric_cols].std(axis=0, ddof=1)
        for _, row in qswaps.iterrows():
            target_unit = str(row["target_unit"])
            for leaf, col in zip(METRIC_LEAVES, metric_cols, strict=True):
                raw_value = float(row[col])
                baseline_value = float(baseline_row[col])
                raw_delta = raw_value - baseline_value
                improvement = orientation(leaf) * raw_delta
                std = float(noise_std[col])
                rows.append(
                    {
                        "shot": shot,
                        "metric": col,
                        "metric_leaf": leaf,
                        "metric_kind": metric_kind(leaf),
                        "intervention_id": row["intervention_id"],
                        "run_name": row["run_name"],
                        "registry_key": row["registry_key"],
                        "target_unit": target_unit,
                        "target_label": topic_label(target_unit),
                        "raw_value": raw_value,
                        "baseline_value": baseline_value,
                        "raw_delta": raw_delta,
                        "improvement": improvement,
                        "proportional_noise_std": std,
                        "z_proportional_noise": improvement / std if std > 0 else np.nan,
                    }
                )
    effects = pd.DataFrame(rows)
    effects["shot_metric"] = effects["shot"] + " / " + effects["metric_leaf"]
    return effects


def build_epoch_multipliers() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_CSV, low_memory=False)
    qswaps = manifest[manifest["intervention_type"].eq("quality_swap")].copy()
    if len(qswaps) != EXPECTED_QUALITY_SWAPS:
        raise ValueError(f"Expected {EXPECTED_QUALITY_SWAPS} manifest quality swaps, found {len(qswaps)}")

    rows: list[dict[str, object]] = []
    for _, row in qswaps.iterrows():
        target_unit = str(row["target_unit"])
        high_domain = str(row["quality_high_domain"])
        low_domain = str(row["quality_low_domain"])
        high_before = float(row["target_mass_before"])
        high_after = float(row["target_mass_after"])
        low_before = float(row["donor_mass_before"])
        low_after = float(row["donor_mass_after"])
        values = [
            ("high", "before", high_domain, {"phase_0": high_before, "phase_1": high_before}, high_before),
            (
                "high",
                "after",
                high_domain,
                {
                    "phase_0": float(row[f"phase_0_{high_domain}"]),
                    "phase_1": float(row[f"phase_1_{high_domain}"]),
                },
                high_before,
            ),
            ("low", "before", low_domain, {"phase_0": low_before, "phase_1": low_before}, low_before),
            (
                "low",
                "after",
                low_domain,
                {
                    "phase_0": float(row[f"phase_0_{low_domain}"]),
                    "phase_1": float(row[f"phase_1_{low_domain}"]),
                },
                low_before,
            ),
        ]
        for quality_level, state, domain, phase_weights, proportional_weight in values:
            domain_tokens = float(TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain])
            phase_epochs = {
                phase: phase_weights[phase] * phase_fraction * TARGET_BUDGET_DOLMA3_COMMON_CRAWL / domain_tokens
                for phase, phase_fraction in PHASE_FRACTIONS.items()
            }
            materialized_epochs = sum(phase_epochs.values())
            proportional_epochs = (
                proportional_weight * TARGET_BUDGET_DOLMA3_COMMON_CRAWL / domain_tokens
                if proportional_weight > 0
                else np.nan
            )
            rows.append(
                {
                    "intervention_id": row["intervention_id"],
                    "run_name": row["run_name"],
                    "target_unit": target_unit,
                    "target_label": topic_label(target_unit),
                    "quality_level": quality_level,
                    "state": state,
                    "domain": domain,
                    "domain_tokens": int(domain_tokens),
                    "phase_0_weight": phase_weights["phase_0"],
                    "phase_1_weight": phase_weights["phase_1"],
                    "phase_0_materialized_epochs": phase_epochs["phase_0"],
                    "phase_1_materialized_epochs": phase_epochs["phase_1"],
                    "materialized_epochs": materialized_epochs,
                    "epoch_multiplier_vs_proportional": materialized_epochs / proportional_epochs
                    if proportional_epochs > 0
                    else np.nan,
                    "quality_swap_fraction": float(row["quality_swap_fraction"]),
                    "quality_swap_mass": float(row["quality_swap_mass"]),
                    "tv_distance": float(row["tv_distance"]),
                }
            )
    return pd.DataFrame(rows)


def make_metric_heatmap(effects: pd.DataFrame, target_order: list[str]) -> object:
    column_order = [f"{shot} / {leaf}" for shot, _ in SHOT_SPECS for leaf in METRIC_LEAVES]
    display = effects.copy()
    display["target_label"] = pd.Categorical(display["target_label"], categories=target_order, ordered=True)
    display["shot_metric"] = pd.Categorical(display["shot_metric"], categories=column_order, ordered=True)
    pivot = display.pivot(index="target_label", columns="shot_metric", values="z_proportional_noise").sort_index()
    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0.0,
        aspect="auto",
        title="HellaSwag quality-swap effects vs proportional (z units; positive helps)",
        labels={
            "x": "shot / metric",
            "y": "quality-swap topic",
            "color": "oriented delta / proportional noise std",
        },
    )
    fig.update_layout(height=620, width=1700, margin=dict(l=220, r=60, t=80, b=120))
    fig.update_xaxes(tickangle=35)
    return fig


def make_metric_facets(effects: pd.DataFrame, target_order: list[str]) -> object:
    display = effects.copy()
    display["target_label"] = pd.Categorical(display["target_label"], categories=target_order, ordered=True)
    display["metric_leaf"] = pd.Categorical(display["metric_leaf"], categories=METRIC_LEAVES, ordered=True)
    display["shot"] = pd.Categorical(display["shot"], categories=[shot for shot, _ in SHOT_SPECS], ordered=True)
    fig = px.bar(
        display.sort_values(["shot", "metric_leaf", "target_label"]),
        x="z_proportional_noise",
        y="target_label",
        color="z_proportional_noise",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0.0,
        facet_row="shot",
        facet_col="metric_leaf",
        orientation="h",
        hover_data=[
            "raw_value",
            "baseline_value",
            "raw_delta",
            "improvement",
            "proportional_noise_std",
            "run_name",
        ],
        title="HellaSwag quality-swap metric effects by shot and metric",
        labels={
            "z_proportional_noise": "z improvement",
            "target_label": "quality-swap topic",
        },
    )
    fig.update_xaxes(matches=None, zeroline=True, zerolinewidth=1, zerolinecolor="#555")
    fig.update_yaxes(matches=None)
    fig.update_layout(height=1050, width=2300, margin=dict(l=220, r=60, t=90, b=70))
    return fig


def make_epoch_plot(epochs: pd.DataFrame, target_order: list[str]) -> object:
    display = epochs.copy()
    display["target_label"] = pd.Categorical(display["target_label"], categories=target_order, ordered=True)
    display["state_label"] = display["state"].str.title()
    display["quality_level"] = pd.Categorical(display["quality_level"], categories=["high", "low"], ordered=True)
    fig = px.bar(
        display.sort_values(["quality_level", "target_label", "state"]),
        x="target_label",
        y="materialized_epochs",
        color="state_label",
        barmode="group",
        facet_col="quality_level",
        hover_data=[
            "domain",
            "domain_tokens",
            "phase_0_weight",
            "phase_1_weight",
            "phase_0_materialized_epochs",
            "phase_1_materialized_epochs",
            "epoch_multiplier_vs_proportional",
            "quality_swap_mass",
            "tv_distance",
        ],
        title=(
            "Materialized epochs before/after 50% low-quality transfer "
            "(simulated-epoch materialized subset; phase 0 + phase 1)"
        ),
        labels={
            "target_label": "quality-swap topic",
            "materialized_epochs": "materialized epochs",
            "state_label": "state",
            "quality_level": "split",
        },
        color_discrete_map={"Before": "#64748b", "After": "#2563eb"},
    )
    fig.update_layout(height=700, width=1700, margin=dict(l=70, r=50, t=90, b=180))
    fig.update_xaxes(tickangle=45)
    return fig


def render_html(metric_heatmap: object, metric_facets: object, epoch_plot: object) -> None:
    sections = [
        pio.to_html(metric_heatmap, full_html=False, include_plotlyjs="cdn", config=PLOTLY_CONFIG),
        pio.to_html(metric_facets, full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG),
        pio.to_html(epoch_plot, full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG),
    ]
    OUTPUT_HTML.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                '<meta charset="utf-8">',
                "<title>HellaSwag quality-swap perturbations</title>",
                "<style>",
                "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 28px; color: #162033; }",
                "h1 { margin-bottom: 0.2rem; }",
                ".note { max-width: 1100px; line-height: 1.45; color: #475569; }",
                ".artifact { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: #f1f5f9; padding: 2px 5px; border-radius: 4px; }",
                "</style>",
                "</head>",
                "<body>",
                "<h1>HellaSwag quality-swap perturbations</h1>",
                "<p class='note'>300M proportional perturbation quality swaps only. Metric panels show oriented change versus the proportional baseline, divided by the proportional variable-subset noise standard deviation; positive values mean better. Raw values, raw deltas, and denominators are available in hover and in the exported CSV. The epoch panel shows actual materialized epochs on the simulated-epoch subset, summed over phase 0 and phase 1, not relative multipliers.</p>",
                f"<p class='note'>Metric table: <span class='artifact'>{html.escape(str(EFFECTS_CSV))}</span><br>Epoch table: <span class='artifact'>{html.escape(str(EPOCHS_CSV))}</span></p>",
                *sections,
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    effects = build_metric_effects()
    epochs = build_epoch_multipliers()
    target_order = (
        epochs.drop_duplicates("target_unit")
        .assign(target_label=lambda frame: frame["target_unit"].map(topic_label))
        ["target_label"]
        .tolist()
    )
    effects.to_csv(EFFECTS_CSV, index=False)
    epochs.to_csv(EPOCHS_CSV, index=False)
    metric_heatmap = make_metric_heatmap(effects, target_order)
    metric_facets = make_metric_facets(effects, target_order)
    epoch_plot = make_epoch_plot(epochs, target_order)
    render_html(metric_heatmap, metric_facets, epoch_plot)

    print(f"Wrote {OUTPUT_HTML}")
    print(f"Wrote {EFFECTS_CSV}")
    print(f"Wrote {EPOCHS_CSV}")
    print("Quality swaps:", effects["target_unit"].nunique())
    print("Metric cells:", len(effects))


if __name__ == "__main__":
    main()
