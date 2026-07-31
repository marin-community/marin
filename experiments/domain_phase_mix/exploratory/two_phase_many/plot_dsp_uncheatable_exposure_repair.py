# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "plotly"]
# ///
"""Visualize aggregate-exposure repair for the DSP uncheatable mixture.

The diagnostic keeps the two-phase phase-contrast vector fixed while changing
the phase-aggregated mass. This tests whether a poor two-phase candidate is
mainly underexposing important buckets in aggregate, rather than using a bad
phase schedule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from plot_one_vs_two_phase_best_mixtures import (
    COMPARISONS,
    OUTPUT_DIR as BEST_MIXTURE_OUTPUT_DIR,
    PHASE_0_FRACTION,
    PHASE_1_FRACTION,
    PLOT_CONFIG,
    clean_domain,
    comparison_frames,
)


OUTPUT_DIR = (
    BEST_MIXTURE_OUTPUT_DIR.parent / "dsp_uncheatable_exposure_repair_20260702"
)
PRIMARY_REPAIR_DOMAINS = [
    "dolmino_synth_code",
    "dolma3_wikipedia",
    "dolmino_stack_edu_fim",
]
COLORS = {
    "two_phase_original": "#6f8190",
    "repaired_top3": "#e36f2c",
    "repaired_all_deficits": "#8b5cf6",
    "single_phase_reference": "#2f9e44",
}


@dataclass(frozen=True)
class RepairedMixture:
    name: str
    label: str
    selected_domains: list[str]
    frame: pd.DataFrame
    contrast_scale: float
    total_selected_mass_increase: float
    donor_scale: float


def proportional_mass(merged: pd.DataFrame) -> pd.Series:
    """Infer the proportional baseline mass from aggregate weights and epochs."""

    inferred = merged["aggregate_weight_single"] / merged["simulated_epochs_single"]
    inferred_two = merged["aggregate_weight_two_phase"] / merged["simulated_epochs_two_phase"]
    max_error = float((inferred - inferred_two).abs().max())
    if max_error > 1e-8:
        raise ValueError(f"inconsistent inferred proportional masses: max_error={max_error}")
    return inferred


def reconstruct_with_aggregate(
    merged: pd.DataFrame, aggregate_weight: pd.Series
) -> tuple[pd.DataFrame, float]:
    """Reconstruct phase weights from aggregate mass and original phase contrast."""

    contrast = merged["phase_0_weight_two_phase"] - merged["phase_1_weight_two_phase"]
    scale = 1.0
    positive = contrast > 0
    negative = contrast < 0
    if positive.any():
        scale = min(
            scale,
            float((aggregate_weight[positive] / (PHASE_0_FRACTION * contrast[positive])).min()),
        )
    if negative.any():
        scale = min(
            scale,
            float(
                (
                    aggregate_weight[negative]
                    / (PHASE_1_FRACTION * (-contrast[negative]))
                ).min()
            ),
        )
    scale = max(0.0, min(1.0, scale))
    repaired = merged[["domain", "domain_short", "domain_group"]].copy()
    repaired["aggregate_weight"] = aggregate_weight
    repaired["phase_0_weight"] = aggregate_weight + PHASE_1_FRACTION * scale * contrast
    repaired["phase_1_weight"] = aggregate_weight - PHASE_0_FRACTION * scale * contrast
    p = proportional_mass(merged)
    repaired["phase_0_epoch_multiplier"] = repaired["phase_0_weight"] / p
    repaired["phase_1_epoch_multiplier"] = repaired["phase_1_weight"] / p
    repaired["simulated_epochs"] = repaired["aggregate_weight"] / p
    for column in ["phase_0_weight", "phase_1_weight", "aggregate_weight"]:
        if abs(float(repaired[column].sum()) - 1.0) > 1e-8:
            raise ValueError(f"{column} does not sum to 1: {repaired[column].sum()}")
    min_weight = float(repaired[["phase_0_weight", "phase_1_weight"]].min().min())
    if min_weight < -1e-10:
        raise ValueError(f"negative reconstructed phase weight: {min_weight}")
    return repaired, scale


def repair_aggregate_exposure(
    merged: pd.DataFrame, selected_domains: list[str], name: str, label: str
) -> RepairedMixture:
    """Raise selected aggregate exposures to the single-phase reference."""

    selected = merged["domain"].isin(selected_domains)
    if not selected.any():
        raise ValueError(f"no selected domains found for {name}")
    aggregate = merged["aggregate_weight_two_phase"].copy()
    target = merged["aggregate_weight_single"]
    increase = (target[selected] - aggregate[selected]).clip(lower=0.0)
    aggregate.loc[selected] = aggregate.loc[selected] + increase
    total_increase = float(increase.sum())
    donors = (~selected) & (aggregate > target)
    donor_surplus = (aggregate[donors] - target[donors]).clip(lower=0.0)
    total_surplus = float(donor_surplus.sum())
    if total_increase > total_surplus + 1e-12:
        raise ValueError(
            f"{name} needs aggregate mass increase {total_increase}, "
            f"but non-selected surplus only has {total_surplus}"
        )
    donor_scale = 1.0 if total_surplus == 0 else 1.0 - total_increase / total_surplus
    aggregate.loc[donors] = target[donors] + donor_scale * donor_surplus
    total_error = float(abs(aggregate.sum() - 1.0))
    if total_error > 1e-10:
        raise ValueError(f"{name} repaired aggregate sum error={total_error}")
    repaired, contrast_scale = reconstruct_with_aggregate(merged, aggregate)
    return RepairedMixture(
        name=name,
        label=label,
        selected_domains=selected_domains,
        frame=repaired,
        contrast_scale=contrast_scale,
        total_selected_mass_increase=total_increase,
        donor_scale=donor_scale,
    )


def original_frame(merged: pd.DataFrame, suffix: str) -> pd.DataFrame:
    frame = merged[["domain", "domain_short", "domain_group"]].copy()
    frame["phase_0_weight"] = merged[f"phase_0_weight_{suffix}"]
    frame["phase_1_weight"] = merged[f"phase_1_weight_{suffix}"]
    frame["aggregate_weight"] = merged[f"aggregate_weight_{suffix}"]
    frame["phase_0_epoch_multiplier"] = merged[f"phase_0_epoch_multiplier_{suffix}"]
    frame["phase_1_epoch_multiplier"] = merged[f"phase_1_epoch_multiplier_{suffix}"]
    frame["simulated_epochs"] = merged[f"simulated_epochs_{suffix}"]
    return frame


def long_frame(named_frames: list[tuple[str, str, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, label, frame in named_frames:
        for _, row in frame.iterrows():
            for phase, weight_column, epoch_column in [
                ("phase_0", "phase_0_weight", "phase_0_epoch_multiplier"),
                ("phase_1", "phase_1_weight", "phase_1_epoch_multiplier"),
                ("aggregate", "aggregate_weight", "simulated_epochs"),
            ]:
                rows.append(
                    {
                        "mixture": name,
                        "mixture_label": label,
                        "phase": phase,
                        "domain": row["domain"],
                        "domain_short": row["domain_short"],
                        "domain_group": row["domain_group"],
                        "weight": float(row[weight_column]),
                        "epoch_multiplier": float(row[epoch_column]),
                    }
                )
    return pd.DataFrame(rows)


def plot_repair(
    title: str,
    long_df: pd.DataFrame,
    order_domains: list[str],
    subtitle: str,
) -> go.Figure:
    domain_to_y = {domain: clean_domain(domain) for domain in order_domains}
    y_order = [domain_to_y[domain] for domain in order_domains]
    fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=[
            "Phase 0 weights",
            "Phase 1 weights",
            "Aggregate weights",
            "Aggregate exposure",
        ],
        shared_yaxes=True,
        horizontal_spacing=0.03,
    )
    panels = [
        ("phase_0", "weight", "mixture weight"),
        ("phase_1", "weight", "mixture weight"),
        ("aggregate", "weight", "mixture weight"),
        ("aggregate", "epoch_multiplier", "realized simulated epochs"),
    ]
    mixtures = list(dict.fromkeys(long_df["mixture"].tolist()))
    for col, (phase, value_column, x_title) in enumerate(panels, start=1):
        for mixture in mixtures:
            data = long_df[
                (long_df["phase"] == phase) & (long_df["mixture"] == mixture)
            ].copy()
            data["domain_short"] = data["domain"].map(domain_to_y)
            data = data.set_index("domain").loc[order_domains].reset_index()
            fig.add_trace(
                go.Bar(
                    x=data[value_column],
                    y=data["domain_short"],
                    orientation="h",
                    name=str(data["mixture_label"].iloc[0]),
                    legendgroup=mixture,
                    showlegend=col == 1,
                    marker_color=COLORS[mixture],
                    opacity=0.86,
                    customdata=data[["domain", "epoch_multiplier", "domain_group"]],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"{x_title}: %{{x:.5f}}<br>"
                        "epochs: %{customdata[1]:.2f}<br>"
                        "group: %{customdata[2]}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
    fig.update_yaxes(categoryorder="array", categoryarray=y_order)
    for col, (_phase, _value_column, x_title) in enumerate(panels, start=1):
        fig.update_xaxes(title_text=x_title, row=1, col=col)
    fig.update_layout(
        title={"text": f"{title}<br><sup>{subtitle}</sup>", "x": 0.5, "xanchor": "center"},
        barmode="group",
        template="plotly_white",
        height=1220,
        width=2250,
        margin={"l": 210, "r": 40, "t": 115, "b": 105},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.055,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255,255,255,0.94)",
            "bordercolor": "#d9e0ea",
            "borderwidth": 1,
        },
    )
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = [s for s in COMPARISONS if s.task == "Uncheatable BPB" and s.method == "DSP"][0]
    _, _, merged = comparison_frames(spec)
    merged = merged.sort_values("domain").reset_index(drop=True)
    p = proportional_mass(merged)
    merged["proportional_mass"] = p
    merged["exposure_deficit_single_minus_two"] = (
        merged["simulated_epochs_single"] - merged["simulated_epochs_two_phase"]
    )
    top_deficits = (
        merged.sort_values("exposure_deficit_single_minus_two", ascending=False)
        .head(12)[
            [
                "domain",
                "aggregate_weight_single",
                "aggregate_weight_two_phase",
                "simulated_epochs_single",
                "simulated_epochs_two_phase",
                "exposure_deficit_single_minus_two",
                "proportional_mass",
            ]
        ]
        .copy()
    )
    all_deficit_domains = merged.loc[
        merged["exposure_deficit_single_minus_two"] > 1e-9, "domain"
    ].tolist()
    top3 = repair_aggregate_exposure(
        merged,
        PRIMARY_REPAIR_DOMAINS,
        name="repaired_top3",
        label="top-3 exposure repair",
    )
    all_deficits = repair_aggregate_exposure(
        merged,
        all_deficit_domains,
        name="repaired_all_deficits",
        label="all-deficit exposure repair",
    )
    two_original = original_frame(merged, "two_phase")
    single_reference = original_frame(merged, "single")
    named_top3 = [
        ("two_phase_original", "original two-phase DSP", two_original),
        (top3.name, top3.label, top3.frame),
        ("single_phase_reference", "single-phase DSP reference", single_reference),
    ]
    named_all = [
        ("two_phase_original", "original two-phase DSP", two_original),
        (all_deficits.name, all_deficits.label, all_deficits.frame),
        ("single_phase_reference", "single-phase DSP reference", single_reference),
    ]
    order_domains = (
        merged.sort_values("exposure_deficit_single_minus_two", ascending=True)["domain"]
        .tolist()
    )
    top3_long = long_frame(named_top3)
    all_long = long_frame(named_all)
    top3_long.to_csv(OUTPUT_DIR / "dsp_uncheatable_top3_exposure_repair_long.csv", index=False)
    all_long.to_csv(OUTPUT_DIR / "dsp_uncheatable_all_deficits_exposure_repair_long.csv", index=False)
    top_deficits.to_csv(OUTPUT_DIR / "dsp_uncheatable_exposure_deficits_top12.csv", index=False)
    top3.frame.to_csv(OUTPUT_DIR / "dsp_uncheatable_top3_exposure_repaired_mixture.csv", index=False)
    all_deficits.frame.to_csv(
        OUTPUT_DIR / "dsp_uncheatable_all_deficits_exposure_repaired_mixture.csv",
        index=False,
    )
    metadata = {
        "phase_0_fraction": PHASE_0_FRACTION,
        "phase_1_fraction": PHASE_1_FRACTION,
        "primary_repair_domains": PRIMARY_REPAIR_DOMAINS,
        "top3_total_selected_mass_increase": top3.total_selected_mass_increase,
        "top3_donor_scale": top3.donor_scale,
        "top3_contrast_scale": top3.contrast_scale,
        "all_deficits_count": len(all_deficit_domains),
        "all_deficits_total_selected_mass_increase": all_deficits.total_selected_mass_increase,
        "all_deficits_donor_scale": all_deficits.donor_scale,
        "all_deficits_contrast_scale": all_deficits.contrast_scale,
    }
    (OUTPUT_DIR / "exposure_repair_metadata.json").write_text(json.dumps(metadata, indent=2))
    top3_fig = plot_repair(
        "DSP uncheatable aggregate-exposure repair",
        top3_long,
        order_domains,
        "Raises dolmino_synth_code, dolma3_wikipedia, and dolmino_stack_edu_fim to the single-phase aggregate exposure; preserves original phase contrast.",
    )
    all_fig = plot_repair(
        "DSP uncheatable aggregate-exposure repair: all deficits",
        all_long,
        order_domains,
        "Raises every underexposed bucket to the single-phase aggregate exposure and scales the remaining buckets down.",
    )
    top3_fig.write_html(
        OUTPUT_DIR / "dsp_uncheatable_top3_exposure_repair.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )
    all_fig.write_html(
        OUTPUT_DIR / "dsp_uncheatable_all_deficits_exposure_repair.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>DSP uncheatable exposure repair</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;color:#172033}"
        "table{border-collapse:collapse;margin:16px 0;width:100%;font-size:13px}"
        "th,td{border:1px solid #d9e0ea;padding:6px 8px;text-align:left;vertical-align:top}"
        "th{background:#eef3f8} code{background:#eef3f8;padding:2px 4px;border-radius:4px}</style>",
        "</head><body>",
        "<h1>DSP uncheatable aggregate-exposure repair</h1>",
        "<p>Aggregate exposure is inferred as aggregate weight divided by proportional baseline mass. "
        "The repair changes aggregate mass and preserves the original two-phase contrast unless a global "
        "contrast shrink is needed for nonnegative phase weights.</p>",
        "<h2>Repair metadata</h2>",
        pd.DataFrame([metadata]).to_html(index=False, escape=True),
        "<h2>Top exposure deficits</h2>",
        top_deficits.to_html(index=False, escape=True),
        "<h2>Top-3 repair</h2>",
        pio.to_html(top3_fig, include_plotlyjs="cdn", full_html=False, config=PLOT_CONFIG),
        "<h2>All-deficit repair</h2>",
        pio.to_html(all_fig, include_plotlyjs=False, full_html=False, config=PLOT_CONFIG),
        "</body></html>",
    ]
    (OUTPUT_DIR / "dsp_uncheatable_exposure_repair.html").write_text("\n".join(parts))
    print(json.dumps(metadata, indent=2))
    print(f"wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
