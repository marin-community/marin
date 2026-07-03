# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "plotly"]
# ///
"""Prepare phase-1-only DSP exposure repair mixtures.

This diagnostic keeps phase 0 fixed at the source two-phase DSP mixture and
tries to repair selected aggregate-exposure deficits entirely through phase 1.
Because phase 1 is only 20% of training, full repair is often infeasible. The
script therefore reports the maximum feasible repair fraction for each donor
policy and writes the corresponding max-feasible candidate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from plot_dsp_uncheatable_exposure_repair import original_frame, proportional_mass
from plot_one_vs_two_phase_best_mixtures import (
    COMPARISONS,
    OUTPUT_DIR as BEST_MIXTURE_OUTPUT_DIR,
    PHASE_0_FRACTION,
    PHASE_1_FRACTION,
    PLOT_CONFIG,
    clean_domain,
    comparison_frames,
)
from prepare_dsp_exposure_repair_validation_mixtures import (
    OBJECTIVE_TO_KEY,
    TARGETED_REPAIR_DOMAINS,
    launch_ready_frame,
)


OUTPUT_DIR = (
    BEST_MIXTURE_OUTPUT_DIR.parent / "dsp_phase1_repair_validation_mixtures_20260702"
)
MIXTURE_DIR = OUTPUT_DIR / "mixtures"

COLORS = {
    "two_phase_original": "#6f8190",
    "phase1_surplus": "#e36f2c",
    "phase1_surplus_rhomax": "#e36f2c",
    "phase1_any_donor": "#8b5cf6",
    "phase1_any_donor_rhomax": "#8b5cf6",
    "single_phase_reference": "#2f9e44",
}


@dataclass(frozen=True)
class Phase1Repair:
    name: str
    label: str
    objective: str
    donor_policy: str
    selected_domains: list[str]
    frame: pd.DataFrame
    rho_max: float
    rho_applied: float
    selected_phase1_delta_full: float
    selected_phase1_delta_applied: float
    selected_aggregate_mass_deficit: float
    selected_aggregate_mass_repaired: float
    donor_phase1_capacity: float
    donor_surplus_scale: float
    individual_rho_cap: float


@dataclass(frozen=True)
class CandidateSummary:
    mixture_id: str
    objective: str
    donor_policy: str
    selected_domain_count: int
    selected_domains: str
    rho_max: float
    rho_applied: float
    selected_aggregate_mass_deficit: float
    selected_aggregate_mass_repaired: float
    selected_deficit_repair_fraction: float
    donor_phase1_capacity: float
    donor_surplus_scale: float
    individual_rho_cap: float
    max_simulated_epochs: float
    q95_simulated_epochs: float
    max_phase_0_epochs: float
    max_phase_1_epochs: float
    phase0_sum: float
    phase1_sum: float
    aggregate_sum: float
    min_phase_weight: float
    remaining_positive_deficit_count: int
    max_remaining_deficit: float
    worsened_original_underexposed_count: int
    output_csv: str


def q95(values: pd.Series) -> float:
    return float(values.quantile(0.95, interpolation="linear"))


def objective_spec(task: str):
    return [spec for spec in COMPARISONS if spec.task == task and spec.method == "DSP"][0]


def selected_phase1_target(merged: pd.DataFrame) -> pd.Series:
    target = (
        merged["aggregate_weight_single"] - PHASE_0_FRACTION * merged["phase_0_weight_two_phase"]
    ) / PHASE_1_FRACTION
    return target.clip(lower=0.0)


def phase1_repair(
    merged: pd.DataFrame,
    *,
    objective: str,
    selected_domains: list[str],
    donor_policy: str,
) -> Phase1Repair:
    selected = merged["domain"].isin(selected_domains)
    if not selected.any():
        raise ValueError(f"{objective}/{donor_policy} selected no domains")

    w0 = merged["phase_0_weight_two_phase"].copy()
    w1 = merged["phase_1_weight_two_phase"].copy()
    target_w1 = selected_phase1_target(merged)
    full_delta = (target_w1 - w1).clip(lower=0.0)
    selected_delta = full_delta.where(selected, 0.0)
    selected_delta_total = float(selected_delta.sum())

    if donor_policy == "surplus":
        donor_lower = target_w1.clip(lower=0.0)
    elif donor_policy == "any_donor":
        donor_lower = pd.Series(0.0, index=merged.index)
    else:
        raise ValueError(f"unknown donor policy: {donor_policy}")

    donors = ~selected
    # A non-selected bucket that is already below its protected target has no
    # surplus to donate; do not raise it while repairing selected buckets.
    donor_floor = donor_lower.where(donor_lower < w1, w1)
    donor_surplus = (w1[donors] - donor_floor[donors]).clip(lower=0.0)
    donor_capacity = float(donor_surplus.sum())
    positive_delta = selected_delta[selected_delta > 0]
    if positive_delta.empty:
        individual_rho_cap = 1.0
    else:
        individual_rho_cap = float(((1.0 - w1[positive_delta.index]) / positive_delta).min())
    donor_rho_cap = 1.0 if selected_delta_total == 0 else donor_capacity / selected_delta_total
    rho_max = max(0.0, min(1.0, individual_rho_cap, donor_rho_cap))
    rho_applied = rho_max

    w1_new = w1.copy()
    applied_delta = rho_applied * selected_delta
    w1_new.loc[selected] = w1_new.loc[selected] + applied_delta.loc[selected]
    total_applied_delta = float(applied_delta.sum())
    if total_applied_delta > donor_capacity + 1e-12:
        raise ValueError(
            f"{objective}/{donor_policy} applied phase1 delta {total_applied_delta} "
            f"exceeds donor capacity {donor_capacity}"
        )
    donor_surplus_scale = (
        1.0 if donor_capacity == 0 else 1.0 - total_applied_delta / donor_capacity
    )
    w1_new.loc[donors] = donor_floor.loc[donors] + donor_surplus_scale * donor_surplus

    phase1_sum_error = float(abs(w1_new.sum() - 1.0))
    if phase1_sum_error > 1e-10:
        raise ValueError(f"{objective}/{donor_policy} phase1 sum error={phase1_sum_error}")
    if float(w1_new.min()) < -1e-10:
        raise ValueError(f"{objective}/{donor_policy} negative phase1 weight={w1_new.min()}")

    aggregate = PHASE_0_FRACTION * w0 + PHASE_1_FRACTION * w1_new
    p = proportional_mass(merged)
    frame = merged[["domain", "domain_short", "domain_group"]].copy()
    frame["phase_0_weight"] = w0
    frame["phase_1_weight"] = w1_new
    frame["aggregate_weight"] = aggregate
    frame["phase_0_epoch_multiplier"] = frame["phase_0_weight"] / p
    frame["phase_1_epoch_multiplier"] = frame["phase_1_weight"] / p
    frame["simulated_epochs"] = frame["aggregate_weight"] / p
    selected_aggregate_deficit = float(
        (
            merged.loc[selected, "aggregate_weight_single"]
            - merged.loc[selected, "aggregate_weight_two_phase"]
        )
        .clip(lower=0.0)
        .sum()
    )
    selected_aggregate_repaired = PHASE_1_FRACTION * total_applied_delta
    policy_label = (
        "phase-1 surplus-donor max repair"
        if donor_policy == "surplus"
        else "phase-1 any-donor max repair"
    )
    objective_key = OBJECTIVE_TO_KEY[objective]
    return Phase1Repair(
        name=f"dsp_{objective_key}_phase1_{donor_policy}_rhomax",
        label=policy_label,
        objective=objective,
        donor_policy=donor_policy,
        selected_domains=selected_domains,
        frame=frame,
        rho_max=rho_max,
        rho_applied=rho_applied,
        selected_phase1_delta_full=selected_delta_total,
        selected_phase1_delta_applied=total_applied_delta,
        selected_aggregate_mass_deficit=selected_aggregate_deficit,
        selected_aggregate_mass_repaired=selected_aggregate_repaired,
        donor_phase1_capacity=donor_capacity,
        donor_surplus_scale=donor_surplus_scale,
        individual_rho_cap=individual_rho_cap,
    )


def remaining_deficit_stats(merged: pd.DataFrame, frame: pd.DataFrame) -> tuple[int, float, int]:
    by_domain = frame.set_index("domain")["simulated_epochs"].sort_index()
    single = merged.set_index("domain")["simulated_epochs_single"].sort_index()
    original = merged.set_index("domain")["simulated_epochs_two_phase"].sort_index()
    remaining = single - by_domain
    original_deficit = single - original
    underexposed = original_deficit > 1e-8
    worsened = ((by_domain + 1e-9) < original) & underexposed
    return (
        int((remaining > 1e-8).sum()),
        float(remaining.max()),
        int(worsened.sum()),
    )


def summarize_candidate(
    repair: Phase1Repair, merged: pd.DataFrame, output_csv: Path
) -> CandidateSummary:
    frame = repair.frame
    positive_remaining, max_remaining, worsened_count = remaining_deficit_stats(merged, frame)
    repair_fraction = (
        0.0
        if repair.selected_aggregate_mass_deficit == 0
        else repair.selected_aggregate_mass_repaired / repair.selected_aggregate_mass_deficit
    )
    return CandidateSummary(
        mixture_id=repair.name,
        objective=repair.objective,
        donor_policy=repair.donor_policy,
        selected_domain_count=len(repair.selected_domains),
        selected_domains=";".join(repair.selected_domains),
        rho_max=repair.rho_max,
        rho_applied=repair.rho_applied,
        selected_aggregate_mass_deficit=repair.selected_aggregate_mass_deficit,
        selected_aggregate_mass_repaired=repair.selected_aggregate_mass_repaired,
        selected_deficit_repair_fraction=repair_fraction,
        donor_phase1_capacity=repair.donor_phase1_capacity,
        donor_surplus_scale=repair.donor_surplus_scale,
        individual_rho_cap=repair.individual_rho_cap,
        max_simulated_epochs=float(frame["simulated_epochs"].max()),
        q95_simulated_epochs=q95(frame["simulated_epochs"]),
        max_phase_0_epochs=float(frame["phase_0_epoch_multiplier"].max()),
        max_phase_1_epochs=float(frame["phase_1_epoch_multiplier"].max()),
        phase0_sum=float(frame["phase_0_weight"].sum()),
        phase1_sum=float(frame["phase_1_weight"].sum()),
        aggregate_sum=float(frame["aggregate_weight"].sum()),
        min_phase_weight=float(frame[["phase_0_weight", "phase_1_weight"]].min().min()),
        remaining_positive_deficit_count=positive_remaining,
        max_remaining_deficit=max_remaining,
        worsened_original_underexposed_count=worsened_count,
        output_csv=str(output_csv),
    )


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


def plot_phase1_repair(
    *,
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


def objective_repairs(task: str) -> tuple[list[CandidateSummary], tuple[str, go.Figure]]:
    spec = objective_spec(task)
    _, _, merged = comparison_frames(spec)
    merged = merged.sort_values("domain").reset_index(drop=True)
    merged["exposure_deficit_single_minus_two"] = (
        merged["simulated_epochs_single"] - merged["simulated_epochs_two_phase"]
    )
    selected_domains = TARGETED_REPAIR_DOMAINS[task]
    repairs = [
        phase1_repair(
            merged,
            objective=task,
            selected_domains=selected_domains,
            donor_policy="surplus",
        ),
        phase1_repair(
            merged,
            objective=task,
            selected_domains=selected_domains,
            donor_policy="any_donor",
        ),
    ]
    summaries: list[CandidateSummary] = []
    named_frames = [
        ("two_phase_original", "original two-phase DSP", original_frame(merged, "two_phase")),
        *[(repair.name.replace(f"dsp_{OBJECTIVE_TO_KEY[task]}_", ""), repair.label, repair.frame) for repair in repairs],
        ("single_phase_reference", "single-phase DSP reference", original_frame(merged, "single")),
    ]
    for repair in repairs:
        output_csv = MIXTURE_DIR / f"{repair.name}.csv"
        launch_ready_frame(repair.frame).to_csv(output_csv, index=False)
        repair.frame.to_csv(OUTPUT_DIR / f"{repair.name}_diagnostic.csv", index=False)
        summaries.append(summarize_candidate(repair, merged, output_csv))
    long_df = long_frame(named_frames)
    objective_key = OBJECTIVE_TO_KEY[task]
    long_df.to_csv(OUTPUT_DIR / f"{objective_key}_phase1_repair_long.csv", index=False)
    order_domains = (
        merged.sort_values("exposure_deficit_single_minus_two", ascending=True)["domain"]
        .tolist()
    )
    figure = plot_phase1_repair(
        title=f"{task} DSP phase-1-only exposure repair",
        long_df=long_df,
        order_domains=order_domains,
        subtitle=(
            "Phase 0 fixed; selected deficits repaired through phase 1 up to feasible rho_max."
        ),
    )
    figure.write_html(
        OUTPUT_DIR / f"{objective_key}_phase1_repair.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )
    return summaries, (task, figure)


def write_index(figures: list[tuple[str, go.Figure]], manifest: pd.DataFrame) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>DSP phase-1-only repair validation mixtures</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;color:#172033}"
        "table{border-collapse:collapse;margin:16px 0;width:100%;font-size:13px}"
        "th,td{border:1px solid #d9e0ea;padding:6px 8px;text-align:left;vertical-align:top}"
        "th{background:#eef3f8} code{background:#eef3f8;padding:2px 4px;border-radius:4px}</style>",
        "</head><body>",
        "<h1>DSP phase-1-only exposure repair</h1>",
        "<p>Phase 0 is fixed at the original two-phase DSP mixture. Selected aggregate-exposure deficits "
        "are repaired through phase 1 only, up to each donor policy's feasible rho_max. The surplus-donor "
        "policy protects non-selected buckets down to the single-phase aggregate reference; the any-donor "
        "policy is aggressive and may zero non-selected phase-1 mass.</p>",
        manifest.to_html(index=False, escape=True),
    ]
    include_js: str | bool = "cdn"
    for title, figure in figures:
        parts.append(f"<h2>{title}</h2>")
        parts.append(pio.to_html(figure, include_plotlyjs=include_js, full_html=False, config=PLOT_CONFIG))
        include_js = False
    parts.append("</body></html>")
    (OUTPUT_DIR / "dsp_phase1_repair_validation_mixtures.html").write_text("\n".join(parts))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[CandidateSummary] = []
    figures: list[tuple[str, go.Figure]] = []
    for task in ["Uncheatable BPB", "Table-9 Macro BPB"]:
        task_summaries, task_figure = objective_repairs(task)
        summaries.extend(task_summaries)
        figures.append(task_figure)
    manifest = pd.DataFrame([asdict(summary) for summary in summaries])
    manifest.to_csv(OUTPUT_DIR / "validation_mixture_manifest.csv", index=False)
    (OUTPUT_DIR / "validation_mixture_manifest.json").write_text(
        json.dumps([asdict(summary) for summary in summaries], indent=2)
    )
    write_index(figures, manifest)
    print(manifest.to_string(index=False))
    print(f"wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
