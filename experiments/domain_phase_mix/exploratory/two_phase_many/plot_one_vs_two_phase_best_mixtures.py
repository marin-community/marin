# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "plotly"]
# ///
"""Compare best one-phase and two-phase mixture candidates.

This script visualizes, for the current best known one-phase and two-phase
candidate in each task/method cell, how weights differ by phase and in the
phase-aggregated mixture.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


REPO_ROOT = Path(__file__).resolve().parents[4]
REFERENCE_OUTPUTS = (
    REPO_ROOT / "experiments" / "domain_phase_mix" / "exploratory" / "two_phase_many" / "reference_outputs"
)
OUTPUT_DIR = REFERENCE_OUTPUTS / "one_vs_two_phase_best_mixture_comparison_20260701"
PHASE_0_FRACTION = 0.8
PHASE_1_FRACTION = 0.2
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class MixtureSpec:
    task: str
    method: str
    phase_family: str
    label: str
    source_path: Path
    schedule: str | None = None
    expected_table9_bpb_3e18: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class ComparisonSpec:
    task: str
    method: str
    single: MixtureSpec
    two_phase: MixtureSpec


COMPARISONS = [
    ComparisonSpec(
        task="Uncheatable BPB",
        method="OLMix",
        single=MixtureSpec(
            task="Uncheatable BPB",
            method="OLMix",
            phase_family="one_phase",
            label="one-phase OLMix, delta=0.01, KL=0.05, cap=4",
            source_path=REFERENCE_OUTPUTS
            / "one_phase_uncheatable_validation_mixtures_300m_20260629"
            / "olmix_onephase_uncheatable_d001_kl005_cap4.csv",
            notes="single-simplex tied phases; fit target eval/uncheatable_eval/bpb",
        ),
        two_phase=MixtureSpec(
            task="Uncheatable BPB",
            method="OLMix",
            phase_family="two_phase",
            label="two-phase OLMix, delta=0.01, KL=0.05, cap=4",
            source_path=REFERENCE_OUTPUTS
            / "scaling_validation_mixture_candidates_20260625"
            / "uncheatable_scaling_validation_final_three_mixtures.csv",
            schedule="OLMix delta=0.01 KL=0.05 cap=4",
            notes="two-phase adapted OLMix used in uncheatable scaling validation",
        ),
    ),
    ComparisonSpec(
        task="Uncheatable BPB",
        method="DSP",
        single=MixtureSpec(
            task="Uncheatable BPB",
            method="DSP",
            phase_family="one_phase",
            label="one-phase effective-exposure DSP, KL=0.1",
            source_path=REFERENCE_OUTPUTS
            / "one_phase_uncheatable_validation_mixtures_300m_20260629"
            / "dsp_onephase_effexp_uncheatable_kl0p1.csv",
            notes="single-simplex tied phases; fit target eval/uncheatable_eval/bpb",
        ),
        two_phase=MixtureSpec(
            task="Uncheatable BPB",
            method="DSP",
            phase_family="two_phase",
            label="two-phase effective-exposure DSP, KL=0.1",
            source_path=REFERENCE_OUTPUTS
            / "scaling_validation_mixture_candidates_20260625"
            / "uncheatable_scaling_validation_final_three_mixtures.csv",
            schedule="Effective-exposure DSP KL=0.1",
            notes="two-phase effective-exposure DSP used in uncheatable scaling validation",
        ),
    ),
    ComparisonSpec(
        task="Table-9 Macro BPB",
        method="OLMix",
        single=MixtureSpec(
            task="Table-9 Macro BPB",
            method="OLMix",
            phase_family="one_phase",
            label="one-phase OLMix, delta=0.01, KL=0.05, cap=4",
            source_path=REFERENCE_OUTPUTS
            / "olmo_base_easy_one_phase_model_sweeps_300m_20260628"
            / "olmix_one_phase_cap4_delta0p01_kl0p05"
            / "proposed_mixture_weights.csv",
            expected_table9_bpb_3e18=1.081359,
            notes="single-simplex tied phases; paper-faithful OLMix objective",
        ),
        two_phase=MixtureSpec(
            task="Table-9 Macro BPB",
            method="OLMix",
            phase_family="two_phase",
            label="two-phase OLMix, delta=0.01, KL=0.05, cap=4",
            source_path=REFERENCE_OUTPUTS
            / "olmo_base_easy_paper_faithful_olmix_300m_20260625"
            / "two_phase_adapted_delta_0p01"
            / "proposed_mixture_weights.csv",
            expected_table9_bpb_3e18=1.102655,
            notes="two-phase adapted OLMix; same componentwise paper-faithful fit",
        ),
    ),
    ComparisonSpec(
        task="Table-9 Macro BPB",
        method="DSP",
        single=MixtureSpec(
            task="Table-9 Macro BPB",
            method="DSP",
            phase_family="one_phase",
            label="one-phase effective-exposure DSP, KL=0.1",
            source_path=REFERENCE_OUTPUTS
            / "olmo_base_easy_one_phase_model_sweeps_300m_20260628"
            / "dsp_one_phase_effexp_linear_reg0p0001_kl0p1"
            / "proposed_mixture_weights.csv",
            expected_table9_bpb_3e18=1.070728,
            notes="single-simplex tied phases; effective-exposure DSP",
        ),
        two_phase=MixtureSpec(
            task="Table-9 Macro BPB",
            method="DSP",
            phase_family="two_phase",
            label="two-phase split-saturation DSP, L2=0.01, KL=0.3",
            source_path=REFERENCE_OUTPUTS
            / "table9_dsp_phase_functional_form_20260630"
            / "validation_mixtures"
            / "dsp_split_table9_l2_0p01_kl0p3.csv",
            expected_table9_bpb_3e18=1.085229,
            notes="best validated split/two-phase DSP candidate as of 2026-07-01",
        ),
    ),
]


def clean_domain(domain: str) -> str:
    return (
        domain.replace("dolma3_cc/", "cc/")
        .replace("dolmino_", "")
        .replace("dolma3_", "")
        .replace("_and_", "/")
        .replace("_", " ")
    )


def domain_group(domain: str) -> str:
    if domain.startswith("dolma3_cc/"):
        return "CC"
    return "Non-CC"


def read_mixture(spec: MixtureSpec) -> pd.DataFrame:
    if not spec.source_path.exists():
        raise FileNotFoundError(spec.source_path)
    frame = pd.read_csv(spec.source_path)
    if spec.schedule is not None:
        if "schedule" not in frame.columns:
            raise ValueError(f"{spec.source_path} has no schedule column")
        frame = frame[frame["schedule"] == spec.schedule].copy()
        if frame.empty:
            raise ValueError(f"schedule {spec.schedule!r} not found in {spec.source_path}")
    rename = {
        "phase0_weight": "phase_0_weight",
        "phase1_weight": "phase_1_weight",
        "phase0_epochs": "phase_0_epoch_multiplier",
        "phase1_epochs": "phase_1_epoch_multiplier",
    }
    frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})
    required = ["domain", "phase_0_weight", "phase_1_weight", "aggregate_weight"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{spec.source_path} missing columns: {missing}")
    frame = frame.copy()
    frame["domain_group"] = frame["domain"].map(domain_group)
    frame["domain_short"] = frame["domain"].map(clean_domain)
    if "phase_0_epoch_multiplier" not in frame.columns:
        frame["phase_0_epoch_multiplier"] = float("nan")
    if "phase_1_epoch_multiplier" not in frame.columns:
        frame["phase_1_epoch_multiplier"] = float("nan")
    if "simulated_epochs" not in frame.columns:
        frame["simulated_epochs"] = (
            PHASE_0_FRACTION * frame["phase_0_epoch_multiplier"]
            + PHASE_1_FRACTION * frame["phase_1_epoch_multiplier"]
        )
    phase_sums = frame[["phase_0_weight", "phase_1_weight"]].sum()
    aggregate_sum = frame["aggregate_weight"].sum()
    if (phase_sums.sub(1.0).abs() > 1e-5).any() or abs(aggregate_sum - 1.0) > 1e-5:
        raise ValueError(
            f"{spec.label} has invalid sums: phase_sums={phase_sums.to_dict()} "
            f"aggregate_sum={aggregate_sum}"
        )
    aggregate_from_phases = (
        PHASE_0_FRACTION * frame["phase_0_weight"] + PHASE_1_FRACTION * frame["phase_1_weight"]
    )
    max_aggregate_error = float((frame["aggregate_weight"] - aggregate_from_phases).abs().max())
    if max_aggregate_error > 1e-5:
        raise ValueError(
            f"{spec.label} aggregate weights are inconsistent with "
            f"{PHASE_0_FRACTION}/{PHASE_1_FRACTION} phases; max error={max_aggregate_error}"
        )
    return frame


def comparison_frames(spec: ComparisonSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    single = read_mixture(spec.single)
    two_phase = read_mixture(spec.two_phase)
    merged = single.merge(
        two_phase,
        on="domain",
        suffixes=("_single", "_two_phase"),
        validate="one_to_one",
    )
    merged["domain_short"] = merged["domain"].map(clean_domain)
    merged["domain_group"] = merged["domain"].map(domain_group)
    merged["aggregate_delta_two_minus_single"] = (
        merged["aggregate_weight_two_phase"] - merged["aggregate_weight_single"]
    )
    merged["aggregate_exposure_delta_two_minus_single"] = (
        merged["simulated_epochs_two_phase"] - merged["simulated_epochs_single"]
    )
    merged["aggregate_abs_delta"] = merged["aggregate_delta_two_minus_single"].abs()
    merged["phase_gap_single"] = (
        merged["phase_1_weight_single"] - merged["phase_0_weight_single"]
    )
    merged["phase_gap_two_phase"] = (
        merged["phase_1_weight_two_phase"] - merged["phase_0_weight_two_phase"]
    )
    merged["phase_gap_delta"] = merged["phase_gap_two_phase"] - merged["phase_gap_single"]
    return single, two_phase, merged


def tv_distance(left: pd.Series, right: pd.Series) -> float:
    return float(0.5 * (left - right).abs().sum())


def summarize_comparison(spec: ComparisonSpec, merged: pd.DataFrame) -> dict[str, object]:
    top_up = (
        merged.sort_values("aggregate_delta_two_minus_single", ascending=False)
        .head(5)["domain"]
        .tolist()
    )
    top_down = (
        merged.sort_values("aggregate_delta_two_minus_single", ascending=True)
        .head(5)["domain"]
        .tolist()
    )
    return {
        "comparison": f"{spec.task} / {spec.method}",
        "task": spec.task,
        "method": spec.method,
        "single_label": spec.single.label,
        "two_phase_label": spec.two_phase.label,
        "single_table9_bpb_3e18_if_known": spec.single.expected_table9_bpb_3e18,
        "two_phase_table9_bpb_3e18_if_known": spec.two_phase.expected_table9_bpb_3e18,
        "aggregate_tv_single_vs_two_phase": tv_distance(
            merged["aggregate_weight_single"], merged["aggregate_weight_two_phase"]
        ),
        "phase0_tv_single_vs_two_phase": tv_distance(
            merged["phase_0_weight_single"], merged["phase_0_weight_two_phase"]
        ),
        "phase1_tv_single_vs_two_phase": tv_distance(
            merged["phase_1_weight_single"], merged["phase_1_weight_two_phase"]
        ),
        "two_phase_phase_tv": tv_distance(
            merged["phase_0_weight_two_phase"], merged["phase_1_weight_two_phase"]
        ),
        "single_phase_phase_tv": tv_distance(
            merged["phase_0_weight_single"], merged["phase_1_weight_single"]
        ),
        "max_single_simulated_epochs": float(merged["simulated_epochs_single"].max()),
        "max_two_phase_simulated_epochs": float(merged["simulated_epochs_two_phase"].max()),
        "top_two_phase_aggregate_up_domains": "; ".join(top_up),
        "top_two_phase_aggregate_down_domains": "; ".join(top_down),
    }


def comparison_long(spec: ComparisonSpec, merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        for family in ["single", "two_phase"]:
            for phase in ["phase_0", "phase_1", "aggregate"]:
                weight_column = f"{phase}_weight_{family}" if phase != "aggregate" else f"aggregate_weight_{family}"
                epoch_column = (
                    f"{phase}_epoch_multiplier_{family}"
                    if phase != "aggregate"
                    else f"simulated_epochs_{family}"
                )
                rows.append(
                    {
                        "comparison": f"{spec.task} / {spec.method}",
                        "task": spec.task,
                        "method": spec.method,
                        "family": family,
                        "family_label": "One-phase best" if family == "single" else "Two-phase best",
                        "phase": phase,
                        "domain": row["domain"],
                        "domain_short": row["domain_short"],
                        "domain_group": row["domain_group"],
                        "weight": float(row[weight_column]),
                        "epoch_multiplier": float(row[epoch_column]),
                        "aggregate_delta_two_minus_single": float(
                            row["aggregate_delta_two_minus_single"]
                        ),
                        "aggregate_exposure_delta_two_minus_single": float(
                            row["aggregate_exposure_delta_two_minus_single"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plot_phase_bars(spec: ComparisonSpec, long_df: pd.DataFrame) -> go.Figure:
    comparison = f"{spec.task} / {spec.method}"
    aggregate_order = (
        long_df[long_df["phase"] == "aggregate"]
        .drop_duplicates(["domain", "aggregate_delta_two_minus_single"])
        .sort_values("aggregate_delta_two_minus_single")["domain"]
        .tolist()
    )
    group_order = {"Non-CC": 1, "CC": 2}
    domain_to_y = {domain: clean_domain(domain) for domain in aggregate_order}
    y_order = [domain_to_y[domain] for domain in aggregate_order]
    colors = {
        "single": "#6f8190",
        "two_phase": "#e36f2c",
    }
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
    panel_specs = [
        ("phase_0", "weight", "mixture weight"),
        ("phase_1", "weight", "mixture weight"),
        ("aggregate", "weight", "mixture weight"),
        ("aggregate", "epoch_multiplier", "realized simulated epochs"),
    ]
    for col, (phase, value_column, x_title) in enumerate(panel_specs, start=1):
        for family in ["single", "two_phase"]:
            data = long_df[(long_df["phase"] == phase) & (long_df["family"] == family)].copy()
            data["domain_short"] = data["domain"].map(domain_to_y)
            data = data.set_index("domain").loc[aggregate_order].reset_index()
            delta_label = (
                "two - one aggregate exposure delta"
                if value_column == "epoch_multiplier"
                else "two - one aggregate weight delta"
            )
            delta_column = (
                "aggregate_exposure_delta_two_minus_single"
                if value_column == "epoch_multiplier"
                else "aggregate_delta_two_minus_single"
            )
            fig.add_trace(
                go.Bar(
                    x=data[value_column],
                    y=data["domain_short"],
                    orientation="h",
                    name="one-phase best" if family == "single" else "two-phase best",
                    legendgroup=family,
                    showlegend=col == 1,
                    marker_color=colors[family],
                    opacity=0.82 if family == "single" else 0.9,
                    customdata=data[
                        [
                            "domain",
                            "epoch_multiplier",
                            delta_column,
                            "domain_group",
                        ]
                    ],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"{x_title}: %{{x:.5f}}<br>"
                        "epochs: %{customdata[1]:.2f}<br>"
                        f"{delta_label}: %{{customdata[2]:+.5f}}<br>"
                        "group: %{customdata[3]}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
    fig.update_layout(
        title={
            "text": f"{comparison}: best one-phase vs best two-phase mixture",
            "x": 0.5,
            "xanchor": "center",
        },
        barmode="group",
        template="plotly_white",
        height=1180,
        width=2200,
        margin={"l": 210, "r": 40, "t": 105, "b": 95},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.055,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255,255,255,0.92)",
            "bordercolor": "#d9e0ea",
            "borderwidth": 1,
        },
    )
    fig.update_yaxes(categoryorder="array", categoryarray=y_order)
    for col, (_phase, _value_column, x_title) in enumerate(panel_specs, start=1):
        fig.update_xaxes(title_text=x_title, row=1, col=col)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1,
        y=-0.105,
        showarrow=False,
        xanchor="right",
        text=(
            "Domains ordered by aggregate two-phase minus one-phase weight. "
            "Phase fractions are 0.8/0.2; lower BPB is better."
        ),
    )
    # Add unobtrusive separators between Non-CC and CC domain groups if both are present.
    ordered_groups = [domain_group(domain) for domain in aggregate_order]
    if "Non-CC" in ordered_groups and "CC" in ordered_groups:
        first_cc = ordered_groups.index("CC")
        if 0 < first_cc < len(y_order):
            fig.add_hline(
                y=first_cc - 0.5,
                line_width=1,
                line_dash="dot",
                line_color="#9ca3af",
            )
    return fig


def plot_aggregate_scatter(all_deltas: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Uncheatable BPB / OLMix",
            "Uncheatable BPB / DSP",
            "Table-9 Macro BPB / OLMix",
            "Table-9 Macro BPB / DSP",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.11,
    )
    positions = {
        "Uncheatable BPB / OLMix": (1, 1),
        "Uncheatable BPB / DSP": (1, 2),
        "Table-9 Macro BPB / OLMix": (2, 1),
        "Table-9 Macro BPB / DSP": (2, 2),
    }
    for comparison, (row, col) in positions.items():
        data = all_deltas[all_deltas["comparison"] == comparison]
        fig.add_trace(
            go.Scatter(
                x=data["aggregate_weight_single"],
                y=data["aggregate_weight_two_phase"],
                mode="markers+text",
                text=data["domain_short"],
                textposition="top center",
                marker={
                    "size": 9,
                    "color": data["aggregate_delta_two_minus_single"],
                    "colorscale": "RdYlGn_r",
                    "cmid": 0,
                    "showscale": comparison == "Uncheatable BPB / DSP",
                    "colorbar": {"title": "two - one<br>aggregate"},
                },
                customdata=data[["domain", "domain_group"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "one-phase aggregate: %{x:.5f}<br>"
                    "two-phase aggregate: %{y:.5f}<br>"
                    "group: %{customdata[1]}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        limit = max(
            0.01,
            float(data["aggregate_weight_single"].max()),
            float(data["aggregate_weight_two_phase"].max()),
        )
        fig.add_trace(
            go.Scatter(
                x=[0, limit],
                y=[0, limit],
                mode="lines",
                line={"color": "#9ca3af", "dash": "dash"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="one-phase aggregate weight", range=[0, limit * 1.08], row=row, col=col)
        fig.update_yaxes(title_text="two-phase aggregate weight", range=[0, limit * 1.08], row=row, col=col)
    fig.update_layout(
        title="Aggregate mixture comparison: one-phase vs two-phase",
        template="plotly_white",
        height=1050,
        width=1450,
        margin={"l": 80, "r": 80, "t": 85, "b": 60},
    )
    return fig


def write_html_index(figures: list[tuple[str, go.Figure]], summary: pd.DataFrame, output_path: Path) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>One-phase vs two-phase best mixtures</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;color:#172033}"
        "table{border-collapse:collapse;margin:16px 0;width:100%;font-size:13px}"
        "th,td{border:1px solid #d9e0ea;padding:6px 8px;text-align:left;vertical-align:top}"
        "th{background:#eef3f8} code{background:#eef3f8;padding:2px 4px;border-radius:4px}</style>",
        "</head><body>",
        "<h1>Best one-phase vs best two-phase mixtures</h1>",
        "<p>Each panel compares the current best known one-phase and two-phase candidates for the task/method cell. "
        "The one-phase candidates have tied phase weights; the two-phase candidates use the standard 0.8/0.2 phase fractions.</p>",
        summary.to_html(index=False, escape=True),
    ]
    include_js = "cdn"
    for title, fig in figures:
        parts.append(f"<h2>{html.escape(title)}</h2>")
        parts.append(pio.to_html(fig, include_plotlyjs=include_js, full_html=False, config=PLOT_CONFIG))
        include_js = False
    parts.append("</body></html>")
    output_path.write_text("\n".join(parts))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    all_long: list[pd.DataFrame] = []
    all_deltas: list[pd.DataFrame] = []
    figures: list[tuple[str, go.Figure]] = []
    manifest: list[dict[str, object]] = []

    for spec in COMPARISONS:
        _, _, merged = comparison_frames(spec)
        comparison = f"{spec.task} / {spec.method}"
        summary_rows.append(summarize_comparison(spec, merged))
        long_df = comparison_long(spec, merged)
        all_long.append(long_df)
        delta_df = merged.copy()
        delta_df["comparison"] = comparison
        delta_df["task"] = spec.task
        delta_df["method"] = spec.method
        all_deltas.append(delta_df)
        fig = plot_phase_bars(spec, long_df)
        safe_name = (
            comparison.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace("=", "")
        )
        fig_path = OUTPUT_DIR / f"{safe_name}_phase_bars.html"
        fig.write_html(fig_path, include_plotlyjs="cdn", config=PLOT_CONFIG)
        figures.append((comparison, fig))
        manifest.append(
            {
                "comparison": comparison,
                "single_source": str(spec.single.source_path),
                "two_phase_source": str(spec.two_phase.source_path),
                "single_schedule": spec.single.schedule,
                "two_phase_schedule": spec.two_phase.schedule,
                "single_label": spec.single.label,
                "two_phase_label": spec.two_phase.label,
            }
        )

    summary = pd.DataFrame(summary_rows)
    long_all = pd.concat(all_long, ignore_index=True)
    delta_all = pd.concat(all_deltas, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "one_vs_two_phase_best_mixture_summary.csv", index=False)
    long_all.to_csv(OUTPUT_DIR / "one_vs_two_phase_best_mixture_long.csv", index=False)
    delta_all.to_csv(OUTPUT_DIR / "one_vs_two_phase_best_mixture_deltas.csv", index=False)
    (OUTPUT_DIR / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2))

    aggregate_fig = plot_aggregate_scatter(delta_all)
    aggregate_fig.write_html(
        OUTPUT_DIR / "one_vs_two_phase_aggregate_scatter.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )
    figures.insert(0, ("Aggregate one-phase vs two-phase scatter", aggregate_fig))
    write_html_index(figures, summary, OUTPUT_DIR / "one_vs_two_phase_best_mixtures.html")
    print(f"Wrote {OUTPUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
