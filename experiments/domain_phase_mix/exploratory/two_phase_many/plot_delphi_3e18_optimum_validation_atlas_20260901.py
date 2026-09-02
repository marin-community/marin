# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
# ]
# ///

"""Build a measured-outcome atlas for Delphi 3e18 optimum validations."""

from __future__ import annotations

import hashlib
import html
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[4]
REFERENCE_OUTPUTS = REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs"
OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_3e18_optimum_validation_atlas_20260901"
HISTORICAL_REGISTRY = REFERENCE_OUTPUTS / "delphi_3e18_append_only_heldouts_20260714/heldout_current.csv"
COMPOSITE_RESULTS = REFERENCE_OUTPUTS / "delphi_3e18_composite_proposal_validation_results_20260727/observed_results.csv"
TV_LADDER_RESULTS = REFERENCE_OUTPUTS / "delphi_3e18_uncheatable_phase_tv_ladder_results_20260727/observed_results.csv"
DSP_CAP_RESULTS = REFERENCE_OUTPUTS / "delphi_one_phase_dsp_epoch_cap_sweep_20260828/measured_results.csv"
AGGREGATE_V_CAP_RESULTS = (
    REFERENCE_OUTPUTS / "delphi_one_phase_surrogate_challenger_validations_20260831/measured_results.csv"
)

MAIN_UNCHEATABLE_LIMIT = 1.10
MAIN_TABLE9_LIMIT = 1.20
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}


@dataclass(frozen=True)
class SweepSpec:
    label: str
    family: str
    completed: date
    expected_rows: int
    kind: str = "validation"


HISTORICAL_SWEEPS: dict[str, SweepSpec] = {
    "delphi_baseline_mixtures_issue6607_20260623": SweepSpec(
        "Proportional + UniMax controls", "Controls", date(2026, 6, 23), 2, "control"
    ),
    "delphi_uncheatable_optimized_mixtures_20260625": SweepSpec(
        "Initial optimized mixtures", "Foundational DSP / OLMix", date(2026, 6, 25), 33
    ),
    "delphi_table9_optimized_mixtures_20260626": SweepSpec(
        "Initial Table-9 optima", "Foundational DSP / OLMix", date(2026, 6, 26), 2
    ),
    "delphi_table9_dsp_kl_sweep_3e18_20260627": SweepSpec(
        "Table-9 DSP KL", "Foundational DSP / OLMix", date(2026, 6, 27), 4
    ),
    "delphi_table9_dsp_validation_mixtures_3e18_20260628": SweepSpec(
        "Table-9 DSP validation", "Foundational DSP / OLMix", date(2026, 6, 28), 11
    ),
    "delphi_one_phase_table9_dsp_kl_sweep_3e18_20260628": SweepSpec(
        "One-phase Table-9 DSP KL", "Foundational DSP / OLMix", date(2026, 6, 28), 5
    ),
    "delphi_one_phase_table9_olmix_scaling_20260628": SweepSpec(
        "One-phase Table-9 OLMix", "Foundational DSP / OLMix", date(2026, 6, 28), 1
    ),
    "delphi_table9_adaptive_shrinkage_mixtures_3e18_20260628": SweepSpec(
        "Table-9 adaptive shrinkage", "Table-9 local refinements", date(2026, 6, 28), 8
    ),
    "delphi_one_phase_uncheatable_validation_20260629": SweepSpec(
        "One-phase Uncheatable", "Foundational DSP / OLMix", date(2026, 6, 29), 2
    ),
    "delphi_table9_phase_split_dsp_validation_20260630": SweepSpec(
        "Phase-split DSP", "Foundational DSP / OLMix", date(2026, 6, 30), 5
    ),
    "delphi_dsp_exposure_repair_validation_20260702": SweepSpec(
        "DSP exposure repair", "Foundational DSP / OLMix", date(2026, 7, 2), 4
    ),
    "delphi_dsp_canonical_bowl_validation_20260703": SweepSpec(
        "Canonical DSP bowl", "Foundational DSP / OLMix", date(2026, 7, 3), 10
    ),
    "delphi_dsp_support_aware_validation_20260703": SweepSpec(
        "Support-aware DSP", "Foundational DSP / OLMix", date(2026, 7, 3), 1
    ),
    "delphi_gamma_capped_bowl_validation_20260704": SweepSpec(
        "Gamma-capped bowl", "Foundational DSP / OLMix", date(2026, 7, 4), 12
    ),
    "delphi_augmented_profile_validation_20260705": SweepSpec(
        "Augmented profile", "Foundational DSP / OLMix", date(2026, 7, 5), 3
    ),
    "delphi_one_phase_olmix_kl_sweep_3e18_20260705": SweepSpec(
        "One-phase OLMix KL", "Foundational DSP / OLMix", date(2026, 7, 5), 16
    ),
    "delphi_sufficiency_floored_validation_20260705": SweepSpec(
        "Sufficiency-floored", "Foundational DSP / OLMix", date(2026, 7, 5), 8
    ),
    "delphi_winner_neighborhood_validation_20260705": SweepSpec(
        "Winner neighborhood", "Foundational DSP / OLMix", date(2026, 7, 5), 8
    ),
    "delphi_table9_controlled_tilt_validation_20260705": SweepSpec(
        "Controlled tilt", "Table-9 local refinements", date(2026, 7, 5), 3
    ),
    "delphi_table9_value_room_20260705": SweepSpec(
        "Table-9 value room", "Table-9 local refinements", date(2026, 7, 5), 3
    ),
    "delphi_table9_value_room_repeats_20260705": SweepSpec(
        "Value-room repeats", "Table-9 local refinements", date(2026, 7, 5), 4
    ),
    "delphi_table9_fresh_anneal_validation_20260705": SweepSpec(
        "Fresh anneal", "Table-9 local refinements", date(2026, 7, 5), 4
    ),
    "delphi_table9_fresh_anneal_v2_20260705": SweepSpec(
        "Fresh anneal v2", "Table-9 local refinements", date(2026, 7, 5), 6
    ),
    "delphi_table9_anneal_repeats_v2_20260705": SweepSpec(
        "Anneal repeats v2", "Table-9 local refinements", date(2026, 7, 5), 4
    ),
    "delphi_sep_lf_kl_sweep_20260707": SweepSpec(
        "Separate-head LF KL", "Separate-head phase models", date(2026, 7, 7), 12
    ),
    "delphi_sep_frontier_tied_validation_20260710": SweepSpec(
        "Separate frontier vs tied", "Separate-head phase models", date(2026, 7, 10), 12
    ),
    "delphi_best_phase_model_validation_20260710": SweepSpec(
        "Best phase-model", "Separate-head phase models", date(2026, 7, 10), 12
    ),
    "delphi_generalized_power_sepheads_reorder_20260710": SweepSpec(
        "Generalized-power reorder", "Separate-head phase models", date(2026, 7, 10), 4
    ),
    "delphi_centered_recency_sepheads_reorder_20260710": SweepSpec(
        "Centered-recency reorder", "Separate-head phase models", date(2026, 7, 10), 4
    ),
    "delphi_symmetric_sepheads_geometry_frontier_3e18_20260711": SweepSpec(
        "Symmetric separate-head geometry", "Separate-head phase models", date(2026, 7, 11), 30
    ),
    "delphi_original_style_matched_sepheads_ablation_3e18_20260712": SweepSpec(
        "Original-style separate-head ablation", "Separate-head phase models", date(2026, 7, 12), 24
    ),
    "hpr_300m_to_3e18_optimum_validation_panel_20260720": SweepSpec(
        "HPR transfer from 300M", "Hierarchical phase replay", date(2026, 7, 20), 62
    ),
    "hpr_3e18_to_3e18_optimum_validation_panel_20260720": SweepSpec(
        "HPR fit at 3e18", "Hierarchical phase replay", date(2026, 7, 20), 62
    ),
    "delphi_compact_optimum_path_validation_panel_20260721": SweepSpec(
        "Compact optimum path", "Compact retained state", date(2026, 7, 21), 15
    ),
    "delphi_compact_sub280_optimum_validation_panel_20260721": SweepSpec(
        "Compact sub-280 learning curve", "Compact retained state", date(2026, 7, 22), 140
    ),
    "delphi-corrective-hpr-280-tied-controls-3e18": SweepSpec(
        "Corrective HPR tied controls", "Hierarchical phase replay", date(2026, 7, 24), 6
    ),
}

POST_REGISTRY_SWEEPS: dict[str, SweepSpec] = {
    "phase_tv_ladder": SweepSpec("Phase-TV ladder", "Phase-specialization probes", date(2026, 7, 27), 27),
    "composite_proposal": SweepSpec("Composite proposal", "Phase-specialization probes", date(2026, 7, 27), 6),
    "shared_shape_dsp_epoch_cap": SweepSpec("Shared-shape DSP epoch cap", "One-phase epoch caps", date(2026, 8, 30), 11),
    "aggregate_v_epoch_cap": SweepSpec("Aggregate-V epoch cap", "One-phase epoch caps", date(2026, 9, 1), 8),
}

FAMILY_ORDER = (
    "Controls",
    "Foundational DSP / OLMix",
    "Table-9 local refinements",
    "Separate-head phase models",
    "Hierarchical phase replay",
    "Compact retained state",
    "Phase-specialization probes",
    "One-phase epoch caps",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_policy_class(value: object) -> str:
    text = str(value)
    if text in {"single_phase_tied", "one_phase", "one-phase"}:
        return "One phase"
    if text in {"two_phase", "two-phase"}:
        return "Two phase"
    return "Control / mixed"


def _load_historical() -> pd.DataFrame:
    source = pd.read_csv(HISTORICAL_REGISTRY)
    selected = source.loc[source["training_series"].isin(HISTORICAL_SWEEPS)].copy()
    observed_counts = selected.groupby("training_series").size().to_dict()
    expected_counts = {key: spec.expected_rows for key, spec in HISTORICAL_SWEEPS.items()}
    if observed_counts != expected_counts:
        raise ValueError(f"Historical sweep inventory changed: observed={observed_counts}, expected={expected_counts}")

    candidate = selected["candidate_id"].fillna(selected["wandb_run_base"]).fillna(selected["wandb_run_name"])
    result = pd.DataFrame(
        {
            "sweep_id": selected["training_series"],
            "candidate_id": candidate.astype(str),
            "objective": selected["objective"].fillna("unspecified").astype(str),
            "policy_class": selected["policy_class"].map(_normalize_policy_class),
            "uncheatable_bpb": pd.to_numeric(selected["uncheatable_bpb"], errors="raise"),
            "table9_macro_bpb": pd.to_numeric(selected["table9_macro_bpb"], errors="raise"),
            "sweep_coordinate": np.nan,
            "coordinate_label": "",
            "wandb_url": selected["wandb_url"].fillna("").astype(str),
            "source_csv": str(HISTORICAL_REGISTRY.relative_to(REPO_ROOT)),
            "source_row": selected.index,
        }
    )
    return result


def _load_post_registry() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    tv = pd.read_csv(TV_LADDER_RESULTS)
    frames.append(
        pd.DataFrame(
            {
                "sweep_id": "phase_tv_ladder",
                "candidate_id": tv["candidate_id"],
                "objective": "uncheatable",
                "policy_class": "Two phase",
                "uncheatable_bpb": tv["uncheatable_bpb"],
                "table9_macro_bpb": tv["table9_macro_bpb"],
                "sweep_coordinate": np.where(tv["sign"].eq("minus"), -tv["phase_tv"], tv["phase_tv"]),
                "coordinate_label": "signed phase TV",
                "wandb_url": tv["training_wandb_url"],
                "source_csv": str(TV_LADDER_RESULTS.relative_to(REPO_ROOT)),
                "source_row": tv.index,
            }
        )
    )

    composite = pd.read_csv(COMPOSITE_RESULTS)
    sign_coordinate = composite["sign"].map({"minus": -1.0, "center": 0.0, "plus": 1.0})
    frames.append(
        pd.DataFrame(
            {
                "sweep_id": "composite_proposal",
                "candidate_id": composite["candidate_id"],
                "objective": "uncheatable",
                "policy_class": "Two phase",
                "uncheatable_bpb": composite["uncheatable_bpb"],
                "table9_macro_bpb": composite["table9_macro_bpb"],
                "sweep_coordinate": sign_coordinate,
                "coordinate_label": "minus / tied / plus",
                "wandb_url": composite["training_wandb_url"],
                "source_csv": str(COMPOSITE_RESULTS.relative_to(REPO_ROOT)),
                "source_row": composite.index,
            }
        )
    )

    for sweep_id, path in (
        ("shared_shape_dsp_epoch_cap", DSP_CAP_RESULTS),
        ("aggregate_v_epoch_cap", AGGREGATE_V_CAP_RESULTS),
    ):
        cap = pd.read_csv(path)
        frames.append(
            pd.DataFrame(
                {
                    "sweep_id": sweep_id,
                    "candidate_id": cap["candidate_id"],
                    "objective": cap["target"].replace({"uncheatable_bpb": "uncheatable", "table9_macro_bpb": "table9"}),
                    "policy_class": "One phase",
                    "uncheatable_bpb": cap["uncheatable_bpb"],
                    "table9_macro_bpb": cap["table9_macro_bpb"],
                    "sweep_coordinate": cap["epoch_cap"],
                    "coordinate_label": "whole-run epoch cap",
                    "wandb_url": cap["training_wandb_url"],
                    "source_csv": str(path.relative_to(REPO_ROOT)),
                    "source_row": cap.index,
                }
            )
        )

    result = pd.concat(frames, ignore_index=True)
    observed_counts = result.groupby("sweep_id").size().to_dict()
    expected_counts = {key: spec.expected_rows for key, spec in POST_REGISTRY_SWEEPS.items()}
    if observed_counts != expected_counts:
        raise ValueError(f"Post-registry inventory changed: observed={observed_counts}, expected={expected_counts}")
    return result


def load_candidates() -> pd.DataFrame:
    candidates = pd.concat([_load_historical(), _load_post_registry()], ignore_index=True)
    specs = HISTORICAL_SWEEPS | POST_REGISTRY_SWEEPS
    candidates["sweep_label"] = candidates["sweep_id"].map(lambda key: specs[str(key)].label)
    candidates["family"] = candidates["sweep_id"].map(lambda key: specs[str(key)].family)
    candidates["completed"] = candidates["sweep_id"].map(lambda key: specs[str(key)].completed.isoformat())
    candidates["kind"] = candidates["sweep_id"].map(lambda key: specs[str(key)].kind)
    candidates["is_main_range"] = candidates["uncheatable_bpb"].le(MAIN_UNCHEATABLE_LIMIT) & candidates[
        "table9_macro_bpb"
    ].le(MAIN_TABLE9_LIMIT)
    candidates["row_id"] = candidates["sweep_id"] + "::" + candidates["candidate_id"]
    if candidates[["uncheatable_bpb", "table9_macro_bpb"]].isna().any().any():
        raise ValueError("Every included candidate must have both measured endpoint metrics")
    if candidates["row_id"].duplicated().any():
        duplicates = candidates.loc[candidates["row_id"].duplicated(keep=False), "row_id"].tolist()
        raise ValueError(f"Duplicate sweep/candidate rows: {duplicates}")
    return candidates.sort_values(["completed", "sweep_label", "candidate_id"]).reset_index(drop=True)


def summarize_sweeps(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sweep_id, group in candidates.groupby("sweep_id", sort=False):
        best_uncheatable = group.loc[group["uncheatable_bpb"].idxmin()]
        best_table9 = group.loc[group["table9_macro_bpb"].idxmin()]
        rows.append(
            {
                "sweep_id": sweep_id,
                "sweep_label": group.iloc[0]["sweep_label"],
                "family": group.iloc[0]["family"],
                "completed": group.iloc[0]["completed"],
                "kind": group.iloc[0]["kind"],
                "rows": len(group),
                "policy_classes": ", ".join(sorted(group["policy_class"].unique())),
                "objectives": ", ".join(sorted(group["objective"].unique())),
                "best_uncheatable_bpb": float(best_uncheatable["uncheatable_bpb"]),
                "best_uncheatable_candidate": best_uncheatable["candidate_id"],
                "best_table9_macro_bpb": float(best_table9["table9_macro_bpb"]),
                "best_table9_candidate": best_table9["candidate_id"],
                "source_csvs": "; ".join(sorted(group["source_csv"].unique())),
            }
        )
    return pd.DataFrame(rows).sort_values(["completed", "sweep_label"]).reset_index(drop=True)


def pareto_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    ordered = candidates.sort_values(["uncheatable_bpb", "table9_macro_bpb"])
    keep: list[int] = []
    best_table9 = float("inf")
    for index, row in ordered.iterrows():
        table9 = float(row["table9_macro_bpb"])
        if table9 < best_table9:
            keep.append(index)
            best_table9 = table9
    return ordered.loc[keep].sort_values("uncheatable_bpb")


def _family_colors() -> dict[str, str]:
    values = np.linspace(0.05, 0.95, len(FAMILY_ORDER))
    return dict(zip(FAMILY_ORDER, px.colors.sample_colorscale("RdYlGn_r", values), strict=True))


def _hover_customdata(group: pd.DataFrame) -> np.ndarray:
    coordinate_hover = [
        f"<br>{label}: {value:g}" if label and pd.notna(value) else ""
        for label, value in zip(group["coordinate_label"], group["sweep_coordinate"], strict=True)
    ]
    return np.column_stack(
        [
            group["sweep_label"],
            group["candidate_id"],
            group["objective"],
            group["policy_class"],
            group["sweep_coordinate"],
            group["coordinate_label"],
            coordinate_hover,
        ]
    )


def outcome_plane(candidates: pd.DataFrame) -> go.Figure:
    colors = _family_colors()
    figure = go.Figure()
    trace_roles: list[str] = []
    symbols = {"One phase": "circle", "Two phase": "diamond", "Control / mixed": "x"}

    for family in FAMILY_ORDER:
        for policy_class in ("One phase", "Two phase", "Control / mixed"):
            group = candidates.loc[
                candidates["family"].eq(family)
                & candidates["policy_class"].eq(policy_class)
                & candidates["is_main_range"]
            ]
            if group.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=group["uncheatable_bpb"],
                    y=group["table9_macro_bpb"],
                    mode="markers",
                    name=family,
                    legendgroup=family,
                    showlegend=not any(trace.legendgroup == family for trace in figure.data),
                    marker={
                        "color": colors[family],
                        "size": 8 if family != "Controls" else 13,
                        "symbol": symbols[policy_class],
                        "opacity": 0.58 if family != "Controls" else 0.95,
                        "line": {"color": "#17324D", "width": 0.5},
                    },
                    customdata=_hover_customdata(group),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                        "Objective: %{customdata[2]}<br>Policy: %{customdata[3]}<br>"
                        "Uncheatable: %{x:.6f}<br>Table-9: %{y:.6f}<br>"
                        "%{customdata[6]}<extra></extra>"
                    ),
                )
            )
            trace_roles.append("main")

    for family in FAMILY_ORDER:
        group = candidates.loc[candidates["family"].eq(family) & ~candidates["is_main_range"]]
        if group.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=group["uncheatable_bpb"],
                y=group["table9_macro_bpb"],
                mode="markers",
                name=f"{family} edge outliers",
                legendgroup=family,
                showlegend=False,
                visible=False,
                marker={
                    "color": colors[family],
                    "size": 8,
                    "symbol": [symbols[value] for value in group["policy_class"]],
                    "opacity": 0.55,
                },
                customdata=_hover_customdata(group),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                    "Uncheatable: %{x:.6f}<br>Table-9: %{y:.6f}<extra></extra>"
                ),
            )
        )
        trace_roles.append("outlier")

    pareto = pareto_rows(candidates)
    figure.add_trace(
        go.Scatter(
            x=pareto["uncheatable_bpb"],
            y=pareto["table9_macro_bpb"],
            mode="lines+markers",
            name="Measured Pareto frontier",
            legendgroup="pareto",
            line={"color": "#17324D", "width": 2, "dash": "dot", "shape": "hv"},
            marker={"color": "#F8F3E8", "size": 12, "symbol": "star", "line": {"color": "#17324D", "width": 2}},
            customdata=_hover_customdata(pareto),
            hovertemplate=(
                "<b>Pareto: %{customdata[0]}</b><br>%{customdata[1]}<br>"
                "Uncheatable: %{x:.6f}<br>Table-9: %{y:.6f}<extra></extra>"
            ),
        )
    )
    trace_roles.append("pareto")

    main_visibility = [role != "outlier" for role in trace_roles]
    all_visibility = [True for _ in trace_roles]
    figure.update_layout(
        height=760,
        margin={"l": 85, "r": 35, "t": 135, "b": 75},
        paper_bgcolor="#F8F3E8",
        plot_bgcolor="#F8F3E8",
        font={"family": "Avenir Next, sans-serif", "size": 15, "color": "#17324D"},
        legend={"orientation": "h", "y": 1.16, "x": 0, "xanchor": "left", "groupclick": "togglegroup"},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 1,
                "xanchor": "right",
                "y": 1.04,
                "buttons": [
                    {
                        "label": "Main range",
                        "method": "update",
                        "args": [
                            {"visible": main_visibility},
                            {
                                "xaxis.range": [0.975, MAIN_UNCHEATABLE_LIMIT],
                                "yaxis.range": [1.055, MAIN_TABLE9_LIMIT],
                            },
                        ],
                    },
                    {
                        "label": "Include edge outliers",
                        "method": "update",
                        "args": [
                            {"visible": all_visibility},
                            {"xaxis.autorange": True, "yaxis.autorange": True},
                        ],
                    },
                ],
            }
        ],
    )
    figure.update_xaxes(
        title="Uncheatable BPB (lower is better)",
        range=[0.975, MAIN_UNCHEATABLE_LIMIT],
        gridcolor="#DCE5EA",
        zeroline=False,
    )
    figure.update_yaxes(
        title="Native Table-9 macro BPB (lower is better)",
        range=[1.055, MAIN_TABLE9_LIMIT],
        gridcolor="#DCE5EA",
        zeroline=False,
    )
    return figure


def sweep_timeline(candidates: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    colors = _family_colors()
    order = summary["sweep_id"].tolist()
    positions = {sweep_id: position for position, sweep_id in enumerate(order)}
    labels = summary["sweep_label"].tolist()
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Uncheatable BPB", "Native Table-9 macro BPB"),
        horizontal_spacing=0.05,
    )

    plot_rows = candidates.loc[candidates["is_main_range"]].copy()
    within_sweep_rank = plot_rows.groupby("sweep_id").cumcount()
    within_sweep_count = plot_rows.groupby("sweep_id")["sweep_id"].transform("size")
    centered_rank = within_sweep_rank - (within_sweep_count - 1) / 2
    denominator = np.maximum(within_sweep_count - 1, 1)
    plot_rows["timeline_x"] = plot_rows["sweep_id"].map(positions) + 0.44 * centered_rank / denominator

    for column, metric in enumerate(("uncheatable_bpb", "table9_macro_bpb"), start=1):
        for family in FAMILY_ORDER:
            group = plot_rows.loc[plot_rows["family"].eq(family)]
            if group.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=group["timeline_x"],
                    y=group[metric],
                    mode="markers",
                    name=family,
                    legendgroup=family,
                    showlegend=column == 1,
                    marker={"color": colors[family], "size": 7, "opacity": 0.52},
                    customdata=_hover_customdata(group),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                        "Objective: %{customdata[2]}<br>BPB: %{y:.6f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )

        best_indices = candidates.groupby("sweep_id")[metric].idxmin()
        best = candidates.loc[best_indices].copy()
        best["timeline_x"] = best["sweep_id"].map(positions)
        figure.add_trace(
            go.Scatter(
                x=best["timeline_x"],
                y=best[metric],
                mode="markers",
                name="Best in each sweep",
                legendgroup="best",
                showlegend=column == 1,
                marker={
                    "color": "#F8F3E8",
                    "size": 12,
                    "symbol": "diamond",
                    "line": {"color": "#17324D", "width": 2},
                },
                customdata=_hover_customdata(best),
                hovertemplate=(
                    "<b>Best in sweep: %{customdata[0]}</b><br>%{customdata[1]}<br>" "BPB: %{y:.6f}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )

    figure.update_layout(
        width=2500,
        height=820,
        margin={"l": 80, "r": 30, "t": 125, "b": 260},
        paper_bgcolor="#F8F3E8",
        plot_bgcolor="#F8F3E8",
        font={"family": "Avenir Next, sans-serif", "size": 13, "color": "#17324D"},
        legend={"orientation": "h", "y": 1.13, "x": 0.01},
    )
    for column in (1, 2):
        figure.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(order))),
            ticktext=labels,
            tickangle=-55,
            gridcolor="#E7DDD0",
            row=1,
            col=column,
        )
        figure.update_yaxes(title="BPB (lower is better)", gridcolor="#DCE5EA", row=1, col=column)
    return figure


def recent_epoch_cap_plot(candidates: pd.DataFrame) -> go.Figure:
    recent = candidates.loc[candidates["sweep_id"].isin(POST_REGISTRY_SWEEPS)].copy()
    recent = recent.loc[recent["sweep_id"].isin({"shared_shape_dsp_epoch_cap", "aggregate_v_epoch_cap"})]
    colors = _family_colors()
    line_colors = px.colors.sample_colorscale("RdYlGn_r", [0.15, 0.85])
    model_colors = {
        "shared_shape_dsp_epoch_cap": line_colors[0],
        "aggregate_v_epoch_cap": line_colors[1],
    }
    model_labels = {
        "shared_shape_dsp_epoch_cap": "Shared-shape DSP",
        "aggregate_v_epoch_cap": "Aggregate-V",
    }
    target_labels = {"uncheatable": "optimized for Uncheatable", "table9": "optimized for Table-9"}
    target_dashes = {"uncheatable": "solid", "table9": "dash"}

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Measured Uncheatable BPB", "Measured native Table-9 macro BPB"),
        horizontal_spacing=0.1,
    )
    for column, metric in enumerate(("uncheatable_bpb", "table9_macro_bpb"), start=1):
        for sweep_id in model_labels:
            for objective in target_labels:
                group = recent.loc[recent["sweep_id"].eq(sweep_id) & recent["objective"].eq(objective)].sort_values(
                    "sweep_coordinate"
                )
                if group.empty:
                    continue
                label = f"{model_labels[sweep_id]} · {target_labels[objective]}"
                figure.add_trace(
                    go.Scatter(
                        x=group["sweep_coordinate"],
                        y=group[metric],
                        mode="lines+markers",
                        name=label,
                        legendgroup=label,
                        showlegend=column == 1,
                        line={"color": model_colors[sweep_id], "width": 3, "dash": target_dashes[objective]},
                        marker={"size": 9, "line": {"color": colors["Controls"], "width": 0.5}},
                        customdata=_hover_customdata(group),
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>" "Cap: %{x}<br>BPB: %{y:.6f}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=column,
                )
    figure.update_layout(
        height=680,
        margin={"l": 75, "r": 35, "t": 150, "b": 80},
        paper_bgcolor="#F8F3E8",
        plot_bgcolor="#F8F3E8",
        font={"family": "Avenir Next, sans-serif", "size": 15, "color": "#17324D"},
        legend={"orientation": "h", "y": 1.18, "x": 0.5, "xanchor": "center"},
    )
    figure.update_xaxes(title="Whole-run materialized epoch cap", dtick=2, gridcolor="#DCE5EA")
    figure.update_yaxes(title="BPB (lower is better)", gridcolor="#DCE5EA")
    return figure


def _figure_html(figure: go.Figure, *, include_plotlyjs: bool) -> str:
    return pio.to_html(
        figure,
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        config=PLOT_CONFIG,
    )


def _summary_table(summary: pd.DataFrame) -> str:
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.completed))}</td>"
            f"<td><strong>{html.escape(str(row.sweep_label))}</strong><br><span>{html.escape(str(row.family))}</span></td>"
            f"<td>{row.rows}</td>"
            f"<td>{html.escape(str(row.policy_classes))}</td>"
            f"<td>{row.best_uncheatable_bpb:.6f}<br><span>{html.escape(str(row.best_uncheatable_candidate))}</span></td>"
            f"<td>{row.best_table9_macro_bpb:.6f}<br><span>{html.escape(str(row.best_table9_candidate))}</span></td>"
            "</tr>"
        )
    return "\n".join(rows)


def write_artifact() -> None:
    candidates = load_candidates()
    summary = summarize_sweeps(candidates)
    pareto = pareto_rows(candidates)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates_path = OUTPUT_DIR / "observed_candidates.csv"
    summary_path = OUTPUT_DIR / "sweep_summary.csv"
    candidates.to_csv(candidates_path, index=False)
    summary.to_csv(summary_path, index=False)

    best_uncheatable = candidates.loc[candidates["uncheatable_bpb"].idxmin()]
    best_table9 = candidates.loc[candidates["table9_macro_bpb"].idxmin()]
    validation_summary = summary.loc[summary["kind"].eq("validation")]
    main_range_rows = int(candidates["is_main_range"].sum())
    edge_outlier_rows = len(candidates) - main_range_rows
    provenance = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "Measured Delphi 3e18 optimum-validation sweeps with both Uncheatable and native Table-9 endpoints",
        "validation_sweeps": len(validation_summary),
        "control_series": int(summary["kind"].eq("control").sum()),
        "candidate_rows": len(candidates),
        "main_range_rows": main_range_rows,
        "edge_outlier_rows": edge_outlier_rows,
        "pareto_rows": len(pareto),
        "sources": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in (
                HISTORICAL_REGISTRY,
                COMPOSITE_RESULTS,
                TV_LADDER_RESULTS,
                DSP_CAP_RESULTS,
                AGGREGATE_V_CAP_RESULTS,
            )
        },
    }
    provenance_path = OUTPUT_DIR / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    outcome_html = _figure_html(outcome_plane(candidates), include_plotlyjs=True)
    timeline_html = _figure_html(sweep_timeline(candidates, summary), include_plotlyjs=False)
    epoch_cap_html = _figure_html(recent_epoch_cap_plot(candidates), include_plotlyjs=False)

    index = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Delphi 3e18 optimum-validation atlas</title>
  <style>
    :root {{ --ink: #17324d; --muted: #62758a; --paper: #f8f3e8; --panel: #fffdf7; --accent: #d95d39; --rule: #dacdbc; }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: "Avenir Next", "Gill Sans", sans-serif;
    }}
    main {{ width: min(1540px, 96vw); margin: 0 auto; padding: 56px 0 80px; }}
    h1, h2 {{ margin: 0; font-family: Georgia, "Times New Roman", serif; letter-spacing: -0.025em; }}
    h1 {{ max-width: 1100px; font-size: clamp(2.7rem, 6vw, 5.4rem); line-height: 0.98; }}
    h2 {{ font-size: clamp(2rem, 4vw, 3.2rem); margin-top: 64px; }}
    p {{ max-width: 1050px; font-size: 1.14rem; line-height: 1.65; }}
    .lede {{ color: var(--muted); font-size: 1.28rem; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 34px 0; }}
    .card {{ min-height: 120px; padding: 20px 22px; border: 1px solid var(--rule); background: var(--panel); }}
    .card b {{ display: block; font-family: Georgia, serif; font-size: 2.2rem; font-weight: 500; }}
    .card span {{ color: var(--muted); line-height: 1.35; }}
    .callout {{ border-left: 8px solid var(--accent); background: var(--panel); padding: 18px 24px; margin: 30px 0; }}
    .plot {{ margin: 22px 0 42px; border: 1px solid var(--rule); background: var(--panel); overflow: hidden; }}
    .scroll {{ overflow-x: auto; border: 1px solid var(--rule); background: var(--panel); margin: 22px 0 42px; }}
    .scroll-inner {{ min-width: 2500px; }}
    .downloads a {{ color: var(--ink); font-weight: 650; margin-right: 18px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); font-size: 0.92rem; }}
    th, td {{ border-bottom: 1px solid var(--rule); padding: 11px 12px; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #eee5d7; z-index: 1; }}
    td span {{ color: var(--muted); font-size: 0.8rem; overflow-wrap: anywhere; }}
    .table-wrap {{ max-height: 760px; overflow: auto; border: 1px solid var(--rule); margin-top: 22px; }}
    code {{ background: #eee5d7; padding: 0.12rem 0.32rem; }}
    @media (max-width: 850px) {{
      .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      main {{ width: 94vw; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>Delphi 3e18 optimum-validation atlas</h1>
  <p class="lede">
    Every dedicated optimum-validation sweep with locally frozen measurements on both Uncheatable and native
    Table-9, from the first June controls through the newest whole-run epoch-cap candidates.
  </p>
  <div class="cards">
    <div class="card">
      <b>{len(validation_summary)}</b><span>validation sweeps, plus one baseline-control series</span>
    </div>
    <div class="card"><b>{len(candidates)}</b><span>trained candidates with both endpoint metrics</span></div>
    <div class="card">
      <b>{best_uncheatable.uncheatable_bpb:.6f}</b>
      <span>lowest measured Uncheatable BPB<br>{html.escape(str(best_uncheatable.sweep_label))}</span>
    </div>
    <div class="card">
      <b>{best_table9.table9_macro_bpb:.6f}</b>
      <span>lowest measured Table-9 macro BPB<br>{html.escape(str(best_table9.sweep_label))}</span>
    </div>
  </div>
  <div class="callout">
    <strong>Scope.</strong> This is an outcome inventory, not a pooled statistical comparison: sweeps differ in
    objective, model class, candidate count, and replication. Broad fit swarms, mechanism surfaces, prefix/branch
    searches, and predicted-but-untrained optima are excluded. The {edge_outlier_rows} poor compact-subset
    extrapolations outside the main plotting range remain available behind the outlier toggle and in the CSV.
  </div>
  <p class="downloads">
    <a href="observed_candidates.csv">Download normalized candidates</a>
    <a href="sweep_summary.csv">Download sweep summary</a>
    <a href="provenance.json">Inspect source hashes</a>
  </p>

  <h2>Outcome plane</h2>
  <p>
    Each point is one trained policy. Circles are one-phase policies, diamonds are two-phase policies, and crosses
    are baseline controls or mixed control panels. The dotted lower-left envelope is the measured Pareto frontier.
    Use the button above the chart to reveal edge outliers.
  </p>
  <div class="plot">{outcome_html}</div>

  <h2>Sweep-by-sweep overlay</h2>
  <p>
    All candidates are spread narrowly around their sweep label; the outlined diamond marks the best measured
    value within that sweep for the displayed metric. Scroll horizontally to retain readable labels rather than
    compressing forty series into one viewport.
  </p>
  <div class="scroll"><div class="scroll-inner">{timeline_html}</div></div>

  <h2>Newest epoch-cap sweeps</h2>
  <p>
    The two recent one-phase model families share the same whole-run cap axis. Solid paths were optimized for
    Uncheatable and dashed paths for Table-9; both measured endpoints are shown, so cross-objective tradeoffs remain
    visible.
  </p>
  <div class="plot">{epoch_cap_html}</div>

  <h2>Provenance by sweep</h2>
  <p>
    The minima below are descriptive best-observed rows, not uncertainty-adjusted winners. Candidate IDs and source
    files are retained in the downloadable tables.
  </p>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Date</th><th>Sweep</th><th>Rows</th><th>Policy class</th>
          <th>Best Uncheatable</th><th>Best Table-9</th>
        </tr>
      </thead>
      <tbody>{_summary_table(summary)}</tbody>
    </table>
  </div>
</main>
</body>
</html>
"""
    (OUTPUT_DIR / "index.html").write_text(index)


if __name__ == "__main__":
    write_artifact()
