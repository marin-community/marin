# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Audit empirical phase-tied restrictions of two-phase surrogate fits.

For each panel with a separately fitted one-phase model, evaluate both that fit
and the corresponding two-phase fit on phase-tied observed policies. This
distinguishes an algebraically valid restriction from an empirically
transferable one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717"
DEFAULT_DASHBOARD = RESEARCH_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "phase_tied_restriction_audit"
PANELS = (
    ("300m", "uncheatable"),
    ("300m", "table9"),
    ("starcoder_cosine", "starcoder_bpb"),
    ("starcoder_wsd80", "starcoder_bpb"),
)
MECHANISTIC_MODELS = (
    "canonical",
    "effective_exposure",
    "effective_exposure_geometry",
    "separate_heads",
    "grp",
    "compact_retained_state",
    "bucket_family_grp",
    "bucket_family_power_separate_heads",
    "hierarchical_phase_bucket_replay",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def finite(value: object) -> bool:
    return value is not None and np.isfinite(float(value))


def panel_label(swarm: str, target: str) -> str:
    labels = {
        "300m": "300M",
        "starcoder_cosine": "StarCoder cosine",
        "starcoder_wsd80": "StarCoder WSD",
        "uncheatable": "Uncheatable",
        "table9": "Table-9",
        "starcoder_bpb": "StarCoder BPB",
    }
    return f"{labels.get(swarm, swarm)} / {labels.get(target, target)}"


def restriction_rows(
    swarm_id: str,
    target: str,
    swarm: dict[str, Any],
) -> tuple[list[dict[str, float | str | int]], list[dict[str, float | str]]]:
    rows = swarm["rows"]
    mask = np.asarray(
        [
            row["policyFamily"] == "single_phase"
            and not row["isSharedAlias"]
            and "single_phase" in row["fitPolicies"]
            and finite(row["observed"].get(target))
            for row in rows
        ],
        dtype=bool,
    )
    observed = np.asarray(
        [float(row["observed"][target]) for row, keep in zip(rows, mask, strict=False) if keep], dtype=float
    )
    predictions = swarm["predictions"][target]
    common = sorted(set(predictions["two_phase"]) & set(predictions["single_phase"]) & set(MECHANISTIC_MODELS))
    metric_rows: list[dict[str, float | str | int]] = []
    prediction_rows: list[dict[str, float | str]] = []
    for model in common:
        two_phase = np.asarray(predictions["two_phase"][model]["prediction"], dtype=float)[mask]
        single_phase = np.asarray(predictions["single_phase"][model]["prediction"], dtype=float)[mask]
        valid = np.isfinite(observed) & np.isfinite(two_phase) & np.isfinite(single_phase)
        if int(valid.sum()) < 3:
            continue
        observed_valid = observed[valid]
        two_valid = two_phase[valid]
        single_valid = single_phase[valid]
        two_summary, _bins = gate.metrics(observed_valid, two_valid)
        single_summary, _bins = gate.metrics(observed_valid, single_valid)
        disagreement = two_valid - single_valid
        for fit_policy, summary in (
            ("two_phase_restricted", two_summary),
            ("one_phase_refit", single_summary),
        ):
            metric_rows.append(
                {
                    "panel": panel_label(swarm_id, target),
                    "swarm": swarm_id,
                    "target": target,
                    "model": model,
                    "fit_policy": fit_policy,
                    "restriction_prediction_rmse": float(np.sqrt(np.mean(np.square(disagreement)))),
                    "restriction_prediction_bias": float(np.mean(disagreement)),
                    "restriction_prediction_max_abs": float(np.max(np.abs(disagreement))),
                    **summary,
                }
            )
        row_ids = [row["id"] for row, keep in zip(rows, mask, strict=False) if keep]
        for row_id, truth, two_prediction, single_prediction in zip(
            np.asarray(row_ids)[valid], observed_valid, two_valid, single_valid, strict=False
        ):
            prediction_rows.append(
                {
                    "panel": panel_label(swarm_id, target),
                    "swarm": swarm_id,
                    "target": target,
                    "model": model,
                    "row_id": row_id,
                    "observed": truth,
                    "two_phase_restricted": two_prediction,
                    "one_phase_refit": single_prediction,
                    "two_minus_one_prediction": two_prediction - single_prediction,
                }
            )
    return metric_rows, prediction_rows


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot(
        index=["panel", "swarm", "target", "model"],
        columns="fit_policy",
        values=[
            "rmse",
            "spearman",
            "regret_at_1",
            "optimism_gt_0p05_count",
            "worst_optimism",
        ],
    )
    pivot.columns = [f"{metric}__{policy}" for metric, policy in pivot.columns]
    pivot = pivot.reset_index()
    pivot["rmse_ratio_restricted_over_refit"] = pivot["rmse__two_phase_restricted"] / pivot["rmse__one_phase_refit"]
    details = metrics.loc[
        metrics["fit_policy"].eq("two_phase_restricted"),
        [
            "panel",
            "model",
            "restriction_prediction_rmse",
            "restriction_prediction_bias",
            "restriction_prediction_max_abs",
        ],
    ]
    return pivot.merge(details, on=["panel", "model"], how="left")


def write_figure(summary: pd.DataFrame, path: Path) -> None:
    plot = summary.copy()
    plot["direction"] = np.where(
        plot["rmse_ratio_restricted_over_refit"] <= 1.0,
        "Two-phase restriction better",
        "One-phase refit better",
    )
    figure = px.scatter(
        plot,
        x="rmse__one_phase_refit",
        y="rmse__two_phase_restricted",
        facet_col="panel",
        facet_col_wrap=2,
        color="rmse_ratio_restricted_over_refit",
        text="model",
        hover_data=[
            "restriction_prediction_rmse",
            "restriction_prediction_bias",
            "restriction_prediction_max_abs",
            "regret_at_1__one_phase_refit",
            "regret_at_1__two_phase_restricted",
        ],
        color_continuous_scale="RdYlGn_r",
        labels={
            "rmse__one_phase_refit": "One-phase refit RMSE",
            "rmse__two_phase_restricted": "Restricted two-phase RMSE",
            "rmse_ratio_restricted_over_refit": "Restricted / refit RMSE",
        },
        title="A valid phase-tied input does not guarantee coefficient transfer",
    )
    figure.update_traces(textposition="top center")
    for annotation in figure.layout.annotations:
        annotation.text = annotation.text.replace("panel=", "")
    maxima = plot.groupby("panel")[["rmse__one_phase_refit", "rmse__two_phase_restricted"]].max().max(axis=1)
    minima = plot.groupby("panel")[["rmse__one_phase_refit", "rmse__two_phase_restricted"]].min().min(axis=1)
    for index, panel in enumerate(plot["panel"].drop_duplicates(), start=1):
        lo = float(minima[panel])
        hi = float(maxima[panel])
        figure.add_shape(
            type="line",
            x0=lo,
            y0=lo,
            x1=hi,
            y1=hi,
            line=dict(color="#52636d", dash="dash"),
            xref=f"x{index if index > 1 else ''}",
            yref=f"y{index if index > 1 else ''}",
        )
    figure.update_layout(height=950, width=1500, margin=dict(l=80, r=100, t=100, b=80))
    figure.write_html(
        path,
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"scale": 4}},
    )


def write_report(summary: pd.DataFrame, path: Path) -> None:
    panel_summary = summary.groupby("panel", as_index=False).agg(
        model_count=("model", "nunique"),
        median_rmse_ratio=("rmse_ratio_restricted_over_refit", "median"),
        max_rmse_ratio=("rmse_ratio_restricted_over_refit", "max"),
        median_prediction_disagreement=("restriction_prediction_rmse", "median"),
        max_prediction_disagreement=("restriction_prediction_rmse", "max"),
        max_pointwise_disagreement=("restriction_prediction_max_abs", "max"),
        restricted_fit_wins=("rmse_ratio_restricted_over_refit", lambda values: int(np.sum(values <= 1.0))),
    )
    worst = summary.sort_values("rmse_ratio_restricted_over_refit", ascending=False).head(12)
    lines = [
        "# Empirical phase-tied restriction audit",
        "",
        "Every tested equation accepts a tied policy algebraically. This audit asks the stronger empirical question: "
        "do coefficients learned from the two-phase panel predict independently trained phase-tied policies as well as "
        "the same form refitted in the one-phase class?",
        "",
        panel_summary.to_markdown(index=False, floatfmt=".5f"),
        "",
        "Largest restricted/refit RMSE ratios:",
        "",
        worst[
            [
                "panel",
                "model",
                "rmse__one_phase_refit",
                "rmse__two_phase_restricted",
                "rmse_ratio_restricted_over_refit",
                "restriction_prediction_rmse",
                "restriction_prediction_max_abs",
            ]
        ].to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Interpretation",
        "",
        "- Algebraic phase tying is necessary but does not identify the shared coefficients. The independently refitted "
        "one-phase form is often materially better on the diagonal.",
        "- The disagreement is a direct symptom of phase-design confounding: a two-phase fit can assign aggregate "
        "effects to phase contrasts while retaining excellent in-panel OOF error.",
        "- This audit does not make the one-phase prediction privileged ground truth; it shows that the claimed limiting "
        "law is not empirically stable under refitting. Any headline model needs a hierarchical or invariant transition "
        "that survives this check without an output calibrator.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    gate.assert_sealed_absent(args.dashboard)
    dashboard = json.loads(args.dashboard.read_text())
    metric_rows: list[dict[str, float | str | int]] = []
    prediction_rows: list[dict[str, float | str]] = []
    for swarm_id, target in PANELS:
        metrics, predictions = restriction_rows(swarm_id, target, dashboard["swarms"][swarm_id])
        metric_rows.extend(metrics)
        prediction_rows.extend(predictions)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "restriction_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "restriction_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "restriction_summary.csv", index=False)
    write_figure(summary, args.output_dir / "phase_tied_restriction.html")
    write_report(summary, args.output_dir / "report.md")


if __name__ == "__main__":
    main()
