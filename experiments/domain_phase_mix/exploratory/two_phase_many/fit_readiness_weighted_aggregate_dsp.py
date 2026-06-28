# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly", "scipy", "scikit-learn"]
# ///
"""Fit DSP to readiness-weighted variants of the Grug-v4 aggregate.

The goal is to test whether heteroskedastic/readiness diagnostics improve the
objective used for mixture optimization. The original collaborator aggregate is
kept as a baseline. New target variants reweight the same 41 selected metrics
using the proportional-controllability readiness labels.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr

from experiments.domain_phase_mix.exploratory.two_phase_many.fit_grug_v4_aggregate_canonical_dsp import (
    DEFAULT_METADATA_CSV,
    DEFAULT_NOISE_CSV,
    DEFAULT_RAW_CSV,
    DEFAULT_VARIANT,
    EPS,
    average_phase_tv,
    entropy,
    mixture_comparison,
    prediction_metrics,
    read_dashboard_v4,
    read_reproduced_lcb,
    weights_frame,
    weights_to_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.reproduce_collaborator_grug_v4_aggregate import (
    DatasetBundle,
    aggregate_targets,
    load_data,
    proportional_weights,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp


SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_DIR / "readiness_weighted_aggregate_dsp_20260616"
DEFAULT_READINESS_CSV = (
    REFERENCE_DIR / "heteroskedastic_optimization_readiness_20260616" / "metric_optimization_readiness.csv"
)
ORIGINAL_CANONICAL_RAW_OPTIMUM_WEIGHTS = (
    REFERENCE_DIR
    / "collaborator_grug_v4_aggregate_repro_20260525"
    / "canonical_dsp_sent_zip"
    / "raw_optimum_weights.csv"
)
TO_IMAGE_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class TargetSpec:
    """One higher-is-better aggregate target to fit."""

    name: str
    description: str
    y: np.ndarray
    metric_weights: pd.DataFrame


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="sent_raw_metric_matrix_300m_zip_readiness_weighted")
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--noise-csv", type=Path, default=DEFAULT_NOISE_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--readiness-csv", type=Path, default=DEFAULT_READINESS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", choices=sorted(dsp.VARIANTS), default=DEFAULT_VARIANT)
    parser.add_argument("--maxiter", type=int, default=dsp.FIT_MAXITER)
    parser.add_argument("--coarse-top-k", type=int, default=dsp.START_TOP_K)
    parser.add_argument("--basin-hopping-iters", type=int, default=3)
    parser.add_argument("--optimum-starts", type=int, default=96)
    parser.add_argument("--max-observed-starts", type=int, default=96)
    parser.add_argument("--drop-incomplete-task-cols", action="store_true")
    return parser.parse_args()


def standardize(values: np.ndarray) -> np.ndarray:
    """Return finite standardized values."""
    y = np.asarray(values, dtype=float)
    sd = float(np.std(y, ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        raise ValueError("Cannot standardize a constant target")
    return (y - float(np.mean(y))) / sd


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute ordinary R^2."""
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


def task_role(metric: str, readiness_by_metric: dict[str, dict[str, Any]]) -> str:
    """Return the optimization-readiness role for one selected metric."""
    row = readiness_by_metric.get(metric)
    if row is not None:
        value = row.get("optimization_role")
        if isinstance(value, str) and value:
            return value
    if metric.startswith(("eval/paloma/", "eval/uncheatable_eval/")):
        return "stable_bpb_anchor_not_locally_calibrated"
    return "not_locally_calibrated"


def target_weight(role: str, policy: str) -> float:
    """Return a metric weight for one readiness policy."""
    direct_roles = {"local_steerable", "finite_effect_steerable", "local_and_finite_steerable"}
    if policy == "strict_steerable":
        return 1.0 if role in direct_roles else 0.0
    if policy == "steerable_guardrail":
        if role in direct_roles:
            return 1.0
        if role == "guardrail_detectable":
            return 0.5
        return 0.0
    if policy == "steerable_guardrail_stabilized":
        if role in direct_roles:
            return 1.0
        if role == "guardrail_detectable":
            return 0.5
        if role == "stable_bpb_anchor_not_locally_calibrated":
            return 0.35
        return 0.0
    if policy == "broad_screened":
        if role in direct_roles:
            return 1.0
        if role == "guardrail_detectable":
            return 0.75
        if role == "stable_bpb_anchor_not_locally_calibrated":
            return 0.35
        if role == "fragile_screen_only":
            return 0.05
        return 0.0
    raise ValueError(f"Unknown readiness policy {policy}")


def metric_weight_frame(
    task_cols: list[str],
    readiness: pd.DataFrame,
    *,
    policy: str,
) -> pd.DataFrame:
    """Build metric weights for one readiness policy."""
    readiness_by_metric = {
        str(row.metric): row._asdict()
        for row in readiness.itertuples(index=False)
        if isinstance(row.metric, str)
    }
    rows: list[dict[str, Any]] = []
    for metric in task_cols:
        role = task_role(metric, readiness_by_metric)
        weight = target_weight(role, policy)
        readiness_row = readiness_by_metric.get(metric, {})
        rows.append(
            {
                "policy": policy,
                "metric": metric,
                "optimization_role": role,
                "raw_weight": weight,
                "normalized_weight": 0.0,
                "log_tilt_optimization_readiness": readiness_row.get("log_tilt_optimization_readiness"),
                "bump_optimization_readiness": readiness_row.get("bump_optimization_readiness"),
                "actuation_max_all_anchor_sd_bh_q_value": readiness_row.get(
                    "actuation_max_all_anchor_sd_bh_q_value"
                ),
                "bump_max_all_anchor_sd_bh_q_value": readiness_row.get("bump_max_all_anchor_sd_bh_q_value"),
                "max_all_over_proportional_sd": readiness_row.get("max_all_over_proportional_sd"),
            }
        )
    frame = pd.DataFrame(rows)
    total = float(frame["raw_weight"].sum())
    if total <= 0.0:
        raise ValueError(f"Readiness policy {policy} selected no metrics")
    frame["normalized_weight"] = frame["raw_weight"] / total
    return frame


def weighted_z_target(z: np.ndarray, weights: pd.DataFrame) -> np.ndarray:
    """Compute a standardized higher-is-better weighted z target."""
    vector = weights["normalized_weight"].to_numpy(dtype=float)
    return standardize(z @ vector)


def build_targets(data: dict[str, object], readiness: pd.DataFrame) -> list[TargetSpec]:
    """Create original and readiness-weighted targets."""
    z = np.asarray(data["z"], dtype=float)
    noise_share = np.asarray(data["noise_share"], dtype=float)
    task_cols = list(data["task_cols"])
    aggregate = aggregate_targets(z, noise_share)
    original = standardize(np.asarray(aggregate["y_factor"], dtype=float))
    original_weights = pd.DataFrame(
        {
            "policy": "original_factor",
            "metric": task_cols,
            "optimization_role": "factor_loading_not_simple_weight",
            "raw_weight": np.nan,
            "normalized_weight": np.nan,
        }
    )
    targets = [
        TargetSpec(
            name="original_factor",
            description="Original collaborator-style 5-factor aggregate, standardized.",
            y=original,
            metric_weights=original_weights,
        )
    ]
    for policy, description in [
        ("strict_steerable", "Only metrics classified as local/finite steerable."),
        ("steerable_guardrail", "Steerable metrics plus half-weight robust guardrail-detectable metrics."),
        (
            "steerable_guardrail_stabilized",
            "Steerable and guardrail metrics plus stable BPB anchors without local diagnostics.",
        ),
        (
            "broad_screened",
            "A broader screened aggregate: steerable, guardrails, stable BPB anchors, and tiny fragile-screen weight.",
        ),
    ]:
        weights = metric_weight_frame(task_cols, readiness, policy=policy)
        targets.append(TargetSpec(policy, description, weighted_z_target(z, weights), weights))
    return targets


def phase_stats(weights: np.ndarray) -> dict[str, float]:
    """Summarize a two-phase mixture."""
    return {
        "phase0_support_gt_1e3": int(np.sum(weights[0] > 1e-3)),
        "phase1_support_gt_1e3": int(np.sum(weights[1] > 1e-3)),
        "phase0_entropy": entropy(weights[0]),
        "phase1_entropy": entropy(weights[1]),
        "phase0_effective_support": float(np.exp(entropy(weights[0]))),
        "phase1_effective_support": float(np.exp(entropy(weights[1]))),
        "phase0_max_weight": float(np.max(weights[0])),
        "phase1_max_weight": float(np.max(weights[1])),
    }


def read_original_raw_optimum(domains: list[str]) -> np.ndarray | None:
    """Read the raw optimum from the original-factor canonical DSP fit."""
    if not ORIGINAL_CANONICAL_RAW_OPTIMUM_WEIGHTS.exists():
        return None
    frame = pd.read_csv(ORIGINAL_CANONICAL_RAW_OPTIMUM_WEIGHTS)
    frame = frame[frame["label"].eq("raw_dsp_optimum")].set_index("domain")
    if frame.empty:
        return None
    aligned = frame.reindex(domains).fillna(0.0)
    weights = np.stack(
        [
            aligned["phase_0_weight"].to_numpy(dtype=float),
            aligned["phase_1_weight"].to_numpy(dtype=float),
        ],
        axis=0,
    )
    totals = weights.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("Original raw optimum has an empty phase")
    return weights / totals


def fit_target(
    target: TargetSpec,
    *,
    raw: pd.DataFrame,
    domains: list[str],
    w0: np.ndarray,
    w1: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    variant: dsp.DSPVariant,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dsp.FittedDSPModel]:
    """Fit one target and return summary, weights, predictions, and model."""
    print(f"fitting target={target.name} rows={len(target.y)} metrics={len(target.metric_weights)}", flush=True)
    packet = weights_to_packet(raw, -target.y, domains, w0, w1, c0, c1)
    model, tuning = dsp.fit_variant(
        packet,
        variant,
        maxiter=args.maxiter,
        coarse_top_k=args.coarse_top_k,
        basin_hopping_iters=args.basin_hopping_iters,
    )
    print(f"optimizing raw optimum target={target.name}", flush=True)
    raw_result, raw_weights = dsp.optimize_raw(
        model,
        num_starts=args.optimum_starts,
        observed_start_weights=packet.w,
        max_observed_starts=args.max_observed_starts,
    )
    train_pred = -dsp.predict(model, packet.w)
    oof_pred = -dsp.oof_predictions(packet, model)
    predictions = packet.frame[["run_name", "run_id"]].copy()
    predictions["target_name"] = target.name
    predictions["actual_target"] = target.y
    predictions["train_pred_target"] = train_pred
    predictions["oof_pred_target"] = oof_pred
    predictions["oof_residual"] = oof_pred - target.y
    predictions["actual_rank_desc"] = predictions["actual_target"].rank(method="min", ascending=False)
    predictions["oof_rank_desc"] = predictions["oof_pred_target"].rank(method="min", ascending=False)

    raw_distances = dsp.average_phase_tv_distance(packet.w, raw_weights[None, :, :])
    nearest_idx = int(np.argmin(raw_distances))
    w0_prop, w1_prop = proportional_weights(c0, c1)
    proportional = np.stack([w0_prop, w1_prop], axis=0)
    mixtures: dict[str, np.ndarray] = {
        "proportional": proportional,
        "raw_dsp_optimum": raw_weights,
    }
    for label, weights in [
        ("dashboard_v4", read_dashboard_v4(domains)),
        ("collaborator_lcb", read_reproduced_lcb(domains)),
        ("original_factor_raw_dsp_optimum", read_original_raw_optimum(domains)),
    ]:
        if weights is not None:
            mixtures[label] = weights

    pred_target_by_label = {
        label: -float(dsp.predict(model, weights[None, :, :])[0]) for label, weights in mixtures.items()
    }
    comparisons = [
        mixture_comparison(
            f"raw_dsp_optimum_vs_{label}",
            raw_weights,
            weights,
            pred_target_by_label["raw_dsp_optimum"],
            pred_target_by_label[label],
        )
        for label, weights in mixtures.items()
        if label != "raw_dsp_optimum"
    ]
    summary = {
        "target_name": target.name,
        "description": target.description,
        "variant": model.variant.name,
        "fit_row_count": int(len(target.y)),
        "target_mean": float(np.mean(target.y)),
        "target_std": float(np.std(target.y, ddof=1)),
        "metric_count": int(len(target.metric_weights)),
        "active_metric_count": int((target.metric_weights["normalized_weight"].fillna(0.0) > 0.0).sum()),
        "total_param_count": int(model.total_param_count),
        "m_dependent_params_per_domain": int(model.m_dependent_params_per_domain),
        "gamma": float(model.params.get("gamma", float("nan"))),
        "active_benefit_coef_count": int(np.sum(model.benefit_coef > EPS)),
        "active_penalty_coef_count": int(np.sum(model.penalty_coef > EPS)),
        "raw_pred_target": -float(raw_result.fun),
        "raw_pred_delta_vs_proportional": pred_target_by_label["raw_dsp_optimum"]
        - pred_target_by_label["proportional"],
        "raw_nearest_observed_tv": float(raw_distances[nearest_idx]),
        "raw_nearest_observed_run_name": str(packet.frame.iloc[nearest_idx][packet.name_col]),
        "raw_nearest_observed_target": float(target.y[nearest_idx]),
        "raw_optimum_success": bool(raw_result.success),
        "raw_optimum_message": str(raw_result.message),
        "pred_target_by_label": pred_target_by_label,
        "mixture_comparisons": comparisons,
        **prediction_metrics(target.y, train_pred, "train"),
        **prediction_metrics(target.y, oof_pred, "oof"),
        **phase_stats(raw_weights),
    }
    weights = pd.concat(
        [weights_frame(label, domains, values, c0, c1, model) for label, values in mixtures.items()],
        ignore_index=True,
    )
    weights.insert(0, "target_name", target.name)
    tuning.insert(0, "target_name", target.name)
    summary["tuning_rows"] = tuning.to_dict(orient="records")
    return summary, weights, predictions, model


def write_target_plots(output_dir: Path, summaries: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """Write overview HTML plots."""
    fig = px.scatter(
        predictions,
        x="actual_target",
        y="oof_pred_target",
        color="target_name",
        facet_col="target_name",
        facet_col_wrap=3,
        hover_name="run_name",
        title="Readiness-weighted aggregate DSP fits: OOF predictions",
        labels={"actual_target": "actual target (standardized utility)", "oof_pred_target": "OOF predicted target"},
    )
    fig.update_layout(width=1450, height=950)
    fig.write_html(output_dir / "oof_prediction_facets.html", config=TO_IMAGE_CONFIG)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("OOF Spearman", "Raw optimum delta vs proportional"))
    order = summaries.sort_values("oof_spearman", ascending=False)["target_name"].tolist()
    sorted_summary = summaries.set_index("target_name").loc[order].reset_index()
    fig.add_trace(go.Bar(x=sorted_summary["target_name"], y=sorted_summary["oof_spearman"], name="OOF Spearman"), row=1, col=1)
    fig.add_trace(
        go.Bar(
            x=sorted_summary["target_name"],
            y=sorted_summary["raw_pred_delta_vs_proportional"],
            name="predicted gain",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(width=1300, height=550, title="Fit quality and predicted raw-optimum gain", showlegend=False)
    fig.write_html(output_dir / "target_summary_bars.html", config=TO_IMAGE_CONFIG)


def write_mixture_plot(
    output_dir: Path,
    target_name: str,
    weights: pd.DataFrame,
) -> None:
    """Plot mixture weights for one target."""
    subset = weights[weights["target_name"].eq(target_name)].copy()
    raw = subset[subset["label"].eq("raw_dsp_optimum")].sort_values("total_epochs", ascending=False)
    domains = raw["domain"].tolist()
    subset["domain"] = pd.Categorical(subset["domain"], categories=domains, ordered=True)
    plot = subset.sort_values("domain")
    fig = px.bar(
        plot,
        x="total_epochs",
        y="domain",
        color="label",
        orientation="h",
        barmode="group",
        title=f"Materialized epochs by mixture: {target_name}",
        hover_data=["phase_0_weight", "phase_1_weight", "phase_0_epochs", "phase_1_epochs"],
    )
    fig.update_layout(width=1300, height=max(720, 24 * len(domains)), yaxis={"categoryorder": "array", "categoryarray": domains})
    fig.write_html(output_dir / f"{target_name}_mixture_epochs.html", config=TO_IMAGE_CONFIG)


def write_report(output_dir: Path, summaries: pd.DataFrame, metric_weights: pd.DataFrame) -> None:
    """Write a concise analysis report."""
    summary_cols = [
        "target_name",
        "active_metric_count",
        "oof_spearman",
        "oof_r2",
        "raw_pred_delta_vs_proportional",
        "raw_nearest_observed_tv",
        "phase0_max_weight",
        "phase1_max_weight",
        "phase0_effective_support",
        "phase1_effective_support",
    ]
    active_weights = metric_weights[metric_weights["normalized_weight"].fillna(0.0) > 0.0].copy()
    role_counts = (
        active_weights.groupby(["target_name", "optimization_role"], dropna=False)
        .size()
        .reset_index(name="metric_count")
    )
    lines = [
        "# Readiness-Weighted Aggregate DSP Analysis",
        "",
        "This analysis rebuilds aggregate targets over the original Grug-v4 selected metric panel, using heteroskedastic optimization-readiness labels from the 300M proportional-controllability diagnostics.",
        "",
        "All fitted targets are standardized higher-is-better utilities. Predicted gains are therefore in target-SD units and should be compared within target, not as observed benchmark deltas.",
        "",
        "## Summary",
        "",
        summaries[summary_cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Active Metric Roles",
        "",
        role_counts.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- `strict_steerable` tests whether the direct-readiness signal alone is enough; if it collapses to a narrow mixture, it is not deployable by itself.",
        "- `steerable_guardrail` adds robust-but-direction-unstable metrics as partial objective terms; these should still be validated as guardrails.",
        "- `steerable_guardrail_stabilized` adds Paloma/uncheatable BPB anchors that have strong prior modeling evidence but no exact local-controllability diagnostic in this experiment.",
        "- `broad_screened` is closest to a conservative replacement for the original aggregate: it keeps steerable and robust metrics, includes stable BPB anchors, and gives fragile metrics only tiny weight.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    """Run the readiness-weighted aggregate DSP comparison."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = DatasetBundle(
        name=args.dataset_name,
        raw_path=args.raw_csv,
        noise_path=args.noise_csv,
        metadata_path=args.metadata_csv,
        drop_incomplete_task_cols=args.drop_incomplete_task_cols,
    )
    print(f"loading aggregate data from {bundle.raw_path}", flush=True)
    data = load_data(bundle)
    readiness = pd.read_csv(args.readiness_csv)
    targets = build_targets(data, readiness)

    raw = data["raw"]
    domains = data["domains"]
    w0 = data["w0"]
    w1 = data["w1"]
    c0 = data["c0"]
    c1 = data["c1"]
    assert isinstance(raw, pd.DataFrame)
    assert isinstance(domains, list)
    assert isinstance(w0, np.ndarray)
    assert isinstance(w1, np.ndarray)
    assert isinstance(c0, np.ndarray)
    assert isinstance(c1, np.ndarray)
    variant = dsp.VARIANTS[args.variant]

    summary_rows: list[dict[str, Any]] = []
    all_weights: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_metric_weights: list[pd.DataFrame] = []
    model_blobs: dict[str, Any] = {}
    for target in targets:
        summary, weights, predictions, model = fit_target(
            target,
            raw=raw,
            domains=domains,
            w0=w0,
            w1=w1,
            c0=c0,
            c1=c1,
            variant=variant,
            args=args,
        )
        summary_rows.append(summary)
        all_weights.append(weights)
        all_predictions.append(predictions)
        metric_weights = target.metric_weights.copy()
        metric_weights.insert(0, "target_name", target.name)
        all_metric_weights.append(metric_weights)
        model_blobs[target.name] = dsp.model_to_json(model, summary)
        target_dir = args.output_dir / target.name
        target_dir.mkdir(exist_ok=True)
        weights.to_csv(target_dir / "mixture_weights.csv", index=False)
        weights[weights["label"].eq("raw_dsp_optimum")].to_csv(target_dir / "raw_optimum_weights.csv", index=False)
        predictions.to_csv(target_dir / "observed_predictions.csv", index=False)
        (target_dir / "model.json").write_text(json.dumps(model_blobs[target.name], indent=2))
        pd.DataFrame(summary["tuning_rows"]).to_csv(target_dir / "tuning.csv", index=False)
        write_mixture_plot(target_dir, target.name, weights)

    summaries_full = pd.DataFrame(summary_rows)
    summaries = summaries_full.drop(columns=["mixture_comparisons", "pred_target_by_label", "tuning_rows"])
    weights = pd.concat(all_weights, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    metric_weights = pd.concat(all_metric_weights, ignore_index=True)

    summaries.to_csv(args.output_dir / "summary.csv", index=False)
    summaries_full.to_json(args.output_dir / "summary_full.json", orient="records", indent=2)
    weights.to_csv(args.output_dir / "all_mixture_weights.csv", index=False)
    predictions.to_csv(args.output_dir / "all_observed_predictions.csv", index=False)
    metric_weights.to_csv(args.output_dir / "target_metric_weights.csv", index=False)
    (args.output_dir / "models.json").write_text(json.dumps(model_blobs, indent=2))

    comparison_rows = []
    for row in summary_rows:
        for comparison in row["mixture_comparisons"]:
            comparison_rows.append({"target_name": row["target_name"], **comparison})
    pd.DataFrame(comparison_rows).to_csv(args.output_dir / "mixture_comparisons.csv", index=False)

    target_score_wide = predictions.pivot(index="run_name", columns="target_name", values="actual_target").reset_index()
    target_score_wide.to_csv(args.output_dir / "target_scores_wide.csv", index=False)
    write_target_plots(args.output_dir, summaries, predictions)
    write_report(args.output_dir, summaries, metric_weights)

    print(
        textwrap.dedent(
            f"""
            wrote {args.output_dir}
            targets={len(targets)}
            best_oof={summaries.sort_values('oof_spearman', ascending=False).iloc[0]['target_name']}
            best_oof_spearman={summaries['oof_spearman'].max():.4f}
            """
        ).strip(),
        flush=True,
    )


if __name__ == "__main__":
    main()
