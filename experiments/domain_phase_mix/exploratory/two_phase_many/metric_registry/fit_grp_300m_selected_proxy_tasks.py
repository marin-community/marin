# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "plotly", "scipy", "scikit-learn", "tabulate", "torch"]
# ///
"""Fit GRP to the current 300M selected task proxies.

The selected-proxy table chooses one smooth proxy per task using fixed-mixture
SNR and proxy-to-accuracy correlation. This script turns those selections into
task-balanced aggregate objectives and fits/optimizes the GRP no-L2 family.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_grp_power_family_penalty_no_l2_retune import (
    REG_FIXED,
    VARIANT_NAME,
    _no_l2_param_keys,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.build_eval_signal_to_noise_all_metrics_300m import (
    DEFAULT_KEEP_DROP_CSV,
    _default_extra_results_csvs,
    _load_signal_frame,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry import (
    fit_grp_300m_perplexity_proxy_benchmark as proxy_benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.build_metric_registry import (
    WEIGHT_PREFIXES,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.fit_grp_300m_mmlu_choice_prob_norm import (
    BLOCK_VARIANTS,
    DEFAULT_COARSE_TOP_K,
    DEFAULT_METHOD,
    DEFAULT_PROB_EPS,
    DEFAULT_RANDOM_STARTS,
    _model_options,
    _refine_rows,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.fit_grp_no_l2_benchmark_aggregates import (
    DEFAULT_FAMILY_SCHEME,
    FAMILY_SCHEMES,
    AggregateObjective,
    _expanded_start_bank,
    _model_target_to_metric,
    _packet_from_frame,
    _plot_predictions,
    _plot_residuals,
    _prediction_rows,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    build_penalty_calibration_surrogate,
    optimize_penalty_calibration_model,
)

SCALE = "300m_6b"
COHORT = "signal"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "reference_outputs" / "grp_300m_selected_proxy_tasks_20260430"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
OPTIMUM_DIAGNOSTICS_CSV = OUTPUT_DIR / "optimum_diagnostics.csv"
TARGET_COMPONENTS_CSV = OUTPUT_DIR / "target_components.csv"
REPORT_MD = OUTPUT_DIR / "report.md"
LOGBOOK_MD = Path(".agents/logbooks/benchmark-proxy-optimization.md")

PRIMARY_TARGET = "selected_proxy_task_balanced"
KEEP_ONLY_TARGET = "selected_proxy_keep_only_task_balanced"
SNR_GT2_TARGET = "selected_proxy_snr_gt2_task_balanced"
FLAT_TARGET = "selected_proxy_flat_weighted"
ACCURACY_SUFFIX = "__accuracy_reference"

EXCLUDED_TASKS = frozenset({"averages", "eval"})
SNR_GT2_THRESHOLD = 2.0


@dataclass(frozen=True)
class TargetSpec:
    """One selected-proxy aggregate target to fit."""

    slug: str
    display_name: str
    selection_mode: str
    collapse_groups: bool


TARGET_SPECS = (
    TargetSpec(
        slug=PRIMARY_TARGET,
        display_name="Task-balanced selected proxies, keep+downweight",
        selection_mode="keep_downweight",
        collapse_groups=True,
    ),
    TargetSpec(
        slug=KEEP_ONLY_TARGET,
        display_name="Task-balanced selected proxies, keep only",
        selection_mode="keep_only",
        collapse_groups=True,
    ),
    TargetSpec(
        slug=SNR_GT2_TARGET,
        display_name="Task-balanced selected proxies, selected SNR > 2",
        selection_mode="snr_gt2",
        collapse_groups=True,
    ),
    TargetSpec(
        slug=FLAT_TARGET,
        display_name="Flat selected proxies, keep+downweight",
        selection_mode="keep_downweight",
        collapse_groups=False,
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--coarse-top-k", type=int, default=DEFAULT_COARSE_TOP_K)
    parser.add_argument("--random-starts", type=int, default=DEFAULT_RANDOM_STARTS)
    parser.add_argument("--prob-eps", type=float, default=DEFAULT_PROB_EPS)
    parser.add_argument("--family-scheme", choices=FAMILY_SCHEMES, default=DEFAULT_FAMILY_SCHEME)
    parser.add_argument("--block-variant", choices=BLOCK_VARIANTS, default="full")
    return parser.parse_args()


def _component_group(task: str) -> str:
    if task == "mmlu_5shot":
        return "mmlu_standard"
    if task.startswith("mmlu_") and task.endswith("_5shot"):
        return "mmlu_standard"
    return task


def _aggregate_objective(slug: str, display_name: str) -> AggregateObjective:
    return AggregateObjective(
        slug=slug,
        source_column="objective_metric",
        display_name=display_name,
        family="selected_task_proxy",
        higher_is_better=True,
        transform="identity",
    )


def _complete_qsplit_signal_frame() -> pd.DataFrame:
    signal = _load_signal_frame(_default_extra_results_csvs())
    data = signal[
        signal["scale"].eq(SCALE) & signal["cohort"].eq(COHORT) & signal["is_qsplit240_core"].fillna(False)
    ].copy()
    if len(data) != 240:
        raise ValueError(f"Expected 240 qsplit-core rows, found {len(data)}")
    for prefix in ("phase_0_", "phase_1_"):
        weight_columns = [column for column in data.columns if column.startswith(prefix)]
        sums = data[weight_columns].sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-6):
            raise ValueError(f"{prefix} weights do not sum to 1")
    return data.reset_index(drop=True)


def _selected_components(selection_mode: str) -> pd.DataFrame:
    keep_drop = pd.read_csv(DEFAULT_KEEP_DROP_CSV)
    if selection_mode == "keep_downweight":
        selected = keep_drop[keep_drop["optimization_recommendation"].isin({"keep", "downweight"})].copy()
        selected["component_raw_weight"] = selected["default_weight"].astype(float) * selected[
            "selected_proxy_score"
        ].astype(float)
    elif selection_mode == "keep_only":
        selected = keep_drop[keep_drop["optimization_recommendation"].eq("keep")].copy()
        selected["component_raw_weight"] = selected["default_weight"].astype(float) * selected[
            "selected_proxy_score"
        ].astype(float)
    elif selection_mode == "snr_gt2":
        selected = keep_drop[pd.to_numeric(keep_drop["selected_proxy_snr"], errors="coerce") > SNR_GT2_THRESHOLD].copy()
        selected["component_raw_weight"] = selected["selected_proxy_snr"].astype(float)
    else:
        raise ValueError(f"Unsupported selection mode: {selection_mode}")

    selected = selected[
        selected["selected_proxy_metric"].notna()
        & selected["accuracy_metric"].notna()
        & ~selected["task"].isin(EXCLUDED_TASKS)
        & (selected["component_raw_weight"] > 0.0)
    ].copy()
    if selected.empty:
        raise ValueError("No selected proxies available for target construction")
    selected["component_group"] = selected["task"].map(_component_group)
    return selected.reset_index(drop=True)


def _standardized_component(data: pd.DataFrame, metric: str, direction: str) -> pd.Series:
    if metric not in data.columns:
        raise ValueError(f"Selected proxy metric is missing from signal frame: {metric}")
    values = pd.to_numeric(data[metric], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Selected proxy metric has missing values: {metric}")
    oriented = values if direction == "maximize" else -values
    std = float(oriented.std(ddof=1))
    if std == 0.0 or not np.isfinite(std):
        raise ValueError(f"Selected proxy metric has zero/invalid std: {metric}")
    return (oriented - float(oriented.mean())) / std


def _weighted_average(columns: list[pd.Series], weights: np.ndarray) -> pd.Series:
    if len(columns) == 1:
        return columns[0].copy()
    matrix = np.column_stack([column.to_numpy(dtype=float) for column in columns])
    return pd.Series(matrix @ weights, index=columns[0].index)


def _add_target_columns(data: pd.DataFrame, spec: TargetSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = _selected_components(selection_mode=spec.selection_mode)
    out = data.copy()
    component_rows: list[dict[str, Any]] = []
    group_proxy: dict[str, pd.Series] = {}
    group_accuracy: dict[str, pd.Series] = {}
    group_scores: dict[str, float] = {}

    for group, group_frame in selected.groupby("component_group", sort=True):
        proxy_columns: list[pd.Series] = []
        accuracy_columns: list[pd.Series] = []
        weights = group_frame["component_raw_weight"].to_numpy(dtype=float)
        weights = weights / weights.sum()
        for local_idx, (_, row) in enumerate(group_frame.iterrows()):
            proxy_metric = str(row["selected_proxy_metric"])
            accuracy_metric = str(row["accuracy_metric"])
            if accuracy_metric not in out.columns:
                raise ValueError(f"Accuracy metric is missing from signal frame: {accuracy_metric}")
            proxy_component = _standardized_component(out, proxy_metric, str(row["selected_proxy_direction"]))
            accuracy_values = pd.to_numeric(out[accuracy_metric], errors="coerce")
            if accuracy_values.isna().any():
                raise ValueError(f"Accuracy metric has missing values: {accuracy_metric}")
            proxy_columns.append(proxy_component)
            accuracy_columns.append(accuracy_values)
            component_rows.append(
                {
                    "target": spec.slug,
                    "task": row["task"],
                    "component_group": group,
                    "selected_proxy_metric": proxy_metric,
                    "selected_proxy_direction": row["selected_proxy_direction"],
                    "selected_proxy_kind": row["selected_proxy_kind"],
                    "selected_proxy_snr": float(row["selected_proxy_snr"]),
                    "selected_proxy_accuracy_spearman": float(row["selected_proxy_accuracy_spearman"]),
                    "selected_proxy_score": float(row["selected_proxy_score"]),
                    "optimization_recommendation": row["optimization_recommendation"],
                    "accuracy_metric": accuracy_metric,
                    "default_weight": float(row["default_weight"]),
                    "component_weight_within_group": float(weights[local_idx]),
                }
            )
        group_proxy[group] = _weighted_average(proxy_columns, weights)
        group_accuracy[group] = _weighted_average(accuracy_columns, weights)
        group_scores[group] = float(group_frame["component_raw_weight"].sum())

    groups = sorted(group_proxy)
    if spec.collapse_groups:
        group_weights = np.full(len(groups), 1.0 / len(groups), dtype=float)
    else:
        group_weights = np.asarray([group_scores[group] for group in groups], dtype=float)
        group_weights = group_weights / group_weights.sum()
    out[spec.slug] = _weighted_average([group_proxy[group] for group in groups], group_weights)
    out[f"{spec.slug}{ACCURACY_SUFFIX}"] = _weighted_average(
        [group_accuracy[group] for group in groups],
        group_weights,
    )
    components = pd.DataFrame.from_records(component_rows)
    components["group_weight"] = components["component_group"].map(
        {group: float(weight) for group, weight in zip(groups, group_weights, strict=True)}
    )
    components["effective_component_weight"] = components["component_weight_within_group"] * components["group_weight"]
    return out, components


def _packet_frame(data: pd.DataFrame, objective_column: str) -> pd.DataFrame:
    weight_columns = sorted(column for column in data.columns if column.startswith(WEIGHT_PREFIXES))
    id_columns = [
        column
        for column in (
            "registry_run_key",
            "run_id",
            "run_name",
            "scale",
            "cohort",
            "source_run_name",
            "source_experiment",
            "wandb_run_id",
            "checkpoint_root",
            "status",
            "is_qsplit240_core",
        )
        if column in data.columns
    ]
    frame = data[id_columns + weight_columns + [objective_column]].rename(columns={objective_column: "objective_metric"})
    frame["objective_metric_key"] = objective_column
    return frame.dropna(axis=1, how="all").reset_index(drop=True)


def _fit_selected_proxy_target(
    data: pd.DataFrame,
    spec: TargetSpec,
    *,
    family_scheme: str,
    model_options: dict[str, bool],
    start_bank: tuple[dict[str, float], ...],
    coarse_top_k: int,
    method: str,
    prob_eps: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    objective = _aggregate_objective(spec.slug, spec.display_name)
    packet = _packet_from_frame(_packet_frame(data, spec.slug), objective, prob_eps, family_scheme)
    coarse, best, refine = _refine_rows(packet, start_bank, coarse_top_k, method, model_options)
    params = {key: float(best[key]) for key in _no_l2_param_keys()}
    params["reg"] = REG_FIXED
    model = build_penalty_calibration_surrogate(
        packet,
        params=params,
        variant_name=VARIANT_NAME,
        **model_options,
    ).fit(packet.base.w, packet.base.y)
    pred_rows = _prediction_rows(packet, params, objective)
    raw_result, phase0, phase1 = optimize_penalty_calibration_model(packet, model, seed=0)
    raw_weights = np.stack([phase0, phase1], axis=0)
    score_model = proxy_benchmark.ScoreModelAdapter(
        lambda weights: _model_target_to_metric(model.predict(weights), objective)
    )
    actual_proxy = data[spec.slug].to_numpy(dtype=float)
    actual_accuracy = data[f"{spec.slug}{ACCURACY_SUFFIX}"].to_numpy(dtype=float)
    diag, weight_map = proxy_benchmark._candidate_diagnostics(
        candidate_id=spec.slug,
        candidate_kind="direct_grp_selected_proxy",
        target=spec.slug,
        packet=packet,
        score_model=score_model,
        actual_choice=actual_proxy,
        actual_accuracy=actual_accuracy,
        actual_primary=actual_proxy,
        raw_metric=float(_model_target_to_metric(float(raw_result.fun), objective)),
        raw_weights=raw_weights,
    )

    out_dir = OUTPUT_DIR / "grp_candidates" / spec.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    coarse.to_csv(out_dir / "coarse.csv", index=False)
    refine.to_csv(out_dir / "refine.csv", index=False)
    pd.DataFrame([params]).to_csv(out_dir / "params.csv", index=False)
    pred_rows.to_csv(out_dir / "oof_predictions.csv", index=False)
    _plot_predictions(pred_rows, out_dir / "predicted_vs_actual.html", spec.display_name)
    _plot_residuals(pred_rows, out_dir / "residuals.html", spec.display_name)
    diag.to_csv(out_dir / "optimum_diagnostics.csv", index=False)
    proxy_benchmark._write_optimum_weights(out_dir / "optimum_weights.csv", packet, weight_map)
    raw_plot = proxy_benchmark._plot_phase_comparison(out_dir, packet, "raw", f"Raw optimum: {spec.display_name}")
    hull_plot = proxy_benchmark._plot_phase_comparison(
        out_dir,
        packet,
        "top8actual_hull",
        f"Top8 hull: {spec.display_name}",
    )

    predicted = pred_rows["predicted_metric"].to_numpy(dtype=float)
    summary = {
        "candidate_id": spec.slug,
        "display_name": spec.display_name,
        "n_rows": len(data),
        "family_scheme": family_scheme,
        "block_variant": "full" if not model_options else json.dumps(model_options, sort_keys=True),
        "raw_phase_plot": str(raw_plot),
        "top8_hull_phase_plot": str(hull_plot),
        **proxy_benchmark._metric_summary(actual_proxy, predicted, prefix="proxy_"),
        **proxy_benchmark._metric_summary(actual_accuracy, predicted, prefix="accuracy_reference_"),
    }
    raw_row = diag.loc[diag["opt_kind"].eq("raw")].iloc[0].to_dict()
    hull_row = diag.loc[diag["opt_kind"].eq("top8actual_hull")].iloc[0].to_dict()
    for key, value in raw_row.items():
        if isinstance(value, int | float | np.integer | np.floating):
            summary[f"raw_{key}"] = float(value)
    for key, value in hull_row.items():
        if isinstance(value, int | float | np.integer | np.floating):
            summary[f"top8actual_hull_{key}"] = float(value)
    return summary, diag


def _write_report(summary: pd.DataFrame, optimum: pd.DataFrame, components: pd.DataFrame) -> None:
    primary_components = components[components["target"].eq(PRIMARY_TARGET)].sort_values(
        ["component_group", "effective_component_weight"],
        ascending=[True, False],
    )
    snr_gt2_components = components[components["target"].eq(SNR_GT2_TARGET)].sort_values(
        ["component_group", "effective_component_weight"],
        ascending=[True, False],
    )
    summary_columns = [
        "candidate_id",
        "proxy_spearman",
        "proxy_regret_at_1",
        "proxy_lower_tail_optimism",
        "accuracy_reference_spearman",
        "accuracy_reference_regret_at_1",
        "raw_predicted_proxy_metric",
        "raw_nearest_observed_tv",
        "raw_nearest_observed_choice",
        "raw_nearest_observed_accuracy",
        "top8actual_hull_nearest_observed_tv",
    ]
    optimum_columns = [
        "candidate_id",
        "opt_kind",
        "predicted_proxy_metric",
        "nearest_observed_tv",
        "nearest_observed_run_name",
        "nearest_observed_choice",
        "nearest_observed_accuracy",
        "nearest_observed_primary_regret",
        "raw_phase0_support_gt_1e4",
        "raw_phase1_support_gt_1e4",
    ]
    body = [
        "# 300M Selected-Proxy Task Optimization",
        "",
        "## Setup",
        "",
        "- Rows: 240 qsplit-core 300M/6B rows.",
        "- Source: current `eval_signal_to_noise_all_metrics_300m_current_keep_drop.csv`.",
        "- Each selected smooth proxy is oriented so higher is better, z-scored, then aggregated.",
        "- Primary target collapses all standard MMLU selected proxies into one group and equally weights task groups.",
        "- The `accuracy_reference_*` metrics evaluate the same selected task basket using hard accuracy values.",
        "- The strict-SNR target keeps only components whose selected proxy has fixed-mixture SNR > 2.0.",
        "",
        "## GRP Fit Summary",
        "",
        summary[summary_columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Optimum Diagnostics",
        "",
        optimum[optimum["opt_kind"].isin(["best_observed", "predicted_best_observed", "raw", "top8actual_hull"])][
            optimum_columns
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Primary Target Components",
        "",
        primary_components[
            [
                "task",
                "component_group",
                "selected_proxy_metric",
                "selected_proxy_direction",
                "optimization_recommendation",
                "selected_proxy_snr",
                "selected_proxy_accuracy_spearman",
                "effective_component_weight",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Strict SNR > 2 Target Components",
        "",
        snr_gt2_components[
            [
                "task",
                "component_group",
                "selected_proxy_metric",
                "selected_proxy_direction",
                "optimization_recommendation",
                "selected_proxy_snr",
                "selected_proxy_accuracy_spearman",
                "effective_component_weight",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    REPORT_MD.write_text("\n".join(body), encoding="utf-8")


def _append_logbook(summary: pd.DataFrame) -> None:
    LOGBOOK_MD.parent.mkdir(parents=True, exist_ok=True)
    best = summary.sort_values(["proxy_spearman", "accuracy_reference_spearman"], ascending=[False, False])
    section = [
        "",
        "### 2026-04-30 - Selected-proxy task optimization",
        "- Hypothesis: optimizing the current selected smooth proxies, with MMLU collapsed into one group, "
        "gives a more task-aligned objective than uncheatable BPB or flat MMLU-heavy averages.",
        f"- Command: `uv run --with matplotlib --with torch python {Path(__file__)}`",
        "- Result summary:",
        best[
            [
                "candidate_id",
                "proxy_spearman",
                "accuracy_reference_spearman",
                "raw_nearest_observed_tv",
                "raw_nearest_observed_choice",
                "raw_nearest_observed_accuracy",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "- Artifacts: "
        "`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
        "grp_300m_selected_proxy_tasks_20260430/`.",
        "",
    ]
    with LOGBOOK_MD.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(section))


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_data = _complete_qsplit_signal_frame()
    start_bank = _expanded_start_bank(args.random_starts)
    model_options = _model_options(args.block_variant)

    summary_rows: list[dict[str, Any]] = []
    optimum_rows: list[pd.DataFrame] = []
    component_frames: list[pd.DataFrame] = []
    for spec in TARGET_SPECS:
        print(f"Fitting {spec.slug}", flush=True)
        data, components = _add_target_columns(base_data, spec)
        data.to_csv(OUTPUT_DIR / f"{spec.slug}_dataset.csv", index=False)
        component_frames.append(components)
        summary, diag = _fit_selected_proxy_target(
            data,
            spec,
            family_scheme=args.family_scheme,
            model_options=model_options,
            start_bank=start_bank,
            coarse_top_k=args.coarse_top_k,
            method=args.method,
            prob_eps=args.prob_eps,
        )
        summary_rows.append(summary)
        optimum_rows.append(diag)

    summary = pd.DataFrame.from_records(summary_rows)
    optimum = pd.concat(optimum_rows, ignore_index=True)
    components = pd.concat(component_frames, ignore_index=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    optimum.to_csv(OPTIMUM_DIAGNOSTICS_CSV, index=False)
    components.to_csv(TARGET_COMPONENTS_CSV, index=False)
    _write_report(summary, optimum, components)
    _append_logbook(summary)
    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
