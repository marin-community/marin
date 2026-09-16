# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "plotly", "scipy", "scikit-learn", "tabulate", "torch"]
# ///
"""Fit GRP no-L2 surrogates to 300M IRT/factor-analysis targets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_grp_power_family_penalty_no_l2_retune import (
    REG_FIXED,
    VARIANT_NAME,
    _no_l2_param_keys,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.build_eval_signal_to_noise_all_metrics_300m import (
    DEFAULT_KEEP_DROP_CSV,
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
from experiments.domain_phase_mix.exploratory.two_phase_many.reproduce_300m_irt_factor_analysis import (
    fit_nonnegative_factor_model,
    horn_parallel_analysis,
    keep_irt_item,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    build_penalty_calibration_surrogate,
    optimize_penalty_calibration_model,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR.parent
RAW_MATRIX_DIR = SCRIPT_DIR / "raw_metric_matrix_300m"
OUTPUT_DIR = TWO_PHASE_MANY_DIR / "reference_outputs" / "grp_300m_irt_targets_20260501"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
OPTIMUM_DIAGNOSTICS_CSV = OUTPUT_DIR / "optimum_diagnostics.csv"
TARGET_COMPONENTS_CSV = OUTPUT_DIR / "target_components.csv"
ITEM_GROUP_VALIDATION_CSV = OUTPUT_DIR / "item_group_validation.csv"
REPORT_MD = OUTPUT_DIR / "report.md"
LOGBOOK_MD = Path(".agents/logbooks/benchmark-proxy-optimization.md")

SCALE = "300m_6b"
COHORT = "signal"
ACCURACY_SUFFIX = "__hard_accuracy_reference"
FIT_ROW_MASK_COLUMN = "is_qsplit240_core"
HYBRID_TASK_WEIGHT = 0.75
HYBRID_EVAL_WEIGHT = 0.25
SNR_GT2_THRESHOLD = 2.0
EPS = 1e-12

EXCLUDED_TASKS = frozenset({"averages", "eval"})
NOISE_MODES = ("variable", "fixed")
PRIMARY_NOISE_MODE = "variable"


@dataclass(frozen=True)
class ItemSpec:
    """One item passed into the Gaussian factor model."""

    metric: str
    display_name: str
    group: str
    direction: str
    accuracy_metric: str | None
    source: str
    weight: float = 1.0


@dataclass(frozen=True)
class FactorTarget:
    """One fitted IRT/factor target family."""

    family: str
    noise_mode: str
    items: tuple[ItemSpec, ...]
    theta: np.ndarray
    loadings: np.ndarray
    psi: np.ndarray
    communality: np.ndarray
    noise_share: np.ndarray
    h2_ceiling: np.ndarray
    k_horn: int
    k_mean: int
    item_frame: pd.DataFrame
    accuracy_reference: np.ndarray


@dataclass(frozen=True)
class TargetSpec:
    """One scalar target column to fit with GRP."""

    slug: str
    family: str
    noise_mode: str
    target_column: str
    display_name: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--coarse-top-k", type=int, default=DEFAULT_COARSE_TOP_K)
    parser.add_argument("--random-starts", type=int, default=DEFAULT_RANDOM_STARTS)
    parser.add_argument("--prob-eps", type=float, default=DEFAULT_PROB_EPS)
    parser.add_argument("--family-scheme", choices=FAMILY_SCHEMES, default=DEFAULT_FAMILY_SCHEME)
    parser.add_argument("--block-variant", choices=BLOCK_VARIANTS, default="full")
    parser.add_argument(
        "--target-family",
        action="append",
        choices=("eval_bpb_irt", "task_proxy_irt", "hybrid_irt"),
        default=[],
        help="Optional subset of target families to fit.",
    )
    parser.add_argument(
        "--noise-mode",
        action="append",
        choices=NOISE_MODES,
        default=[],
        help="Optional subset of noise modes to fit.",
    )
    return parser.parse_args()


def _load_raw_matrix(name: str) -> pd.DataFrame:
    path = RAW_MATRIX_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _load_signal_and_noise() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    signal = _load_raw_matrix("raw_metric_matrix_300m.csv")
    fixed_noise = _load_raw_matrix("noise_baseline_run00097_fixed_subset_300m.csv")
    variable_noise = _load_raw_matrix("noise_baseline_run00097_variable_subset_300m.csv")
    completed = signal[signal["status"].eq("completed")].copy().reset_index(drop=True)
    if len(completed) != 242:
        raise ValueError(f"Expected 242 completed signal rows, got {len(completed)}")
    if len(fixed_noise) != 10:
        raise ValueError(f"Expected 10 fixed-subset noise rows, got {len(fixed_noise)}")
    if len(variable_noise) != 10:
        raise ValueError(f"Expected 10 variable-subset noise rows, got {len(variable_noise)}")
    if int(completed[FIT_ROW_MASK_COLUMN].fillna(False).sum()) != 240:
        raise ValueError("Expected exactly 240 qsplit-core signal rows")
    for frame_name, frame in (("signal", completed), ("fixed_noise", fixed_noise), ("variable_noise", variable_noise)):
        for prefix in ("phase_0_", "phase_1_"):
            weight_columns = [column for column in frame.columns if column.startswith(prefix)]
            sums = frame[weight_columns].sum(axis=1)
            if not np.allclose(sums, 1.0, atol=1e-6):
                raise ValueError(f"{frame_name}: {prefix} weights do not sum to 1")
    return completed, {"fixed": fixed_noise, "variable": variable_noise}


def _semantic_group_from_task(task: str) -> str:
    lower = task.lower()
    if lower.startswith("mmlu_"):
        return "mmlu"
    if lower.startswith("arc_"):
        return "arc"
    if "hellaswag" in lower or lower.startswith("swag"):
        return "hellaswag_swag"
    if lower.startswith("truthfulqa"):
        return "truthfulqa"
    if lower.startswith("gsm8k"):
        return "gsm8k"
    if lower.startswith("humaneval"):
        return "humaneval"
    if any(token in lower for token in ("boolq", "copa", "csqa", "winogrande", "wsc", "socialiqa", "piqa", "sciq")):
        return "commonsense_qa"
    if "medmcqa" in lower or "openbookqa" in lower:
        return "commonsense_qa"
    return "other_task"


def _task_from_metric(metric: str) -> str:
    parts = metric.split("/")
    if metric.startswith("teacher_forced/gsm8k"):
        return "gsm8k"
    if metric.startswith("teacher_forced/humaneval"):
        return "humaneval"
    if len(parts) >= 2:
        return parts[1]
    return metric


def _accuracy_metric_for_task_metric(metric: str, signal: pd.DataFrame) -> str | None:
    task = _task_from_metric(metric)
    candidates = []
    if task == "gsm8k":
        candidates.extend(
            [
                "lm_eval/gsm8k/exact_match,flexible-extract",
                "lm_eval/gsm8k/exact_match,strict-match",
            ]
        )
    elif task == "humaneval":
        candidates.append("lm_eval/humaneval/pass@1,create_test")
    elif metric.startswith("mcq_smooth/"):
        candidates.append(f"lm_eval/{task}/acc")
    elif metric.startswith("lm_eval/"):
        candidates.append(f"lm_eval/{task}/acc")
    for candidate in candidates:
        if candidate in signal.columns and signal[candidate].notna().all():
            return candidate
    return None


def _component_group(task: str) -> str:
    if task == "mmlu_5shot":
        return "mmlu_standard"
    if task.startswith("mmlu_") and task.endswith("_5shot"):
        return "mmlu_standard"
    return task


def _selected_proxy_components() -> pd.DataFrame:
    keep_drop = pd.read_csv(DEFAULT_KEEP_DROP_CSV)
    selected = keep_drop[
        keep_drop["optimization_recommendation"].isin({"keep", "downweight"})
        & keep_drop["selected_proxy_metric"].notna()
        & keep_drop["accuracy_metric"].notna()
        & ~keep_drop["task"].isin(EXCLUDED_TASKS)
    ].copy()
    selected["component_group"] = selected["task"].map(_component_group)
    selected["semantic_group"] = selected["task"].map(_semantic_group_from_task)
    selected["component_raw_weight"] = selected["default_weight"].astype(float) * selected[
        "selected_proxy_score"
    ].astype(float)
    selected = selected[selected["component_raw_weight"] > 0.0].copy()
    if selected.empty:
        raise ValueError("No selected task proxies available")
    return selected.reset_index(drop=True)


def _oriented_values(frame: pd.DataFrame, item: ItemSpec) -> np.ndarray:
    values = pd.to_numeric(frame[item.metric], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(values).any():
        raise ValueError(f"Missing values for item {item.metric}")
    if item.direction == "maximize":
        return values
    if item.direction == "minimize":
        return -values
    raise ValueError(f"Unsupported direction {item.direction}")


def _zscore_oriented_items(
    signal: pd.DataFrame,
    items: tuple[ItemSpec, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.column_stack([_oriented_values(signal, item) for item in items])
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0)
    if np.any(stds <= EPS):
        bad = [items[i].metric for i in np.where(stds <= EPS)[0]]
        raise ValueError(f"Constant IRT item columns: {bad}")
    return (matrix - means) / stds, means, stds


def _zscore_noise_items(
    noise: pd.DataFrame,
    items: tuple[ItemSpec, ...],
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    matrix = np.column_stack([_oriented_values(noise, item) for item in items])
    return (matrix - means) / stds


def _noise_share_for_oriented_items(
    signal: pd.DataFrame,
    noise: pd.DataFrame,
    items: tuple[ItemSpec, ...],
) -> tuple[np.ndarray, np.ndarray]:
    signal_matrix = np.column_stack([_oriented_values(signal, item) for item in items])
    noise_matrix = np.column_stack([_oriented_values(noise, item) for item in items])
    noise_sd = noise_matrix.std(axis=0, ddof=1)
    signal_sd = signal_matrix.std(axis=0, ddof=1)
    if np.any(signal_sd <= EPS):
        bad = [items[i].metric for i in np.where(signal_sd <= EPS)[0]]
        raise ValueError(f"Zero signal scale for IRT item columns: {bad}")
    noise_share = (noise_sd / signal_sd) ** 2
    h2_ceiling = np.clip(1.0 - noise_share, 0.0, 1.0)
    return noise_share, h2_ceiling


def _group_weighted_series(
    frame: pd.DataFrame,
    components: pd.DataFrame,
    value_column: str,
    direction_column: str,
) -> pd.Series:
    columns: list[np.ndarray] = []
    weights = components["component_raw_weight"].to_numpy(dtype=float)
    weights = weights / weights.sum()
    for _, row in components.iterrows():
        metric = str(row[value_column])
        values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=np.float64)
        if np.isnan(values).any():
            raise ValueError(f"Missing values for grouped item {metric}")
        direction = str(row[direction_column])
        columns.append(values if direction == "maximize" else -values)
    matrix = np.column_stack(columns)
    values = matrix @ weights
    return pd.Series(values, index=frame.index)


def _build_task_group_items(
    signal: pd.DataFrame,
    noise_by_mode: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], tuple[ItemSpec, ...]]:
    selected = _selected_proxy_components()
    task_signal = signal.copy()
    task_noise = {mode: noise.copy() for mode, noise in noise_by_mode.items()}
    item_specs: list[ItemSpec] = []
    component_rows: list[dict[str, Any]] = []
    for group, group_frame in selected.groupby("component_group", sort=True):
        proxy_col = f"irt_task_group/{group}/selected_proxy"
        acc_col = f"irt_task_group/{group}/accuracy_reference"
        task_signal[proxy_col] = _group_weighted_series(
            task_signal,
            group_frame,
            "selected_proxy_metric",
            "selected_proxy_direction",
        )
        for noise in task_noise.values():
            noise[proxy_col] = _group_weighted_series(
                noise,
                group_frame,
                "selected_proxy_metric",
                "selected_proxy_direction",
            )

        acc_weights = group_frame["component_raw_weight"].to_numpy(dtype=float)
        acc_weights = acc_weights / acc_weights.sum()
        acc_values = []
        for _, row in group_frame.iterrows():
            metric = str(row["accuracy_metric"])
            values = pd.to_numeric(task_signal[metric], errors="coerce").to_numpy(dtype=np.float64)
            if np.isnan(values).any():
                raise ValueError(f"Missing values for accuracy reference {metric}")
            acc_values.append(values)
            component_rows.append(
                {
                    "target_family": "task_proxy_irt",
                    "component_group": group,
                    "task": row["task"],
                    "semantic_group": row["semantic_group"],
                    "selected_proxy_metric": row["selected_proxy_metric"],
                    "selected_proxy_direction": row["selected_proxy_direction"],
                    "selected_proxy_kind": row["selected_proxy_kind"],
                    "selected_proxy_snr": float(row["selected_proxy_snr"]),
                    "selected_proxy_accuracy_spearman": float(row["selected_proxy_accuracy_spearman"]),
                    "optimization_recommendation": row["optimization_recommendation"],
                    "accuracy_metric": metric,
                    "component_weight_within_group": float(acc_weights[len(acc_values) - 1]),
                }
            )
        task_signal[acc_col] = np.column_stack(acc_values) @ acc_weights
        item_specs.append(
            ItemSpec(
                metric=proxy_col,
                display_name=group,
                group=str(group_frame["semantic_group"].iloc[0]),
                direction="maximize",
                accuracy_metric=acc_col,
                source="task_proxy_group",
                weight=1.0,
            )
        )
    pd.DataFrame.from_records(component_rows).to_csv(OUTPUT_DIR / "task_proxy_source_components.csv", index=False)
    return task_signal, task_noise, tuple(item_specs)


def _build_eval_bpb_items(signal: pd.DataFrame) -> tuple[ItemSpec, ...]:
    items = []
    for metric in sorted(column for column in signal.columns if keep_irt_item(column)):
        task = _task_from_metric(metric)
        group = (
            "generic_eval"
            if metric.startswith(("eval/paloma/", "eval/uncheatable_eval/"))
            else _semantic_group_from_task(task)
        )
        items.append(
            ItemSpec(
                metric=metric,
                display_name=metric,
                group=group,
                direction="minimize",
                accuracy_metric=_accuracy_metric_for_task_metric(metric, signal),
                source="bpb_item",
                weight=1.0,
            )
        )
    if len(items) != 43:
        raise ValueError(f"Expected 43 BPB IRT items, got {len(items)}")
    return tuple(items)


def _accuracy_reference(signal: pd.DataFrame, items: tuple[ItemSpec, ...]) -> np.ndarray:
    columns = []
    for item in items:
        if item.accuracy_metric is None:
            continue
        if item.accuracy_metric not in signal.columns:
            continue
        values = pd.to_numeric(signal[item.accuracy_metric], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isnan(values).any():
            columns.append(values)
    if not columns:
        return np.full(len(signal), np.nan, dtype=np.float64)
    return np.column_stack(columns).mean(axis=1)


def _run_factor_target(
    family: str,
    noise_mode: str,
    signal: pd.DataFrame,
    noise: pd.DataFrame,
    items: tuple[ItemSpec, ...],
) -> FactorTarget:
    if len(items) < 2:
        raise ValueError(f"{family}/{noise_mode}: expected at least two IRT items")
    z, _, _ = _zscore_oriented_items(signal, items)
    k_horn, k_mean, _, _, _ = horn_parallel_analysis(z)
    k = max(1, int(k_horn))
    noise_share, h2_ceiling = _noise_share_for_oriented_items(signal, noise, items)
    loadings, psi, theta, communality = fit_nonnegative_factor_model(z, k, noise_share)
    rows = []
    for idx, item in enumerate(items):
        row = {
            "target_family": family,
            "noise_mode": noise_mode,
            "metric": item.metric,
            "display_name": item.display_name,
            "semantic_group": item.group,
            "direction": item.direction,
            "accuracy_metric": item.accuracy_metric,
            "source": item.source,
            "weight": item.weight,
            "noise_share": float(noise_share[idx]),
            "h2_ceiling": float(h2_ceiling[idx]),
            "communality": float(communality[idx]),
            "psi": float(psi[idx]),
            "dominant_factor": int(np.argmax(loadings[idx]) + 1),
            "max_loading": float(loadings[idx].max()),
        }
        for factor_idx in range(loadings.shape[1]):
            row[f"loading_factor_{factor_idx + 1}"] = float(loadings[idx, factor_idx])
        rows.append(row)
    return FactorTarget(
        family=family,
        noise_mode=noise_mode,
        items=items,
        theta=theta,
        loadings=loadings,
        psi=psi,
        communality=communality,
        noise_share=noise_share,
        h2_ceiling=h2_ceiling,
        k_horn=k_horn,
        k_mean=k_mean,
        item_frame=pd.DataFrame.from_records(rows),
        accuracy_reference=_accuracy_reference(signal, items),
    )


def _standardized(values: np.ndarray) -> np.ndarray:
    std = float(np.nanstd(values, ddof=0))
    if std <= EPS:
        raise ValueError("Cannot standardize constant values")
    return (values - float(np.nanmean(values))) / std


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(stats.spearmanr(x[mask], y[mask]).statistic)


def _factor_weighted_score(theta: np.ndarray, accuracy_reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = []
    for factor_idx in range(theta.shape[1]):
        corr = _spearman(theta[:, factor_idx], accuracy_reference)
        weights.append(max(0.0, 0.0 if np.isnan(corr) else corr))
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.sum() <= EPS:
        weight_array = np.ones(theta.shape[1], dtype=float)
    weight_array = weight_array / weight_array.sum()
    return theta @ weight_array, weight_array


def _add_factor_target_columns(signal: pd.DataFrame, target: FactorTarget) -> tuple[pd.DataFrame, list[TargetSpec]]:
    out = signal.copy()
    prefix = f"{target.family}_{target.noise_mode}"
    out[f"{prefix}_aggregate_mean_theta"] = target.theta.mean(axis=1)
    out[f"{prefix}{ACCURACY_SUFFIX}"] = target.accuracy_reference
    specs = [
        TargetSpec(
            slug=f"{prefix}_aggregate_mean_theta",
            family=target.family,
            noise_mode=target.noise_mode,
            target_column=f"{prefix}_aggregate_mean_theta",
            display_name=f"{target.family} ({target.noise_mode}) aggregate mean theta",
        )
    ]
    for factor_idx in range(min(3, target.theta.shape[1])):
        column = f"{prefix}_theta_{factor_idx + 1}"
        out[column] = target.theta[:, factor_idx]
        specs.append(
            TargetSpec(
                slug=column,
                family=target.family,
                noise_mode=target.noise_mode,
                target_column=column,
                display_name=f"{target.family} ({target.noise_mode}) theta {factor_idx + 1}",
            )
        )
    weighted, weights = _factor_weighted_score(target.theta, target.accuracy_reference)
    out[f"{prefix}_accuracy_weighted_theta"] = weighted
    (OUTPUT_DIR / "factor_accuracy_weights").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "target_family": target.family,
            "noise_mode": target.noise_mode,
            "factor": [f"theta_{idx + 1}" for idx in range(len(weights))],
            "accuracy_weight": weights,
        }
    ).to_csv(OUTPUT_DIR / "factor_accuracy_weights" / f"{prefix}.csv", index=False)
    specs.append(
        TargetSpec(
            slug=f"{prefix}_accuracy_weighted_theta",
            family=target.family,
            noise_mode=target.noise_mode,
            target_column=f"{prefix}_accuracy_weighted_theta",
            display_name=f"{target.family} ({target.noise_mode}) accuracy-weighted theta",
        )
    )
    return out, specs


def _add_hybrid_columns(
    signal: pd.DataFrame,
    task_target: FactorTarget,
    eval_target: FactorTarget,
) -> tuple[pd.DataFrame, list[TargetSpec]]:
    if task_target.noise_mode != eval_target.noise_mode:
        raise ValueError("Hybrid target requires matching noise modes")
    out = signal.copy()
    prefix = f"hybrid_irt_{task_target.noise_mode}"
    out[f"{prefix}{ACCURACY_SUFFIX}"] = task_target.accuracy_reference
    specs = []

    task_columns = {
        "aggregate_mean_theta": task_target.theta.mean(axis=1),
        "accuracy_weighted_theta": _factor_weighted_score(task_target.theta, task_target.accuracy_reference)[0],
    }
    eval_columns = {
        "aggregate_mean_theta": eval_target.theta.mean(axis=1),
        "accuracy_weighted_theta": _factor_weighted_score(eval_target.theta, task_target.accuracy_reference)[0],
    }
    for factor_idx in range(min(3, task_target.theta.shape[1], eval_target.theta.shape[1])):
        task_columns[f"theta_{factor_idx + 1}"] = task_target.theta[:, factor_idx]
        eval_columns[f"theta_{factor_idx + 1}"] = eval_target.theta[:, factor_idx]

    for suffix, task_values in task_columns.items():
        eval_values = eval_columns[suffix]
        column = f"{prefix}_{suffix}"
        out[column] = HYBRID_TASK_WEIGHT * _standardized(task_values) + HYBRID_EVAL_WEIGHT * _standardized(eval_values)
        specs.append(
            TargetSpec(
                slug=column,
                family="hybrid_irt",
                noise_mode=task_target.noise_mode,
                target_column=column,
                display_name=f"hybrid IRT ({task_target.noise_mode}) {suffix}",
            )
        )
    return out, specs


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
            FIT_ROW_MASK_COLUMN,
        )
        if column in data.columns
    ]
    frame = data[id_columns + weight_columns + [objective_column]].rename(columns={objective_column: "objective_metric"})
    frame["objective_metric_key"] = objective_column
    return frame.dropna(axis=1, how="all").reset_index(drop=True)


def _objective(spec: TargetSpec) -> AggregateObjective:
    return AggregateObjective(
        slug=spec.slug,
        source_column="objective_metric",
        display_name=spec.display_name,
        family="irt_target",
        higher_is_better=True,
        transform="identity",
    )


def _fit_one_target(
    all_data: pd.DataFrame,
    spec: TargetSpec,
    *,
    family_scheme: str,
    model_options: dict[str, bool],
    start_bank: tuple[dict[str, float], ...],
    coarse_top_k: int,
    method: str,
    prob_eps: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fit_data = all_data[all_data[FIT_ROW_MASK_COLUMN].fillna(False)].copy().reset_index(drop=True)
    if len(fit_data) != 240:
        raise ValueError(f"{spec.slug}: expected 240 fit rows, got {len(fit_data)}")
    objective = _objective(spec)
    fit_packet = _packet_from_frame(_packet_frame(fit_data, spec.target_column), objective, prob_eps, family_scheme)
    observed_packet = _packet_from_frame(_packet_frame(all_data, spec.target_column), objective, prob_eps, family_scheme)

    coarse, best, refine = _refine_rows(fit_packet, start_bank, coarse_top_k, method, model_options)
    params = {key: float(best[key]) for key in _no_l2_param_keys()}
    params["reg"] = REG_FIXED
    model = build_penalty_calibration_surrogate(
        fit_packet,
        params=params,
        variant_name=VARIANT_NAME,
        **model_options,
    ).fit(fit_packet.base.w, fit_packet.base.y)
    pred_rows = _prediction_rows(fit_packet, params, objective)

    raw_result, phase0, phase1 = optimize_penalty_calibration_model(fit_packet, model, seed=0)
    raw_weights = np.stack([phase0, phase1], axis=0)
    score_model = proxy_benchmark.ScoreModelAdapter(
        lambda weights: _model_target_to_metric(model.predict(weights), objective)
    )
    accuracy_column = f"{spec.family}_{spec.noise_mode}{ACCURACY_SUFFIX}"
    if spec.family == "hybrid_irt":
        accuracy_column = f"hybrid_irt_{spec.noise_mode}{ACCURACY_SUFFIX}"
    actual_target_observed = all_data[spec.target_column].to_numpy(dtype=float)
    actual_accuracy_observed = all_data[accuracy_column].to_numpy(dtype=float)
    diag, weight_map = proxy_benchmark._candidate_diagnostics(
        candidate_id=spec.slug,
        candidate_kind="direct_grp_irt_target",
        target=spec.target_column,
        packet=observed_packet,
        score_model=score_model,
        actual_choice=actual_target_observed,
        actual_accuracy=actual_accuracy_observed,
        actual_primary=actual_target_observed,
        raw_metric=float(_model_target_to_metric(float(raw_result.fun), objective)),
        raw_weights=raw_weights,
    )

    target_dir = OUTPUT_DIR / "grp_candidates" / spec.slug
    target_dir.mkdir(parents=True, exist_ok=True)
    coarse.to_csv(target_dir / "coarse.csv", index=False)
    refine.to_csv(target_dir / "refine.csv", index=False)
    pd.DataFrame([params]).to_csv(target_dir / "params.csv", index=False)
    pred_rows.to_csv(target_dir / "oof_predictions.csv", index=False)
    _plot_predictions(pred_rows, target_dir / "predicted_vs_actual.html", spec.display_name)
    _plot_residuals(pred_rows, target_dir / "residuals.html", spec.display_name)
    diag.to_csv(target_dir / "optimum_diagnostics.csv", index=False)
    proxy_benchmark._write_optimum_weights(target_dir / "optimum_weights.csv", observed_packet, weight_map)
    raw_plot = proxy_benchmark._plot_phase_comparison(
        target_dir,
        observed_packet,
        "raw",
        f"Raw optimum: {spec.display_name}",
    )
    hull_plot = proxy_benchmark._plot_phase_comparison(
        target_dir,
        observed_packet,
        "top8actual_hull",
        f"Top8 hull: {spec.display_name}",
    )

    fit_predicted = pred_rows["predicted_metric"].to_numpy(dtype=float)
    fit_actual = pred_rows["actual_metric"].to_numpy(dtype=float)
    fit_accuracy = fit_data[accuracy_column].to_numpy(dtype=float)
    summary: dict[str, Any] = {
        "candidate_id": spec.slug,
        "target_family": spec.family,
        "noise_mode": spec.noise_mode,
        "target_column": spec.target_column,
        "display_name": spec.display_name,
        "n_fit_rows": len(fit_data),
        "n_observed_rows": len(all_data),
        "family_scheme": family_scheme,
        "block_variant": json.dumps(model_options, sort_keys=True),
        "raw_phase_plot": str(raw_plot),
        "top8_hull_phase_plot": str(hull_plot),
        **proxy_benchmark._metric_summary(fit_actual, fit_predicted, prefix="target_"),
        **proxy_benchmark._metric_summary(fit_accuracy, fit_predicted, prefix="accuracy_reference_"),
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


def _heldout_reference(
    signal: pd.DataFrame,
    items: tuple[ItemSpec, ...],
) -> tuple[np.ndarray, np.ndarray]:
    z, _, _ = _zscore_oriented_items(signal, items)
    proxy_reference = z.mean(axis=1)
    accuracy_reference = _accuracy_reference(signal, items)
    return proxy_reference, accuracy_reference


def _item_group_validation_rows(
    family: str,
    noise_mode: str,
    signal: pd.DataFrame,
    noise: pd.DataFrame,
    items: tuple[ItemSpec, ...],
) -> list[dict[str, Any]]:
    rows = []
    groups = sorted({item.group for item in items})
    for heldout_group in groups:
        train_items = tuple(item for item in items if item.group != heldout_group)
        heldout_items = tuple(item for item in items if item.group == heldout_group)
        if len(train_items) < 2 or not heldout_items:
            continue
        target = _run_factor_target(f"{family}_leaveout_{heldout_group}", noise_mode, signal, noise, train_items)
        train_score = target.theta.mean(axis=1)
        heldout_proxy, heldout_accuracy = _heldout_reference(signal, heldout_items)
        rows.append(
            {
                "target_family": family,
                "noise_mode": noise_mode,
                "heldout_group": heldout_group,
                "train_item_count": len(train_items),
                "heldout_item_count": len(heldout_items),
                "target_vs_heldout_proxy_spearman": _spearman(train_score, heldout_proxy),
                "target_vs_heldout_accuracy_spearman": _spearman(train_score, heldout_accuracy),
            }
        )
    return rows


def _write_factor_outputs(targets: list[FactorTarget]) -> None:
    target_dir = OUTPUT_DIR / "irt_targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    leaderboards = []
    for target in targets:
        prefix = f"{target.family}_{target.noise_mode}"
        target.item_frame.to_csv(target_dir / f"{prefix}_loadings.csv", index=False)
        frames.append(target.item_frame)
        leaderboard = pd.DataFrame(
            {
                "run_name": target_signal_run_names,
                "target_family": target.family,
                "noise_mode": target.noise_mode,
                "aggregate_mean_theta": target.theta.mean(axis=1),
                "accuracy_reference": target.accuracy_reference,
            }
        )
        for factor_idx in range(target.theta.shape[1]):
            leaderboard[f"theta_{factor_idx + 1}"] = target.theta[:, factor_idx]
        leaderboard = leaderboard.sort_values("aggregate_mean_theta", ascending=False)
        leaderboard.to_csv(target_dir / f"{prefix}_leaderboard.csv", index=False)
        leaderboards.append(leaderboard)
    pd.concat(frames, ignore_index=True).to_csv(TARGET_COMPONENTS_CSV, index=False)
    pd.concat(leaderboards, ignore_index=True).to_csv(OUTPUT_DIR / "irt_leaderboards.csv", index=False)


target_signal_run_names: np.ndarray


def _write_report(summary: pd.DataFrame, optimum: pd.DataFrame, item_validation: pd.DataFrame) -> None:
    prior_sections = []
    previous_choice = (
        TWO_PHASE_MANY_DIR / "reference_outputs" / "grp_300m_mean_choice_prob_norm_no_mmlu_pro_20260428" / "summary.json"
    )
    selected_proxy = TWO_PHASE_MANY_DIR / "reference_outputs" / "grp_300m_selected_proxy_tasks_20260430" / "summary.csv"
    if previous_choice.exists():
        previous = json.loads(previous_choice.read_text(encoding="utf-8"))
        if isinstance(previous, list):
            previous = previous[0] if previous else {}
        prior_sections.append(
            "- Direct mean `choice_prob_norm` GRP previously reached "
            f"OOF Spearman `{previous.get('metric_oof_spearman', previous.get('oof_spearman', 'n/a'))}` "
            "and had raw-optimum collapse; "
            "this remains the failure mode to beat."
        )
    if selected_proxy.exists():
        selected = pd.read_csv(selected_proxy)
        best = selected.sort_values("proxy_spearman", ascending=False).head(1)
        if not best.empty:
            row = best.iloc[0]
            prior_sections.append(
                "- Selected-proxy GRP best prior row: "
                f"`{row['candidate_id']}` with proxy Spearman `{row['proxy_spearman']:.3f}`, "
                f"accuracy-reference Spearman `{row['accuracy_reference_spearman']:.3f}`, "
                f"raw nearest-observed TV `{row['raw_nearest_observed_tv']:.3f}`."
            )

    summary_columns = [
        "candidate_id",
        "target_family",
        "noise_mode",
        "target_spearman",
        "target_regret_at_1",
        "target_lower_tail_optimism",
        "accuracy_reference_spearman",
        "accuracy_reference_regret_at_1",
        "raw_predicted_proxy_metric",
        "raw_nearest_observed_tv",
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
    display_summary = summary.sort_values(
        ["noise_mode", "target_family", "target_spearman"],
        ascending=[False, True, False],
    )
    variable_task = summary[
        summary["noise_mode"].eq(PRIMARY_NOISE_MODE) & summary["target_family"].isin({"task_proxy_irt", "hybrid_irt"})
    ].copy()
    best_task_accuracy = variable_task.sort_values(
        ["accuracy_reference_spearman", "target_spearman"],
        ascending=[False, False],
    ).head(1)
    best_task_fit = variable_task.sort_values(
        ["target_spearman", "accuracy_reference_spearman"],
        ascending=[False, False],
    ).head(1)
    unsafe_raw_count = int((variable_task["raw_nearest_observed_tv"] > 0.25).sum())
    best_task_accuracy_text = "No variable-noise task/hybrid candidates were fit."
    best_task_fit_text = "No variable-noise task/hybrid candidates were fit."
    if not best_task_accuracy.empty:
        row = best_task_accuracy.iloc[0]
        best_task_accuracy_text = (
            f"Best task-aligned candidate by hard-accuracy reference is `{row['candidate_id']}` "
            f"(accuracy-reference Spearman `{row['accuracy_reference_spearman']:.3f}`, "
            f"target Spearman `{row['target_spearman']:.3f}`, raw nearest-observed TV "
            f"`{row['raw_nearest_observed_tv']:.3f}`)."
        )
    if not best_task_fit.empty:
        row = best_task_fit.iloc[0]
        best_task_fit_text = (
            f"Best variable-noise task/hybrid target fit is `{row['candidate_id']}` "
            f"(target Spearman `{row['target_spearman']:.3f}`, accuracy-reference Spearman "
            f"`{row['accuracy_reference_spearman']:.3f}`)."
        )
    body = [
        "# 300M GRP Modeling of IRT Scores",
        "",
        "## Setup",
        "",
        "- Rows: 242 completed 300M/6B signal rows; GRP fits use the 240 qsplit-core rows.",
        "- Primary noise estimate: variable-subset `run_00097`; fixed-subset is a sensitivity check.",
        "- Targets: BPB-item IRT, task-proxy IRT, and a 75/25 task/eval hybrid.",
        "- Hard accuracy is used only as reference/validation, not as a fitted target.",
        "",
        "## Prior Baselines",
        "",
        *(prior_sections or ["- Prior baseline files were not found in this workspace."]),
        "",
        "## Recommendation",
        "",
        f"- {best_task_fit_text}",
        f"- {best_task_accuracy_text}",
        f"- `{unsafe_raw_count}/{len(variable_task)}` variable-noise task/hybrid raw optima have "
        "nearest-observed TV > 0.25, so raw GRP optima remain diagnostic-only.",
        "- IRT improves the denoised target construction and rank fit, but this run does not produce a "
        "deployable raw optimum.",
        "- Use observed top candidates or top8-hull/trustblend candidates for validation; do not launch a "
        "raw IRT-GRP optimum without a separate sanity constraint.",
        "",
        "## GRP Fit Summary",
        "",
        display_summary[summary_columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Optimum Diagnostics",
        "",
        optimum[optimum["opt_kind"].isin(["best_observed", "predicted_best_observed", "raw", "top8actual_hull"])][
            optimum_columns
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Item-Group Validation",
        "",
        item_validation.sort_values(["target_family", "noise_mode", "heldout_group"]).to_markdown(
            index=False,
            floatfmt=".6f",
        ),
        "",
        "## Interpretation",
        "",
        "- IRT improves target construction if its GRP fit has stronger rank metrics than direct choice-prob targets.",
        "- It fixes optimization only if raw optima stop moving far from the observed mixture manifold.",
        "- Top8-hull and trustblend candidates remain the deployable candidates when raw nearest-observed TV is high.",
        "",
    ]
    REPORT_MD.write_text("\n".join(body), encoding="utf-8")


def _append_logbook(summary: pd.DataFrame) -> None:
    LOGBOOK_MD.parent.mkdir(parents=True, exist_ok=True)
    best = summary.sort_values(["target_spearman", "accuracy_reference_spearman"], ascending=[False, False]).head(8)
    section = [
        "",
        "### 2026-05-01 - GRP-style modeling of IRT scores",
        "- Hypothesis: denoised IRT/factor targets provide a better task-optimization surrogate target than raw "
        "`mean(choice_prob_norm)`.",
        f"- Command: `uv run --with matplotlib --with torch python {Path(__file__)}`",
        "- Result summary:",
        best[
            [
                "candidate_id",
                "target_spearman",
                "accuracy_reference_spearman",
                "raw_nearest_observed_tv",
                "raw_nearest_observed_accuracy",
                "top8actual_hull_nearest_observed_tv",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "- Artifacts: "
        "`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
        "grp_300m_irt_targets_20260501/`.",
        "",
    ]
    with LOGBOOK_MD.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(section))


def main() -> None:
    global target_signal_run_names
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signal, noise_by_mode = _load_signal_and_noise()
    target_signal_run_names = signal["run_name"].astype(str).to_numpy()

    task_signal, task_noise_by_mode, task_items = _build_task_group_items(signal, noise_by_mode)
    eval_items = _build_eval_bpb_items(signal)
    target_families = set(args.target_family or ["eval_bpb_irt", "task_proxy_irt", "hybrid_irt"])
    noise_modes = tuple(args.noise_mode or NOISE_MODES)

    factor_targets: list[FactorTarget] = []
    data_with_targets = task_signal.copy()
    target_specs: list[TargetSpec] = []
    item_validation_rows: list[dict[str, Any]] = []
    by_family_noise: dict[tuple[str, str], FactorTarget] = {}

    for noise_mode in noise_modes:
        if "eval_bpb_irt" in target_families or "hybrid_irt" in target_families:
            target = _run_factor_target("eval_bpb_irt", noise_mode, signal, noise_by_mode[noise_mode], eval_items)
            factor_targets.append(target)
            by_family_noise[(target.family, noise_mode)] = target
            data_with_targets, specs = _add_factor_target_columns(data_with_targets, target)
            target_specs.extend(specs if "eval_bpb_irt" in target_families else [])
            item_validation_rows.extend(
                _item_group_validation_rows("eval_bpb_irt", noise_mode, signal, noise_by_mode[noise_mode], eval_items)
            )
        if "task_proxy_irt" in target_families or "hybrid_irt" in target_families:
            target = _run_factor_target(
                "task_proxy_irt",
                noise_mode,
                task_signal,
                task_noise_by_mode[noise_mode],
                task_items,
            )
            factor_targets.append(target)
            by_family_noise[(target.family, noise_mode)] = target
            data_with_targets, specs = _add_factor_target_columns(data_with_targets, target)
            target_specs.extend(specs if "task_proxy_irt" in target_families else [])
            item_validation_rows.extend(
                _item_group_validation_rows(
                    "task_proxy_irt",
                    noise_mode,
                    task_signal,
                    task_noise_by_mode[noise_mode],
                    task_items,
                )
            )
        if "hybrid_irt" in target_families:
            task_target = by_family_noise[("task_proxy_irt", noise_mode)]
            eval_target = by_family_noise[("eval_bpb_irt", noise_mode)]
            data_with_targets, specs = _add_hybrid_columns(data_with_targets, task_target, eval_target)
            target_specs.extend(specs)

    _write_factor_outputs(factor_targets)
    pd.DataFrame.from_records(item_validation_rows).to_csv(ITEM_GROUP_VALIDATION_CSV, index=False)
    data_with_targets.to_csv(OUTPUT_DIR / "target_dataset.csv", index=False)

    start_bank = _expanded_start_bank(args.random_starts)
    model_options = _model_options(args.block_variant)
    summary_rows: list[dict[str, Any]] = []
    optimum_rows: list[pd.DataFrame] = []
    for spec in target_specs:
        print(f"Fitting {spec.slug}", flush=True)
        summary, diag = _fit_one_target(
            data_with_targets,
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
    item_validation = pd.read_csv(ITEM_GROUP_VALIDATION_CSV)
    summary.to_csv(SUMMARY_CSV, index=False)
    optimum.to_csv(OPTIMUM_DIAGNOSTICS_CSV, index=False)
    _write_report(summary, optimum, item_validation)
    _append_logbook(summary)
    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
