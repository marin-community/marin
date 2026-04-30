# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "plotly", "scipy", "scikit-learn", "tabulate", "torch"]
# ///
"""Fit and optimize task-ensemble GRP models for 300M choice_prob_norm."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_grp_power_family_penalty_no_l2_retune import (
    REG_FIXED,
    VARIANT_NAME,
    _no_l2_param_keys,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.build_metric_registry import (
    METRICS_WIDE_CSV,
    WEIGHT_PREFIXES,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.fit_grp_300m_mean_choice_prob_norm import (
    COHORT,
    DISPLAY_NAME,
    EXCLUDED_SUITES,
    TARGET_COLUMN,
    _choice_prob_columns,
    _plot_optimum_phase_comparison,
    _suite_name,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.fit_grp_300m_mmlu_choice_prob_norm import (
    DEFAULT_COARSE_TOP_K,
    DEFAULT_METHOD,
    DEFAULT_PROB_EPS,
    DEFAULT_RANDOM_STARTS,
    RUN_SET,
    SCALE,
    _coarse_rows,
    _metric_oof_summary,
    _model_options,
    _refine_rows,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.fit_grp_no_l2_benchmark_aggregates import (
    AggregateObjective,
    _expanded_start_bank,
    _family_shares,
    _model_target_to_metric,
    _packet_from_frame,
    _plot_predictions,
    _plot_residuals,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    optimize_generic_family_convex_hull,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    build_penalty_calibration_surrogate,
    optimize_penalty_calibration_model,
)
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance

CV_SEED = 0
DEFAULT_FAMILY_SCHEME = "default"
DEFAULT_BLOCK_VARIANT = "full"
OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "reference_outputs"
    / "grp_300m_choice_prob_norm_ensembles_no_mmlu_pro_20260428"
)
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
TASK_FITS_CSV = OUTPUT_DIR / "task_fits.csv"
OPTIMUM_DIAGNOSTICS_CSV = OUTPUT_DIR / "optimum_diagnostics.csv"
REPORT_MD = OUTPUT_DIR / "report.md"
TARGET_COLUMNS_JSON = OUTPUT_DIR / "target_columns.json"
SUPPORT_TOP_K = 12
TOP_CANDIDATE_SUPPORTS = 16


@dataclass(frozen=True)
class EnsembleFit:
    """Fitted task-ensemble state."""

    variant: str
    packet: Any
    task_columns: tuple[str, ...]
    actual_metric: np.ndarray
    oof_predicted_metric: np.ndarray
    models: tuple[Any, ...]
    task_fit_rows: tuple[dict[str, Any], ...]


class EnsemblePredictionModel:
    """Small adapter exposing averaged task probability as a minimization model."""

    def __init__(self, models: tuple[Any, ...], objective: AggregateObjective):
        self.models = tuple(models)
        self.objective = objective
        self.coef_ = np.ones(1, dtype=float)
        self.intercept_ = 0.0

    def predict_metric(self, weights: np.ndarray) -> np.ndarray:
        target_predictions = np.column_stack([model.predict(weights) for model in self.models])
        return np.mean(_model_target_to_metric(target_predictions, self.objective), axis=1)

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return -self.predict_metric(weights)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-starts", type=int, default=DEFAULT_RANDOM_STARTS)
    parser.add_argument("--coarse-top-k", type=int, default=DEFAULT_COARSE_TOP_K)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--prob-eps", type=float, default=DEFAULT_PROB_EPS)
    parser.add_argument("--family-scheme", default=DEFAULT_FAMILY_SCHEME)
    parser.add_argument("--block-variant", default=DEFAULT_BLOCK_VARIANT)
    parser.add_argument("--max-tasks", type=int, default=0, help="Debug limit; 0 means all tasks.")
    return parser.parse_args()


def _aggregate_objective() -> AggregateObjective:
    return AggregateObjective(
        slug="mean_choice_prob_norm_no_mmlu_pro_logit",
        source_column="objective_metric",
        display_name=DISPLAY_NAME,
        family="choice_prob_norm",
        higher_is_better=True,
        transform="logit_probability",
    )


def _base_frame(max_tasks: int) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frame = pd.read_csv(METRICS_WIDE_CSV, low_memory=False)
    target_columns = tuple(_choice_prob_columns(frame))
    if max_tasks > 0:
        target_columns = target_columns[: int(max_tasks)]
    mask = frame["scale"].eq(SCALE) & frame["cohort"].eq(COHORT) & frame["is_qsplit240_core"].fillna(False)
    source = frame.loc[mask].copy()
    source = source.dropna(subset=list(target_columns)).copy()
    source[TARGET_COLUMN] = source[list(target_columns)].mean(axis=1)

    weight_columns = sorted(column for column in source.columns if column.startswith(WEIGHT_PREFIXES))
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
        if column in source.columns
    ]
    out = source[id_columns + weight_columns + list(target_columns) + [TARGET_COLUMN]].copy()
    out = out.dropna(axis=1, how="all").reset_index(drop=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "ensemble_fit_dataset.csv", index=False)
    TARGET_COLUMNS_JSON.write_text(
        json.dumps(
            {
                "target": TARGET_COLUMN,
                "excluded_suites": sorted(EXCLUDED_SUITES),
                "n_rows": len(out),
                "n_target_columns": len(target_columns),
                "target_columns": list(target_columns),
                "target_suites": [_suite_name(column) for column in target_columns],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return out, target_columns


def _task_frame(base: pd.DataFrame, column: str) -> pd.DataFrame:
    frame = base.copy()
    frame["objective_metric"] = frame[column]
    frame["objective_metric_key"] = column
    return frame.drop(columns=[TARGET_COLUMN, *[c for c in base.columns if c.startswith("lm_eval/")]], errors="ignore")


def _aggregate_frame(base: pd.DataFrame) -> pd.DataFrame:
    frame = base.copy()
    frame["objective_metric"] = frame[TARGET_COLUMN]
    frame["objective_metric_key"] = TARGET_COLUMN
    return frame.drop(columns=[TARGET_COLUMN, *[c for c in base.columns if c.startswith("lm_eval/")]], errors="ignore")


def _fit_model(packet, params: dict[str, float], model_options: dict[str, bool]):
    full_params = dict(params)
    full_params["reg"] = REG_FIXED
    return build_penalty_calibration_surrogate(
        packet,
        params=full_params,
        variant_name=VARIANT_NAME,
        **model_options,
    ).fit(packet.base.w, packet.base.y)


def _oof_task_predictions(
    packets: tuple[Any, ...],
    params_by_task: tuple[dict[str, float], ...],
    objective: AggregateObjective,
    model_options: dict[str, bool],
) -> np.ndarray:
    n_rows = len(packets[0].base.y)
    task_predictions = np.zeros((n_rows, len(packets)), dtype=float)
    kf = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    for train_idx, test_idx in kf.split(packets[0].base.w):
        for task_idx, packet in enumerate(packets):
            model = build_penalty_calibration_surrogate(
                packet,
                params={**params_by_task[task_idx], "reg": REG_FIXED},
                variant_name=VARIANT_NAME,
                **model_options,
            ).fit(packet.base.w[train_idx], packet.base.y[train_idx])
            target_pred = model.predict(packet.base.w[test_idx])
            task_predictions[test_idx, task_idx] = _model_target_to_metric(target_pred, objective)
    return np.mean(task_predictions, axis=1)


def _fit_independent(
    *,
    base: pd.DataFrame,
    task_columns: tuple[str, ...],
    objective: AggregateObjective,
    family_scheme: str,
    model_options: dict[str, bool],
    random_starts: int,
    prob_eps: float,
) -> EnsembleFit:
    start_bank = _expanded_start_bank(random_starts)
    packets: list[Any] = []
    params_by_task: list[dict[str, float]] = []
    task_fit_rows: list[dict[str, Any]] = []
    for task_idx, column in enumerate(task_columns):
        packet = _packet_from_frame(_task_frame(base, column), objective, prob_eps, family_scheme)
        coarse = _coarse_rows(packet, start_bank, model_options)
        best = dict(coarse.iloc[0])
        params = {key: float(best[key]) for key in _no_l2_param_keys()}
        packets.append(packet)
        params_by_task.append(params)
        task_fit_rows.append(
            {
                "variant": "independent_task_grps",
                "task_index": task_idx,
                "metric_column": column,
                "suite": _suite_name(column),
                "start_id": int(best["start_id"]),
                "task_target_cv_rmse": float(best["target_cv_rmse"]),
                "task_target_cv_spearman": float(best["target_cv_spearman"]),
                "task_objective": float(best["objective"]),
                **params,
            }
        )

    oof_pred = _oof_task_predictions(tuple(packets), tuple(params_by_task), objective, model_options)
    models = tuple(
        _fit_model(packet, params, model_options) for packet, params in zip(packets, params_by_task, strict=True)
    )
    return EnsembleFit(
        variant="independent_task_grps",
        packet=packets[0],
        task_columns=task_columns,
        actual_metric=base[TARGET_COLUMN].to_numpy(dtype=float),
        oof_predicted_metric=oof_pred,
        models=models,
        task_fit_rows=tuple(task_fit_rows),
    )


def _fit_shared_body(
    *,
    base: pd.DataFrame,
    task_columns: tuple[str, ...],
    objective: AggregateObjective,
    family_scheme: str,
    model_options: dict[str, bool],
    random_starts: int,
    coarse_top_k: int,
    method: str,
    prob_eps: float,
) -> EnsembleFit:
    start_bank = _expanded_start_bank(random_starts)
    aggregate_packet = _packet_from_frame(_aggregate_frame(base), objective, prob_eps, family_scheme)
    _coarse, best, _refine = _refine_rows(aggregate_packet, start_bank, coarse_top_k, method, model_options)
    shared_params = {key: float(best[key]) for key in _no_l2_param_keys()}
    packets = tuple(
        _packet_from_frame(_task_frame(base, column), objective, prob_eps, family_scheme) for column in task_columns
    )
    params_by_task = tuple(dict(shared_params) for _ in packets)
    oof_pred = _oof_task_predictions(packets, params_by_task, objective, model_options)
    models = tuple(_fit_model(packet, shared_params, model_options) for packet in packets)
    task_fit_rows = tuple(
        {
            "variant": "shared_body_task_heads",
            "task_index": task_idx,
            "metric_column": column,
            "suite": _suite_name(column),
            "start_id": int(best["start_id"]),
            "task_target_cv_rmse": np.nan,
            "task_target_cv_spearman": np.nan,
            "task_objective": np.nan,
            **shared_params,
        }
        for task_idx, column in enumerate(task_columns)
    )
    return EnsembleFit(
        variant="shared_body_task_heads",
        packet=aggregate_packet,
        task_columns=task_columns,
        actual_metric=base[TARGET_COLUMN].to_numpy(dtype=float),
        oof_predicted_metric=oof_pred,
        models=models,
        task_fit_rows=task_fit_rows,
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    out = np.exp(shifted)
    return out / np.sum(out)


def _optimize_on_support(
    *,
    ensemble_model: EnsemblePredictionModel,
    n_domains: int,
    support0: np.ndarray,
    support1: np.ndarray,
    start_weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    def weights_from_z(z: np.ndarray) -> np.ndarray:
        p0 = np.zeros(n_domains, dtype=float)
        p1 = np.zeros(n_domains, dtype=float)
        p0[support0] = _softmax(z[: len(support0)])
        p1[support1] = _softmax(z[len(support0) :])
        return np.stack([p0, p1], axis=0)

    start0 = np.log(np.maximum(start_weights[0, support0], 1e-10))
    start1 = np.log(np.maximum(start_weights[1, support1], 1e-10))
    start = np.concatenate([start0, start1])

    def objective(z: np.ndarray) -> float:
        return float(ensemble_model.predict(weights_from_z(np.asarray(z, dtype=float))[None, :, :])[0])

    starts = [start, np.zeros_like(start)]
    best_result = None
    for current_start in starts:
        result = minimize(objective, current_start, method="L-BFGS-B", options={"maxiter": 200, "ftol": 1e-10})
        if best_result is None or float(result.fun) < float(best_result.fun):
            best_result = result
    if best_result is None:
        raise RuntimeError("Support optimization failed")
    best_weights = weights_from_z(np.asarray(best_result.x, dtype=float))
    return float(-ensemble_model.predict(best_weights[None, :, :])[0]), best_weights


def _candidate_optimum(fit: EnsembleFit, objective: AggregateObjective) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    ensemble_model = EnsemblePredictionModel(fit.models, objective)
    packet = fit.packet
    actual = fit.actual_metric
    best_idx = int(np.argmax(actual))
    predicted_observed = ensemble_model.predict_metric(packet.base.w)
    predicted_best_observed_idx = int(np.argmax(predicted_observed))

    candidates: list[tuple[str, np.ndarray]] = [
        ("best_observed", packet.base.w[best_idx]),
        ("predicted_best_observed", packet.base.w[predicted_best_observed_idx]),
    ]
    raw_task_values: list[float] = []
    for task_idx, model in enumerate(fit.models):
        raw_result, phase0, phase1 = optimize_penalty_calibration_model(packet, model, n_random=0, seed=CV_SEED)
        raw_weights = np.stack([phase0, phase1], axis=0)
        candidates.append((f"task_raw_{task_idx:03d}", raw_weights))
        raw_task_values.append(float(raw_result.fun))

    candidate_values = np.asarray(
        [ensemble_model.predict_metric(weights[None, :, :])[0] for _name, weights in candidates],
        dtype=float,
    )
    best_candidate_idx = int(np.argmax(candidate_values))
    best_candidate_weights = candidates[best_candidate_idx][1]

    top_candidate_order = np.argsort(-candidate_values)[: min(TOP_CANDIDATE_SUPPORTS, len(candidates))]
    support0 = set(np.argsort(-best_candidate_weights[0])[:SUPPORT_TOP_K].tolist())
    support1 = set(np.argsort(-best_candidate_weights[1])[:SUPPORT_TOP_K].tolist())
    for idx in top_candidate_order:
        weights = candidates[int(idx)][1]
        support0.update(np.where(weights[0] > 1e-4)[0].tolist())
        support1.update(np.where(weights[1] > 1e-4)[0].tolist())
    support0_array = np.asarray(sorted(support0), dtype=int)
    support1_array = np.asarray(sorted(support1), dtype=int)
    raw_refined_metric, raw_refined_weights = _optimize_on_support(
        ensemble_model=ensemble_model,
        n_domains=packet.base.m,
        support0=support0_array,
        support1=support1_array,
        start_weights=best_candidate_weights,
    )

    top_indices = np.argsort(-actual)[: min(8, len(actual))]
    hull_value, hull_coeffs, hull_weights = optimize_generic_family_convex_hull(
        ensemble_model,
        packet.base.w[top_indices],
        start_indices=np.arange(min(8, len(top_indices)), dtype=int),
    )

    weight_map = {
        "best_observed": packet.base.w[best_idx],
        "predicted_best_observed": packet.base.w[predicted_best_observed_idx],
        "raw_candidate": best_candidate_weights,
        "raw_refined": raw_refined_weights,
        "top8actual_hull": hull_weights,
    }
    rows: list[dict[str, Any]] = []
    best_observed_metric = float(actual[best_idx])

    def add_row(
        opt_kind: str, weights: np.ndarray, predicted_metric: float, extra: dict[str, Any] | None = None
    ) -> None:
        distances = average_phase_tv_distance(packet.base.w, weights[None, :, :])
        nearest_idx = int(np.argmin(distances))
        row = {
            "variant": fit.variant,
            "opt_kind": opt_kind,
            "predicted_metric": float(predicted_metric),
            "optimism_vs_best_observed_metric": float(predicted_metric - best_observed_metric),
            "nearest_observed_run_name": str(packet.base.frame.loc[nearest_idx, packet.base.name_col]),
            "nearest_observed_metric": float(actual[nearest_idx]),
            "nearest_observed_regret": float(best_observed_metric - actual[nearest_idx]),
            "nearest_observed_tv": float(distances[nearest_idx]),
            **_family_shares(packet, weights),
        }
        if extra:
            row.update(extra)
        rows.append(row)

    add_row(
        "best_observed",
        packet.base.w[best_idx],
        float(predicted_observed[best_idx]),
        {
            "anchor_run_name": str(packet.base.frame.loc[best_idx, packet.base.name_col]),
            "actual_metric": best_observed_metric,
        },
    )
    add_row(
        "predicted_best_observed",
        packet.base.w[predicted_best_observed_idx],
        float(predicted_observed[predicted_best_observed_idx]),
        {
            "anchor_run_name": str(packet.base.frame.loc[predicted_best_observed_idx, packet.base.name_col]),
            "actual_metric": float(actual[predicted_best_observed_idx]),
        },
    )
    add_row(
        "raw_candidate",
        best_candidate_weights,
        float(candidate_values[best_candidate_idx]),
        {
            "candidate_source": candidates[best_candidate_idx][0],
            "mean_task_raw_target": float(np.mean(raw_task_values)),
        },
    )
    add_row("raw_refined", raw_refined_weights, raw_refined_metric)
    add_row(
        "top8actual_hull",
        hull_weights,
        float(-hull_value),
        {
            "hull_nonzero_coeff_count": int(np.sum(hull_coeffs > 1e-6)),
            "hull_top_coeffs": json.dumps([float(v) for v in hull_coeffs[np.argsort(-hull_coeffs)[:4]]]),
        },
    )
    return pd.DataFrame.from_records(rows), weight_map


def _write_optimum_weights(path: Path, packet, weight_map: dict[str, np.ndarray]) -> None:
    rows: list[dict[str, Any]] = []
    for opt_kind, weights in weight_map.items():
        for domain_idx, domain_name in enumerate(packet.base.domain_names):
            rows.append(
                {
                    "opt_kind": opt_kind,
                    "domain_name": domain_name,
                    "phase0_weight": float(weights[0, domain_idx]),
                    "phase0_epochs": float(weights[0, domain_idx] * packet.base.c0[domain_idx]),
                    "phase1_weight": float(weights[1, domain_idx]),
                    "phase1_epochs": float(weights[1, domain_idx] * packet.base.c1[domain_idx]),
                }
            )
    pd.DataFrame.from_records(rows).to_csv(path, index=False)


def _materialize_fit(fit: EnsembleFit, objective: AggregateObjective) -> tuple[dict[str, Any], pd.DataFrame]:
    variant_dir = OUTPUT_DIR / fit.variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    pred_rows = pd.DataFrame(
        {
            "run_name": fit.packet.base.frame[fit.packet.base.name_col].astype(str).to_numpy(),
            "actual_metric": fit.actual_metric,
            "predicted_metric": fit.oof_predicted_metric,
        }
    )
    pred_rows["residual"] = pred_rows["predicted_metric"] - pred_rows["actual_metric"]
    pred_rows.to_csv(variant_dir / "oof_predictions.csv", index=False)
    _plot_predictions(pred_rows, variant_dir / "predicted_vs_actual.html", f"{fit.variant}: {DISPLAY_NAME}")
    _plot_residuals(pred_rows, variant_dir / "residuals.html", f"{fit.variant}: {DISPLAY_NAME}")

    optimum_rows, weight_map = _candidate_optimum(fit, objective)
    optimum_rows.to_csv(variant_dir / "optimum_diagnostics.csv", index=False)
    _write_optimum_weights(variant_dir / "optimum_weights.csv", fit.packet, weight_map)
    raw_plot = _plot_optimum_phase_comparison(
        objective_dir=variant_dir,
        packet=fit.packet,
        opt_kind="raw_refined",
        target_columns=list(fit.task_columns),
    )
    top8_plot = _plot_optimum_phase_comparison(
        objective_dir=variant_dir,
        packet=fit.packet,
        opt_kind="top8actual_hull",
        target_columns=list(fit.task_columns),
    )

    metric_summary = _metric_oof_summary(pred_rows, higher_is_better=True)
    best_observed_idx = int(np.argmax(fit.actual_metric))
    predicted_best_idx = int(np.argmax(fit.oof_predicted_metric))
    raw_refined = optimum_rows.loc[optimum_rows["opt_kind"].eq("raw_refined")].iloc[0].to_dict()
    top8_hull = optimum_rows.loc[optimum_rows["opt_kind"].eq("top8actual_hull")].iloc[0].to_dict()
    summary = {
        "variant": fit.variant,
        "scale": SCALE,
        "run_set": RUN_SET,
        "n": len(fit.actual_metric),
        "n_target_columns": len(fit.task_columns),
        "best_observed_run_name": str(fit.packet.base.frame.loc[best_observed_idx, fit.packet.base.name_col]),
        "best_observed_metric": float(fit.actual_metric[best_observed_idx]),
        "predicted_observed_run_name": str(fit.packet.base.frame.loc[predicted_best_idx, fit.packet.base.name_col]),
        "predicted_observed_metric": float(fit.actual_metric[predicted_best_idx]),
        "predicted_observed_regret": float(fit.actual_metric[best_observed_idx] - fit.actual_metric[predicted_best_idx]),
        "raw_refined_predicted_metric": float(raw_refined["predicted_metric"]),
        "raw_refined_optimism_vs_best_observed_metric": float(raw_refined["optimism_vs_best_observed_metric"]),
        "raw_refined_nearest_observed_run_name": str(raw_refined["nearest_observed_run_name"]),
        "raw_refined_nearest_observed_metric": float(raw_refined["nearest_observed_metric"]),
        "raw_refined_nearest_observed_regret": float(raw_refined["nearest_observed_regret"]),
        "raw_refined_nearest_observed_tv": float(raw_refined["nearest_observed_tv"]),
        "top8actual_hull_predicted_metric": float(top8_hull["predicted_metric"]),
        "top8actual_hull_optimism_vs_best_observed_metric": float(top8_hull["optimism_vs_best_observed_metric"]),
        "top8actual_hull_nearest_observed_metric": float(top8_hull["nearest_observed_metric"]),
        "top8actual_hull_nearest_observed_regret": float(top8_hull["nearest_observed_regret"]),
        "top8actual_hull_nearest_observed_tv": float(top8_hull["nearest_observed_tv"]),
        "raw_refined_phase_comparison_png": str(raw_plot),
        "top8actual_hull_phase_comparison_png": str(top8_plot),
        **metric_summary,
        **{f"raw_refined_{key}": value for key, value in _family_shares(fit.packet, weight_map["raw_refined"]).items()},
    }
    return summary, optimum_rows


def _write_report(summary: pd.DataFrame, optimum: pd.DataFrame) -> None:
    columns = [
        "variant",
        "n",
        "n_target_columns",
        "metric_oof_rmse",
        "metric_oof_mae",
        "metric_oof_spearman",
        "metric_oof_pearson",
        "metric_oof_regret_at_1",
        "best_observed_metric",
        "predicted_observed_metric",
        "predicted_observed_regret",
        "raw_refined_predicted_metric",
        "raw_refined_optimism_vs_best_observed_metric",
        "raw_refined_nearest_observed_metric",
        "raw_refined_nearest_observed_regret",
        "raw_refined_nearest_observed_tv",
        "top8actual_hull_predicted_metric",
        "top8actual_hull_nearest_observed_regret",
        "top8actual_hull_nearest_observed_tv",
    ]
    body = [
        "# 300M choice_prob_norm task-ensemble GRP fits",
        "",
        "## Setup",
        "",
        f"- Scale: `{SCALE}`.",
        f"- Run set: `{RUN_SET}`.",
        "- Target columns: all `choice_prob_norm` metrics with matching `acc`, excluding `mmlu_pro_5shot`.",
        "- Variant 1: independent GRP nonlinear start-bank selection and linear head per task.",
        "- Variant 2: shared GRP nonlinear body selected on the aggregate target, per-task linear heads.",
        "- Raw refined optima are support-refined candidates seeded from per-task raw optima, "
        "not trusted deployment mixtures.",
        "",
        "## Fit Summary",
        "",
        summary[columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Optimum Diagnostics",
        "",
        optimum[
            [
                "variant",
                "opt_kind",
                "predicted_metric",
                "optimism_vs_best_observed_metric",
                "nearest_observed_metric",
                "nearest_observed_regret",
                "nearest_observed_tv",
                "raw_phase0_support_gt_1e4",
                "raw_phase1_support_gt_1e4",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    REPORT_MD.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    objective = _aggregate_objective()
    base, task_columns = _base_frame(args.max_tasks)
    model_options = _model_options(args.block_variant)

    print(f"Fitting independent task GRPs for {len(task_columns)} tasks", flush=True)
    independent = _fit_independent(
        base=base,
        task_columns=task_columns,
        objective=objective,
        family_scheme=args.family_scheme,
        model_options=model_options,
        random_starts=args.random_starts,
        prob_eps=args.prob_eps,
    )
    print("Fitting shared-body task-head ensemble", flush=True)
    shared = _fit_shared_body(
        base=base,
        task_columns=task_columns,
        objective=objective,
        family_scheme=args.family_scheme,
        model_options=model_options,
        random_starts=args.random_starts,
        coarse_top_k=args.coarse_top_k,
        method=args.method,
        prob_eps=args.prob_eps,
    )

    summary_rows: list[dict[str, Any]] = []
    optimum_rows: list[pd.DataFrame] = []
    task_fit_rows: list[dict[str, Any]] = []
    for fit in (independent, shared):
        print(f"Optimizing ensemble variant {fit.variant}", flush=True)
        summary, optimum = _materialize_fit(fit, objective)
        summary_rows.append(summary)
        optimum_rows.append(optimum)
        task_fit_rows.extend(fit.task_fit_rows)
        print(
            f"{fit.variant}: rmse={summary['metric_oof_rmse']:.6f} "
            f"spearman={summary['metric_oof_spearman']:.6f} "
            f"raw_tv={summary['raw_refined_nearest_observed_tv']:.3f}",
            flush=True,
        )

    summary_frame = pd.DataFrame.from_records(summary_rows)
    optimum_frame = pd.concat(optimum_rows, ignore_index=True)
    task_fit_frame = pd.DataFrame.from_records(task_fit_rows)
    summary_frame.to_csv(SUMMARY_CSV, index=False)
    optimum_frame.to_csv(OPTIMUM_DIAGNOSTICS_CSV, index=False)
    task_fit_frame.to_csv(TASK_FITS_CSV, index=False)
    _write_report(summary_frame, optimum_frame)
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {TASK_FITS_CSV}")
    print(f"Wrote {OPTIMUM_DIAGNOSTICS_CSV}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
