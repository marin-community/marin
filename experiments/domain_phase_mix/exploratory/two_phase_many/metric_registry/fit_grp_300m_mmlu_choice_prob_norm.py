# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly", "scipy", "scikit-learn", "tabulate"]
# ///
"""Fit GRP no-L2 surrogates to 300M MMLU choice_prob_norm metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_grp_power_family_penalty_no_l2_retune import (
    REG_FIXED,
    VARIANT_NAME,
    _no_l2_param_keys,
    _parameter_counts,
    _pack_no_l2_params,
    _unpack_no_l2_params,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.fit_grp_no_l2_benchmark_aggregates import (
    AggregateObjective,
    DEFAULT_FAMILY_SCHEME,
    FAMILY_SCHEMES,
    _expanded_start_bank,
    _family_shares,
    _model_target_to_metric,
    _packet_from_frame,
    _plot_predictions,
    _plot_residuals,
    _prediction_rows,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.materialize_fit_dataset import (
    materialize_fit_dataset,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    CALIBRATION_CV_WEIGHT,
    CALIBRATION_FOLDMEAN_WEIGHT,
    CALIBRATION_TAIL_WEIGHT,
    LOWER_TAIL_FRAC,
    build_penalty_calibration_surrogate,
    compute_penalty_calibration_metrics,
    optimize_penalty_calibration_model,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    optimize_generic_family_convex_hull,
)
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance

SCALE = "300m_6b"
RUN_SET = "qsplit240_core"
COHORT = "signal"
CV_SEED = 0
DEFAULT_METHOD = "Powell"
DEFAULT_COARSE_TOP_K = 8
DEFAULT_RANDOM_STARTS = 24
DEFAULT_PROB_EPS = 1e-4
BLOCK_VARIANTS = ("full", "no_pairs", "signals_only")
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "reference_outputs" / "grp_300m_mmlu_choice_prob_norm_20260428"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
PARAMS_CSV = OUTPUT_DIR / "params.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"
REPORT_MD = OUTPUT_DIR / "report.md"
OPTIMUM_DIAGNOSTICS_CSV = OUTPUT_DIR / "optimum_diagnostics.csv"
TRUSTBLEND_TV_CAPS = (0.05, 0.10, 0.15, 0.20)
TRUSTBLEND_GRID_SIZE = 401


@dataclass(frozen=True)
class ChoiceProbObjective:
    """One MMLU choice probability objective."""

    slug: str
    metric_key: str
    display_name: str
    transform: str


OBJECTIVES = (
    ChoiceProbObjective(
        slug="mmlu_5shot_choice_prob_norm_raw",
        metric_key="lm_eval/mmlu_5shot/choice_prob_norm",
        display_name="MMLU 5-shot choice_prob_norm raw-probability",
        transform="raw_probability",
    ),
    ChoiceProbObjective(
        slug="mmlu_5shot_choice_prob_norm_logit",
        metric_key="lm_eval/mmlu_5shot/choice_prob_norm",
        display_name="MMLU 5-shot choice_prob_norm logit-probability",
        transform="logit_probability",
    ),
    ChoiceProbObjective(
        slug="mmlu_5shot_choice_prob_norm_arcsin_sqrt",
        metric_key="lm_eval/mmlu_5shot/choice_prob_norm",
        display_name="MMLU 5-shot choice_prob_norm arcsin-sqrt probability",
        transform="arcsin_sqrt_probability",
    ),
    ChoiceProbObjective(
        slug="mmlu_5shot_choice_prob_norm_probit",
        metric_key="lm_eval/mmlu_5shot/choice_prob_norm",
        display_name="MMLU 5-shot choice_prob_norm probit probability",
        transform="probit_probability",
    ),
    ChoiceProbObjective(
        slug="mmlu_5shot_choice_prob_norm_rank_normal",
        metric_key="lm_eval/mmlu_5shot/choice_prob_norm",
        display_name="MMLU 5-shot choice_prob_norm rank-normal diagnostic",
        transform="rank_normal",
    ),
    ChoiceProbObjective(
        slug="mmlu_sl_verb_5shot_choice_prob_norm_raw",
        metric_key="lm_eval/mmlu_sl_verb_5shot/choice_prob_norm",
        display_name="MMLU SL-Verb 5-shot choice_prob_norm raw-probability",
        transform="raw_probability",
    ),
    ChoiceProbObjective(
        slug="mmlu_sl_verb_5shot_choice_prob_norm_logit",
        metric_key="lm_eval/mmlu_sl_verb_5shot/choice_prob_norm",
        display_name="MMLU SL-Verb 5-shot choice_prob_norm logit-probability",
        transform="logit_probability",
    ),
    ChoiceProbObjective(
        slug="mmlu_sl_verb_5shot_choice_prob_norm_arcsin_sqrt",
        metric_key="lm_eval/mmlu_sl_verb_5shot/choice_prob_norm",
        display_name="MMLU SL-Verb 5-shot choice_prob_norm arcsin-sqrt probability",
        transform="arcsin_sqrt_probability",
    ),
    ChoiceProbObjective(
        slug="mmlu_sl_verb_5shot_choice_prob_norm_probit",
        metric_key="lm_eval/mmlu_sl_verb_5shot/choice_prob_norm",
        display_name="MMLU SL-Verb 5-shot choice_prob_norm probit probability",
        transform="probit_probability",
    ),
    ChoiceProbObjective(
        slug="mmlu_sl_verb_5shot_choice_prob_norm_rank_normal",
        metric_key="lm_eval/mmlu_sl_verb_5shot/choice_prob_norm",
        display_name="MMLU SL-Verb 5-shot choice_prob_norm rank-normal diagnostic",
        transform="rank_normal",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--coarse-top-k", type=int, default=DEFAULT_COARSE_TOP_K)
    parser.add_argument("--random-starts", type=int, default=DEFAULT_RANDOM_STARTS)
    parser.add_argument("--prob-eps", type=float, default=DEFAULT_PROB_EPS)
    parser.add_argument("--only-slug-prefix", action="append", default=[])
    parser.add_argument("--family-scheme", action="append", choices=FAMILY_SCHEMES, default=[])
    parser.add_argument("--block-variant", action="append", choices=BLOCK_VARIANTS, default=[])
    return parser.parse_args()


def _aggregate_objective(spec: ChoiceProbObjective) -> AggregateObjective:
    return AggregateObjective(
        slug=spec.slug,
        source_column="objective_metric",
        display_name=spec.display_name,
        family="choice_prob_norm",
        higher_is_better=True,
        transform=spec.transform,
    )


def _model_options(block_variant: str) -> dict[str, bool]:
    if block_variant == "full":
        return {}
    if block_variant == "no_pairs":
        return {"include_pairs": False}
    if block_variant == "signals_only":
        return {"include_family_group_penalty": False}
    raise ValueError(f"Unsupported block variant: {block_variant}")


def _oof_target_metrics(packet, params: dict[str, float], model_options: dict[str, bool]) -> dict[str, float]:
    y = packet.base.y
    kf = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    oof = np.zeros_like(y)
    fold_regrets: list[float] = []
    for train_idx, test_idx in kf.split(packet.base.w):
        model = build_penalty_calibration_surrogate(
            packet,
            params=params,
            variant_name=VARIANT_NAME,
            **model_options,
        ).fit(packet.base.w[train_idx], y[train_idx])
        pred = model.predict(packet.base.w[test_idx])
        oof[test_idx] = pred
        fold_regrets.append(float(y[test_idx][int(np.argmin(pred))] - np.min(y[test_idx])))
    residuals = oof - y
    tail_count = max(5, int(np.ceil(float(LOWER_TAIL_FRAC) * float(len(y)))))
    tail_idx = np.argsort(oof)[:tail_count]
    lower_tail_optimism = float(np.mean(np.maximum(y[tail_idx] - oof[tail_idx], 0.0)))
    mean_regret = float(np.mean(fold_regrets))
    cv_rmse = float(np.sqrt(np.mean(residuals**2)))
    objective = (
        CALIBRATION_CV_WEIGHT * cv_rmse
        + CALIBRATION_FOLDMEAN_WEIGHT * mean_regret
        + CALIBRATION_TAIL_WEIGHT * lower_tail_optimism
    )
    return {
        "target_cv_rmse": cv_rmse,
        "target_cv_mae": float(np.mean(np.abs(residuals))),
        "target_cv_spearman": float(stats.spearmanr(y, oof).statistic),
        "target_cv_foldmean_regret_at_1": mean_regret,
        "target_lower_tail_optimism": lower_tail_optimism,
        "objective": float(objective),
    }


def _coarse_rows(packet, start_bank: tuple[dict[str, float], ...], model_options: dict[str, bool]) -> pd.DataFrame:
    rows = [
        {
            "stage": "coarse",
            "start_id": int(start_id),
            **params,
            **_oof_target_metrics(packet, params, model_options),
        }
        for start_id, params in enumerate(start_bank)
    ]
    return pd.DataFrame.from_records(rows).sort_values(
        ["objective", "target_cv_rmse", "target_cv_foldmean_regret_at_1"],
        ascending=[True, True, True],
    )


def _refine_rows(
    packet,
    start_bank: tuple[dict[str, float], ...],
    coarse_top_k: int,
    method: str,
    model_options: dict[str, bool],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    coarse = _coarse_rows(packet, start_bank, model_options)
    chosen_ids = coarse["start_id"].head(int(coarse_top_k)).tolist()
    best_row: dict[str, Any] | None = None
    refine_rows: list[dict[str, Any]] = []
    for chosen_rank, start_id in enumerate(chosen_ids):
        start = _pack_no_l2_params(start_bank[start_id])
        cache: dict[tuple[float, ...], float] = {}

        def objective(z: np.ndarray, _cache: dict[tuple[float, ...], float] = cache) -> float:
            key = tuple(np.round(np.asarray(z, dtype=float), 8))
            if key not in _cache:
                params = _unpack_no_l2_params(z)
                _cache[key] = _oof_target_metrics(packet, params, model_options)["objective"]
            return _cache[key]

        options = {
            "L-BFGS-B": {"maxiter": 160, "ftol": 1e-8},
            "Nelder-Mead": {"maxiter": 900, "xatol": 1e-4, "fatol": 1e-8},
            "Powell": {"maxiter": 120, "xtol": 1e-4, "ftol": 1e-8},
        }.get(method, {"maxiter": 240})
        result = minimize(objective, start, method=method, options=options)
        params = _unpack_no_l2_params(np.asarray(result.x, dtype=float))
        row = {
            "stage": "refine",
            "chosen_rank": int(chosen_rank),
            "start_id": int(start_id),
            "success": bool(result.success),
            "message": str(result.message),
            **params,
            **_oof_target_metrics(packet, params, model_options),
        }
        refine_rows.append(row)
        if best_row is None or float(row["objective"]) < float(best_row["objective"]):
            best_row = row
    if best_row is None:
        raise RuntimeError("No refined fit produced a best row")
    return coarse, best_row, pd.DataFrame.from_records(refine_rows).sort_values("objective")


def _metric_oof_summary(pred_rows: pd.DataFrame, *, higher_is_better: bool) -> dict[str, float]:
    actual = pred_rows["actual_metric"].to_numpy(dtype=float)
    predicted = pred_rows["predicted_metric"].to_numpy(dtype=float)
    residual = predicted - actual
    best_value = float(np.max(actual) if higher_is_better else np.min(actual))
    selected_value = float(actual[int(np.argmax(predicted) if higher_is_better else np.argmin(predicted))])
    return {
        "metric_oof_rmse": float(np.sqrt(np.mean(residual**2))),
        "metric_oof_mae": float(np.mean(np.abs(residual))),
        "metric_oof_spearman": float(stats.spearmanr(actual, predicted).statistic),
        "metric_oof_pearson": float(stats.pearsonr(actual, predicted).statistic),
        "metric_oof_regret_at_1": float(
            best_value - selected_value if higher_is_better else selected_value - best_value
        ),
        "actual_metric_min": float(np.min(actual)),
        "actual_metric_max": float(np.max(actual)),
        "actual_metric_std": float(np.std(actual, ddof=1)),
        "predicted_metric_std": float(np.std(predicted, ddof=1)),
    }


def _metric_from_target(value: float, objective: AggregateObjective) -> float:
    return float(_model_target_to_metric(float(value), objective))


def _nearest_observed_summary(packet, pred_rows: pd.DataFrame, weights: np.ndarray) -> dict[str, Any]:
    distances = average_phase_tv_distance(packet.base.w, weights[None, :, :])
    nearest_idx = int(np.argmin(distances))
    best_observed_metric = float(np.max(pred_rows["actual_metric"].to_numpy(dtype=float)))
    nearest_metric = float(pred_rows.loc[nearest_idx, "actual_metric"])
    return {
        "nearest_observed_run_name": str(pred_rows.loc[nearest_idx, "run_name"]),
        "nearest_observed_metric": nearest_metric,
        "nearest_observed_regret": float(best_observed_metric - nearest_metric),
        "nearest_observed_tv": float(distances[nearest_idx]),
    }


def _objective_diagnostics(
    packet,
    model,
    objective: AggregateObjective,
    pred_rows: pd.DataFrame,
    raw_value: float,
    raw_weights: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compare raw and locality-constrained optima for bounded choice-probability targets."""
    rows: list[dict[str, Any]] = []
    weight_map: dict[str, np.ndarray] = {"raw": raw_weights}
    actual_metric = pred_rows["actual_metric"].to_numpy(dtype=float)
    best_idx = int(np.argmax(actual_metric))
    best_observed_metric = float(actual_metric[best_idx])
    predicted_rows_target = model.predict(packet.base.w)
    predicted_observed_idx = int(np.argmin(predicted_rows_target))

    def add_row(
        *,
        opt_kind: str,
        predicted_value: float,
        weights: np.ndarray,
        extra: dict[str, Any] | None = None,
    ) -> None:
        metric = _metric_from_target(predicted_value, objective)
        row = {
            "slug": objective.slug,
            "opt_kind": opt_kind,
            "predicted_target": float(predicted_value),
            "predicted_metric": float(metric),
            "optimism_vs_best_observed_metric": float(metric - best_observed_metric),
            **_nearest_observed_summary(packet, pred_rows, weights),
            **_family_shares(packet, weights),
        }
        if extra:
            row.update(extra)
        rows.append(row)
        weight_map[opt_kind] = weights

    add_row(
        opt_kind="best_observed",
        predicted_value=float(predicted_rows_target[best_idx]),
        weights=packet.base.w[best_idx],
        extra={
            "anchor_run_name": str(pred_rows.loc[best_idx, "run_name"]),
            "actual_metric": best_observed_metric,
            "observed_rank": 1,
        },
    )
    add_row(
        opt_kind="predicted_best_observed",
        predicted_value=float(predicted_rows_target[predicted_observed_idx]),
        weights=packet.base.w[predicted_observed_idx],
        extra={
            "anchor_run_name": str(pred_rows.loc[predicted_observed_idx, "run_name"]),
            "actual_metric": float(actual_metric[predicted_observed_idx]),
            "observed_rank": int(np.where(np.argsort(-actual_metric) == predicted_observed_idx)[0][0] + 1),
        },
    )
    add_row(opt_kind="raw", predicted_value=raw_value, weights=raw_weights)

    top_indices = np.argsort(packet.base.y)[: min(8, len(packet.base.y))]
    hull_value, hull_coeffs, hull_weights = optimize_generic_family_convex_hull(
        model,
        packet.base.w[top_indices],
        start_indices=np.arange(min(8, len(top_indices)), dtype=int),
    )
    coeff_order = np.argsort(hull_coeffs)[::-1]
    top_coeff_indices = [int(idx) for idx in coeff_order[: min(4, len(coeff_order))] if float(hull_coeffs[idx]) > 1e-6]
    add_row(
        opt_kind="top8actual_hull",
        predicted_value=float(hull_value),
        weights=hull_weights,
        extra={
            "hull_nonzero_coeff_count": int(np.sum(hull_coeffs > 1e-6)),
            "hull_top_run_names": json.dumps(
                [str(pred_rows.loc[int(top_indices[idx]), "run_name"]) for idx in top_coeff_indices]
            ),
            "hull_top_coeffs": json.dumps([float(hull_coeffs[idx]) for idx in top_coeff_indices]),
        },
    )

    raw_hull_distance = float(average_phase_tv_distance(hull_weights[None, :, :], raw_weights[None, :, :])[0])
    best_by_cap: dict[float, tuple[float, float, np.ndarray]] = {}
    for delta in np.linspace(0.0, 1.0, TRUSTBLEND_GRID_SIZE):
        weights = (1.0 - delta) * hull_weights + delta * raw_weights
        distance = float(average_phase_tv_distance(hull_weights[None, :, :], weights[None, :, :])[0])
        value = float(model.predict(weights[None, :, :])[0])
        for cap in TRUSTBLEND_TV_CAPS:
            if distance <= cap + 1e-12:
                current = best_by_cap.get(cap)
                if current is None or value < current[0]:
                    best_by_cap[cap] = (value, float(delta), weights)

    for cap in TRUSTBLEND_TV_CAPS:
        if cap not in best_by_cap:
            continue
        predicted_value, delta, weights = best_by_cap[cap]
        add_row(
            opt_kind=f"trustblend_top8actual_to_raw_cap{cap:.2f}",
            predicted_value=predicted_value,
            weights=weights,
            extra={
                "tv_cap_from_hull": float(cap),
                "delta_to_raw": float(delta),
                "raw_hull_tv": raw_hull_distance,
            },
        )
    return pd.DataFrame.from_records(rows), weight_map


def _write_optimum_weight_tables(objective_dir: Path, packet, weight_map: dict[str, np.ndarray]) -> None:
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
    pd.DataFrame.from_records(rows).to_csv(objective_dir / "optimum_weights.csv", index=False)


def _fit_one(
    spec: ChoiceProbObjective,
    *,
    family_scheme: str,
    block_variant: str,
    method: str,
    coarse_top_k: int,
    random_starts: int,
    prob_eps: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit_frame = materialize_fit_dataset(
        spec.metric_key,
        scale=SCALE,
        cohort=COHORT,
        run_set=RUN_SET,
        output=OUTPUT_DIR / "fit_datasets" / f"{spec.slug}.csv",
    )
    objective = _aggregate_objective(spec)
    packet = _packet_from_frame(fit_frame, objective, prob_eps, family_scheme)
    start_bank = _expanded_start_bank(random_starts)
    model_options = _model_options(block_variant)
    coarse, best, refine = _refine_rows(packet, start_bank, coarse_top_k, method, model_options)
    params = {key: float(best[key]) for key in _no_l2_param_keys()}
    params["reg"] = REG_FIXED

    model = build_penalty_calibration_surrogate(
        packet,
        params=params,
        variant_name=VARIANT_NAME,
        **model_options,
    ).fit(packet.base.w, packet.base.y)
    full_metrics = compute_penalty_calibration_metrics(packet, model, seed=CV_SEED)
    raw_result, phase0, phase1 = optimize_penalty_calibration_model(packet, model, seed=CV_SEED)
    raw_weights = np.stack([phase0, phase1], axis=0)
    raw_distances = average_phase_tv_distance(packet.base.w, raw_weights[None, :, :])
    nearest_idx = int(np.argmin(raw_distances))

    train_pred_target = model.predict(packet.base.w)
    pred_rows = _prediction_rows(packet, params, objective)
    best_observed_idx = int(np.argmax(pred_rows["actual_metric"]))
    predicted_observed_idx = int(np.argmin(train_pred_target))
    raw_metric = float(_model_target_to_metric(float(raw_result.fun), objective))
    raw_nearest_metric = float(pred_rows.loc[nearest_idx, "actual_metric"])
    best_metric = float(pred_rows.loc[best_observed_idx, "actual_metric"])
    predicted_observed_metric = float(pred_rows.loc[predicted_observed_idx, "actual_metric"])

    objective_dir = (
        OUTPUT_DIR / spec.slug
        if family_scheme == DEFAULT_FAMILY_SCHEME and block_variant == "full"
        else OUTPUT_DIR / f"{spec.slug}__{family_scheme}__{block_variant}"
    )
    objective_dir.mkdir(parents=True, exist_ok=True)
    coarse.to_csv(objective_dir / "coarse.csv", index=False)
    refine.to_csv(objective_dir / "refine.csv", index=False)
    pred_rows.to_csv(objective_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(
        {
            "domain_name": packet.base.domain_names,
            "phase0_weight": phase0,
            "phase0_epochs": phase0 * packet.base.c0,
            "phase1_weight": phase1,
            "phase1_epochs": phase1 * packet.base.c1,
        }
    ).to_csv(objective_dir / "raw_optimum_weights.csv", index=False)
    optimum_rows, optimum_weight_map = _objective_diagnostics(
        packet,
        model,
        objective,
        pred_rows,
        float(raw_result.fun),
        raw_weights,
    )
    optimum_rows.to_csv(objective_dir / "optimum_diagnostics.csv", index=False)
    _write_optimum_weight_tables(objective_dir, packet, optimum_weight_map)
    _plot_predictions(pred_rows, objective_dir / "predicted_vs_actual.html", spec.display_name)
    _plot_residuals(pred_rows, objective_dir / "residuals.html", spec.display_name)
    deployment = {
        f"{row.opt_kind!s}_{key}": value
        for row in optimum_rows.itertuples(index=False)
        for key, value in {
            "predicted_metric": float(row.predicted_metric),
            "optimism_vs_best_observed_metric": float(row.optimism_vs_best_observed_metric),
            "nearest_observed_metric": float(row.nearest_observed_metric),
            "nearest_observed_regret": float(row.nearest_observed_regret),
            "nearest_observed_tv": float(row.nearest_observed_tv),
        }.items()
    }

    summary = {
        "slug": spec.slug,
        "metric_key": spec.metric_key,
        "display_name": spec.display_name,
        "family_scheme": family_scheme,
        "block_variant": block_variant,
        "scale": SCALE,
        "run_set": RUN_SET,
        "cohort": COHORT,
        "transform": spec.transform,
        "n": len(packet.base.y),
        "method": method,
        "coarse_top_k": int(coarse_top_k),
        "start_bank_size": len(start_bank),
        "best_observed_run_name": str(pred_rows.loc[best_observed_idx, "run_name"]),
        "best_observed_metric": best_metric,
        "predicted_observed_run_name": str(pred_rows.loc[predicted_observed_idx, "run_name"]),
        "predicted_observed_metric": predicted_observed_metric,
        "predicted_observed_regret": float(best_metric - predicted_observed_metric),
        "raw_predicted_optimum_metric": raw_metric,
        "raw_nearest_observed_run_name": str(pred_rows.loc[nearest_idx, "run_name"]),
        "raw_nearest_observed_metric": raw_nearest_metric,
        "raw_nearest_observed_regret": float(best_metric - raw_nearest_metric),
        "raw_nearest_observed_tv": float(raw_distances[nearest_idx]),
        **deployment,
        **_metric_oof_summary(pred_rows, higher_is_better=True),
        **{
            key: float(value)
            for key, value in full_metrics.items()
            if isinstance(value, int | float | np.integer | np.floating)
        },
        **{
            key: float(value) for key, value in best.items() if isinstance(value, int | float | np.integer | np.floating)
        },
        **_family_shares(packet, raw_weights),
    }
    params_row = {
        "slug": spec.slug,
        "family_scheme": family_scheme,
        "block_variant": block_variant,
        **params,
        **_parameter_counts(packet),
    }
    return summary, params_row, optimum_rows


def _write_report(summary: pd.DataFrame, optimum: pd.DataFrame) -> None:
    columns = [
        "family_scheme",
        "block_variant",
        "display_name",
        "transform",
        "n",
        "metric_oof_rmse",
        "metric_oof_spearman",
        "metric_oof_regret_at_1",
        "actual_metric_min",
        "actual_metric_max",
        "actual_metric_std",
        "predicted_metric_std",
        "best_observed_metric",
        "predicted_observed_metric",
        "predicted_observed_regret",
        "raw_predicted_optimum_metric",
        "raw_nearest_observed_metric",
        "raw_nearest_observed_regret",
        "raw_nearest_observed_tv",
        "raw_phase0_broad_text_share",
        "raw_phase0_tech_code_share",
        "raw_phase0_reasoning_share",
        "raw_phase1_broad_text_share",
        "raw_phase1_tech_code_share",
        "raw_phase1_reasoning_share",
    ]
    best_by_metric = summary.sort_values(["metric_oof_spearman", "metric_oof_rmse"], ascending=[False, True])
    optimum_columns = [
        "slug",
        "family_scheme",
        "block_variant",
        "opt_kind",
        "predicted_metric",
        "optimism_vs_best_observed_metric",
        "nearest_observed_metric",
        "nearest_observed_regret",
        "nearest_observed_tv",
        "raw_phase0_broad_text_share",
        "raw_phase0_tech_code_share",
        "raw_phase0_reasoning_share",
        "raw_phase1_broad_text_share",
        "raw_phase1_tech_code_share",
        "raw_phase1_reasoning_share",
    ]
    body = [
        "# 300M MMLU choice_prob_norm GRP Fits",
        "",
        "## Setup",
        "",
        f"- Scale: `{SCALE}`.",
        f"- Run set: `{RUN_SET}`.",
        "- Model: GRP power-family-penalty no-L2 body.",
        "- Metrics: standard MMLU and MMLU SL-Verb `choice_prob_norm`.",
        "- Transforms: raw probability and logit probability. Higher is better; the fitted model target is "
        "negated so the existing minimization path can be reused.",
        "",
        "## Fit Summary",
        "",
        summary[columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Ranking",
        "",
        best_by_metric[
            [
                "display_name",
                "metric_oof_spearman",
                "metric_oof_rmse",
                "metric_oof_regret_at_1",
                "predicted_observed_run_name",
                "predicted_observed_regret",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Optimum Diagnostics",
        "",
        optimum[optimum_columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "- `choice_prob_norm` is fit much more like an accuracy surrogate than MMLU BPB: the target is "
        "bounded and higher-is-better.",
        "- BPB has a wider, lower-is-better loss range; `choice_prob_norm` has a narrow bounded "
        "probability range, so raw simplex optimization is more exposed to small extrapolation errors.",
        "- Logit fits are the best current default for standard MMLU `choice_prob_norm`: they slightly "
        "improve rank and make bounded probabilities additive in odds space.",
        "- Standard MMLU has materially more choice-probability spread than SL-Verb in this panel, so "
        "the SL-Verb absolute RMSE is less informative than rank/regret.",
        "- The raw standard-MMLU optima substantially exceed the best observed metric and sit far from "
        "observed rows; this is tail optimism, not a deployable mixture.",
        "- Top-actual hull and trust-blend rows are the safer deployment diagnostics for `choice_prob_norm` "
        "until we have validated raw benchmark-optimal mixtures.",
        "",
    ]
    REPORT_MD.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    family_schemes = tuple(args.family_scheme) if args.family_scheme else (DEFAULT_FAMILY_SCHEME,)
    block_variants = tuple(args.block_variant) if args.block_variant else ("full",)
    objectives = OBJECTIVES
    if args.only_slug_prefix:
        prefixes = tuple(str(prefix) for prefix in args.only_slug_prefix)
        objectives = tuple(spec for spec in objectives if spec.slug.startswith(prefixes))
        if not objectives:
            raise ValueError(f"No objectives matched prefixes {prefixes!r}")
    rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    optimum_rows: list[pd.DataFrame] = []
    for family_scheme in family_schemes:
        for block_variant in block_variants:
            for spec in objectives:
                print(f"fitting {spec.slug} family_scheme={family_scheme} block_variant={block_variant}", flush=True)
                summary, params, optimum = _fit_one(
                    spec,
                    family_scheme=family_scheme,
                    block_variant=block_variant,
                    method=args.method,
                    coarse_top_k=args.coarse_top_k,
                    random_starts=args.random_starts,
                    prob_eps=args.prob_eps,
                )
                rows.append(summary)
                param_rows.append(params)
                optimum.insert(0, "block_variant", block_variant)
                optimum.insert(0, "family_scheme", family_scheme)
                optimum_rows.append(optimum)
                print(
                    f"fit {spec.slug} family_scheme={family_scheme} block_variant={block_variant}: "
                    f"metric_oof_rmse={summary['metric_oof_rmse']:.6f} "
                    f"metric_oof_spearman={summary['metric_oof_spearman']:.6f}",
                    flush=True,
                )
    summary_frame = pd.DataFrame.from_records(rows)
    params_frame = pd.DataFrame.from_records(param_rows)
    optimum_frame = pd.concat(optimum_rows, ignore_index=True) if optimum_rows else pd.DataFrame()
    summary_frame.to_csv(SUMMARY_CSV, index=False)
    params_frame.to_csv(PARAMS_CSV, index=False)
    optimum_frame.to_csv(OPTIMUM_DIAGNOSTICS_CSV, index=False)
    SUMMARY_JSON.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(summary_frame, optimum_frame)
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {PARAMS_CSV}")
    print(f"Wrote {OPTIMUM_DIAGNOSTICS_CSV}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
