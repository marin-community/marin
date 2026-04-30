# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "plotly", "scipy", "scikit-learn", "tabulate", "torch"]
# ///
"""Fit perplexity-proxy models for 300M benchmark optimization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import warnings

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

import matplotlib.pyplot as plt

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
    EXCLUDED_SUITES,
    _choice_prob_columns,
    _suite_name,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.fit_grp_300m_mmlu_choice_prob_norm import (
    DEFAULT_COARSE_TOP_K,
    DEFAULT_METHOD,
    DEFAULT_PROB_EPS,
    DEFAULT_RANDOM_STARTS,
    SCALE,
    _coarse_rows,
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
    _prediction_rows,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    optimize_generic_family_convex_hull,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    build_penalty_calibration_surrogate,
    optimize_penalty_calibration_model,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.plot_grp_phase_comparison import (
    TEXT_MUTED_COLOR,
    _plot_cc_block,
    _plot_non_cc_block,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.plot_grp_vs_proportional import (
    _display_non_cc_label,
    _grp_domain_order,
)
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance

COHORT = "signal"
CV_SEED = 0
PROB_EPS = 1e-5
TARGET_CHOICE = "mean_choice_prob_norm_no_mmlu_pro"
TARGET_ACCURACY = "mean_accuracy_no_mmlu_pro"
OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "reference_outputs" / "grp_300m_perplexity_proxy_benchmark_20260428"
)
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
PROXY_FITS_CSV = OUTPUT_DIR / "proxy_fits.csv"
SELECTED_FEATURES_CSV = OUTPUT_DIR / "selected_features.csv"
OPTIMUM_DIAGNOSTICS_CSV = OUTPUT_DIR / "optimum_diagnostics.csv"
REPORT_MD = OUTPUT_DIR / "report.md"
LOGBOOK_MD = Path(".agents/logbooks/benchmark-proxy-optimization.md")
SUPPORT_TOP_K = 12
TRUSTBLEND_TV_CAPS = (0.05, 0.10, 0.15, 0.20)
TRUSTBLEND_GRID_SIZE = 401


@dataclass(frozen=True)
class ProxyCandidate:
    """A selected proxy regression candidate."""

    candidate_id: str
    target: str
    feature_set: str
    model_type: str
    transform: str
    feature_columns: tuple[str, ...]
    pipeline: Pipeline
    oof_metric: np.ndarray
    oof_transformed: np.ndarray
    selected_features: tuple[str, ...]
    selected_pipeline: Pipeline
    selected_oof_metric: np.ndarray


class ScoreModelAdapter:
    """Expose a higher-is-better score as a minimization-compatible model."""

    def __init__(self, predict_metric_fn):
        self._predict_metric_fn = predict_metric_fn
        self.coef_ = np.ones(1, dtype=float)
        self.intercept_ = 0.0

    def predict_metric(self, weights: np.ndarray) -> np.ndarray:
        return np.asarray(self._predict_metric_fn(weights), dtype=float)

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return -self.predict_metric(weights)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-starts", type=int, default=DEFAULT_RANDOM_STARTS)
    parser.add_argument("--coarse-top-k", type=int, default=DEFAULT_COARSE_TOP_K)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--prob-eps", type=float, default=DEFAULT_PROB_EPS)
    parser.add_argument("--family-scheme", default="default")
    parser.add_argument("--block-variant", default="full")
    parser.add_argument("--max-selected-features", type=int, default=12)
    return parser.parse_args()


def _is_perplexity_feature(column: str) -> bool:
    lower = column.lower()
    return (
        column.startswith(("eval/", "lm_eval/"))
        and any(token in lower for token in ("/bpb", "/loss", "perplexity"))
        and "logprob" not in lower
    )


def _target_task_columns(frame: pd.DataFrame) -> tuple[tuple[str, ...], tuple[str, ...]]:
    choice_columns = tuple(_choice_prob_columns(frame))
    accuracy_columns: list[str] = []
    for column in choice_columns:
        suite = _suite_name(column)
        if suite in EXCLUDED_SUITES:
            continue
        acc_column = column.rsplit("/", 1)[0] + "/acc"
        if acc_column not in frame.columns:
            raise ValueError(f"Missing matched accuracy column for {column}")
        accuracy_columns.append(acc_column)
    return choice_columns, tuple(accuracy_columns)


def _build_dataset() -> tuple[pd.DataFrame, dict[str, tuple[str, ...]], tuple[str, ...], tuple[str, ...]]:
    frame = pd.read_csv(METRICS_WIDE_CSV, low_memory=False)
    choice_columns, accuracy_columns = _target_task_columns(frame)
    mask = frame["scale"].eq(SCALE) & frame["cohort"].eq(COHORT) & frame["is_qsplit240_core"].fillna(False)
    data = frame.loc[mask].copy()
    data = data.dropna(subset=[*choice_columns, *accuracy_columns]).copy()
    data[TARGET_CHOICE] = data[list(choice_columns)].mean(axis=1)
    data[TARGET_ACCURACY] = data[list(accuracy_columns)].mean(axis=1)

    weight_columns = tuple(sorted(column for column in data.columns if column.startswith(WEIGHT_PREFIXES)))
    feature_columns = tuple(column for column in data.columns if _is_perplexity_feature(column))
    complete_features = tuple(column for column in feature_columns if data[column].notna().all())
    deployable_features = tuple(column for column in complete_features if column.startswith("eval/"))
    diagnostic_features = tuple(complete_features)

    id_columns = tuple(
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
    )
    dataset = data[
        list(id_columns)
        + list(weight_columns)
        + [TARGET_CHOICE, TARGET_ACCURACY]
        + list(choice_columns)
        + list(accuracy_columns)
        + list(complete_features)
    ].copy()
    dataset = dataset.dropna(axis=1, how="all").reset_index(drop=True)
    feature_sets = {"eval_only": deployable_features, "eval_plus_lm_eval": diagnostic_features}

    if len(dataset) != 240:
        raise ValueError(f"Expected 240 qsplit-core rows, got {len(dataset)}")
    if len(choice_columns) != 119 or len(accuracy_columns) != 119:
        raise ValueError(f"Expected 119 matched tasks, got {len(choice_columns)} choice and {len(accuracy_columns)} acc")
    for prefix in ("phase_0_", "phase_1_"):
        columns = [column for column in dataset.columns if column.startswith(prefix)]
        sums = dataset[columns].sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-6):
            raise ValueError(f"{prefix} weights do not sum to 1")
    if not deployable_features:
        raise ValueError("No complete eval/* perplexity features found")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_DIR / "proxy_dataset.csv", index=False)
    (OUTPUT_DIR / "dataset_summary.json").write_text(
        json.dumps(
            {
                "n_rows": len(dataset),
                "n_choice_tasks": len(choice_columns),
                "n_accuracy_tasks": len(accuracy_columns),
                "n_eval_only_features": len(deployable_features),
                "n_eval_plus_lm_eval_features": len(diagnostic_features),
                "excluded_suites": sorted(EXCLUDED_SUITES),
                "choice_columns": list(choice_columns),
                "accuracy_columns": list(accuracy_columns),
                "feature_sets": {key: list(value) for key, value in feature_sets.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset, feature_sets, choice_columns, accuracy_columns


def _transform_target(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return values.astype(float)
    if transform == "logit":
        return logit(np.clip(values, PROB_EPS, 1.0 - PROB_EPS))
    raise ValueError(f"Unsupported target transform {transform}")


def _inverse_target(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return values.astype(float)
    if transform == "logit":
        return expit(values)
    raise ValueError(f"Unsupported target transform {transform}")


def _make_pipeline(model_type: str) -> Pipeline:
    if model_type == "ridge":
        model = RidgeCV(alphas=np.logspace(-5, 5, 41))
    elif model_type == "elasticnet":
        model = ElasticNetCV(
            l1_ratio=(0.1, 0.5, 0.9),
            alphas=np.logspace(-4, 1, 31),
            cv=5,
            max_iter=5000,
            random_state=CV_SEED,
            tol=1e-4,
        )
    else:
        raise ValueError(f"Unsupported model type {model_type}")
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def _metric_summary(actual: np.ndarray, predicted: np.ndarray, prefix: str = "") -> dict[str, float]:
    residual = predicted - actual
    best_idx = int(np.argmax(actual))
    selected_idx = int(np.argmax(predicted))
    tail_count = max(5, int(np.ceil(0.15 * len(actual))))
    tail_idx = np.argsort(-predicted)[:tail_count]
    optimism = float(np.mean(np.maximum(predicted[tail_idx] - actual[tail_idx], 0.0)))
    return {
        f"{prefix}rmse": float(np.sqrt(np.mean(residual**2))),
        f"{prefix}mae": float(np.mean(np.abs(residual))),
        f"{prefix}spearman": float(stats.spearmanr(actual, predicted).statistic),
        f"{prefix}pearson": float(stats.pearsonr(actual, predicted).statistic),
        f"{prefix}regret_at_1": float(actual[best_idx] - actual[selected_idx]),
        f"{prefix}lower_tail_optimism": optimism,
        f"{prefix}actual_min": float(np.min(actual)),
        f"{prefix}actual_max": float(np.max(actual)),
        f"{prefix}actual_std": float(np.std(actual, ddof=1)),
        f"{prefix}predicted_std": float(np.std(predicted, ddof=1)),
    }


def _fit_proxy_oof(
    data: pd.DataFrame,
    *,
    target: str,
    feature_columns: tuple[str, ...],
    model_type: str,
    transform: str,
) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    x = data[list(feature_columns)].to_numpy(dtype=float)
    y_metric = data[target].to_numpy(dtype=float)
    y = _transform_target(y_metric, transform)
    oof_transformed = np.zeros_like(y)
    kf = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    for train_idx, test_idx in kf.split(x):
        pipeline = _make_pipeline(model_type)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            pipeline.fit(x[train_idx], y[train_idx])
        oof_transformed[test_idx] = pipeline.predict(x[test_idx])
    final = _make_pipeline(model_type)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        final.fit(x, y)
    return final, _inverse_target(oof_transformed, transform), oof_transformed


def _coef_table(pipeline: Pipeline, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    coefficients = np.asarray(model.coef_, dtype=float)
    if coefficients.shape != (len(feature_columns),):
        raise ValueError(f"Unexpected coefficient shape {coefficients.shape}")
    rows = [
        {"feature": feature, "coefficient": float(coef), "abs_coefficient": float(abs(coef))}
        for feature, coef in zip(feature_columns, coefficients, strict=True)
    ]
    return pd.DataFrame.from_records(rows).sort_values("abs_coefficient", ascending=False)


def _select_features(pipeline: Pipeline, feature_columns: tuple[str, ...], max_features: int) -> tuple[str, ...]:
    table = _coef_table(pipeline, feature_columns)
    nonzero = table.loc[table["abs_coefficient"] > 1e-10]
    selected = nonzero if len(nonzero) > 0 else table
    return tuple(selected["feature"].head(max_features).astype(str).tolist())


def _fit_proxy_candidates(
    data: pd.DataFrame,
    feature_sets: dict[str, tuple[str, ...]],
    *,
    max_selected_features: int,
    targets: tuple[str, ...] = (TARGET_CHOICE, TARGET_ACCURACY),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, ProxyCandidate]]:
    proxy_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    candidates: dict[str, ProxyCandidate] = {}
    for target in targets:
        for feature_set, feature_columns in feature_sets.items():
            for model_type in ("ridge", "elasticnet"):
                for transform in ("identity", "logit"):
                    candidate_id = f"{target}__{feature_set}__{model_type}__{transform}"
                    pipeline, oof_metric, oof_transformed = _fit_proxy_oof(
                        data,
                        target=target,
                        feature_columns=feature_columns,
                        model_type=model_type,
                        transform=transform,
                    )
                    selected_features = _select_features(pipeline, feature_columns, max_selected_features)
                    selected_pipeline, selected_oof_metric, _selected_oof_transformed = _fit_proxy_oof(
                        data,
                        target=target,
                        feature_columns=selected_features,
                        model_type=model_type,
                        transform=transform,
                    )
                    actual = data[target].to_numpy(dtype=float)
                    summary = {
                        "candidate_id": candidate_id,
                        "target": target,
                        "feature_set": feature_set,
                        "model_type": model_type,
                        "transform": transform,
                        "n_features": len(feature_columns),
                        "n_selected_features": len(selected_features),
                        **_metric_summary(actual, oof_metric, prefix="full_proxy_"),
                        **_metric_summary(actual, selected_oof_metric, prefix="selected_proxy_"),
                    }
                    proxy_rows.append(summary)
                    coef_table = _coef_table(pipeline, feature_columns)
                    for rank, row in enumerate(coef_table.head(max_selected_features).itertuples(index=False), start=1):
                        feature_rows.append(
                            {
                                "candidate_id": candidate_id,
                                "rank": rank,
                                "feature": str(row.feature),
                                "coefficient": float(row.coefficient),
                                "abs_coefficient": float(row.abs_coefficient),
                            }
                        )
                    candidates[candidate_id] = ProxyCandidate(
                        candidate_id=candidate_id,
                        target=target,
                        feature_set=feature_set,
                        model_type=model_type,
                        transform=transform,
                        feature_columns=feature_columns,
                        pipeline=pipeline,
                        oof_metric=oof_metric,
                        oof_transformed=oof_transformed,
                        selected_features=selected_features,
                        selected_pipeline=selected_pipeline,
                        selected_oof_metric=selected_oof_metric,
                    )
    proxy_frame = pd.DataFrame.from_records(proxy_rows)
    feature_frame = pd.DataFrame.from_records(feature_rows)
    proxy_frame.to_csv(PROXY_FITS_CSV, index=False)
    feature_frame.to_csv(SELECTED_FEATURES_CSV, index=False)
    return proxy_frame, feature_frame, candidates


def _best_single_feature_baselines(
    data: pd.DataFrame,
    feature_sets: dict[str, tuple[str, ...]],
    *,
    targets: tuple[str, ...] = (TARGET_CHOICE, TARGET_ACCURACY),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in targets:
        actual = data[target].to_numpy(dtype=float)
        for feature_set, features in feature_sets.items():
            for feature in features:
                values = data[feature].to_numpy(dtype=float)
                predicted = -values
                rows.append(
                    {
                        "target": target,
                        "feature_set": feature_set,
                        "feature": feature,
                        **_metric_summary(actual, predicted, prefix="single_feature_"),
                    }
                )
    out = pd.DataFrame.from_records(rows).sort_values(
        ["target", "feature_set", "single_feature_spearman"],
        ascending=[True, True, False],
    )
    out.to_csv(OUTPUT_DIR / "single_feature_baselines.csv", index=False)
    return out


def _objective_for_probability(slug: str, source_column: str, display_name: str) -> AggregateObjective:
    return AggregateObjective(
        slug=slug,
        source_column=source_column,
        display_name=display_name,
        family="probability",
        higher_is_better=True,
        transform="logit_probability",
    )


def _objective_for_loss(slug: str, source_column: str, display_name: str) -> AggregateObjective:
    return AggregateObjective(
        slug=slug,
        source_column=source_column,
        display_name=display_name,
        family="perplexity",
        higher_is_better=False,
        transform="identity",
    )


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


def _fit_scalar_grp(
    data: pd.DataFrame,
    candidate: ProxyCandidate,
    *,
    family_scheme: str,
    model_options: dict[str, bool],
    start_bank: tuple[dict[str, float], ...],
    coarse_top_k: int,
    method: str,
    prob_eps: float,
) -> tuple[Any, dict[str, float], pd.DataFrame]:
    target_column = f"{candidate.candidate_id}__scalar_proxy_target"
    scalar_data = data.copy()
    scalar_data[target_column] = np.clip(candidate.oof_metric, PROB_EPS, 1.0 - PROB_EPS)
    objective = _objective_for_probability(candidate.candidate_id, "objective_metric", candidate.candidate_id)
    packet = _packet_from_frame(_packet_frame(scalar_data, target_column), objective, prob_eps, family_scheme)
    _coarse, best, _refine = _refine_rows(packet, start_bank, coarse_top_k, method, model_options)
    params = {key: float(best[key]) for key in _no_l2_param_keys()}
    params["reg"] = REG_FIXED
    model = build_penalty_calibration_surrogate(
        packet,
        params=params,
        variant_name=VARIANT_NAME,
        **model_options,
    ).fit(packet.base.w, packet.base.y)
    pred_rows = _prediction_rows(packet, params, objective)
    return packet, params, pred_rows, model, objective


def _fit_component_grps(
    data: pd.DataFrame,
    candidate: ProxyCandidate,
    *,
    family_scheme: str,
    model_options: dict[str, bool],
    start_bank: tuple[dict[str, float], ...],
    prob_eps: float,
) -> tuple[Any, tuple[Any, ...], tuple[str, ...], tuple[dict[str, float], ...], tuple[AggregateObjective, ...]]:
    models: list[Any] = []
    params_rows: list[dict[str, float]] = []
    objectives: list[AggregateObjective] = []
    packet0 = None
    for feature in candidate.selected_features:
        objective = _objective_for_loss(feature, "objective_metric", feature)
        packet = _packet_from_frame(_packet_frame(data, feature), objective, prob_eps, family_scheme)
        coarse = _coarse_rows(packet, start_bank, model_options)
        best = dict(coarse.iloc[0])
        params = {key: float(best[key]) for key in _no_l2_param_keys()}
        params["reg"] = REG_FIXED
        model = build_penalty_calibration_surrogate(
            packet,
            params=params,
            variant_name=VARIANT_NAME,
            **model_options,
        ).fit(packet.base.w, packet.base.y)
        models.append(model)
        params_rows.append(params)
        objectives.append(objective)
        if packet0 is None:
            packet0 = packet
    if packet0 is None:
        raise ValueError("No selected component features")
    return packet0, tuple(models), candidate.selected_features, tuple(params_rows), tuple(objectives)


def _component_oof_metric(
    data: pd.DataFrame,
    candidate: ProxyCandidate,
    *,
    family_scheme: str,
    model_options: dict[str, bool],
    prob_eps: float,
    selected_features: tuple[str, ...],
    params_rows: tuple[dict[str, float], ...],
    objectives: tuple[AggregateObjective, ...],
) -> np.ndarray:
    component_predictions: list[np.ndarray] = []
    for feature, params, objective in zip(selected_features, params_rows, objectives, strict=True):
        packet = _packet_from_frame(_packet_frame(data, feature), objective, prob_eps, family_scheme)
        oof_target = np.zeros_like(packet.base.y)
        kf = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)
        for train_idx, test_idx in kf.split(packet.base.w):
            model = build_penalty_calibration_surrogate(
                packet,
                params=params,
                variant_name=VARIANT_NAME,
                **model_options,
            ).fit(packet.base.w[train_idx], packet.base.y[train_idx])
            oof_target[test_idx] = model.predict(packet.base.w[test_idx])
        component_predictions.append(_model_target_to_metric(oof_target, objective))
    feature_matrix = np.column_stack(component_predictions)
    return _predict_selected_proxy(candidate, feature_matrix)


def _predict_selected_proxy(candidate: ProxyCandidate, features: np.ndarray) -> np.ndarray:
    transformed = candidate.selected_pipeline.predict(features)
    return np.clip(_inverse_target(transformed, candidate.transform), PROB_EPS, 1.0 - PROB_EPS)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    out = np.exp(shifted)
    return out / np.sum(out)


def _optimize_on_support(
    score_model: ScoreModelAdapter,
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
    starts = [np.concatenate([start0, start1]), np.zeros(len(support0) + len(support1), dtype=float)]

    def objective(z: np.ndarray) -> float:
        return float(score_model.predict(weights_from_z(np.asarray(z, dtype=float))[None, :, :])[0])

    best = None
    for start in starts:
        result = minimize(objective, start, method="L-BFGS-B", options={"maxiter": 250, "ftol": 1e-10})
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError("Support optimization failed")
    weights = weights_from_z(np.asarray(best.x, dtype=float))
    return float(score_model.predict_metric(weights[None, :, :])[0]), weights


def _raw_weights_for_score_model(
    packet,
    score_model: ScoreModelAdapter,
    seed_weights: list[np.ndarray],
) -> tuple[float, np.ndarray]:
    candidate_values = np.asarray(
        [score_model.predict_metric(weights[None, :, :])[0] for weights in seed_weights],
        dtype=float,
    )
    best_idx = int(np.argmax(candidate_values))
    best_weights = seed_weights[best_idx]
    support0 = set(np.argsort(-best_weights[0])[:SUPPORT_TOP_K].tolist())
    support1 = set(np.argsort(-best_weights[1])[:SUPPORT_TOP_K].tolist())
    for weights in seed_weights[: min(16, len(seed_weights))]:
        support0.update(np.where(weights[0] > 1e-4)[0].tolist())
        support1.update(np.where(weights[1] > 1e-4)[0].tolist())
    return _optimize_on_support(
        score_model,
        packet.base.m,
        np.asarray(sorted(support0), dtype=int),
        np.asarray(sorted(support1), dtype=int),
        best_weights,
    )


def _trustblend_rows(
    score_model: ScoreModelAdapter,
    hull_weights: np.ndarray,
    raw_weights: np.ndarray,
) -> list[tuple[str, float, np.ndarray, dict[str, float]]]:
    out: list[tuple[str, float, np.ndarray, dict[str, float]]] = []
    raw_hull_distance = float(average_phase_tv_distance(hull_weights[None, :, :], raw_weights[None, :, :])[0])
    best_by_cap: dict[float, tuple[float, float, np.ndarray]] = {}
    for delta in np.linspace(0.0, 1.0, TRUSTBLEND_GRID_SIZE):
        weights = (1.0 - delta) * hull_weights + delta * raw_weights
        distance = float(average_phase_tv_distance(hull_weights[None, :, :], weights[None, :, :])[0])
        value = float(score_model.predict_metric(weights[None, :, :])[0])
        for cap in TRUSTBLEND_TV_CAPS:
            if distance <= cap + 1e-12:
                current = best_by_cap.get(cap)
                if current is None or value > current[0]:
                    best_by_cap[cap] = (value, float(delta), weights)
    for cap, (value, delta, weights) in best_by_cap.items():
        out.append(
            (
                f"trustblend_top8actual_to_raw_cap{cap:.2f}",
                value,
                weights,
                {"tv_cap_from_hull": float(cap), "delta_to_raw": delta, "raw_hull_tv": raw_hull_distance},
            )
        )
    return out


def _candidate_diagnostics(
    *,
    candidate_id: str,
    candidate_kind: str,
    target: str,
    packet,
    score_model: ScoreModelAdapter,
    actual_choice: np.ndarray,
    actual_accuracy: np.ndarray,
    actual_primary: np.ndarray,
    raw_metric: float,
    raw_weights: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    predicted_observed = score_model.predict_metric(packet.base.w)
    best_idx = int(np.argmax(actual_primary))
    predicted_best_idx = int(np.argmax(predicted_observed))
    top_indices = np.argsort(-actual_primary)[: min(8, len(actual_primary))]
    hull_value, hull_coeffs, hull_weights = optimize_generic_family_convex_hull(
        score_model,
        packet.base.w[top_indices],
        start_indices=np.arange(min(8, len(top_indices)), dtype=int),
    )
    weight_map = {
        "best_observed": packet.base.w[best_idx],
        "predicted_best_observed": packet.base.w[predicted_best_idx],
        "raw": raw_weights,
        "top8actual_hull": hull_weights,
    }
    rows: list[dict[str, Any]] = []

    def add_row(
        opt_kind: str,
        predicted_metric: float,
        weights: np.ndarray,
        extra: dict[str, Any] | None = None,
    ) -> None:
        distances = average_phase_tv_distance(packet.base.w, weights[None, :, :])
        nearest_idx = int(np.argmin(distances))
        row = {
            "candidate_id": candidate_id,
            "candidate_kind": candidate_kind,
            "target": target,
            "opt_kind": opt_kind,
            "predicted_proxy_metric": float(predicted_metric),
            "predicted_proxy_gain_vs_best_observed": float(predicted_metric - predicted_observed[best_idx]),
            "nearest_observed_run_name": str(packet.base.frame.loc[nearest_idx, packet.base.name_col]),
            "nearest_observed_tv": float(distances[nearest_idx]),
            "nearest_observed_choice": float(actual_choice[nearest_idx]),
            "nearest_observed_accuracy": float(actual_accuracy[nearest_idx]),
            "nearest_observed_primary_regret": float(actual_primary[best_idx] - actual_primary[nearest_idx]),
            "best_observed_choice": float(actual_choice[best_idx]),
            "best_observed_accuracy": float(actual_accuracy[best_idx]),
            **_family_shares(packet, weights),
        }
        if extra:
            row.update(extra)
        rows.append(row)

    add_row(
        "best_observed",
        float(predicted_observed[best_idx]),
        packet.base.w[best_idx],
        {"anchor_run_name": str(packet.base.frame.loc[best_idx, packet.base.name_col])},
    )
    add_row(
        "predicted_best_observed",
        float(predicted_observed[predicted_best_idx]),
        packet.base.w[predicted_best_idx],
        {"anchor_run_name": str(packet.base.frame.loc[predicted_best_idx, packet.base.name_col])},
    )
    add_row("raw", raw_metric, raw_weights)
    add_row(
        "top8actual_hull",
        float(-hull_value),
        hull_weights,
        {
            "hull_nonzero_coeff_count": int(np.sum(hull_coeffs > 1e-6)),
            "hull_top_coeffs": json.dumps([float(v) for v in hull_coeffs[np.argsort(-hull_coeffs)[:4]]]),
        },
    )
    for opt_kind, value, weights, extra in _trustblend_rows(score_model, hull_weights, raw_weights):
        weight_map[opt_kind] = weights
        add_row(opt_kind, value, weights, extra)
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


def _plot_phase_comparison(objective_dir: Path, packet, opt_kind: str, title: str) -> Path:
    weights_long = pd.read_csv(objective_dir / "optimum_weights.csv")
    selected = weights_long.loc[weights_long["opt_kind"].eq(opt_kind)].copy()
    weights_by_domain = selected.set_index("domain_name")
    weights = np.array(
        [
            weights_by_domain.loc[packet.base.domain_names, "phase0_weight"].to_numpy(dtype=float),
            weights_by_domain.loc[packet.base.domain_names, "phase1_weight"].to_numpy(dtype=float),
        ]
    )
    non_cc_indices, cc_indices = _grp_domain_order(packet.base.domain_names, weights)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(24, 15),
        gridspec_kw={"width_ratios": [1.0, 1.65], "wspace": 0.30},
        facecolor="white",
    )
    _plot_non_cc_block(
        ax=axes[0],
        indices=non_cc_indices,
        labels=[_display_non_cc_label(packet.base.domain_names[idx]) for idx in non_cc_indices],
        weights=weights,
        phase0_multipliers=packet.base.c0,
        phase1_multipliers=packet.base.c1,
        title="Non-CC Domains",
    )
    _plot_cc_block(
        ax=axes[1],
        domain_names=packet.base.domain_names,
        indices=cc_indices,
        weights=weights,
        phase0_multipliers=packet.base.c0,
        phase1_multipliers=packet.base.c1,
        title="CC Domains",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=29, y=0.985, fontweight="bold")
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=18, bbox_to_anchor=(0.5, 0.925))
    fig.text(0.5, 0.07, "Epoch labels; 80/20 WSD.", ha="center", va="center", fontsize=14, color=TEXT_MUTED_COLOR)
    fig.subplots_adjust(top=0.88, left=0.12, right=0.985, bottom=0.10, wspace=0.30)
    path = objective_dir / f"{opt_kind}_phase_comparison.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _run_scalar_candidate(
    data: pd.DataFrame,
    candidate: ProxyCandidate,
    *,
    family_scheme: str,
    model_options: dict[str, bool],
    start_bank: tuple[dict[str, float], ...],
    coarse_top_k: int,
    method: str,
    prob_eps: float,
    choice_target: str = TARGET_CHOICE,
    accuracy_target: str = TARGET_ACCURACY,
) -> tuple[dict[str, Any], pd.DataFrame]:
    packet, _params, pred_rows, model, objective = _fit_scalar_grp(
        data,
        candidate,
        family_scheme=family_scheme,
        model_options=model_options,
        start_bank=start_bank,
        coarse_top_k=coarse_top_k,
        method=method,
        prob_eps=prob_eps,
    )
    raw_result, phase0, phase1 = optimize_penalty_calibration_model(packet, model, seed=CV_SEED)
    raw_weights = np.stack([phase0, phase1], axis=0)
    score_model = ScoreModelAdapter(lambda weights: _model_target_to_metric(model.predict(weights), objective))
    actual_choice = data[choice_target].to_numpy(dtype=float)
    actual_accuracy = data[accuracy_target].to_numpy(dtype=float)
    diag, weight_map = _candidate_diagnostics(
        candidate_id=candidate.candidate_id,
        candidate_kind="scalar_proxy_target",
        target=candidate.target,
        packet=packet,
        score_model=score_model,
        actual_choice=actual_choice,
        actual_accuracy=actual_accuracy,
        actual_primary=data[candidate.target].to_numpy(dtype=float),
        raw_metric=float(_model_target_to_metric(float(raw_result.fun), objective)),
        raw_weights=raw_weights,
    )
    out_dir = OUTPUT_DIR / "grp_candidates" / f"{candidate.candidate_id}__scalar"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_rows.to_csv(out_dir / "oof_predictions.csv", index=False)
    _plot_predictions(pred_rows, out_dir / "predicted_vs_actual.html", candidate.candidate_id)
    _plot_residuals(pred_rows, out_dir / "residuals.html", candidate.candidate_id)
    diag.to_csv(out_dir / "optimum_diagnostics.csv", index=False)
    _write_optimum_weights(out_dir / "optimum_weights.csv", packet, weight_map)
    raw_plot = _plot_phase_comparison(
        out_dir,
        packet,
        "raw",
        f"Scalar proxy GRP raw optimum: {candidate.candidate_id}",
    )
    hull_plot = _plot_phase_comparison(
        out_dir,
        packet,
        "top8actual_hull",
        f"Scalar proxy GRP top8 hull: {candidate.candidate_id}",
    )
    predicted = pred_rows["predicted_metric"].to_numpy(dtype=float)
    summary = {
        "candidate_id": candidate.candidate_id,
        "candidate_kind": "scalar_proxy_target",
        "target": candidate.target,
        "feature_set": candidate.feature_set,
        "model_type": candidate.model_type,
        "transform": candidate.transform,
        "n_selected_features": len(candidate.selected_features),
        "raw_phase_plot": str(raw_plot),
        "top8_hull_phase_plot": str(hull_plot),
        **_metric_summary(actual_choice, predicted, prefix="choice_"),
        **_metric_summary(actual_accuracy, predicted, prefix="accuracy_"),
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


def _run_direct_target_candidate(
    data: pd.DataFrame,
    target: str,
    *,
    family_scheme: str,
    model_options: dict[str, bool],
    start_bank: tuple[dict[str, float], ...],
    coarse_top_k: int,
    method: str,
    prob_eps: float,
    choice_target: str = TARGET_CHOICE,
    accuracy_target: str = TARGET_ACCURACY,
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidate_id = f"direct_grp_{target}"
    target_column = f"{candidate_id}__target"
    direct_data = data.copy()
    direct_data[target_column] = np.clip(data[target].to_numpy(dtype=float), PROB_EPS, 1.0 - PROB_EPS)
    objective = _objective_for_probability(candidate_id, "objective_metric", target)
    packet = _packet_from_frame(_packet_frame(direct_data, target_column), objective, prob_eps, family_scheme)
    _coarse, best, _refine = _refine_rows(packet, start_bank, coarse_top_k, method, model_options)
    params = {key: float(best[key]) for key in _no_l2_param_keys()}
    params["reg"] = REG_FIXED
    model = build_penalty_calibration_surrogate(
        packet,
        params=params,
        variant_name=VARIANT_NAME,
        **model_options,
    ).fit(packet.base.w, packet.base.y)
    pred_rows = _prediction_rows(packet, params, objective)
    raw_result, phase0, phase1 = optimize_penalty_calibration_model(packet, model, seed=CV_SEED)
    raw_weights = np.stack([phase0, phase1], axis=0)
    score_model = ScoreModelAdapter(lambda weights: _model_target_to_metric(model.predict(weights), objective))
    actual_choice = data[choice_target].to_numpy(dtype=float)
    actual_accuracy = data[accuracy_target].to_numpy(dtype=float)
    diag, weight_map = _candidate_diagnostics(
        candidate_id=candidate_id,
        candidate_kind="direct_grp_target",
        target=target,
        packet=packet,
        score_model=score_model,
        actual_choice=actual_choice,
        actual_accuracy=actual_accuracy,
        actual_primary=data[target].to_numpy(dtype=float),
        raw_metric=float(_model_target_to_metric(float(raw_result.fun), objective)),
        raw_weights=raw_weights,
    )
    out_dir = OUTPUT_DIR / "grp_candidates" / candidate_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_rows.to_csv(out_dir / "oof_predictions.csv", index=False)
    _plot_predictions(pred_rows, out_dir / "predicted_vs_actual.html", candidate_id)
    _plot_residuals(pred_rows, out_dir / "residuals.html", candidate_id)
    diag.to_csv(out_dir / "optimum_diagnostics.csv", index=False)
    _write_optimum_weights(out_dir / "optimum_weights.csv", packet, weight_map)
    raw_plot = _plot_phase_comparison(out_dir, packet, "raw", f"Direct GRP raw optimum: {target}")
    hull_plot = _plot_phase_comparison(out_dir, packet, "top8actual_hull", f"Direct GRP top8 hull: {target}")
    predicted = pred_rows["predicted_metric"].to_numpy(dtype=float)
    summary = {
        "candidate_id": candidate_id,
        "candidate_kind": "direct_grp_target",
        "target": target,
        "feature_set": "direct_target",
        "model_type": "grp_no_l2",
        "transform": "probability",
        "n_selected_features": 0,
        "raw_phase_plot": str(raw_plot),
        "top8_hull_phase_plot": str(hull_plot),
        **_metric_summary(actual_choice, predicted, prefix="choice_"),
        **_metric_summary(actual_accuracy, predicted, prefix="accuracy_"),
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


def _run_component_candidate(
    data: pd.DataFrame,
    candidate: ProxyCandidate,
    *,
    family_scheme: str,
    model_options: dict[str, bool],
    start_bank: tuple[dict[str, float], ...],
    prob_eps: float,
    choice_target: str = TARGET_CHOICE,
    accuracy_target: str = TARGET_ACCURACY,
) -> tuple[dict[str, Any], pd.DataFrame]:
    packet, component_models, selected_features, params_rows, objectives = _fit_component_grps(
        data,
        candidate,
        family_scheme=family_scheme,
        model_options=model_options,
        start_bank=start_bank,
        prob_eps=prob_eps,
    )

    def predict_metric(weights: np.ndarray) -> np.ndarray:
        features = np.column_stack([model.predict(weights) for model in component_models])
        return _predict_selected_proxy(candidate, features)

    score_model = ScoreModelAdapter(predict_metric)
    actual_choice = data[choice_target].to_numpy(dtype=float)
    actual_accuracy = data[accuracy_target].to_numpy(dtype=float)
    seed_weights = [packet.base.w[int(idx)] for idx in np.argsort(-data[candidate.target].to_numpy(dtype=float))[:8]]
    for model in component_models:
        _result, phase0, phase1 = optimize_penalty_calibration_model(packet, model, n_random=0, seed=CV_SEED)
        seed_weights.append(np.stack([phase0, phase1], axis=0))
    raw_metric, raw_weights = _raw_weights_for_score_model(packet, score_model, seed_weights)
    diag, weight_map = _candidate_diagnostics(
        candidate_id=candidate.candidate_id,
        candidate_kind="component_proxy_ensemble",
        target=candidate.target,
        packet=packet,
        score_model=score_model,
        actual_choice=actual_choice,
        actual_accuracy=actual_accuracy,
        actual_primary=data[candidate.target].to_numpy(dtype=float),
        raw_metric=raw_metric,
        raw_weights=raw_weights,
    )
    out_dir = OUTPUT_DIR / "grp_candidates" / f"{candidate.candidate_id}__component"
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_pred = _component_oof_metric(
        data,
        candidate,
        family_scheme=family_scheme,
        model_options=model_options,
        prob_eps=prob_eps,
        selected_features=selected_features,
        params_rows=params_rows,
        objectives=objectives,
    )
    pred_rows = pd.DataFrame(
        {
            "run_name": packet.base.frame[packet.base.name_col].astype(str).to_numpy(),
            "actual_metric": data[candidate.target].to_numpy(dtype=float),
            "predicted_metric": oof_pred,
        }
    )
    pred_rows.to_csv(out_dir / "oof_predictions.csv", index=False)
    _plot_predictions(pred_rows, out_dir / "predicted_vs_actual.html", candidate.candidate_id)
    _plot_residuals(pred_rows, out_dir / "residuals.html", candidate.candidate_id)
    diag.to_csv(out_dir / "optimum_diagnostics.csv", index=False)
    _write_optimum_weights(out_dir / "optimum_weights.csv", packet, weight_map)
    raw_plot = _plot_phase_comparison(
        out_dir,
        packet,
        "raw",
        f"Component proxy raw optimum: {candidate.candidate_id}",
    )
    hull_plot = _plot_phase_comparison(
        out_dir,
        packet,
        "top8actual_hull",
        f"Component proxy top8 hull: {candidate.candidate_id}",
    )
    summary = {
        "candidate_id": candidate.candidate_id,
        "candidate_kind": "component_proxy_ensemble",
        "target": candidate.target,
        "feature_set": candidate.feature_set,
        "model_type": candidate.model_type,
        "transform": candidate.transform,
        "n_selected_features": len(selected_features),
        "raw_phase_plot": str(raw_plot),
        "top8_hull_phase_plot": str(hull_plot),
        **_metric_summary(actual_choice, oof_pred, prefix="choice_"),
        **_metric_summary(actual_accuracy, oof_pred, prefix="accuracy_"),
    }
    raw_row = diag.loc[diag["opt_kind"].eq("raw")].iloc[0].to_dict()
    hull_row = diag.loc[diag["opt_kind"].eq("top8actual_hull")].iloc[0].to_dict()
    for key, value in raw_row.items():
        if isinstance(value, int | float | np.integer | np.floating):
            summary[f"raw_{key}"] = float(value)
    for key, value in hull_row.items():
        if isinstance(value, int | float | np.integer | np.floating):
            summary[f"top8actual_hull_{key}"] = float(value)
    pd.DataFrame({"feature": selected_features}).to_csv(out_dir / "component_features.csv", index=False)
    return summary, diag


def _selected_proxy_candidates(
    proxy_frame: pd.DataFrame,
    candidates: dict[str, ProxyCandidate],
    targets: tuple[str, ...] = (TARGET_CHOICE, TARGET_ACCURACY),
) -> tuple[ProxyCandidate, ...]:
    selected_ids: list[str] = []
    for target in targets:
        for feature_set in ("eval_only", "eval_plus_lm_eval"):
            subset = proxy_frame.loc[(proxy_frame["target"].eq(target)) & (proxy_frame["feature_set"].eq(feature_set))]
            if subset.empty:
                continue
            row = subset.sort_values(
                ["selected_proxy_spearman", "selected_proxy_regret_at_1", "selected_proxy_rmse"],
                ascending=[False, True, True],
            ).iloc[0]
            selected_ids.append(str(row["candidate_id"]))
    return tuple(candidates[candidate_id] for candidate_id in selected_ids)


def _write_report(
    proxy_frame: pd.DataFrame,
    single_feature: pd.DataFrame,
    summary: pd.DataFrame,
    optimum: pd.DataFrame,
) -> None:
    proxy_columns = [
        "candidate_id",
        "target",
        "feature_set",
        "model_type",
        "transform",
        "n_features",
        "n_selected_features",
        "full_proxy_spearman",
        "selected_proxy_spearman",
        "selected_proxy_regret_at_1",
        "selected_proxy_rmse",
    ]
    summary_columns = [
        "candidate_id",
        "candidate_kind",
        "target",
        "feature_set",
        "choice_spearman",
        "accuracy_spearman",
        "choice_regret_at_1",
        "accuracy_regret_at_1",
        "raw_predicted_proxy_metric",
        "raw_nearest_observed_tv",
        "raw_nearest_observed_choice",
        "raw_nearest_observed_accuracy",
        "top8actual_hull_nearest_observed_tv",
    ]
    body = [
        "# 300M Perplexity-Proxy Benchmark Optimization",
        "",
        "## Setup",
        "",
        f"- Rows: 240 qsplit-core `{SCALE}` rows.",
        "- Targets: mean `choice_prob_norm` and mean `acc` over 119 matched tasks, excluding `mmlu_pro_5shot`.",
        "- Feature sets: `eval_only` is deployable BPB/loss features; `eval_plus_lm_eval` is a diagnostic upper bound.",
        "",
        "## Best Proxy Fits",
        "",
        proxy_frame.sort_values(["target", "feature_set", "selected_proxy_spearman"], ascending=[True, True, False])[
            proxy_columns
        ]
        .groupby(["target", "feature_set"], as_index=False)
        .head(2)
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Best Single Perplexity Baselines",
        "",
        single_feature.groupby(["target", "feature_set"], as_index=False)
        .head(1)[["target", "feature_set", "feature", "single_feature_spearman", "single_feature_regret_at_1"]]
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "## GRP Proxy Candidates",
        "",
        summary[summary_columns]
        .sort_values(
            ["target", "candidate_kind", "choice_spearman"],
            ascending=[True, True, False],
        )
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Optimum Diagnostics",
        "",
        optimum[
            [
                "candidate_id",
                "candidate_kind",
                "target",
                "opt_kind",
                "predicted_proxy_metric",
                "nearest_observed_tv",
                "nearest_observed_choice",
                "nearest_observed_accuracy",
                "nearest_observed_primary_regret",
                "raw_phase0_support_gt_1e4",
                "raw_phase1_support_gt_1e4",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "- Proxy regressions are evaluated as selectors, not just calibrated predictors.",
        "- `eval_plus_lm_eval` is an upper bound and may be benchmark-specific leakage; "
        "deployable claims should use `eval_only`.",
        "- Raw optima with large nearest-observed TV or low support are not deployable, "
        "even when predicted proxy scores look high.",
        "",
    ]
    REPORT_MD.write_text("\n".join(body), encoding="utf-8")


def _write_logbook(summary: pd.DataFrame) -> None:
    LOGBOOK_MD.parent.mkdir(parents=True, exist_ok=True)
    best_rows = summary.sort_values(["choice_spearman", "accuracy_spearman"], ascending=[False, False]).head(4)
    text = [
        "# Benchmark Proxy Optimization: Research Logbook",
        "",
        "## Scope",
        "- Goal: test whether smooth perplexity/BPB proxies can improve 300M benchmark mixture optimization.",
        "- Primary metrics: OOF rank/regret against mean choice probability and mean accuracy, excluding MMLU-Pro.",
        "- Constraints: local-only modeling, no new training/eval launches.",
        "",
        "## Experiment Log",
        "### 2026-04-28 - Perplexity proxy sprint",
        "- Hypothesis: GRP models smooth BPB/perplexity surfaces better than bounded "
        "benchmark probabilities, so a learned BPB proxy may produce better selectors "
        "or saner optima.",
        f"- Command: `uv run --with matplotlib --with torch python {Path(__file__)}`",
        "- Result summary:",
        best_rows[
            [
                "candidate_id",
                "candidate_kind",
                "target",
                "feature_set",
                "choice_spearman",
                "accuracy_spearman",
                "raw_nearest_observed_tv",
                "raw_nearest_observed_choice",
                "raw_nearest_observed_accuracy",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "- Artifacts: "
        "`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
        "grp_300m_perplexity_proxy_benchmark_20260428/`.",
        "",
    ]
    LOGBOOK_MD.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    data, feature_sets, _choice_columns, _accuracy_columns = _build_dataset()
    single_feature = _best_single_feature_baselines(data, feature_sets)
    proxy_frame, _feature_frame, candidates = _fit_proxy_candidates(
        data,
        feature_sets,
        max_selected_features=args.max_selected_features,
    )
    selected_candidates = _selected_proxy_candidates(proxy_frame, candidates)
    start_bank = _expanded_start_bank(args.random_starts)
    model_options = _model_options(args.block_variant)

    summary_rows: list[dict[str, Any]] = []
    optimum_rows: list[pd.DataFrame] = []
    for target in (TARGET_CHOICE, TARGET_ACCURACY):
        print(f"Running direct GRP target baseline {target}", flush=True)
        summary, diag = _run_direct_target_candidate(
            data,
            target,
            family_scheme=args.family_scheme,
            model_options=model_options,
            start_bank=start_bank,
            coarse_top_k=args.coarse_top_k,
            method=args.method,
            prob_eps=args.prob_eps,
        )
        summary_rows.append(summary)
        optimum_rows.append(diag)
    for candidate in selected_candidates:
        print(f"Running scalar GRP candidate {candidate.candidate_id}", flush=True)
        summary, diag = _run_scalar_candidate(
            data,
            candidate,
            family_scheme=args.family_scheme,
            model_options=model_options,
            start_bank=start_bank,
            coarse_top_k=args.coarse_top_k,
            method=args.method,
            prob_eps=args.prob_eps,
        )
        summary_rows.append(summary)
        optimum_rows.append(diag)
        print(f"Running component GRP candidate {candidate.candidate_id}", flush=True)
        summary, diag = _run_component_candidate(
            data,
            candidate,
            family_scheme=args.family_scheme,
            model_options=model_options,
            start_bank=start_bank,
            prob_eps=args.prob_eps,
        )
        summary_rows.append(summary)
        optimum_rows.append(diag)

    summary_frame = pd.DataFrame.from_records(summary_rows)
    optimum_frame = pd.concat(optimum_rows, ignore_index=True)
    summary_frame.to_csv(SUMMARY_CSV, index=False)
    optimum_frame.to_csv(OPTIMUM_DIAGNOSTICS_CSV, index=False)
    _write_report(proxy_frame, single_feature, summary_frame, optimum_frame)
    _write_logbook(summary_frame)
    print(f"Wrote {PROXY_FITS_CSV}")
    print(f"Wrote {SELECTED_FEATURES_CSV}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {OPTIMUM_DIAGNOSTICS_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {LOGBOOK_MD}")


if __name__ == "__main__":
    main()
