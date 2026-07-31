# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Predicted GRP optima with nonlinear parameters retuned per observed-run subset."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    load_two_phase_many_candidate_summary_spec,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GENERIC_FAMILY_NAMES,
    TUNED_GENERIC_FAMILY_PARAMS,
    GenericFamilyPacket,
    GenericFamilyRetainedTotalSurrogate,
    family_shares,
    load_generic_family_packet,
    optimize_generic_family_model,
)
from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights
from experiments.domain_phase_mix.static_batch_selection import retrospective_generic_selection
from experiments.domain_phase_mix.two_phase_many_ccglobalpremium_baselines import (
    ccglobalpremium_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_ccpairtotal_baseline import (
    ccpairtotal_retainedtotal_summary,
)

GENERICFAMILY_RETUNED_SUBSET_OPTIMA_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_retuned_subset_optima_rep_uncheatable_bpb"
)
GENERICFAMILY_RETUNED_SUBSET_OPTIMA_BASE_RUN_ID = 340
GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES = tuple(range(20, 240, 20))
GENERICFAMILY_RETUNED_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES = (40, 100, 180, 220)
GENERICFAMILY_RETUNED_SUBSET_OPTIMA_POLICY = "feature_bayes_linear_observed_retuned"
GENERICFAMILY_RETUNED_SUBSET_OPTIMA_OBJECTIVE = "cv_rmse+anchor_mae+0.02*cv_foldmean_regret"
CSV_PATH = Path(__file__).resolve().parent / "exploratory" / "two_phase_many" / "two_phase_many.csv"
OBJECTIVE_METRIC = "eval/uncheatable_eval/bpb"
VALIDATED_GLOBAL_BPB = 1.04381
VALIDATED_PAIR_BPB = 1.04794
CV_WEIGHT = 1.0
ANCHOR_WEIGHT = 1.0
REGRET_WEIGHT = 0.02
DEFAULT_TUNING_METHOD = "L-BFGS-B"
DEFAULT_TUNING_OBJECTIVE_NAME = "single_foldmean"
BROAD_BETA_GENERIC_FAMILY_PARAMS = {
    "alpha": 11.533461482593735,
    "eta": 10.859113730214359,
    "lam": 0.3422735488822989,
    "tau": 2.843180828656475,
    "reg": 0.0001896587113845684,
    "beta": 0.9324427249160729,
}


@dataclass(frozen=True)
class GenericFamilyRetunedSubsetOptimumSummary:
    """Summary for one GRP optimum with subset-specific nonlinear retuning."""

    subset_size: int
    run_id: int
    run_name: str
    policy: str
    objective_metric: str
    tuning_method: str
    tuning_objective_name: str
    tuning_objective: float
    tuning_cv_rmse: float
    tuning_cv_r2: float
    tuning_cv_spearman: float
    tuning_cv_regret_at_1: float
    tuning_cv_foldmean_regret_at_1: float
    tuning_anchor_mae: float
    tuning_anchor_rmse: float
    tuned_params: dict[str, float]
    predicted_optimum_value: float
    fullswarm_chosen_run_name: str
    fullswarm_chosen_value: float
    fullswarm_regret_at_1: float
    observed_best_run_name: str
    observed_best_value: float
    gap_below_observed_best: float
    nearest_observed_run_name: str
    nearest_observed_value: float
    nearest_observed_tv_distance: float
    optimum_move_mean_phase_tv_vs_prev: float | None
    phase0_max_weight: float
    phase1_max_weight: float
    phase0_support_below_1e4: int
    phase1_support_below_1e4: int
    phase0_top_domains: list[dict[str, float | str]]
    phase1_top_domains: list[dict[str, float | str]]
    optimizer_success: bool
    optimizer_message: str
    tuning_success: bool
    tuning_message: str
    family_shares: dict[str, float]
    phase_weights: dict[str, dict[str, float]]


def _subset_packet(packet: GenericFamilyPacket, indices: np.ndarray) -> GenericFamilyPacket:
    indices = np.asarray(indices, dtype=int)
    return GenericFamilyPacket(
        base=replace(
            packet.base,
            frame=packet.base.frame.iloc[indices].reset_index(drop=True),
            y=packet.base.y[indices],
            w=packet.base.w[indices],
        ),
        pairs=packet.pairs,
        pair_topics=packet.pair_topics,
        singletons=packet.singletons,
        family_map=packet.family_map,
    )


def _top_domains(
    domain_names: list[str],
    weights: np.ndarray,
    epochs: np.ndarray,
    *,
    top_k: int = 10,
) -> list[dict[str, float | str]]:
    frame = pd.DataFrame({"domain": domain_names, "weight": weights, "epochs": epochs})
    return frame.sort_values(["weight", "epochs"], ascending=False).head(top_k).to_dict(orient="records")


def _phase_weights_from_array(domain_names: list[str], weights: np.ndarray) -> dict[str, dict[str, float]]:
    return normalize_phase_weights(
        {
            "phase_0": {
                domain_name: float(weight) for domain_name, weight in zip(domain_names, weights[0], strict=True)
            },
            "phase_1": {
                domain_name: float(weight) for domain_name, weight in zip(domain_names, weights[1], strict=True)
            },
        }
    )


def _mean_phase_tv_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return 0.5 * float(np.mean(np.sum(np.abs(lhs - rhs), axis=1)))


def _summary_weights(summary: dict[str, object], domain_names: list[str]) -> np.ndarray:
    phase_weights = summary["phase_weights"]
    phase0 = np.asarray([float(phase_weights["phase_0"][domain_name]) for domain_name in domain_names], dtype=float)
    phase1 = np.asarray([float(phase_weights["phase_1"][domain_name]) for domain_name in domain_names], dtype=float)
    return np.stack([phase0, phase1], axis=0)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0))))


def _pack_params(params: dict[str, float]) -> np.ndarray:
    beta = float(np.clip(params["beta"], 1e-8, 1.0 - 1e-8))
    return np.asarray(
        [
            np.log(float(params["alpha"])),
            np.log(float(params["eta"])),
            np.log(float(params["lam"])),
            float(params["tau"]),
            np.log(float(params["reg"])),
            np.log(beta / (1.0 - beta)),
        ],
        dtype=float,
    )


def _unpack_params(z: np.ndarray) -> dict[str, float]:
    return {
        "alpha": float(np.exp(np.clip(z[0], -8.0, 8.0))),
        "eta": float(np.exp(np.clip(z[1], -8.0, 8.0))),
        "lam": float(np.exp(np.clip(z[2], -12.0, 4.0))),
        "tau": float(np.clip(z[3], -2.0, 8.0)),
        "reg": float(np.exp(np.clip(z[4], -18.0, -2.0))),
        "beta": float(np.clip(_sigmoid(float(z[5])), 1e-6, 1.0 - 1e-6)),
    }


def _evaluate_params(
    z: np.ndarray,
    packet: GenericFamilyPacket,
    valid_weights: np.ndarray,
    valid_y: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, float | bool]:
    params = _unpack_params(z)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros_like(packet.base.y)
    fold_regrets: list[float] = []

    for tr, te in kf.split(packet.base.w):
        model = GenericFamilyRetainedTotalSurrogate(
            packet,
            params=params,
            family_totals=GENERIC_FAMILY_NAMES,
            quality_discount=True,
        ).fit(packet.base.w[tr], packet.base.y[tr])
        pred = model.predict(packet.base.w[te])
        oof[te] = pred
        fold_regrets.append(float(packet.base.y[te][int(np.argmin(pred))] - np.min(packet.base.y[te])))

    full_model = GenericFamilyRetainedTotalSurrogate(
        packet,
        params=params,
        family_totals=GENERIC_FAMILY_NAMES,
        quality_discount=True,
    ).fit(packet.base.w, packet.base.y)
    train_pred = full_model.predict(packet.base.w)
    anchor_pred = full_model.predict(valid_weights)
    anchor_err = anchor_pred - valid_y

    train_res = train_pred - packet.base.y
    cv_res = oof - packet.base.y
    sst = float(np.sum((packet.base.y - np.mean(packet.base.y)) ** 2))
    cv_rmse = float(np.sqrt(np.mean(cv_res**2)))
    anchor_mae = float(np.mean(np.abs(anchor_err)))
    foldmean_regret = float(np.mean(fold_regrets))
    objective = CV_WEIGHT * cv_rmse + ANCHOR_WEIGHT * anchor_mae + REGRET_WEIGHT * foldmean_regret

    return {
        **params,
        "objective": objective,
        "train_rmse": float(np.sqrt(np.mean(train_res**2))),
        "train_r2": float(1.0 - float(np.sum(train_res**2)) / sst),
        "train_spearman": float(spearmanr(packet.base.y, train_pred).statistic),
        "cv_rmse": cv_rmse,
        "cv_r2": float(1.0 - float(np.sum(cv_res**2)) / sst),
        "cv_spearman": float(spearmanr(packet.base.y, oof).statistic),
        "cv_regret_at_1": float(packet.base.y[int(np.argmin(oof))] - np.min(packet.base.y)),
        "cv_foldmean_regret_at_1": foldmean_regret,
        "anchor_mae": anchor_mae,
        "anchor_rmse": float(np.sqrt(np.mean(anchor_err**2))),
        "anchor_rank_correct": bool(int(np.argmin(anchor_pred)) == int(np.argmin(valid_y))),
        "pred_validated_global": float(anchor_pred[0]),
        "pred_validated_pair": float(anchor_pred[1]),
    }


def _objective_value_from_metrics(metrics: dict[str, float | bool], objective_name: str) -> float:
    if objective_name == "single_foldmean":
        return float(metrics["objective"])
    if objective_name == "single_cvregret":
        return float(metrics["cv_rmse"]) + float(metrics["anchor_mae"]) + 0.2 * float(metrics["cv_regret_at_1"])
    if objective_name == "single_both":
        return (
            float(metrics["cv_rmse"])
            + float(metrics["anchor_mae"])
            + 0.2 * float(metrics["cv_regret_at_1"])
            + 0.02 * float(metrics["cv_foldmean_regret_at_1"])
        )
    raise ValueError(f"Unsupported tuning objective: {objective_name}")


def tune_genericfamily_subset_params(
    packet: GenericFamilyPacket,
    valid_weights: np.ndarray,
    valid_y: np.ndarray,
    *,
    method: str = DEFAULT_TUNING_METHOD,
    objective_name: str = DEFAULT_TUNING_OBJECTIVE_NAME,
    start_params: dict[str, float] | None = None,
    seed: int = 0,
) -> tuple[dict[str, float | bool], Any]:
    start = _pack_params(TUNED_GENERIC_FAMILY_PARAMS if start_params is None else start_params)

    def objective(z: np.ndarray) -> float:
        metrics = _evaluate_params(z, packet, valid_weights, valid_y, seed=seed)
        return _objective_value_from_metrics(metrics, objective_name)

    result = minimize(
        objective,
        start,
        method=method,
        options={
            "L-BFGS-B": {"maxiter": 250, "ftol": 1e-6},
            "Nelder-Mead": {"maxiter": 900, "xatol": 1e-4, "fatol": 1e-6},
            "Powell": {"maxiter": 400, "xtol": 1e-4, "ftol": 1e-6},
        }.get(method, {"maxiter": 250}),
    )
    metrics = _evaluate_params(np.asarray(result.x, dtype=float), packet, valid_weights, valid_y, seed=seed)
    metrics = {
        "success": bool(result.success),
        "message": str(result.message),
        "method": method,
        "objective_name": objective_name,
        **metrics,
        "objective": _objective_value_from_metrics(metrics, objective_name),
    }
    return metrics, result


def genericfamily_retuned_subset_optimum_run_id(subset_size: int) -> int:
    """Return the canonical run id for one retuned subset-fit GRP optimum."""
    if subset_size not in GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES:
        raise ValueError(f"Unsupported subset size: {subset_size}")
    return GENERICFAMILY_RETUNED_SUBSET_OPTIMA_BASE_RUN_ID + GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES.index(
        subset_size
    )


def genericfamily_retuned_subset_optimum_run_name(subset_size: int) -> str:
    """Return the canonical run name for one retuned subset-fit GRP optimum."""
    return f"baseline_genericfamily_retuned_k{subset_size:03d}_uncheatable_bpb"


def parse_subset_sizes(spec: str) -> tuple[int, ...]:
    """Parse a comma-separated subset-size spec."""
    cleaned = spec.strip().lower()
    if cleaned == "all":
        return GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES
    values = tuple(int(part.strip()) for part in spec.split(",") if part.strip())
    if not values:
        raise ValueError("subset size spec must not be empty")
    invalid = [value for value in values if value not in GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES]
    if invalid:
        raise ValueError(f"Unsupported subset sizes: {invalid}")
    return values


def _summary_to_dict(summary: GenericFamilyRetunedSubsetOptimumSummary) -> dict[str, Any]:
    return {
        "subset_size": summary.subset_size,
        "run_id": summary.run_id,
        "run_name": summary.run_name,
        "policy": summary.policy,
        "objective_metric": summary.objective_metric,
        "tuning_method": summary.tuning_method,
        "tuning_objective_name": summary.tuning_objective_name,
        "tuning_objective": summary.tuning_objective,
        "tuning_cv_rmse": summary.tuning_cv_rmse,
        "tuning_cv_r2": summary.tuning_cv_r2,
        "tuning_cv_spearman": summary.tuning_cv_spearman,
        "tuning_cv_regret_at_1": summary.tuning_cv_regret_at_1,
        "tuning_cv_foldmean_regret_at_1": summary.tuning_cv_foldmean_regret_at_1,
        "tuning_anchor_mae": summary.tuning_anchor_mae,
        "tuning_anchor_rmse": summary.tuning_anchor_rmse,
        "tuned_params": summary.tuned_params,
        "predicted_optimum_value": summary.predicted_optimum_value,
        "fullswarm_chosen_run_name": summary.fullswarm_chosen_run_name,
        "fullswarm_chosen_value": summary.fullswarm_chosen_value,
        "fullswarm_regret_at_1": summary.fullswarm_regret_at_1,
        "observed_best_run_name": summary.observed_best_run_name,
        "observed_best_value": summary.observed_best_value,
        "gap_below_observed_best": summary.gap_below_observed_best,
        "nearest_observed_run_name": summary.nearest_observed_run_name,
        "nearest_observed_value": summary.nearest_observed_value,
        "nearest_observed_tv_distance": summary.nearest_observed_tv_distance,
        "optimum_move_mean_phase_tv_vs_prev": summary.optimum_move_mean_phase_tv_vs_prev,
        "phase0_max_weight": summary.phase0_max_weight,
        "phase1_max_weight": summary.phase1_max_weight,
        "phase0_support_below_1e4": summary.phase0_support_below_1e4,
        "phase1_support_below_1e4": summary.phase1_support_below_1e4,
        "phase0_top_domains": summary.phase0_top_domains,
        "phase1_top_domains": summary.phase1_top_domains,
        "optimizer_success": summary.optimizer_success,
        "optimizer_message": summary.optimizer_message,
        "tuning_success": summary.tuning_success,
        "tuning_message": summary.tuning_message,
        "family_shares": summary.family_shares,
        "phase_weights": summary.phase_weights,
    }


@cache
def genericfamily_retuned_subset_optima_summaries(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_RETUNED_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
    tuning_method: str = DEFAULT_TUNING_METHOD,
) -> tuple[GenericFamilyRetunedSubsetOptimumSummary, ...]:
    """Return predicted GRP optima with nonlinear params retuned per subset."""
    _, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many_genericfamily_retuned_subset_optima",
    )
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    valid_weights = np.stack(
        [
            _summary_weights(ccglobalpremium_retainedtotal_summary(), packet.base.domain_names),
            _summary_weights(ccpairtotal_retainedtotal_summary(), packet.base.domain_names),
        ],
        axis=0,
    )
    valid_y = np.asarray([VALIDATED_GLOBAL_BPB, VALIDATED_PAIR_BPB], dtype=float)
    best_idx = int(np.argmin(packet.base.y))
    best_value = float(packet.base.y[best_idx])
    previous_optimum: np.ndarray | None = None
    summaries: list[GenericFamilyRetunedSubsetOptimumSummary] = []

    for subset_size in subset_sizes:
        selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=subset_size, seed=0)
        subset_indices = np.asarray(selection.selected_indices, dtype=int)
        train_packet = _subset_packet(packet, subset_indices)
        tuning_metrics, _ = tune_genericfamily_subset_params(
            train_packet,
            valid_weights,
            valid_y,
            method=tuning_method,
            seed=0,
        )
        tuned_params = {key: float(tuning_metrics[key]) for key in ("alpha", "eta", "lam", "tau", "reg", "beta")}
        model = GenericFamilyRetainedTotalSurrogate(train_packet, params=tuned_params).fit(
            train_packet.base.w, train_packet.base.y
        )
        fullswarm_predictions = model.predict(packet.base.w)
        chosen_idx = int(np.argmin(fullswarm_predictions))
        result, phase0, phase1 = optimize_generic_family_model(train_packet, model, seed=0)
        optimum = np.stack([phase0, phase1], axis=0)
        distances = 0.5 * np.abs(packet.base.w - optimum[None, :, :]).sum(axis=2).mean(axis=1)
        nearest_idx = int(np.argmin(distances))

        summaries.append(
            GenericFamilyRetunedSubsetOptimumSummary(
                subset_size=subset_size,
                run_id=genericfamily_retuned_subset_optimum_run_id(subset_size),
                run_name=genericfamily_retuned_subset_optimum_run_name(subset_size),
                policy=GENERICFAMILY_RETUNED_SUBSET_OPTIMA_POLICY,
                objective_metric=OBJECTIVE_METRIC,
                tuning_method=tuning_method,
                tuning_objective_name=GENERICFAMILY_RETUNED_SUBSET_OPTIMA_OBJECTIVE,
                tuning_objective=float(tuning_metrics["objective"]),
                tuning_cv_rmse=float(tuning_metrics["cv_rmse"]),
                tuning_cv_r2=float(tuning_metrics["cv_r2"]),
                tuning_cv_spearman=float(tuning_metrics["cv_spearman"]),
                tuning_cv_regret_at_1=float(tuning_metrics["cv_regret_at_1"]),
                tuning_cv_foldmean_regret_at_1=float(tuning_metrics["cv_foldmean_regret_at_1"]),
                tuning_anchor_mae=float(tuning_metrics["anchor_mae"]),
                tuning_anchor_rmse=float(tuning_metrics["anchor_rmse"]),
                tuned_params=tuned_params,
                predicted_optimum_value=float(result.fun),
                fullswarm_chosen_run_name=str(packet.base.frame.iloc[chosen_idx][packet.base.name_col]),
                fullswarm_chosen_value=float(packet.base.y[chosen_idx]),
                fullswarm_regret_at_1=float(packet.base.y[chosen_idx] - best_value),
                observed_best_run_name=str(packet.base.frame.iloc[best_idx][packet.base.name_col]),
                observed_best_value=best_value,
                gap_below_observed_best=float(result.fun - best_value),
                nearest_observed_run_name=str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
                nearest_observed_value=float(packet.base.y[nearest_idx]),
                nearest_observed_tv_distance=float(distances[nearest_idx]),
                optimum_move_mean_phase_tv_vs_prev=(
                    None if previous_optimum is None else _mean_phase_tv_distance(optimum, previous_optimum)
                ),
                phase0_max_weight=float(phase0.max()),
                phase1_max_weight=float(phase1.max()),
                phase0_support_below_1e4=int(np.sum(phase0 < 1e-4)),
                phase1_support_below_1e4=int(np.sum(phase1 < 1e-4)),
                phase0_top_domains=_top_domains(packet.base.domain_names, phase0, phase0 * packet.base.c0),
                phase1_top_domains=_top_domains(packet.base.domain_names, phase1, phase1 * packet.base.c1),
                optimizer_success=bool(result.success),
                optimizer_message=str(result.message),
                tuning_success=bool(tuning_metrics["success"]),
                tuning_message=str(tuning_metrics["message"]),
                family_shares=family_shares(packet, optimum),
                phase_weights=_phase_weights_from_array(packet.base.domain_names, optimum),
            )
        )
        previous_optimum = optimum

    return tuple(summaries)


def genericfamily_retuned_subset_optima_summaries_json(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_RETUNED_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
    tuning_method: str = DEFAULT_TUNING_METHOD,
) -> str:
    """Return the retuned subset-optima summaries as JSON."""
    return json.dumps(
        [
            _summary_to_dict(summary)
            for summary in genericfamily_retuned_subset_optima_summaries(subset_sizes, tuning_method)
        ],
        indent=2,
    )


def genericfamily_retuned_subset_optima_summaries_frame(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_RETUNED_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
    tuning_method: str = DEFAULT_TUNING_METHOD,
) -> pd.DataFrame:
    """Return a flat summary frame for the retuned subset-optimum sweep."""
    return pd.DataFrame(
        [
            {
                "subset_size": summary.subset_size,
                "run_id": summary.run_id,
                "run_name": summary.run_name,
                "policy": summary.policy,
                "tuning_method": summary.tuning_method,
                "predicted_optimum_value": summary.predicted_optimum_value,
                "fullswarm_chosen_run_name": summary.fullswarm_chosen_run_name,
                "fullswarm_chosen_value": summary.fullswarm_chosen_value,
                "fullswarm_regret_at_1": summary.fullswarm_regret_at_1,
                "observed_best_value": summary.observed_best_value,
                "gap_below_observed_best": summary.gap_below_observed_best,
                "nearest_observed_run_name": summary.nearest_observed_run_name,
                "nearest_observed_value": summary.nearest_observed_value,
                "nearest_observed_tv_distance": summary.nearest_observed_tv_distance,
                "optimum_move_mean_phase_tv_vs_prev": summary.optimum_move_mean_phase_tv_vs_prev,
                "phase0_max_weight": summary.phase0_max_weight,
                "phase1_max_weight": summary.phase1_max_weight,
                "phase0_support_below_1e4": summary.phase0_support_below_1e4,
                "phase1_support_below_1e4": summary.phase1_support_below_1e4,
                "tuning_objective": summary.tuning_objective,
                "tuning_cv_rmse": summary.tuning_cv_rmse,
                "tuning_cv_r2": summary.tuning_cv_r2,
                "tuning_cv_spearman": summary.tuning_cv_spearman,
                "tuning_cv_regret_at_1": summary.tuning_cv_regret_at_1,
                "tuning_cv_foldmean_regret_at_1": summary.tuning_cv_foldmean_regret_at_1,
                "tuning_anchor_mae": summary.tuning_anchor_mae,
                "tuning_anchor_rmse": summary.tuning_anchor_rmse,
                "alpha": summary.tuned_params["alpha"],
                "eta": summary.tuned_params["eta"],
                "lam": summary.tuned_params["lam"],
                "tau": summary.tuned_params["tau"],
                "reg": summary.tuned_params["reg"],
                "beta": summary.tuned_params["beta"],
            }
            for summary in genericfamily_retuned_subset_optima_summaries(subset_sizes, tuning_method)
        ]
    )


def create_genericfamily_retuned_subset_optimum_weight_config(subset_size: int) -> WeightConfig:
    """Return the weight config for one retuned subset-fit predicted optimum."""
    summary = next(
        summary
        for summary in genericfamily_retuned_subset_optima_summaries(
            GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES
        )
        if summary.subset_size == subset_size
    )
    return WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)
