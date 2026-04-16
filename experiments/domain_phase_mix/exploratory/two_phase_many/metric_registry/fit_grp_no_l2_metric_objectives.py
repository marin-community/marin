# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Fit no-L2 GRP power-family-penalty models for registry metrics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_grp_power_family_penalty_no_l2_retune import (
    REG_FIXED,
    VARIANT_NAME,
    _no_l2_param_keys,
    _parameter_counts,
    _pack_no_l2_params,
    _start_bank,
    _unpack_no_l2_params,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.materialize_fit_dataset import (
    materialize_fit_dataset,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GENERIC_FAMILY_NAMES,
    GenericFamilyPacket,
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
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    PacketData,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import build_two_phase_many_loop_config
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance, build_dataset_spec_from_frame

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "grp_no_l2_metric_objective_fits"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
PARAMS_CSV = OUTPUT_DIR / "params.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"
MARKDOWN_MD = OUTPUT_DIR / "summary.md"
MODEL_TARGET_COLUMN = "model_target"
CV_SEED = 0


@dataclass(frozen=True)
class ObjectiveSpec:
    """One registry metric objective to fit."""

    slug: str
    metric_key: str
    display_name: str
    lower_is_better: bool


OBJECTIVES = (
    ObjectiveSpec(
        slug="uncheatable_macro_bpb",
        metric_key="eval/uncheatable_eval/macro_bpb",
        display_name="Uncheatable Eval Macro BPB",
        lower_is_better=True,
    ),
    ObjectiveSpec(
        slug="piqa_5shot_bpb",
        metric_key="lm_eval/piqa_5shot/bpb",
        display_name="PIQA 5-shot BPB",
        lower_is_better=True,
    ),
    ObjectiveSpec(
        slug="piqa_5shot_choice_logprob",
        metric_key="lm_eval/piqa_5shot/choice_logprob",
        display_name="PIQA 5-shot Choice Logprob",
        lower_is_better=False,
    ),
    ObjectiveSpec(
        slug="paloma_macro_bpb",
        metric_key="eval/paloma/macro_bpb",
        display_name="Paloma Macro BPB",
        lower_is_better=True,
    ),
)


def _generic_family_packet_from_base(base: PacketData) -> GenericFamilyPacket:
    pairs: list[tuple[int, int]] = []
    pair_topics: list[str] = []
    paired: set[int] = set()

    for idx, domain_name in enumerate(base.domain_names):
        if idx in paired:
            continue
        if domain_name.startswith("dolma3_cc/") and domain_name.endswith("_high"):
            low_name = domain_name[:-5] + "_low"
            if low_name in base.domain_names:
                low_idx = base.domain_names.index(low_name)
                pairs.append((idx, low_idx))
                pair_topics.append(domain_name[len("dolma3_cc/") : -5])
                paired.add(idx)
                paired.add(low_idx)

    singletons = [idx for idx in range(base.m) if idx not in paired]
    family_map = {family_name: [] for family_name in GENERIC_FAMILY_NAMES}
    for idx, domain_name in enumerate(base.domain_names):
        is_broad = (
            domain_name.startswith("dolma3_cc/")
            or domain_name
            in {
                "dolma3_wikipedia",
                "dolmino_common_crawl_hq",
                "dolmino_olmocr_pdfs_hq",
                "dolmino_stem_heavy_crawl",
            }
            or domain_name.endswith("synth_qa")
        )
        is_tech = any(token in domain_name for token in ("stack_edu", "synth_code", "synth_math")) or domain_name in {
            "dolma3_arxiv",
            "dolma3_finemath_3plus",
        }
        is_reasoning = domain_name in {"dolmino_synth_instruction", "dolmino_synth_thinking"}

        if is_broad:
            family_map["broad_text"].append(idx)
        if is_tech:
            family_map["tech_code"].append(idx)
        if is_reasoning:
            family_map["reasoning"].append(idx)

    return GenericFamilyPacket(
        base=base,
        pairs=pairs,
        pair_topics=pair_topics,
        singletons=singletons,
        family_map=family_map,
    )


def _load_packet(spec: ObjectiveSpec) -> tuple[GenericFamilyPacket, pd.DataFrame]:
    fit_frame = materialize_fit_dataset(
        spec.metric_key,
        scale="60m_1p2b",
        cohort="signal",
        run_set="fit_swarm_60m_default",
    )
    fit_frame = fit_frame.copy()
    fit_frame[MODEL_TARGET_COLUMN] = (
        fit_frame["objective_metric"] if spec.lower_is_better else -fit_frame["objective_metric"]
    )
    loop = build_two_phase_many_loop_config(
        objective_metric=MODEL_TARGET_COLUMN,
        name=f"grp_no_l2_{spec.slug}",
    )
    dataset_spec = build_dataset_spec_from_frame(
        fit_frame,
        objective_metric=MODEL_TARGET_COLUMN,
        name=f"grp_no_l2_{spec.slug}",
        loop=loop,
    )
    name_col = "candidate_run_name" if "candidate_run_name" in fit_frame.columns else "run_name"
    base = PacketData(
        frame=fit_frame.reset_index(drop=True),
        name_col=name_col,
        y=dataset_spec.y,
        w=dataset_spec.weights,
        m=dataset_spec.M,
        c0=np.asarray(dataset_spec.epoch_multipliers[0], dtype=float),
        c1=np.asarray(dataset_spec.epoch_multipliers[1], dtype=float),
        domain_names=list(dataset_spec.domain_names),
    )
    return _generic_family_packet_from_base(base), fit_frame


def _model_target_to_metric(value: float, spec: ObjectiveSpec) -> float:
    return float(value if spec.lower_is_better else -value)


def _regret(model_target_value: float, best_model_target_value: float) -> float:
    return float(model_target_value - best_model_target_value)


def _prediction_quality(packet: GenericFamilyPacket, params: dict[str, float]) -> dict[str, float]:
    y = packet.base.y
    kf = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    oof = np.zeros_like(y)
    for train_idx, test_idx in kf.split(packet.base.w):
        model = build_penalty_calibration_surrogate(packet, params=params, variant_name=VARIANT_NAME).fit(
            packet.base.w[train_idx],
            y[train_idx],
        )
        oof[test_idx] = model.predict(packet.base.w[test_idx])

    residuals = oof - y
    train_model = build_penalty_calibration_surrogate(packet, params=params, variant_name=VARIANT_NAME).fit(
        packet.base.w,
        y,
    )
    train_pred = train_model.predict(packet.base.w)
    train_residuals = train_pred - y
    chosen_idx = int(np.argmin(oof))
    best_idx = int(np.argmin(y))
    return {
        "oof_rmse": float(np.sqrt(np.mean(residuals**2))),
        "oof_r2": float(1.0 - np.sum(residuals**2) / np.sum((y - np.mean(y)) ** 2)),
        "oof_spearman": float(stats.spearmanr(y, oof).statistic),
        "oof_regret_at_1": _regret(float(y[chosen_idx]), float(y[best_idx])),
        "train_r2": float(1.0 - np.sum(train_residuals**2) / np.sum((y - np.mean(y)) ** 2)),
        "train_spearman": float(stats.spearmanr(y, train_pred).statistic),
    }


def _fast_oof_metrics(packet: GenericFamilyPacket, params: dict[str, float]) -> dict[str, float]:
    y = packet.base.y
    kf = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    oof = np.zeros_like(y)
    fold_regrets: list[float] = []

    for train_idx, test_idx in kf.split(packet.base.w):
        model = build_penalty_calibration_surrogate(packet, params=params, variant_name=VARIANT_NAME).fit(
            packet.base.w[train_idx],
            y[train_idx],
        )
        pred = model.predict(packet.base.w[test_idx])
        oof[test_idx] = pred
        fold_regrets.append(float(y[test_idx][int(np.argmin(pred))] - np.min(y[test_idx])))

    residuals = oof - y
    cv_rmse = float(np.sqrt(np.mean(residuals**2)))
    tail_count = max(5, int(np.ceil(float(LOWER_TAIL_FRAC) * float(len(y)))))
    tail_idx = np.argsort(oof)[:tail_count]
    lower_tail_optimism = float(np.mean(np.maximum(y[tail_idx] - oof[tail_idx], 0.0)))
    mean_regret = float(np.mean(fold_regrets))
    objective = (
        CALIBRATION_CV_WEIGHT * cv_rmse
        + CALIBRATION_FOLDMEAN_WEIGHT * mean_regret
        + CALIBRATION_TAIL_WEIGHT * lower_tail_optimism
    )
    return {
        "cv_rmse": cv_rmse,
        "cv_regret_at_1": float(y[int(np.argmin(oof))] - np.min(y)),
        "cv_foldmean_regret_at_1": mean_regret,
        "lower_tail_optimism": lower_tail_optimism,
        "objective": float(objective),
    }


def _coarse_rows_fast(packet: GenericFamilyPacket, start_bank: tuple[dict[str, float], ...]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for start_id, params in enumerate(start_bank):
        rows.append(
            {
                "variant": "power_family_penalty_no_l2",
                "surrogate_variant": VARIANT_NAME,
                "stage": "coarse",
                "start_id": int(start_id),
                **params,
                **_fast_oof_metrics(packet, params),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["objective", "cv_rmse", "cv_foldmean_regret_at_1"],
        ascending=[True, True, True],
    )


def _refine_rows_fast(
    packet: GenericFamilyPacket,
    start_bank: tuple[dict[str, float], ...],
    *,
    coarse_top_k: int,
    method: str,
) -> tuple[pd.DataFrame, dict[str, float | bool | str], pd.DataFrame]:
    coarse_frame = _coarse_rows_fast(packet, start_bank)
    chosen_ids = coarse_frame["start_id"].head(int(coarse_top_k)).tolist()

    best_metrics: dict[str, float | bool | str] | None = None
    best_objective = float("inf")
    refine_rows: list[dict[str, float | bool | str | int]] = []

    for chosen_rank, start_id in enumerate(chosen_ids):
        start = _pack_no_l2_params(start_bank[start_id])
        cache: dict[tuple[float, ...], float] = {}

        def objective(z: np.ndarray, _cache: dict[tuple[float, ...], float] = cache) -> float:
            key = tuple(np.round(np.asarray(z, dtype=float), 8))
            if key not in _cache:
                params = _unpack_no_l2_params(z)
                _cache[key] = _fast_oof_metrics(packet, params)["objective"]
            return _cache[key]

        options = {
            "L-BFGS-B": {"maxiter": 120, "ftol": 1e-7},
            "Nelder-Mead": {"maxiter": 600, "xatol": 1e-4, "fatol": 1e-7},
            "Powell": {"maxiter": 80, "xtol": 1e-4, "ftol": 1e-7},
        }.get(method, {"maxiter": 160})
        result = minimize(objective, start, method=method, options=options)
        params = _unpack_no_l2_params(np.asarray(result.x, dtype=float))
        metrics = _fast_oof_metrics(packet, params)
        row = {
            "variant": "power_family_penalty_no_l2",
            "surrogate_variant": VARIANT_NAME,
            "stage": "refine",
            "chosen_rank": int(chosen_rank),
            "start_id": int(start_id),
            "success": bool(result.success),
            "message": str(result.message),
            **params,
            **metrics,
        }
        refine_rows.append(row)
        if float(row["objective"]) < best_objective:
            best_objective = float(row["objective"])
            best_metrics = row

    if best_metrics is None:
        raise RuntimeError("No-L2 metric retune failed to produce a best result")
    return coarse_frame, best_metrics, pd.DataFrame.from_records(refine_rows)


def _fit_objective(spec: ObjectiveSpec, *, method: str, coarse_top_k: int) -> tuple[dict[str, Any], dict[str, Any]]:
    packet, _fit_frame = _load_packet(spec)
    coarse_frame, best_metrics, refine_frame = _refine_rows_fast(
        packet,
        _start_bank(),
        coarse_top_k=coarse_top_k,
        method=method,
    )
    params = {key: float(best_metrics[key]) for key in _no_l2_param_keys()}
    params["reg"] = REG_FIXED

    model = build_penalty_calibration_surrogate(packet, params=params, variant_name=VARIANT_NAME).fit(
        packet.base.w,
        packet.base.y,
    )
    full_metrics = compute_penalty_calibration_metrics(packet, model, seed=CV_SEED)
    extra_quality = _prediction_quality(packet, params)
    raw_result, phase0, phase1 = optimize_penalty_calibration_model(packet, model, seed=CV_SEED)
    raw_weights = np.stack([phase0, phase1], axis=0)
    raw_distances = average_phase_tv_distance(packet.base.w, raw_weights[None, :, :])
    nearest_idx = int(np.argmin(raw_distances))
    weights_csv = OUTPUT_DIR / f"{spec.slug}_raw_optimum_weights.csv"
    train_pred = model.predict(packet.base.w)
    predicted_observed_idx = int(np.argmin(train_pred))
    best_observed_idx = int(np.argmin(packet.base.y))

    best_model_target = float(packet.base.y[best_observed_idx])
    raw_nearest_model_target = float(packet.base.y[nearest_idx])
    predicted_observed_model_target = float(packet.base.y[predicted_observed_idx])

    summary = {
        "slug": spec.slug,
        "metric_key": spec.metric_key,
        "display_name": spec.display_name,
        "lower_is_better": spec.lower_is_better,
        "n": len(packet.base.y),
        "method": method,
        "coarse_top_k": coarse_top_k,
        "best_observed_run_name": str(packet.base.frame.iloc[best_observed_idx][packet.base.name_col]),
        "best_observed_metric": _model_target_to_metric(best_model_target, spec),
        "predicted_observed_run_name": str(packet.base.frame.iloc[predicted_observed_idx][packet.base.name_col]),
        "predicted_observed_metric": _model_target_to_metric(predicted_observed_model_target, spec),
        "predicted_observed_regret": _regret(predicted_observed_model_target, best_model_target),
        "raw_predicted_optimum_metric": _model_target_to_metric(float(raw_result.fun), spec),
        "raw_nearest_observed_run_name": str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
        "raw_nearest_observed_metric": _model_target_to_metric(raw_nearest_model_target, spec),
        "raw_nearest_observed_regret": _regret(raw_nearest_model_target, best_model_target),
        "raw_nearest_observed_tv": float(raw_distances[nearest_idx]),
        "raw_optimum_weights_csv": str(weights_csv),
        "raw_phase0_lt_1e4": int(np.sum(phase0 < 1e-4)),
        "raw_phase1_lt_1e4": int(np.sum(phase1 < 1e-4)),
        **{
            key: float(value)
            for key, value in full_metrics.items()
            if isinstance(value, int | float | np.integer | np.floating)
        },
        **extra_quality,
    }
    summary["raw_predicted_optimum_value"] = _model_target_to_metric(
        float(full_metrics["raw_predicted_optimum_value"]), spec
    )
    summary["raw_nearest_observed_value"] = _model_target_to_metric(
        float(full_metrics["raw_nearest_observed_value"]), spec
    )

    param_row = {
        "slug": spec.slug,
        "metric_key": spec.metric_key,
        **params,
        **_parameter_counts(packet),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "domain_name": packet.base.domain_names,
            "phase0_weight": phase0,
            "phase0_epochs": phase0 * packet.base.c0,
            "phase1_weight": phase1,
            "phase1_epochs": phase1 * packet.base.c1,
        }
    ).to_csv(weights_csv, index=False)
    coarse_frame.to_csv(OUTPUT_DIR / f"{spec.slug}_coarse.csv", index=False)
    refine_frame.to_csv(OUTPUT_DIR / f"{spec.slug}_refine.csv", index=False)
    return summary, param_row


def _markdown(summary: pd.DataFrame) -> str:
    columns = [
        "display_name",
        "n",
        "train_rmse",
        "train_r2",
        "train_spearman",
        "oof_rmse",
        "oof_r2",
        "oof_spearman",
        "cv_foldmean_regret_at_1",
        "lower_tail_optimism",
        "best_observed_metric",
        "predicted_observed_metric",
        "predicted_observed_regret",
        "raw_predicted_optimum_metric",
        "raw_nearest_observed_metric",
        "raw_nearest_observed_regret",
        "raw_nearest_observed_tv",
    ]
    return summary[columns].to_markdown(index=False, floatfmt=".6f") + "\n"


def main() -> None:
    method = "Powell"
    coarse_top_k = 3
    rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    for spec in OBJECTIVES:
        summary, params = _fit_objective(spec, method=method, coarse_top_k=coarse_top_k)
        rows.append(summary)
        param_rows.append(params)
        print(f"fit {spec.slug}: oof_rmse={summary['oof_rmse']:.6f} oof_spearman={summary['oof_spearman']:.6f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame.from_records(rows)
    params_frame = pd.DataFrame.from_records(param_rows)
    summary_frame.to_csv(SUMMARY_CSV, index=False)
    params_frame.to_csv(PARAMS_CSV, index=False)
    SUMMARY_JSON.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    MARKDOWN_MD.write_text(_markdown(summary_frame))
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {PARAMS_CSV}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {MARKDOWN_MD}")


if __name__ == "__main__":
    main()
