# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Compare single-head GRP no-L2 against uncheatable-component aggregation.

The component variant fits one independent GRP no-L2 model per
``eval/uncheatable_eval/*/bpb`` subdomain on the 60M fit swarm, then aggregates
their predicted sub-losses back to overall ``eval/uncheatable_eval/bpb`` using a
fixed non-negative decomposition learned from the observed metrics.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_grp_power_family_penalty_no_l2_retune import (
    BEST_CSV,
    CV_SEED,
    REG_FIXED,
    VARIANT_NAME,
    _no_l2_param_keys,
    _parameter_counts,
    _start_bank,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.fit_grp_no_l2_metric_objectives import (
    _refine_rows_fast,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GenericFamilyPacket,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    build_penalty_calibration_surrogate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    PacketData,
    regression_metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = SCRIPT_DIR / "grp_power_family_penalty_no_l2_uncheatable_components.csv"
OUTPUT_JSON = SCRIPT_DIR / "grp_power_family_penalty_no_l2_uncheatable_components_summary.json"
OUTPUT_MD = SCRIPT_DIR / "grp_power_family_penalty_no_l2_uncheatable_components.md"
OUTPUT_COMPONENT_PARAMS_CSV = SCRIPT_DIR / "grp_power_family_penalty_no_l2_uncheatable_component_params.csv"
OUTPUT_COMPONENT_PREDICTIONS_CSV = SCRIPT_DIR / "grp_power_family_penalty_no_l2_uncheatable_component_predictions.csv"

COMPONENT_TARGETS = (
    "eval/uncheatable_eval/ao3_english/bpb",
    "eval/uncheatable_eval/arxiv_computer_science/bpb",
    "eval/uncheatable_eval/arxiv_physics/bpb",
    "eval/uncheatable_eval/bbc_news/bpb",
    "eval/uncheatable_eval/github_cpp/bpb",
    "eval/uncheatable_eval/github_python/bpb",
    "eval/uncheatable_eval/wikipedia_english/bpb",
)


def _subset_packet(packet: GenericFamilyPacket) -> GenericFamilyPacket:
    required = [MANY_DOMAIN_TARGET, *COMPONENT_TARGETS]
    mask = packet.base.frame.loc[:, required].notna().all(axis=1).to_numpy()
    frame = packet.base.frame.loc[mask].reset_index(drop=True)
    base = PacketData(
        frame=frame,
        name_col=packet.base.name_col,
        y=frame[MANY_DOMAIN_TARGET].to_numpy(float),
        w=packet.base.w[mask],
        m=packet.base.m,
        c0=packet.base.c0,
        c1=packet.base.c1,
        domain_names=packet.base.domain_names,
    )
    return replace(packet, base=base)


def _packet_with_target(packet: GenericFamilyPacket, metric_key: str) -> GenericFamilyPacket:
    y = packet.base.frame[metric_key].to_numpy(float)
    base = PacketData(
        frame=packet.base.frame,
        name_col=packet.base.name_col,
        y=y,
        w=packet.base.w,
        m=packet.base.m,
        c0=packet.base.c0,
        c1=packet.base.c1,
        domain_names=packet.base.domain_names,
    )
    return replace(packet, base=base)


def _foldmean_regret(y_true: np.ndarray, y_pred: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]) -> float:
    regrets: list[float] = []
    for _tr, te in splits:
        chosen = int(np.argmin(y_pred[te]))
        regrets.append(float(y_true[te][chosen] - np.min(y_true[te])))
    return float(np.mean(regrets))


def _lower_tail_optimism(y_true: np.ndarray, y_pred: np.ndarray, frac: float = 0.15) -> float:
    tail_count = max(5, int(np.ceil(frac * len(y_true))))
    idx = np.argsort(y_pred)[:tail_count]
    return float(np.mean(np.maximum(y_true[idx] - y_pred[idx], 0.0)))


def _load_overall_no_l2_params() -> dict[str, float]:
    row = pd.read_csv(BEST_CSV).iloc[0].to_dict()
    params = {key: float(row[key]) for key in _no_l2_param_keys()}
    params["reg"] = REG_FIXED
    return params


def _fit_single_head_predictions(
    packet: GenericFamilyPacket,
    params: dict[str, float],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    model = build_penalty_calibration_surrogate(packet, params=params, variant_name=VARIANT_NAME).fit(
        packet.base.w,
        packet.base.y,
    )
    train_pred = model.predict(packet.base.w)
    oof = np.zeros_like(packet.base.y)
    for tr, te in splits:
        fold_model = build_penalty_calibration_surrogate(packet, params=params, variant_name=VARIANT_NAME).fit(
            packet.base.w[tr],
            packet.base.y[tr],
        )
        oof[te] = fold_model.predict(packet.base.w[te])
    return train_pred, oof


def _component_aggregation_weights(frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    x = frame.loc[:, COMPONENT_TARGETS].to_numpy(float)
    y = frame[MANY_DOMAIN_TARGET].to_numpy(float)
    weights, _ = nnls(x, y)
    intercept = float(np.mean(y - x @ weights))
    return weights, intercept


def _fit_component_models(
    packet: GenericFamilyPacket,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    component_rows: list[dict[str, Any]] = []
    train_components: list[np.ndarray] = []
    oof_components: list[np.ndarray] = []
    for metric_key in COMPONENT_TARGETS:
        component_packet = _packet_with_target(packet, metric_key)
        coarse_frame, best_metrics, refine_frame = _refine_rows_fast(
            component_packet,
            _start_bank(),
            coarse_top_k=3,
            method="Powell",
        )
        del coarse_frame, refine_frame
        params = {key: float(best_metrics[key]) for key in _no_l2_param_keys()}
        params["reg"] = REG_FIXED
        train_pred, oof_pred = _fit_single_head_predictions(component_packet, params, splits)
        component_rows.append(
            {
                "metric_key": metric_key,
                **params,
                "objective": float(best_metrics["objective"]),
                "cv_rmse_component": float(best_metrics["cv_rmse"]),
                "cv_foldmean_regret_component": float(best_metrics["cv_foldmean_regret_at_1"]),
                "lower_tail_optimism_component": float(best_metrics["lower_tail_optimism"]),
            }
        )
        train_components.append(train_pred)
        oof_components.append(oof_pred)
    return (
        pd.DataFrame.from_records(component_rows),
        np.column_stack(train_components),
        np.column_stack(oof_components),
    )


def _summary_row(
    *,
    label: str,
    packet: GenericFamilyPacket,
    train_pred: np.ndarray,
    oof_pred: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    total_param_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    y = packet.base.y
    train = regression_metrics(packet.base.frame, packet.base.name_col, y, train_pred)
    cv = regression_metrics(packet.base.frame, packet.base.name_col, y, oof_pred)
    row = {
        "label": label,
        "n": len(y),
        "total_param_count": int(total_param_count),
        "train_rmse": float(train["rmse"]),
        "train_r2": float(train["r2"]),
        "train_spearman": float(train["spearman"]),
        "train_regret_at_1": float(train["regret_at_1"]),
        "cv_rmse": float(cv["rmse"]),
        "cv_r2": float(cv["r2"]),
        "cv_spearman": float(cv["spearman"]),
        "cv_regret_at_1": float(cv["regret_at_1"]),
        "cv_foldmean_regret_at_1": _foldmean_regret(y, oof_pred, splits),
        "lower_tail_optimism": _lower_tail_optimism(y, oof_pred),
        "chosen_candidate": str(cv["chosen_candidate"]),
        "best_candidate": str(cv["best_candidate"]),
        "chosen_value": float(cv["chosen_value"]),
        "best_value": float(cv["best_value"]),
    }
    if extra is not None:
        row.update(extra)
    return row


def _markdown(summary: pd.DataFrame) -> str:
    columns = [
        "label",
        "total_param_count",
        "train_rmse",
        "cv_rmse",
        "cv_spearman",
        "cv_foldmean_regret_at_1",
        "lower_tail_optimism",
        "chosen_candidate",
        "chosen_value",
    ]
    return summary.loc[:, columns].to_markdown(index=False, floatfmt=".6f") + "\n"


def main() -> None:
    packet = _subset_packet(load_generic_family_packet(target=MANY_DOMAIN_TARGET))
    splits = list(KFold(n_splits=5, shuffle=True, random_state=CV_SEED).split(packet.base.w))

    agg_weights, agg_intercept = _component_aggregation_weights(packet.base.frame)
    actual_component_matrix = packet.base.frame.loc[:, COMPONENT_TARGETS].to_numpy(float)
    agg_reconstruction = agg_intercept + actual_component_matrix @ agg_weights

    baseline_params = _load_overall_no_l2_params()
    baseline_train_pred, baseline_oof_pred = _fit_single_head_predictions(packet, baseline_params, splits)
    baseline_counts = _parameter_counts(packet)
    baseline_summary = _summary_row(
        label="single_head_no_l2",
        packet=packet,
        train_pred=baseline_train_pred,
        oof_pred=baseline_oof_pred,
        splits=splits,
        total_param_count=int(baseline_counts["total_param_count"]),
        extra={"variant": "power_family_penalty_no_l2"},
    )

    component_params, component_train_pred_matrix, component_oof_pred_matrix = _fit_component_models(packet, splits)
    component_train_pred = agg_intercept + component_train_pred_matrix @ agg_weights
    component_oof_pred = agg_intercept + component_oof_pred_matrix @ agg_weights
    per_model_param_count = int(_parameter_counts(packet)["total_param_count"])
    component_total_param_count = len(COMPONENT_TARGETS) * per_model_param_count + len(COMPONENT_TARGETS) + 1
    component_summary = _summary_row(
        label="component_aggregate_no_l2",
        packet=packet,
        train_pred=component_train_pred,
        oof_pred=component_oof_pred,
        splits=splits,
        total_param_count=component_total_param_count,
        extra={
            "variant": "power_family_penalty_no_l2_uncheatable_components",
            "aggregation_intercept": agg_intercept,
            "aggregation_reconstruction_rmse": float(np.sqrt(np.mean((agg_reconstruction - packet.base.y) ** 2))),
        },
    )
    for metric_key, weight in zip(COMPONENT_TARGETS, agg_weights, strict=True):
        component_summary[f"agg_weight::{metric_key}"] = float(weight)

    summary_frame = pd.DataFrame.from_records([baseline_summary, component_summary])
    prediction_frame = packet.base.frame.loc[:, [packet.base.name_col, MANY_DOMAIN_TARGET, *COMPONENT_TARGETS]].copy()
    prediction_frame["pred::single_head_no_l2"] = baseline_train_pred
    prediction_frame["oof::single_head_no_l2"] = baseline_oof_pred
    prediction_frame["pred::component_aggregate_no_l2"] = component_train_pred
    prediction_frame["oof::component_aggregate_no_l2"] = component_oof_pred
    for idx, metric_key in enumerate(COMPONENT_TARGETS):
        slug = metric_key.removeprefix("eval/uncheatable_eval/").removesuffix("/bpb")
        prediction_frame[f"pred_component::{slug}"] = component_train_pred_matrix[:, idx]
        prediction_frame[f"oof_component::{slug}"] = component_oof_pred_matrix[:, idx]

    OUTPUT_CSV.write_text(summary_frame.to_csv(index=False))
    OUTPUT_JSON.write_text(json.dumps(summary_frame.to_dict(orient="records"), indent=2, sort_keys=True) + "\n")
    OUTPUT_MD.write_text(_markdown(summary_frame))
    component_params.to_csv(OUTPUT_COMPONENT_PARAMS_CSV, index=False)
    prediction_frame.to_csv(OUTPUT_COMPONENT_PREDICTIONS_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_COMPONENT_PARAMS_CSV}")
    print(f"Wrote {OUTPUT_COMPONENT_PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
