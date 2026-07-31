# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Probe a per-domain-exponent GRP variant starting from power_family_penalty."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    load_generic_family_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    build_penalty_calibration_surrogate,
    compute_penalty_calibration_metrics,
    deploy_penalty_calibration_gaincapped_topkactual,
    domain_exponent_key,
    penalty_calibration_variant_parameter_counts,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_STATE_PATH = SCRIPT_DIR / "chatgpt_pro_grp_recovery_packet" / "data" / "current_reference_state.json"
BEST_CSV = SCRIPT_DIR / "grp_penalty_calibration_variants_best.csv"
DEPLOY_CSV = SCRIPT_DIR / "grp_penalty_calibration_variants_deployments.csv"
OUT_JSON = SCRIPT_DIR / "grp_power_domain_penalty_probe_summary.json"
OUT_MD = SCRIPT_DIR / "grp_power_domain_penalty_probe_table.md"

SOURCE_VARIANT = "power_family_penalty"
TARGET_VARIANT = "power_domain_penalty"


def _validated_anchor_arrays() -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(REFERENCE_STATE_PATH.read_text())
    weights = np.stack(
        [
            np.asarray(payload["validated_global"]["phase_weights"], dtype=float),
            np.asarray(payload["validated_pair"]["phase_weights"], dtype=float),
        ],
        axis=0,
    )
    targets = np.asarray([payload["validated_global_bpb"], payload["validated_pair_bpb"]], dtype=float)
    return weights, targets


def _load_best_row(variant_name: str) -> dict[str, Any]:
    frame = pd.read_csv(BEST_CSV)
    subset = frame.loc[frame["variant"] == variant_name]
    if subset.empty:
        raise ValueError(f"Missing variant {variant_name!r} in {BEST_CSV}")
    return subset.sort_values("objective", ascending=True).iloc[0].to_dict()


def _load_deploy_row(variant_name: str) -> dict[str, Any]:
    frame = pd.read_csv(DEPLOY_CSV)
    subset = frame.loc[frame["variant"] == variant_name]
    if subset.empty:
        raise ValueError(f"Missing deployment row for {variant_name!r} in {DEPLOY_CSV}")
    return subset.sort_values("predicted_optimum_value", ascending=True).iloc[0].to_dict()


def _family_best_params(best_row: dict[str, Any]) -> dict[str, float]:
    return {
        "eta": float(best_row["eta"]),
        "lam": float(best_row["lam"]),
        "reg": float(best_row["reg"]),
        "beta": float(best_row["beta"]),
        "tau_broad_text": float(best_row["tau_broad_text"]),
        "tau_tech_code": float(best_row["tau_tech_code"]),
        "tau_reasoning": float(best_row["tau_reasoning"]),
    }


def _lift_family_exponents(packet, best_row: dict[str, Any]) -> dict[str, float]:
    params = _family_best_params(best_row)
    for domain_idx in range(packet.base.m):
        family_name = next(family_name for family_name, members in packet.family_map.items() if domain_idx in members)
        params[domain_exponent_key(domain_idx)] = float(best_row[f"a_{family_name}"])
    return params


def _train_rmse(packet, params: dict[str, float]) -> float:
    model = build_penalty_calibration_surrogate(packet, params=params, variant_name=TARGET_VARIANT).fit(
        packet.base.w,
        packet.base.y,
    )
    pred = model.predict(packet.base.w)
    return float(np.sqrt(np.mean((pred - packet.base.y) ** 2)))


def _tune_domain_exponents(packet, init_params: dict[str, float]) -> dict[str, float]:
    exponent_keys = [domain_exponent_key(domain_idx) for domain_idx in range(packet.base.m)]
    z0 = np.log(np.asarray([float(init_params[key]) for key in exponent_keys], dtype=float))

    def objective(z: np.ndarray) -> float:
        params = dict(init_params)
        clipped = np.exp(np.clip(np.asarray(z, dtype=float), np.log(0.02), np.log(2.0)))
        for key, value in zip(exponent_keys, clipped, strict=True):
            params[key] = float(value)
        return _train_rmse(packet, params)

    result = minimize(
        objective,
        z0,
        method="L-BFGS-B",
        options={"maxiter": 25, "ftol": 1e-6},
    )
    tuned = dict(init_params)
    clipped = np.exp(np.clip(np.asarray(result.x, dtype=float), np.log(0.02), np.log(2.0)))
    for key, value in zip(exponent_keys, clipped, strict=True):
        tuned[key] = float(value)
    tuned["optimizer_success"] = bool(result.success)
    tuned["optimizer_message"] = str(result.message)
    tuned["optimizer_fun"] = float(result.fun)
    return tuned


def _domain_exponent_summary(packet, params: dict[str, float]) -> dict[str, Any]:
    values = np.asarray(
        [float(params[domain_exponent_key(domain_idx)]) for domain_idx in range(packet.base.m)],
        dtype=float,
    )
    rows: list[dict[str, Any]] = []
    for domain_idx, domain_name in enumerate(packet.base.domain_names):
        family_name = next(family_name for family_name, members in packet.family_map.items() if domain_idx in members)
        rows.append(
            {
                "domain_idx": int(domain_idx),
                "domain_name": str(domain_name),
                "family": family_name,
                "a_domain": float(params[domain_exponent_key(domain_idx)]),
            }
        )
    return {
        "min_a": float(np.min(values)),
        "max_a": float(np.max(values)),
        "mean_a": float(np.mean(values)),
        "std_a": float(np.std(values)),
        "domains": rows,
    }


def main() -> None:
    packet = load_generic_family_packet()
    valid_weights, valid_y = _validated_anchor_arrays()
    family_best = _load_best_row(SOURCE_VARIANT)
    family_deploy = _load_deploy_row(SOURCE_VARIANT)

    init_params = _lift_family_exponents(packet, family_best)
    tuned_params = _tune_domain_exponents(packet, init_params)
    model = build_penalty_calibration_surrogate(packet, params=tuned_params, variant_name=TARGET_VARIANT).fit(
        packet.base.w,
        packet.base.y,
    )
    tuned_metrics = compute_penalty_calibration_metrics(
        packet,
        model,
        seed=0,
        valid_weights=valid_weights,
        valid_y=valid_y,
    )
    tuned_deploy = deploy_penalty_calibration_gaincapped_topkactual(packet, model, tuned_metrics)

    tuned_counts = penalty_calibration_variant_parameter_counts(packet, TARGET_VARIANT)
    family_counts = penalty_calibration_variant_parameter_counts(packet, SOURCE_VARIANT)

    summary = {
        "source_variant": SOURCE_VARIANT,
        "target_variant": TARGET_VARIANT,
        "family_counts": family_counts,
        "domain_counts": tuned_counts,
        "family_best": family_best,
        "family_deploy": family_deploy,
        "domain_tuned_metrics": tuned_metrics,
        "domain_tuned_deploy": {
            "predicted_optimum_value": float(tuned_deploy["predicted_optimum_value"]),
            "raw_predicted_optimum_value": float(tuned_deploy["raw_predicted_optimum_value"]),
            "gain_budget": float(tuned_deploy["gain_budget"]),
            "delta": float(tuned_deploy["delta"]),
        },
        "domain_tuned_params": tuned_params,
        "domain_exponent_summary": _domain_exponent_summary(packet, tuned_params),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    table = pd.DataFrame(
        [
            {
                "variant": SOURCE_VARIANT,
                "nonlinear_params": int(family_counts["nonlinear_param_count"]),
                "total_params": int(family_counts["total_param_count"]),
                "train_rmse": float(family_best["train_rmse"]),
                "cv_rmse": float(family_best["cv_rmse"]),
                "tail_opt": float(family_best["lower_tail_optimism"]),
                "cv_depopt8": float(family_best["cv_depopt_best8"]),
                "cv_raw_tv": float(family_best["cv_rawopt_nearest_tv"]),
                "anchor_mae": float(family_best["anchor_mae"]),
                "deploy_bpb": float(family_deploy["predicted_optimum_value"]),
            },
            {
                "variant": TARGET_VARIANT,
                "nonlinear_params": int(tuned_counts["nonlinear_param_count"]),
                "total_params": int(tuned_counts["total_param_count"]),
                "train_rmse": float(tuned_metrics["train_rmse"]),
                "cv_rmse": float(tuned_metrics["cv_rmse"]),
                "tail_opt": float(tuned_metrics["lower_tail_optimism"]),
                "cv_depopt8": float(tuned_metrics["cv_depopt_best8"]),
                "cv_raw_tv": float(tuned_metrics["cv_rawopt_nearest_tv"]),
                "anchor_mae": float(tuned_metrics["anchor_mae"]),
                "deploy_bpb": float(tuned_deploy["predicted_optimum_value"]),
            },
        ]
    )
    OUT_MD.write_text(table.to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
