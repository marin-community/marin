# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Benchmark plausible GRP tuning procedures against the published tuned model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GENERIC_FAMILY_NAMES,
    TUNED_GENERIC_FAMILY_PARAMS,
    GenericFamilyRetainedTotalSurrogate,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.two_phase_many_ccglobalpremium_baselines import (
    ccglobalpremium_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_ccpairtotal_baseline import (
    ccpairtotal_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    VALIDATED_GLOBAL_BPB,
    VALIDATED_PAIR_BPB,
    _evaluate_params,
    _pack_params,
    _summary_weights,
    _unpack_params,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_tuned_baseline import (
    GENERICFAMILY_TUNED_ANCHOR_COEFFICIENTS,
    genericfamily_tuned_summary,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = SCRIPT_DIR / "genericfamily_tuner_gap_benchmark.csv"
RESULTS_JSON = SCRIPT_DIR / "genericfamily_tuner_gap_benchmark.json"
REPEATED_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Procedure:
    name: str
    start_label: str
    method: str
    objective_name: str
    objective: str


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / ez.sum()


def _mean_phase_tv(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return 0.5 * float(np.mean(np.sum(np.abs(lhs - rhs), axis=1)))


def _convex_hull_deployment(
    packet,
    params: dict[str, float],
    valid_weights: np.ndarray,
    valid_y: np.ndarray,
) -> dict[str, object]:
    aug_w = np.concatenate([packet.base.w, valid_weights], axis=0)
    aug_y = np.concatenate([packet.base.y, valid_y], axis=0)
    deploy_model = GenericFamilyRetainedTotalSurrogate(
        packet,
        params=params,
        family_totals=GENERIC_FAMILY_NAMES,
        quality_discount=True,
    ).fit(aug_w, aug_y)

    frame = packet.base.frame
    best_idx = int(np.argmin(packet.base.y))
    best_obs = packet.base.w[best_idx]
    prop_idx = int(frame.index[frame["run_name"] == "baseline_proportional"][0])
    prop = packet.base.w[prop_idx]
    anchors = np.stack([best_obs, valid_weights[0], valid_weights[1], prop], axis=0)

    def objective(z: np.ndarray) -> float:
        coeffs = _softmax(z)
        ww = np.tensordot(coeffs, anchors, axes=1)[None, :, :]
        return float(deploy_model.predict(ww)[0])

    starts = [np.zeros(anchors.shape[0], dtype=float)] + [
        np.eye(anchors.shape[0], dtype=float)[i] * 4.0 for i in range(anchors.shape[0])
    ]
    best_result = None
    best_value = float("inf")
    for start in starts:
        result = minimize(objective, start, method="L-BFGS-B", options={"maxiter": 100})
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best_result = result

    if best_result is None:
        raise RuntimeError("Convex hull deployment optimization failed")

    coeffs = _softmax(np.asarray(best_result.x, dtype=float))
    weights = np.tensordot(coeffs, anchors, axes=1)
    return {
        "predicted_value": best_value,
        "coefficients": {
            "best_observed": float(coeffs[0]),
            "validated_global": float(coeffs[1]),
            "validated_pair": float(coeffs[2]),
            "baseline_proportional": float(coeffs[3]),
        },
        "weights": weights,
    }


def _objective_value(
    kind: str,
    single_metrics: dict[str, float | bool],
    repeated_rows: list[dict[str, float | bool]],
) -> float:
    if kind == "single_foldmean":
        return (
            float(single_metrics["cv_rmse"])
            + float(single_metrics["anchor_mae"])
            + 0.02 * float(single_metrics["cv_foldmean_regret_at_1"])
        )
    if kind == "single_cvregret":
        return (
            float(single_metrics["cv_rmse"])
            + float(single_metrics["anchor_mae"])
            + 0.2 * float(single_metrics["cv_regret_at_1"])
        )
    if kind == "single_both":
        return (
            float(single_metrics["cv_rmse"])
            + float(single_metrics["anchor_mae"])
            + 0.2 * float(single_metrics["cv_regret_at_1"])
            + 0.02 * float(single_metrics["cv_foldmean_regret_at_1"])
        )
    if kind == "repeated_both":
        mean_cv_rmse = float(np.mean([float(row["cv_rmse"]) for row in repeated_rows]))
        mean_cv_regret = float(np.mean([float(row["cv_regret_at_1"]) for row in repeated_rows]))
        mean_fold_regret = float(np.mean([float(row["cv_foldmean_regret_at_1"]) for row in repeated_rows]))
        return mean_cv_rmse + float(single_metrics["anchor_mae"]) + 0.2 * mean_cv_regret + 0.02 * mean_fold_regret
    raise ValueError(f"Unknown objective kind: {kind}")


def _evaluate_metrics(
    z: np.ndarray,
    packet,
    valid_weights: np.ndarray,
    valid_y: np.ndarray,
    objective_kind: str,
) -> tuple[float, dict[str, object]]:
    single = _evaluate_params(z, packet, valid_weights, valid_y, seed=0)
    repeated = [_evaluate_params(z, packet, valid_weights, valid_y, seed=seed) for seed in REPEATED_SEEDS]
    objective = _objective_value(objective_kind, single, repeated)
    params = _unpack_params(np.asarray(z, dtype=float))
    return objective, {"single": single, "repeated": repeated, "params": params}


def main() -> None:
    packet = load_generic_family_packet()
    valid_weights = np.stack(
        [
            _summary_weights(ccglobalpremium_retainedtotal_summary(), packet.base.domain_names),
            _summary_weights(ccpairtotal_retainedtotal_summary(), packet.base.domain_names),
        ],
        axis=0,
    )
    valid_y = np.asarray([VALIDATED_GLOBAL_BPB, VALIDATED_PAIR_BPB], dtype=float)

    target_summary = genericfamily_tuned_summary()
    target_weights = _summary_weights(target_summary, packet.base.domain_names)
    target_coeffs = dict(GENERICFAMILY_TUNED_ANCHOR_COEFFICIENTS)
    target_z = _pack_params(TUNED_GENERIC_FAMILY_PARAMS)

    starts = {
        "tuned": _pack_params(TUNED_GENERIC_FAMILY_PARAMS),
        "broad_beta": _pack_params(
            {
                "alpha": 11.533461482593735,
                "eta": 10.859113730214359,
                "lam": 0.3422735488822989,
                "tau": 2.843180828656475,
                "reg": 0.0001896587113845684,
                "beta": 0.9324427249160729,
            }
        ),
        "neutral": np.array([0.0, 0.0, -2.0, 2.0, -8.0, 0.0], dtype=float),
    }

    procedures = [
        Procedure("current_from_tuned_lbfgsb", "tuned", "L-BFGS-B", "single_foldmean", "current"),
        Procedure("current_from_broad_beta_lbfgsb", "broad_beta", "L-BFGS-B", "single_foldmean", "current"),
        Procedure("cvregret_from_broad_beta_lbfgsb", "broad_beta", "L-BFGS-B", "single_cvregret", "cvregret"),
        Procedure("cvregret_from_broad_beta_powell", "broad_beta", "Powell", "single_cvregret", "cvregret"),
        Procedure("both_from_broad_beta_powell", "broad_beta", "Powell", "single_both", "both"),
        Procedure("repeated_both_from_broad_beta_powell", "broad_beta", "Powell", "repeated_both", "repeated_both"),
        Procedure("current_from_neutral_powell", "neutral", "Powell", "single_foldmean", "current"),
        Procedure("both_from_neutral_powell", "neutral", "Powell", "single_both", "both"),
    ]

    rows: list[dict[str, object]] = []
    for procedure in procedures:
        start = starts[procedure.start_label]
        objective_name = procedure.objective_name

        def objective(z: np.ndarray, objective_name: str = objective_name) -> float:
            value, _ = _evaluate_metrics(z, packet, valid_weights, valid_y, objective_name)
            return value

        t0 = perf_counter()
        result = minimize(
            objective,
            start,
            method=procedure.method,
            options={
                "L-BFGS-B": {"maxiter": 250, "ftol": 1e-6},
                "Powell": {"maxiter": 300, "xtol": 1e-4, "ftol": 1e-6},
            }.get(procedure.method, {"maxiter": 250}),
        )
        elapsed = float(perf_counter() - t0)
        objective_value, payload = _evaluate_metrics(
            np.asarray(result.x, dtype=float),
            packet,
            valid_weights,
            valid_y,
            procedure.objective_name,
        )
        params = payload["params"]
        deployment = _convex_hull_deployment(packet, params, valid_weights, valid_y)
        repeated = payload["repeated"]
        coeffs = deployment["coefficients"]
        deploy_weights = deployment["weights"]

        row = {
            "procedure": procedure.name,
            "start_label": procedure.start_label,
            "method": procedure.method,
            "objective_name": procedure.objective_name,
            "success": bool(result.success),
            "message": str(result.message),
            "nit": int(getattr(result, "nit", -1)),
            "nfev": int(getattr(result, "nfev", -1)),
            "duration": elapsed,
            "objective": float(objective_value),
            "alpha": float(params["alpha"]),
            "eta": float(params["eta"]),
            "lam": float(params["lam"]),
            "tau": float(params["tau"]),
            "reg": float(params["reg"]),
            "beta": float(params["beta"]),
            "param_z_l2_to_target": float(np.linalg.norm(_pack_params(params) - target_z)),
            "single_cv_rmse": float(payload["single"]["cv_rmse"]),
            "single_cv_r2": float(payload["single"]["cv_r2"]),
            "single_cv_regret_at_1": float(payload["single"]["cv_regret_at_1"]),
            "single_cv_foldmean_regret_at_1": float(payload["single"]["cv_foldmean_regret_at_1"]),
            "anchor_mae": float(payload["single"]["anchor_mae"]),
            "repeated_cv_rmse_mean": float(np.mean([float(r["cv_rmse"]) for r in repeated])),
            "repeated_cv_regret_mean": float(np.mean([float(r["cv_regret_at_1"]) for r in repeated])),
            "repeated_foldmean_regret_mean": float(np.mean([float(r["cv_foldmean_regret_at_1"]) for r in repeated])),
            "deploy_predicted_value": float(deployment["predicted_value"]),
            "deploy_tv_to_target": _mean_phase_tv(deploy_weights, target_weights),
            "deploy_coeff_l1_to_target": float(
                sum(abs(float(coeffs[k]) - float(target_coeffs[k])) for k in target_coeffs)
            ),
            "deploy_best_observed_coeff": float(coeffs["best_observed"]),
            "deploy_validated_global_coeff": float(coeffs["validated_global"]),
            "deploy_validated_pair_coeff": float(coeffs["validated_pair"]),
            "deploy_baseline_proportional_coeff": float(coeffs["baseline_proportional"]),
        }
        rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        ["deploy_tv_to_target", "deploy_coeff_l1_to_target", "param_z_l2_to_target"], ascending=True
    )
    RESULTS_CSV.write_text(results.to_csv(index=False))
    RESULTS_JSON.write_text(
        json.dumps(
            {
                "target_params": TUNED_GENERIC_FAMILY_PARAMS,
                "target_anchor_coefficients": target_coeffs,
                "target_predicted_value": float(target_summary["predicted_optimum_value"]),
                "best_by_deploy_tv": results.iloc[0].to_dict(),
                "best_by_param_distance": results.sort_values("param_z_l2_to_target").iloc[0].to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Results CSV: {RESULTS_CSV}")
    print(f"Results JSON: {RESULTS_JSON}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
