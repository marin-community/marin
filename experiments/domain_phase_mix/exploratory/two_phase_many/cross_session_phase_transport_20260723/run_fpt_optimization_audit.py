# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.14",
#   "tabulate>=0.9",
# ]
# ///

"""Audit raw tied and two-phase optima of the two-parameter SFOS-FPT model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_phase_transport_synthesis import (
    TARGETS,
    _full_fit_ridge,
    fit_phase,
    fit_sfos,
    load_panel,
    phase_design,
    predict_phase,
    predict_sfos,
)
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "reference_outputs" / "cross_session_phase_transport_20260723"
MODEL_ID = "fpt_total_global"
EPS = 1e-12


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(np.clip(shifted, -60.0, 60.0))
    return values / values.sum()


def unpack(logits: np.ndarray, bucket_count: int, tied: bool) -> np.ndarray:
    if tied:
        weights = softmax(logits)
        return np.stack([weights, weights])
    return np.stack(
        [
            softmax(logits[:bucket_count]),
            softmax(logits[bucket_count:]),
        ]
    )


def logits_for(weights: np.ndarray, tied: bool) -> np.ndarray:
    if tied:
        return np.log(np.maximum(weights[0], 1e-12))
    return np.concatenate(
        [
            np.log(np.maximum(weights[0], 1e-12)),
            np.log(np.maximum(weights[1], 1e-12)),
        ]
    )


def kl(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sum(np.maximum(left, EPS) * np.log(np.maximum(left, EPS) / np.maximum(right, EPS))))


def standardized_nearest_distance(
    fit_weights: np.ndarray,
    query: np.ndarray,
) -> float:
    flattened = fit_weights.reshape(len(fit_weights), -1)
    query_flat = query.reshape(1, -1)
    scale = np.std(flattened, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    distance = np.linalg.norm((flattened - query_flat) / scale[None, :], axis=1)
    return float(np.min(distance) / np.sqrt(flattened.shape[1]))


def optimize_target(target_name: str) -> tuple[list[dict[str, object]], pd.DataFrame]:
    panel = load_panel()
    aggregate_fit = fit_sfos(
        panel,
        panel.one_weights[:, 0],
        panel.one_targets[target_name],
    )
    ridge = _full_fit_ridge(panel, MODEL_ID, target_name)
    phase_train, names, constrained = phase_design(
        panel,
        aggregate_fit,
        panel.two_weights,
        MODEL_ID,
    )
    phase_fit = fit_phase(
        panel,
        MODEL_ID,
        phase_train,
        panel.two_targets[target_name] - panel.one_targets[target_name],
        constrained,
        ridge,
        names,
    )

    def value(weights: np.ndarray) -> float:
        aggregate = panel.alpha0 * weights[0] + panel.alpha1 * weights[1]
        aggregate_value = predict_sfos(panel, aggregate_fit, aggregate[None])[0]
        phase_features, _, _ = phase_design(
            panel,
            aggregate_fit,
            weights[None],
            MODEL_ID,
        )
        return float(aggregate_value + predict_phase(phase_fit, phase_features)[0])

    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    for tied in (True, False):
        fit_weights = panel.one_weights if tied else panel.two_weights
        fit_predictions = np.asarray([value(weights) for weights in fit_weights])
        starts = [fit_weights[index] for index in np.argsort(fit_predictions)[:10]]
        starts.append(np.stack([panel.proportional, panel.proportional]))
        records: list[tuple[float, object, np.ndarray]] = []
        for start_index, start in enumerate(starts):
            initial = logits_for(start, tied)

            def objective(
                logits: np.ndarray,
                tied_policy: bool = tied,
            ) -> float:
                weights = unpack(
                    logits,
                    panel.one_weights.shape[2],
                    tied_policy,
                )
                return value(weights)

            result = minimize(
                objective,
                initial,
                method="L-BFGS-B",
                options={
                    "maxiter": 800,
                    "ftol": 1e-13,
                    "gtol": 1e-8,
                    "maxls": 50,
                },
            )
            weights = unpack(result.x, panel.one_weights.shape[2], tied)
            records.append((float(result.fun), result, weights))
            rows.append(
                {
                    "target": target_name,
                    "policy_class": "tied" if tied else "two_phase",
                    "start_index": start_index,
                    "objective": float(result.fun),
                    "success": bool(result.success),
                    "iterations": int(result.nit),
                    "message": str(result.message),
                }
            )
        objective, result, weights = min(records, key=lambda record: record[0])
        aggregate = panel.alpha0 * weights[0] + panel.alpha1 * weights[1]
        phase_information = panel.alpha0 * kl(weights[0], aggregate) + panel.alpha1 * kl(weights[1], aggregate)
        nearest = standardized_nearest_distance(fit_weights, weights)
        phase_epochs = np.stack([weights[0] * panel.c0, weights[1] * panel.c1])
        summary = {
            "target": target_name,
            "policy_class": "tied" if tied else "two_phase",
            "predicted_bpb": objective,
            "success": bool(result.success),
            "iterations": int(result.nit),
            "max_phase_weight": float(np.max(weights)),
            "max_phase_epochs": float(np.max(phase_epochs)),
            "phase_tv": float(0.5 * np.sum(np.abs(weights[1] - weights[0]))),
            "phase_information": phase_information,
            "aggregate_kl_to_proportional": kl(aggregate, panel.proportional),
            "standardized_nearest_fit_distance": nearest,
            "phase_ridge": ridge,
            "odd_coefficient": float(phase_fit.coefficients[0]),
            "jensen_coefficient": float(phase_fit.coefficients[1]),
            "implied_recency_share": float(panel.alpha0 + phase_fit.coefficients[0] * panel.alpha0 * panel.alpha1),
        }
        rows.append({**summary, "start_index": "selected"})
        for phase in range(2):
            for domain, weight in zip(panel.domains, weights[phase], strict=True):
                weight_rows.append(
                    {
                        **summary,
                        "phase": phase,
                        "domain": domain,
                        "weight": weight,
                    }
                )
    return rows, pd.DataFrame(weight_rows)


def main() -> None:
    rows: list[dict[str, object]] = []
    weight_frames: list[pd.DataFrame] = []
    for target_name in TARGETS:
        target_rows, target_weights = optimize_target(target_name)
        rows.extend(target_rows)
        weight_frames.append(target_weights)
    audit = pd.DataFrame(rows)
    weights = pd.concat(weight_frames, ignore_index=True)
    audit.to_csv(OUTPUT / "raw_optimization_audit.csv", index=False)
    weights.to_csv(OUTPUT / "raw_optimum_weights.csv", index=False)
    selected = audit.loc[audit["start_index"].astype(str) == "selected"]
    (OUTPUT / "raw_optimization_summary.json").write_text(
        json.dumps(selected.to_dict(orient="records"), indent=2, sort_keys=True) + "\n"
    )
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
