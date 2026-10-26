# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "threadpoolctl"]
# ///
"""Uncapped numerical policy search for the synthetic timing diagnostic.

This generalizes round four's fixed SLSQP search to one synthetic response.
Every feasible starting point and result remains eligible; failed solves are
retained in the trace. The returned values are numerical incumbents, not bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits


class Predictor(Protocol):
    def predict(self, early: np.ndarray, late: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class PolicyGeometry:
    phase0: np.ndarray
    phase1: np.ndarray
    scales: np.ndarray
    pool_tokens: np.ndarray
    alpha: float


def optimize_fit(
    fit: Predictor,
    geometry: PolicyGeometry,
    *,
    seed: int,
    phase_blind: bool,
) -> dict[str, Any]:
    """Search both simplexes using the frozen five-start round-four protocol."""
    alpha, scales = geometry.alpha, geometry.scales
    width = len(scales)
    natural = geometry.pool_tokens / geometry.pool_tokens.sum()
    candidates: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    def predict(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return fit.predict(alpha * scales * first, (1.0 - alpha) * scales * second)

    def consider(first: np.ndarray, second: np.ndarray, tied: bool, source: str, success: bool) -> None:
        residual = max(abs(first.sum() - 1.0), abs(second.sum() - 1.0))
        feasible = residual < 1e-6 and min(first.min(), second.min()) >= -1e-10
        value = float(predict(first[None], second[None])[0])
        record = {
            "source": source,
            "tied": tied,
            "predicted": value,
            "feasible": bool(feasible),
            "success": success,
            "simplex_residual": float(residual),
            "phase0": first.tolist(),
            "phase1": second.tolist(),
        }
        traces.append(record)
        if feasible and np.isfinite(value):
            candidates.append(record)

    def execute(start: np.ndarray, tied: bool, source: str) -> None:
        consider(start[:width], start[:width] if tied else start[width:], tied, source + ":start", False)
        constraints = [{"type": "eq", "fun": lambda x: x[:width].sum() - 1.0}]
        if not tied:
            constraints.append({"type": "eq", "fun": lambda x: x[width:].sum() - 1.0})
        result = minimize(
            lambda x: float(predict(x[None, :width], x[None, :width] if tied else x[None, width:])[0]),
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(start),
            constraints=constraints,
            options={"maxiter": 600, "ftol": 1e-10},
        )
        consider(result.x[:width], result.x[:width] if tied else result.x[width:], tied, source, bool(result.success))
        traces[-1].update(iterations=int(result.nit), message=str(result.message))

    ranked = np.argsort(predict(geometry.phase0, geometry.phase1), kind="stable")[:3]
    starts = [
        natural,
        np.full(width, 1.0 / width),
        *(alpha * geometry.phase0[i] + (1.0 - alpha) * geometry.phase1[i] for i in ranked),
    ]
    with threadpool_limits(limits=1):
        for index, start in enumerate(starts):
            execute(start, True, f"tied{index}")
        tied_best = min(candidates, key=lambda record: record["predicted"])
        if not phase_blind:
            rng = np.random.default_rng(seed)
            free_starts = [
                np.r_[tied_best["phase0"], tied_best["phase0"]],
                *(np.r_[geometry.phase0[i], geometry.phase1[i]] for i in ranked),
                np.r_[rng.dirichlet(np.ones(width)), rng.dirichlet(np.ones(width))],
            ]
            for index, start in enumerate(free_starts):
                execute(start, False, f"free{index}")
            best = min(candidates, key=lambda record: record["predicted"])
            execute(
                alpha * np.asarray(best["phase0"]) + (1.0 - alpha) * np.asarray(best["phase1"]), True, "free_aggregate"
            )
    best = min(candidates, key=lambda record: record["predicted"])
    tied_best = min((record for record in candidates if record["tied"]), key=lambda record: record["predicted"])
    first, second = np.asarray(best["phase0"]), np.asarray(best["phase1"])
    aggregate = alpha * first + (1.0 - alpha) * second
    aggregate_value = float(predict(aggregate[None], aggregate[None])[0])
    distances = np.linalg.norm(
        np.c_[np.sqrt(geometry.phase0), np.sqrt(geometry.phase1)] - np.r_[np.sqrt(first), np.sqrt(second)], axis=1
    )
    return {
        "best": best,
        "best_static": tied_best,
        "aggregate": aggregate.tolist(),
        "predicted_aggregate_tied": aggregate_value,
        "predicted_gain_same_aggregate": aggregate_value - best["predicted"],
        "predicted_gain_vs_static": tied_best["predicted"] - best["predicted"],
        "phase_tv": float(0.5 * np.abs(first - second).sum()),
        "max_total_epochs": float(np.max(scales * aggregate)),
        "nearest_policy_sqrt_distance": float(distances.min()),
        "phase_blind_search": phase_blind,
        "traces": traces,
        "claim": "Best feasible numerical incumbents; no global optimality or observed-gain claim.",
    }


def evaluate_selected_truth(
    result: dict[str, Any], truth: Predictor, geometry: PolicyGeometry, truth_optimum: dict[str, Any]
) -> dict[str, float]:
    """Separate phase-gain calibration from aggregate-policy extrapolation."""
    first = np.asarray([result["best"]["phase0"], result["best_static"]["phase0"], result["aggregate"]])
    second = np.asarray([result["best"]["phase1"], result["best_static"]["phase1"], result["aggregate"]])
    values = truth.predict(geometry.alpha * geometry.scales * first, (1.0 - geometry.alpha) * geometry.scales * second)
    true_phase_gain = float(values[2] - values[0])
    return {
        "true_selected_free": float(values[0]),
        "true_selected_static": float(values[1]),
        "true_selected_aggregate_tied": float(values[2]),
        "true_gain_same_aggregate": true_phase_gain,
        "true_gain_vs_selected_static": float(values[1] - values[0]),
        "phase_gain_overprediction": result["predicted_gain_same_aggregate"] - true_phase_gain,
        "selected_optimism": float(values[0] - result["best"]["predicted"]),
        "regret_vs_truth_numerical_incumbent": float(values[0] - truth_optimum["best"]["predicted"]),
        "regret_vs_truth_static_incumbent": float(values[0] - truth_optimum["best_static"]["predicted"]),
    }
