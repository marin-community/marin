# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "threadpoolctl"]
# ///
"""Deterministic two-bucket numerical incumbents, with a separate replay callback.

No data loading, fitting, persistent output, or scientific execution is included.
The frozen older optimizer and its signed truth-accounting formulas are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.optimize import minimize, minimize_scalar
from threadpoolctl import threadpool_limits

from experiments.domain_phase_mix.exploratory.two_phase_many.tpp40_tangent_optimization_20261015 import (
    evaluate_selected_truth,
)

MESH = np.linspace(0.0, 1.0, 17)
SCALAR_OPTIONS = {"xatol": 1e-8, "maxiter": 48}
POWELL_OPTIONS = {"xtol": 1e-7, "ftol": 1e-10, "maxiter": 100, "maxfev": 400}
SEARCH_CALL_CAP = 8929
REPLAY_CHANGE_LIMIT = 1e-8
ScalarPredictor = Callable[[np.ndarray, np.ndarray], float]


class Restriction(StrEnum):
    AGGREGATE = "registered_exact_aggregate_or_null"
    TEMPORAL = "temporal"


@dataclass(frozen=True)
class Predictors:
    """Scalar physical-epoch callbacks with explicit scientific/numerical identity."""

    search: ScalarPredictor
    replay: ScalarPredictor
    identity: dict[str, Any]
    search_numerics: dict[str, Any]
    replay_numerics: dict[str, Any]


class EvaluationFailure(ArithmeticError):
    """A recorded invalid evaluation terminates only the current local solve."""


def coordinates_key(xy: tuple[float, float]) -> tuple[str, str]:
    return float(xy[0]).hex(), float(xy[1]).hex()


def materialize(xy: tuple[float, float], alpha: float, scales: np.ndarray) -> dict[str, Any]:
    """Convert code shares without clipping, renormalization, or projection."""
    x, y = xy
    feasible = bool(np.isfinite(xy).all() and 0 <= x <= 1 and 0 <= y <= 1)
    if not feasible:
        return {
            "coordinates": [float(value) if np.isfinite(value) else None for value in xy],
            "coordinate_repr": [repr(value) for value in xy],
            "feasible": False,
            "phase0": None,
            "phase1": None,
            "early_epochs": None,
            "late_epochs": None,
            "simplex_residual": None,
            "tied": False,
            "boundary": None,
        }
    first, second = np.array([1 - x, x]), np.array([1 - y, y])
    return {
        "coordinates": [float(x), float(y)],
        "feasible": True,
        "phase0": first.tolist(),
        "phase1": second.tolist(),
        "early_epochs": (alpha * scales * first).tolist(),
        "late_epochs": ((1 - alpha) * scales * second).tolist(),
        "simplex_residual": float(max(abs(first.sum() - 1), abs(second.sum() - 1))),
        "tied": x == y,
        "boundary": x in (0, 1) or y in (0, 1),
    }


def candidate_key(record: dict[str, Any]) -> tuple:
    return record["predicted"], not record["tied"], *record["coordinates"], record["source"], record["trial_id"]


def start_key(record: dict[str, Any]) -> tuple:
    return record["predicted"], *record["coordinates"], record["source"], record["trial_id"]


class Evaluator:
    """Keep every attempted trial and cache only exact coordinate/identity pairs."""

    def __init__(self, predictors: Predictors, alpha: float, scales: np.ndarray):
        self.predictors, self.alpha, self.scales = predictors, alpha, scales
        self.identity = {
            "predictor": predictors.identity,
            "alpha": alpha,
            "scales": scales.tolist(),
            "search_numerics": predictors.search_numerics,
            "replay_numerics": predictors.replay_numerics,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        }
        self.identity_key = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        self.cache: dict[tuple, dict[str, Any]] = {}
        self.traces: list[dict[str, Any]] = []
        self.calls = {"search": 0, "accounting": 0, "replay": 0}

    def evaluate(self, xy: tuple[float, float], source: str, kind: str = "search") -> dict[str, Any]:
        xy = float(xy[0]), float(xy[1])
        record = {
            **materialize(xy, self.alpha, self.scales),
            "source": source,
            "kind": kind,
            "trial_id": len(self.traces),
            "predicted": None,
            "success": False,
        }
        self.traces.append(record)
        if not record["feasible"]:
            record["status"] = "infeasible_coordinates"
            return record
        numerical_kind = "replay" if kind == "replay" else "search"
        key = self.identity_key, numerical_kind, *coordinates_key(xy)
        if key in self.cache:
            record.update(self.cache[key], cache_hit=True)
            return record
        if kind == "search" and self.calls["search"] >= SEARCH_CALL_CAP:
            record.update(status="search_call_cap", cache_hit=False)
            return record
        self.calls[kind] += 1
        function = self.predictors.replay if kind == "replay" else self.predictors.search
        try:
            value = float(function(np.asarray(record["early_epochs"]), np.asarray(record["late_epochs"])))
        except ArithmeticError as error:
            result = {
                "status": "numerical_failure",
                "predicted": None,
                "exception": type(error).__name__,
                "message": str(error),
                "success": False,
            }
        else:
            result = {
                "status": "finite" if np.isfinite(value) else "nonfinite_prediction",
                "predicted": value if np.isfinite(value) else None,
                "success": bool(np.isfinite(value)),
            }
        self.cache[key] = result
        record.update(result, cache_hit=False)
        return record


def finite_candidates(evaluator: Evaluator, records: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    return [
        record
        for record in (evaluator.traces if records is None else records)
        if record["kind"] != "replay" and record["status"] == "finite" and record["feasible"]
    ]


def execute_solver(
    evaluator: Evaluator,
    start: tuple[float, float],
    source: str,
    line: Callable[[float], tuple[float, float]] | None,
    interval: tuple[float, float] | None,
) -> dict[str, Any]:
    first_trial = len(evaluator.traces)
    initial = evaluator.evaluate(start, source + ":start")
    callback_calls = 0
    cap = SCALAR_OPTIONS["maxiter"] if line is not None else POWELL_OPTIONS["maxfev"]

    def objective(point: Any) -> float:
        nonlocal callback_calls
        callback_calls += 1
        if callback_calls > cap:
            raise EvaluationFailure("Prescribed per-solve callback budget exhausted")
        xy = line(float(point)) if line is not None else tuple(map(float, point))
        record = evaluator.evaluate(xy, source + f":trial{callback_calls}")
        if record["status"] != "finite":
            raise EvaluationFailure(f"Recorded evaluation failed: {record['status']}")
        return record["predicted"]

    outcome: dict[str, Any] = {"source": source, "start": list(start), "kind": "line" if line else "powell"}
    try:
        if initial["status"] != "finite":
            raise EvaluationFailure(f"Recorded starting evaluation failed: {initial['status']}")
        if line is not None:
            result = minimize_scalar(objective, bounds=interval, method="bounded", options=SCALAR_OPTIONS)
            endpoint = line(float(result.x))
        else:
            result = minimize(
                objective,
                np.asarray(start),
                method="Powell",
                bounds=[(0, 1), (0, 1)],
                options={**POWELL_OPTIONS, "direc": np.eye(2)},
            )
            endpoint = tuple(map(float, result.x))
        final = evaluator.evaluate(endpoint, source + ":endpoint")
        outcome.update(
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            nfev=int(result.nfev),
            nit=int(result.nit) if "nit" in result else None,
            endpoint=list(endpoint),
            endpoint_trial=final["trial_id"],
            termination="success" if result.success else "solver_unsuccessful",
        )
    except EvaluationFailure as error:
        outcome.update(
            success=False,
            status=None,
            message=str(error),
            nfev=callback_calls,
            nit=None,
            endpoint=None,
            endpoint_trial=None,
            termination="recorded_evaluation_failure",
        )
    records = evaluator.traces[first_trial:]
    for record in records:
        record["solver_success"] = outcome["success"]
        record["success"] = outcome["success"] and record["status"] == "finite"
    finite = finite_candidates(evaluator, records)
    outcome["first_trial"] = first_trial
    outcome["last_trial"] = len(evaluator.traces) - 1
    outcome["callback_calls"] = callback_calls
    outcome["best_trial"] = min(finite, key=candidate_key)["trial_id"] if finite else None
    outcome["finite_trial_count"] = len(finite)
    return outcome


def powell_starts(evaluator: Evaluator, mesh: list[dict[str, Any]]) -> list[tuple[float, float]]:
    eligible = finite_candidates(evaluator)
    mesh = finite_candidates(evaluator, mesh)
    ordered = []
    groups = [[record for record in eligible if record["tied"]]]
    groups.extend(
        [record for record in eligible if record["coordinates"][axis] == endpoint]
        for axis, endpoint in ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    groups.extend(
        [
            record
            for record in mesh
            if (record["coordinates"][0] >= 0.5) == right and (record["coordinates"][1] >= 0.5) == upper
        ]
        for right, upper in ((False, False), (False, True), (True, False), (True, True))
    )
    for group in groups:
        if group:
            ordered.append(min(group, key=start_key))
    ordered.extend(sorted(mesh, key=start_key))
    starts, seen = [], set()
    for record in ordered:
        xy = tuple(record["coordinates"])
        if coordinates_key(xy) not in seen:
            starts.append(xy)
            seen.add(coordinates_key(xy))
        if len(starts) == 12:
            break
    return starts


def replay_finalists(
    evaluator: Evaluator,
    mesh: list[dict[str, Any]],
    solves: list[dict[str, Any]],
    reporting: list[dict[str, Any]],
) -> dict[str, Any]:
    finalists = {coordinates_key(tuple(record["coordinates"])): record for record in mesh if record["feasible"]}
    for solve in solves:
        indices = [solve["first_trial"], solve["last_trial"], solve["best_trial"]]
        for index in indices:
            if index is not None and evaluator.traces[index]["feasible"]:
                record = evaluator.traces[index]
                finalists[coordinates_key(tuple(record["coordinates"]))] = record
    for record in reporting:
        if record["feasible"]:
            finalists[coordinates_key(tuple(record["coordinates"]))] = record
    replays = []
    for index, selected in enumerate(finalists.values()):
        record = evaluator.evaluate(tuple(selected["coordinates"]), f"replay:{index}", "replay")
        change = (
            None
            if selected["predicted"] is None or record["predicted"] is None
            else record["predicted"] - selected["predicted"]
        )
        replays.append(
            {
                "search_trial": selected["trial_id"],
                "replay_trial": record["trial_id"],
                "coordinates": selected["coordinates"],
                "search_value": selected["predicted"],
                "replay_value": record["predicted"],
                "signed_change": change,
                "status": record["status"],
            }
        )
    paired = [item for item in replays if item["signed_change"] is not None]
    values = np.asarray([item["search_value"] for item in paired])
    repeated = np.asarray([item["replay_value"] for item in paired])
    reversals = int(np.sum(np.triu((values[:, None] - values) * (repeated[:, None] - repeated) < 0, k=1)))
    max_change = max((abs(item["signed_change"]) for item in paired), default=0.0)
    report_keys = [coordinates_key(tuple(record["coordinates"])) for record in reporting]
    selected_replays = [
        next(item for item in replays if coordinates_key(tuple(item["coordinates"])) == key) for key in report_keys
    ]
    return {
        "records": replays,
        "count": len(replays),
        "maximum_absolute_change": max_change,
        "selected_reporting_replays": dict(
            zip(("best", "best_static", "aggregate_tied"), selected_replays, strict=True)
        ),
        "strict_pairwise_ordering_reversals": reversals,
        "change_limit_bpb": REPLAY_CHANGE_LIMIT,
        "unresolved": max_change > REPLAY_CHANGE_LIMIT or any(item["status"] != "finite" for item in replays),
        "selection_reranked": False,
    }


def optimize_two_bucket(
    predictors: Predictors,
    *,
    alpha: float,
    scales: np.ndarray,
    restriction: Restriction,
    observed_policies: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return numerical incumbents and all failures under the frozen bounded search."""
    scales = np.asarray(scales, float)
    if scales.shape != (2,) or np.any(scales <= 0) or not np.isfinite(scales).all() or not 0 < alpha < 1:
        raise ValueError("Require two positive exposure scales and an interior phase fraction")
    if restriction not in (Restriction.AGGREGATE, Restriction.TEMPORAL):
        raise ValueError("Declare the exact registered model restriction")
    evaluator = Evaluator(predictors, alpha, scales)
    solves = []
    with threadpool_limits(limits=1):
        points = (
            [(float(z), float(z)) for z in MESH]
            if restriction == Restriction.AGGREGATE
            else [(float(x), float(y)) for x in MESH for y in MESH]
        )
        mesh = [evaluator.evaluate(point, f"mesh:{index}") for index, point in enumerate(points)]
        lines: list[tuple[str, Callable]] = [("tied", lambda z: (z, z))]
        if restriction == Restriction.TEMPORAL:
            lines.extend(
                (
                    ("early0", lambda z: (0.0, z)),
                    ("early1", lambda z: (1.0, z)),
                    ("late0", lambda z: (z, 0.0)),
                    ("late1", lambda z: (z, 1.0)),
                )
            )
        for name, line in lines:
            for index, (left, right) in enumerate(pairwise(MESH)):
                solves.append(
                    execute_solver(evaluator, line(float(left)), f"{name}:{index}", line, (float(left), float(right)))
                )
        starts = []
        if restriction == Restriction.TEMPORAL:
            starts = powell_starts(evaluator, mesh)
            for index, start in enumerate(starts):
                solves.append(execute_solver(evaluator, start, f"powell:{index}", None, None))
        candidates = finite_candidates(evaluator)
        tied_candidates = [record for record in candidates if record["tied"]]
        if not candidates or not tied_candidates:
            return {
                "identity": evaluator.identity,
                "identity_key": evaluator.identity_key,
                "best": None,
                "best_static": None,
                "numerical_audit_unresolved": True,
                "failure": "No finite feasible tied incumbent",
                "traces": evaluator.traces,
                "solves": solves,
                "calls": evaluator.calls,
                "claim": "No valid numerical policy result.",
            }
        original_best = min(candidates, key=candidate_key)
        initial_xy = original_best["coordinates"]
        z = float(alpha * initial_xy[0] + (1 - alpha) * initial_xy[1])
        accounting = evaluator.evaluate((z, z), "selected_aggregate_tied", "accounting")
        candidates = finite_candidates(evaluator)
        if (
            accounting["coordinates"] == original_best["coordinates"]
            and accounting["predicted"] == original_best["predicted"]
        ):
            candidates = [record for record in candidates if record["trial_id"] != accounting["trial_id"]]
        best, best_static = (
            min(candidates, key=candidate_key),
            min((r for r in candidates if r["tied"]), key=candidate_key),
        )
        # A successful replacement is itself tied, so a single accounting evaluation closes the step.
        replaced = best["trial_id"] != original_best["trial_id"]
        first, second = np.asarray(best["phase0"]), np.asarray(best["phase1"])
        aggregate = alpha * first + (1 - alpha) * second
        aggregate_value = best["predicted"] if replaced else accounting["predicted"]
        if restriction == Restriction.AGGREGATE:
            best = best_static
            aggregate, aggregate_value = np.asarray(best["phase0"]), best["predicted"]
        final_aggregate = best if replaced else accounting
        replay = replay_finalists(evaluator, mesh, solves, [best, best_static, final_aggregate])
    distance = None
    if observed_policies is not None:
        observed = np.asarray(observed_policies, float)
        if (
            observed.ndim != 3
            or observed.shape[1:] != (2, 2)
            or not len(observed)
            or np.any(observed < 0)
            or not np.isfinite(observed).all()
        ):
            raise ValueError("Observed covariates must be nonnegative policy-by-phase-by-two weights")
        if not np.allclose(observed.sum(axis=2), 1, atol=1e-12, rtol=0):
            raise ValueError("Observed policy weights violate the simplexes")
        distance = float(
            np.min(
                np.linalg.norm(np.sqrt(observed).reshape(len(observed), 4) - np.sqrt([first, second]).ravel(), axis=1)
            )
        )
    bad_evaluations = [r["trial_id"] for r in evaluator.traces if r["status"] != "finite"]
    missing_starts = restriction == Restriction.TEMPORAL and len(starts) != 12
    unsuccessful_lines = [item["source"] for item in solves if item["kind"] == "line" and not item["success"]]
    return {
        "identity": evaluator.identity,
        "identity_key": evaluator.identity_key,
        "best": best,
        "best_static": best_static,
        "original_search_best": original_best,
        "accounting_tied_replaced_incumbent": replaced,
        "accounting_tied_trial": accounting["trial_id"],
        "accounting_replacement_reason": (
            "Better finite tied policy or exact-value tied/coordinate preference" if replaced else None
        ),
        "aggregate": aggregate.tolist(),
        "predicted_aggregate_tied": aggregate_value,
        "predicted_gain_same_aggregate": None if aggregate_value is None else aggregate_value - best["predicted"],
        "predicted_gain_vs_static": best_static["predicted"] - best["predicted"],
        "phase_tv": float(0.5 * abs(first - second).sum()),
        "max_total_epochs": float(np.max(scales * aggregate)),
        "nearest_policy_sqrt_distance": distance,
        "phase_blind_search": restriction == Restriction.AGGREGATE,
        "powell_starts": [list(point) for point in starts],
        "missing_prescribed_starts": missing_starts,
        "calls": evaluator.calls,
        "search_call_cap": SEARCH_CALL_CAP,
        "reporting_call_cap": 1,
        "failed_evaluation_trials": bad_evaluations,
        "failed_local_solves": sum(not item["success"] for item in solves),
        "unsuccessful_required_line_solves": unsuccessful_lines,
        "numerical_audit_unresolved": bool(
            bad_evaluations or missing_starts or unsuccessful_lines or replay["unresolved"]
        ),
        "replay": replay,
        "traces": evaluator.traces,
        "solves": solves,
        "claim": "Best feasible numerical incumbents under bounded search; no global optimum or observed-gain claim.",
    }


# Keep this symbol available for callers adopting the previous exact signed accounting schema.
SIGNED_TRUTH_ACCOUNTING = evaluate_selected_truth
