# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Audit raw simplex optima of the frozen two-phase log-link transfer fits.

Prediction matches the separate aggregate/correction overflow guards in the
temporal fitter. Gradients are analytic in the interior. The Weibull derivative
uses a 1e-12-epoch numerical floor at zero, where powers below one have no finite
derivative; this floor never alters function values or imposes a policy bound.
All starts and solver failures are retained. No policy is trained or submitted.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import fit_two_phase_link_transfer_20260907 as driver
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.special import expit

DERIVATIVE_EPOCH_FLOOR = 1e-12
LOG_CLIP = 30.0
START_SEED = 20260907
FEASIBILITY_TOLERANCE = 1e-7
OPTIMIZER_OPTIONS = {"maxiter": 500, "ftol": 1e-11}
PARITY_TOLERANCE = 1e-8
GRADIENT_TOLERANCE = 2e-6


@dataclass(frozen=True)
class ObjectiveModel:
    """Vectorized immutable task parameters for one target and fit context."""

    aggregation: np.ndarray
    rate: np.ndarray
    power: np.ndarray
    threshold: np.ndarray
    intercept: np.ndarray
    floor: np.ndarray
    benefit_amplitudes: np.ndarray
    harm_amplitudes: np.ndarray
    theta: np.ndarray
    c0: np.ndarray
    c1: np.ndarray

    @property
    def buckets(self) -> int:
        return len(self.c0)

    @property
    def phase_fraction(self) -> np.ndarray:
        return self.c0 / (self.c0 + self.c1)

    def laws(self, exposure: np.ndarray) -> tuple[np.ndarray, ...]:
        """Task by bucket values and exposure derivatives at one policy state."""
        values = np.maximum(exposure[None, :], 0.0)
        rate, power = self.rate[:, None], self.power[:, None]
        scaled = (rate * values) ** power
        benefit = -np.expm1(-scaled)
        safe = np.maximum(values, DERIVATIVE_EPOCH_FLOOR)
        safe_scaled = (rate * safe) ** power
        benefit_derivative = power * safe_scaled * np.exp(-safe_scaled) / safe
        argument = np.log1p(values) - self.threshold[:, None]
        softplus = np.logaddexp(argument, 0.0)
        harm = softplus**2
        harm_derivative = 2.0 * softplus * expit(argument) / (1.0 + values)
        return benefit, harm, benefit_derivative, harm_derivative

    def value_gradient(self, weights: np.ndarray) -> tuple[float, np.ndarray, dict[str, Any]]:
        """Predict one two-phase policy and its gradient in unconstrained weights."""
        weights = np.asarray(weights, float).reshape(2, self.buckets)
        early = self.c0 * weights[0]
        total = early + self.c1 * weights[1]
        tied_early = self.phase_fraction * total
        tied_early = np.where(weights[0] == weights[1], early, tied_early)
        benefit, harm, db, dh = self.laws(total)
        b0, h0, db0, dh0 = self.laws(early)
        bt, ht, dbt, dht = self.laws(tied_early)
        a, b = self.benefit_amplitudes, self.harm_amplitudes
        eta = self.intercept + np.sum(-a * benefit + b * harm, axis=1)
        z_b = np.sum(a * (b0 - bt), axis=1)
        z_h = np.sum(b * (h0 - ht), axis=1)
        correction = self.theta[:, 0] * z_b + self.theta[:, 1] * z_h
        deficit = np.exp(np.clip(eta, -LOG_CLIP, LOG_CLIP)) * np.exp(np.clip(correction, -LOG_CLIP, LOG_CLIP))
        prediction = self.floor + deficit
        eta_active = (np.abs(eta) < LOG_CLIP).astype(float)
        correction_active = (np.abs(correction) < LOG_CLIP).astype(float)
        aggregate_derivative = -a * db + b * dh
        common = self.theta[:, 0, None] * a * dbt + self.theta[:, 1, None] * b * dht
        early_only = self.theta[:, 0, None] * a * db0 + self.theta[:, 1, None] * b * dh0
        correction_gradient0 = self.c0 * (early_only - self.phase_fraction * common)
        correction_gradient1 = -self.c1 * self.phase_fraction * common
        weighted_deficit = self.aggregation * deficit
        gradient = np.stack(
            [
                np.sum(
                    weighted_deficit[:, None]
                    * (
                        eta_active[:, None] * self.c0 * aggregate_derivative
                        + correction_active[:, None] * correction_gradient0
                    ),
                    axis=0,
                ),
                np.sum(
                    weighted_deficit[:, None]
                    * (
                        eta_active[:, None] * self.c1 * aggregate_derivative
                        + correction_active[:, None] * correction_gradient1
                    ),
                    axis=0,
                ),
            ]
        )
        diagnostic = {
            "aggregate_clip_components": int(np.sum(np.abs(eta) >= LOG_CLIP)),
            "correction_clip_components": int(np.sum(np.abs(correction) >= LOG_CLIP)),
            "combined_linear_over_30_components": int(np.sum(np.abs(eta + correction) >= LOG_CLIP)),
            "max_abs_aggregate_linear": float(np.max(np.abs(eta))),
            "max_abs_correction": float(np.max(np.abs(correction))),
            "derivative_floor_total_buckets": int(np.sum(total < DERIVATIVE_EPOCH_FLOOR)),
            "derivative_floor_early_buckets": int(np.sum(early < DERIVATIVE_EPOCH_FLOOR)),
            "derivative_floor_tied_early_buckets": int(np.sum(tied_early < DERIVATIVE_EPOCH_FLOOR)),
        }
        return float(prediction @ self.aggregation), gradient, diagnostic


def load_model(output: Path, context: str, objective: str, arm: str) -> ObjectiveModel:
    _, panel, _ = driver.inputs(str(output))
    width = len(panel["buckets"])
    spines = [driver.load_spine(output, context, objective, i) for i in range(len(panel[f"{objective}_components"]))]
    theta = np.zeros((len(spines), 2))
    if arm != "aggregate":
        for i in range(len(spines)):
            path = output / "temporal" / context / f"{objective}_c{i}.json"
            theta[i] = json.loads(path.read_text())["fits"][arm]["theta"]
    return ObjectiveModel(
        panel[f"{objective}_aggregation_weights"],
        np.asarray([s.shape["rate"] for s in spines]),
        np.asarray([s.shape["power"] for s in spines]),
        np.asarray([s.shape["threshold"] for s in spines]),
        np.asarray([s.head.intercept for s in spines]),
        np.asarray([s.head.floor for s in spines]),
        np.stack([s.head.coefficients[:width] for s in spines]),
        np.stack([s.head.coefficients[width:] for s in spines]),
        theta,
        panel["c0"],
        panel["c1"],
    )


def check_model(output: Path, context: str, objective: str, arm: str, model: ObjectiveModel) -> dict[str, Any]:
    """Check independent component prediction and interior tangent derivatives."""
    module, panel, _ = driver.inputs(str(output))
    reference = np.zeros(len(panel["runs"]))
    for component, weight in enumerate(model.aggregation):
        spine = driver.load_spine(output, context, objective, component)
        base, q, columns = driver.basis_and_prediction(module, panel, spine)
        reference += weight * (base + driver.predict_bpb_delta(q, columns, model.theta[component]))
    predicted = np.asarray([model.value_gradient(w)[0] for w in panel["weights"]])
    parity = float(np.max(np.abs(predicted - reference)))
    if parity > PARITY_TOLERANCE:
        raise ValueError(f"Prediction parity failed at {context}/{objective}/{arm}: {parity}")
    rng = np.random.default_rng(START_SEED)
    gradient_error = 0.0
    for _ in range(4):
        policy = rng.dirichlet(np.full(model.buckets, 3.0), size=2)
        _, gradient, _ = model.value_gradient(policy)
        for _ in range(8):
            direction = rng.normal(size=policy.shape)
            direction -= direction.mean(axis=1, keepdims=True)
            direction /= np.linalg.norm(direction)
            step = 1e-6
            plus = model.value_gradient(policy + step * direction)[0]
            minus = model.value_gradient(policy - step * direction)[0]
            numerical = (plus - minus) / (2 * step)
            analytic = float(np.sum(gradient * direction))
            gradient_error = max(gradient_error, abs(numerical - analytic) / max(1.0, abs(analytic)))
    if gradient_error > GRADIENT_TOLERANCE:
        raise ValueError(f"Gradient parity failed at {context}/{objective}/{arm}: {gradient_error}")
    proportional = 1.0 / (model.c0 + model.c1)
    proportional /= proportional.sum()
    tied = np.stack([proportional, proportional])
    null = replace(model, theta=np.zeros_like(model.theta))
    tied_error = abs(model.value_gradient(tied)[0] - null.value_gradient(tied)[0])
    if tied_error != 0.0:
        raise ValueError(f"Tied restriction failed: {tied_error}")
    return {
        "max_prediction_error": parity,
        "max_interior_tangent_gradient_relative_error": gradient_error,
        "tied_prediction_error": tied_error,
        "derivative_epoch_floor": DERIVATIVE_EPOCH_FLOOR,
        "gradient_scope": "analytic in the interior; finite numerical derivative floor at zero only",
    }


def policy_tv(left: np.ndarray, right: np.ndarray, alpha: float) -> float:
    phase = np.sum(np.abs(left - right), axis=1) / 2
    return float(alpha * phase[0] + (1 - alpha) * phase[1])


def solve_start(model: ObjectiveModel, start: np.ndarray, tied: bool, name: str) -> tuple[np.ndarray, dict[str, Any]]:
    width = model.buckets

    def lift(flat: np.ndarray) -> np.ndarray:
        return np.stack([flat, flat]) if tied else flat.reshape(2, width)

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient, _ = model.value_gradient(lift(flat))
        return value, gradient.sum(axis=0) if tied else gradient.ravel()

    dimensions = width if tied else 2 * width
    constraints = np.ones((1, width)) if tied else np.kron(np.eye(2), np.ones((1, width)))
    initial = start[0].copy() if tied else start.ravel().copy()
    initial_value = objective(initial)[0]
    result = minimize(
        objective,
        initial,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * dimensions,
        constraints={"type": "eq", "fun": lambda x: constraints @ x - 1, "jac": lambda _x: constraints},
        options=OPTIMIZER_OPTIONS,
    )
    violation = float(max(np.max(np.abs(constraints @ result.x - 1)), -np.min(result.x), np.max(result.x) - 1))
    candidate = start.copy()
    endpoint_accepted = np.isfinite(result.x).all() and violation <= FEASIBILITY_TOLERANCE
    endpoint_value = None
    projection_tv = None
    if endpoint_accepted:
        endpoint = np.maximum(lift(result.x), 0.0)
        endpoint /= endpoint.sum(axis=1, keepdims=True)
        projection_tv = policy_tv(endpoint, lift(result.x), float(model.phase_fraction[0]))
        endpoint_value = model.value_gradient(endpoint)[0]
        if math.isfinite(endpoint_value) and endpoint_value < initial_value:
            candidate = endpoint
    value, _, clips = model.value_gradient(candidate)
    return candidate, {
        "name": name,
        "initial_weights": start.tolist(),
        "initial_bpb": initial_value,
        "endpoint_bpb": endpoint_value,
        "retained_bpb": value,
        "success": bool(result.success and endpoint_accepted),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "feasibility_violation": violation if math.isfinite(violation) else None,
        "endpoint_projection_tv": projection_tv,
        "retained_initial_start": bool(np.array_equal(candidate, start)),
        "clips": clips,
        "weights": candidate.tolist(),
    }


def support_audit(policy: np.ndarray, observed: np.ndarray, alpha: float) -> dict[str, Any]:
    """Exact weighted policy-TV hull distance via the existing L1 LP formulation.

    Phase scaling makes the flattened policies probability vectors. The LP is
    the same L1 epigraph formulation used by optimize_delphi_matched_policies;
    it is kept local because that script imports the full historical fitting
    stack, whereas this audit consumes only frozen standalone parameters.
    """
    phase_scale = np.asarray([alpha, 1 - alpha])[:, None]
    cloud = (observed * phase_scale).reshape(len(observed), -1)
    point = (policy * phase_scale).ravel()
    nearest = np.abs(cloud - point).sum(axis=1) / 2
    aggregate = (observed * phase_scale).sum(axis=1)
    candidate_aggregate = (policy * phase_scale).sum(axis=0)
    rows, columns = cloud.shape
    identity = np.eye(columns)
    solution = linprog(
        np.r_[np.zeros(rows), np.full(columns, 0.5)],
        A_ub=np.vstack([np.hstack([cloud.T, -identity]), np.hstack([-cloud.T, -identity])]),
        b_ub=np.r_[point, -point],
        A_eq=np.r_[np.ones(rows), np.zeros(columns)][None],
        b_eq=[1.0],
        bounds=(0, None),
        method="highs",
    )
    if not solution.success:
        raise ValueError(f"Convex support distance failed: {solution.message}")
    return {
        "nearest_policy_tv": float(nearest.min()),
        "nearest_policy_index": int(np.argmin(nearest)),
        "nearest_aggregate_tv": float(np.min(np.abs(aggregate - candidate_aggregate).sum(axis=1) / 2)),
        "convex_hull_policy_tv": float(solution.fun),
    }


def starts_for(model: ObjectiveModel, observed: np.ndarray, tied: bool, tied_optimum: np.ndarray | None):
    rng = np.random.default_rng(START_SEED)
    proportional = 1.0 / (model.c0 + model.c1)
    proportional /= proportional.sum()
    starts = [("proportional", np.stack([proportional, proportional]))]
    if tied_optimum is not None:
        starts.insert(0, ("fitted_tied_optimum", tied_optimum))
    for index in rng.choice(len(observed), size=min(4, len(observed)), replace=False):
        policy = observed[index]
        if tied:
            aggregate = (model.c0 * policy[0] + model.c1 * policy[1]) / (model.c0 + model.c1)
            policy = np.stack([aggregate, aggregate])
        starts.append((f"training_policy_{index}", policy))
    for index in range(2):
        policy = rng.dirichlet(np.ones(model.buckets), size=1 if tied else 2)
        if tied:
            policy = np.repeat(policy, 2, axis=0)
        starts.append((f"dirichlet_{index}", policy))
    return starts


def run_cell(output: Path, context: str, objective: str, arm: str, tied_result: dict | None) -> dict[str, Any]:
    root = output / "optima"
    root.mkdir(exist_ok=True)
    path = root / f"{context}_{objective}_{arm}.json"
    _, panel, splits = driver.inputs(str(output))
    source_paths = [
        Path(__file__),
        Path(driver.__file__),
        Path(driver.__file__).with_name("two_phase_link_residual_20260907.py"),
        output / "inputs/single_phase.py",
        output / "inputs/panel.npz",
        output / "inputs/splits.npz",
    ]
    for component in range(len(panel[f"{objective}_components"])):
        source_paths.append(output / "spines" / context / f"{objective}_c{component}.json")
        if arm != "aggregate":
            source_paths.append(output / "temporal" / context / f"{objective}_c{component}.json")
    identity = {str(p): driver.file_hash(p) for p in source_paths}
    if path.exists():
        record = json.loads(path.read_text())
        if record["input_hashes"] != identity:
            raise ValueError(f"Optimum audit source changed: {path}")
        return record
    model = load_model(output, context, objective, arm)
    checks = check_model(output, context, objective, arm, model)
    train = driver.training_rows(panel, splits, context)
    observed = panel["weights"][train]
    tied = arm == "aggregate"
    tied_optimum = None if tied_result is None else np.asarray(tied_result["weights"])
    results = [solve_start(model, start, tied, name) for name, start in starts_for(model, observed, tied, tied_optimum)]
    policy, best = min(results, key=lambda pair: pair[1]["retained_bpb"])
    value = best["retained_bpb"]
    alpha = float(panel["alpha"])
    stability = [policy_tv(left[0], right[0], alpha) for i, left in enumerate(results) for right in results[i + 1 :]]
    support = support_audit(policy, observed, alpha)
    full_support = support_audit(policy, panel["weights"], alpha)
    epochs = panel["c0"] * policy[0] + panel["c1"] * policy[1]
    response = panel[f"{objective}_outcomes"] @ panel[f"{objective}_aggregation_weights"]
    nearest_training = int(train[support["nearest_policy_index"]])
    record = {
        "context": context,
        "objective": objective,
        "arm": arm,
        "exact_phase_null": bool(np.all(model.theta == 0.0)),
        "input_hashes": identity,
        "checks": checks,
        "predicted_bpb": value,
        "fitted_tied_optimum_bpb": value if tied_result is None else tied_result["predicted_bpb"],
        "predicted_gain_over_fitted_tied_optimum": 0.0 if tied_result is None else tied_result["predicted_bpb"] - value,
        "best_observed_bpb": float(response.min()),
        "predicted_improvement_over_best_observed": float(response.min() - value),
        "best_training_bpb": float(response[train].min()),
        "max_phase_bucket_weight": float(policy.max()),
        "max_total_epochs": float(epochs.max()),
        "phase_tv": float(np.sum(np.abs(policy[0] - policy[1])) / 2),
        "weights": policy.tolist(),
        "total_epochs": epochs.tolist(),
        "selected_start": best["name"],
        "selected_solver_success": best["success"],
        "successful_starts": sum(result[1]["success"] for result in results),
        "starts": [result[1] for result in results],
        "all_start_pairwise_policy_tv_max": max(stability, default=0.0),
        "all_start_pairwise_policy_tv_median": float(np.median(stability)) if stability else 0.0,
        "all_start_bpb_spread": float(np.ptp([result[1]["retained_bpb"] for result in results])),
        "training_support": support,
        "full_design_support": full_support,
        "nearest_training_row": nearest_training,
        "nearest_training_measured_bpb": float(response[nearest_training]),
        "clips": best["clips"],
        "interpretation": "Unregularized simplex optimum found by multistart SLSQP; no new measured outcome.",
    }
    driver.write_json(path, record)
    print(json.dumps({k: record[k] for k in ("context", "objective", "arm", "predicted_bpb", "phase_tv")}), flush=True)
    return record


def summarize(output: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "context",
        "objective",
        "arm",
        "predicted_bpb",
        "predicted_gain_over_fitted_tied_optimum",
        "predicted_improvement_over_best_observed",
        "max_phase_bucket_weight",
        "max_total_epochs",
        "phase_tv",
        "selected_start",
        "selected_solver_success",
        "successful_starts",
        "all_start_pairwise_policy_tv_max",
        "all_start_pairwise_policy_tv_median",
        "all_start_bpb_spread",
    ]
    rows = []
    for record in records:
        row = {key: record[key] for key in columns}
        row.update({f"training_{key}": value for key, value in record["training_support"].items()})
        row.update({f"full_{key}": value for key, value in record["full_design_support"].items()})
        row.update(record["clips"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "optima/summary.csv", index=False)
    _, panel, _ = driver.inputs(str(output))
    distances = []
    for objective in driver.OBJECTIVES:
        for arm in driver.ARMS:
            selected = [r for r in records if r["objective"] == objective and r["arm"] == arm]
            for i, left in enumerate(selected):
                for right in selected[i + 1 :]:
                    a, b = np.asarray(left["weights"]), np.asarray(right["weights"])
                    aggregate_a = (panel["c0"] * a[0] + panel["c1"] * a[1]) / panel["inventory"]
                    aggregate_b = (panel["c0"] * b[0] + panel["c1"] * b[1]) / panel["inventory"]
                    distances.append(
                        {
                            "objective": objective,
                            "arm": arm,
                            "left_context": left["context"],
                            "right_context": right["context"],
                            "policy_tv": policy_tv(a, b, float(panel["alpha"])),
                            "aggregate_tv": float(np.abs(aggregate_a - aggregate_b).sum() / 2),
                        }
                    )
    pd.DataFrame(distances).to_csv(output / "optima/context_stability.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=driver.DEFAULT_OUTPUT)
    parser.add_argument("--contexts", nargs="+", choices=driver.CONTEXTS, default=list(driver.CONTEXTS))
    parser.add_argument("--objectives", nargs="+", choices=driver.OBJECTIVES, default=list(driver.OBJECTIVES))
    args = parser.parse_args()
    _, panel, _ = driver.inputs(str(args.output))
    fractions = panel["c0"] / (panel["c0"] + panel["c1"])
    if not np.allclose(fractions, float(panel["alpha"]), rtol=0.0, atol=1e-10):
        raise ValueError("Policy-TV geometry requires a common physical phase fraction")
    required = [
        args.output / "temporal" / context / f"{objective}_c{component}.json"
        for context in args.contexts
        for objective in args.objectives
        for component in range(len(panel[f"{objective}_components"]))
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(f"Wait for the registered phase fits before optimizing; {len(missing)} files missing")
    records = []
    for context in args.contexts:
        for objective in args.objectives:
            baseline = run_cell(args.output, context, objective, "aggregate", None)
            records.append(baseline)
            for arm in driver.ARMS[1:]:
                records.append(run_cell(args.output, context, objective, arm, baseline))
    summarize(args.output, records)


if __name__ == "__main__":
    main()
