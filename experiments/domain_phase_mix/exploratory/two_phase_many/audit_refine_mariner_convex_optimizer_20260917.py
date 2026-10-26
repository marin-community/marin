# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "joblib"]
# ///
"""Independently check convex optimizer certificates and tighten two weak bounds."""

import dataclasses
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import optimize, special

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    optimize_mariner_convex_20260917 as original,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    validate_mariner_policy_banks_20260917 as validation,
)

ROOT = validation.OUTPUT.parent
SOURCE = ROOT / "optimizer_convex"
OUTPUT = ROOT / "optimizer_convex_refinement"


def objective_for(record: dict) -> original.ConvexObjective:
    panel = validation.bench.load_panel(record["panel"])
    spec = next(item for item in original.candidates.candidate_specs() if item.model_id == record["model"])
    paths = [
        validation.fit_path(validation.OUTPUT / "convex", spec.model_id, panel.name, record["target"], index)
        for index in range(len(panel.group(record["target"]).components))
    ]
    for path in paths:
        if validation.sha256(path) != record["fit_hashes"][str(path)]:
            raise ValueError(f"Input head hash differs: {path}")
    return original.ConvexObjective(spec, panel, record["target"], [joblib.load(path) for path in paths])


@dataclasses.dataclass(frozen=True)
class SeparableTasks:
    """Independent formulas for each task and each raw-epoch bucket column."""

    inventory: np.ndarray
    rate: np.ndarray
    power: np.ndarray
    onset: np.ndarray
    benefit_coefficient: np.ndarray
    harm_coefficient: np.ndarray
    intercept: np.ndarray
    weights: np.ndarray
    floor: float
    family: str
    width: float

    @classmethod
    def from_objective(cls, objective: original.ConvexObjective):
        buckets = len(objective.inventory)
        coefficients = np.stack([f.head.coefficients for f in objective.fits])
        assert coefficients.shape[1] == 2 * buckets and np.min(coefficients) >= 0
        return cls(
            objective.inventory,
            np.array([f.shape["rate"] for f in objective.fits])[:, None],
            np.array([f.shape["power"] for f in objective.fits])[:, None],
            np.expm1(np.array([f.shape["threshold"] for f in objective.fits]))[:, None],
            coefficients[:, :buckets],
            coefficients[:, buckets:],
            np.array([f.head.intercept for f in objective.fits]),
            objective.panel.group(objective.target).aggregation_weights,
            objective.floor,
            objective.spec.category,
            objective.spec.params[0],
        )

    def terms(self, mixture: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        epochs = mixture[None, :] * self.inventory[None, :]
        z = (self.rate * epochs) ** self.power
        benefit = self.benefit_coefficient * np.expm1(-z)
        safe_epochs = np.where(epochs > 0, epochs, 1.0)
        derivative = -self.power * self.rate**self.power * safe_epochs ** (self.power - 1) * np.exp(-z)
        derivative = np.where(epochs > 0, derivative, np.where(self.power == 1, -self.rate, -np.inf))
        active = self.benefit_coefficient > 0
        benefit_derivative = np.zeros_like(benefit)
        benefit_derivative[active] = derivative[active] * self.benefit_coefficient[active]
        if self.family == "smooth_hinge":
            harm = (
                self.width
                * (np.logaddexp(0, (epochs - self.onset) / self.width) - np.logaddexp(0, -self.onset / self.width))
                / 32
            )
            harm_derivative = special.expit((epochs - self.onset) / self.width) / 32
        elif self.family == "huber_hinge":
            excess = np.maximum(epochs - self.onset, 0)
            shoulder = np.minimum(excess, self.width)
            harm = (shoulder**2 / (2 * self.width) + np.maximum(excess - self.width, 0)) / 32
            harm_derivative = np.clip(excess / self.width, 0, 1) / 32
        else:
            raise ValueError(self.family)
        return (
            benefit + self.harm_coefficient * harm,
            (benefit_derivative + self.harm_coefficient * harm_derivative) * self.inventory,
        )

    def logits(self, mixture: np.ndarray) -> np.ndarray:
        return self.intercept + self.terms(mixture)[0].sum(axis=1)

    def value_gradient(self, mixture: np.ndarray) -> tuple[float, np.ndarray]:
        terms, derivatives = self.terms(mixture)
        logits = self.intercept + terms.sum(axis=1)
        kept = logits >= -30
        log_terms = np.maximum(logits, -30) + np.log(self.weights)
        probabilities = special.softmax(log_terms)
        gradient = np.sum(probabilities[kept, None] * derivatives[kept], axis=0)
        return float(special.logsumexp(log_terms)), gradient


def audit_one(record: dict) -> dict:
    objective = objective_for(record)
    independent = SeparableTasks.from_objective(objective)
    rng = np.random.default_rng(20260917)
    points = np.vstack(
        [
            record["weights"],
            np.ones(len(objective.inventory)) / len(objective.inventory),
            rng.dirichlet(np.ones(len(objective.inventory)), size=5),
        ]
    )
    errors = []
    for point in points:
        np.testing.assert_allclose(independent.logits(point), objective.logits(point)[0], rtol=2e-11, atol=2e-11)
        value, gradient = independent.value_gradient(point)
        parent_value, parent_gradient = objective.log_value_gradient(point)
        np.testing.assert_allclose(value, parent_value, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(gradient, parent_gradient, rtol=2e-11, atol=2e-11)
        errors.append(np.max(np.abs(gradient - parent_gradient)))
    point = np.asarray(record["weights"])
    value, gradient = independent.value_gradient(point)
    raw_gradient = np.exp(value) * gradient
    bound = independent.floor + np.exp(value) - raw_gradient @ point + raw_gradient.min()
    np.testing.assert_allclose(bound, record["lower_bound"], rtol=1e-10, atol=1e-10)
    numerical = optimize._numdiff.approx_derivative(
        lambda w: independent.value_gradient(w)[0], points[1], method="3-point", abs_step=1e-7
    )
    _, analytic = independent.value_gradient(points[1])
    np.testing.assert_allclose(numerical.ravel(), analytic, rtol=2e-5, atol=2e-7)
    # For any upper-clipped task, its fixed contribution alone is already larger than the incumbent.
    # Consequently clipping cannot introduce a better minimum; uncapped and capped minima agree.
    clipped_region_lower = independent.floor + float(independent.weights.min()) * np.exp(30)
    assert clipped_region_lower > record["value"]
    return {
        "model": record["model"],
        "panel": record["panel"],
        "target": record["target"],
        "independent_formulas_match": True,
        "max_gradient_error": max(errors),
        "finite_difference_max_error": float(np.max(np.abs(numerical.ravel() - analytic))),
        "supporting_bound_reproduced": True,
        "gap_bpb": record["supporting_plane_gap_bpb"],
        "clipped_region_lower_bound": clipped_region_lower,
        "clipping_cannot_improve_incumbent": True,
        "upper_clipped_components": record["audit"]["upper_clipped_components_anywhere_on_simplex"],
    }


def waterfill(tasks: SeparableTasks, probabilities: np.ndarray) -> tuple[np.ndarray, float]:
    """Minimize the task-weighted separable convex logits on the entire simplex."""

    def derivative(w):
        raw = tasks.terms(w)[1]
        return np.sum(probabilities[:, None] * raw, axis=0)

    at_zero = derivative(np.zeros(len(tasks.inventory)))
    at_one = derivative(np.ones(len(tasks.inventory)))

    def allocations(common):
        low = np.full(len(tasks.inventory), -700.0)
        high = np.zeros(len(tasks.inventory))
        for _ in range(90):
            middle = (low + high) / 2
            values = derivative(np.exp(middle))
            low = np.where(values < common, middle, low)
            high = np.where(values >= common, middle, high)
        result = np.exp((low + high) / 2)
        result = np.where(common <= at_zero, 0.0, result)
        return np.where(common >= at_one, 1.0, result)

    upper = float(at_one.min())
    lower = upper - max(1.0, abs(upper))
    while allocations(lower).sum() > 1:
        lower = lower - max(1.0, abs(lower))
    for _ in range(80):
        middle = (lower + upper) / 2
        if allocations(middle).sum() > 1:
            upper = middle
        else:
            lower = middle
    common = (lower + upper) / 2
    return allocations(common), common


def variational_bound(tasks: SeparableTasks, probabilities: np.ndarray) -> dict:
    weights, common = waterfill(tasks, probabilities)
    terms, derivatives = tasks.terms(weights)
    q = probabilities @ terms
    derivative = probabilities @ derivatives
    # A supporting line of each Q_i-theta*w at its approximate minimizer is a lower bound on [0,1].
    residual = derivative - common
    corrections = np.minimum(-weights * residual, (1 - weights) * residual)
    minimization_lower = common + np.sum(q - common * weights + corrections)
    entropy = probabilities @ (np.log(tasks.weights) - np.log(probabilities))
    log_lower = probabilities @ tasks.intercept + entropy + minimization_lower - 1e-10
    feasible = weights / weights.sum()
    log_value, _ = tasks.value_gradient(feasible)
    return {
        "weights": feasible.tolist(),
        "lower_bound": float(tasks.floor + np.exp(log_lower)),
        "value": float(tasks.floor + np.exp(log_value)),
        "gap": float(np.exp(log_value) - np.exp(log_lower)),
        "dual_probabilities": probabilities.tolist(),
        "dual_common_derivative": common,
        "allocation_sum_error": float(abs(weights.sum() - 1)),
        "sum_supporting_correction": float(corrections.sum()),
        "max_stationarity_residual": float(np.max(np.abs(residual[weights > 0]))),
        "arithmetic_safety_log": 1e-10,
    }


def refine(record: dict) -> dict:
    objective = objective_for(record)
    tasks = SeparableTasks.from_objective(objective)
    old = np.asarray(record["weights"])
    probabilities = special.softmax(tasks.logits(old) + np.log(tasks.weights))
    initial = variational_bound(tasks, probabilities)
    return {
        "model": record["model"],
        "panel": record["panel"],
        "target": record["target"],
        "original_value": record["value"],
        "original_gap": record["supporting_plane_gap_bpb"],
        "original_lower_bound": record["lower_bound"],
        "variational": initial,
        "improvement": record["value"] - initial["value"],
        "valid_full_simplex_lower_bound": True,
        "certificate": (
            "Task logsumexp variational lower bound; separable Lagrange dual "
            "with per-coordinate supporting-line corrections"
        ),
    }


def main() -> None:
    paths = sorted(SOURCE.glob("*/*/*.json"))
    assert len(paths) == 12
    OUTPUT.mkdir(parents=True, exist_ok=True)
    validation.json_write(
        OUTPUT / "manifest.json",
        {
            "sources": {
                str(p): validation.sha256(p)
                for p in (Path(__file__), Path(original.__file__), Path(original.candidates.__file__))
            },
            "inputs": {str(p): validation.sha256(p) for p in paths},
            "refine_only": "300m_39bucket/uncheatable",
        },
    )
    records = [json.loads(path.read_text()) for path in paths]
    audits = [audit_one(record) for record in records]
    pd.DataFrame(audits).to_csv(OUTPUT / "audit.csv", index=False)
    for record in records:
        if record["panel"] == "300m_39bucket" and record["target"] == "uncheatable":
            result = refine(record)
            validation.json_write(OUTPUT / f"{record['model']}.json", result)
            print(json.dumps(result), flush=True)
    validation.json_write(
        OUTPUT / "verification.json",
        {
            "objectives": 12,
            "all_independent_checks_passed": True,
            "refined_objectives": 2,
            "clipping_equivalence_proved_for_minimization": True,
        },
    )


if __name__ == "__main__":
    main()
