# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "threadpoolctl"]
# ///
"""Pure fixed-aggregate phase bounds for TP40-028's physical-harm law.

This module reads no artifacts, fits no coefficients, and optimizes no nonlinear
policy. Its only minimization is a linear capacitated-knapsack problem. Bounds
are floating-point diagnostics, not interval-arithmetic certificates. They say
nothing about changing aggregate exposure or changing the fitted head.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_late_refresh_20261015 as law

LOGIT_LIMIT = 30.0


def normalized_exposure_cost(c0: np.ndarray, c1: np.ndarray, alpha: float) -> np.ndarray:
    """Derive pool/reference-budget costs from frozen phase exposure scales.

    If E=c0*w0+c1*w1, c0=alpha*K and c1=(1-alpha)*K,
    then sum E/K=alpha*sum w0+(1-alpha)*sum w1=1. These are
    exposure-reference token fractions, not actual proxy-training token counts.
    """
    c0, c1 = np.asarray(c0, dtype=float), np.asarray(c1, dtype=float)
    if c0.ndim != 1 or c0.shape != c1.shape or not np.isfinite([alpha]).all() or not 0 < alpha < 1:
        raise ValueError("Require matching phase-scale vectors and alpha in (0,1)")
    if not np.isfinite(c0).all() or not np.isfinite(c1).all() or np.any(c0 <= 0) or np.any(c1 <= 0):
        raise ValueError("Phase exposure scales must be finite and positive")
    total_scale = c0 + c1
    if not np.allclose(c0 / total_scale, alpha, atol=64 * np.finfo(float).eps, rtol=0):
        raise ValueError("Frozen scales do not have a common phase fraction")
    return 1.0 / total_scale


@dataclass(frozen=True)
class TaskHead:
    """One fixed physical-harm head; amplitudes follow native nonnegative signs."""

    rate: float
    power: float
    threshold: float
    strength: float
    intercept: float
    benefit: np.ndarray
    harm: np.ndarray
    floor: float


@dataclass(frozen=True)
class TaskPoint:
    """Smooth logit and gradient, plus native clipped prediction at one point."""

    logit: float
    native_loss: float
    logit_gradient: np.ndarray
    singular_indices: tuple[int, ...]
    logit_box_upper: float


@dataclass(frozen=True)
class LinearMinimum:
    """Greedy primal and Lagrange dual values for a box-and-budget linear program."""

    allocation: np.ndarray
    primal_value: float
    dual_value: float
    dual_multiplier: float
    primal_minus_dual: float
    budget_residual: float


def task_point(head: TaskHead, total: np.ndarray, late: np.ndarray, beta: float) -> TaskPoint:
    """Evaluate the smooth TP028 logit and its partial derivative in late dose.

    Total dose is held fixed. Active zero-late coordinates with power below one
    have derivative minus infinity; no finite derivative floor is substituted.
    The frozen implementation's roundoff-only tied-row snapping is intentionally
    absent from this mathematical derivative. Values agree to floating precision.
    """
    total, late = np.asarray(total, dtype=float), np.asarray(late, dtype=float)
    benefit, harm = np.asarray(head.benefit, dtype=float), np.asarray(head.harm, dtype=float)
    parameters = [head.rate, head.power, head.threshold, head.strength, head.intercept, head.floor, beta]
    if not np.isfinite(parameters).all() or not 0 < beta < 1:
        raise ValueError("Require finite parameters and beta in (0,1)")
    if head.rate <= 0 or not 0 < head.power <= 1 or not 0 <= head.strength <= 1:
        raise ValueError("Require positive rate, power in (0,1], strength in [0,1]")
    if total.ndim != 1 or any(value.shape != total.shape for value in (late, benefit, harm)):
        raise ValueError("Exposure and coefficient vectors must have one common shape")
    if not all(np.isfinite(value).all() for value in (total, late, benefit, harm)):
        raise ValueError("Exposure and coefficients must be finite")
    if np.any(late < 0) or np.any(late > total) or np.any(benefit < 0) or np.any(harm < 0):
        raise ValueError("Require 0 <= late <= total and nonnegative amplitudes")

    hazard = (head.rate * total) ** head.power
    denominator = law.mariner.benefit(beta * total, head.rate, head.power)
    active = (total > 0) & (benefit > 0) & (head.strength > 0)
    if not np.isfinite(hazard).all() or np.any(active & ((hazard == 0) | (denominator == 0))):
        raise FloatingPointError("Active hazard or normalization is outside the floating-point domain")
    refresh = np.divide(head.strength * hazard, denominator, out=np.zeros_like(total), where=denominator > 0)
    baseline_hazard = (1 - head.strength) * hazard
    effective_hazard = baseline_hazard + refresh * law.mariner.benefit(late, head.rate, head.power)
    physical_harm = law.mariner.harm(total, head.threshold)
    constant = head.intercept + float(harm @ physical_harm)
    logit = constant + float(benefit @ np.expm1(-effective_hazard))
    box_upper = constant + float(benefit @ np.expm1(-baseline_hazard))
    if not np.isfinite([logit, box_upper]).all():
        raise FloatingPointError("Logit or its fixed-aggregate box upper bound is nonfinite")

    gradient = np.zeros_like(total)
    positive = active & (late > 0)
    with np.errstate(over="ignore", under="ignore"):
        log_magnitude = (
            np.log(benefit[positive])
            + np.log(refresh[positive])
            + np.log(head.power)
            + head.power * np.log(head.rate)
            + (head.power - 1) * np.log(late[positive])
            - (head.rate * late[positive]) ** head.power
            - effective_hazard[positive]
        )
        gradient[positive] = -np.exp(log_magnitude)
    if not np.isfinite(gradient[positive]).all():
        raise FloatingPointError("Interior derivative is outside the floating-point domain")
    zero_active = active & (late == 0)
    singular = zero_active & (head.power < 1)
    gradient[singular] = -np.inf
    if head.power == 1:
        gradient[zero_active] = (
            -benefit[zero_active] * refresh[zero_active] * head.rate * np.exp(-effective_hazard[zero_active])
        )
    return TaskPoint(
        logit=logit,
        native_loss=float(head.floor + np.exp(np.clip(logit, -LOGIT_LIMIT, LOGIT_LIMIT))),
        logit_gradient=gradient,
        singular_indices=tuple(int(index) for index in np.flatnonzero(singular)),
        logit_box_upper=box_upper,
    )


def linear_minimum(gradient: np.ndarray, upper: np.ndarray, cost: np.ndarray, budget: float) -> LinearMinimum:
    """Minimize g dot z with 0 <= z <= upper and cost dot z = budget.

    The dual is mu*budget + sum upper_i*min(0,g_i-mu*cost_i). Any finite mu
    supplies a lower bound independently of the greedy primal's roundoff error.
    Equal marginal costs are ordered by their original coordinate index.
    """
    gradient, upper, cost = (np.asarray(value, dtype=float) for value in (gradient, upper, cost))
    if gradient.ndim != 1 or upper.shape != gradient.shape or cost.shape != gradient.shape:
        raise ValueError("Require matching one-dimensional gradient, capacities, and costs")
    if not all(np.isfinite(value).all() for value in (gradient, upper, cost)) or not np.isfinite(budget):
        raise ValueError("The linear problem must be finite")
    if np.any(upper < 0) or np.any(cost <= 0) or not 0 <= budget <= float(cost @ upper):
        raise ValueError("The linear budget must lie within nonnegative box capacity")
    ratio = gradient / cost
    order = np.argsort(ratio, kind="stable")
    allocation = np.zeros_like(upper)
    remaining = budget
    multiplier = float(np.min(ratio)) if len(ratio) else 0.0
    for index in order:
        if remaining <= 0:
            break
        multiplier = float(ratio[index])
        tokens = min(remaining, float(cost[index] * upper[index]))
        allocation[index] = tokens / cost[index]
        remaining -= tokens
    primal = float(gradient @ allocation)
    dual = float(multiplier * budget + upper @ np.minimum(0.0, gradient - multiplier * cost))
    return LinearMinimum(
        allocation=allocation,
        primal_value=primal,
        dual_value=dual,
        dual_multiplier=multiplier,
        primal_minus_dual=primal - dual,
        budget_residual=float(cost @ allocation - budget),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def conditional_diagnostic(
    heads: tuple[TaskHead, ...],
    weights: np.ndarray,
    total: np.ndarray,
    late: np.ndarray,
    cost: np.ndarray,
    beta: float,
    total_budget: float,
    *,
    interior_fraction: float,
    feasibility_tolerance: float,
) -> dict[str, Any]:
    """Bound remaining phase-placement improvement, keeping total and heads fixed.

    A singular reference is blended once toward beta*total by the specified
    fraction. No policy search, coefficient update, or feasibility repair occurs.
    An original point outside the declared feasibility tolerance returns no gap
    bound. All values remain numerical estimates rather than rigorous certificates.
    """
    total, late, cost, weights = (np.asarray(value, dtype=float) for value in (total, late, cost, weights))
    if total.ndim != 1 or late.shape != total.shape or cost.shape != total.shape:
        raise ValueError("Require matching exposure and cost vectors")
    if len(heads) == 0 or weights.shape != (len(heads),):
        raise ValueError("Every task needs one explicit objective weight")
    if not all(np.isfinite(value).all() for value in (total, late, cost, weights)):
        raise ValueError("All exposures, costs, and weights must be finite")
    if np.any(cost <= 0) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Require positive costs and nonnegative, nonzero task weights")
    if not np.isfinite([beta, total_budget, interior_fraction, feasibility_tolerance]).all():
        raise ValueError("Diagnostic parameters must be finite")
    if not 0 < beta < 1 or total_budget <= 0 or not 0 < interior_fraction <= 1 or feasibility_tolerance < 0:
        raise ValueError("Invalid phase fraction, budget, interior fraction, or tolerance")
    aggregate_residual = float(cost @ total - total_budget)
    late_budget = beta * total_budget
    late_residual = float(cost @ late - late_budget)
    box_feasible = bool(np.all(total >= 0) and np.all(late >= 0) and np.all(late <= total))
    feasible = (
        box_feasible
        and abs(aggregate_residual) <= feasibility_tolerance * total_budget
        and abs(late_residual) <= feasibility_tolerance * total_budget
        and late_budget <= float(cost @ total)
    )
    result: dict[str, Any] = {
        "scope": "fixed aggregate and fixed heads; no outer optimum certificate",
        "numerics": "floating-point diagnostic, not an interval-arithmetic certificate",
        "reference_convention": "smooth unsnapped logit; native loss clips logit to [-30,30]",
        "objective_weights": weights,
        "total": total,
        "original_late": late,
        "cost": cost,
        "total_budget": total_budget,
        "late_budget": late_budget,
        "aggregate_budget_residual": aggregate_residual,
        "original_late_budget_residual": late_residual,
        "box_feasible": box_feasible,
        "feasible_at_tolerance": feasible,
        "feasibility_tolerance_relative_total_budget": feasibility_tolerance,
        "bounds_available": False,
    }
    if not feasible:
        result["unavailable_reason"] = "Original policy violates box or declared budget tolerance; no repair"
        return _jsonable(result)

    original = [task_point(head, total, late, beta) for head in heads]
    needs_interior = any(point.singular_indices for point in original)
    reference = (1 - interior_fraction) * late + interior_fraction * beta * total if needs_interior else late.copy()
    points = [task_point(head, total, reference, beta) for head in heads]
    if any(point.singular_indices for point in points):
        raise FloatingPointError("Declared interiorization did not resolve a singular derivative")
    task_results = []
    task_lower_bounds = []
    for head, original_point, point in zip(heads, original, points, strict=True):
        linear = linear_minimum(point.logit_gradient, total, cost, late_budget)
        lower_logit = point.logit + linear.dual_value - float(point.logit_gradient @ reference)
        native_lower = float(head.floor + np.exp(np.clip(lower_logit, -LOGIT_LIMIT, LOGIT_LIMIT)))
        task_lower_bounds.append(native_lower)
        task_results.append(
            {
                "original": asdict(original_point),
                "reference": asdict(point),
                "linear": asdict(linear),
                "lower_logit": lower_logit,
                "native_loss_lower_bound": native_lower,
                "original_native_minus_lower": original_point.native_loss - native_lower,
            }
        )

    active_weights = weights > 0
    upper_cap_excluded = all(
        point.logit_box_upper <= LOGIT_LIMIT for point, active in zip(points, active_weights, strict=True) if active
    )
    native_original = float(weights @ np.array([point.native_loss for point in original]))
    native_reference = float(weights @ np.array([point.native_loss for point in points]))
    taskwise_lower = float(weights @ np.asarray(task_lower_bounds))
    weighted: dict[str, Any] = {
        "reference_convention": "unclipped exp(logit) value and gradient; no clipped-gradient substitution",
        "upper_cap_excluded_over_full_box": upper_cap_excluded,
        "native_original_value": native_original,
        "native_reference_value": native_reference,
        "native_taskwise_lower_bound": taskwise_lower,
        "unclipped_support_available": False,
        "unclipped_support_is_native_lower_bound": False,
    }
    active = [(head, point, weight) for head, point, weight in zip(heads, points, weights, strict=True) if weight > 0]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        deficits = np.exp(np.array([point.logit for _, point, _ in active]))
        unclipped_value = float(
            sum(weight * (head.floor + deficit) for (head, _, weight), deficit in zip(active, deficits, strict=True))
        )
        loss_gradient = sum(
            (
                weight * deficit * point.logit_gradient
                for (_, point, weight), deficit in zip(active, deficits, strict=True)
            ),
            start=np.zeros_like(total),
        )
    if np.isfinite(unclipped_value) and np.isfinite(loss_gradient).all():
        loss_linear = linear_minimum(loss_gradient, total, cost, late_budget)
        unclipped_lower = unclipped_value + loss_linear.dual_value - float(loss_gradient @ reference)
        weighted.update(
            {
                "unclipped_support_available": True,
                "unclipped_reference_value": unclipped_value,
                "unclipped_reference_gradient": loss_gradient,
                "unclipped_linear": asdict(loss_linear),
                "unclipped_lower_bound": unclipped_lower,
                "unclipped_support_is_native_lower_bound": upper_cap_excluded,
            }
        )
        native_lower = max(taskwise_lower, unclipped_lower) if upper_cap_excluded else taskwise_lower
    else:
        weighted["unclipped_unavailable_reason"] = "Unclipped value or gradient overflows floating-point range"
        native_lower = taskwise_lower
    weighted["native_combined_lower_bound"] = native_lower
    weighted["native_remaining_improvement_upper_estimate"] = native_original - native_lower
    result.update(
        {
            "bounds_available": True,
            "interiorization_applied": needs_interior,
            "interior_fraction": interior_fraction if needs_interior else 0.0,
            "reference_late": reference,
            "reference_late_budget_residual": float(cost @ reference - late_budget),
            "task_results": task_results,
            "weighted": weighted,
        }
    )
    return _jsonable(result)
