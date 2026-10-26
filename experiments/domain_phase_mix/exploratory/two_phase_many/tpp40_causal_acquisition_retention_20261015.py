# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""Pure cumulative acquisition/retention hypothesis, with no fitting or data I/O.

The primary evaluator uses positive integrating-factor quadrature on the actual
piecewise-linear epoch path. Paired-order rules adapt until their estimated relative
integral error meets the requested tolerance; this is not an interval proof.
Independent acquisition/forgetting ordering bounds are supplied as a slower
check. Group labels, masses and numerical controls are explicit inputs. No
semantic partition, kinetic grid, empirical geometry or estimator is selected.

At fixed aggregate, the terminal readout is native benefit plus
exp(-R_tied)*(m_path-m_tied). It attenuates the retained-mastery difference;
the equivalent positive S_path+R_tied computation avoids saturated subtraction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

RULES = {order: leggauss(order) for order in (16, 32)}
LOG_HALF = np.log(0.5)


@dataclass(frozen=True)
class Law:
    rate: float
    power: float
    mu: float
    groups: tuple[str, ...]
    masses: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite([self.rate, self.power, self.mu]).all():
            raise ValueError("Law parameters must be finite")
        if self.rate <= 0 or not 0 < self.power <= 1 or self.mu < 0:
            raise ValueError("Require rate>0, power in (0,1], and mu>=0")
        if self.masses.shape != (len(self.groups),) or not len(self.groups):
            raise ValueError("One fixed mass and group are required per bucket")
        if np.any(self.masses < 0) or not np.isfinite(self.masses).all():
            raise ValueError("Fixed masses must be finite and nonnegative")


@dataclass(frozen=True)
class Quadrature:
    relative_tolerance: float
    max_panels: int

    def __post_init__(self) -> None:
        if not 0 < self.relative_tolerance < 1 or self.max_panels < 1:
            raise ValueError("Specify a positive relative tolerance and panel budget")


@dataclass(frozen=True)
class State:
    epochs: np.ndarray
    log_mastery: np.ndarray
    log_excess_survival: np.ndarray
    evaluations: int
    panels: int
    maximum_relative_error_estimate: float

    def survival_hazard(self, law: Law) -> np.ndarray:
        """Return -log(q), choosing the numerically stable complement."""
        validate_state(self)
        hazard = native_hazard(self.epochs, law)
        log_survival = np.logaddexp(-hazard, self.log_excess_survival)
        result = -log_survival
        small = self.log_mastery < LOG_HALF
        result[small] = -np.log1p(-np.exp(self.log_mastery[small]))
        return result

    def deficit(self, law: Law) -> np.ndarray:
        """Return H-S without subtracting nearly equal large hazards."""
        validate_state(self)
        return np.logaddexp(0.0, native_hazard(self.epochs, law) + self.log_excess_survival)


@dataclass(frozen=True)
class PolicyResult:
    benefit: np.ndarray
    effective_hazard: np.ndarray
    path: State
    tied: State
    exact_native_branch: bool


@dataclass(frozen=True)
class DerivativeResult:
    derivative: np.ndarray
    path_integral: np.ndarray
    tied_integral: np.ndarray
    evaluations: int


@dataclass(frozen=True)
class Bounds:
    lower: np.ndarray
    upper: np.ndarray
    cells_per_phase: int


@dataclass(frozen=True)
class Integral:
    value: np.ndarray
    relative_error: float
    evaluations: int
    panels: int


def validate_state(state: State) -> None:
    """Reject nonfinite physical epochs or corrupt positive/logarithmic state."""
    if state.epochs.ndim != 1 or any(
        value.shape != state.epochs.shape for value in (state.log_mastery, state.log_excess_survival)
    ):
        raise ValueError("State arrays must share a one-dimensional bucket axis")
    if not np.isfinite(state.epochs).all() or np.any(state.epochs < 0):
        raise ValueError("State epochs must be finite and nonnegative")
    for value in (state.log_mastery, state.log_excess_survival):
        if np.any(np.isnan(value) | np.isposinf(value)):
            raise FloatingPointError(
                "Log state may contain negative infinity for zero, but not NaN or positive infinity"
            )


def native_hazard(epochs: np.ndarray, law: Law) -> np.ndarray:
    """Evaluate the native physical cumulative hazard."""
    value = (law.rate * epochs) ** law.power
    if not np.isfinite(value).all():
        raise FloatingPointError("Native hazard overflowed")
    return value


def hazard_increment(start: np.ndarray, increment: np.ndarray, law: Law) -> np.ndarray:
    """Avoid subtracting nearly equal cumulative hazards."""
    start, increment = np.broadcast_arrays(start, increment)
    result = native_hazard(increment, law)
    positive = start > 0
    result[positive] = native_hazard(start[positive], law) * np.expm1(
        law.power * np.log1p(increment[positive] / start[positive])
    )
    return result


def log_acquired(hazard: np.ndarray) -> np.ndarray:
    """Log of 1-exp(-H), retaining zero and tiny acquisition."""
    with np.errstate(divide="ignore"):
        return np.log(-np.expm1(-hazard))


def outside_logsum(log_values: np.ndarray, law: Law) -> np.ndarray:
    """Sum fixed outside-group masses in log space."""
    result = np.full_like(log_values, -np.inf)
    with np.errstate(divide="ignore"):
        log_mass = np.log(law.masses)
    for i, group in enumerate(law.groups):
        mask = np.array([other != group for other in law.groups])
        if mask.any():
            result[..., i] = logsumexp(log_values[..., mask] + log_mass[mask], axis=-1)
    return result


def phase_coordinates(
    unit: np.ndarray, start: np.ndarray, increment: np.ndarray, law: Law
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Smooth physical origins by t=u**k, retaining the exact Jacobian."""
    changing = increment > 0
    exponent = 1 if np.all(start[changing] > 0) else int(np.ceil(4 / law.power))
    time = unit[:, None] ** exponent
    epochs = start + time * increment
    hazard = native_hazard(epochs, law)
    log_derivative = np.full_like(epochs, -np.inf)
    if changing.any():
        zero_start = changing & (start == 0)
        log_derivative[:, zero_start] = (
            law.power * np.log(law.rate * increment[zero_start])
            + np.log(law.power * exponent)
            + (law.power * exponent - 1) * np.log(unit[:, None])
        )
        positive_start = changing & (start > 0)
        log_derivative[:, positive_start] = (
            law.power * np.log(law.rate)
            + np.log(law.power * exponent)
            + np.log(increment[positive_start])
            + (exponent - 1) * np.log(unit[:, None])
            + (law.power - 1) * np.log(epochs[:, positive_start])
        )
    return epochs, hazard, log_derivative, time


def positive_log_quadrature(function: Callable[[np.ndarray], np.ndarray], control: Quadrature) -> Integral:
    """Adapt a positive log integral; raise if the requested estimate is unmet."""

    def panel(left: float, right: float) -> tuple[float, float, np.ndarray, np.ndarray]:
        estimates = []
        for order in (16, 32):
            nodes, weights = RULES[order]
            points = left + (right - left) * (nodes + 1) / 2
            logs = function(points)
            if np.any(np.isnan(logs) | np.isposinf(logs)):
                raise FloatingPointError(
                    "Log integrand may contain negative infinity for zero, but not NaN or positive infinity"
                )
            estimates.append(logsumexp(logs + np.log(weights[:, None] * (right - left) / 2), axis=0))
        low, high = estimates
        error = np.full_like(high, -np.inf)
        finite = np.isfinite(high) & np.isfinite(low)
        difference = abs(low[finite] - high[finite])
        with np.errstate(divide="ignore"):
            error[finite] = np.maximum(low[finite], high[finite]) + np.log(-np.expm1(-difference))
        error[np.isfinite(high) & ~np.isfinite(low)] = high[np.isfinite(high) & ~np.isfinite(low)]
        error[np.isfinite(low) & ~np.isfinite(high)] = low[np.isfinite(low) & ~np.isfinite(high)]
        return left, right, high, error

    leaves = [panel(0.0, 1.0)]
    evaluations = 48
    while True:
        total = logsumexp(np.stack([item[2] for item in leaves]), axis=0)
        error = logsumexp(np.stack([item[3] for item in leaves]), axis=0)
        active = np.isfinite(total)
        ratios = np.full_like(total, -np.inf)
        ratios[active] = error[active] - total[active]
        maximum = float(np.max(ratios))
        if maximum <= np.log(control.relative_tolerance):
            return Integral(total, float(np.exp(maximum)), evaluations, len(leaves))
        if len(leaves) >= control.max_panels:
            raise ArithmeticError("Positive quadrature did not converge within its panel budget")
        priorities = [float(np.max(item[3][active] - total[active])) for item in leaves]
        index = int(np.argmax(priorities))
        left, right, _value, _error = leaves.pop(index)
        midpoint = (left + right) / 2
        if midpoint == left or midpoint == right:
            raise ArithmeticError("Quadrature reached floating-point subdivision resolution")
        leaves.extend((panel(left, midpoint), panel(midpoint, right)))
        evaluations += 96


def empty_state(buckets: int) -> State:
    return State(np.zeros(buckets), np.full(buckets, -np.inf), np.full(buckets, -np.inf), 0, 0, 0.0)


def advance(state: State, increment: np.ndarray, law: Law, control: Quadrature) -> State:
    """Propagate a causal prefix through one constant physical allocation segment."""
    validate_state(state)
    increment = np.asarray(increment, dtype=float)
    if increment.shape != state.epochs.shape or increment.shape != law.masses.shape:
        raise ValueError("State, increment and fixed bucket axis differ")
    if np.any(increment < 0) or not np.isfinite(increment).all():
        raise ValueError("Epoch increments must be finite and nonnegative")
    start = state.epochs
    final = start + increment
    initial_hazard = native_hazard(start, law)
    delta_hazard = hazard_increment(start, increment, law)
    log_delta_j = outside_logsum(-initial_hazard + log_acquired(delta_hazard), law)
    log_k = log_delta_j + np.log(law.mu) if law.mu else np.full_like(log_delta_j, -np.inf)
    kinetic = np.exp(log_k)
    decay = delta_hazard + kinetic
    if not np.isfinite(decay).all():
        raise FloatingPointError("Combined acquisition/interference clock overflowed")
    log_m, log_d = state.log_mastery.copy(), state.log_excess_survival.copy()
    acquisition_only = ~np.isfinite(log_k)
    forgetting_only = (delta_hazard == 0) & ~acquisition_only
    active = ~(acquisition_only | forgetting_only)
    log_q = np.logaddexp(-initial_hazard, log_d)
    log_m[acquisition_only] = np.logaddexp(
        log_m[acquisition_only], log_q[acquisition_only] + log_acquired(delta_hazard[acquisition_only])
    )
    log_d[acquisition_only] -= delta_hazard[acquisition_only]
    log_one_minus_decay = log_k.copy()
    representable = log_k > -36
    log_one_minus_decay[representable] = log_acquired(kinetic[representable])
    log_d[forgetting_only] = np.logaddexp(
        log_d[forgetting_only], log_m[forgetting_only] + log_one_minus_decay[forgetting_only]
    )
    log_m[forgetting_only] -= kinetic[forgetting_only]
    evaluations, panels, estimate = 0, 0, 0.0
    if active.any():

        def integrand(unit: np.ndarray) -> np.ndarray:
            epochs, hazard, log_dh, time = phase_coordinates(unit, start, increment, law)
            remaining = (1 - time) * increment
            delta = hazard_increment(epochs, remaining, law)
            log_remaining_j = outside_logsum(-hazard + log_acquired(delta), law)
            attenuation = delta + np.exp(np.log(law.mu) + log_remaining_j)
            log_dj = outside_logsum(-hazard + log_dh, law)
            mastery = (log_dh - attenuation)[:, active]
            excess = (np.log(law.mu) + log_dj + log_acquired(hazard) - attenuation)[:, active]
            return np.hstack([mastery, excess])

        integral = positive_log_quadrature(integrand, control)
        count = int(active.sum())
        log_m[active] = np.logaddexp(log_m[active] - decay[active], integral.value[:count])
        log_d[active] = np.logaddexp(log_d[active] - decay[active], integral.value[count:])
        evaluations, panels, estimate = integral.evaluations, integral.panels, integral.relative_error
    result = State(
        final,
        log_m,
        log_d,
        state.evaluations + evaluations,
        state.panels + panels,
        max(state.maximum_relative_error_estimate, estimate),
    )
    conservation = np.exp(log_m) + np.exp(np.logaddexp(-native_hazard(final, law), log_d))
    if not np.isfinite(conservation).all() or (
        np.max(abs(conservation - 1)) > 20 * control.relative_tolerance + 256 * np.finfo(float).eps
    ):
        raise ArithmeticError("Independent mastery/survival integrals do not conserve probability")
    return result


def integrate(phase_epochs: np.ndarray, law: Law, control: Quadrature) -> State:
    """Integrate a whole path, retaining prefix state rather than final-normalized rates."""
    phase_epochs = np.asarray(phase_epochs, dtype=float)
    if phase_epochs.ndim != 2 or phase_epochs.shape[1] != len(law.groups):
        raise ValueError("Expected a phase-by-bucket epoch matrix")
    state = empty_state(len(law.groups))
    for increment in phase_epochs:
        state = advance(state, increment, law, control)
    return state


def policy_epochs(weights: np.ndarray, fractions: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Materialize an explicit simplex policy into physical phase epochs."""
    weights, fractions, scales = (np.asarray(value, dtype=float) for value in (weights, fractions, scales))
    if weights.ndim != 2 or weights.shape != (len(fractions), len(scales)):
        raise ValueError("Policy dimensions differ")
    if any(not np.isfinite(value).all() for value in (weights, fractions, scales)):
        raise ValueError("Policy covariates must be finite")
    if np.any(weights < 0) or np.any(scales < 0) or np.any(fractions <= 0):
        raise ValueError("Require nonnegative weights/scales and positive phase fractions")
    if not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-13) or not np.isclose(
        fractions.sum(), 1, rtol=0, atol=1e-13
    ):
        raise ValueError("Policy and phase fractions must be simplexes")
    return fractions[:, None] * weights * scales


def evaluate_policy(
    weights: np.ndarray, fractions: np.ndarray, scales: np.ndarray, law: Law, control: Quadrature
) -> PolicyResult:
    """Return the terminal counterfactual readout; it is not a prefix state."""
    epochs = policy_epochs(weights, fractions, scales)
    total = epochs.sum(axis=0)
    tied = integrate(total[None], law, control)
    exact_tied = np.array_equal(weights, np.broadcast_to(weights[0], weights.shape))
    if law.mu == 0 or exact_tied:
        hazard = native_hazard(total, law)
        return PolicyResult(-np.expm1(-hazard), hazard, tied, tied, True)
    path = integrate(epochs, law, control)
    hazard = path.survival_hazard(law) + tied.deficit(law)
    return PolicyResult(-np.expm1(-hazard), hazard, path, tied, False)


def ordering_states(phase_epochs: np.ndarray, law: Law, cells_per_phase: int) -> tuple[State, State]:
    """Bound mastery using opposite acquisition/forgetting orderings per cell.

    The inequalities are exact over the real numbers. Floating-point evaluation
    is not a formally outward-rounded interval certificate. Width is generally
    first order in cell size, so this is a reference check, not the fast route.
    """
    if cells_per_phase < 1:
        raise ValueError("Ordering bounds need at least one cell per phase")
    lower, upper = empty_state(len(law.groups)), empty_state(len(law.groups))
    for increment in np.asarray(phase_epochs, dtype=float):
        origin = lower.epochs.copy()
        times = np.linspace(0, 1, cells_per_phase + 1) ** (1 / law.power)
        for left, right in pairwise(times):
            start, step = origin + left * increment, (right - left) * increment
            hazard = native_hazard(start, law)
            delta = hazard_increment(start, step, law)
            log_j = outside_logsum(-hazard + log_acquired(delta), law)
            log_k = np.log(law.mu) + log_j if law.mu else np.full_like(log_j, -np.inf)
            kinetic = np.exp(log_k)
            log_loss = log_k.copy()
            ordinary = log_k > -36
            log_loss[ordinary] = log_acquired(kinetic[ordinary])
            states = []
            for state, order in ((lower, "acquire_then_forget"), (upper, "forget_then_acquire")):
                log_m, log_d = state.log_mastery.copy(), state.log_excess_survival.copy()
                if order == "acquire_then_forget":
                    log_q = np.logaddexp(-hazard, log_d)
                    log_m = np.logaddexp(log_m, log_q + log_acquired(delta))
                    log_d -= delta
                    log_d = np.logaddexp(log_d, log_m + log_loss)
                    log_m -= kinetic
                else:
                    log_d = np.logaddexp(log_d, log_m + log_loss)
                    log_m -= kinetic
                    log_q = np.logaddexp(-hazard, log_d)
                    log_m = np.logaddexp(log_m, log_q + log_acquired(delta))
                    log_d -= delta
                states.append(State(origin + right * increment, log_m, log_d, 0, cells_per_phase, 0.0))
            lower, upper = states
    return lower, upper


def policy_ordering_bounds(
    weights: np.ndarray, fractions: np.ndarray, scales: np.ndarray, law: Law, cells_per_phase: int
) -> Bounds:
    """Bound the benefit readout using positive S(path)+R(tied)."""
    epochs = policy_epochs(weights, fractions, scales)
    low_path, high_path = ordering_states(epochs, law, cells_per_phase)
    low_tied, high_tied = ordering_states(epochs.sum(axis=0)[None], law, cells_per_phase)
    lower = low_path.survival_hazard(law) + high_tied.deficit(law)
    upper = high_path.survival_hazard(law) + low_tied.deficit(law)
    return Bounds(-np.expm1(-lower), -np.expm1(-upper), cells_per_phase)


def zero_mu_integral(phase_epochs: np.ndarray, law: Law, control: Quadrature) -> Integral:
    """Integrate K=integral exp(-(H_T-H))*b(H) dJ directly, with no exp(H)."""
    phase_epochs = np.asarray(phase_epochs, dtype=float)
    if phase_epochs.ndim != 2 or phase_epochs.shape[1] != len(law.groups):
        raise ValueError("Expected a phase-by-bucket epoch matrix")
    if not np.isfinite(phase_epochs).all() or np.any(phase_epochs < 0):
        raise ValueError("Epoch increments must be finite and nonnegative")
    future = np.zeros_like(phase_epochs)
    for index in range(len(phase_epochs) - 2, -1, -1):
        future[index] = future[index + 1] + phase_epochs[index + 1]
    start = np.zeros(phase_epochs.shape[1])
    logs = []
    evaluations, panels, error = 0, 0, 0.0
    for increment, suffix in zip(phase_epochs, future, strict=True):

        def integrand(
            unit: np.ndarray,
            start: np.ndarray = start,
            increment: np.ndarray = increment,
            suffix: np.ndarray = suffix,
        ) -> np.ndarray:
            epochs, hazard, log_dh, time = phase_coordinates(unit, start, increment, law)
            remaining = suffix + (1 - time) * increment
            remaining_hazard = hazard_increment(epochs, remaining, law)
            return -remaining_hazard + log_acquired(hazard) + outside_logsum(-hazard + log_dh, law)

        integral = positive_log_quadrature(integrand, control)
        logs.append(integral.value)
        evaluations += integral.evaluations
        panels += integral.panels
        error = max(error, integral.relative_error)
        start = start + increment
    return Integral(logsumexp(np.stack(logs), axis=0), error, evaluations, panels)


def zero_mu_response(
    weights: np.ndarray, fractions: np.ndarray, scales: np.ndarray, law: Law, control: Quadrature
) -> DerivativeResult:
    """Return the stable benefit derivative K(tied)-K(path) at zero kinetics."""
    epochs = policy_epochs(weights, fractions, scales)
    path = zero_mu_integral(epochs, law, control)
    tied = zero_mu_integral(epochs.sum(axis=0)[None], law, control)
    difference = np.zeros(len(law.groups))
    for i, (first, second) in enumerate(zip(tied.value, path.value, strict=True)):
        if first > second:
            difference[i] = np.exp(first) * -np.expm1(second - first)
        elif second > first:
            difference[i] = -np.exp(second) * -np.expm1(first - second)
    if np.array_equal(weights, np.broadcast_to(weights[0], weights.shape)):
        difference[:] = 0
    return DerivativeResult(difference, np.exp(path.value), np.exp(tied.value), path.evaluations + tied.evaluations)
