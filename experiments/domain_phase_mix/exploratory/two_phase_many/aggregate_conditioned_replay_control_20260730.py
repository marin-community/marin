# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate response plus replay-limited phase control.

Let ``a = beta0 * w0 + beta1 * w1`` and ``delta = w1 - w0``. The surrogate is

    L(a, delta) = A(a)
        + xi * P_q(a) * grad A(a)^T delta
        + gamma * I_beta(a, delta)
        + zeta * J(a, delta).

``A`` is the retained-power-law model's hierarchical aggregate backbone: a
power-law benefit in aggregate token share plus repetition damage in aggregate
materialized epochs. It is identified only from tied policies.

``P_q(a) = R(a)^(-q)`` is a dimensionless plasticity factor. The replay pressure

    R(a) = sum_i k_i a_i^2 / min_p sum_i k_i p_i^2

is the expected materialized epoch count of a sampled token, normalized to one
at the token-proportional policy. Concentrating weight on small pools raises
``R`` and reduces the phase-control leverage when ``q > 0``.

``I_beta`` is the Fisher-quadratic phase-information cost. ``J`` is the Jensen
gap of a convex phase-local replay-rate cost. Both are dimensionless,
nonnegative, and exactly zero for tied policies; only ``I_beta`` is exactly
even under contrast reversal when the phase fractions differ.

The follow-up curvature candidate replaces both generic costs with

    delta^T H_A(a) delta.

This is the second directional derivative of the fitted tied response. It is
nonnegative for this convex aggregate backbone, vanishes when tied, and supplies
the second-order term omitted by the linear control approximation.

The control-energy candidate instead adds

    omega * (grad A(a)^T delta)^2.

This is the second-order response in the one-dimensional control coordinate
already identified by the aggregate model. It limits finite phase actions
without introducing a separate direction or an aggregate-specific calibration.

All three phase amplitudes are nonnegative. At an interior tied optimum of
``A``, ``grad A`` is constant on the simplex and therefore orthogonal to every
feasible contrast. The ordering term vanishes while both even terms remain
nonnegative, so the fitted tied optimum cannot be improved anywhere on its
phase fiber. Away from that optimum, the aggregate gradient supplies a signed
phase-control direction and the model can prefer a genuinely two-phase policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import lsq_linear

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    aggregate_conditioned_phase_control_model_20260729 as aggregate_control,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import retained_power_law_model_20260728 as retained

BENEFIT_EXPONENTS = retained.BENEFIT_EXPONENTS
BENEFIT_OFFSETS = retained.BENEFIT_OFFSETS
DAMAGE_EXPONENTS = retained.DAMAGE_EXPONENTS
DAMAGE_THRESHOLDS = retained.DAMAGE_THRESHOLDS
RIDGE_GRID = retained.RIDGE_GRID
MIN_TIED_ROWS = 4
MIN_TIED_TRAIN_ROWS = 3

Geometry = retained.Geometry


@dataclass(frozen=True)
class AggregateShape:
    """Nonlinear shape of the tied aggregate response."""

    benefit_exponent: float
    benefit_offset: float
    damage_exponent: float
    damage_threshold: float

    def control_shape(self) -> aggregate_control.Shape:
        return aggregate_control.Shape(
            benefit_exponent=self.benefit_exponent,
            benefit_offset=self.benefit_offset,
            damage_exponent=self.damage_exponent,
            damage_threshold=self.damage_threshold,
            phase_leverage=0.0,
        )


@dataclass(frozen=True)
class PhaseConfig:
    """Frozen phase mechanism for one candidate."""

    name: str
    replay_exponent: float
    use_phase_information: bool = True
    use_replay_jensen: bool = True
    use_control_energy: bool = False
    use_aggregate_curvature: bool = False
    use_reactivation_bregman: bool = False


@dataclass(frozen=True)
class AggregateFitted:
    shape: AggregateShape
    ridge: float
    intercept: float
    coefficients: np.ndarray
    geometry: Geometry

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design = aggregate_design_matrix(weights, self.geometry, self.shape)
        return self.intercept + design @ self.coefficients


@dataclass(frozen=True)
class Fitted:
    aggregate: AggregateFitted
    phase: PhaseConfig
    phase_coefficients: np.ndarray

    def predict(self, weights: np.ndarray) -> np.ndarray:
        baseline = self.aggregate.predict(weights)
        design = phase_design_matrix(weights, self.aggregate, self.phase)
        return baseline + design @ self.phase_coefficients

    @property
    def phase_control(self) -> float:
        return float(self.phase_coefficients[0])


def aggregate_shape_grid() -> tuple[AggregateShape, ...]:
    return tuple(
        AggregateShape(*values)
        for values in product(
            BENEFIT_EXPONENTS,
            BENEFIT_OFFSETS,
            DAMAGE_EXPONENTS,
            DAMAGE_THRESHOLDS,
        )
    )


def tied_rows(weights: np.ndarray) -> np.ndarray:
    return np.max(np.abs(weights[:, 0, :] - weights[:, 1, :]), axis=1) < 1e-9


def aggregate_mixture(weights: np.ndarray, geometry: Geometry) -> np.ndarray:
    return aggregate_control.aggregate_mixture(weights, geometry)


def epoch_scale(geometry: Geometry) -> np.ndarray:
    return aggregate_control.epoch_scale(geometry)


def proportional_mixture(geometry: Geometry) -> np.ndarray:
    inverse = 1.0 / epoch_scale(geometry)
    return inverse / inverse.sum()


def normalized_replay_pressure(weights: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Token-weighted expected epochs, normalized to one at proportional."""
    aggregate = aggregate_mixture(weights, geometry)
    scale = epoch_scale(geometry)
    proportional = proportional_mixture(geometry)
    minimum = float(np.sum(scale * proportional**2))
    pressure = np.sum(scale[None, :] * aggregate**2, axis=1) / minimum
    assert np.all(pressure >= 1.0 - 1e-9), "proportional must minimize replay pressure"
    return pressure


def phase_information_cost(weights: np.ndarray, geometry: Geometry) -> np.ndarray:
    return aggregate_control.phase_information_cost(weights, geometry)


def replay_jensen_cost(weights: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Jensen gap of convex replay rate beyond one full-budget epoch."""
    beta0 = geometry.phase_0_fraction
    beta1 = geometry.phase_1_fraction
    scale = epoch_scale(geometry)
    phase0 = weights[:, 0, :] * scale[None, :]
    phase1 = weights[:, 1, :] * scale[None, :]
    aggregate = beta0 * phase0 + beta1 * phase1

    def cost(epochs: np.ndarray) -> np.ndarray:
        return np.maximum(epochs - 1.0, 0.0) ** 2

    gap = beta0 * cost(phase0) + beta1 * cost(phase1) - cost(aggregate)
    gap = gap.sum(axis=1)
    assert np.all(gap >= -1e-9), "convex replay Jensen gap must be nonnegative"
    return np.maximum(gap, 0.0)


def aggregate_design_matrix(weights: np.ndarray, geometry: Geometry, shape: AggregateShape) -> np.ndarray:
    return aggregate_control.aggregate_design_matrix(weights, geometry, shape.control_shape())


def directional_design_matrix(weights: np.ndarray, geometry: Geometry, shape: AggregateShape) -> np.ndarray:
    return aggregate_control.directional_design_matrix(weights, geometry, shape.control_shape())


def second_directional_design_matrix(
    weights: np.ndarray,
    geometry: Geometry,
    shape: AggregateShape,
) -> np.ndarray:
    """Second derivative of every aggregate design column along phase contrast."""
    aggregate = aggregate_mixture(weights, geometry)
    contrast_squared = (weights[:, 1, :] - weights[:, 0, :]) ** 2

    benefit_scale = aggregate + shape.benefit_offset
    benefit_curvature = (
        shape.benefit_exponent
        * (shape.benefit_exponent + 1.0)
        * benefit_scale ** (-(shape.benefit_exponent + 2.0))
        * contrast_squared
    )

    scale = epoch_scale(geometry)
    epochs = aggregate * scale
    excess = np.maximum(epochs - shape.damage_threshold, 0.0)
    active = excess > 0.0
    damage_curvature = np.zeros_like(excess)
    scale_squared = np.broadcast_to(scale**2, excess.shape)
    damage_curvature[active] = (
        shape.damage_exponent
        * (shape.damage_exponent - 1.0)
        * scale_squared[active]
        * excess[active] ** (shape.damage_exponent - 2.0)
        * contrast_squared[active]
    )
    return np.column_stack(
        [
            retained._hierarchical_block(benefit_curvature, geometry),
            retained._hierarchical_block(damage_curvature, geometry),
        ]
    )


def late_reactivation_state(weights: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Aggregate benefit retained after one-epoch late-phase reactivation."""
    aggregate = aggregate_mixture(weights, geometry)
    late_epochs = np.maximum(geometry.c1[None, :] * weights[:, 1, :], 0.0)
    tied_late_epochs = geometry.c1[None, :] * aggregate

    def activation(epochs: np.ndarray) -> np.ndarray:
        return epochs / (1.0 + epochs)

    numerator = activation(late_epochs)
    denominator = activation(tied_late_epochs)
    ratio = np.ones_like(aggregate)
    np.divide(numerator, denominator, out=ratio, where=denominator > 0.0)
    return aggregate * ratio


def reactivation_bregman_design_matrix(
    weights: np.ndarray,
    geometry: Geometry,
    shape: AggregateShape,
) -> np.ndarray:
    """Bregman divergence of the convex tied benefit under retained state."""
    aggregate = aggregate_mixture(weights, geometry)
    retained_state = late_reactivation_state(weights, geometry)
    aggregate_scale = aggregate + shape.benefit_offset
    retained_scale = retained_state + shape.benefit_offset
    benefit = aggregate_scale ** (-shape.benefit_exponent)
    retained_benefit = retained_scale ** (-shape.benefit_exponent)
    benefit_gradient = -shape.benefit_exponent * aggregate_scale ** (-(shape.benefit_exponent + 1.0))
    divergence = retained_benefit - benefit - benefit_gradient * (retained_state - aggregate)
    assert np.all(divergence >= -1e-9), "convex benefit Bregman divergence must be nonnegative"
    return retained._hierarchical_block(np.maximum(divergence, 0.0), geometry)


def phase_design_matrix(weights: np.ndarray, aggregate: AggregateFitted, phase: PhaseConfig) -> np.ndarray:
    directional = directional_design_matrix(weights, aggregate.geometry, aggregate.shape)
    ordering = directional @ aggregate.coefficients
    plasticity = normalized_replay_pressure(weights, aggregate.geometry) ** (-phase.replay_exponent)
    columns = [plasticity * ordering]
    if phase.use_control_energy:
        columns.append(ordering**2)
    if phase.use_phase_information:
        columns.append(phase_information_cost(weights, aggregate.geometry))
    if phase.use_replay_jensen:
        columns.append(replay_jensen_cost(weights, aggregate.geometry))
    if phase.use_aggregate_curvature:
        curvature = second_directional_design_matrix(weights, aggregate.geometry, aggregate.shape)
        aggregate_curvature = curvature @ aggregate.coefficients
        assert np.all(aggregate_curvature >= -1e-9), "convex aggregate response must have nonnegative curvature"
        columns.append(np.maximum(aggregate_curvature, 0.0))
    if phase.use_reactivation_bregman:
        divergence = reactivation_bregman_design_matrix(
            weights,
            aggregate.geometry,
            aggregate.shape,
        )
        benefit_coefficients = aggregate.coefficients[: divergence.shape[1]]
        reactivation_cost = divergence @ benefit_coefficients
        assert np.all(reactivation_cost >= -1e-9), "positive benefit amplitudes must preserve nonnegativity"
        columns.append(np.maximum(reactivation_cost, 0.0))
    return np.column_stack(columns)


def aggregate_penalty_multipliers(geometry: Geometry) -> np.ndarray:
    return aggregate_control.aggregate_penalty_multipliers(geometry)


def _indices(selection: np.ndarray) -> np.ndarray:
    selection = np.asarray(selection)
    return np.flatnonzero(selection) if selection.dtype == bool else selection.astype(int)


def fit_aggregate(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: Geometry,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> AggregateFitted:
    """Select the aggregate shape and ridge using tied evidence only."""
    tied = np.flatnonzero(tied_rows(weights))
    if len(tied) < MIN_TIED_ROWS:
        raise ValueError(f"aggregate fit needs at least {MIN_TIED_ROWS} tied rows, found {len(tied)}")
    multipliers = aggregate_penalty_multipliers(geometry)
    best_score = np.inf
    best_shape = None
    best_ridge = None
    for shape in aggregate_shape_grid():
        design = aggregate_design_matrix(weights, geometry, shape)
        for ridge in RIDGE_GRID:
            residuals = []
            for train, test in folds:
                train_tied = np.intersect1d(_indices(train), tied)
                test_tied = np.intersect1d(_indices(test), tied)
                if len(train_tied) < MIN_TIED_TRAIN_ROWS or not len(test_tied):
                    continue
                intercept, coefficients = retained.solve_head(
                    design[train_tied],
                    target[train_tied],
                    ridge,
                    multipliers,
                )
                residuals.append(intercept + design[test_tied] @ coefficients - target[test_tied])
            if not residuals:
                continue
            score = float(np.sqrt(np.mean(np.concatenate(residuals) ** 2)))
            if score < best_score:
                best_score = score
                best_shape = shape
                best_ridge = ridge
    if best_shape is None or best_ridge is None:
        raise ValueError("no aggregate shape had usable tied folds")
    design = aggregate_design_matrix(weights, geometry, best_shape)
    intercept, coefficients = retained.solve_head(
        design[tied],
        target[tied],
        best_ridge,
        multipliers,
    )
    return AggregateFitted(
        shape=best_shape,
        ridge=best_ridge,
        intercept=intercept,
        coefficients=coefficients,
        geometry=geometry,
    )


def _fit_nonnegative_phase_head(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Robust nonnegative phase amplitudes with no intercept."""
    scale = np.maximum(np.abs(design).max(axis=0), retained.COLUMN_SCALE_FLOOR)
    normalized = design / scale
    row_weights = np.ones(len(target))
    coefficients = np.zeros(design.shape[1])
    for _ in range(retained.HUBER_ITERATIONS):
        root = np.sqrt(row_weights)
        solved = lsq_linear(
            normalized * root[:, None],
            target * root,
            bounds=(np.zeros(design.shape[1]), np.full(design.shape[1], np.inf)),
            method="trf",
            tol=1e-10,
            max_iter=200,
        )
        updated = solved.x
        residual = normalized @ updated - target
        spread = retained.MAD_TO_SIGMA * float(np.median(np.abs(residual - np.median(residual))))
        shift = float(np.max(np.abs(normalized @ (updated - coefficients))))
        coefficients = updated
        if spread <= 0.0 or shift < retained.HUBER_TOLERANCE * spread:
            break
        cut = retained.HUBER_SCALE * spread
        row_weights = np.minimum(1.0, cut / np.maximum(np.abs(residual), 1e-12))
    return coefficients / scale


def fit_phase(
    aggregate: AggregateFitted,
    weights: np.ndarray,
    target: np.ndarray,
    phase: PhaseConfig,
    paired_tied_target: np.ndarray | None = None,
) -> Fitted:
    """Fit phase amplitudes after freezing the tied aggregate model.

    ``paired_tied_target`` contains the observed tied score for an asymmetric
    row when an exact aggregate-matched counterpart exists, and NaN otherwise.
    Paired differences therefore identify phase response without aggregate
    model error. Unpaired rows use the frozen aggregate prediction.
    """
    asymmetric = ~tied_rows(weights)
    rows = np.flatnonzero(asymmetric)
    if not len(rows):
        raise ValueError("phase fit needs asymmetric rows")
    baseline = aggregate.predict(weights[rows])
    if paired_tied_target is not None:
        paired = np.asarray(paired_tied_target, dtype=float)[rows]
        baseline = np.where(np.isfinite(paired), paired, baseline)
    phase_target = target[rows] - baseline
    design = phase_design_matrix(weights[rows], aggregate, phase)
    coefficients = _fit_nonnegative_phase_head(design, phase_target)
    return Fitted(aggregate=aggregate, phase=phase, phase_coefficients=coefficients)


def without_phase(model: Fitted) -> Fitted:
    return Fitted(
        aggregate=model.aggregate,
        phase=model.phase,
        phase_coefficients=np.zeros_like(model.phase_coefficients),
    )
