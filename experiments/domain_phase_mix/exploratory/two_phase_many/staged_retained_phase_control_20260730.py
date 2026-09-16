# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tied aggregate response plus a nonlinear retained-state phase residual.

Let ``a = beta0 * w0 + beta1 * w1``. The model is

    L(w0, w1) = A(a) + xi * Delta_B(a, w0, w1) + gamma * I_beta(a, w1 - w0).

``A`` is fitted only on tied policies. ``Delta_B`` applies the fitted aggregate
benefit amplitudes to the change induced by a normalized retained state,

    s_i = [beta0 * w0_i * exp(g(lambda * (w1_i - w0_i)))
           + m * beta1 * w1_i] / [beta0 + m * beta1],

    Delta_B = B_A(s) - B_A(a).

The normalization makes ``s = a`` for every tied policy, so the phase residual
cannot rewrite the fitted one-phase surface. Retention ``lambda`` makes the
value of early data depend on what arrives late, while ``m`` represents the
distinct leverage of the late optimization window. ``I_beta`` is the
nonnegative Fisher-quadratic cost of making the phases distinguishable.

Exact aggregate-matched pairs identify the phase response from score
differences. Unpaired asymmetric rows use the frozen aggregate prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    aggregate_conditioned_replay_control_20260730 as replay_control,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import retained_power_law_model_20260728 as retained

RETENTIONS = retained.RETENTIONS
LATE_MULTIPLIERS = retained.LATE_MULTIPLIERS


@dataclass(frozen=True)
class Shape:
    retention: float
    late_multiplier: float


@dataclass(frozen=True)
class Fitted:
    aggregate: replay_control.AggregateFitted
    shape: Shape
    family_resolved: bool
    phase_coefficients: np.ndarray

    def predict(self, weights: np.ndarray) -> np.ndarray:
        baseline = self.aggregate.predict(weights)
        design = phase_design_matrix(
            weights,
            self.aggregate,
            self.shape,
            family_resolved=self.family_resolved,
        )
        return baseline + design @ self.phase_coefficients


@dataclass(frozen=True)
class JointFitted:
    aggregate: replay_control.AggregateFitted
    shape: Shape
    use_ordering: bool
    balance_policy_classes: bool
    ridge: float
    intercept: float
    coefficients: np.ndarray

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design = joint_design_matrix(
            weights,
            self.aggregate,
            self.shape,
            use_ordering=self.use_ordering,
        )
        return self.intercept + design @ self.coefficients


def shape_grid() -> tuple[Shape, ...]:
    return tuple(Shape(*values) for values in product(RETENTIONS, LATE_MULTIPLIERS))


def normalized_retained_state(
    weights: np.ndarray,
    geometry: retained.Geometry,
    shape: Shape,
) -> np.ndarray:
    """Retained token share normalized to preserve every tied policy."""
    beta0 = geometry.phase_0_fraction
    beta1 = geometry.phase_1_fraction
    phase0 = weights[:, 0, :]
    phase1 = weights[:, 1, :]
    contrast = phase1 - phase0
    survival = np.exp(retained.GATE_CLIP * np.tanh(shape.retention * contrast / retained.GATE_CLIP))
    normalizer = beta0 + shape.late_multiplier * beta1
    state = (beta0 * phase0 * survival + shape.late_multiplier * beta1 * phase1) / normalizer
    assert np.all(state >= 0.0), "retained state must remain nonnegative"
    return state


def aggregate_benefit_coefficients(aggregate: replay_control.AggregateFitted) -> np.ndarray:
    block_size = len(np.unique(aggregate.geometry.families)) + len(aggregate.geometry.excess_domains)
    return aggregate.coefficients[:block_size]


def retained_benefit_residual(
    weights: np.ndarray,
    aggregate: replay_control.AggregateFitted,
    shape: Shape,
) -> np.ndarray:
    """Change in fitted benefit response induced by the retained state."""
    mixture = replay_control.aggregate_mixture(weights, aggregate.geometry)
    state = normalized_retained_state(weights, aggregate.geometry, shape)
    exponent = aggregate.shape.benefit_exponent
    offset = aggregate.shape.benefit_offset
    tied_basis = retained._hierarchical_block((mixture + offset) ** (-exponent), aggregate.geometry)
    phase_basis = retained._hierarchical_block((state + offset) ** (-exponent), aggregate.geometry)
    return (phase_basis - tied_basis) @ aggregate_benefit_coefficients(aggregate)


def retained_family_residual(
    weights: np.ndarray,
    aggregate: replay_control.AggregateFitted,
    shape: Shape,
) -> np.ndarray:
    """One retained-benefit response column per predeclared family."""
    mixture = replay_control.aggregate_mixture(weights, aggregate.geometry)
    state = normalized_retained_state(weights, aggregate.geometry, shape)
    exponent = aggregate.shape.benefit_exponent
    offset = aggregate.shape.benefit_offset
    tied_basis = (mixture + offset) ** (-exponent)
    phase_basis = (state + offset) ** (-exponent)
    return retained._family_totals(phase_basis - tied_basis, aggregate.geometry)


def phase_design_matrix(
    weights: np.ndarray,
    aggregate: replay_control.AggregateFitted,
    shape: Shape,
    family_resolved: bool,
) -> np.ndarray:
    response = (
        retained_family_residual(weights, aggregate, shape)
        if family_resolved
        else retained_benefit_residual(weights, aggregate, shape)[:, None]
    )
    return np.column_stack(
        [
            response,
            replay_control.phase_information_cost(weights, aggregate.geometry),
        ]
    )


def joint_design_matrix(
    weights: np.ndarray,
    aggregate: replay_control.AggregateFitted,
    shape: Shape,
    use_ordering: bool,
) -> np.ndarray:
    """Aggregate response plus family-pooled phase columns that vanish when tied."""
    columns = [
        replay_control.aggregate_design_matrix(
            weights,
            aggregate.geometry,
            aggregate.shape,
        ),
        retained_family_residual(weights, aggregate, shape),
        replay_control.phase_information_cost(weights, aggregate.geometry),
    ]
    if use_ordering:
        columns.append(
            retained.marginal_phase_block(
                weights,
                aggregate.geometry,
                retained.Shape(
                    benefit_exponent=aggregate.shape.benefit_exponent,
                    benefit_offset=aggregate.shape.benefit_offset,
                    damage_exponent=aggregate.shape.damage_exponent,
                    damage_threshold=aggregate.shape.damage_threshold,
                    retention=shape.retention,
                    late_multiplier=shape.late_multiplier,
                    ordering_channel=True,
                ),
            )
        )
    return np.column_stack(columns)


def joint_penalty_multipliers(
    aggregate: replay_control.AggregateFitted,
    use_ordering: bool,
) -> np.ndarray:
    aggregate_penalty = replay_control.aggregate_penalty_multipliers(aggregate.geometry)
    phase_columns = len(np.unique(aggregate.geometry.families)) + 1
    multipliers = [aggregate_penalty, np.ones(phase_columns)]
    if use_ordering:
        families = len(np.unique(aggregate.geometry.families))
        multipliers.append(np.concatenate([np.ones(4 * families), np.zeros(2)]))
    return np.concatenate(multipliers)


def _phase_baseline(
    aggregate: replay_control.AggregateFitted,
    weights: np.ndarray,
    rows: np.ndarray,
    paired_tied_target: np.ndarray | None,
) -> np.ndarray:
    baseline = aggregate.predict(weights[rows])
    if paired_tied_target is None:
        return baseline
    paired = np.asarray(paired_tied_target, dtype=float)[rows]
    return np.where(np.isfinite(paired), paired, baseline)


def _fit_shape(
    aggregate: replay_control.AggregateFitted,
    weights: np.ndarray,
    target: np.ndarray,
    shape: Shape,
    family_resolved: bool,
    rows: np.ndarray,
    paired_tied_target: np.ndarray | None,
) -> np.ndarray:
    baseline = _phase_baseline(aggregate, weights, rows, paired_tied_target)
    design = phase_design_matrix(
        weights[rows],
        aggregate,
        shape,
        family_resolved=family_resolved,
    )
    return replay_control._fit_nonnegative_phase_head(design, target[rows] - baseline)


def fit(
    aggregate: replay_control.AggregateFitted,
    weights: np.ndarray,
    target: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    paired_tied_target: np.ndarray | None = None,
    family_resolved: bool = False,
) -> Fitted:
    """Select the phase transition out of fold after freezing ``A``."""
    asymmetric = ~replay_control.tied_rows(weights)
    best_score = np.inf
    best_shape = None
    for shape in shape_grid():
        residuals = []
        for train, test in folds:
            train_rows = np.intersect1d(train, np.flatnonzero(asymmetric))
            test_rows = np.intersect1d(test, np.flatnonzero(asymmetric))
            if not len(train_rows) or not len(test_rows):
                continue
            coefficients = _fit_shape(
                aggregate,
                weights,
                target,
                shape,
                family_resolved,
                train_rows,
                paired_tied_target,
            )
            baseline = _phase_baseline(
                aggregate,
                weights,
                test_rows,
                paired_tied_target,
            )
            predicted = (
                baseline
                + phase_design_matrix(
                    weights[test_rows],
                    aggregate,
                    shape,
                    family_resolved=family_resolved,
                )
                @ coefficients
            )
            residuals.append(predicted - target[test_rows])
        if not residuals:
            continue
        score = float(np.sqrt(np.mean(np.concatenate(residuals) ** 2)))
        if score < best_score:
            best_score = score
            best_shape = shape
    if best_shape is None:
        raise ValueError("no retained-state phase shape had usable folds")
    rows = np.flatnonzero(asymmetric)
    coefficients = _fit_shape(
        aggregate,
        weights,
        target,
        best_shape,
        family_resolved,
        rows,
        paired_tied_target,
    )
    return Fitted(
        aggregate=aggregate,
        shape=best_shape,
        family_resolved=family_resolved,
        phase_coefficients=coefficients,
    )


def fit_joint(
    aggregate: replay_control.AggregateFitted,
    weights: np.ndarray,
    target: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    use_ordering: bool = False,
    balance_policy_classes: bool = False,
) -> JointFitted:
    """Jointly estimate aggregate amplitudes and a tied-invariant phase residual."""
    best_score = np.inf
    best_shape = None
    best_ridge = None
    multipliers = joint_penalty_multipliers(aggregate, use_ordering)
    tied = replay_control.tied_rows(weights)

    def fit_rows(rows: np.ndarray) -> np.ndarray:
        if not balance_policy_classes:
            return rows
        tied_rows = rows[tied[rows]]
        asymmetric_rows = rows[~tied[rows]]
        if not len(tied_rows) or not len(asymmetric_rows):
            return rows
        if len(tied_rows) < len(asymmetric_rows):
            extra = np.resize(tied_rows, len(asymmetric_rows) - len(tied_rows))
        else:
            extra = np.resize(asymmetric_rows, len(tied_rows) - len(asymmetric_rows))
        return np.concatenate([rows, extra])

    for shape in shape_grid():
        design = joint_design_matrix(
            weights,
            aggregate,
            shape,
            use_ordering=use_ordering,
        )
        for ridge in retained.RIDGE_GRID:
            residuals = []
            for train, test in folds:
                fitting = fit_rows(train)
                intercept, coefficients = retained.solve_head(
                    design[fitting],
                    target[fitting],
                    ridge,
                    multipliers,
                )
                residuals.append(intercept + design[test] @ coefficients - target[test])
            score = float(np.sqrt(np.mean(np.concatenate(residuals) ** 2)))
            if score < best_score:
                best_score = score
                best_shape = shape
                best_ridge = ridge
    if best_shape is None or best_ridge is None:
        raise ValueError("no joint retained-state phase shape had usable folds")
    design = joint_design_matrix(
        weights,
        aggregate,
        best_shape,
        use_ordering=use_ordering,
    )
    intercept, coefficients = retained.solve_head(
        design[fit_rows(np.arange(len(target)))],
        target[fit_rows(np.arange(len(target)))],
        best_ridge,
        multipliers,
    )
    return JointFitted(
        aggregate=aggregate,
        shape=best_shape,
        use_ordering=use_ordering,
        balance_policy_classes=balance_policy_classes,
        ridge=best_ridge,
        intercept=intercept,
        coefficients=coefficients,
    )
