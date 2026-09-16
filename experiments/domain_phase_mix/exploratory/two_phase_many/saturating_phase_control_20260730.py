# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate-conditioned phase control with a bounded odd response.

Let ``a = beta0 * w0 + beta1 * w1`` and ``delta = w1 - w0``. A tied-only
aggregate model ``A`` supplies a tangent gradient ``g(a)``. Define

    u(a, delta) = g(a)^T delta
    c(a) = max_i g_i(a) - min_i g_i(a)
    S_tau(u; a) = tau * c(a) * tanh(u / (tau * c(a))).

``c(a)`` is the pointwise control authority of the aggregate response on the
mixture simplex. Every feasible contrast obeys ``|u| <= c(a)``. The candidate
response is

    L(a, delta) = A(a)
        + xi * S_tau(u; a)
        + gamma * I_beta(a, delta)
        + zeta * J(a, delta),

where ``I_beta`` is phase-information cost and ``J`` is the phase-local replay
Jensen gap. The amplitudes are nonnegative. ``tau`` is selected from a frozen
dimensionless grid; ``tau = infinity`` is the exact linear-response ablation.

The bounded odd response is the new mechanism. It is not a monotone output
link, an even control-energy term, or saturation inside a retained-state
transition. At a tied policy all phase terms vanish. At an interior optimum of
``A``, all tangent-gradient components agree, so ``c = u = 0`` and nonnegative
phase costs make that optimum fiber-optimal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    aggregate_conditioned_replay_control_20260730 as replay_control,
)

TAU_GRID = (np.inf, 2.0, 1.0, 0.5, 0.25)
MIN_ASYMMETRIC_TRAIN_ROWS = 3
CONTROL_TOLERANCE = 1e-8

Geometry = replay_control.Geometry
AggregateFitted = replay_control.AggregateFitted


@dataclass(frozen=True)
class Fitted:
    """Fitted bounded phase response on a frozen tied aggregate model."""

    aggregate: AggregateFitted
    tau: float
    phase_coefficients: np.ndarray
    selection_scores: tuple[tuple[float, float], ...]

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.aggregate.predict(weights) + phase_design_matrix(weights, self.aggregate, self.tau) @ (
            self.phase_coefficients
        )

    @property
    def phase_control(self) -> float:
        return float(self.phase_coefficients[0])

    @property
    def phase_information(self) -> float:
        return float(self.phase_coefficients[1])

    @property
    def replay_jensen(self) -> float:
        return float(self.phase_coefficients[2])


def tangent_gradient(weights: np.ndarray, aggregate: AggregateFitted) -> np.ndarray:
    """Return aggregate gradients modulo the simplex-normal constant.

    The last bucket is the reference. Each other component is the directional
    derivative along ``e_i - e_ref``. This is sufficient because every valid
    phase contrast sums to zero, and both ``u`` and ``max(g)-min(g)`` are
    invariant to adding a constant to all gradient components.
    """

    mixture = replay_control.aggregate_mixture(weights, aggregate.geometry)
    rows, domains = mixture.shape
    beta0 = aggregate.geometry.phase_0_fraction
    beta1 = aggregate.geometry.phase_1_fraction
    gradient = np.zeros((rows, domains), dtype=float)
    reference = domains - 1
    for domain in range(reference):
        contrast = np.zeros_like(mixture)
        contrast[:, domain] = 1.0
        contrast[:, reference] = -1.0
        synthetic = np.stack(
            [
                mixture - beta1 * contrast,
                mixture + beta0 * contrast,
            ],
            axis=1,
        )
        directional = replay_control.directional_design_matrix(
            synthetic,
            aggregate.geometry,
            aggregate.shape,
        )
        gradient[:, domain] = directional @ aggregate.coefficients
    return gradient


def control_statistics(
    weights: np.ndarray,
    aggregate: AggregateFitted,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return signed control, pointwise authority, and normalized magnitude."""

    contrast = weights[:, 1, :] - weights[:, 0, :]
    if np.max(np.abs(contrast.sum(axis=1))) > CONTROL_TOLERANCE:
        raise ValueError("phase contrasts must be tangent to the mixture simplex")
    gradient = tangent_gradient(weights, aggregate)
    control = np.sum(gradient * contrast, axis=1)
    authority = np.ptp(gradient, axis=1)
    ratio = np.zeros_like(control)
    np.divide(np.abs(control), authority, out=ratio, where=authority > CONTROL_TOLERANCE)
    if np.max(ratio) > 1.0 + CONTROL_TOLERANCE:
        raise ValueError(f"control exceeded simplex authority: {np.max(ratio):.6f}")
    return control, authority, ratio


def saturating_control(
    control: np.ndarray,
    authority: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Bound the signed response while preserving its derivative at zero."""

    if np.isinf(tau):
        return control.copy()
    scale = tau * authority
    response = np.zeros_like(control)
    active = scale > CONTROL_TOLERANCE
    response[active] = scale[active] * np.tanh(control[active] / scale[active])
    return response


def phase_design_matrix(
    weights: np.ndarray,
    aggregate: AggregateFitted,
    tau: float,
) -> np.ndarray:
    """Three response columns: bounded odd control and two nonnegative costs."""

    control, authority, _ratio = control_statistics(weights, aggregate)
    return np.column_stack(
        [
            saturating_control(control, authority, tau),
            replay_control.phase_information_cost(weights, aggregate.geometry),
            replay_control.replay_jensen_cost(weights, aggregate.geometry),
        ]
    )


def _indices(selection: np.ndarray) -> np.ndarray:
    selection = np.asarray(selection)
    return np.flatnonzero(selection) if selection.dtype == bool else selection.astype(int)


def phase_target(
    aggregate: AggregateFitted,
    weights: np.ndarray,
    target: np.ndarray,
    paired_tied_target: np.ndarray | None,
) -> np.ndarray:
    """Subtract exact tied controls when available, otherwise the aggregate fit."""

    baseline = aggregate.predict(weights)
    if paired_tied_target is not None:
        paired = np.asarray(paired_tied_target, dtype=float)
        baseline = np.where(np.isfinite(paired), paired, baseline)
    return target - baseline


def fit_fixed_tau(
    aggregate: AggregateFitted,
    weights: np.ndarray,
    target: np.ndarray,
    tau: float,
    paired_tied_target: np.ndarray | None = None,
) -> Fitted:
    """Fit phase amplitudes for one prespecified response shape."""

    rows = np.flatnonzero(~replay_control.tied_rows(weights))
    if len(rows) < MIN_ASYMMETRIC_TRAIN_ROWS:
        raise ValueError(f"phase fit needs at least {MIN_ASYMMETRIC_TRAIN_ROWS} asymmetric rows")
    design = phase_design_matrix(weights, aggregate, tau)
    residual_target = phase_target(aggregate, weights, target, paired_tied_target)
    coefficients = replay_control._fit_nonnegative_phase_head(design[rows], residual_target[rows])
    return Fitted(
        aggregate=aggregate,
        tau=tau,
        phase_coefficients=coefficients,
        selection_scores=((tau, np.nan),),
    )


def fit(
    aggregate: AggregateFitted,
    weights: np.ndarray,
    target: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    paired_tied_target: np.ndarray | None = None,
) -> Fitted:
    """Select ``tau`` inside training folds and refit phase amplitudes."""

    asymmetric = ~replay_control.tied_rows(weights)
    residual_target = phase_target(aggregate, weights, target, paired_tied_target)
    scores = []
    for tau in TAU_GRID:
        design = phase_design_matrix(weights, aggregate, tau)
        residuals = []
        for train, test in folds:
            train_rows = np.intersect1d(_indices(train), np.flatnonzero(asymmetric))
            test_rows = np.intersect1d(_indices(test), np.flatnonzero(asymmetric))
            if len(train_rows) < MIN_ASYMMETRIC_TRAIN_ROWS or not len(test_rows):
                continue
            coefficients = replay_control._fit_nonnegative_phase_head(
                design[train_rows],
                residual_target[train_rows],
            )
            residuals.append(design[test_rows] @ coefficients - residual_target[test_rows])
        if not residuals:
            continue
        score = float(np.sqrt(np.mean(np.concatenate(residuals) ** 2)))
        scores.append((tau, score))
    if not scores:
        raise ValueError("no tau candidate had usable phase folds")
    best_tau, _best_score = min(
        scores,
        key=lambda item: (item[1], 0 if np.isinf(item[0]) else 1),
    )
    fitted = fit_fixed_tau(
        aggregate,
        weights,
        target,
        best_tau,
        paired_tied_target=paired_tied_target,
    )
    return Fitted(
        aggregate=fitted.aggregate,
        tau=fitted.tau,
        phase_coefficients=fitted.phase_coefficients,
        selection_scores=tuple(scores),
    )
