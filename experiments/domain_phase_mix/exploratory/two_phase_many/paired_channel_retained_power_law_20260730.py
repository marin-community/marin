# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit retained power law in aggregate and exact matched-phase channels.

For every asymmetric policy with an exact tied aggregate counterpart, the
invertible transformation

    (y_tied, y_asymmetric) -> (y_tied, y_asymmetric - y_tied)

exposes the phase response directly. Giving the aggregate and phase channels
equal total weight is a generalized least-squares objective with a non-diagonal
per-pair weight matrix in the original coordinates. It changes the estimation
loss, not the retained-power-law equation or parameter count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from experiments.domain_phase_mix.exploratory.two_phase_many import retained_power_law_model_20260728 as rpl


@dataclass(frozen=True)
class ChannelSystem:
    """Transformed regression system and its fixed design weights."""

    design: np.ndarray
    target: np.ndarray
    intercept_column: np.ndarray
    row_weights: np.ndarray
    column_scale: np.ndarray
    aggregate_count: int
    phase_count: int

    @property
    def equation_count(self) -> int:
        return len(self.target)


@dataclass(frozen=True)
class Fitted:
    """A retained-power-law fit with paired-channel diagnostics."""

    model: rpl.Fitted
    active_coefficients: int
    aggregate_count: int
    phase_count: int

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.model.predict(weights)


def tied_rows(weights: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    """Return policies whose two phase mixtures are equal."""
    return np.all(np.abs(weights[:, 0, :] - weights[:, 1, :]) <= tolerance, axis=1)


def build_channel_system(
    design: np.ndarray,
    target: np.ndarray,
    correspondence_keys: np.ndarray,
    weights: np.ndarray,
) -> ChannelSystem:
    """Replace every asymmetric absolute equation with its matched phase delta."""
    if len(design) != len(target) or len(target) != len(correspondence_keys) or len(target) != len(weights):
        raise ValueError("design, target, keys, and weights must have the same number of rows")

    tied = tied_rows(weights)
    aggregate_indices = np.flatnonzero(tied)
    phase_indices = np.flatnonzero(~tied)
    if not len(aggregate_indices) or not len(phase_indices):
        raise ValueError("paired-channel fitting requires both tied and asymmetric policies")

    tied_by_key: dict[str, list[int]] = {}
    for index in aggregate_indices:
        tied_by_key.setdefault(str(correspondence_keys[index]), []).append(int(index))

    matched_indices = []
    for index in phase_indices:
        matches = tied_by_key.get(str(correspondence_keys[index]), [])
        if len(matches) != 1:
            raise ValueError(f"asymmetric row {index} has {len(matches)} tied counterparts; exactly one is required")
        matched_indices.append(matches[0])
    matched = np.asarray(matched_indices, dtype=int)

    transformed_design = np.vstack(
        [
            design[aggregate_indices],
            design[phase_indices] - design[matched],
        ]
    )
    transformed_target = np.concatenate(
        [
            target[aggregate_indices],
            target[phase_indices] - target[matched],
        ]
    )
    intercept_column = np.concatenate(
        [
            np.ones(len(aggregate_indices)),
            np.zeros(len(phase_indices)),
        ]
    )

    equation_count = len(aggregate_indices) + len(phase_indices)
    if equation_count != len(target):
        raise ValueError(f"expected one transformed equation per input row, got {equation_count} for {len(target)}")

    # Each channel contributes half the total data weight while the total stays
    # equal to the ordinary absolute-response fit. This keeps the nominal ridge
    # strength comparable across estimators.
    channel_total = equation_count / 2.0
    row_weights = np.concatenate(
        [
            np.full(len(aggregate_indices), channel_total / len(aggregate_indices)),
            np.full(len(phase_indices), channel_total / len(phase_indices)),
        ]
    )
    if not np.isclose(row_weights.sum(), equation_count):
        raise AssertionError("paired-channel row weights must preserve total data weight")

    # The scale comes from the original absolute design, not the differenced
    # rows. Otherwise the transformation silently changes the ridge prior.
    column_scale = np.maximum(np.abs(design).max(axis=0), rpl.COLUMN_SCALE_FLOOR)
    return ChannelSystem(
        design=transformed_design,
        target=transformed_target,
        intercept_column=intercept_column,
        row_weights=row_weights,
        column_scale=column_scale,
        aggregate_count=len(aggregate_indices),
        phase_count=len(phase_indices),
    )


def _bounded_solve(
    augmented: np.ndarray,
    response: np.ndarray,
    data_count: int,
    columns: int,
    row_weights: np.ndarray,
) -> np.ndarray:
    weighted_design = augmented.copy()
    weighted_response = response.copy()
    root = np.sqrt(row_weights)[:, None]
    weighted_design[:data_count] *= root
    weighted_response[:data_count] *= root[:, 0]
    bounds = (
        np.concatenate([[-np.inf], np.zeros(columns)]),
        np.full(columns + 1, np.inf),
    )
    solved = lsq_linear(
        weighted_design,
        weighted_response,
        bounds=bounds,
        method="trf",
        tol=1e-10,
        max_iter=200,
    )
    predicted = augmented[:data_count] @ solved.x
    if not np.all(np.isfinite(predicted)):
        raise ValueError("paired-channel solve produced non-finite predictions")
    limit = rpl.PREDICTION_SCALE_LIMIT * max(float(np.max(np.abs(response[:data_count]))), 1e-12)
    if float(np.max(np.abs(predicted))) > limit:
        raise ValueError("paired-channel solve produced predictions outside the response-scale guard")
    return solved.x


def solve_head(
    system: ChannelSystem,
    ridge: float,
    multipliers: np.ndarray,
) -> tuple[float, np.ndarray, int]:
    """Fit the shared nonnegative RPL head under the paired-channel loss."""
    columns = system.design.shape[1]
    if len(multipliers) != columns:
        raise ValueError("one ridge multiplier is required per design column")

    data_design = np.column_stack(
        [
            system.intercept_column,
            system.design / system.column_scale,
        ]
    )
    penalty = np.diag(np.concatenate([[0.0], np.sqrt(ridge * multipliers)]))
    augmented = np.vstack([data_design, penalty])
    response = np.concatenate([system.target, np.zeros(len(penalty))])
    robust_weights = system.row_weights.copy()
    coefficients = _bounded_solve(
        augmented,
        response,
        system.equation_count,
        columns,
        robust_weights,
    )

    if rpl.HUBER_SCALE is not None:
        aggregate = slice(0, system.aggregate_count)
        phase = slice(system.aggregate_count, system.equation_count)
        for _ in range(rpl.HUBER_ITERATIONS):
            residual = data_design @ coefficients - system.target
            updated_weights = []
            spreads = []
            for channel, base_weights in (
                (aggregate, system.row_weights[aggregate]),
                (phase, system.row_weights[phase]),
            ):
                channel_residual = residual[channel]
                spread = rpl.MAD_TO_SIGMA * float(np.median(np.abs(channel_residual - np.median(channel_residual))))
                spreads.append(spread)
                if spread <= 0.0:
                    updated_weights.append(base_weights)
                    continue
                cut = rpl.HUBER_SCALE * spread
                updated_weights.append(base_weights * np.minimum(1.0, cut / np.maximum(np.abs(channel_residual), 1e-12)))
            robust_weights = np.concatenate(updated_weights)
            updated = _bounded_solve(
                augmented,
                response,
                system.equation_count,
                columns,
                robust_weights,
            )
            shift = float(np.max(np.abs(data_design @ (updated - coefficients))))
            coefficients = updated
            if shift < rpl.HUBER_TOLERANCE * max(max(spreads), 1e-12):
                break

    unscaled = coefficients[1:] / system.column_scale
    active = int(np.count_nonzero(unscaled > 1e-10))
    return float(coefficients[0]), unscaled, active


def fit_fixed_shape(
    weights: np.ndarray,
    target: np.ndarray,
    correspondence_keys: np.ndarray,
    geometry: rpl.Geometry,
    shape: rpl.Shape,
    ridge: float,
) -> Fitted:
    """Fit paired channels with a shape and ridge selected independently."""
    absolute_design = rpl.design_matrix(weights, geometry, shape)
    system = build_channel_system(absolute_design, target, correspondence_keys, weights)
    intercept, coefficients, active = solve_head(
        system,
        ridge,
        rpl.penalty_multipliers(geometry, shape),
    )
    model = rpl.Fitted(
        shape=shape,
        ridge=ridge,
        intercept=intercept,
        coefficients=coefficients,
        geometry=geometry,
    )
    return Fitted(
        model=model,
        active_coefficients=active,
        aggregate_count=system.aggregate_count,
        phase_count=system.phase_count,
    )
