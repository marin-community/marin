# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two exposure indices with different memory horizons.

Every phase mechanism tried on this problem so far modifies a single exposure index: a retention
factor, a recency weight, a replay term, all reweighting one cumulative dose. That whole family is
refuted by one measurement. If loss depends on the policy only through one index, then a two-phase
policy has the same index value as some tied policy and therefore the same predicted loss, so it can
never beat the best tied policy. The 80/20 WSD panel measures a `0.009594` BPB gain over the entire
tied class. One index cannot produce that, whatever form it takes.

Two indices can. Give each bucket two effective mixtures, one integrated over a long horizon and one
over a short one:

    E_k,i = (1 - phi_k) w0_i + phi_k w1_i        for k in {slow, fast}

The tied policy class is exactly the diagonal `E_slow = E_fast = w`. A two-phase policy leaves that
diagonal, so if the response surface has its minimum off the diagonal, the best two-phase policy beats
every constant mixture. Setting `phi_slow = phi_fast` collapses the two indices into one and recovers
the phase-weighted-dose null exactly.

The reading is that different capabilities integrate the data stream over different horizons and their
preferred mixtures differ, so a schedule that is light on a domain overall but heavy on it late can
serve both at once. That is a claim about memory, and it is falsifiable: the two horizons have to come
out separated and interior, and the fitted optimum has to sit off the diagonal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from experiments.domain_phase_mix.exploratory.two_phase_many import interference_evidence_model_20260806 as ile

# `phi` is the weight the index puts on the phase-1 mixture. The realized phase-1 token share is about
# 0.20, so a horizon at 0.20 integrates the run uniformly and anything above it is recency-weighted.
# The grid reaches 1.0, where only the decay phase counts.
# Both horizons live on the same interval and the only real constraint between them is slow < fast, so
# they share one grid. The first version gave `slow` its own grid capped at 0.45, which was an unforced
# choice and pinned the selection at that cap; the cap was removed after seeing that, which makes this a
# post-outcome grid correction and any result under it provisional until independently confirmed.
HORIZON_GRID = (0.05, 0.10, 0.20, 0.30, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00)
SLOW_GRID = HORIZON_GRID
FAST_GRID = HORIZON_GRID
RHO_GRID = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
CURVATURE_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, math.inf)
# Marginal damage from over-exposure. One recovers a term linear in epochs, which measurement showed
# gives too broad a bowl at the tied optimum; above one, repeated data hurts superlinearly.
DAMAGE_EXPONENT_GRID = (1.0, 1.5, 2.0, 3.0)
HEAD_RIDGE_GRID = ile.HEAD_RIDGE_GRID


@dataclass(frozen=True)
class Shape:
    """Two memory horizons and the shared acquisition curve they both feed."""

    slow: float
    fast: float
    rho: float
    curvature: float = math.inf
    damage_exponent: float = 1.0

    @property
    def single_index(self) -> bool:
        """True when the two horizons coincide, which is the phase-weighted-dose null exactly."""
        return self.slow == self.fast


def effective_mixture(weights: np.ndarray, horizon: float) -> np.ndarray:
    """The mixture as seen by a readout that weights the decay phase by `horizon`."""
    return (1.0 - horizon) * weights[:, 0, :] + horizon * weights[:, 1, :]


def evidence(weights: np.ndarray, geometry: ile.Geometry, shape: Shape, horizon: float) -> np.ndarray:
    """Retained evidence per bucket at one horizon, bounded in [0, 1).

    Total epochs per unit weight is `c0 + c1`, so an index of `w` corresponds to `(c0 + c1) w` epochs.
    """
    epochs = (geometry.c0 + geometry.c1) * effective_mixture(weights, horizon)
    return ile.acquired_share(epochs, ile.Shape(rho=shape.rho, interference=0.0, curvature=shape.curvature))


def design_matrix(weights: np.ndarray, geometry: ile.Geometry, shape: Shape) -> np.ndarray:
    """Intercept, family evidence at each horizon, family over-exposure, per-bucket departures.

    Departures are carried on the slow index only, and only when a family actually contains more than
    one bucket. The previous round shipped a design that was exactly rank deficient because the family
    column is the mean of its members, so with one bucket per family the two blocks are identical
    columns. Dropping the departures in that case removes the deficiency instead of leaving the ridge
    to paper over it.
    """
    slow = evidence(weights, geometry, shape, shape.slow)
    fast = evidence(weights, geometry, shape, shape.fast)
    damage = ile.overexposure(weights, geometry) ** shape.damage_exponent
    blocks = [
        np.ones(len(weights)),
        -ile.family_sums(slow, geometry),
        -ile.family_sums(fast, geometry),
        ile.family_sums(damage, geometry),
    ]
    if geometry.n_domains > geometry.n_families:
        blocks.append(-slow)
    return np.column_stack(blocks)


def has_departures(geometry: ile.Geometry) -> bool:
    return geometry.n_domains > geometry.n_families


def head_bounds(width: int, geometry: ile.Geometry) -> tuple[np.ndarray, np.ndarray]:
    """Amplitudes non-negative, intercept free, departures signed but small enough to keep the net positive.

    The previous round found that unbounded departures let a bucket's net evidence coefficient go
    negative, which quietly broke a structural argument that assumed otherwise. Here the departure is
    bounded by the amplitude limit's own scale so the family level keeps the sign.
    """
    families = geometry.n_families
    lower = np.zeros(width)
    upper = np.full(width, ile.AMPLITUDE_LIMIT)
    lower[0], upper[0] = -np.inf, np.inf
    if has_departures(geometry):
        # Bounded by the family level's own scale so the net evidence coefficient keeps its sign.
        lower[1 + 3 * families :] = -ile.AMPLITUDE_LIMIT
    return lower, upper


def penalty_rows(geometry: ile.Geometry, ridge: float) -> np.ndarray:
    """Shrink bucket departures hard, family levels barely, never the intercept."""
    families = geometry.n_families
    departures = geometry.n_domains if has_departures(geometry) else 0
    width = 1 + 3 * families + departures
    scales = np.concatenate([[0.0], np.full(3 * families, 1e-3), np.ones(departures)])
    return np.sqrt(ridge) * np.diag(scales)[:width, :width]


def solve_head(design: np.ndarray, response: np.ndarray, geometry: ile.Geometry, ridge: float) -> np.ndarray:
    penalty = penalty_rows(geometry, ridge)
    augmented = np.vstack([design, penalty])
    target = np.concatenate([response, np.zeros(len(penalty))])
    lower, upper = head_bounds(design.shape[1], geometry)
    solved = lsq_linear(augmented, target, bounds=(lower, upper), method="trf", max_iter=500)
    predicted = design @ solved.x
    assert np.all(np.isfinite(predicted)), "bounded solve produced non-finite predictions"
    limit = ile.PREDICTION_SCALE_LIMIT * max(float(np.max(np.abs(response))), 1e-12)
    assert np.max(np.abs(predicted)) <= limit, f"bounded solve produced predictions above {limit:.4g}"
    return solved.x


def shape_grid(
    include_single_index: bool = True,
    damage_grid: tuple[float, ...] = DAMAGE_EXPONENT_GRID,
) -> tuple[Shape, ...]:
    """Every horizon pair with slow below fast, plus the collapsed single-index null."""
    shapes = [
        Shape(slow=slow, fast=fast, rho=rho, curvature=curvature, damage_exponent=tau)
        for tau in damage_grid
        for curvature in CURVATURE_GRID
        for rho in RHO_GRID
        for slow in SLOW_GRID
        for fast in FAST_GRID
        if fast > slow
    ]
    if include_single_index:
        shapes.extend(
            Shape(slow=horizon, fast=horizon, rho=rho, curvature=curvature, damage_exponent=tau)
            for tau in damage_grid
            for curvature in CURVATURE_GRID
            for rho in RHO_GRID
            for horizon in FAST_GRID
        )
    return tuple(shapes)
