# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Interference-limited evidence: a two-state surrogate with composition-driven forgetting.

Training converts exposure into retained evidence, and two things limit how much sticks. A bucket's
unique pool saturates, so extra epochs buy progressively less. And evidence acquired in the stable
phase decays during the decay phase in proportion to how much *other* material is trained there.

That second mechanism is the point of this module. Every acquisition-forgetting state tried before on
this problem forgets as a function of elapsed time, which assigns identical decay to two policies with
the same total exposure and the same phase lengths. Those policies are exactly the fixed-aggregate
fiber that carries the whole two-phase question, so a time-driven decay cannot separate them. Here the
retention factor reads the phase-1 mixture instead, so a policy that concentrates a bucket late both
adds evidence and shields it.

Setting the interference rate to zero collapses the state to ``1 - exp(-rho * total_epochs)``, a
function of cumulative dose alone. The phase-weighted-dose null is therefore nested exactly rather than
approximately, and it is reachable inside the frozen parameter grid.

The head is linear in the state given the two nonlinear parameters, so fitting profiles the head out
with a bounded least-squares solve and searches the nonlinear pair on a grid. That keeps the
joint-versus-independent comparison exact: identical form, identical head, identical folds, and the
only difference is whether one ``theta`` serves every target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
from scipy.optimize import lsq_linear

# Amplitudes are bounded rather than merely non-negative so that a rank-deficient design cannot be
# solved by two enormous cancelling coefficients. The limit is far above any physically meaningful
# BPB-per-unit-evidence slope.
AMPLITUDE_LIMIT = 50.0
PREDICTION_SCALE_LIMIT = 100.0

RHO_GRID = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
MU_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
# Infinity is the exponential acquisition curve the family started from; the finite values let the
# curve approach saturation as a power of exposure instead.
CURVATURE_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, math.inf)
HEAD_RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


@dataclass(frozen=True)
class Geometry:
    """Physical panel constants: epochs per unit weight in each phase, families, phase lengths."""

    c0: np.ndarray
    c1: np.ndarray
    phase_1_fraction: float
    family_index: np.ndarray

    @property
    def n_domains(self) -> int:
        return len(self.c0)

    @property
    def n_families(self) -> int:
        return int(np.unique(self.family_index).size)


class InterferenceLaw(StrEnum):
    """What phase-1 quantity drives the loss of phase-0 evidence.

    ``ABSOLUTE`` charges a bucket for every phase-1 token spent elsewhere. It is the literal reading of
    interference, and it is also not tied-neutral: on a one-phase policy the retention factor still
    varies with the mixture, so the interference rate can be identified from the shape of the tied
    response instead of from phase order. That is a confound, not a feature.

    ``SHARE_DROP`` charges a bucket only for the fall in its own representation between the phases. It is
    exactly one on any tied policy, so it contributes nothing to the aggregate response and the rate is
    identified from asymmetric policies alone. It is also one-sided, and on a simplex every share that
    rises is exactly some other bucket's share that falls, so this law can only ever subtract evidence.
    It cannot produce a two-phase gain on a fixed-aggregate fiber at all.

    ``RECENCY_EXPOSURE`` keeps tied-neutrality but stops treating the factor as a survival probability.
    A share that falls between the phases discounts the bucket's early exposure; a share that rises
    credits it, because the late material reactivates what was learned earlier. It is the two-sided
    version of the one-sided reactivation state, it is bounded because it acts on exposure rather than
    on acquired mass, and it is the only one of the three that is both tied-neutral and able to move a
    fixed-aggregate contrast in either direction.
    """

    ABSOLUTE = "absolute"
    SHARE_DROP = "share_drop"
    RECENCY_EXPOSURE = "recency_exposure"


@dataclass(frozen=True)
class Shape:
    """The shared nonlinear transition parameters.

    ``curvature`` controls how the acquisition curve approaches saturation. Infinity is the exponential
    ``1 - exp(-rho x)``; finite values give ``1 - (1 + rho x / nu)^-nu``, which approaches saturation as
    a power of exposure rather than exponentially. The exponential is the ``nu -> inf`` limit of the
    power form, so the two are nested exactly and the extension cannot fit worse than the original.
    Power-law approach is also the form the scaling-law literature supports, and an exponential
    acquisition curve turns out not to reach the WSD80 tied response at any ``(rho, mu)``.
    """

    rho: float
    interference: float
    law: InterferenceLaw = InterferenceLaw.ABSOLUTE
    curvature: float = math.inf


@dataclass(frozen=True)
class Head:
    """One target's linear response over the shared state."""

    intercept: float
    family_benefit: np.ndarray
    family_damage: np.ndarray
    bucket_departure: np.ndarray


@dataclass(frozen=True)
class Model:
    shape: Shape
    geometry: Geometry
    head: Head
    ridge: float

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return design_matrix(weights, self.geometry, self.shape) @ coefficient_vector(self.head)

    @property
    def phase_blind(self) -> Model:
        """The exact phase-weighted-dose ablation: same head, interference switched off."""
        return replace(self, shape=replace(self.shape, interference=0.0))


def phase_epochs(weights: np.ndarray, geometry: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """Materialized epochs of each bucket inside each phase."""
    return geometry.c0 * weights[:, 0, :], geometry.c1 * weights[:, 1, :]


def interference_load(weights: np.ndarray, geometry: Geometry, law: InterferenceLaw) -> np.ndarray:
    """Phase-1 quantity that competes with each bucket's stored evidence, in units of total tokens.

    Signed for ``RECENCY_EXPOSURE``: a bucket whose share rises late gets a negative load, which credits
    rather than discounts its early exposure.
    """
    if law is InterferenceLaw.ABSOLUTE:
        return geometry.phase_1_fraction * (1.0 - weights[:, 1, :])
    difference = weights[:, 0, :] - weights[:, 1, :]
    if law is InterferenceLaw.SHARE_DROP:
        return geometry.phase_1_fraction * np.maximum(difference, 0.0)
    return geometry.phase_1_fraction * difference


def acquired_share(exposure: np.ndarray, shape: Shape) -> np.ndarray:
    """Fraction of a bucket's learnable content acquired from `exposure` epochs, in [0, 1)."""
    if math.isinf(shape.curvature):
        return -np.expm1(-shape.rho * exposure)
    return 1.0 - (1.0 + shape.rho * exposure / shape.curvature) ** (-shape.curvature)


def unacquired_share(exposure: np.ndarray, shape: Shape) -> np.ndarray:
    """The complement of `acquired_share`, written directly to stay accurate for small exposure."""
    if math.isinf(shape.curvature):
        return np.exp(-shape.rho * exposure)
    return (1.0 + shape.rho * exposure / shape.curvature) ** (-shape.curvature)


def evidence_state(weights: np.ndarray, geometry: Geometry, shape: Shape) -> np.ndarray:
    """Terminal retained evidence per bucket, bounded in [0, 1).

    Under the two retention laws, phase-0 exposure acquires a saturating share, the phase-1 mixture
    decides how much of that survives, and phase-1 exposure acquires a share of whatever remains
    unlearned. Under the recency law the factor discounts or credits the phase-0 exposure itself, which
    keeps the state bounded without needing the factor to be a probability.
    """
    epochs_0, epochs_1 = phase_epochs(weights, geometry)
    factor = np.exp(-shape.interference * interference_load(weights, geometry, shape.law))
    if shape.law is InterferenceLaw.RECENCY_EXPOSURE:
        return acquired_share(factor * epochs_0 + epochs_1, shape)
    acquired = acquired_share(epochs_0, shape)
    return 1.0 - (1.0 - factor * acquired) * unacquired_share(epochs_1, shape)


def overexposure(weights: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Epochs consumed beyond one pass of each bucket's unique pool. A function of total exposure only."""
    epochs_0, epochs_1 = phase_epochs(weights, geometry)
    return np.maximum(epochs_0 + epochs_1 - 1.0, 0.0)


def family_sums(values: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Mean over the buckets of each predeclared family."""
    families = np.unique(geometry.family_index)
    return np.column_stack([values[:, geometry.family_index == family].mean(axis=1) for family in families])


def design_matrix(weights: np.ndarray, geometry: Geometry, shape: Shape) -> np.ndarray:
    """Columns: intercept, family evidence, family over-exposure, per-bucket evidence departures.

    Evidence columns are negated so that every fitted amplitude is non-negative under the sign
    convention "more evidence lowers BPB, more over-exposure raises it".
    """
    state = evidence_state(weights, geometry, shape)
    damage = overexposure(weights, geometry)
    return np.column_stack(
        [
            np.ones(len(weights)),
            -family_sums(state, geometry),
            family_sums(damage, geometry),
            -state,
        ]
    )


def coefficient_vector(head: Head) -> np.ndarray:
    return np.concatenate([[head.intercept], head.family_benefit, head.family_damage, head.bucket_departure])


def unpack_head(coefficients: np.ndarray, geometry: Geometry) -> Head:
    n_families = geometry.n_families
    return Head(
        intercept=float(coefficients[0]),
        family_benefit=coefficients[1 : 1 + n_families].copy(),
        family_damage=coefficients[1 + n_families : 1 + 2 * n_families].copy(),
        bucket_departure=coefficients[1 + 2 * n_families :].copy(),
    )


def _penalty_rows(geometry: Geometry, ridge: float) -> np.ndarray:
    """Shrink bucket departures hard and family amplitudes barely; never shrink the intercept.

    The family amplitude carries the pooled signal and the bucket departure is the deviation from it,
    which is the hierarchical pooling this project already uses. A single ridge on every column would
    instead shrink the pooled level toward zero.
    """
    n_families = geometry.n_families
    width = 1 + 2 * n_families + geometry.n_domains
    scales = np.concatenate(
        [
            [0.0],
            np.full(2 * n_families, 1e-3),
            np.ones(geometry.n_domains),
        ]
    )
    return np.sqrt(ridge) * np.diag(scales)[:width, :width]


def _head_bounds(width: int, geometry: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """Amplitudes are non-negative and bounded; the intercept is free.

    Bucket departures are deviations from their family level and may take either sign; whether the
    total amplitude stays non-negative is reported as a diagnostic rather than clipped here.
    """
    lower = np.full(width, 0.0)
    upper = np.full(width, AMPLITUDE_LIMIT)
    lower[0], upper[0] = -np.inf, np.inf
    lower[1 + 2 * geometry.n_families :] = -AMPLITUDE_LIMIT
    return lower, upper


def solve_head(
    design: np.ndarray,
    response: np.ndarray,
    geometry: Geometry,
    ridge: float,
) -> Head:
    """Bounded ridge solve for one target."""
    penalty = _penalty_rows(geometry, ridge)
    augmented = np.vstack([design, penalty])
    target = np.concatenate([response, np.zeros(len(penalty))])
    lower, upper = _head_bounds(design.shape[1], geometry)

    solved = lsq_linear(augmented, target, bounds=(lower, upper), method="trf", max_iter=500)
    predicted = design @ solved.x
    assert np.all(np.isfinite(predicted)), "bounded solve produced non-finite predictions"
    limit = PREDICTION_SCALE_LIMIT * max(float(np.max(np.abs(response))), 1e-12)
    assert np.max(np.abs(predicted)) <= limit, f"bounded solve produced predictions above {limit:.4g}"
    return unpack_head(solved.x, geometry)


def solve_coefficients_batch(
    design: np.ndarray,
    responses: np.ndarray,
    geometry: Geometry,
    ridge: float,
) -> np.ndarray:
    """Coefficients for many targets that share one design and one row mask, as a `width x targets` array.

    The bound constraints are inactive for most targets, and when the unconstrained ridge solution is
    already feasible it is the exact solution of the bounded problem. So solve every target at once
    through a single factorization and fall back to the bounded solver only for the columns that come
    back infeasible. Same answer, far fewer solves.

    Returning the raw array rather than `Head` objects matters: the selection loop evaluates this tens
    of thousands of times and packing and unpacking dataclasses per target dominated the runtime.
    """
    penalty = _penalty_rows(geometry, ridge)
    augmented = np.vstack([design, penalty])
    stacked = np.vstack([responses, np.zeros((len(penalty), responses.shape[1]))])
    coefficients, *_ = np.linalg.lstsq(augmented, stacked, rcond=None)

    lower, upper = _head_bounds(design.shape[1], geometry)
    infeasible = np.flatnonzero(
        ~np.all((coefficients >= lower[:, None] - 1e-12) & (coefficients <= upper[:, None] + 1e-12), axis=0)
    )
    for column in infeasible:
        coefficients[:, column] = coefficient_vector(solve_head(design, responses[:, column], geometry, ridge))
    return coefficients


def solve_heads_batch(
    design: np.ndarray,
    responses: np.ndarray,
    geometry: Geometry,
    ridge: float,
) -> list[Head]:
    coefficients = solve_coefficients_batch(design, responses, geometry, ridge)
    return [unpack_head(coefficients[:, column], geometry) for column in range(responses.shape[1])]


def shape_grid(
    rho_grid: tuple[float, ...] = RHO_GRID,
    mu_grid: tuple[float, ...] = MU_GRID,
    law: InterferenceLaw = InterferenceLaw.ABSOLUTE,
    curvature_grid: tuple[float, ...] = (math.inf,),
) -> tuple[Shape, ...]:
    return tuple(
        Shape(rho=rho, interference=mu, law=law, curvature=curvature)
        for curvature in curvature_grid
        for rho in rho_grid
        for mu in mu_grid
    )


def tied_weights(single_phase: np.ndarray) -> np.ndarray:
    """Lift a one-phase mixture to the two-phase representation used everywhere else."""
    return np.stack([single_phase, single_phase], axis=1)
