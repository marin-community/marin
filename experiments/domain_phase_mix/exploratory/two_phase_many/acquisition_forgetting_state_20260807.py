# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A training state that acquires and forgets, with one memory rate governing both.

Every earlier candidate on this problem reweights cumulative dose. The two-timescale family looked
different but is not: with two distinct horizons the pair of effective mixtures is an invertible linear
map of the original phase coordinates, so it can represent a phase effect without ever claiming a
mechanism for one. This module claims a mechanism.

Per bucket the state is the fraction of that bucket's learnable content currently held, and it obeys

    ds_i/du = rho_i R_k,i (1 - s_i) - lambda s_i

over run fraction u, where R_k,i is the epoch delivery rate of bucket i during phase k. Acquisition is
proportional to how much material is arriving and to remaining headroom; forgetting is proportional to
how much is held and runs whether or not the bucket is being trained. Delivery rate is constant inside
a phase, so each phase is a linear ODE with a closed-form solution and the run is two of them composed.

The same lambda discounts repetition damage. Over-exposure is delivered at a known rate once a bucket's
cumulative epochs pass one, and what it does to the weights fades on the same memory:

    X_i = integral of (over-exposure delivery rate) * exp(-lambda (1 - u)) du,   damage_i = X_i^tau

That is not decoration. SUR-093 found that reading benefit from a horizon-weighted exposure while
reading damage from raw total epochs made early weight about four times cheaper to cut on the damage
side than on the benefit side, a free lunch that pinned every fitted optimum to the support boundary.
Discounting both against the same memory removes the mismatch by construction rather than by fitting a
second horizon to cancel it.

What makes the family falsifiable is the null it nests. At lambda = 0 the state integrates to
``1 - exp(-rho (g0 + g1))`` and the discounted over-exposure to ``max(0, e - 1)``, both exact functions
of cumulative dose alone. A single-index surrogate assigns a two-phase policy the same loss as the tied
policy sharing its index, so it can never beat the best tied policy; the WSD80 panel measures a
0.009594 BPB gain over that entire class. lambda = 0 sits at an interior grid point, so the refuted
family is reachable and lambda is exactly the parameter that has to carry the advantage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

# Below this rate the closed forms are evaluated through their lambda -> 0 limits, which are the
# single-index null. The threshold is where the series and the direct expression agree to well past
# double precision on this panel's time scales.
NEGLIGIBLE_RATE = 1e-9
# Amplitudes are bounded rather than merely non-negative so a near-collinear design cannot be solved by
# two enormous cancelling coefficients. Far above any physical BPB-per-unit-evidence slope.
AMPLITUDE_LIMIT = 50.0


@dataclass(frozen=True)
class Geometry:
    """Panel constants: epochs per unit weight over the whole run, and the decay phase's length."""

    epochs_per_unit_weight: np.ndarray
    phase_1_fraction: float

    @property
    def phase_0_fraction(self) -> float:
        return 1.0 - self.phase_1_fraction

    @property
    def n_domains(self) -> int:
        return len(self.epochs_per_unit_weight)


@dataclass(frozen=True)
class Shape:
    """The state's two rates, the readout's exponents, and the repetition exponent.

    ``rho`` and ``forgetting`` govern the state. ``exponent`` and ``offset`` govern how retained content
    is read out as loss: a power law in retained epochs, per domain. Measurement on this project already
    settled that an exponential aggregate cannot reach the WSD80 tied response while an inverse power
    can, so the saturation lives in the readout rather than in the state's approach to equilibrium.

    ``forgetting`` carries one rate per retained component. A single rate cannot serve this problem:
    measurement on the WSD80 panel showed that any rate large enough for phase order to matter also
    erases the stable phase, because it runs for four fifths of the training run. The two benefit
    columns then correlate at 0.963 and the state stops depending on early exposure at all. Several
    components at different rates let slow capability hold what was learned early while fast capability
    tracks what arrived late. That is not the two-horizon reweighting this project already rejected: an
    effective-mixture pair is a linear map of the phase coordinates, whereas these states are nonlinear
    compositions of two phase solves and do not factor through any index.
    """

    rho: float
    forgetting: tuple[float, ...]
    exponent: tuple[float, ...]
    offset: float
    damage_exponent: float

    @property
    def single_index(self) -> bool:
        """True at the exact phase-weighted-dose null, which the impossibility argument refutes."""
        return all(rate <= NEGLIGIBLE_RATE for rate in self.forgetting)


def delivery_rates(weights: np.ndarray, geometry: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """Epochs of each bucket delivered per unit run fraction, inside each phase.

    Constant within a phase because the mixture is. Multiplying by the phase length recovers the epochs
    materialized in that phase, so ``rate_0 * T0 + rate_1 * T1`` is total epochs.
    """
    scale = geometry.epochs_per_unit_weight
    return scale * weights[:, 0, :], scale * weights[:, 1, :]


def _relax(start: np.ndarray, target: np.ndarray, rate: np.ndarray, duration: float) -> np.ndarray:
    """Exponential relaxation of a first-order linear ODE from `start` toward `target`."""
    return target + (start - target) * np.exp(-rate * duration)


def unheld_fraction(weights: np.ndarray, geometry: Geometry, rho: float, forgetting: float) -> np.ndarray:
    """Fraction of each bucket's learnable content NOT held at the end of the run.

    Two phases composed. Within a phase the equilibrium held fraction is
    ``acquisition / (acquisition + forgetting)`` and the approach rate is their sum, so a bucket trained
    harder both aims higher and gets there faster. Forgetting keeps running while the bucket is absent,
    which is what makes late material worth more than the same material early.

    Carried as the complement because the readout needs its logarithm, and a bucket whose pool is small
    relative to the run saturates hard enough that the held fraction rounds to one.
    """
    rate_0, rate_1 = delivery_rates(weights, geometry)
    approach_0 = rho * rate_0 + forgetting
    approach_1 = rho * rate_1 + forgetting
    # Equilibrium unheld fraction: zero without forgetting, so the state fills completely.
    equilibrium_0 = forgetting / approach_0
    equilibrium_1 = forgetting / approach_1
    after_phase_0 = _relax(np.ones_like(rate_0), equilibrium_0, approach_0, geometry.phase_0_fraction)
    return _relax(after_phase_0, equilibrium_1, approach_1, geometry.phase_1_fraction)


def held_fraction(weights: np.ndarray, geometry: Geometry, rho: float, forgetting: float) -> np.ndarray:
    return 1.0 - unheld_fraction(weights, geometry, rho, forgetting)


def retained_epochs(weights: np.ndarray, geometry: Geometry, rho: float, forgetting: float) -> np.ndarray:
    """Exposure that would have produced the surviving state had nothing been forgotten.

    The readout needs a quantity in the units the scaling law is written in, and the state is a
    saturating fraction. Inverting the no-forgetting acquisition curve puts it back on the exposure
    scale: retained epochs are what a run with perfect memory would have had to deliver to end up here.

    This is the step that keeps the null exact. At zero forgetting the inversion returns cumulative
    epochs identically, so the readout becomes a power law in total dose, which is the aggregate form
    the cross-scale sweep selected, and the single-index impossibility applies to it unchanged.
    """
    rate_0, rate_1 = delivery_rates(weights, geometry)
    if forgetting <= NEGLIGIBLE_RATE:
        return rate_0 * geometry.phase_0_fraction + rate_1 * geometry.phase_1_fraction
    return -np.log(unheld_fraction(weights, geometry, rho, forgetting)) / rho


def _discounted_span(rate: np.ndarray, start: np.ndarray, end: np.ndarray, forgetting: float) -> np.ndarray:
    """Integral of `rate * exp(-forgetting * (1 - u))` over `u` in [start, end], zero where end <= start.

    At `forgetting = 0` this is the undiscounted amount delivered, `rate * (end - start)`, which is the
    single-index damage term the earlier families used.
    """
    span = np.maximum(end - start, 0.0)
    if forgetting <= NEGLIGIBLE_RATE:
        return rate * span
    # Written through expm1 on the span so the small-span case stays accurate.
    return (rate / forgetting) * np.exp(-forgetting * (1.0 - end)) * -np.expm1(-forgetting * span)


def discounted_overexposure(weights: np.ndarray, geometry: Geometry, forgetting: float) -> np.ndarray:
    """Repetition delivered beyond one pass of a bucket's unique pool, discounted to the end of the run.

    Cumulative epochs rise linearly inside each phase, so the moment a bucket starts repeating itself is
    a closed-form crossing time and the repeated portion is one or two spans of constant delivery.
    """
    rate_0, rate_1 = delivery_rates(weights, geometry)
    boundary = geometry.phase_0_fraction
    epochs_0 = rate_0 * boundary

    # Crossing inside phase 0 happens only where that phase alone exhausts the pool.
    crosses_early = epochs_0 > 1.0
    start_0 = np.where(
        crosses_early, np.divide(1.0, rate_0, out=np.full_like(rate_0, np.inf), where=rate_0 > 0), boundary
    )
    early = _discounted_span(rate_0, np.minimum(start_0, boundary), np.full_like(start_0, boundary), forgetting)

    # Otherwise the crossing is inside phase 1, at the point where the running total reaches one.
    remaining = np.maximum(1.0 - epochs_0, 0.0)
    offset = np.divide(remaining, rate_1, out=np.full_like(rate_1, np.inf), where=rate_1 > 0)
    start_1 = np.minimum(boundary + offset, 1.0)
    late = _discounted_span(rate_1, start_1, np.ones_like(start_1), forgetting)
    return early + late


def damage(weights: np.ndarray, geometry: Geometry, shape: Shape) -> np.ndarray:
    """Repetition harm, discounted on the slowest component's memory.

    Overfitting is the persistent consequence of a run, so it is read off the longest-lived state rather
    than given a free horizon of its own. Sharing a rate with the slow benefit column is also what keeps
    early weight from being cheaper to cut on the damage side than on the benefit side, which is the
    asymmetry that pinned every earlier candidate's optimum to the support boundary.
    """
    return discounted_overexposure(weights, geometry, min(shape.forgetting)) ** shape.damage_exponent


def design_matrix(weights: np.ndarray, geometry: Geometry, shape: Shape) -> np.ndarray:
    """Intercept, an inverse power of retained epochs per bucket, and repetition damage per bucket.

    Both non-intercept blocks fall as their driver improves and rise as it worsens, so every amplitude
    is non-negative under one sign convention. Columns identically zero across the panel are dropped: a
    bucket whose pool is never exhausted contributes no damage column, and keeping it would leave the
    design rank deficient for the solver to paper over.
    """
    exponent = np.asarray(shape.exponent, dtype=float)
    benefit = [
        (retained_epochs(weights, geometry, shape.rho, rate) + shape.offset) ** -exponent for rate in shape.forgetting
    ]
    repetition = damage(weights, geometry, shape)
    columns = np.column_stack([np.ones(len(weights)), *benefit, repetition])
    return columns[:, np.any(np.abs(columns) > 0.0, axis=0)]


def solve(design: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Least squares with the sign convention imposed on every amplitude but not on the intercept.

    The unconstrained solution is the exact answer to the bounded problem whenever it is feasible, which
    it is for most shapes, so the bounded solver is a fallback rather than the default path.
    """
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    if np.all(coefficients[1:] >= 0.0) and np.all(coefficients[1:] <= AMPLITUDE_LIMIT):
        return coefficients
    width = design.shape[1]
    lower = np.concatenate([[-np.inf], np.zeros(width - 1)])
    upper = np.concatenate([[np.inf], np.full(width - 1, AMPLITUDE_LIMIT)])
    return lsq_linear(design, response, bounds=(lower, upper), method="trf").x
