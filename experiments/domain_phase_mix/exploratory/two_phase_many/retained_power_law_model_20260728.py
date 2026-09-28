# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A surrogate built from three mechanisms, one per term, and nothing else.

The incumbents fail on the 80/20 WSD StarCoder surface in the aggregate channel before they get
anywhere near phase order: exponential saturation cannot reproduce a power-law data-scaling response,
so the fitted one-phase curve misplaces the best constant mixture by ten aggregate points. Fixing that
is the first term. The other two carry phase order, and they have to be interactions rather than
reweightings, because a model in which the schedule enters only through a per-domain reweighted
cumulative exposure predicts exactly zero two-phase gain -- the tied policy class already sweeps the
whole effective-exposure simplex, so nothing is left for a schedule to win.

    L = b + sum_i A_i * (S_i + E0)^-a          benefit, a power law in retained token share
          + sum_i B_i * max(D_i - T, 0)^g      damage, from re-reading a finite pool
          + J * concentration                  cost of cramming a domain into one window

**Benefit.** Loss falls as a power of data seen, not exponentially, which is the one ingredient that
separates a sub-sigma fit from a three-sigma fit on the tied diagonal. The argument of the power law is
*token share*, not epochs, so that a single shared offset can serve domains whose pools differ by more
than an order of magnitude; epochs enter only where repetition is the mechanism. Exposure is *retained*
share: material studied in phase 0 survives according to the signed phase contrast, while phase-1
material receives a learned endpoint multiplier,
``S_i = exp(c*tanh(lambda*(w1_i-w0_i)/c))*alpha0*w0_i + m*alpha1*w1_i``. The contrast gate is what makes
the term an interaction -- the value of early data depends on the late mixture, which no reweighting of
cumulative exposure can express -- and it is the only term that can make a tilt *helpful*.

**Damage.** Re-reading a pool past a threshold hurts, as a power of the excess epochs. Damage is
charged on raw epochs rather than retained share because over-fitting a pool does not un-happen when
later data arrives.

**Concentration.** Splitting a fixed number of epochs unevenly across the two phases is worse than
spreading them, if within-window intensity has convex cost. That is a Jensen gap: exactly zero at the
tied policy, positive otherwise, and not a function of any linear combination of the two phase
mixtures. It charges for asymmetry and so competes with the retention gate, which is the
ordering-effect-against-asymmetry-cost trade expressed structurally.

Amplitudes are nonnegative with a free intercept, solved by bounded least squares. The five shape
parameters are selected on a discrete grid by out-of-fold error, jointly with the ridge, which is the
same protocol the incumbent Observatory models get. An earlier continuous shape search fitted the
training folds better and cross-validated three times worse, so the grid is doing real work.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from itertools import product

import numpy as np
from scipy.optimize import lsq_linear

# Shape grid. Deliberately coarse: the surface has 346 coordinates, and the failure mode being regularised
# against is a shape search that chases the extreme corners of the surface and then swings on
# held-out points near zero exposure.
# Nested cross-validation repeats this search inside every outer fold, so the grid is kept small
# enough that the honest estimate is affordable. Each axis keeps the spacing that mattered in the
# coarse sweep and drops the values that were never selected.
BENEFIT_EXPONENTS = (0.25, 0.5, 1.0)
# Two offsets, not four. A larger grid was tried and is worse under this model's robust head: on the
# StarCoder panel 2880 combinations gave 10.53 sigma RMSE and 2.31 sigma median residual where 720 give
# 9.67 and 1.41. An earlier sweep appeared to show the opposite, but it ran with the harness's
# least-squares head substituted for this model's, so it was measuring a different estimator. Under the
# real head the extra shapes buy selection variance rather than fit.
BENEFIT_OFFSETS = (0.01, 0.1)
DAMAGE_EXPONENTS = (1.5, 2.0, 3.0)
DAMAGE_THRESHOLDS = (0.0,)
# The top of this range exists for one specific failure. Policies that remove a domain from the late
# phase entirely are punished far harder in reality than a moderate gate can express: at retention 5
# the fifteen such coordinates on the StarCoder panel were under-predicted by a median of 34 training
# sigma. Allowing 10 halves that and also improves the interior, from 1.01 to 0.91 sigma. Raising the
# saturation bound instead reaches the boundary faster but degrades the interior, which is the wrong
# trade, so the bound is left alone.
# All five values stay, even though no sweep on either panel ever selected 1 or 5 and the count is larger
# than is comfortable. Thinning the grid on that evidence would be target-informed pruning: those sweeps
# scored full panels that the nested comparison then reports out-of-fold error on, so every outer test
# fold would have helped choose the candidate set, and re-selecting within the training folds does not
# undo it. A deleted value could have won an inner fold and changed that fold's prediction. Values above
# 10 stay out on a different basis: 20 and 40 are worse everywhere they were tried.
RETENTIONS = (0.0, 1.0, 2.5, 5.0, 10.0)
# How much a token seen in the late phase counts toward what is still present at the end, relative to
# its raw share of the budget. Fixing this at one was the model's largest single error at thirty-nine
# buckets. The gate discounted early exposure but late exposure always counted exactly its token
# weight, so the whole phase response had one free scalar; the leading incumbent searches a late
# multiplier over an eight-fold range alongside its forgetting rate, and that second dimension was the
# difference. Freeing it is worth 12 percent of out-of-fold error on the Delphi 3e18 uncheatable panel
# and 12 percent on precisely the moved policies a residual localization had indicted, while the tied
# policies this model already led on give up little. The original ceiling of 4 was appropriate for the
# 3e18 panel but bound every 300M fold. A follow-up sweep through 64 established an interior optimum at
# 8--16 on both 300M targets, with 64 clearly worse. The multiplier trades against the benefit offset,
# which sets the exposure scale it rescales, and the two are selected together.
LATE_MULTIPLIERS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
# Ridge zero is not offered. Every multi-member family column is exactly the sum of its member columns,
# so the design is rank-deficient by construction and only the penalty on the departures identifies it.
# With no penalty the bounded solve hit its iteration cap without converging and returned that iterate
# anyway, which scored as a valid fold: predictions near 7e10 and fold errors near 2.5e10 appeared in the
# grid landscape. Those were failed optimizations, and a shape that would have won at ridge zero under a
# real solve was being eliminated by one.
RIDGE_GRID = (1e-4, 1e-2, 1.0)
# The family-pooled ordering block is a selected structural choice, never a pinned one. It was removed
# entirely at one point because a two-fiber panel could not identify it and no honest criterion
# separated the case where it helps from the case where it hurts. The StarCoder panel now measures
# eight fixed-aggregate fibers rather than two, and across them the best contrast changes sign exactly
# at the one-phase optimum, which is the behaviour this block exists to represent. It is therefore
# offered to cross-validation again, as a grid dimension chosen by nested out-of-fold error alone.
ORDERING_CHANNELS = (False, True)
COLUMN_SCALE_FLOOR = 1e-12
# How far past the response scale a fitted value may go before the solve is treated as unusable rather
# than merely imprecise. Generous, because the point is to catch divergence by orders of magnitude, not
# to police a poor fit: the failure this exists for overshot by ten orders.
PREDICTION_SCALE_LIMIT = 100.0
# The signed gate exponent is bounded so a large retention value on a near-corner policy cannot produce
# a design column orders of magnitude out of scale. Tightening it to 1.5, with a correspondingly
# smaller retention grid, was tried and cost 39 percent on both fold protocols; because the phase
# columns are identically zero at tied policies and the tied-row error moved from 12.6 to 17.5 sigma,
# the damage was traceable to the gate alone rather than to any phase term.
GATE_CLIP = 4.0
# Squared error is the wrong loss for this surface. The response is visibly smooth and the scatter
# around it is measurement noise with occasional far outliers, so a least-squares head spends its
# amplitudes explaining points that carry no signal. The head is therefore fitted by iteratively
# reweighted least squares with a Huber weight, and the cut is set from the residual median absolute
# deviation rather than from an absolute BPB threshold, so the same setting transfers to panels whose
# noise scale differs. Setting the scale to None recovers plain least squares.
HUBER_SCALE = 2.5
HUBER_ITERATIONS = 50
HUBER_TOLERANCE = 1e-3
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class Shape:
    """The parameters the response is not linear in."""

    benefit_exponent: float
    benefit_offset: float
    damage_exponent: float
    damage_threshold: float
    retention: float
    late_multiplier: float
    ordering_channel: bool


def shape_grid() -> tuple[Shape, ...]:
    return tuple(
        Shape(*values)
        for values in product(
            BENEFIT_EXPONENTS,
            BENEFIT_OFFSETS,
            DAMAGE_EXPONENTS,
            DAMAGE_THRESHOLDS,
            RETENTIONS,
            LATE_MULTIPLIERS,
            ORDERING_CHANNELS,
        )
    )


@dataclass(frozen=True)
class Geometry:
    """Everything about a panel the model needs that does not depend on the policy.

    ``family_index`` assigns each domain to a group whose amplitude it shares by default. Writing a
    bucket amplitude as its family's value plus a bucket-specific excess, and penalising only the
    excess, is what lets thirty-nine buckets be fitted without spending thirty-nine free amplitudes on
    each term. With one domain per family the excess columns would duplicate the family columns
    exactly, so they are dropped and the parameterisation collapses to a plain per-domain amplitude.
    """

    c0: np.ndarray
    c1: np.ndarray
    phase_0_fraction: float
    family_index: np.ndarray | None = None

    @property
    def phase_1_fraction(self) -> float:
        return 1.0 - self.phase_0_fraction

    @property
    def families(self) -> np.ndarray:
        if self.family_index is None:
            return np.arange(len(self.c0))
        return np.asarray(self.family_index)

    @property
    def pooled_families(self) -> np.ndarray:
        """Families with more than one member, the only ones an excess column is identified for."""
        families = self.families
        counts = np.bincount(families, minlength=families.max() + 1)
        return np.flatnonzero(counts > 1)

    @property
    def excess_domains(self) -> np.ndarray:
        return np.flatnonzero(np.isin(self.families, self.pooled_families))


@dataclass(frozen=True)
class Fitted:
    shape: Shape
    ridge: float
    intercept: float
    coefficients: np.ndarray
    geometry: Geometry

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.intercept + design_matrix(weights, self.geometry, self.shape) @ self.coefficients

    @property
    def concentration(self) -> float:
        """Net coefficient on the concentration gap, whose two signed columns follow the damage block."""
        families = len(np.unique(self.geometry.families))
        start = 2 * (families + len(self.geometry.excess_domains))
        return float(self.coefficients[start] - self.coefficients[start + 1])


def retained_share(weights: np.ndarray, geometry: Geometry, retention: float, late_multiplier: float) -> np.ndarray:
    """Fraction of the token budget spent on each domain that is still present at the end.

    Phase-0 share is discounted by how thoroughly the late phase displaced that domain, so a domain
    kept present late retains what it learned early and one dropped for the whole late phase does not.
    Phase-1 share counts at ``late_multiplier`` times its token weight, because what a run sees last is
    more present in the final weights than what it saw first, and that recency is what makes any
    two-phase policy able to beat the tied policy at the same aggregate.

    At a tied policy this returns ``k * w`` with ``k = b0 + late * b1``, so every explicit phase column
    and the concentration gap stay identically zero there, which is the invariant two earlier gate
    designs broke. The tied response is not left completely alone, though: the benefit basis becomes
    ``(k*w + E0)**-a``, and pulling ``k`` out gives ``k**-a * (w + E0/k)**-a``, so an amplitude absorbs it
    only if the grid also offers offset ``E0/k``. At ``late`` 4 that would need 0.0062 or 0.0622 and
    ``BENEFIT_OFFSETS`` has neither. Selecting the multiplier therefore also searches tied-diagonal
    shapes, and its measured gain is not purely an off-diagonal effect.
    """
    phase_0, phase_1 = weights[:, 0, :], weights[:, 1, :]
    # The gate is a function of the phase contrast, and it has to be two-sided and centred at the tied
    # policy. Two earlier versions each failed one of those. Gating on ``1 - w1`` varies along the tied
    # diagonal, so the phase term silently rewrote the one-phase response and cost a factor of three on
    # tied-row error. Gating on ``max(w0 - w1, 0)`` is neutral at tied but one-sided: it can penalise
    # putting a domain early and can never reward putting it late, which collapsed the fitted optimum
    # onto the diagonal and destroyed the contrast-sign profile entirely. Gating on the signed contrast
    # is exactly one at every tied policy and moves in both directions.
    # Saturation is smooth rather than clipped. A hard clip bounds the factor but puts a kink in the
    # response wherever it engages, and that kink is visible in the fitted sheet as a fold across the
    # surface that the measurements have no counterpart for. Passing the exponent through a scaled
    # tanh keeps the same slope near the tied policy and the same bound far from it, without the
    # derivative jump.
    survival = np.exp(GATE_CLIP * np.tanh(retention * (phase_1 - phase_0) / GATE_CLIP))
    return survival * (geometry.phase_0_fraction * phase_0) + late_multiplier * geometry.phase_1_fraction * phase_1


def total_epochs(weights: np.ndarray, geometry: Geometry) -> np.ndarray:
    return geometry.c0 * weights[:, 0, :] + geometry.c1 * weights[:, 1, :]


def concentration_gap(weights: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Jensen gap of a convex within-window intensity cost, summed over domains."""
    alpha_0, alpha_1 = geometry.phase_0_fraction, geometry.phase_1_fraction
    intensity_0 = geometry.c0 * weights[:, 0, :] / alpha_0
    intensity_1 = geometry.c1 * weights[:, 1, :] / alpha_1
    average = alpha_0 * intensity_0 + alpha_1 * intensity_1

    def cost(value: np.ndarray) -> np.ndarray:
        return np.maximum(value - 1.0, 0.0) ** 2

    return (alpha_0 * cost(intensity_0) + alpha_1 * cost(intensity_1) - cost(average)).sum(axis=1)


def _family_totals(values: np.ndarray, geometry: Geometry) -> np.ndarray:
    families = geometry.families
    return np.stack([values[:, families == family].sum(axis=1) for family in np.unique(families)], axis=1)


def _signed(column: np.ndarray) -> np.ndarray:
    # Accepts either a single column or a block, and returns it followed by its negation.
    """A column and its negation, so a nonnegative head can still give it either sign.

    The even part of the phase response is not a cost everywhere. Measured on the eight fixed-aggregate
    fibers, twelve of seventy-nine antithetic pairs have a negative orientation-averaged term, all at
    aggregates 0.50 and above: the tied policy is locally worse than the average of the two arms. A
    head constrained to nonnegative coefficients cannot represent that at all, so every even column is
    entered twice with opposite signs rather than being forced to be a penalty.
    """
    return np.column_stack([column, -column])


def _hierarchical_block(values: np.ndarray, geometry: Geometry) -> np.ndarray:
    """One column per family, plus one per domain in a multi-member family.

    The family column carries the amplitude every member shares; the domain column carries how far
    that domain departs from it. Only the departures are penalised, so shrinkage pulls buckets toward
    their family rather than toward zero.
    """
    pooled = _family_totals(values, geometry)
    excess = geometry.excess_domains
    if not len(excess):
        return pooled
    return np.column_stack([pooled, values[:, excess]])


def marginal_phase_block(weights: np.ndarray, geometry: Geometry, shape: Shape) -> np.ndarray:
    """Ordering and asymmetry columns taken from the benefit function's own derivatives.

    Expanding the benefit term in the phase contrast ``d_i = w1_i - w0_i`` about the tied policy at the
    same aggregate gives a first-order piece proportional to ``(abar_i + E0)^-(a+1) * d_i`` and a
    second-order piece proportional to ``(abar_i + E0)^-(a+2) * d_i^2``. Using those instead of a free
    phase term makes the ordering coefficient scale with a bucket's marginal value, so it is large
    where a domain is undersupplied, small where the aggregate response has flattened, and it changes
    sign where the marginal value does. The eight measured fibers show exactly that: the best contrast
    is late-heavy below the one-phase optimum, zero at it, and early-heavy above it.

    Ordering columns are pooled by family and entered with both signs because the head is nonnegative.
    Every column here is exactly zero at the tied policy.

    Per-bucket ordering columns were tried and are worse, which is worth recording because the
    diagnostic that motivated them was sound. Pinned at its selected shape on the Delphi 3e18
    uncheatable panel, this pooled block beats every incumbent on the forty-two tied policies (0.00341
    against 0.00530) and loses to ``hierarchical_phase_replay`` on all four quartiles of moved policies,
    with all twelve of its worst rows above 0.39 phase total variation. Losing on exactly the rows the
    phase machinery exists to describe pointed at the single global retention scalar. Resolving the
    ordering block per bucket under family shrinkage takes the design from 100 columns to 256 on 280
    rows and makes those same moved rows worse, 0.01217 at best against 0.01087 here, whether the pooled
    amplitude is shrunk or free. The contrast columns are near-collinear across buckets because they are
    all driven by the same phase contrast, so per-bucket freedom buys conditioning trouble rather than
    resolution.
    """
    aggregate = geometry.phase_0_fraction * weights[:, 0, :] + geometry.phase_1_fraction * weights[:, 1, :]
    contrast = weights[:, 1, :] - weights[:, 0, :]
    scale = aggregate + shape.benefit_offset

    # The aggregate response has two terms, so its derivative in the contrast direction has two terms
    # as well, and both are needed. A single column built from the benefit derivative alone is monotone
    # in the aggregate and never changes sign: it can shrink the ordering effect at high exposure but
    # cannot reverse it. The measured fibers do reverse -- late-heavy below the one-phase optimum,
    # early-heavy above it -- because the damage derivative grows with exposure while the benefit
    # derivative decays. Fitting the two separately lets their difference cross zero where the net
    # marginal value does, which is what the sign flip is.
    benefit_slope = scale ** (-(shape.benefit_exponent + 1.0))
    epochs = total_epochs(weights, geometry)
    damage_slope = np.maximum(epochs - shape.damage_threshold, 0.0) ** max(shape.damage_exponent - 1.0, 0.0)

    ordering_benefit = _family_totals(benefit_slope * contrast, geometry)
    ordering_damage = _family_totals(damage_slope * contrast, geometry)
    asymmetry = (scale ** (-(shape.benefit_exponent + 2.0)) * contrast**2).sum(axis=1)
    # Sign freedom comes from entering each column and its negation, not from splitting it at zero.
    # A ``max(o, 0)`` / ``max(-o, 0)`` pair with independent nonnegative coefficients is kinked wherever
    # the column crosses zero, and for a family-pooled ordering column that locus is a curve across the
    # mixture square: the fitted sheet showed a fold there that the measurements have no counterpart
    # for. Entering ``o`` and ``-o`` leaves the fitted combination linear in ``o`` and therefore smooth,
    # while still reaching either sign.
    return np.column_stack([_signed(ordering_benefit), _signed(ordering_damage), _signed(asymmetry)])


def design_matrix(weights: np.ndarray, geometry: Geometry, shape: Shape) -> np.ndarray:
    """Columns whose coefficients enter the response linearly."""
    retained = retained_share(weights, geometry, shape.retention, shape.late_multiplier)
    benefit = (retained + shape.benefit_offset) ** (-shape.benefit_exponent)
    excess = np.maximum(total_epochs(weights, geometry) - shape.damage_threshold, 0.0)
    return np.column_stack(
        [
            _hierarchical_block(benefit, geometry),
            _hierarchical_block(excess**shape.damage_exponent, geometry),
            _signed(concentration_gap(weights, geometry)),
        ]
        + ([marginal_phase_block(weights, geometry, shape)] if shape.ordering_channel else [])
    )


def penalty_multipliers(geometry: Geometry, shape: Shape) -> np.ndarray:
    """Ridge multiplier per design column: zero on pooled terms, one on bucket departures.

    A hierarchical prior in this parameterisation means the family amplitude is free and the
    bucket-level departure from it is shrunk, which is how the incumbent leaders spend far fewer
    effective parameters at thirty-nine buckets than they have columns.
    """
    families = len(np.unique(geometry.families))
    excess = len(geometry.excess_domains)
    block = np.concatenate([np.zeros(families), np.ones(excess)])
    phase = np.concatenate([np.ones(4 * families), np.zeros(2)]) if shape.ordering_channel else np.empty(0)
    return np.concatenate([block, block, np.zeros(2), phase])


def _bounded_solve(
    augmented: np.ndarray, response: np.ndarray, columns: int, row_weights: np.ndarray | None
) -> np.ndarray:
    """One bounded least-squares solve, optionally with per-row weights."""
    if row_weights is not None:
        root = np.sqrt(row_weights)[:, None]
        augmented = augmented.copy()
        augmented[: len(row_weights)] *= root
        response = response.copy()
        response[: len(row_weights)] *= root[:, 0]
    bounds = (np.concatenate([[-np.inf], np.zeros(columns)]), np.full(augmented.shape[1], np.inf))
    solved = lsq_linear(augmented, response, bounds=bounds, method="trf", tol=1e-10, max_iter=200)
    # The solver's own success flag is not the right test. At the top of the ridge grid it reports failure
    # on every fold of the Delphi panel, yet the objective is identical to eight significant figures at
    # 200, 2000 and 10000 iterations with the same coefficients pinned at their bound: the iterate is
    # optimal and only the termination criterion never fires. Asserting on the flag would discard fits
    # that are correct. Where the flag is trustworthy the capped iterate matches the converged solution
    # to within 0.0 BPB on every prediction, so the flag adds nothing there either.
    #
    # What does need catching is a solve whose answer is unusable. Ridge zero made the design exactly
    # rank-deficient and produced predictions near 7e10 against a target near 1, and those scored as
    # ordinary cross-validation results. Ridge zero is gone, so this guard is a backstop rather than the
    # fix, and it tests the fitted scale, which is what scoring actually depends on.
    predicted = augmented[: len(response)] @ solved.x
    assert np.all(np.isfinite(predicted)), "bounded solve produced non-finite predictions"
    limit = PREDICTION_SCALE_LIMIT * max(float(np.max(np.abs(response))), 1e-12)
    assert np.max(np.abs(predicted)) <= limit, (
        f"bounded solve produced predictions up to {np.max(np.abs(predicted)):.3e} against a response "
        f"bounded by {np.max(np.abs(response)):.3e}; the design is not identified at this ridge"
    )
    return solved.x


def solve_head(
    design: np.ndarray,
    target: np.ndarray,
    ridge: float,
    multipliers: np.ndarray | None = None,
    huber_scale: float | None = HUBER_SCALE,
) -> tuple[float, np.ndarray]:
    """Nonnegative amplitudes with a free intercept, fitted robustly and column-scaled for the ridge.

    ``huber_scale`` defaults to the module setting, so every existing caller is unaffected. Passing None
    recovers plain bounded least squares, which costs a single solve instead of an iterated one. That is
    roughly three hundred times cheaper here and is the only way a grid this size is affordable as a
    screen; it must not be used for a reported fit, because a least-squares screen measures a different
    estimator, which has already inverted one conclusion on this model.
    """
    scale = np.maximum(np.abs(design).max(axis=0), COLUMN_SCALE_FLOOR)
    augmented = np.column_stack([np.ones(len(target)), design / scale])
    response = target
    if ridge > 0.0:
        weights = np.ones(design.shape[1]) if multipliers is None else np.asarray(multipliers, dtype=float)
        assert len(weights) == design.shape[1], "one penalty multiplier per design column"
        penalty = np.diag(np.concatenate([[0.0], np.sqrt(ridge * weights)]))
        augmented = np.vstack([augmented, penalty])
        response = np.concatenate([target, np.zeros(penalty.shape[0])])

    coefficients = _bounded_solve(augmented, response, design.shape[1], None)
    if huber_scale is not None:
        for _ in range(HUBER_ITERATIONS):
            residual = augmented[: len(target)] @ coefficients - target
            spread = MAD_TO_SIGMA * float(np.median(np.abs(residual - np.median(residual))))
            if spread <= 0.0:
                break
            cut = huber_scale * spread
            row_weights = np.minimum(1.0, cut / np.maximum(np.abs(residual), 1e-12))
            updated = _bounded_solve(augmented, response, design.shape[1], row_weights)
            # Convergence is tested on predictions, not coefficients. Collinear columns let the
            # coefficient vector keep drifting long after the fitted response has stopped moving, so a
            # coefficient criterion almost never fires and every solve runs to the cap -- measured as a
            # linear cost in the cap, 17 ms at one iteration against 352 ms at fifty. The criterion
            # that matters is whether predictions have settled relative to the residual scale.
            shift = float(np.max(np.abs(augmented[: len(target)] @ (updated - coefficients))))
            coefficients = updated
            if shift < HUBER_TOLERANCE * spread:
                break
    return float(coefficients[0]), coefficients[1:] / scale


def _shape_score(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: Geometry,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    shape: Shape,
) -> tuple[float, Shape, float]:
    design = design_matrix(weights, geometry, shape)
    if not np.all(np.isfinite(design)):
        return np.inf, shape, RIDGE_GRID[0]
    multipliers = penalty_multipliers(geometry, shape)
    best_score = np.inf
    best_ridge = RIDGE_GRID[0]
    for ridge in RIDGE_GRID:
        errors = []
        for train, test in folds:
            intercept, coefficients = solve_head(design[train], target[train], ridge, multipliers)
            errors.append(intercept + design[test] @ coefficients - target[test])
        score = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
        if score < best_score:
            best_score = score
            best_ridge = ridge
    return best_score, shape, best_ridge


def _shape_batch_scores(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: Geometry,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    shapes: tuple[Shape, ...],
) -> list[tuple[float, Shape, float]]:
    return [_shape_score(weights, target, geometry, folds, shape) for shape in shapes]


def _best_shape_and_ridge(
    scores: list[tuple[float, Shape, float]],
    shapes: tuple[Shape, ...],
) -> tuple[Shape, float]:
    """Select by score with the canonical grid order as the tie-breaker."""
    score_by_shape = {shape: (score, ridge) for score, shape, ridge in scores}
    best_shape = min(shapes, key=lambda shape: score_by_shape[shape][0])
    _, best_ridge = score_by_shape[best_shape]
    return best_shape, best_ridge


def fit(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: Geometry,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    workers: int = 1,
) -> Fitted:
    """Select shape and ridge by out-of-fold error on the given folds, then refit on everything."""
    if workers < 1:
        raise ValueError("workers must be positive")
    shapes = shape_grid()
    if workers == 1:
        scores = [_shape_score(weights, target, geometry, folds, shape) for shape in shapes]
    else:
        worker_count = min(workers, len(shapes))
        batch_count = min(len(shapes), worker_count * 4)
        batch_size = (len(shapes) + batch_count - 1) // batch_count
        batches = tuple(shapes[start : start + batch_size] for start in range(0, len(shapes), batch_size))
        scores = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_shape_batch_scores, weights, target, geometry, folds, batch) for batch in batches
            ]
            for future in as_completed(futures):
                scores.extend(future.result())

    best_shape, best_ridge = _best_shape_and_ridge(scores, shapes)

    design = design_matrix(weights, geometry, best_shape)
    intercept, coefficients = solve_head(design, target, best_ridge, penalty_multipliers(geometry, best_shape))
    return Fitted(shape=best_shape, ridge=best_ridge, intercept=intercept, coefficients=coefficients, geometry=geometry)


def without_phase_terms(model: Fitted) -> Fitted:
    """The same fit with both interaction channels off, for ablation.

    Retention zero makes phase-0 share count in full and a unit late multiplier makes phase-1 share count
    at its token weight, which together collapse the benefit term to a function of the aggregate mixture;
    dropping the concentration coefficient removes the only other term that distinguishes a schedule from
    its aggregate. What remains cannot express any two-phase gain.

    Both phase parameters have to be reset, not just retention. Leaving the multiplier at 2 or 4 keeps
    retained exposure at ``a0*w0 + late*a1*w1``, which still separates two schedules that share an
    aggregate, so the supposedly phase-free ablation would report a phase gain.
    """
    scale = model.geometry.phase_0_fraction + model.shape.late_multiplier * model.geometry.phase_1_fraction
    families = len(np.unique(model.geometry.families))
    block_width = families + len(model.geometry.excess_domains)
    coefficients = model.coefficients[: 2 * block_width + 2].copy()
    coefficients[:block_width] *= scale ** (-model.shape.benefit_exponent)
    coefficients[2 * block_width :] = 0.0
    return replace(
        model,
        shape=replace(
            model.shape,
            benefit_offset=model.shape.benefit_offset / scale,
            retention=0.0,
            late_multiplier=1.0,
            ordering_channel=False,
        ),
        coefficients=coefficients,
    )
