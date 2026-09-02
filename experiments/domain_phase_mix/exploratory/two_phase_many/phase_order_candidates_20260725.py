# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "scipy",
# ]
# ///
"""Conditional phase-order candidates and reference baselines.

Every candidate predicts the *paired* contrast

    Delta(a, d) = L(a, d) - L(a, 0) = O(a, d) + C(a, d),

with ``O`` odd and ``C`` even in ``d``. The odd and even blocks are always
returned separately so that the split can be tested against the antithetic
panel, where ``O`` and ``C`` are observed individually.

Exposure conventions
--------------------
``c0[i]`` and ``c1[i]`` convert a phase weight into simulated epochs for bucket
``i``. Because ``c1 = c0 * (1 - alpha) / alpha`` in these swarms, total physical
exposure depends only on the aggregate:

    E_i(a) = c0_i * p0_i + c1_i * p1_i = (c0_i / alpha) * a_i,

so phase order never changes physical exposure. That is what makes the paired
contrast a clean phase estimand.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

EPSILON = 1e-12


@dataclass(frozen=True)
class Geometry:
    """Fixed, target-independent coordinates of a paired panel."""

    alpha: float
    aggregate: np.ndarray  # (n, k)
    contrast: np.ndarray  # (n, k)
    c0: np.ndarray  # (k,) phase-0 epochs per unit weight
    c1: np.ndarray  # (k,) phase-1 epochs per unit weight
    family_index: np.ndarray  # (k,) family id per bucket
    family_count: int

    @property
    def phase0(self) -> np.ndarray:
        return self.aggregate - (1.0 - self.alpha) * self.contrast

    @property
    def phase1(self) -> np.ndarray:
        return self.aggregate + self.alpha * self.contrast

    @property
    def phase_tv(self) -> np.ndarray:
        return 0.5 * np.abs(self.contrast).sum(axis=1)

    @property
    def epochs(self) -> np.ndarray:
        """Total simulated epochs per bucket, a function of the aggregate only."""
        return (self.c0 / self.alpha) * self.aggregate

    def family_pool(self, values: np.ndarray) -> np.ndarray:
        """Sum bucket-indexed columns within each family."""
        out = np.zeros((values.shape[0], self.family_count))
        for f in range(self.family_count):
            out[:, f] = values[:, self.family_index == f].sum(axis=1)
        return out

    def reflect(self) -> Geometry:
        """Same aggregate, reversed contrast."""
        return Geometry(
            alpha=self.alpha,
            aggregate=self.aggregate,
            contrast=-self.contrast,
            c0=self.c0,
            c1=self.c1,
            family_index=self.family_index,
            family_count=self.family_count,
        )


# --------------------------------------------------------------------------
# Even-cost functionals. Each maps a geometry to a single nonnegative column.
# --------------------------------------------------------------------------


def even_euclidean(geometry: Geometry) -> np.ndarray:
    return 0.5 * geometry.alpha * (1 - geometry.alpha) * (geometry.contrast**2).sum(axis=1)


def even_fisher_chi2(geometry: Geometry) -> np.ndarray:
    """Retained survivor: Fisher-weighted even asymmetry cost.

    Equals the leading term of the mixture Jensen gap in KL, so it carries the
    natural alpha(1-alpha) scaling and is dimensionless.
    """
    weight = np.clip(geometry.aggregate, EPSILON, None)
    return 0.5 * geometry.alpha * (1 - geometry.alpha) * (geometry.contrast**2 / weight).sum(axis=1)


def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, None)
    q = np.clip(q, EPSILON, None)
    ratio = np.where(p > EPSILON, np.log(np.clip(p, EPSILON, None) / q), 0.0)
    return (p * ratio).sum(axis=1)


def even_kl_jensen(geometry: Geometry) -> np.ndarray:
    """Exact mixture Jensen gap ``alpha KL(p0||a) + (1-alpha) KL(p1||a)``.

    Supra-quadratic growth in the contrast radius is a prediction of the shape,
    not a fitted exponent.
    """
    alpha = geometry.alpha
    return alpha * _kl(geometry.phase0, geometry.aggregate) + (1 - alpha) * _kl(geometry.phase1, geometry.aggregate)


def even_tv_power(power: float) -> Callable[[Geometry], np.ndarray]:
    def functional(geometry: Geometry) -> np.ndarray:
        return geometry.phase_tv**power

    return functional


def _overload(geometry: Geometry, threshold: float, starvation: float) -> np.ndarray:
    """Physical within-phase overload plus late-starvation load (not yet symmetrized)."""
    epochs0 = geometry.c0 * geometry.phase0 / geometry.alpha
    epochs1 = geometry.c1 * geometry.phase1 / (1.0 - geometry.alpha)
    overload = np.maximum(epochs0 - threshold, 0.0) ** 2 + np.maximum(epochs1 - threshold, 0.0) ** 2
    share = np.clip(geometry.phase1 / np.clip(geometry.aggregate, EPSILON, None), 0.0, None)
    starve = np.maximum(starvation - share, 0.0) ** 2 * geometry.aggregate
    return overload.sum(axis=1) + starve.sum(axis=1)


def even_boundary_overload(threshold: float = 1.0, starvation: float = 0.25) -> Callable[[Geometry], np.ndarray]:
    """Even part of a physical overload/starvation functional.

    Symmetrizing an arbitrary physical functional in ``d`` yields an admissible
    even cost by construction. Because overload depends on how far individual
    buckets are pushed toward repetition or toward zero late presence, it is
    close to direction-independent at fixed radius, which is the observed
    behaviour of the measured even cost.
    """

    def functional(geometry: Geometry) -> np.ndarray:
        forward = _overload(geometry, threshold, starvation)
        reverse = _overload(geometry.reflect(), threshold, starvation)
        tied = _overload(
            Geometry(
                alpha=geometry.alpha,
                aggregate=geometry.aggregate,
                contrast=np.zeros_like(geometry.contrast),
                c0=geometry.c0,
                c1=geometry.c1,
                family_index=geometry.family_index,
                family_count=geometry.family_count,
            ),
            threshold,
            starvation,
        )
        return 0.5 * (forward + reverse) - tied

    return functional


EVEN_FUNCTIONALS: dict[str, Callable[[Geometry], np.ndarray]] = {
    "euclidean": even_euclidean,
    "fisher_chi2": even_fisher_chi2,
    "kl_jensen": even_kl_jensen,
    "tv_squared": even_tv_power(2.0),
    "tv_cubed": even_tv_power(3.0),
    "boundary_overload": even_boundary_overload(),
}


# --------------------------------------------------------------------------
# Odd-field constructions. Each maps a geometry to an (n, p) odd design whose
# columns are exactly odd in the contrast.
# --------------------------------------------------------------------------


def odd_free_bucket(geometry: Geometry) -> np.ndarray:
    """Unrestricted 39-dimensional odd field (upper bound on odd capacity)."""
    return geometry.contrast


def odd_family_pooled(geometry: Geometry) -> np.ndarray:
    """Odd field constrained to the canonical family subspace."""
    return geometry.family_pool(geometry.contrast)


def odd_effective_exposure(geometry: Geometry) -> np.ndarray:
    """Effective-exposure DSP odd field: the epoch-scaled contrast.

    DSP values a bucket by phase-weighted exposure, so displacing mass late
    changes its effective exposure in proportion to its epochs-per-unit-weight.
    """
    scale = geometry.alpha * (1 - geometry.alpha)
    return (scale * geometry.c0 * geometry.contrast).sum(axis=1, keepdims=True)


def odd_marginal_value(tau: float) -> Callable[[Geometry], np.ndarray]:
    """Marginal-learnability transport, the declared prior-route reproduction.

    ``m_i(a) = 1 / (tau + E_i(a))`` is the remaining learnability of bucket ``i``
    at the aggregate. This is the PMVT form and is included as a reference,
    not as a new mechanism. ``tau`` is in simulated epochs and must be positive
    because unsampled buckets have exactly zero exposure.
    """
    assert tau > 0.0, "marginal-value transport needs a positive epoch offset"

    def design(geometry: Geometry) -> np.ndarray:
        m = 1.0 / (tau + geometry.epochs)
        scale = geometry.alpha * (1 - geometry.alpha)
        return (scale * m * geometry.contrast).sum(axis=1, keepdims=True)

    return design


def odd_retention_exchange(tau: float) -> Callable[[Geometry], np.ndarray]:
    """Retention-exchange field: marginal value reweighted per family.

    Physical claim: phase-0 evidence survives into the terminal state with a
    family-specific retention ``r_f``, so displacing bucket mass late changes
    retained evidence by ``alpha (1 - alpha) (1 - r_f) d_i``. The response is
    that displacement valued at the aggregate marginal learnability. With
    heterogeneous ``r_f`` the resulting field is not proportional to the
    aggregate gradient, which is what escapes the finite-potential-transport
    obstruction. One column per family carries ``(1 - r_f)``.
    """

    assert tau > 0.0, "retention exchange needs a positive epoch offset"

    def design(geometry: Geometry) -> np.ndarray:
        m = 1.0 / (tau + geometry.epochs)
        scale = geometry.alpha * (1 - geometry.alpha)
        return geometry.family_pool(scale * m * geometry.contrast)

    return design


# --------------------------------------------------------------------------
# Shared-curvature transport: one concave curve generates O and C jointly.
# --------------------------------------------------------------------------


def shared_curvature_blocks(geometry: Geometry, tau: float, retention: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (odd, even) designs implied by one saturating acquisition curve.

    With ``G(x) = 1 - exp(-x / tau)`` acting on retained exposure
    ``x_i = retention * c0_i * p0_i + c1_i * p1_i``, the paired contrast expands
    as an odd term in ``G'`` and an even term in ``G''`` at the *same* bucket.
    Both blocks carry the same per-bucket amplitude, so a single family-pooled
    coefficient vector fixes the odd and even amplitudes together. The ratio of
    the two is therefore a prediction with no free parameter.
    """
    retained_tied = retention * geometry.c0 * geometry.aggregate + geometry.c1 * geometry.aggregate
    displacement = (1.0 - geometry.alpha) * (1.0 - retention) * geometry.c0 * geometry.contrast
    curve_first = np.exp(-retained_tied / tau) / tau
    curve_second = -np.exp(-retained_tied / tau) / tau**2
    odd = geometry.family_pool(-curve_first * displacement)
    even = geometry.family_pool(-0.5 * curve_second * displacement**2)
    return odd, even


# --------------------------------------------------------------------------
# Ridge fitting with an explicit odd/even split
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockFit:
    odd_coef: np.ndarray
    even_coef: np.ndarray
    intercept: float
    l2: float

    def predict_parts(self, odd_design: np.ndarray, even_design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        odd = odd_design @ self.odd_coef if odd_design.size else np.zeros(len(even_design))
        even = (
            even_design @ self.even_coef + self.intercept
            if even_design.size
            else np.full(len(odd_design), self.intercept)
        )
        return odd, even

    def predict(self, odd_design: np.ndarray, even_design: np.ndarray) -> np.ndarray:
        odd, even = self.predict_parts(odd_design, even_design)
        return odd + even


def fit_blocks(
    odd_design: np.ndarray,
    even_design: np.ndarray,
    target: np.ndarray,
    l2_odd: float,
    l2_even: float = 0.0,
) -> BlockFit:
    """Ridge-fit odd and even blocks jointly with no intercept on the odd block.

    The odd block is penalized; the even block is not, because the even response
    is the well-identified part of the paired contrast and its columns are few.
    """
    n_odd = odd_design.shape[1] if odd_design.size else 0
    parts = [p for p in (odd_design, even_design) if p.size]
    design = np.hstack(parts) if parts else np.zeros((len(target), 0))
    design = np.hstack([design, np.ones((len(target), 1))])
    scale = np.maximum(np.sqrt((design**2).mean(axis=0)), 1e-12)
    scaled = design / scale
    penalty = np.full(design.shape[1], 1e-8)
    penalty[:n_odd] = l2_odd
    penalty[n_odd : design.shape[1] - 1] = max(l2_even, 1e-8)
    gram = scaled.T @ scaled + np.diag(penalty)
    # Even blocks can contain near-collinear radius powers, so use a
    # least-squares solve rather than an exact inverse.
    coef = np.linalg.lstsq(gram, scaled.T @ target, rcond=None)[0] / scale
    n_even = even_design.shape[1] if even_design.size else 0
    return BlockFit(
        odd_coef=coef[:n_odd],
        even_coef=coef[n_odd : n_odd + n_even],
        intercept=float(coef[-1]),
        l2=l2_odd,
    )


def grouped_folds(labels: Sequence[object] | np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-group-out folds over an arbitrary grouping label.

    A single-group label would produce an empty training set, which is a silent
    way to lose a validation gate, so it is rejected.
    """
    values = np.asarray(labels)
    distinct = sorted(set(values.tolist()))
    assert len(distinct) > 1, f"grouped_folds needs at least two groups, got {distinct}"
    folds = []
    for value in distinct:
        test = values == value
        folds.append((~test, test))
    return folds


def quantile_blocks(values: np.ndarray, n_blocks: int) -> np.ndarray:
    """Deterministic equal-count blocks, used to build grouped hold-out folds."""
    order = np.argsort(values, kind="stable")
    labels = np.empty(len(values), dtype=int)
    labels[order] = np.floor(np.arange(len(values)) * n_blocks / len(values)).astype(int)
    return labels


def aggregate_region_blocks(aggregate: np.ndarray, n_blocks: int) -> np.ndarray:
    """Block the aggregate anchors by their leading principal coordinate.

    Holding out a whole region of aggregate space is the paired-panel analogue of
    leave-anchor-out: it asks whether the phase response learned near one part of
    the mixture simplex predicts a part never seen during fitting.
    """
    centered = aggregate - aggregate.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    return quantile_blocks(centered @ right[0], n_blocks)
