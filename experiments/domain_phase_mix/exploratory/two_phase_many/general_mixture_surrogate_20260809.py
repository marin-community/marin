# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A two-phase mixture surrogate with no semantic bucket assignment (GEN-001).

The audited WSD80 model assigned three of its five structures by hand, using knowledge that one domain
was code and the other was off-target text: an early-boundary kernel on code, repetition damage on code,
and a signed late-share control on broad. Production swarms do not supply that. Buckets arrive classified
by topic, with quality splits inside each topic, and nothing tells the surrogate which bucket matches the
eval. A model that needs to be told cannot be deployed.

Here every structure is a sum over all buckets, with nonlinear parameters pooled at the family (topic)
level and per-bucket freedom entering only as shrunk amplitude departures. Quality splits inside a topic
therefore share that topic's shape and differ only in how much of it they get, so the quality label is
used as a grouping and never as a feature.

Two of the three semantic structures were geometric all along and only needed rewriting as sums. Damage
fires through ``[E - 1]_+``, which is non-zero only for buckets whose pool is small relative to the run,
so a bucket that is never repeated contributes an identically zero column and needs no special-casing.
The early-boundary kernel corrects a failure mode that is likewise geometric: on WSD80 an early epoch
costs about four times what a late one does per unit weight (21.089 against 5.369), so starving the
stable phase is a cheap way to buy exposure and the fitted optimum collapses to zero early share. Which
buckets are exposed to that is readable from pool size relative to the run, not from what they contain —
see ``Panel.boundary_risk``, and note the correction recorded there about which quantity actually varies.

The signed late control needed a real generalisation rather than a rewrite. It read the off-target
domain's late share, which presumes a designated eval domain. Here it becomes a free-signed late share
per family: on a two-domain panel that spans the original exactly, because the shares sum to one, and on
thirty-nine buckets it becomes one term per topic with nothing designated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

# Excess epochs at which repetition harm saturates, measured on the 300M panel with WSD80 never
# consulted (stable 102-110 across seeds). Fixed rather than fitted: a free saturation scale with a wide
# bound is exactly what made the unbounded form explode, and this quantity is in epochs, which the term
# specification records as transferable across panels.
DAMAGE_KNEE = 105.0


@dataclass(frozen=True)
class Panel:
    """Everything the surrogate is allowed to know about a swarm.

    `family_index` is the topic grouping. Quality splits inside a topic share a family. No field
    identifies what a bucket contains or which bucket the eval is about.
    """

    weights: np.ndarray  # (rows, 2, buckets), phase-0 and phase-1 mixtures
    epochs_early: np.ndarray  # per-bucket epochs delivered at unit weight during the stable phase
    epochs_late: np.ndarray  # ... and during the decay phase
    family_index: np.ndarray

    @property
    def n_families(self) -> int:
        return int(np.unique(self.family_index).size)

    @property
    def total_epochs_per_unit(self) -> np.ndarray:
        return self.epochs_early + self.epochs_late

    def exposure(self, horizon: float) -> np.ndarray:
        """Epochs of every bucket as seen by a channel weighting the decay phase by `horizon`."""
        return self.total_epochs_per_unit * ((1.0 - horizon) * self.weights[:, 0, :] + horizon * self.weights[:, 1, :])

    def early_epochs(self) -> np.ndarray:
        return self.epochs_early * self.weights[:, 0, :]

    def n_exposure_strata(self, requested: int = 3) -> int:
        """Strata actually supportable: never more than there are distinct exposure rates.

        A two-bucket panel cut into three strata leaves one EMPTY, and its boundary scale is then fitted
        from no data and wanders freely -- observed at 504.6, 459.6, 0.545, 4.4, 0.322 and 328.6 across
        six WSD80 seeds before this was fixed.
        """
        return int(min(requested, np.unique(np.round(self.total_epochs_per_unit, 9)).size))

    def exposure_stratum(self, n_strata: int | None = None) -> np.ndarray:
        """Bucket grouping by how fast each exhausts its own pool, in log epochs per unit weight.

        Topic is the right pooling for parameters that say how much a family helps an eval, but the wrong
        one for parameters that describe geometry: on the 39-bucket panel a single topic spans 4.80 to
        1723.89 epochs per unit weight, a 359-fold range, and no single boundary scale can serve all of
        it. Strata are cut on the observed distribution, so this needs no knowledge of bucket contents --
        only the token counts the exposure columns are already built from.
        """
        count = self.n_exposure_strata() if n_strata is None else n_strata
        log_rate = np.log10(np.maximum(self.total_epochs_per_unit, 1e-12))
        if count <= 1:
            return np.zeros(len(log_rate), dtype=int)
        edges = np.quantile(log_rate, np.linspace(0.0, 1.0, count + 1)[1:-1])
        return np.searchsorted(edges, log_rate)

    def boundary_risk(self) -> np.ndarray:
        """Total epochs a bucket receives at unit weight: how fast it exhausts its own pool.

        A bucket with a large value crosses one epoch at a tiny mixture weight, so the fitted optimum can
        buy its exposure almost entirely in the decay phase and drive its early share to zero. Those are
        the buckets the early-boundary kernel exists to protect, and it is knowable before any fit.

        An earlier version returned ``epochs_early / epochs_late``, which is USELESS as a discriminator:
        that ratio is ``phase_0_fraction / phase_1_fraction`` and is therefore identical for every bucket
        by construction — 3.93 on every WSD80 bucket and 4.00 on all thirty-nine 300M buckets. Pool size
        relative to the run is what actually varies, from 4.8 to 1723.9 epochs per unit weight on 300M.
        """
        return self.total_epochs_per_unit


@dataclass(frozen=True)
class Shape:
    """Nonlinear parameters. Per-family where a topic can plausibly differ, shared otherwise."""

    near_horizon: float
    damage_horizon: float
    offset: float
    damage_exponent: float
    readout_exponent: tuple[float, ...]  # one per FAMILY: how much a topic helps this eval
    boundary_scale: tuple[float, ...]  # one per EXPOSURE STRATUM: geometry, not taste


def family_sums(values: np.ndarray, family_index: np.ndarray) -> np.ndarray:
    """Mean over the buckets of each family, the project's standard pooling."""
    return np.column_stack([values[:, family_index == f].mean(axis=1) for f in np.unique(family_index)])


def design(panel: Panel, shape: Shape) -> tuple[np.ndarray, np.ndarray]:
    """Free-sign columns and non-negative columns.

    Free: the intercept, and one late-share term per family. The late-share signs are free because a
    topic being heavy late can help or hurt depending on the eval, and nothing here knows which.
    """
    exponent = np.asarray(shape.readout_exponent, dtype=float)[panel.family_index]
    # Geometry parameters follow pool size, not topic: a topic can span a 359-fold range of epochs per
    # unit weight, and no single boundary scale serves all of it.
    scale = np.asarray(shape.boundary_scale, dtype=float)[panel.exposure_stratum()]

    near = (panel.exposure(shape.near_horizon) + shape.offset) ** -exponent
    late = (panel.exposure(1.0) + shape.offset) ** -exponent
    boundary = np.exp(-panel.early_epochs() / scale)
    # Saturating damage, BOUNDED in [0, 1). The unbounded `excess ** tau` reached 1.2e24 on this panel
    # (tau up to 10, excess up to 255 epochs) and made the model violently extrapolative: `fit_head`
    # normalises columns by their TRAINING norm, so a test row with slightly larger excess is amplified
    # by the tau-th power. Measured consequence of the old form -- random parameters drawn from this
    # model's own bounds predicted up to 23x worse than the mean, and one ordinary 300M component fit
    # returned an RMSE of 562 BPB. Below the knee this is still a power law, so panels whose excess never
    # approaches DAMAGE_KNEE (WSD80 tops out at 25.46) are essentially unaffected.
    excess = np.maximum(panel.exposure(shape.damage_horizon) - 1.0, 0.0) / DAMAGE_KNEE
    powered = excess**shape.damage_exponent
    damage = powered / (1.0 + powered)

    free = np.column_stack([np.ones(len(panel.weights)), family_sums(panel.weights[:, 1, :], panel.family_index)])
    constrained = np.column_stack(
        [
            family_sums(near, panel.family_index),
            family_sums(late, panel.family_index),
            family_sums(boundary, panel.family_index),
            family_sums(damage, panel.family_index),
            near,  # per-bucket departures on the main benefit block, shrunk in the solve
        ]
    )
    return free, constrained


def pooled_width(panel: Panel) -> int:
    """Columns carrying the pooled signal; everything after them is a shrunk departure."""
    return 4 * panel.n_families


def column_space(free: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    """Orthonormal basis for the columns free ACTUALLY span, dropping null directions.

    ``free`` is rank deficient by construction: family shares sum to one, so the family-size-weighted
    sum of the late-share columns reproduces the intercept exactly. On WSD80 that makes a 346x3 matrix
    of rank 2 with a zero singular value.

    A plain ``np.linalg.qr`` returns one Q column per INPUT column, and for a rank-deficient input the
    surplus column is an arbitrary direction lying entirely outside the column space -- measured at
    distance 1.0 from it. Projecting with that basis removes real signal from the response and from every
    constrained column, and which signal it removes depends on the order the families happen to be
    numbered in. Swapping the two WSD80 family labels, with parameters swapped to match, moved primary
    predictions by RMS 0.07054 BPB against gates of about 0.008.

    Truncating on the singular values keeps the projector equal to the one onto span(free), which is what
    partialling out the free block is supposed to mean, and makes the fit invariant to relabelling.
    """
    u, singular, _ = np.linalg.svd(free, full_matrices=False)
    return u[:, singular > tolerance * max(singular[0], 1e-300)]


def fit_head(
    free: np.ndarray, constrained: np.ndarray, response: np.ndarray, ridge: float, pooled: int
) -> tuple[np.ndarray, np.ndarray]:
    """Free columns unconstrained, the rest non-negative, departures shrunk hard and levels barely.

    Columns are normalised before the sign-constrained solve and unscaled after. That does not change the
    model, since non-negativity survives positive rescaling, but the damage column spans many orders of
    magnitude with a fitted exponent and an unscaled solve misses its iteration cap on some folds.
    """
    basis = column_space(free)
    columns = constrained - basis @ (basis.T @ constrained)
    target = response - basis @ (basis.T @ response)
    scale = np.maximum(np.linalg.norm(columns, axis=0), 1e-300)
    scaled = columns / scale
    if ridge > 0:
        strength = np.sqrt(ridge) * np.concatenate([np.full(pooled, 1e-3), np.ones(scaled.shape[1] - pooled)])
        scaled = np.vstack([scaled, np.diag(strength)])
        target = np.concatenate([target, np.zeros(scaled.shape[1])])
    amplitudes, _ = nnls(scaled, target, maxiter=20000)
    amplitudes = amplitudes / scale
    return np.linalg.lstsq(free, response - constrained @ amplitudes, rcond=None)[0], amplitudes


def predictions_escape_range(predictions: np.ndarray, observed: np.ndarray, slack: float = 3.0) -> bool:
    """Do held-out predictions leave the range the training targets actually occupy?

    Selection guard for extrapolation. The design contains columns that are large where the panel is
    sparse -- the readout is capped only by ``offset ** -gamma`` on buckets with exactly zero weight --
    so a parameter vector can fit the training rows well and still send one held-out row far outside any
    plausible value. Measured on the 300M arc_challenge component: 516 of 520 rows had a median absolute
    error of 0.027 while a single row predicted 37.2 against an observed 1.16, which alone moved RMSE
    from about 0.04 to 1.58.

    This rejects such a vector during selection rather than clipping its output afterwards, so the fitted
    model is never one that produces impossible predictions. The window is the observed spread widened by
    ``slack``, which is deliberately loose: the point is to exclude the absurd, not to shrink toward the
    mean.
    """
    low, high = float(np.min(observed)), float(np.max(observed))
    margin = slack * max(high - low, 1e-12)
    return bool(np.min(predictions) < low - margin or np.max(predictions) > high + margin)


def unpack(vector: np.ndarray, n_families: int, n_strata: int = 3) -> tuple[Shape, float]:
    """Selection vector -> shape and ridge. Offsets and scales are searched in log space."""
    near, damage_horizon, log_offset, tau, log_ridge = vector[:5]
    exponents = tuple(vector[5 : 5 + n_families])
    scales = tuple(10.0 ** vector[5 + n_families : 5 + n_families + n_strata])
    return (
        Shape(
            near_horizon=float(near),
            damage_horizon=float(damage_horizon),
            offset=10.0 ** float(log_offset),
            damage_exponent=float(tau),
            readout_exponent=exponents,
            boundary_scale=scales,
        ),
        10.0 ** float(log_ridge),
    )


def bounds(n_families: int, n_strata: int = 3) -> tuple[tuple[float, float], ...]:
    return (
        (0.0, 1.0),  # near horizon
        (0.0, 1.0),  # damage horizon
        # Log offset. The floor is 1e-2 EPOCHS, not 1e-5, and the difference is a robustness fix rather
        # than a taste choice. The readout (E + offset)^-gamma is regularised only by the offset on
        # buckets with EXACTLY zero weight -- 41 such entries on the 300M panel -- so a 1e-5 floor lets a
        # single design entry reach offset^-gamma = 1e10. Random parameters drawn from the old box gave
        # out-of-fold RMSE up to 15894 BPB against a 0.0178 intercept; at this floor the worst is 0.0434.
        # It costs nothing: nested 300M Uncheatable RMSE is 0.005665 at this floor against 0.005671 at
        # 1e-5. WSD80 is structurally immune either way, its smallest real exposure being 0.1 epochs.
        (-2.0, -0.3),  # log offset
        (0.2, 10.0),  # damage exponent
        (-6.0, 1.0),  # log ridge
        # Readout exponent per family. The upper bound is 12, not 2, and the difference is measured
        # rather than chosen: at 2 this parameter sits EXACTLY on its bound in 7 of 11 WSD80 fits, and a
        # parameter railed against its limit in most fits is a specification defect. Widening it lifts the
        # WSD80 acceptance score from 40/44 to 43/44, recovers Regret@1 to 11/11, and unpins every seed.
        # It is safe elsewhere: on the 39-bucket panel the exponent pins in 1 fold of 9 and widening
        # changes nested Uncheatable RMSE only from 0.005909 to 0.005919. The asymmetry is structural --
        # WSD80 has one bucket per family, so the per-bucket departure block duplicates the family mean
        # and this exponent is the only flexibility that design has.
        *(((0.005, 12.0),) * n_families),
        *(((-1.0, 3.5),) * n_strata),  # log boundary scale per exposure stratum
    )
