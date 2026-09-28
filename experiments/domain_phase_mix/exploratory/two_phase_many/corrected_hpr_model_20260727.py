# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HPR with the structural defects an audit found, each fixed behind its own switch.

Six defects were confirmed by arithmetic on the built design rather than by fitting, so each has a
determinate fix. They are separable switches so the benchmark can attribute any change in fit to a
specific correction instead of to the bundle:

``identifiable_hierarchy``
    The pooled family base column is the exact sum of its members' excess columns -- measured
    relative gap 0.0, rank 49 of 52 with deficiency exactly the family count. Ridge still returns a
    unique fit, so this is not a prediction bug, but the split between a family coefficient and a
    bucket deviation is chosen by the penalty rather than by the data, which is what the Observatory
    displays as two separate mechanisms. The fix drops the base column and puts one total coefficient
    on each bucket, moving the pooling into a non-diagonal penalty that shrinks within-family spread
    separately from the family mean.

    This is a *different* prior, not a reparameterization, and the earlier claim that it was
    equivalent was wrong. Minimizing the original penalty over the family coefficient gives
    ``min_{0 <= b <= min_i theta_i} [b^2 + r sum_i (theta_i - b)^2]``, whose constraint binds when a
    family's smallest total coefficient falls below the unconstrained optimum; the replacement uses
    ``mean_f^2 + r sum_i (theta_i - mean_f)^2``, which has no such kink. Both are sensible
    partial-pooling priors and they are empirically indistinguishable in fit here, but the honest
    description is a changed prior that buys full rank, not an algebraic identity.

``transition``
    Retained exposure is ``exp(-lambda(1 - w1)) w0 c0 + eta w1 c1``, which for a *tied* policy still
    depends on both transition parameters and on where the phase boundary was drawn: holding the
    policy and the total token budget fixed and moving the boundary from 0.5 to 0.9 swings predicted
    exposure by 144 percent. A single-phase policy cannot have a boundary-dependent prediction. The
    fix writes exposure as aggregate token exposure times a tilt factor that is exactly one when the
    phases are tied, so one-phase is the algebraic restriction of the two-phase model and the
    transition parameters act only on genuine phase asymmetry, scaled by late token mass so they
    carry a schedule rather than absorbing one. Because the legacy law is also nonlinear in the
    mixture weight itself, a curved variant is provided that keeps that nonlinearity on a
    boundary-free quantity, separating the cost of the invariance from the cost of the form.

``bounded_returns``
    The exponent grid reaches 1.2, which is increasing returns to repetition. Worse, the replay
    penalty is a squared softplus of ``log1p(exposure) - tau`` and with the selected tau it is flat
    until exposure 169 while the panel's median exposure is 2.0. Over the whole realized range the
    benefit therefore never meets a countervailing harm, which is what lets optima run to corners.
    The fix caps the exponent below one and switches bucket replay to the quadratic excess form, so
    harm eventually outgrows benefit inside the range the panel actually samples.

``deduplicated_ledgers``
    For a singleton family, family overexposure and average member replay are literally the same
    column: 39 identical pairs appear under a one-bucket-per-family partition. The three-family dolma
    cells have none, so this matters for StarCoder and for singleton-ish partitions, not for the main
    fit panel. The fix emits one column.

``normalized_family_ledger``
    One threshold ``tau`` gates both family-total exposure and individual bucket exposure, but family
    totals scale with family size and bucket exposures do not, so the same tau means different things
    in the two ledgers. The fix normalizes the family ledger by its proportional reference before
    thresholding, making tau dimensionless in both.

``bounded_link``
    Not from the audit. Borrowed from the compact retained-state model, which is the only model in the
    Observatory whose *relative* rank improves from in-sample to held out on every metric, and which
    leads lower-tail optimism by a wide margin. Its design block is unremarkable; the difference is
    that it fits ``log(target - floor)`` and predicts ``floor + exp(eta)``, with the floor fixed at
    0.95 of the smallest observed target rather than cross-validated. An additive head can predict
    below any entropy floor, which is the mechanism behind out-of-support optimism -- an optimizer
    walks toward a region the model calls arbitrarily good and the panel never contradicted it. Under
    the bounded link that region is unreachable rather than merely penalized, which is a stronger
    answer to the audit's returns-to-repetition objection than capping the exponent.

``smooth_phase_cost``
    The phase term is a nonnegative multiple of total variation. It can only ever penalize asymmetry,
    so it cannot express which data belongs early or late, and total variation has a cusp at the tied
    policy, charging a first-order price for an infinitesimal phase tilt. A local dynamics model gives
    a quadratic leading order. The fix uses a squared tilt for the cost and adds signed family
    *contrasts* so the model can say a tilt is *good*, with genuinely unconstrained coefficients under
    mixed box bounds. Two earlier versions of this term were rank-deficient and both are gone: an
    opposite-sign column pair, and one column per family. The latter is collinear because the family
    tilts sum to zero on the simplex, so only ``F - 1`` contrasts against a reference family are
    identifiable and that is what is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear, nnls

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_hierarchical_coverage_grp_20260715 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_production_grp_quality_variants as family_grp,
)

EPSILON = 1e-12
# Ceiling on the benefit exponent. Strictly below one keeps returns to repetition diminishing.
MAX_EXPONENT = 0.95
# Fraction of the smallest observed target used as the bounded link's floor, matching the compact
# retained-state model this correction is borrowed from. Held fixed rather than cross-validated,
# because cross-validation would choose it against in-support fit and the bound's whole purpose is
# to constrain behaviour out of support.
DEFICIT_FLOOR_FRACTION = 0.95
# Numerical guard so a runaway linear predictor cannot overflow the exponential.
LINK_CLIP = 30.0


class TransitionForm(StrEnum):
    """How retained exposure depends on the two phase mixtures.

    ``LEGACY`` is the current law. ``TIED_INVARIANT`` makes a tied policy's exposure exactly its
    aggregate token exposure, which removes the boundary dependence but also removes the curvature in
    weight that the legacy law happens to supply. ``TIED_INVARIANT_CURVED`` keeps that curvature while
    staying boundary-invariant, so the two together separate "the invariance costs fit" from "my
    replacement functional form is worse" -- claims that the single fix cannot distinguish.
    """

    LEGACY = "legacy"
    TIED_INVARIANT = "tied_invariant"
    TIED_INVARIANT_CURVED = "tied_invariant_curved"
    RECENCY_KERNEL = "recency_kernel"


@dataclass(frozen=True)
class Corrections:
    """Which audit fixes are active. Defaults reproduce the current model's structure."""

    transition: TransitionForm = TransitionForm.LEGACY
    identifiable_hierarchy: bool = False
    bounded_returns: bool = False
    deduplicated_ledgers: bool = False
    normalized_family_ledger: bool = False
    smooth_phase_cost: bool = False
    bounded_link: bool = False

    def label(self) -> str:
        active = [field for field in self.__dataclass_fields__ if field != "transition" and getattr(self, field)]
        if self.transition is not TransitionForm.LEGACY:
            active.insert(0, self.transition.value)
        if not active:
            return "baseline"
        return "+".join(active)


@dataclass(frozen=True)
class CorrectedDesign:
    """Design plus an arbitrary penalty matrix, since hierarchical pooling is not diagonal."""

    values: np.ndarray
    names: tuple[str, ...]
    penalty: np.ndarray
    # Columns whose coefficient may take either sign. Everything else is constrained nonnegative,
    # which is what makes the benefit and harm channels interpretable. A directional phase term has no
    # such prior -- the model should be able to say a tilt helps -- so it gets a genuinely free
    # coefficient rather than an opposite-sign column pair, which would be exactly collinear.
    free_sign: np.ndarray


@dataclass(frozen=True)
class CorrectedModel:
    dataset: family_grp.Dataset
    config: bench.Config
    corrections: Corrections
    intercept: float
    coefficients: np.ndarray
    # Log-deficit floor when the bounded link is active, otherwise NaN. Estimated from the training
    # rows alone, so a held-out row never contributes to the bound that predicts it.
    floor: float = float("nan")

    def predict(self, weights: np.ndarray) -> np.ndarray:
        candidate = replace(
            self.dataset,
            weights=np.asarray(weights, dtype=float),
            target=np.zeros(len(weights), dtype=float),
        )
        design = build_corrected_design(candidate, self.config, self.corrections)
        linear = self.intercept + design.values @ self.coefficients
        if not self.corrections.bounded_link:
            return np.asarray(linear, dtype=float)
        return np.asarray(self.floor + np.exp(np.clip(linear, -LINK_CLIP, LINK_CLIP)), dtype=float)


def clamp_shape(shape: family_grp.Shape, corrections: Corrections) -> family_grp.Shape:
    """Hold the benefit exponent strictly below one when bounded returns are requested."""
    if not corrections.bounded_returns or shape.exponent <= MAX_EXPONENT:
        return shape
    return replace(shape, exponent=MAX_EXPONENT)


def late_token_fraction(dataset: family_grp.Dataset) -> np.ndarray:
    """Per-bucket share of its token budget spent in the late phase."""
    total = np.maximum(dataset.c0 + dataset.c1, EPSILON)
    return dataset.c1 / total


def tied_invariant_exposure(
    dataset: family_grp.Dataset,
    shape: family_grp.Shape,
    curved: bool,
) -> np.ndarray:
    """Aggregate token exposure, tilted by phase asymmetry and nothing else.

    ``aggregate`` is the bucket's total exposure across both phases, which for a tied policy depends
    only on the total token budget and not on the boundary. ``tilt`` is the late-minus-early weight
    difference scaled by late token mass, so it vanishes for tied policies and grows with how much
    late training there is to be recent about; the late multiplier acts on it symmetrically.

    ``curved`` adds a saturating factor in the *aggregate* weight, which reproduces the shape of the
    legacy law's ``exp(-lambda (1 - w1))`` term while depending only on boundary-free quantities. The
    legacy term is nonlinear in the mixture weight itself, so for a tied policy it supplies curvature
    that has nothing to do with any transition. Carrying that curvature separately is what lets the
    benchmark tell the two roles apart.
    """
    phase0_weight = dataset.weights[:, 0, :]
    phase1_weight = dataset.weights[:, 1, :]
    total = np.maximum(dataset.c0 + dataset.c1, EPSILON)
    aggregate = phase0_weight * dataset.c0[None, :] + phase1_weight * dataset.c1[None, :]
    tilt = late_token_fraction(dataset)[None, :] * (phase1_weight - phase0_weight)
    # Linear in the tilt, matching the legacy law's linearity in the phase weights. An exponential
    # here responds far more sharply to phase asymmetry than the legacy law does and costs several
    # run sigma of fit for that reason alone, which would confound the invariance with the form.
    exposure = aggregate * np.maximum(1.0 + (shape.late_multiplier - 1.0) * tilt, 0.0)
    if curved:
        aggregate_weight = aggregate / total[None, :]
        exposure = exposure * np.exp(-shape.forgetting_rate * (1.0 - aggregate_weight))
    return np.maximum(exposure, 0.0)


def kernel_weight(start: float, end: float, decay: float) -> float:
    """Average of the recency kernel ``exp(decay (t - 1))`` over normalized training time.

    ``t`` runs from 0 at the start of training to 1 at the end, so the kernel says a token's
    contribution depends on how long ago it was seen relative to the whole run.
    """
    if abs(decay) < 1e-9:
        return 1.0
    return float((np.exp(decay * (end - 1.0)) - np.exp(decay * (start - 1.0))) / (decay * (end - start)))


def recency_kernel_exposure(dataset: family_grp.Dataset, shape: family_grp.Shape) -> np.ndarray:
    """Token exposure weighted by when in training each token was seen.

    Each phase occupies an interval of normalized training time, and the kernel is averaged over that
    interval, so late tokens can count more per token than early ones -- the freedom the legacy law
    actually uses -- while the *total* weight over a tied policy is an integral over the whole run and
    therefore does not depend on where the phases were cut. That is the property the legacy law
    lacks: it reweights by phase index instead of by elapsed time, so it cannot be carried to a run
    with a different boundary or schedule.
    """
    phase0_weight = dataset.weights[:, 0, :]
    phase1_weight = dataset.weights[:, 1, :]
    decay = shape.late_multiplier - 1.0
    late_fraction = late_token_fraction(dataset)
    early = np.asarray([kernel_weight(0.0, 1.0 - f, decay) for f in late_fraction], dtype=float)
    late = np.asarray([kernel_weight(1.0 - f, 1.0, decay) for f in late_fraction], dtype=float)
    exposure = early[None, :] * dataset.c0[None, :] * phase0_weight + late[None, :] * dataset.c1[None, :] * phase1_weight
    if shape.forgetting_rate > 0.0:
        total = np.maximum(dataset.c0 + dataset.c1, EPSILON)
        aggregate_weight = (phase0_weight * dataset.c0[None, :] + phase1_weight * dataset.c1[None, :]) / total[None, :]
        exposure = exposure * np.exp(-shape.forgetting_rate * (1.0 - aggregate_weight))
    return np.maximum(exposure, 0.0)


def corrected_exposure(
    dataset: family_grp.Dataset,
    shape: family_grp.Shape,
    corrections: Corrections,
) -> np.ndarray:
    if corrections.transition is TransitionForm.LEGACY:
        return bench.retained_exposure(dataset, shape)
    if corrections.transition is TransitionForm.RECENCY_KERNEL:
        return recency_kernel_exposure(dataset, shape)
    return tied_invariant_exposure(
        dataset,
        shape,
        curved=corrections.transition is TransitionForm.TIED_INVARIANT_CURVED,
    )


def hierarchical_penalty(dataset: family_grp.Dataset, residual_shrink: float) -> np.ndarray:
    """Penalty rows shrinking each family's mean coefficient and its members' spread separately.

    Replaces the collinear ``pooled base + bucket excess`` pair. A family's mean is penalized once at
    unit strength and each member's deviation from that mean at ``residual_shrink``, which is the
    same partial-pooling prior the original parameterization expressed, but written on coefficients
    the data can identify.
    """
    rows = []
    for members in dataset.family_members:
        indicator = np.zeros(dataset.m, dtype=float)
        indicator[members] = 1.0 / len(members)
        rows.append(indicator)
        if len(members) == 1:
            continue
        for member in members:
            deviation = -indicator.copy()
            deviation[member] += 1.0
            rows.append(np.sqrt(residual_shrink) * deviation)
    return np.asarray(rows, dtype=float)


def build_corrected_design(
    dataset: family_grp.Dataset,
    config: bench.Config,
    corrections: Corrections,
) -> CorrectedDesign:
    """Mirror of the benchmark's design with each confirmed defect optionally repaired."""
    shape = clamp_shape(config.shape, corrections)
    exposure = corrected_exposure(dataset, shape, corrections)
    bucket_signal = bench.power_response(exposure, shape.exponent)
    family_total = np.column_stack([exposure[:, members].sum(axis=1) for members in dataset.family_members])
    nonsingleton = tuple(index for index, members in enumerate(dataset.family_members) if len(members) > 1)

    pieces: list[np.ndarray] = []
    names: list[str] = []
    diagonal: list[float] = []
    extra_penalty: np.ndarray | None = None

    if corrections.identifiable_hierarchy:
        pieces.append(-bucket_signal)
        names.extend(f"bucket_total_signal:{domain}" for domain in dataset.domains)
        diagonal.extend([np.nan] * dataset.m)
        extra_penalty = hierarchical_penalty(dataset, config.residual_shrink)
    else:
        groups = bench.pooling_groups(dataset, config.variant)
        singleton = [members[0] for _name, members in groups if len(members) == 1]
        if singleton:
            pieces.append(-bucket_signal[:, singleton])
            names.extend(f"singleton_signal:{dataset.domains[index]}" for index in singleton)
            diagonal.extend([1.0] * len(singleton))
        nonsingleton_groups = [(name, members) for name, members in groups if len(members) > 1]
        for group_name, members in nonsingleton_groups:
            pieces.append(-bucket_signal[:, members].sum(axis=1, keepdims=True))
            names.append(f"pooled_base_signal:{group_name}")
            diagonal.append(1.0)
        if nonsingleton_groups:
            residual_members = np.concatenate([members for _name, members in nonsingleton_groups])
            pieces.append(-bucket_signal[:, residual_members])
            names.extend(f"bucket_excess_signal:{dataset.domains[index]}" for index in residual_members)
            diagonal.extend([config.residual_shrink] * len(residual_members))

    if nonsingleton:
        pieces.append(-bench.power_response(family_total[:, nonsingleton], shape.exponent))
        names.extend(f"family_coverage_signal:{dataset.family_names[index]}" for index in nonsingleton)
        diagonal.extend([1.0] * len(nonsingleton))

    if corrections.normalized_family_ledger:
        reference = bench.proportional_family_exposure(dataset, shape)
        family_ledger = family_total / np.maximum(reference[None, :], EPSILON)
    else:
        family_ledger = family_total
    pieces.append(bench.overexposure_harm(family_ledger, shape.penalty_threshold))
    names.extend(f"family_overexposure:{name}" for name in dataset.family_names)
    diagonal.extend([1.0] * len(dataset.family_names))

    replay = bench.excess_replay_harm if corrections.bounded_returns else bench.overexposure_harm
    bucket_harm = replay(exposure, shape.penalty_threshold)
    replay_families = [
        index
        for index, members in enumerate(dataset.family_members)
        if not (corrections.deduplicated_ledgers and len(members) == 1)
    ]
    if replay_families:
        pieces.append(
            np.column_stack([bucket_harm[:, dataset.family_members[index]].mean(axis=1) for index in replay_families])
        )
        names.extend(f"family_member_replay:{dataset.family_names[index]}" for index in replay_families)
        diagonal.extend([1.0] * len(replay_families))

    free: list[bool] = [False] * len(names)
    phase0_weight = dataset.weights[:, 0, :]
    phase1_weight = dataset.weights[:, 1, :]
    tilt = phase1_weight - phase0_weight
    if corrections.smooth_phase_cost:
        pieces.append(np.sum(tilt**2, axis=1, keepdims=True))
        names.append("phase_tilt_squared")
        diagonal.append(1.0)
        free.append(False)
        # One signed column per family sums to exactly zero across families, because both phase
        # mixtures lie on the simplex and so the family tilts sum to 1 - 1 = 0. Emitting all of them
        # is rank-deficient by one and leaves each coefficient gauge-dependent, set by the ridge
        # rather than by the data. Only F - 1 contrasts are identifiable, so the last family is held
        # as the reference and every coefficient is read as a difference against it.
        directional = np.column_stack([tilt[:, members].sum(axis=1) for members in dataset.family_members])
        contrasts = directional[:, :-1] - directional[:, [-1]]
        reference = dataset.family_names[-1]
        pieces.append(contrasts)
        names.extend(f"phase_direction:{name}_vs_{reference}" for name in dataset.family_names[:-1])
        diagonal.extend([1.0] * contrasts.shape[1])
        free.extend([True] * contrasts.shape[1])
    else:
        pieces.append(0.5 * np.abs(tilt).sum(axis=1, keepdims=True))
        names.append("phase_shift_tv")
        diagonal.append(1.0)
        free.append(False)

    values = np.hstack(pieces)
    penalty = np.zeros((0, values.shape[1]), dtype=float)
    diagonal_array = np.asarray(diagonal, dtype=float)
    plain = np.flatnonzero(np.isfinite(diagonal_array))
    if plain.size:
        block = np.zeros((plain.size, values.shape[1]), dtype=float)
        block[np.arange(plain.size), plain] = np.sqrt(diagonal_array[plain])
        penalty = np.vstack([penalty, block])
    if extra_penalty is not None:
        padded = np.zeros((extra_penalty.shape[0], values.shape[1]), dtype=float)
        padded[:, : extra_penalty.shape[1]] = extra_penalty
        penalty = np.vstack([penalty, padded])
    return CorrectedDesign(values, tuple(names), penalty, np.asarray(free, dtype=bool))


def fit_corrected(
    dataset: family_grp.Dataset,
    config: bench.Config,
    corrections: Corrections,
    indices: np.ndarray,
) -> CorrectedModel:
    design = build_corrected_design(dataset, config, corrections)
    train_design = design.values[indices]
    train_target = dataset.target[indices]
    floor = float("nan")
    if corrections.bounded_link:
        # Fit the log deficit above a floor rather than the target itself, so no linear predictor can
        # push the prediction below the floor however far the design is extrapolated.
        floor = DEFICIT_FLOOR_FRACTION * float(np.min(train_target))
        train_target = np.log(np.maximum(train_target - floor, 1e-9))
    design_mean = train_design.mean(axis=0, keepdims=True)
    target_mean = float(train_target.mean())
    centered_design = train_design - design_mean
    centered_target = train_target - target_mean
    if config.l2 > 0.0 and design.penalty.shape[0]:
        centered_design = np.vstack([centered_design, np.sqrt(config.l2) * design.penalty])
        centered_target = np.concatenate([centered_target, np.zeros(design.penalty.shape[0], dtype=float)])
    if design.free_sign.any():
        # Mixed bounds: nonnegative on the interpretable benefit and harm channels, unconstrained on
        # the directional phase terms. Nonnegative least squares cannot express that, so this drops to
        # a box-constrained solve, which reduces to the same problem when no column is free.
        lower = np.where(design.free_sign, -np.inf, 0.0)
        upper = np.full(design.values.shape[1], np.inf)
        coefficients = lsq_linear(centered_design, centered_target, bounds=(lower, upper)).x
    else:
        coefficients, _residual = nnls(centered_design, centered_target, maxiter=40 * centered_design.shape[1])
    intercept = target_mean - float((design_mean @ coefficients).item())
    return CorrectedModel(dataset, config, corrections, intercept, coefficients, floor)


def corrected_oof_prediction(
    dataset: family_grp.Dataset,
    config: bench.Config,
    corrections: Corrections,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for train, test in splits:
        prediction[test] = fit_corrected(dataset, config, corrections, train).predict(dataset.weights[test])
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete OOF prediction for {corrections.label()}")
    return prediction


def design_diagnostics(
    dataset: family_grp.Dataset,
    config: bench.Config,
    corrections: Corrections,
) -> dict[str, Any]:
    """Rank, conditioning and effective degrees of freedom for one built design.

    Effective degrees of freedom is the trace of the ridge hat matrix restricted to the columns the
    nonnegative solver leaves active, which is the honest count for a penalized active-set fit and is
    what the audit asked for in place of a raw parameter count.
    """
    design = build_corrected_design(dataset, config, corrections)
    values = design.values
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    positive = singular[singular > singular.max() * 1e-14]
    model = fit_corrected(dataset, config, corrections, np.arange(dataset.n))
    # A free-sign coefficient is active when nonzero in either direction, not merely when positive.
    active = np.flatnonzero(np.abs(model.coefficients) > 1e-10)
    effective = float("nan")
    if active.size:
        sub = centered[:, active]
        penalty = design.penalty[:, active] if design.penalty.shape[0] else np.zeros((0, active.size))
        gram = sub.T @ sub + config.l2 * (penalty.T @ penalty)
        effective = float(np.trace(sub @ np.linalg.solve(gram, sub.T)))
    return {
        "columns": int(values.shape[1]),
        "rank": int(np.linalg.matrix_rank(values)),
        "deficiency": int(values.shape[1] - np.linalg.matrix_rank(values)),
        "condition_number": float(positive.max() / positive.min()),
        "active_columns": int(active.size),
        "effective_dof": effective,
    }
