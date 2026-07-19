# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mechanistic feature models for mixture-policy surrogate discovery."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import nnls
from scipy.special import logsumexp


@dataclass(frozen=True)
class Panel:
    """A fixed-budget mixture-policy panel."""

    name: str
    target: str
    weights: np.ndarray
    observed: np.ndarray
    phase_epoch_factors: np.ndarray
    phase_fractions: np.ndarray
    domains: tuple[str, ...]
    proportional: np.ndarray
    family_names: tuple[str, ...]
    family_members: tuple[np.ndarray, ...]
    group_names: tuple[str, ...]
    group_members: tuple[np.ndarray, ...]
    group_family_indices: np.ndarray

    @property
    def n(self) -> int:
        return len(self.observed)

    @property
    def m(self) -> int:
        return len(self.domains)


@dataclass(frozen=True)
class ModelConfig:
    """One prespecified nonlinear mechanism and shape."""

    family: str
    parameters: tuple[tuple[str, float], ...] = ()

    @property
    def key(self) -> str:
        if not self.parameters:
            return self.family
        suffix = "__".join(f"{name}-{value:g}" for name, value in self.parameters)
        return f"{self.family}__{suffix}"

    def parameter(self, name: str) -> float:
        return dict(self.parameters)[name]


@dataclass(frozen=True)
class Design:
    """A response design whose non-intercept amplitudes are nonnegative."""

    values: np.ndarray
    names: tuple[str, ...]


@dataclass(frozen=True)
class FittedModel:
    """A constrained ridge fit for one mechanistic design."""

    config: ModelConfig
    l2: float
    intercept: float
    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    effective_degrees_of_freedom: float

    def predict_design(self, design: Design) -> np.ndarray:
        if design.names != self.feature_names:
            raise ValueError("Prediction design does not match fitted feature order")
        return self.intercept + design.values @ self.coefficients


def simulated_epochs(panel: Panel, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1:] != (2, panel.m):
        raise ValueError(f"Expected weights (n, 2, {panel.m}), got {weights.shape}")
    phase0 = weights[:, 0] * panel.phase_epoch_factors[0]
    phase1 = weights[:, 1] * panel.phase_epoch_factors[1]
    return phase0, phase1, phase0 + phase1


def unique_coverage(exposure: np.ndarray) -> np.ndarray:
    return -np.expm1(-np.maximum(exposure, 0.0))


def literal_replay(exposure: np.ndarray) -> np.ndarray:
    return np.maximum(exposure, 0.0) - unique_coverage(exposure)


def finite_subset_coverage(exposure: np.ndarray) -> np.ndarray:
    """Exact unique fraction for a shuffled finite subset recycled by epoch."""

    return np.clip(exposure, 0.0, 1.0)


def finite_subset_replay(exposure: np.ndarray) -> np.ndarray:
    """Exact repeated passes after exhausting a shuffled finite subset."""

    return np.maximum(exposure - 1.0, 0.0)


def foundation_family_index(panel: Panel) -> int:
    """Return the largest predeclared family, interpreted as broad foundation data."""

    family_mass = np.asarray([panel.proportional[members].sum() for members in panel.family_members])
    return int(np.argmax(family_mass))


def foundation_gated_exposure(
    panel: Panel,
    weights: np.ndarray,
    acquisition: float,
    boost: float,
) -> np.ndarray:
    """Integrate specialist exposure under a broad-foundation learning state.

    Normalized training time runs from zero to one. The foundation state is
    ``1 - exp(-acquisition * cumulative_foundation_mass)``. Specialist
    examples retain unit baseline efficiency and gain ``boost`` times this
    state. The integral is exact for each piecewise-constant phase.
    """

    weights = np.asarray(weights, dtype=float)
    phase0, phase1, _total = simulated_epochs(panel, weights)
    foundation_index = foundation_family_index(panel)
    foundation_members = panel.family_members[foundation_index]
    specialist = np.ones(panel.m, dtype=bool)
    specialist[foundation_members] = False
    cumulative_foundation = np.zeros(len(weights), dtype=float)
    effective = np.zeros_like(phase0)
    for phase_exposure, phase_weight, duration in (
        (phase0, weights[:, 0], panel.phase_fractions[0]),
        (phase1, weights[:, 1], panel.phase_fractions[1]),
    ):
        foundation_mass = phase_weight[:, foundation_members].sum(axis=1)
        phase_dose = foundation_mass * duration
        hazard = acquisition * phase_dose
        mean_survival = np.divide(
            -np.expm1(-hazard),
            hazard,
            out=np.ones_like(hazard),
            where=hazard > 1e-12,
        )
        mean_survival *= np.exp(-acquisition * cumulative_foundation)
        mean_state = 1.0 - mean_survival
        phase_effective = phase_exposure.copy()
        phase_effective[:, specialist] *= 1.0 + boost * mean_state[:, None]
        effective += phase_effective
        cumulative_foundation += phase_dose
    return effective


def normalized_group_exposure(exposure: np.ndarray, reference: np.ndarray, panel: Panel) -> np.ndarray:
    grouped = group_sum(exposure, panel)
    grouped_reference = group_sum(reference, panel)[0]
    return grouped / np.maximum(grouped_reference[None, :], 1e-12)


def two_level_prior_floor(
    panel: Panel,
    foundation_floor: float,
    specialist_floor: float,
) -> np.ndarray:
    """Assign equivalent prior exposure to foundation and specialist groups."""

    foundation_index = foundation_family_index(panel)
    return np.where(
        panel.group_family_indices == foundation_index,
        foundation_floor,
        specialist_floor,
    )


def family_sum(values: np.ndarray, panel: Panel, *, weighted: bool = False) -> np.ndarray:
    output = np.empty((len(values), len(panel.family_names)), dtype=float)
    for family_index, members in enumerate(panel.family_members):
        if weighted:
            weights = panel.proportional[members]
            weights = weights / weights.sum()
            output[:, family_index] = values[:, members] @ weights
        else:
            output[:, family_index] = values[:, members].sum(axis=1)
    return output


def family_weight(weights: np.ndarray, panel: Panel) -> np.ndarray:
    output = np.empty((len(weights), len(panel.family_names)), dtype=float)
    for family_index, members in enumerate(panel.family_members):
        output[:, family_index] = weights[:, members].sum(axis=1)
    return output


def group_sum(values: np.ndarray, panel: Panel, *, weighted: bool = False) -> np.ndarray:
    output = np.empty((len(values), len(panel.group_names)), dtype=float)
    for group_index, members in enumerate(panel.group_members):
        if weighted:
            weights = panel.proportional[members]
            weights = weights / weights.sum()
            output[:, group_index] = values[:, members] @ weights
        else:
            output[:, group_index] = values[:, members].sum(axis=1)
    return output


def group_weight(weights: np.ndarray, panel: Panel) -> np.ndarray:
    output = np.empty((len(weights), len(panel.group_names)), dtype=float)
    for group_index, members in enumerate(panel.group_members):
        output[:, group_index] = weights[:, members].sum(axis=1)
    return output


def reference_family_unique(panel: Panel) -> np.ndarray:
    proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
    _phase0, _phase1, total = simulated_epochs(panel, proportional)
    return family_sum(unique_coverage(total), panel, weighted=True)[0]


def normalized_family_coverage(exposure: np.ndarray, panel: Panel) -> np.ndarray:
    coverage = family_sum(unique_coverage(exposure), panel, weighted=True)
    return coverage / np.maximum(reference_family_unique(panel), 1e-8)


def normalized_group_coverage(exposure: np.ndarray, panel: Panel) -> np.ndarray:
    proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
    _phase0, _phase1, reference_exposure = simulated_epochs(panel, proportional)
    coverage = group_sum(unique_coverage(exposure), panel, weighted=True)
    reference = group_sum(unique_coverage(reference_exposure), panel, weighted=True)
    return coverage / np.maximum(reference, 1e-8)


def family_power_coverage(group_coverage: np.ndarray, panel: Panel, order: float, floor: float) -> np.ndarray:
    output = np.empty((len(group_coverage), len(panel.family_names)), dtype=float)
    group_mass = np.asarray(
        [panel.proportional[members].sum() for members in panel.group_members],
        dtype=float,
    )
    safe = np.maximum(group_coverage, 0.0) + floor
    for family_index in range(len(panel.family_names)):
        selected = np.flatnonzero(panel.group_family_indices == family_index)
        weights = group_mass[selected]
        weights /= weights.sum()
        if abs(order) < 1e-10:
            mean = np.exp(np.log(safe[:, selected]) @ weights)
        else:
            mean = np.power(np.power(safe[:, selected], order) @ weights, 1.0 / order)
        output[:, family_index] = np.maximum(mean - floor, 0.0)
    return output


def family_ces_deficit(
    group_ratio: np.ndarray,
    panel: Panel,
    substitution_order: float,
    floor: float,
    alpha: float,
) -> np.ndarray:
    """Return family deficits from a complementary CES production law.

    The productive state is the generalized mean of order
    ``-substitution_order`` over normalized group evidence. Group shares are
    fixed by proportional-policy mass rather than learned from the target.
    The output is zero at the proportional reference and grows when the
    family's productive state falls below it.
    """

    if substitution_order <= 0.0:
        raise ValueError("CES substitution order must be positive")
    group_mass = np.asarray(
        [panel.proportional[members].sum() for members in panel.group_members],
        dtype=float,
    )
    safe = np.maximum(group_ratio, 0.0) + floor
    output = np.empty((len(group_ratio), len(panel.family_names)), dtype=float)
    reference = math.pow(1.0 + floor, -alpha)
    for family_index in range(len(panel.family_names)):
        selected = np.flatnonzero(panel.group_family_indices == family_index)
        shares = group_mass[selected]
        shares /= shares.sum()
        inverse_power_mean = np.power(safe[:, selected], -substitution_order) @ shares
        output[:, family_index] = np.power(inverse_power_mean, alpha / substitution_order) - reference
    return output


def parallel_family_failure(
    group_ratio: np.ndarray,
    panel: Panel,
    prior: float,
) -> np.ndarray:
    """Return failure debt for parallel group support within each family.

    A structural group's learned-capability probability is the stationary
    availability of competing acquisition and loss hazards,
    ``success = ratio / (prior + ratio)``. Groups inside a semantic family are
    parallel substitutes, so the family succeeds unless every group fails.
    The weighted geometric failure probability uses proportional group mass
    as a fixed, target-independent importance measure. The output is log
    failure debt relative to the proportional policy and is zero at ratio one.
    """

    if prior <= 0.0:
        raise ValueError("Reliability prior must be positive")
    group_mass = np.asarray(
        [panel.proportional[members].sum() for members in panel.group_members],
        dtype=float,
    )
    failure = prior / (prior + np.maximum(group_ratio, 0.0))
    reference_failure = prior / (prior + 1.0)
    output = np.empty((len(group_ratio), len(panel.family_names)), dtype=float)
    for family_index in range(len(panel.family_names)):
        selected = np.flatnonzero(panel.group_family_indices == family_index)
        weights = group_mass[selected]
        weights /= weights.sum()
        all_fail = np.exp(np.log(np.maximum(failure[:, selected], 1e-12)) @ weights)
        reference_all_fail = reference_failure
        family_success = np.maximum(1.0 - all_fail, 1e-12)
        reference_success = max(1.0 - reference_all_fail, 1e-12)
        output[:, family_index] = np.log(reference_success) - np.log(family_success)
    return output


def recency_phase_masses(phase_fractions: np.ndarray, rate: float) -> np.ndarray:
    alpha0 = float(phase_fractions[0])
    if abs(rate) < 1e-10:
        return np.asarray([alpha0, 1.0 - alpha0], dtype=float)
    denominator = 1.0 - math.exp(-rate)
    early = (math.exp(-rate * (1.0 - alpha0)) - math.exp(-rate)) / denominator
    return np.asarray([early, 1.0 - early], dtype=float)


def recency_exposure(panel: Panel, weights: np.ndarray, rate: float) -> np.ndarray:
    phase0, phase1, _total = simulated_epochs(panel, weights)
    kernel_mass = recency_phase_masses(panel.phase_fractions, rate)
    phase_scale = kernel_mass / panel.phase_fractions
    return phase_scale[0] * phase0 + phase_scale[1] * phase1


def learning_rate_plasticity_exposure(panel: Panel, weights: np.ndarray, power: float) -> np.ndarray:
    """Integrate exposure under the panel's normalized learning-rate schedule.

    ``power=0`` is physical token exposure. For positive powers, the useful
    acquisition dose per token is proportional to ``learning_rate**power``.
    The kernel is normalized to unit mean over training, so a phase-tied
    policy has the same total dose as physical exposure. WSD panels use a
    plateau followed by a cosine decay; the historical StarCoder cosine panel
    uses a full-training cosine schedule.
    """

    if power < 0.0:
        raise ValueError("Plasticity power must be nonnegative")
    if abs(power) < 1e-12:
        _phase0, _phase1, total = simulated_epochs(panel, weights)
        return total

    boundary = float(panel.phase_fractions[0])
    time = np.linspace(0.0, 1.0, 8193)
    if panel.name.startswith("starcoder_cosine"):
        learning_rate = 0.5 * (1.0 + np.cos(np.pi * time))
    else:
        decay_progress = np.maximum((time - boundary) / max(1.0 - boundary, 1e-12), 0.0)
        learning_rate = np.where(
            time <= boundary,
            1.0,
            0.5 * (1.0 + np.cos(np.pi * np.minimum(decay_progress, 1.0))),
        )
    plasticity = learning_rate**power
    total_mean = float(np.trapezoid(plasticity, time))
    phase_means = np.asarray(
        [
            np.trapezoid(plasticity[time <= boundary], time[time <= boundary]) / boundary,
            np.trapezoid(plasticity[time >= boundary], time[time >= boundary]) / (1.0 - boundary),
        ],
        dtype=float,
    )
    phase_means /= total_mean
    phase0, phase1, _total = simulated_epochs(panel, weights)
    return phase_means[0] * phase0 + phase_means[1] * phase1


def gradient_noise_limited_exposure(panel: Panel, weights: np.ndarray, sensitivity: float) -> np.ndarray:
    """Reduce useful updates by phase-local mixture-induced gradient variance.

    ``sum_i w_i**2 / p_i`` is one at the proportional reference and is the
    second moment of policy-to-reference importance weights. ``sensitivity``
    maps excess variance to lost useful optimizer steps. Replay remains a
    function of physical exposure in the response model.
    """

    if sensitivity < 0.0:
        raise ValueError("Gradient-noise sensitivity must be nonnegative")
    phase0, phase1, _total = simulated_epochs(panel, weights)
    inverse_reference = 1.0 / np.maximum(panel.proportional, 1e-12)
    variance = np.sum(weights**2 * inverse_reference[None, None, :], axis=2)
    efficiency = 1.0 / (1.0 + sensitivity * np.maximum(variance - 1.0, 0.0))
    return efficiency[:, 0, None] * phase0 + efficiency[:, 1, None] * phase1


def retained_state(panel: Panel, weights: np.ndarray, acquisition: float, forgetting: float) -> np.ndarray:
    phase0, phase1, _total = simulated_epochs(panel, weights)
    state = np.zeros_like(phase0)
    for phase_exposure, phase_weight, duration in (
        (phase0, weights[:, 0], panel.phase_fractions[0]),
        (phase1, weights[:, 1], panel.phase_fractions[1]),
    ):
        learning_hazard = acquisition * phase_exposure
        forgetting_hazard = forgetting * duration * np.maximum(1.0 - phase_weight, 0.0)
        total_hazard = learning_hazard + forgetting_hazard
        equilibrium = np.divide(
            learning_hazard,
            total_hazard,
            out=np.zeros_like(total_hazard),
            where=total_hazard > 1e-12,
        )
        state = equilibrium + (state - equilibrium) * np.exp(-total_hazard)
    return state


def family_compatible_state(
    panel: Panel,
    weights: np.ndarray,
    acquisition: float,
    forgetting: float,
) -> np.ndarray:
    phase0, phase1, _total = simulated_epochs(panel, weights)
    state = np.zeros((len(weights), len(panel.group_names)), dtype=float)
    for phase_exposure, phase_weight, duration in (
        (phase0, weights[:, 0], panel.phase_fractions[0]),
        (phase1, weights[:, 1], panel.phase_fractions[1]),
    ):
        group_exposure = group_sum(phase_exposure, panel)
        family_mass = family_weight(phase_weight, panel)
        compatible_mass = family_mass[:, panel.group_family_indices]
        learning_hazard = acquisition * group_exposure
        forgetting_hazard = forgetting * duration * np.maximum(1.0 - compatible_mass, 0.0)
        total_hazard = learning_hazard + forgetting_hazard
        equilibrium = np.divide(
            learning_hazard,
            total_hazard,
            out=np.zeros_like(total_hazard),
            where=total_hazard > 1e-12,
        )
        state = equilibrium + (state - equilibrium) * np.exp(-total_hazard)
    return state


def sequential_error_mass(
    panel: Panel,
    weights: np.ndarray,
    acquisition: float,
    forgetting: float,
    competition: float,
) -> np.ndarray:
    """Return unresolved bucket error after sequential evidence accumulation.

    Dimensionless evidence follows ``dI/dt = a*w - d*(1-w_family)*I``.
    The exact constant-mixture transition is used within each phase, so merely
    subdividing a phase leaves the state unchanged. The optional competition
    factor reduces the acquisition rate in proportion to simultaneous mass
    outside the bucket's declared semantic family. Surviving error is
    ``exp(-I)``.
    """
    phase0, phase1, _total = simulated_epochs(panel, weights)
    evidence = np.zeros_like(phase0)
    for phase_exposure, phase_weight, duration in (
        (phase0, weights[:, 0], panel.phase_fractions[0]),
        (phase1, weights[:, 1], panel.phase_fractions[1]),
    ):
        family_mass = family_weight(phase_weight, panel)
        compatible_mass = family_mass[:, panel.group_family_indices]
        # Structural groups never cross semantic families; map each bucket
        # through its singleton/paired group to the corresponding family mass.
        bucket_family_mass = np.empty_like(phase_weight)
        for group_index, members in enumerate(panel.group_members):
            bucket_family_mass[:, members] = compatible_mass[:, group_index, None]
        effective_exposure = phase_exposure / (1.0 + competition * np.maximum(1.0 - bucket_family_mass, 0.0))
        hazard = forgetting * duration * np.maximum(1.0 - bucket_family_mass, 0.0)
        retention = np.exp(-hazard)
        transition_gain = np.divide(
            -np.expm1(-hazard),
            hazard,
            out=np.ones_like(hazard),
            where=hazard > 1e-12,
        )
        evidence = retention * evidence + acquisition * effective_exposure * transition_gain
    return np.exp(-evidence)


def bounded_coverage_state(
    panel: Panel,
    weights: np.ndarray,
    forgetting: float,
) -> np.ndarray:
    """Accumulate unique coverage with compatible-family forgetting.

    Coverage is a fraction in ``[0, 1]``. Within each phase, retained state
    first decays under exposure outside the bucket's semantic family, then
    unseen evidence is acquired with the exact Poisson unique-coverage law.
    At zero forgetting, phase subdivision is an exact semigroup:
    ``u = 1 - exp(-sum_t e_t)``.
    """

    phase0, phase1, _total = simulated_epochs(panel, weights)
    state = np.zeros_like(phase0)
    for phase_exposure, phase_weight, duration in (
        (phase0, weights[:, 0], panel.phase_fractions[0]),
        (phase1, weights[:, 1], panel.phase_fractions[1]),
    ):
        family_mass = family_weight(phase_weight, panel)
        group_compatible = family_mass[:, panel.group_family_indices]
        bucket_compatible = np.empty_like(phase_weight)
        for group_index, members in enumerate(panel.group_members):
            bucket_compatible[:, members] = group_compatible[:, group_index, None]
        state *= np.exp(-forgetting * duration * np.maximum(1.0 - bucket_compatible, 0.0))
        acquisition = -np.expm1(-np.maximum(phase_exposure, 0.0))
        state += (1.0 - state) * acquisition
    return np.clip(state, 0.0, 1.0)


def replay_hazard_state(
    panel: Panel,
    weights: np.ndarray,
    hazard_rate: float,
    integration_steps: int = 64,
) -> np.ndarray:
    """Integrate competence acquired from novel samples and lost to replay.

    Cumulative bucket exposure ``E_i`` makes the unseen probability
    ``exp(-E_i)``. The global duplicate-token fraction is therefore
    ``sum_i w_i * (1 - exp(-E_i))``. Competence receives novel evidence at
    rate ``d(1-exp(-E_i))/dt`` and decays under the duplicate-token hazard.
    A vectorized RK4 integration preserves the ordering of the two phases.
    """

    weights = np.asarray(weights, dtype=float)
    phase0, phase1, _total = simulated_epochs(panel, weights)
    if hazard_rate == 0.0:
        return unique_coverage(phase0 + phase1)

    state = np.zeros_like(phase0)
    cumulative = np.zeros_like(phase0)

    def derivative(
        current_state: np.ndarray,
        current_exposure: np.ndarray,
        phase_weight: np.ndarray,
        exposure_rate: np.ndarray,
    ) -> np.ndarray:
        unseen = np.exp(-np.maximum(current_exposure, 0.0))
        duplicate_fraction = np.sum(phase_weight * (1.0 - unseen), axis=1)
        return exposure_rate * unseen - hazard_rate * duplicate_fraction[:, None] * current_state

    for phase_exposure, phase_weight, duration in (
        (phase0, weights[:, 0], panel.phase_fractions[0]),
        (phase1, weights[:, 1], panel.phase_fractions[1]),
    ):
        steps = max(1, math.ceil(integration_steps * duration))
        step = duration / steps
        exposure_rate = phase_exposure / duration
        for _ in range(steps):
            half_exposure = cumulative + 0.5 * step * exposure_rate
            full_exposure = cumulative + step * exposure_rate
            k1 = derivative(state, cumulative, phase_weight, exposure_rate)
            k2 = derivative(state + 0.5 * step * k1, half_exposure, phase_weight, exposure_rate)
            k3 = derivative(state + 0.5 * step * k2, half_exposure, phase_weight, exposure_rate)
            k4 = derivative(state + step * k3, full_exposure, phase_weight, exposure_rate)
            state += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            cumulative = full_exposure
    return np.maximum(state, 0.0)


def posterior_precision_state(
    panel: Panel,
    weights: np.ndarray,
    forgetting: float,
) -> np.ndarray:
    """Accumulate group-level information with out-of-family process noise.

    Precision is measured in equivalent independent dataset passes. During a
    phase it follows ``dP_g/dt = x_g - h_g P_g``, where ``x_g`` is the group
    evidence rate and ``h_g`` is proportional to the policy mass outside the
    group's declared semantic family. The exact constant-phase transition is
    used. At zero forgetting, precision is physical aggregate exposure and is
    therefore exactly invariant to phase subdivision and ordering.
    """

    phase0, phase1, _total = simulated_epochs(panel, weights)
    state = np.zeros((len(weights), len(panel.group_names)), dtype=float)
    for phase_exposure, phase_weight, duration in (
        (phase0, weights[:, 0], panel.phase_fractions[0]),
        (phase1, weights[:, 1], panel.phase_fractions[1]),
    ):
        group_exposure = group_sum(phase_exposure, panel)
        family_mass = family_weight(phase_weight, panel)
        compatible_mass = family_mass[:, panel.group_family_indices]
        hazard = forgetting * duration * np.maximum(1.0 - compatible_mass, 0.0)
        retention = np.exp(-hazard)
        transition_gain = np.divide(
            -np.expm1(-hazard),
            hazard,
            out=np.ones_like(hazard),
            where=hazard > 1e-12,
        )
        state = retention * state + transition_gain * group_exposure
    return np.maximum(state, 0.0)


def riccati_uncertainty_state(
    panel: Panel,
    weights: np.ndarray,
    prior_variance: float,
    process_noise: float,
) -> np.ndarray:
    """Propagate family uncertainty under evidence and interference.

    For group ``g``, posterior variance follows the scalar Kalman--Bucy
    Riccati equation ``dV_g/dt = q_g - r_g V_g**2``. The evidence rate
    ``r_g`` is measured in equivalent dataset passes per unit normalized
    training time. The process-variance rate ``q_g`` is proportional to
    policy mass outside the group's declared family. Both are constant within
    a phase, so the transition is exact and composes across arbitrary phase
    subdivisions. ``V`` has units of inverse equivalent passes; process noise
    has the same units per normalized training time.
    """

    if prior_variance <= 0.0 or process_noise < 0.0:
        raise ValueError("Riccati uncertainty parameters are out of range")
    phase0, phase1, _total = simulated_epochs(panel, weights)
    state = np.full((len(weights), len(panel.group_names)), prior_variance, dtype=float)
    for phase_exposure, phase_weight, duration in (
        (phase0, weights[:, 0], panel.phase_fractions[0]),
        (phase1, weights[:, 1], panel.phase_fractions[1]),
    ):
        evidence_rate = group_sum(phase_exposure, panel) / duration
        family_mass = family_weight(phase_weight, panel)
        compatible_mass = family_mass[:, panel.group_family_indices]
        variance_rate = process_noise * np.maximum(1.0 - compatible_mass, 0.0)

        no_process = variance_rate <= 1e-15
        no_evidence = evidence_rate <= 1e-15
        only_evidence = no_process & ~no_evidence
        state[only_evidence] /= 1.0 + evidence_rate[only_evidence] * state[only_evidence] * duration
        only_process = ~no_process & no_evidence
        state[only_process] += variance_rate[only_process] * duration

        coupled = ~no_process & ~no_evidence
        if np.any(coupled):
            equilibrium = np.sqrt(variance_rate[coupled] / evidence_rate[coupled])
            transition = np.tanh(np.sqrt(variance_rate[coupled] * evidence_rate[coupled]) * duration)
            previous = state[coupled]
            state[coupled] = equilibrium * (previous + equilibrium * transition) / (equilibrium + previous * transition)
    return np.maximum(state, 1e-15)


def two_pool_consolidation_state(
    panel: Panel,
    weights: np.ndarray,
    acquisition: float,
    forgetting: float,
    consolidation: float,
    slow_weight: float,
    integration_steps: int = 64,
) -> np.ndarray:
    """Track fast competence and its slower consolidated copy.

    Fast competence learns from in-group evidence and is displaced by
    out-of-family updates. Slow competence follows the fast state without a
    direct forgetting channel:

    ``df/dt = a r (1-f) - h o f`` and ``ds/dt = k (f-s)``.

    The terminal capability is a fixed convex mixture of the two pools. All
    rates are per normalized training time; both states are dimensionless in
    ``[0, 1]``. The ODE is autonomous for a fixed policy, so tied schedules
    are invariant to artificial phase subdivision.
    """

    if min(acquisition, forgetting, consolidation) < 0.0 or not 0.0 <= slow_weight <= 1.0:
        raise ValueError("Invalid two-pool consolidation parameters")
    phase0, phase1, _total = simulated_epochs(panel, weights)
    fast = np.zeros((len(weights), len(panel.group_names)), dtype=float)
    slow = np.zeros_like(fast)
    for phase_exposure, phase_weight, duration in (
        (phase0, weights[:, 0], panel.phase_fractions[0]),
        (phase1, weights[:, 1], panel.phase_fractions[1]),
    ):
        evidence_rate = group_sum(phase_exposure, panel) / duration
        family_mass = family_weight(phase_weight, panel)
        compatible_mass = family_mass[:, panel.group_family_indices]
        outside_mass = np.maximum(1.0 - compatible_mass, 0.0)
        steps = max(1, math.ceil(integration_steps * duration))
        step = duration / steps

        def derivative(
            fast_state: np.ndarray,
            slow_state: np.ndarray,
            evidence_rate: np.ndarray = evidence_rate,
            outside_mass: np.ndarray = outside_mass,
        ) -> tuple[np.ndarray, np.ndarray]:
            fast_change = acquisition * evidence_rate * (1.0 - fast_state)
            fast_change -= forgetting * outside_mass * fast_state
            slow_change = consolidation * (fast_state - slow_state)
            return fast_change, slow_change

        for _ in range(steps):
            k1_fast, k1_slow = derivative(fast, slow)
            k2_fast, k2_slow = derivative(fast + 0.5 * step * k1_fast, slow + 0.5 * step * k1_slow)
            k3_fast, k3_slow = derivative(fast + 0.5 * step * k2_fast, slow + 0.5 * step * k2_slow)
            k4_fast, k4_slow = derivative(fast + step * k3_fast, slow + step * k3_slow)
            fast += step * (k1_fast + 2.0 * k2_fast + 2.0 * k3_fast + k4_fast) / 6.0
            slow += step * (k1_slow + 2.0 * k2_slow + 2.0 * k3_slow + k4_slow) / 6.0
    capability = (1.0 - slow_weight) * fast + slow_weight * slow
    return np.clip(capability, 0.0, 1.0)


def concentration_displacement_state(
    panel: Panel,
    weights: np.ndarray,
    acquisition: float,
    displacement: float,
) -> np.ndarray:
    """Track competence lost to concentrated out-of-family updates.

    Family concentration is normalized to the proportional policy's
    Herfindahl index. Excess concentration is a global gradient-alignment load;
    it displaces a group's state in proportion to mass outside that group's
    family. The resulting rank-one competition channel is
    ``h_g = displacement * excess_concentration * outside_mass_g``.
    """

    if acquisition < 0.0 or displacement < 0.0:
        raise ValueError("Concentration-displacement rates must be nonnegative")
    phase0, phase1, _total = simulated_epochs(panel, weights)
    proportional_family = family_weight(panel.proportional[None, :], panel)[0]
    reference_concentration = float(proportional_family @ proportional_family)
    state = np.zeros((len(weights), len(panel.group_names)), dtype=float)
    for phase_exposure, phase_weight, duration in (
        (phase0, weights[:, 0], panel.phase_fractions[0]),
        (phase1, weights[:, 1], panel.phase_fractions[1]),
    ):
        group_exposure = group_sum(phase_exposure, panel)
        family_mass = family_weight(phase_weight, panel)
        compatible_mass = family_mass[:, panel.group_family_indices]
        concentration = np.sum(family_mass**2, axis=1) / reference_concentration
        excess = np.maximum(concentration - 1.0, 0.0)
        learning_hazard = acquisition * group_exposure
        displacement_hazard = displacement * duration * excess[:, None] * np.maximum(1.0 - compatible_mass, 0.0)
        total_hazard = learning_hazard + displacement_hazard
        equilibrium = np.divide(
            learning_hazard,
            total_hazard,
            out=np.zeros_like(total_hazard),
            where=total_hazard > 1e-12,
        )
        state = equilibrium + (state - equilibrium) * np.exp(-total_hazard)
    return np.clip(state, 0.0, 1.0)


def diversity_gated_exposure(
    panel: Panel,
    weights: np.ndarray,
    floor: float,
    sensitivity: float,
) -> np.ndarray:
    """Scale acquisition by proportional-mass-weighted family support.

    For each constant-mixture phase, family mass is divided by its
    proportional reference. The global acquisition efficiency is the weighted
    geometric mean of these ratios, with proportional family mass as the
    weights and ``floor`` equivalent reference support. Efficiency is capped
    at one: balanced surplus cannot make an optimizer more than fully
    efficient, while starving a foundation family slows every acquisition
    channel. ``sensitivity=0`` is exactly physical exposure.

    Ratios, floor, and efficiency are dimensionless. The output has units of
    simulated passes, matching physical exposure. A tied policy applies the
    same autonomous gate in both phases and is invariant to the phase split.
    """

    if floor <= 0.0 or sensitivity < 0.0:
        raise ValueError("Diversity-gate floor must be positive and sensitivity nonnegative")
    proportional_family = family_weight(panel.proportional[None, :], panel)[0]
    family_weights = proportional_family / proportional_family.sum()
    phase0, phase1, _total = simulated_epochs(panel, weights)
    effective = np.zeros_like(phase0)
    for phase_exposure, phase_weight in ((phase0, weights[:, 0]), (phase1, weights[:, 1])):
        phase_family = family_weight(phase_weight, panel)
        ratio = phase_family / np.maximum(proportional_family[None, :], 1e-12)
        log_health = np.log((np.maximum(ratio, 0.0) + floor) / (1.0 + floor)) @ family_weights
        health = np.minimum(np.exp(log_health), 1.0)
        efficiency = np.power(health, sensitivity)
        effective += phase_exposure * efficiency[:, None]
    return effective


def learned_state_competition(
    panel: Panel,
    weights: np.ndarray,
    acquisition: float,
    competition: float,
    integration_steps: int = 128,
) -> np.ndarray:
    """Track bounded group competence under competition from learned families.

    Group competence ``z_g`` is dimensionless and follows

    ``dz_g/dt = a r_g (1 - z_g) - c z_g sum_{f != f(g)} q_f z_f``.

    Here ``r_g`` is the physical group-exposure rate, ``q_f`` is the current
    policy mass of family ``f``, and ``z_f`` is the proportional-mass-weighted
    competence of that family. Competition therefore requires both an active
    competing data stream and a learned competing representation. Rates
    ``a`` and ``c`` have units of inverse normalized training time. At
    ``c=0`` the groups learn independently; a tied policy is one autonomous
    ODE and is invariant to an artificial phase boundary.
    """

    if acquisition <= 0.0 or competition < 0.0:
        raise ValueError("Acquisition must be positive and competition nonnegative")
    phase0, phase1, _total = simulated_epochs(panel, weights)
    state = np.zeros((len(weights), len(panel.group_names)), dtype=float)
    family_group_weights: list[tuple[np.ndarray, np.ndarray]] = []
    group_mass = np.asarray(
        [panel.proportional[members].sum() for members in panel.group_members],
        dtype=float,
    )
    for family_index in range(len(panel.family_names)):
        group_indices = np.flatnonzero(panel.group_family_indices == family_index)
        relative_mass = group_mass[group_indices]
        relative_mass /= relative_mass.sum()
        family_group_weights.append((group_indices, relative_mass))

    for phase_exposure, phase_weight, duration in (
        (phase0, weights[:, 0], panel.phase_fractions[0]),
        (phase1, weights[:, 1], panel.phase_fractions[1]),
    ):
        evidence_rate = group_sum(phase_exposure, panel) / duration
        family_mass = family_weight(phase_weight, panel)
        steps = max(1, math.ceil(integration_steps * duration))
        step = duration / steps

        def derivative(
            current: np.ndarray,
            evidence_rate: np.ndarray = evidence_rate,
            family_mass: np.ndarray = family_mass,
        ) -> np.ndarray:
            family_state = np.empty((len(weights), len(panel.family_names)), dtype=float)
            for family_index, (group_indices, relative_mass) in enumerate(family_group_weights):
                family_state[:, family_index] = current[:, group_indices] @ relative_mass
            active_family = family_mass * family_state
            total_active = active_family.sum(axis=1, keepdims=True)
            own_active = active_family[:, panel.group_family_indices]
            pressure = np.maximum(total_active - own_active, 0.0)
            gain = acquisition * evidence_rate * (1.0 - current)
            loss = competition * pressure * current
            return gain - loss

        for _ in range(steps):
            k1 = derivative(state)
            k2 = derivative(state + 0.5 * step * k1)
            k3 = derivative(state + 0.5 * step * k2)
            k4 = derivative(state + step * k3)
            state += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            state = np.clip(state, 0.0, 1.0)
    return state


def representation_capacity_state(
    panel: Panel,
    weights: np.ndarray,
    adaptation_rate: float,
) -> np.ndarray:
    """Track finite group-level representation capacity across phases.

    The state is a dimensionless share of a fixed representation budget and
    relaxes toward the current mixture's group mass:
    ``dz_g/dt = rate * (q_g - z_g)``. The exact transition makes the state
    phase-subdivision invariant. Small rate recovers an aggregate-mixture
    model; large rate approaches a final-phase representation allocation.
    """

    if adaptation_rate <= 0.0:
        raise ValueError("Representation adaptation rate must be positive")
    state = np.zeros((len(weights), len(panel.group_names)), dtype=float)
    for phase_weight, duration in (
        (weights[:, 0], panel.phase_fractions[0]),
        (weights[:, 1], panel.phase_fractions[1]),
    ):
        target = group_weight(phase_weight, panel)
        retention = math.exp(-adaptation_rate * duration)
        state = target + (state - target) * retention
    return np.maximum(state, 0.0)


def finite_subset_retained_state(
    panel: Panel,
    weights: np.ndarray,
    forgetting: float,
) -> np.ndarray:
    """Track retained competence from exact finite-subset traversal.

    Each bucket's materialized subset is traversed without replacement and
    then recycled. The cumulative unique fraction is therefore
    ``min(E, 1)``. Previously learned competence decays only under updates
    outside the bucket's declared family; newly encountered examples add
    linearly. At zero forgetting the terminal state is exactly
    ``min(E0 + E1, 1)`` and is independent of phase order.
    """

    phase0, phase1, _total = simulated_epochs(panel, weights)
    state = np.zeros_like(phase0)
    cumulative = np.zeros_like(phase0)
    for phase_exposure, phase_weight, duration in (
        (phase0, weights[:, 0], panel.phase_fractions[0]),
        (phase1, weights[:, 1], panel.phase_fractions[1]),
    ):
        family_mass = family_weight(phase_weight, panel)
        group_compatible = family_mass[:, panel.group_family_indices]
        bucket_compatible = np.empty_like(phase_weight)
        for group_index, members in enumerate(panel.group_members):
            bucket_compatible[:, members] = group_compatible[:, group_index, None]
        retention = np.exp(-forgetting * duration * np.maximum(1.0 - bucket_compatible, 0.0))
        previous_unique = finite_subset_coverage(cumulative)
        cumulative += phase_exposure
        new_unique = finite_subset_coverage(cumulative) - previous_unique
        state = retention * state + new_unique
    return np.clip(state, 0.0, 1.0)


def power_law_memory_exposure(
    panel: Panel,
    weights: np.ndarray,
    timescale: float,
    exponent: float,
    quadrature_steps: int = 32,
) -> np.ndarray:
    """Integrate evidence retained by a power-law interference kernel.

    An example acquired at time ``t`` survives future out-of-family update
    mass ``A`` with probability ``(1 + A / timescale)**(-exponent)``. This is
    the survival law induced by a decreasing hazard, unlike the memoryless
    exponential kernel. The future interference integral is exact within each
    constant-mixture phase; midpoint quadrature integrates acquisition time.
    A phase-tied policy is invariant to where the phase boundary is drawn.
    """

    if timescale <= 0.0 or exponent <= 0.0:
        raise ValueError("Power-law memory parameters must be positive")
    phase0, phase1, _total = simulated_epochs(panel, weights)
    phase_exposures = (phase0, phase1)
    phase_weights = (weights[:, 0], weights[:, 1])
    durations = panel.phase_fractions
    family_masses = [family_weight(phase_weight, panel) for phase_weight in phase_weights]
    retained = np.zeros_like(phase0)
    for phase_index, (phase_exposure, duration) in enumerate(zip(phase_exposures, durations, strict=True)):
        exposure_rate = phase_exposure / duration
        later_interference = np.zeros((len(weights), len(panel.family_names)), dtype=float)
        for later_index in range(phase_index + 1, 2):
            later_interference += durations[later_index] * (1.0 - family_masses[later_index])
        step = duration / quadrature_steps
        current_outside = 1.0 - family_masses[phase_index]
        for step_index in range(quadrature_steps):
            remaining = duration - (step_index + 0.5) * step
            future_family_interference = later_interference + remaining * current_outside
            bucket_interference = np.empty_like(phase_exposure)
            for group_index, members in enumerate(panel.group_members):
                family_index = panel.group_family_indices[group_index]
                bucket_interference[:, members] = future_family_interference[:, family_index, None]
            survival = np.power(1.0 + bucket_interference / timescale, -exponent)
            retained += exposure_rate * step * survival
    return retained


def _bucket_names(prefix: str, panel: Panel) -> tuple[str, ...]:
    return tuple(f"{prefix}:{domain}" for domain in panel.domains)


def _family_names(prefix: str, panel: Panel) -> tuple[str, ...]:
    return tuple(f"{prefix}:{family}" for family in panel.family_names)


def _group_names(prefix: str, panel: Panel) -> tuple[str, ...]:
    return tuple(f"{prefix}:{group}" for group in panel.group_names)


def build_design(panel: Panel, weights: np.ndarray, config: ModelConfig) -> Design:
    phase0, phase1, total = simulated_epochs(panel, weights)
    replay = literal_replay(total)
    if config.family == "unique_replay":
        values = np.column_stack([-unique_coverage(total), replay])
        names = (*_bucket_names("benefit", panel), *_bucket_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family == "recency_unique_replay":
        effective = recency_exposure(panel, weights, config.parameter("recency"))
        values = np.column_stack([-unique_coverage(effective), replay])
        names = (*_bucket_names("recency_benefit", panel), *_bucket_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family == "retained_state_ode":
        state = retained_state(
            panel,
            weights,
            acquisition=config.parameter("acquisition"),
            forgetting=config.parameter("forgetting"),
        )
        values = np.column_stack([-state, replay])
        names = (*_bucket_names("retained_benefit", panel), *_bucket_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family == "family_survival":
        coverage = normalized_family_coverage(total, panel)
        rate = config.parameter("rate")
        deficit = np.exp(-rate * coverage) - math.exp(-rate)
        values = np.column_stack([deficit, replay])
        names = (*_family_names("unresolved_mass", panel), *_bucket_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family == "weibull_shortage":
        scale = config.parameter("scale")
        power = config.parameter("power")
        unresolved = np.exp(-np.power(np.maximum(total, 0.0) / scale, power))
        values = np.column_stack([unresolved, replay])
        names = (*_bucket_names("weibull_unresolved", panel), *_bucket_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family == "soft_bottleneck":
        alpha = config.parameter("alpha")
        temperature = config.parameter("temperature")
        delta = 0.05
        coverage = normalized_family_coverage(total, panel)
        deficit = np.power(coverage + delta, -alpha) - math.pow(1.0 + delta, -alpha)
        shortage = np.maximum(deficit, 0.0)
        bottleneck = temperature * (logsumexp(shortage / temperature, axis=1) - math.log(shortage.shape[1]))
        values = np.column_stack([deficit, bottleneck, replay])
        names = (
            *_family_names("family_deficit", panel),
            "soft_weakest_capability",
            *_bucket_names("literal_replay", panel),
        )
        return Design(values=values, names=names)

    if config.family == "competition_unique_replay":
        strength = config.parameter("strength")
        proportional_family = family_weight(panel.proportional[None, :], panel)[0]
        phase0_family = family_weight(weights[:, 0], panel)
        phase1_family = family_weight(weights[:, 1], panel)
        reference_concentration = float(np.sum(proportional_family**2))
        concentration0 = np.maximum(np.sum(phase0_family**2, axis=1) - reference_concentration, 0.0)
        concentration1 = np.maximum(np.sum(phase1_family**2, axis=1) - reference_concentration, 0.0)
        effective = phase0 / (1.0 + strength * concentration0[:, None])
        effective += phase1 / (1.0 + strength * concentration1[:, None])
        values = np.column_stack([-unique_coverage(effective), replay])
        names = (*_bucket_names("competition_adjusted_benefit", panel), *_bucket_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family == "phase_unique_heads":
        values = np.column_stack([-unique_coverage(phase0), -unique_coverage(phase1), replay])
        names = (
            *_bucket_names("early_benefit", panel),
            *_bucket_names("late_benefit", panel),
            *_bucket_names("literal_replay", panel),
        )
        return Design(values=values, names=names)

    if config.family in {"group_coverage_replay", "hierarchical_coverage", "series_reliability"}:
        recency = config.parameter("recency")
        effective = recency_exposure(panel, weights, recency)
        group_coverage = normalized_group_coverage(effective, panel)
        floor = config.parameter("floor")
        alpha = config.parameter("alpha")
        group_deficit = np.power(group_coverage + floor, -alpha) - math.pow(1.0 + floor, -alpha)
        physical_replay = group_sum(literal_replay(total), panel)
        pieces = [group_deficit]
        names: list[str] = list(_group_names("group_deficit", panel))
        if config.family == "hierarchical_coverage":
            family_coverage = family_power_coverage(
                group_coverage,
                panel,
                order=-config.parameter("bottleneck"),
                floor=floor,
            )
            family_deficit = np.power(family_coverage + floor, -alpha) - math.pow(1.0 + floor, -alpha)
            pieces.append(family_deficit)
            names.extend(_family_names("family_coverage_deficit", panel))
        elif config.family == "series_reliability":
            rate = config.parameter("rate")
            learned = np.maximum(-np.expm1(-rate * np.maximum(group_coverage, 0.0)), 1e-12)
            group_mass = np.asarray(
                [panel.proportional[members].sum() for members in panel.group_members],
                dtype=float,
            )
            reliability = np.empty((len(weights), len(panel.family_names)), dtype=float)
            reference_reliability = -math.expm1(-rate)
            for family_index in range(len(panel.family_names)):
                selected = np.flatnonzero(panel.group_family_indices == family_index)
                family_weights = group_mass[selected]
                family_weights /= family_weights.sum()
                reliability[:, family_index] = np.exp(np.log(learned[:, selected]) @ family_weights)
            pieces.append(reference_reliability - reliability)
            names.extend(_family_names("series_failure_mass", panel))
        pieces.append(physical_replay)
        names.extend(_group_names("literal_replay", panel))
        return Design(values=np.column_stack(pieces), names=tuple(names))

    if config.family == "overload_hazard_state":
        recency = config.parameter("recency")
        acquisition = config.parameter("acquisition")
        overload = config.parameter("overload")
        effective = recency_exposure(panel, weights, recency)
        physical_replay = literal_replay(total)
        state = -np.expm1(-acquisition * np.maximum(effective, 0.0))
        state *= np.exp(-overload * physical_replay)
        group_state = group_sum(state, panel, weighted=True)
        group_replay = group_sum(physical_replay, panel)
        values = np.column_stack([-group_state, group_replay])
        names = (*_group_names("overload_adjusted_state", panel), *_group_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family == "family_cross_forgetting":
        state = family_compatible_state(
            panel,
            weights,
            acquisition=config.parameter("acquisition"),
            forgetting=config.parameter("forgetting"),
        )
        group_replay = group_sum(literal_replay(total), panel)
        values = np.column_stack([-state, group_replay])
        names = (*_group_names("compatible_retained_state", panel), *_group_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family in {"sequential_error_mass", "competition_error_mass"}:
        unresolved = sequential_error_mass(
            panel,
            weights,
            acquisition=config.parameter("acquisition"),
            forgetting=config.parameter("forgetting"),
            competition=(config.parameter("competition") if config.family == "competition_error_mass" else 0.0),
        )
        group_replay = group_sum(literal_replay(total), panel)
        values = np.column_stack([unresolved, group_replay])
        names = (*_bucket_names("unresolved_error_mass", panel), *_group_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family in {
        "physical_scaling_deficit",
        "foundation_transfer_deficit",
        "learning_rate_plasticity_deficit",
        "gradient_noise_limited_deficit",
    }:
        if config.family == "physical_scaling_deficit":
            effective = total
        elif config.family == "learning_rate_plasticity_deficit":
            effective = learning_rate_plasticity_exposure(panel, weights, config.parameter("power"))
        elif config.family == "gradient_noise_limited_deficit":
            effective = gradient_noise_limited_exposure(panel, weights, config.parameter("sensitivity"))
        else:
            effective = foundation_gated_exposure(
                panel,
                weights,
                acquisition=config.parameter("acquisition"),
                boost=config.parameter("boost"),
            )
        proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
        if config.family == "physical_scaling_deficit":
            _reference0, _reference1, reference = simulated_epochs(panel, proportional)
        elif config.family == "learning_rate_plasticity_deficit":
            reference = learning_rate_plasticity_exposure(panel, proportional, config.parameter("power"))
        elif config.family == "gradient_noise_limited_deficit":
            reference = gradient_noise_limited_exposure(panel, proportional, config.parameter("sensitivity"))
        else:
            reference = foundation_gated_exposure(
                panel,
                proportional,
                acquisition=config.parameter("acquisition"),
                boost=config.parameter("boost"),
            )
        ratio = normalized_group_exposure(effective, reference, panel)
        floor = config.parameter("floor")
        alpha = config.parameter("alpha")
        deficit = np.power(np.maximum(ratio, 0.0) + floor, -alpha)
        deficit -= math.pow(1.0 + floor, -alpha)
        physical_replay = group_sum(literal_replay(total), panel)
        values = np.column_stack([deficit, physical_replay])
        names = (*_group_names("scaling_deficit", panel), *_group_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family in {"two_level_prior_deficit", "two_level_prior_recency_deficit"}:
        recency = config.parameter("recency") if config.family == "two_level_prior_recency_deficit" else 0.0
        effective = recency_exposure(panel, weights, recency)
        proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
        reference = recency_exposure(panel, proportional, recency)
        ratio = normalized_group_exposure(effective, reference, panel)
        floor = two_level_prior_floor(
            panel,
            foundation_floor=config.parameter("foundation_floor"),
            specialist_floor=config.parameter("specialist_floor"),
        )
        alpha = config.parameter("alpha")
        deficit = np.power(np.maximum(ratio, 0.0) + floor[None, :], -alpha)
        deficit -= np.power(1.0 + floor[None, :], -alpha)
        physical_replay = group_sum(literal_replay(total), panel)
        values = np.column_stack([deficit, physical_replay])
        names = (*_group_names("prior_adjusted_deficit", panel), *_group_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family == "bounded_coverage_deficit":
        forgetting = config.parameter("forgetting")
        state = bounded_coverage_state(panel, weights, forgetting)
        proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
        reference = bounded_coverage_state(panel, proportional, forgetting)
        grouped = group_sum(state, panel, weighted=True)
        grouped_reference = group_sum(reference, panel, weighted=True)[0]
        ratio = grouped / np.maximum(grouped_reference[None, :], 1e-12)
        floor = config.parameter("floor")
        alpha = config.parameter("alpha")
        deficit = np.power(np.maximum(ratio, 0.0) + floor, -alpha)
        deficit -= math.pow(1.0 + floor, -alpha)
        physical_replay = group_sum(literal_replay(total), panel)
        values = np.column_stack([deficit, physical_replay])
        names = (*_group_names("bounded_coverage_deficit", panel), *_group_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family in {"physical_ces_production", "retained_ces_production"}:
        proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
        if config.family == "physical_ces_production":
            effective = total
            _reference0, _reference1, reference = simulated_epochs(panel, proportional)
        else:
            forgetting = config.parameter("forgetting")
            effective = bounded_coverage_state(panel, weights, forgetting)
            reference = bounded_coverage_state(panel, proportional, forgetting)
        ratio = normalized_group_exposure(effective, reference, panel)
        deficit = family_ces_deficit(
            ratio,
            panel,
            substitution_order=config.parameter("substitution_order"),
            floor=config.parameter("floor"),
            alpha=config.parameter("alpha"),
        )
        family_replay = family_sum(literal_replay(total), panel)
        values = np.column_stack([deficit, family_replay])
        names = (*_family_names("ces_deficit", panel), *_family_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family == "parallel_reliability_network":
        effective = recency_exposure(panel, weights, config.parameter("recency"))
        proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
        reference = recency_exposure(panel, proportional, config.parameter("recency"))
        ratio = normalized_group_exposure(effective, reference, panel)
        family_failure = parallel_family_failure(ratio, panel, config.parameter("prior"))
        physical_replay = group_sum(literal_replay(total), panel)
        values = np.column_stack([family_failure, physical_replay])
        names = (*_family_names("parallel_failure_debt", panel), *_group_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family == "replay_hazard_deficit":
        state = replay_hazard_state(panel, weights, config.parameter("hazard_rate"))
        proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
        reference = replay_hazard_state(panel, proportional, config.parameter("hazard_rate"))
        ratio = normalized_group_exposure(state, reference, panel)
        floor = config.parameter("floor")
        alpha = config.parameter("alpha")
        deficit = np.power(np.maximum(ratio, 0.0) + floor, -alpha)
        deficit -= math.pow(1.0 + floor, -alpha)
        return Design(values=deficit, names=_group_names("replay_hazard_deficit", panel))

    if config.family == "posterior_precision_debt":
        forgetting = config.parameter("forgetting")
        prior = config.parameter("prior")
        state = posterior_precision_state(panel, weights, forgetting)
        proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
        reference = posterior_precision_state(panel, proportional, forgetting)
        debt = np.log(prior + reference) - np.log(prior + state)
        physical_replay = group_sum(literal_replay(total), panel)
        values = np.column_stack([debt, physical_replay])
        names = (*_group_names("posterior_log_precision_debt", panel), *_group_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family == "riccati_uncertainty_debt":
        prior_variance = config.parameter("prior_variance")
        process_noise = config.parameter("process_noise")
        state = riccati_uncertainty_state(panel, weights, prior_variance, process_noise)
        proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
        reference = riccati_uncertainty_state(panel, proportional, prior_variance, process_noise)
        uncertainty_debt = np.log(state) - np.log(reference)
        physical_replay = group_sum(literal_replay(total), panel)
        values = np.column_stack([uncertainty_debt, physical_replay])
        names = (*_group_names("riccati_log_variance_debt", panel), *_group_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family == "two_pool_consolidation":
        state = two_pool_consolidation_state(
            panel,
            weights,
            config.parameter("acquisition"),
            config.parameter("forgetting"),
            config.parameter("consolidation"),
            config.parameter("slow_weight"),
        )
        proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
        reference = two_pool_consolidation_state(
            panel,
            proportional,
            config.parameter("acquisition"),
            config.parameter("forgetting"),
            config.parameter("consolidation"),
            config.parameter("slow_weight"),
        )
        capability_debt = reference - state
        physical_replay = group_sum(literal_replay(total), panel)
        values = np.column_stack([capability_debt, physical_replay])
        names = (*_group_names("two_pool_capability_debt", panel), *_group_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family == "concentration_displacement":
        state = concentration_displacement_state(
            panel,
            weights,
            config.parameter("acquisition"),
            config.parameter("displacement"),
        )
        proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
        reference = concentration_displacement_state(
            panel,
            proportional,
            config.parameter("acquisition"),
            config.parameter("displacement"),
        )
        capability_debt = reference - state
        physical_replay = group_sum(literal_replay(total), panel)
        values = np.column_stack([capability_debt, physical_replay])
        names = (*_group_names("concentration_displacement_debt", panel), *_group_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family == "diversity_gated_deficit":
        effective = diversity_gated_exposure(
            panel,
            weights,
            config.parameter("diversity_floor"),
            config.parameter("sensitivity"),
        )
        proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
        reference = diversity_gated_exposure(
            panel,
            proportional,
            config.parameter("diversity_floor"),
            config.parameter("sensitivity"),
        )
        ratio = normalized_group_exposure(effective, reference, panel)
        response_floor = config.parameter("response_floor")
        alpha = config.parameter("alpha")
        deficit = np.power(np.maximum(ratio, 0.0) + response_floor, -alpha)
        deficit -= math.pow(1.0 + response_floor, -alpha)
        physical_replay = group_sum(literal_replay(total), panel)
        values = np.column_stack([deficit, physical_replay])
        names = (*_group_names("diversity_gated_deficit", panel), *_group_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family == "capacity_gated_precision":
        adaptation_rate = config.parameter("adaptation_rate")
        evidence_prior = config.parameter("evidence_prior")
        capacity_prior = config.parameter("capacity_prior")
        evidence = group_sum(total, panel)
        capacity = representation_capacity_state(panel, weights, adaptation_rate)
        proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
        _reference0, _reference1, reference_total = simulated_epochs(panel, proportional)
        reference_evidence = group_sum(reference_total, panel)
        reference_capacity = representation_capacity_state(panel, proportional, adaptation_rate)
        log_capability_debt = (
            np.log(evidence_prior + reference_evidence)
            + np.log(capacity_prior + reference_capacity)
            - np.log(evidence_prior + evidence)
            - np.log(capacity_prior + capacity)
        )
        physical_replay = group_sum(literal_replay(total), panel)
        values = np.column_stack([log_capability_debt, physical_replay])
        names = (*_group_names("capacity_gated_log_precision_debt", panel), *_group_names("literal_replay", panel))
        return Design(values=values, names=names)

    if config.family == "finite_subset_retained_debt":
        forgetting = config.parameter("forgetting")
        prior = config.parameter("prior")
        state = finite_subset_retained_state(panel, weights, forgetting)
        proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
        reference = finite_subset_retained_state(panel, proportional, forgetting)
        grouped_state = group_sum(state, panel, weighted=True)
        grouped_reference = group_sum(reference, panel, weighted=True)
        debt = np.log(prior + grouped_reference) - np.log(prior + grouped_state)
        replay = group_sum(finite_subset_replay(total), panel)
        values = np.column_stack([debt, replay])
        names = (*_group_names("finite_subset_log_debt", panel), *_group_names("finite_subset_replay", panel))
        return Design(values=values, names=names)

    if config.family == "power_law_memory_deficit":
        effective = power_law_memory_exposure(
            panel,
            weights,
            config.parameter("timescale"),
            config.parameter("memory_exponent"),
        )
        proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
        reference = power_law_memory_exposure(
            panel,
            proportional,
            config.parameter("timescale"),
            config.parameter("memory_exponent"),
        )
        ratio = normalized_group_exposure(effective, reference, panel)
        floor = config.parameter("floor")
        alpha = config.parameter("alpha")
        deficit = np.power(np.maximum(ratio, 0.0) + floor, -alpha)
        deficit -= math.pow(1.0 + floor, -alpha)
        _phase0, _phase1, total_exposure = simulated_epochs(panel, weights)
        replay = group_sum(finite_subset_replay(total_exposure), panel)
        values = np.column_stack([deficit, replay])
        names = (*_group_names("power_law_memory_deficit", panel), *_group_names("finite_subset_replay", panel))
        return Design(values=values, names=names)

    if config.family == "learned_state_competition":
        acquisition = config.parameter("acquisition")
        competition = config.parameter("competition")
        state = learned_state_competition(panel, weights, acquisition, competition)
        proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
        reference = learned_state_competition(panel, proportional, acquisition, competition)
        debt = reference - state
        physical_replay = group_sum(literal_replay(total), panel)
        values = np.column_stack([debt, physical_replay])
        names = (*_group_names("learned_state_debt", panel), *_group_names("literal_replay", panel))
        return Design(values=values, names=names)

    raise ValueError(f"Unknown mechanistic family {config.family!r}")


def fit_nonnegative_ridge(
    design: Design,
    observed: np.ndarray,
    indices: np.ndarray,
    config: ModelConfig,
    l2: float,
) -> FittedModel:
    x = np.asarray(design.values[indices], dtype=float)
    y = np.asarray(observed[indices], dtype=float)
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    centered_x = x - x_mean
    centered_y = y - y_mean
    if l2 > 0:
        augmented_x = np.vstack([centered_x, math.sqrt(l2) * np.eye(x.shape[1])])
        augmented_y = np.concatenate([centered_y, np.zeros(x.shape[1], dtype=float)])
    else:
        augmented_x = centered_x
        augmented_y = centered_y
    coefficients, _residual = nnls(augmented_x, augmented_y, maxiter=max(3 * x.shape[1], 1000))
    intercept = y_mean - float(x_mean @ coefficients)
    active = coefficients > max(1e-10, 1e-6 * float(np.max(coefficients, initial=0.0)))
    if active.any():
        active_x = centered_x[:, active]
        gram = active_x.T @ active_x
        hat_trace = float(np.trace(np.linalg.solve(gram + l2 * np.eye(gram.shape[0]), gram)))
    else:
        hat_trace = 0.0
    return FittedModel(
        config=config,
        l2=l2,
        intercept=intercept,
        coefficients=coefficients,
        feature_names=design.names,
        effective_degrees_of_freedom=1.0 + hat_trace,
    )


def candidate_configs() -> tuple[ModelConfig, ...]:
    configs: list[ModelConfig] = [ModelConfig("unique_replay"), ModelConfig("phase_unique_heads")]
    configs.extend(ModelConfig("recency_unique_replay", (("recency", value),)) for value in (0.5, 1.0, 2.0, 4.0, 8.0))
    configs.extend(
        ModelConfig("retained_state_ode", (("acquisition", acquisition), ("forgetting", forgetting)))
        for acquisition in (0.5, 1.0, 2.0)
        for forgetting in (0.25, 0.5, 1.0, 2.0, 4.0)
    )
    configs.extend(ModelConfig("family_survival", (("rate", value),)) for value in (0.5, 1.0, 2.0, 4.0))
    configs.extend(
        ModelConfig("weibull_shortage", (("scale", scale), ("power", power)))
        for scale in (0.5, 1.0, 2.0, 4.0)
        for power in (0.5, 1.0, 2.0)
    )
    configs.extend(
        ModelConfig("soft_bottleneck", (("alpha", alpha), ("temperature", temperature)))
        for alpha in (0.5, 1.0, 2.0)
        for temperature in (0.1, 0.25, 0.5, 1.0)
    )
    configs.extend(ModelConfig("competition_unique_replay", (("strength", value),)) for value in (0.5, 1.0, 2.0, 4.0))
    return tuple(configs)


def round2_candidate_configs() -> tuple[ModelConfig, ...]:
    configs: list[ModelConfig] = []
    common = [
        (("recency", recency), ("floor", floor), ("alpha", alpha))
        for recency in (0.0, 2.0, 4.0, 8.0)
        for floor in (0.03, 0.1)
        for alpha in (0.5, 1.0)
    ]
    configs.extend(ModelConfig("group_coverage_replay", parameters) for parameters in common)
    configs.extend(
        ModelConfig("hierarchical_coverage", (*parameters, ("bottleneck", bottleneck)))
        for parameters in common
        for bottleneck in (0.5, 1.0, 2.0)
    )
    configs.extend(
        ModelConfig("series_reliability", (*parameters, ("rate", rate)))
        for parameters in common
        for rate in (1.0, 2.0, 4.0)
    )
    configs.extend(
        ModelConfig(
            "overload_hazard_state",
            (("recency", recency), ("acquisition", acquisition), ("overload", overload)),
        )
        for recency in (0.0, 4.0, 8.0)
        for acquisition in (0.5, 1.0, 2.0)
        for overload in (0.1, 0.25, 0.5)
    )
    configs.extend(
        ModelConfig(
            "family_cross_forgetting",
            (("acquisition", acquisition), ("forgetting", forgetting)),
        )
        for acquisition in (0.5, 1.0, 2.0)
        for forgetting in (0.25, 0.5, 1.0, 2.0, 4.0)
    )
    return tuple(configs)


def round3_dynamics_candidate_configs() -> tuple[ModelConfig, ...]:
    """Prespecified sequential error-mass and competition falsification grid."""
    configs = [
        ModelConfig(
            "sequential_error_mass",
            (("acquisition", acquisition), ("forgetting", forgetting)),
        )
        for acquisition in (0.5, 1.0, 2.0)
        for forgetting in (0.0, 0.25, 0.5, 1.0, 2.0)
    ]
    configs.extend(
        ModelConfig(
            "competition_error_mass",
            (
                ("acquisition", acquisition),
                ("forgetting", forgetting),
                ("competition", competition),
            ),
        )
        for acquisition in (0.5, 1.0, 2.0)
        for forgetting in (0.25, 1.0, 2.0)
        for competition in (0.5, 1.0, 2.0, 4.0)
    )
    return tuple(configs)


def round4_foundation_candidate_configs() -> tuple[ModelConfig, ...]:
    """Directional broad-foundation transfer and its physical-exposure ablation."""

    configs = [
        ModelConfig("physical_scaling_deficit", (("floor", floor), ("alpha", alpha)))
        for floor in (0.03, 0.1, 0.3, 1.0)
        for alpha in (0.1, 0.25, 0.5)
    ]
    configs.extend(
        ModelConfig(
            "foundation_transfer_deficit",
            (
                ("acquisition", acquisition),
                ("boost", boost),
                ("floor", floor),
                ("alpha", alpha),
            ),
        )
        for acquisition in (1.0, 2.0, 4.0, 8.0)
        for boost in (0.5, 1.0, 2.0)
        for floor in (0.03, 0.1, 0.3, 1.0)
        for alpha in (0.1, 0.25, 0.5)
    )
    return tuple(configs)


def round5_prior_candidate_configs() -> tuple[ModelConfig, ...]:
    """Two-level equivalent-prior exposure with an optional recency state."""

    common = [
        (
            ("foundation_floor", foundation_floor),
            ("specialist_floor", specialist_floor),
            ("alpha", alpha),
        )
        for foundation_floor in (0.1, 0.3, 1.0)
        for specialist_floor in (0.03, 0.1, 0.3, 1.0)
        for alpha in (0.1, 0.25, 0.5)
    ]
    configs = [ModelConfig("two_level_prior_deficit", parameters) for parameters in common]
    configs.extend(
        ModelConfig("two_level_prior_recency_deficit", (*parameters, ("recency", recency)))
        for parameters in common
        for recency in (2.0, 8.0)
    )
    return tuple(configs)


def round8_bounded_coverage_candidate_configs() -> tuple[ModelConfig, ...]:
    """Bounded unique-coverage state with compatible-family forgetting."""

    return tuple(
        ModelConfig(
            "bounded_coverage_deficit",
            (("forgetting", forgetting), ("floor", floor), ("alpha", alpha)),
        )
        for forgetting in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
        for floor in (0.03, 0.1, 0.3, 1.0)
        for alpha in (0.1, 0.25, 0.5)
    )


def round9_ces_candidate_configs() -> tuple[ModelConfig, ...]:
    """Full family CES production laws over physical or retained evidence."""

    shapes = tuple(
        (
            ("substitution_order", substitution_order),
            ("floor", floor),
            ("alpha", alpha),
        )
        for substitution_order in (0.25, 1.0, 4.0, 16.0)
        for floor in (0.03, 0.1, 0.3, 1.0)
        for alpha in (0.1, 0.25, 0.5)
    )
    configs = [ModelConfig("physical_ces_production", parameters) for parameters in shapes]
    configs.extend(
        ModelConfig("retained_ces_production", (("forgetting", forgetting), *parameters))
        for forgetting in (0.5, 2.0, 4.0)
        for parameters in shapes
    )
    return tuple(configs)


def round10_replay_hazard_candidate_configs() -> tuple[ModelConfig, ...]:
    """Novel-sample acquisition with global duplicate-token forgetting."""

    return tuple(
        ModelConfig(
            "replay_hazard_deficit",
            (("hazard_rate", hazard_rate), ("floor", floor), ("alpha", alpha)),
        )
        for hazard_rate in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
        for floor in (0.03, 0.1, 0.3, 1.0)
        for alpha in (0.1, 0.25, 0.5)
    )


def round16_plasticity_candidate_configs() -> tuple[ModelConfig, ...]:
    """Learning-rate-weighted acquisition with no fitted phase multiplier."""

    return tuple(
        ModelConfig(
            "learning_rate_plasticity_deficit",
            (("power", power), ("floor", floor), ("alpha", alpha)),
        )
        for power in (0.0, 0.5, 1.0, 2.0)
        for floor in (0.03, 0.1, 0.3, 1.0)
        for alpha in (0.1, 0.25, 0.5)
    )


def round17_gradient_noise_candidate_configs() -> tuple[ModelConfig, ...]:
    """Phase-local effective update budget under mixture gradient variance."""

    return tuple(
        ModelConfig(
            "gradient_noise_limited_deficit",
            (("sensitivity", sensitivity), ("floor", floor), ("alpha", alpha)),
        )
        for sensitivity in (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
        for floor in (0.03, 0.1, 0.3, 1.0)
        for alpha in (0.1, 0.25, 0.5)
    )


def round18_parallel_reliability_candidate_configs() -> tuple[ModelConfig, ...]:
    """Two-level reliability network with parallel within-family support."""

    return tuple(
        ModelConfig(
            "parallel_reliability_network",
            (("prior", prior), ("recency", recency)),
        )
        for prior in (0.03, 0.1, 0.3, 1.0, 3.0)
        for recency in (0.0, 2.0, 8.0)
    )


def round19_posterior_precision_candidate_configs() -> tuple[ModelConfig, ...]:
    """Bayesian information accumulation with compatible-family retention."""

    return tuple(
        ModelConfig(
            "posterior_precision_debt",
            (("forgetting", forgetting), ("prior", prior)),
        )
        for forgetting in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
        for prior in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    )


def round20_capacity_gated_candidate_configs() -> tuple[ModelConfig, ...]:
    """Finite representation capacity gating independent evidence precision."""

    return tuple(
        ModelConfig(
            "capacity_gated_precision",
            (
                ("adaptation_rate", adaptation_rate),
                ("evidence_prior", evidence_prior),
                ("capacity_prior", capacity_prior),
            ),
        )
        for adaptation_rate in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
        for evidence_prior in (0.03, 0.1, 0.3, 1.0)
        for capacity_prior in (0.01, 0.03, 0.1, 0.3)
    )


def round21_finite_subset_candidate_configs() -> tuple[ModelConfig, ...]:
    """Exact simulated-subset occupancy with compatible-family retention."""

    return tuple(
        ModelConfig(
            "finite_subset_retained_debt",
            (("forgetting", forgetting), ("prior", prior)),
        )
        for forgetting in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
        for prior in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    )


def round23_power_law_memory_candidate_configs() -> tuple[ModelConfig, ...]:
    """Continuous acquisition under a power-law retention kernel."""

    return tuple(
        ModelConfig(
            "power_law_memory_deficit",
            (
                ("timescale", timescale),
                ("memory_exponent", memory_exponent),
                ("floor", floor),
                ("alpha", alpha),
            ),
        )
        for timescale in (0.03, 0.1, 0.3, 1.0)
        for memory_exponent in (0.5, 1.0, 2.0, 4.0)
        for floor in (0.03, 0.3, 1.0)
        for alpha in (0.1, 0.5)
    )


def round24_riccati_uncertainty_candidate_configs() -> tuple[ModelConfig, ...]:
    """Kalman--Bucy uncertainty with out-of-family process variance."""

    return tuple(
        ModelConfig(
            "riccati_uncertainty_debt",
            (("prior_variance", prior_variance), ("process_noise", process_noise)),
        )
        for prior_variance in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
        for process_noise in (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    )


def round25_two_pool_consolidation_candidate_configs() -> tuple[ModelConfig, ...]:
    """Fast acquisition, vulnerable working memory, and slow consolidation."""

    return tuple(
        ModelConfig(
            "two_pool_consolidation",
            (
                ("acquisition", acquisition),
                ("forgetting", forgetting),
                ("consolidation", consolidation),
                ("slow_weight", slow_weight),
            ),
        )
        for acquisition in (0.5, 2.0, 8.0)
        for forgetting in (0.5, 2.0, 8.0)
        for consolidation in (0.5, 2.0, 8.0)
        for slow_weight in (0.25, 0.5, 0.75)
    )


def round26_concentration_displacement_candidate_configs() -> tuple[ModelConfig, ...]:
    """Rank-one competition induced by excess family concentration."""

    return tuple(
        ModelConfig(
            "concentration_displacement",
            (("acquisition", acquisition), ("displacement", displacement)),
        )
        for acquisition in (0.5, 2.0, 8.0)
        for displacement in (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    )


def round27_diversity_gated_candidate_configs() -> tuple[ModelConfig, ...]:
    """Global acquisition bottleneck from missing foundation-family support."""

    return tuple(
        ModelConfig(
            "diversity_gated_deficit",
            (
                ("diversity_floor", diversity_floor),
                ("sensitivity", sensitivity),
                ("response_floor", response_floor),
                ("alpha", alpha),
            ),
        )
        for diversity_floor in (0.03, 0.1, 0.3, 1.0)
        for sensitivity in (0.0, 0.25, 0.5, 1.0, 2.0)
        for response_floor in (0.03, 0.3, 1.0)
        for alpha in (0.1, 0.5)
    )


def round28_learned_state_competition_candidate_configs() -> tuple[ModelConfig, ...]:
    """Bounded competence with competition mediated by learned family state."""

    return tuple(
        ModelConfig(
            "learned_state_competition",
            (("acquisition", acquisition), ("competition", competition)),
        )
        for acquisition in (0.5, 1.0, 2.0, 4.0, 8.0)
        for competition in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    )


def record(model: FittedModel) -> dict[str, Any]:
    active = model.coefficients > max(1e-10, 1e-6 * float(np.max(model.coefficients, initial=0.0)))
    return {
        "config": model.config.key,
        "family": model.config.family,
        "parameters": dict(model.config.parameters),
        "l2": model.l2,
        "intercept": model.intercept,
        "parameter_count": 1 + len(model.coefficients),
        "active_parameter_count": 1 + int(active.sum()),
        "effective_degrees_of_freedom": model.effective_degrees_of_freedom,
    }
