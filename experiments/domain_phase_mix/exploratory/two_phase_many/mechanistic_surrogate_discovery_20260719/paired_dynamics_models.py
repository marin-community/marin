# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mechanistic aggregate/phase decompositions identified from matched policies."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

NUMERICAL_FLOOR = 1e-12


class PhaseFamily(StrEnum):
    PMVT = "paired_marginal_value_transport"
    COMMUTATOR = "family_commutator_flow"
    FAST_SLOW = "identified_fast_slow_consolidation"
    QUASI_STEADY = "quasi_steady_consolidation"
    TERMINAL_EQUILIBRIUM = "terminal_equilibrium_adaptation"


@dataclass(frozen=True)
class PairedPanel:
    """A two-phase panel with phase-tied outcomes at matched aggregate policies."""

    name: str
    target: str
    frame: pd.DataFrame
    domain_names: tuple[str, ...]
    family_names: tuple[str, ...]
    family_members: tuple[np.ndarray, ...]
    weights: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    two_phase_target: np.ndarray
    one_phase_target: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.two_phase_target)
        if self.weights.shape != (n, 2, len(self.domain_names)):
            raise ValueError(f"Unexpected weight shape for {self.name}: {self.weights.shape}")
        if self.one_phase_target.shape != (n,):
            raise ValueError(f"Unexpected paired-target shape for {self.name}: {self.one_phase_target.shape}")
        if not np.allclose(self.weights.sum(axis=2), 1.0, atol=1e-9):
            raise ValueError(f"Non-normalized policy in {self.name}")
        if not np.isfinite(self.two_phase_target).all():
            raise ValueError(f"Non-finite two-phase target in {self.name}")
        covered = np.concatenate(self.family_members)
        if sorted(covered.tolist()) != list(range(len(self.domain_names))):
            raise ValueError(f"Families do not partition the domains in {self.name}")

    @property
    def n(self) -> int:
        return len(self.two_phase_target)

    @property
    def m(self) -> int:
        return len(self.domain_names)

    @property
    def paired_mask(self) -> np.ndarray:
        return np.isfinite(self.one_phase_target)

    @property
    def alpha0(self) -> float:
        fractions = self.c0 / np.maximum(self.c0 + self.c1, NUMERICAL_FLOOR)
        if np.ptp(fractions) > 1e-10:
            raise ValueError(f"Domain-dependent phase fraction in {self.name}")
        return float(np.mean(fractions))

    @property
    def alpha1(self) -> float:
        return 1.0 - self.alpha0

    @property
    def aggregate_weights(self) -> np.ndarray:
        return self.alpha0 * self.weights[:, 0, :] + self.alpha1 * self.weights[:, 1, :]

    @property
    def proportional_weights(self) -> np.ndarray:
        token_proxy = 1.0 / np.maximum(self.c0 + self.c1, NUMERICAL_FLOOR)
        return token_proxy / token_proxy.sum()

    @property
    def aggregate_exposure(self) -> np.ndarray:
        return self.weights[:, 0, :] * self.c0[None, :] + self.weights[:, 1, :] * self.c1[None, :]

    @property
    def proportional_exposure(self) -> np.ndarray:
        return self.proportional_weights * (self.c0 + self.c1)


@dataclass(frozen=True)
class AggregateConfig:
    shortage_power: float
    shortage_offset: float
    l2: float

    @property
    def key(self) -> str:
        return f"p={self.shortage_power:g},offset={self.shortage_offset:g},l2={self.l2:g}"


@dataclass(frozen=True)
class PMVTConfig:
    remaining_offset: float
    l2: float
    include_signed_transport: bool = True
    include_quadratic_mismatch: bool = True
    transport_level: str = "family"
    mismatch_level: str = "family"

    @property
    def key(self) -> str:
        return (
            f"remaining={self.remaining_offset:g},l2={self.l2:g},"
            f"signed={int(self.include_signed_transport)},quadratic={int(self.include_quadratic_mismatch)},"
            f"transport={self.transport_level},mismatch={self.mismatch_level}"
        )


@dataclass(frozen=True)
class CommutatorConfig:
    remaining_offset: float
    l2: float

    @property
    def key(self) -> str:
        return f"remaining={self.remaining_offset:g},l2={self.l2:g}"


@dataclass(frozen=True)
class FastSlowConfig:
    learn_rate: float
    forget_rate: float
    consolidate_rate: float
    slow_weight: float
    l2: float
    state_level: str = "family"

    @property
    def key(self) -> str:
        return (
            f"learn={self.learn_rate:g},forget={self.forget_rate:g},"
            f"consolidate={self.consolidate_rate:g},slow={self.slow_weight:g},l2={self.l2:g},"
            f"state={self.state_level}"
        )


@dataclass(frozen=True)
class QuasiSteadyConfig:
    saturation_ratio: float
    consolidate_rate: float
    slow_weight: float
    l2: float

    @property
    def key(self) -> str:
        return (
            f"ratio={self.saturation_ratio:g},consolidate={self.consolidate_rate:g},"
            f"slow={self.slow_weight:g},l2={self.l2:g}"
        )


@dataclass(frozen=True)
class TerminalEquilibriumConfig:
    saturation_ratio: float
    l2: float

    @property
    def key(self) -> str:
        return f"ratio={self.saturation_ratio:g},l2={self.l2:g}"


PhaseConfig = PMVTConfig | CommutatorConfig | FastSlowConfig | QuasiSteadyConfig | TerminalEquilibriumConfig


@dataclass(frozen=True)
class LinearHead:
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    intercept: float
    coefficients: np.ndarray

    def predict(self, design: np.ndarray) -> np.ndarray:
        standardized = (design - self.feature_mean[None, :]) / self.feature_scale[None, :]
        return np.asarray(self.intercept + standardized @ self.coefficients, dtype=float)

    @property
    def coefficients_in_natural_units(self) -> np.ndarray:
        return self.coefficients / self.feature_scale


@dataclass(frozen=True)
class AggregateModel:
    panel: PairedPanel
    config: AggregateConfig
    head: LinearHead

    def predict_weights(self, weights: np.ndarray) -> np.ndarray:
        design, _names = aggregate_design(self.panel, weights, self.config)
        return self.head.predict(design)


@dataclass(frozen=True)
class PhaseModel:
    panel: PairedPanel
    family: PhaseFamily
    config: PhaseConfig
    head: LinearHead

    def predict_delta(self, weights: np.ndarray) -> np.ndarray:
        design, _names, _signs = phase_design(self.panel, weights, self.family, self.config)
        return self.head.predict(design)


@dataclass(frozen=True)
class JointModel:
    """The same aggregate/phase equation fitted without matched-policy orthogonalization."""

    panel: PairedPanel
    aggregate_config: AggregateConfig
    family: PhaseFamily
    phase_config: PhaseConfig
    aggregate_width: int
    head: LinearHead

    def predict_weights(self, weights: np.ndarray) -> np.ndarray:
        aggregate, _aggregate_names = aggregate_design(
            self.panel,
            tied_weights(self.panel, weights),
            self.aggregate_config,
        )
        phase, _phase_names, _phase_signs = phase_design(
            self.panel,
            weights,
            self.family,
            self.phase_config,
        )
        return self.head.predict(np.column_stack([aggregate, phase]))


def fit_linear_head(
    design: np.ndarray,
    target: np.ndarray,
    feature_names: Iterable[str],
    coefficient_signs: np.ndarray,
    l2: float,
) -> LinearHead:
    """Fit a standardized ridge head with explicit coefficient signs."""

    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    signs = np.asarray(coefficient_signs, dtype=int)
    if design.ndim != 2 or len(target) != len(design):
        raise ValueError("Design and target dimensions do not agree")
    if design.shape[1] != len(signs):
        raise ValueError("Coefficient signs do not match design width")
    feature_mean = design.mean(axis=0)
    centered = design - feature_mean[None, :]
    feature_scale = np.sqrt(np.mean(centered**2, axis=0))
    feature_scale = np.maximum(feature_scale, 1e-8)
    standardized = centered / feature_scale[None, :]
    target_mean = float(np.mean(target))
    centered_target = target - target_mean
    if l2 > 0.0:
        standardized = np.vstack([standardized, np.sqrt(l2) * np.eye(design.shape[1])])
        centered_target = np.concatenate([centered_target, np.zeros(design.shape[1])])
    lower = np.where(signs > 0, 0.0, -np.inf)
    upper = np.where(signs < 0, 0.0, np.inf)
    result = lsq_linear(
        standardized,
        centered_target,
        bounds=(lower, upper),
        method="trf",
        lsmr_tol="auto",
        max_iter=1000,
    )
    if not result.success:
        raise RuntimeError(f"Constrained ridge fit failed: {result.message}")
    return LinearHead(
        feature_names=tuple(feature_names),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        intercept=target_mean,
        coefficients=np.asarray(result.x, dtype=float),
    )


def fit_zero_intercept_head(
    design: np.ndarray,
    target: np.ndarray,
    feature_names: Iterable[str],
    coefficient_signs: np.ndarray,
    l2: float,
) -> LinearHead:
    """Fit a ridge head constrained to predict zero for a tied phase policy."""

    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    signs = np.asarray(coefficient_signs, dtype=int)
    if design.ndim != 2 or design.shape[1] != len(signs) or len(target) != len(design):
        raise ValueError("Zero-intercept design, signs, and target do not agree")
    feature_scale = np.sqrt(np.mean(design**2, axis=0))
    feature_scale = np.maximum(feature_scale, 1e-8)
    standardized = design / feature_scale[None, :]
    fitted_target = target
    if l2 > 0.0:
        standardized = np.vstack([standardized, np.sqrt(l2) * np.eye(design.shape[1])])
        fitted_target = np.concatenate([target, np.zeros(design.shape[1])])
    lower = np.where(signs > 0, 0.0, -np.inf)
    upper = np.where(signs < 0, 0.0, np.inf)
    result = lsq_linear(
        standardized,
        fitted_target,
        bounds=(lower, upper),
        method="trf",
        lsmr_tol="auto",
        max_iter=1000,
    )
    if not result.success:
        raise RuntimeError(f"Zero-intercept constrained ridge fit failed: {result.message}")
    return LinearHead(
        feature_names=tuple(feature_names),
        feature_mean=np.zeros(design.shape[1]),
        feature_scale=feature_scale,
        intercept=0.0,
        coefficients=np.asarray(result.x, dtype=float),
    )


def tied_weights(panel: PairedPanel, weights: np.ndarray) -> np.ndarray:
    aggregate = panel.alpha0 * weights[:, 0, :] + panel.alpha1 * weights[:, 1, :]
    return np.stack([aggregate, aggregate], axis=1)


def aggregate_design(
    panel: PairedPanel,
    weights: np.ndarray,
    config: AggregateConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build deficit and literal replay-mass features from total exposure."""

    weights = np.asarray(weights, dtype=float)
    exposure = weights[:, 0, :] * panel.c0[None, :] + weights[:, 1, :] * panel.c1[None, :]
    reference = panel.proportional_exposure
    relative = exposure / np.maximum(reference[None, :], NUMERICAL_FLOOR)
    deficit = (relative + config.shortage_offset) ** (-config.shortage_power)
    deficit -= (1.0 + config.shortage_offset) ** (-config.shortage_power)
    duplicate_mass = exposure - (1.0 - np.exp(-exposure))
    reference_duplicate = reference - (1.0 - np.exp(-reference))
    family_replay: list[np.ndarray] = []
    for members in panel.family_members:
        family_mass = panel.proportional_weights[members]
        family_mass /= family_mass.sum()
        centered = duplicate_mass[:, members] - reference_duplicate[None, members]
        family_replay.append(centered @ family_mass)
    names = tuple(f"deficit::{domain}" for domain in panel.domain_names) + tuple(
        f"literal_replay::{family}" for family in panel.family_names
    )
    return np.column_stack([deficit, *family_replay]), names


def fit_aggregate(
    panel: PairedPanel,
    indices: np.ndarray,
    config: AggregateConfig,
) -> AggregateModel:
    paired = panel.paired_mask[indices]
    source_indices = np.asarray(indices, dtype=int)[paired]
    if len(source_indices) < 3:
        raise ValueError(f"Too few paired rows to fit aggregate response in {panel.name}")
    weights = tied_weights(panel, panel.weights[source_indices])
    design, names = aggregate_design(panel, weights, config)
    head = fit_linear_head(
        design,
        panel.one_phase_target[source_indices],
        names,
        coefficient_signs=np.ones(len(names), dtype=int),
        l2=config.l2,
    )
    return AggregateModel(panel=panel, config=config, head=head)


def relative_phase_contrast(panel: PairedPanel, weights: np.ndarray) -> np.ndarray:
    proportional = panel.proportional_weights
    return (
        panel.alpha0
        * panel.alpha1
        * (weights[:, 1, :] - weights[:, 0, :])
        / np.maximum(proportional[None, :], NUMERICAL_FLOOR)
    )


def relative_aggregate_exposure(panel: PairedPanel, weights: np.ndarray) -> np.ndarray:
    exposure = weights[:, 0, :] * panel.c0[None, :] + weights[:, 1, :] * panel.c1[None, :]
    return exposure / np.maximum(panel.proportional_exposure[None, :], NUMERICAL_FLOOR)


def pmvt_design(
    panel: PairedPanel,
    weights: np.ndarray,
    config: PMVTConfig,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    contrast = relative_phase_contrast(panel, weights)
    remaining = 1.0 / (config.remaining_offset + relative_aggregate_exposure(panel, weights))
    transported = remaining * contrast
    columns: list[np.ndarray] = []
    names: list[str] = []
    signs: list[int] = []
    if config.include_signed_transport and config.transport_level == "bucket":
        columns.extend(transported[:, index] for index in range(panel.m))
        names.extend(f"signed_transport::{domain}" for domain in panel.domain_names)
        signs.extend([0] * panel.m)
    elif config.include_signed_transport and config.transport_level != "family":
        raise ValueError(f"Unknown PMVT transport level {config.transport_level}")
    if config.include_quadratic_mismatch and config.mismatch_level == "bucket":
        columns.extend(transported[:, index] ** 2 for index in range(panel.m))
        names.extend(f"quadratic_mismatch::{domain}" for domain in panel.domain_names)
        signs.extend([1] * panel.m)
    elif config.include_quadratic_mismatch and config.mismatch_level != "family":
        raise ValueError(f"Unknown PMVT mismatch level {config.mismatch_level}")
    for family, members in zip(panel.family_names, panel.family_members, strict=True):
        family_mass = panel.proportional_weights[members]
        family_mass /= family_mass.sum()
        if config.include_signed_transport and config.transport_level == "family":
            columns.append(transported[:, members] @ family_mass)
            names.append(f"signed_transport::{family}")
            signs.append(0)
        if config.include_quadratic_mismatch and config.mismatch_level == "family":
            columns.append((transported[:, members] ** 2) @ family_mass)
            names.append(f"quadratic_mismatch::{family}")
            signs.append(1)
    if not columns:
        raise ValueError("PMVT must retain at least one mechanism")
    return np.column_stack(columns), tuple(names), np.asarray(signs, dtype=int)


def family_weight_mass(panel: PairedPanel, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase0 = np.column_stack([weights[:, 0, members].sum(axis=1) for members in panel.family_members])
    phase1 = np.column_stack([weights[:, 1, members].sum(axis=1) for members in panel.family_members])
    aggregate = panel.alpha0 * phase0 + panel.alpha1 * phase1
    return phase0, phase1, aggregate


def commutator_design(
    panel: PairedPanel,
    weights: np.ndarray,
    config: CommutatorConfig,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    phase0, phase1, aggregate = family_weight_mass(panel, weights)
    reference = np.asarray([panel.proportional_weights[members].sum() for members in panel.family_members])
    relative = aggregate / np.maximum(reference[None, :], NUMERICAL_FLOOR)
    columns: list[np.ndarray] = []
    names: list[str] = []
    for left, right in combinations(range(len(panel.family_names)), 2):
        bracket = phase0[:, left] * phase1[:, right] - phase0[:, right] * phase1[:, left]
        remaining = 2.0 / (2.0 * config.remaining_offset + relative[:, left] + relative[:, right])
        columns.append(panel.alpha0 * panel.alpha1 * bracket * remaining)
        names.append(f"commutator::{panel.family_names[left]}->{panel.family_names[right]}")
    return np.column_stack(columns), tuple(names), np.zeros(len(columns), dtype=int)


def update_fast_slow(
    fast: np.ndarray,
    slow: np.ndarray,
    input_mass: np.ndarray,
    duration: float,
    config: FastSlowConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact constant-input update for coupled fast and slow learning states."""

    rate = config.learn_rate * input_mass + config.forget_rate * (1.0 - input_mass)
    rate = np.maximum(rate, NUMERICAL_FLOOR)
    equilibrium = config.learn_rate * input_mass / rate
    fast_offset = fast - equilibrium
    exp_fast = np.exp(-rate * duration)
    exp_slow = np.exp(-config.consolidate_rate * duration)
    next_fast = equilibrium + fast_offset * exp_fast
    separation = config.consolidate_rate - rate
    near_equal = np.abs(separation) < 1e-8
    driven = np.empty_like(rate)
    driven[near_equal] = config.consolidate_rate * fast_offset[near_equal] * duration * exp_slow
    driven[~near_equal] = (
        config.consolidate_rate * fast_offset[~near_equal] * (exp_fast[~near_equal] - exp_slow) / separation[~near_equal]
    )
    next_slow = slow * exp_slow + equilibrium * (1.0 - exp_slow) + driven
    return next_fast, next_slow


def fast_slow_terminal(
    panel: PairedPanel,
    weights: np.ndarray,
    config: FastSlowConfig,
) -> np.ndarray:
    if config.state_level == "family":
        phase0, phase1, _aggregate = family_weight_mass(panel, weights)
    elif config.state_level == "bucket":
        phase0, phase1 = weights[:, 0, :], weights[:, 1, :]
    else:
        raise ValueError(f"Unknown fast-slow state level {config.state_level}")
    fast = np.zeros_like(phase0)
    slow = np.zeros_like(phase0)
    for input_mass, duration in ((phase0, panel.alpha0), (phase1, panel.alpha1)):
        for row in range(len(weights)):
            fast[row], slow[row] = update_fast_slow(
                fast[row],
                slow[row],
                input_mass[row],
                duration,
                config,
            )
    return (1.0 - config.slow_weight) * fast + config.slow_weight * slow


def fast_slow_design(
    panel: PairedPanel,
    weights: np.ndarray,
    config: FastSlowConfig,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    terminal = fast_slow_terminal(panel, weights, config)
    tied_terminal = fast_slow_terminal(panel, tied_weights(panel, weights), config)
    # Higher retained capability should reduce loss, hence the leading minus sign.
    design = -(terminal - tied_terminal)
    state_names = panel.family_names if config.state_level == "family" else panel.domain_names
    names = tuple(f"consolidated_capability::{name}" for name in state_names)
    return design, names, np.ones(len(names), dtype=int)


def quasi_steady_terminal(
    panel: PairedPanel,
    weights: np.ndarray,
    config: QuasiSteadyConfig,
) -> np.ndarray:
    phase0 = weights[:, 0, :]
    phase1 = weights[:, 1, :]

    def equilibrium(input_weight: np.ndarray) -> np.ndarray:
        denominator = input_weight + config.saturation_ratio * (1.0 - input_weight)
        return input_weight / np.maximum(denominator, NUMERICAL_FLOOR)

    fast0 = equilibrium(phase0)
    fast1 = equilibrium(phase1)
    retain0 = np.exp(-config.consolidate_rate * panel.alpha0)
    retain1 = np.exp(-config.consolidate_rate * panel.alpha1)
    slow0 = (1.0 - retain0) * fast0
    slow1 = retain1 * slow0 + (1.0 - retain1) * fast1
    return (1.0 - config.slow_weight) * fast1 + config.slow_weight * slow1


def quasi_steady_design(
    panel: PairedPanel,
    weights: np.ndarray,
    config: QuasiSteadyConfig,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    terminal = quasi_steady_terminal(panel, weights, config)
    tied_terminal = quasi_steady_terminal(panel, tied_weights(panel, weights), config)
    design = -(terminal - tied_terminal)
    names = tuple(f"quasi_steady_capability::{domain}" for domain in panel.domain_names)
    return design, names, np.ones(len(names), dtype=int)


def terminal_equilibrium_design(
    panel: PairedPanel,
    weights: np.ndarray,
    config: TerminalEquilibriumConfig,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    tied = tied_weights(panel, weights)

    def equilibrium(input_weight: np.ndarray) -> np.ndarray:
        denominator = input_weight + config.saturation_ratio * (1.0 - input_weight)
        return input_weight / np.maximum(denominator, NUMERICAL_FLOOR)

    terminal = equilibrium(weights[:, 1, :])
    tied_terminal = equilibrium(tied[:, 1, :])
    design = -(terminal - tied_terminal)
    names = tuple(f"terminal_equilibrium::{domain}" for domain in panel.domain_names)
    return design, names, np.ones(len(names), dtype=int)


def phase_design(
    panel: PairedPanel,
    weights: np.ndarray,
    family: PhaseFamily,
    config: PhaseConfig,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    if family is PhaseFamily.PMVT and isinstance(config, PMVTConfig):
        return pmvt_design(panel, weights, config)
    if family is PhaseFamily.COMMUTATOR and isinstance(config, CommutatorConfig):
        return commutator_design(panel, weights, config)
    if family is PhaseFamily.FAST_SLOW and isinstance(config, FastSlowConfig):
        return fast_slow_design(panel, weights, config)
    if family is PhaseFamily.QUASI_STEADY and isinstance(config, QuasiSteadyConfig):
        return quasi_steady_design(panel, weights, config)
    if family is PhaseFamily.TERMINAL_EQUILIBRIUM and isinstance(config, TerminalEquilibriumConfig):
        return terminal_equilibrium_design(panel, weights, config)
    raise TypeError(f"Configuration {type(config).__name__} does not match {family}")


def fit_phase(
    panel: PairedPanel,
    indices: np.ndarray,
    family: PhaseFamily,
    config: PhaseConfig,
) -> PhaseModel:
    source_indices = np.asarray(indices, dtype=int)
    source_indices = source_indices[panel.paired_mask[source_indices]]
    if len(source_indices) < 3:
        raise ValueError(f"Too few paired rows to fit phase response in {panel.name}")
    design, names, signs = phase_design(panel, panel.weights[source_indices], family, config)
    delta = panel.two_phase_target[source_indices] - panel.one_phase_target[source_indices]
    # A tied policy has exactly zero phase correction, so no free intercept is admissible.
    head = fit_zero_intercept_head(
        design,
        delta,
        names,
        coefficient_signs=signs,
        l2=config.l2,
    )
    return PhaseModel(panel=panel, family=family, config=config, head=head)


def predict_combined(
    aggregate_model: AggregateModel,
    phase_model: PhaseModel,
    weights: np.ndarray,
) -> np.ndarray:
    return aggregate_model.predict_weights(tied_weights(aggregate_model.panel, weights)) + phase_model.predict_delta(
        weights
    )


def fit_joint(
    panel: PairedPanel,
    indices: np.ndarray,
    aggregate_config: AggregateConfig,
    family: PhaseFamily,
    phase_config: PhaseConfig,
) -> JointModel:
    """Fit the candidate equation jointly when matched phase-tied outcomes are unavailable."""

    indices = np.asarray(indices, dtype=int)
    aggregate, aggregate_names = aggregate_design(
        panel,
        tied_weights(panel, panel.weights[indices]),
        aggregate_config,
    )
    phase, phase_names, phase_signs = phase_design(panel, panel.weights[indices], family, phase_config)
    design = np.column_stack([aggregate, phase])
    signs = np.concatenate([np.ones(len(aggregate_names), dtype=int), phase_signs])
    # The aggregate ridge is already selected on phase-tied data where possible;
    # joint-only panels use the phase ridge as a common standardized penalty.
    head = fit_linear_head(
        design,
        panel.two_phase_target[indices],
        (*aggregate_names, *phase_names),
        signs,
        l2=phase_config.l2,
    )
    return JointModel(
        panel=panel,
        aggregate_config=aggregate_config,
        family=family,
        phase_config=phase_config,
        aggregate_width=len(aggregate_names),
        head=head,
    )
