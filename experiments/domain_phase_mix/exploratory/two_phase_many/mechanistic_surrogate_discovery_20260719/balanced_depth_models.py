# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Balanced deep-linear trunk and task-head learning dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INTEGRATION_STEPS_PER_UNIT = 256
INITIAL_TRUNK_FACTOR = 0.9
INITIAL_HEAD = 0.1


@dataclass(frozen=True)
class BalancedDepthConfig:
    """Architecture-fixed depth and dimensionless optimizer clocks."""

    depth: int
    trunk_relaxation: float
    head_rate_ratio: float
    evaluation_rare_weight: float
    transition: str

    @property
    def key(self) -> str:
        return (
            f"depth={self.depth},trunk={self.trunk_relaxation:g},"
            f"head_ratio={self.head_rate_ratio:g},eval={self.evaluation_rare_weight:g},"
            f"transition={self.transition}"
        )


def initial_state(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.full(n, INITIAL_TRUNK_FACTOR), np.full((n, 2), INITIAL_HEAD)


def derivative(
    trunk: np.ndarray,
    heads: np.ndarray,
    rare_weight: np.ndarray,
    config: BalancedDepthConfig,
) -> tuple[np.ndarray, np.ndarray]:
    product = trunk**config.depth
    errors = product[:, None] * heads - 1.0
    mixture = np.column_stack([1.0 - rare_weight, rare_weight])
    if config.transition == "frozen_trunk":
        trunk_derivative = np.zeros_like(trunk)
    elif config.transition in {"declared_depth", "depth_one"}:
        trunk_gradient = config.depth * trunk ** (config.depth - 1) * np.sum(mixture * errors * heads, axis=1)
        trunk_derivative = -config.trunk_relaxation * trunk_gradient
    else:
        raise ValueError(f"Unknown transition {config.transition}")
    head_derivative = -config.trunk_relaxation * config.head_rate_ratio * mixture * errors * product[:, None]
    return trunk_derivative, head_derivative


def phase_update(
    trunk: np.ndarray,
    heads: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: BalancedDepthConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> tuple[np.ndarray, np.ndarray]:
    # Deep products become stiff at high physical clocks; scaling the numerical
    # resolution with that clock preserves the same ODE and frozen grid.
    clock_resolution = max(1.0, config.trunk_relaxation / 4.0)
    steps = max(1, int(np.ceil(duration * steps_per_unit * clock_resolution)))
    step_size = duration / steps
    trunk = np.asarray(trunk, dtype=float).copy()
    heads = np.asarray(heads, dtype=float).copy()
    rare_weight = np.asarray(rare_weight, dtype=float)
    for _ in range(steps):
        a1, h1 = derivative(trunk, heads, rare_weight, config)
        a2, h2 = derivative(trunk + 0.5 * step_size * a1, heads + 0.5 * step_size * h1, rare_weight, config)
        a3, h3 = derivative(trunk + 0.5 * step_size * a2, heads + 0.5 * step_size * h2, rare_weight, config)
        a4, h4 = derivative(trunk + step_size * a3, heads + step_size * h3, rare_weight, config)
        trunk += step_size * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0
        heads += step_size * (h1 + 2.0 * h2 + 2.0 * h3 + h4) / 6.0
    if not np.isfinite(trunk).all() or not np.isfinite(heads).all():
        raise FloatingPointError(f"Non-finite balanced-depth state for {config.key}")
    if np.any(trunk <= 0.0) or np.any(heads <= 0.0):
        raise FloatingPointError(f"Balanced-depth state crossed the positive invariant for {config.key}")
    return trunk, heads


def terminal_state(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: BalancedDepthConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1:] != (2, 2):
        raise ValueError(f"Expected [policy, phase, domain] weights, got {weights.shape}")
    trunk, heads = initial_state(len(weights))
    trunk, heads = phase_update(
        trunk,
        heads,
        weights[:, 0, 1],
        phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )
    return phase_update(
        trunk,
        heads,
        weights[:, 1, 1],
        1.0 - phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )


def response_feature(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: BalancedDepthConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    trunk, heads = terminal_state(weights, phase0_optimizer_fraction, config, steps_per_unit=steps_per_unit)
    errors = trunk[:, None] ** config.depth * heads - 1.0
    q = config.evaluation_rare_weight
    return 0.5 * ((1.0 - q) * errors[:, 0] ** 2 + q * errors[:, 1] ** 2)


def integration_error(weights: np.ndarray, phase0_optimizer_fraction: float, config: BalancedDepthConfig) -> float:
    coarse = response_feature(weights, phase0_optimizer_fraction, config, steps_per_unit=256)
    fine = response_feature(weights, phase0_optimizer_fraction, config, steps_per_unit=768)
    return float(np.max(np.abs(coarse - fine)))
