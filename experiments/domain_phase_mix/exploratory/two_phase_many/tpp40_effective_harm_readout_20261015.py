# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "threadpoolctl"]
# ///
"""Matched late-refresh readout control; no fitting or data access.

TP40-028 supplies the unchanged effective hazard. Here both native benefit and
harm read its corresponding effective exposure. This tests a shared readout
coordinate, not a causal damage, retention or forgetting mechanism. At lambda=1,
an all-early bucket contributes zero benefit and h(0) harm regardless of its
physical exposure; that invisibility is a known property, not a rejected input.
"""

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_late_refresh_20261015 as refresh

mariner = refresh.mariner
MODEL_NAMES = ("aggregate_mariner", "late_refresh_effective_harm")
STRENGTHS = refresh.STRENGTHS


def parameter_grid(model: str) -> tuple[dict[str, float], ...]:
    """Reuse the refresh strengths with no additional damage parameter."""
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown effective-harm readout: {model}")
    return refresh.parameter_grid("aggregate_mariner" if model == "aggregate_mariner" else "late_refresh_hazard")


def effective_exposure(
    early: np.ndarray,
    late: np.ndarray,
    alpha: float,
    rate: float,
    power: float,
    strength: float,
) -> np.ndarray:
    """Convert the unchanged refresh hazard to its effective exposure.

    The relative-hazard form preserves exact physical exposures at true ties
    and avoids the unnecessary inverse-power round trip through the rate.
    Zero total exposure maps to zero. Finite representability of the effective
    exposure is required; no cap or clipping hides an overflow.
    """
    hazard = refresh.effective_hazard(early, late, alpha, rate, power, strength)
    total = np.asarray(early, dtype=float) + np.asarray(late, dtype=float)
    if strength == 0:
        return total
    native_hazard = np.maximum(rate * total, 0.0) ** power
    ratio = np.divide(hazard, native_hazard, out=np.zeros_like(total), where=native_hazard > 0)
    with np.errstate(over="raise", invalid="raise"):
        return total * ratio ** (1 / power)


def design_and_offset(
    early: np.ndarray,
    late: np.ndarray,
    alpha: float,
    shape: dict[str, float],
    model: str,
    parameters: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Read both native response channels from the same effective exposure."""
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown effective-harm readout: {model}")
    if model == "aggregate_mariner" or parameters["lambda"] == 0:
        return mariner.design_matrix(early + late, shape), np.zeros(len(early))
    exposure = effective_exposure(early, late, alpha, shape["rate"], shape["power"], parameters["lambda"])
    return mariner.design_matrix(exposure, shape), np.zeros(len(early))
