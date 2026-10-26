# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "threadpoolctl"]
# ///
"""Pure late-refresh hazard hypothesis; no fitting or data access.

One taskwise strength interpolates native acquisition hazard with a normalized
late-benefit multiplier. Harm keeps physical cumulative exposure. The terminal
normalization depends on final exposure and is not a causal state equation.
"""

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_temporal_models_20261014 as temporal

mariner = temporal.mariner
MODEL_NAMES = ("aggregate_mariner", "late_refresh_hazard")
STRENGTHS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def parameter_grid(model: str) -> tuple[dict[str, float], ...]:
    """Return the proposed finite oracle grid, with its exact native null first."""
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown late-refresh family: {model}")
    return ({},) if model == "aggregate_mariner" else tuple({"lambda": value} for value in STRENGTHS)


def effective_hazard(
    early: np.ndarray,
    late: np.ndarray,
    alpha: float,
    rate: float,
    power: float,
    strength: float,
) -> np.ndarray:
    """Return H*((1-lambda)+lambda*b(E1)/b((1-alpha)*E)).

    Zero total exposure maps to zero hazard. A relative exposure tolerance
    recognizes tied rows without classifying every sufficiently small dose as
    tied. Evaluating the convex combination directly retains tiny positive
    refresh when lambda=1; subtracting one from the ratio would lose it.
    """
    early, late = np.asarray(early, dtype=float), np.asarray(late, dtype=float)
    if early.shape != late.shape or early.ndim != 2:
        raise ValueError("Phase exposures must be matching row-by-bucket matrices")
    if not np.isfinite([alpha, rate, power, strength]).all():
        raise ValueError("Law parameters must be finite")
    if not 0 < alpha < 1 or rate <= 0 or not 0 < power <= 1 or not 0 <= strength <= 1:
        raise ValueError("Require alpha in (0,1), rate>0, power in (0,1], and lambda in [0,1]")
    total = early + late
    if not np.isfinite(total).all() or np.any(early < 0) or np.any(late < 0):
        raise ValueError("Physical phase exposures must be finite and nonnegative")
    hazard = np.maximum(rate * total, 0.0) ** power
    if not np.isfinite(hazard).all():
        raise FloatingPointError("Native hazard overflowed")
    if strength == 0:
        return hazard
    beta = 1 - alpha
    numerator = mariner.benefit(late, rate, power)
    denominator = mariner.benefit(beta * total, rate, power)
    ratio = np.divide(numerator, denominator, out=np.zeros_like(total), where=denominator > 0)
    result = hazard * ((1 - strength) + strength * ratio)
    early_fraction = np.divide(early, total, out=np.full_like(total, alpha), where=total > 0)
    late_fraction = np.divide(late, total, out=np.full_like(total, beta), where=total > 0)
    tolerance = temporal.TIED_ROUNDOFF_MULTIPLIER * np.finfo(float).eps
    tied = np.all(
        (abs(early_fraction - alpha) <= tolerance * alpha) & (abs(late_fraction - beta) <= tolerance * beta),
        axis=1,
    )
    result[tied] = hazard[tied]
    return result


def design_and_offset(
    early: np.ndarray,
    late: np.ndarray,
    alpha: float,
    shape: dict[str, float],
    model: str,
    parameters: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return native-sign benefit/harm columns, with exact native null delegation."""
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown late-refresh family: {model}")
    base = mariner.design_matrix(early + late, shape)
    offset = np.zeros(len(early))
    if model == "aggregate_mariner" or parameters["lambda"] == 0:
        return base, offset
    hazard = effective_hazard(early, late, alpha, shape["rate"], shape["power"], parameters["lambda"])
    matrix = base.copy()
    matrix[:, : early.shape[1]] = np.expm1(-hazard)
    return matrix, offset
