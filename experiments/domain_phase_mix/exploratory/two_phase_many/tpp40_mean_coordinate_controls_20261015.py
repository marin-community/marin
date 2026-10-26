# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "threadpoolctl"]
# ///
"""Pure hazard/benefit-coordinate means with native physical cumulative harm.

These are constrained readout-curvature controls, with no fitter or data access.
The benefit mean constrains the existing cumulative-recency DSP feature family.
The hazard mean equals the normalized benefit clock at power one. Both retain
the native tied class and the fixed-aggregate convexity restriction for p<=1.
"""

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_late_refresh_20261015 as refresh

mariner = refresh.mariner
MODEL_NAMES = ("aggregate_mariner", "hazard_coordinate_mean", "benefit_coordinate_mean")
STRENGTHS = refresh.STRENGTHS


def parameter_grid(model: str) -> tuple[dict[str, float], ...]:
    """Return the unchanged seven-strength grid with the exact native null first."""
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown mean-coordinate family: {model}")
    return ({},) if model == "aggregate_mariner" else tuple({"lambda": value} for value in STRENGTHS)


def relative_tied_rows(early: np.ndarray, late: np.ndarray, alpha: float) -> np.ndarray:
    """Recognize tied rows by relative errors in both physical phase fractions."""
    total, beta = early + late, 1 - alpha
    first = np.divide(early, total, out=np.full_like(total, alpha), where=total > 0)
    last = np.divide(late, total, out=np.full_like(total, beta), where=total > 0)
    tolerance = refresh.temporal.TIED_ROUNDOFF_MULTIPLIER * np.finfo(float).eps
    return np.all((abs(first - alpha) <= tolerance * alpha) & (abs(last - beta) <= tolerance * beta), axis=1)


def mean_benefit(
    early: np.ndarray,
    late: np.ndarray,
    alpha: float,
    rate: float,
    power: float,
    model: str,
    strength: float,
) -> np.ndarray:
    """Return the bounded mean benefit using L=late/(1-alpha).

    Hazard mean: 1-exp(-(1-lambda)*(r*E)^p-lambda*(r*L)^p).
    Benefit mean: (1-lambda)*b(E)+lambda*b(L). Tied rows and the native
    null use the native benefit evaluator exactly. Float64 overflow raises;
    there is no fitted cap, denominator floor, or small-dose threshold.
    """
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown mean-coordinate family: {model}")
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
    with np.errstate(over="raise", invalid="raise"):
        native = mariner.benefit(total, rate, power)
        if model == "aggregate_mariner" or strength == 0:
            return native
        normalized_late = late / (1 - alpha)
        if model == "hazard_coordinate_mean":
            hazard = (1 - strength) * (rate * total) ** power + strength * (rate * normalized_late) ** power
            result = -np.expm1(-hazard)
        else:
            result = (1 - strength) * native + strength * mariner.benefit(normalized_late, rate, power)
    tied = relative_tied_rows(early, late, alpha)
    result[tied] = native[tied]
    return result


def design_and_offset(
    early: np.ndarray,
    late: np.ndarray,
    alpha: float,
    shape: dict[str, float],
    model: str,
    parameters: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return native-sign benefit and physical-harm columns, with zero offset."""
    strength = 0.0 if model == "aggregate_mariner" else parameters["lambda"]
    benefit = mean_benefit(early, late, alpha, shape["rate"], shape["power"], model, strength)
    base = mariner.design_matrix(early + late, shape)
    offset = np.zeros(len(early))
    if model == "aggregate_mariner" or strength == 0:
        return base, offset
    matrix = base.copy()
    matrix[:, : early.shape[1]] = -benefit
    return matrix, offset
