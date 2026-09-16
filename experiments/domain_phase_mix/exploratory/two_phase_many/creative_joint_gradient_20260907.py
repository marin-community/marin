# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Analytical policy gradients for frozen recency WSPU ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fit_two_phase_creative_joint_20260907 import DEFAULT_OUTPUT as INITIAL_OUTPUT
from fit_two_phase_creative_joint_20260907 import load_predictor as load_initial
from fit_two_phase_creative_joint_followup_20260907 import DEFAULT_OUTPUT as FOLLOWUP_OUTPUT
from fit_two_phase_creative_joint_followup_20260907 import load_predictor as load_followup
from scipy.special import expit


@dataclass(frozen=True)
class RecencyGradient:
    """Return BPB and ambient weight gradients, with no boundary smoothing.

    Active benefit derivatives can be infinite at exactly zero exposure when
    their inherited power is below one. Interior softmax policies avoid that
    mathematical boundary. The numerical log clipping has derivative zero
    outside and at its two endpoints.
    """

    c0: np.ndarray
    c1: np.ndarray
    records: tuple[dict[str, Any], ...]
    aggregation_weights: np.ndarray

    def value_and_gradient(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate batched policies shaped (n, 2, 39)."""
        if weights.ndim != 3 or weights.shape[1:] != (2, 39):
            raise ValueError("weights must have shape (n,2,39)")
        if np.any(weights < 0) or not np.isfinite(weights).all():
            raise ValueError("weights must be finite and nonnegative")
        inventory = self.c0 + self.c1
        rho = self.c0 / inventory
        early, late = self.c0 * weights[:, 0], self.c1 * weights[:, 1]
        tied = weights[:, 0] == weights[:, 1]
        total = np.where(tied, inventory * weights[:, 0], early + late)
        state = np.asarray([record["state"] for record in self.records])[:, None, None]
        rate = np.asarray([record["shape"]["rate"] for record in self.records])[:, None, None]
        power = np.asarray([record["shape"]["power"] for record in self.records])[:, None, None]
        threshold = np.asarray([record["shape"]["threshold"] for record in self.records])[:, None, None]
        denominator = state * rho + 1 - rho
        effective = (state * early + late) / denominator
        effective = np.where(tied[None], total[None], effective)
        effective = np.where(state == 0, inventory * weights[:, 1], effective)
        z = rate * effective
        z_power = z**power
        benefit = -np.expm1(-z_power)
        log_argument = np.log1p(total)[None] - threshold
        softplus = np.logaddexp(log_argument, 0)
        harm = softplus**2 - np.logaddexp(-threshold, 0) ** 2
        coefficients = np.asarray([record["coefficients"] for record in self.records])
        alpha, beta = coefficients[:, :39], coefficients[:, 39:]
        intercept = np.asarray([record["intercept"] for record in self.records])[:, None]
        floor = np.asarray([record["floor"] for record in self.records])[:, None]
        eta = intercept - np.einsum("cnb,cb->cn", benefit, alpha) + np.einsum("cnb,cb->cn", harm, beta)
        deficit = np.exp(np.clip(eta, -30, 30))
        values = self.aggregation_weights @ (floor + deficit)
        with np.errstate(divide="ignore", invalid="ignore"):
            derivative_b = rate * power * z ** (power - 1) * np.exp(-z_power)
            weighted_b = np.where(alpha[:, None] == 0, 0, alpha[:, None] * derivative_b)
        derivative_h = 2 * softplus * expit(log_argument) / (1 + total)[None]
        effective0 = state * self.c0 / denominator
        effective1 = self.c1 / denominator
        # At r=0 the implemented full-duration suffix index is exactly I*w1.
        effective1 = np.where(state == 0, inventory, effective1)
        weighted_h = beta[:, None] * derivative_h
        with np.errstate(invalid="ignore"):
            early_b = np.where(effective0 == 0, 0, weighted_b * effective0)
            late_b = np.where(effective1 == 0, 0, weighted_b * effective1)
        derivative0 = -early_b + weighted_h * self.c0
        derivative1 = -late_b + weighted_h * self.c1
        active = (eta > -30) & (eta < 30)
        response_derivative = np.where(active, deficit, 0)
        with np.errstate(invalid="ignore"):
            gradient0 = np.where(active[:, :, None], derivative0 * response_derivative[:, :, None], 0)
            gradient1 = np.where(active[:, :, None], derivative1 * response_derivative[:, :, None], 0)
        gradient = np.stack(
            [
                np.einsum("c,cnb->nb", self.aggregation_weights, gradient0),
                np.einsum("c,cnb->nb", self.aggregation_weights, gradient1),
            ],
            axis=1,
        )
        return values, gradient

    def tied_value_and_gradient(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the exact tied restriction on mixtures shaped (n,39)."""
        if weights.ndim != 2 or weights.shape[1] != 39:
            raise ValueError("tied weights must have shape (n,39)")
        values, gradient = self.value_and_gradient(np.repeat(weights[:, None], 2, axis=1))
        return values, gradient.sum(axis=1)


def load_gradient(model: str, objective: str, context: str, output: Path | None = None) -> RecencyGradient:
    """Load CRE2-002 or CRE2-013 gradients without refitting any model."""
    if model == "CRE2-002":
        predictor = load_initial(model, objective, context, INITIAL_OUTPUT if output is None else output)
    elif model == "CRE2-013":
        predictor = load_followup(model, objective, context, FOLLOWUP_OUTPUT if output is None else output)
    else:
        raise ValueError("the recency gradient supports CRE2-002 and CRE2-013")
    return RecencyGradient(predictor.c0, predictor.c1, predictor.records, predictor.aggregation_weights)
