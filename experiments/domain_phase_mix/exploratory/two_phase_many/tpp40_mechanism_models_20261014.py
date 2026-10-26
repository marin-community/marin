# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "threadpoolctl"]
# ///
"""Bounded benefit controls and cumulative rate-sensitive repetition damage.

The aggregate MARINER shape, ridge and floor multiplier are supplied by a
previous asymmetric-only fit. Conditional inner selection refits amplitudes
and training-local floors. The odds and hazard variants change the readout of
the earlier persistence departure; neither identifies a physical memory law.
Rate-sensitive damage weights cumulative harm increments without resetting
the repetition clock at the phase boundary.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_temporal_models_20261014 as temporal

mariner = temporal.mariner
MODEL_NAMES = ("bounded_benefit_odds", "rate_damage", "bounded_benefit_rate", "hazard_recency", "contrast_magnitude")
ODDS_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
RATE_GRID = (0.0, 0.5, 1.0, 2.0)
HAZARD_GRID = (0.0, 0.25, 0.5, 1.0)
MAGNITUDE_GRID = (0.0, -0.01, 0.01, -0.03, 0.03, -0.1, 0.1, -0.3, 0.3, -1.0, 1.0)


def parameter_grid(model: str) -> tuple[dict[str, float], ...]:
    """Enumerate each registered conditional search with its native null first."""
    if model == "bounded_benefit_odds":
        return tuple({"kappa": value} for value in ODDS_GRID)
    if model == "rate_damage":
        return tuple({"nu": value} for value in RATE_GRID)
    if model == "bounded_benefit_rate":
        return tuple({"kappa": kappa, "nu": nu} for kappa, nu in itertools.product(ODDS_GRID, RATE_GRID))
    if model == "hazard_recency":
        return tuple({"rho": value} for value in HAZARD_GRID)
    if model == "contrast_magnitude":
        return tuple({"theta": value} for value in MAGNITUDE_GRID)
    raise ValueError(f"Unknown mechanism model: {model}")


def design_and_offset(
    early: np.ndarray,
    late: np.ndarray,
    alpha: float,
    shape: dict[str, float],
    model: str,
    parameters: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return native-sign features, exactly native on null and tied schedules."""
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown mechanism model: {model}")
    total = early + late
    base = mariner.design_matrix(total, shape)
    offset = np.zeros(len(total))
    if not any(parameters.values()):
        return base, offset

    matrix = base.copy()
    buckets = total.shape[1]
    rate, power, threshold = shape["rate"], shape["power"], shape["threshold"]
    if model in ("bounded_benefit_odds", "bounded_benefit_rate") and parameters["kappa"]:
        benefit = -base[:, :buckets]
        departure = mariner.benefit(alpha * total, rate, power) - mariner.benefit(early, rate, power)
        multiplier = np.exp(parameters["kappa"] * departure)
        # This algebraic odds transform remains defined at b=0 and b=1,
        # unlike evaluating logit(b) explicitly at saturated exposures.
        matrix[:, :buckets] = -(benefit * multiplier / (1.0 - benefit + benefit * multiplier))
    if model == "hazard_recency":
        rho = parameters["rho"]
        if not 0.0 <= rho <= 1.0:
            raise ValueError("Hazard departure requires rho in [0, 1]")
        whole_hazard = (rate * total) ** power
        first_hazard = (rate * early) ** power
        tied_hazard = (rate * alpha * total) ** power
        # Rearranging q(E) + rho*(q(alpha E)-q(E0)) keeps its
        # nonnegativity explicit even at the early-only, rho=1 limit.
        hazard = (1.0 - rho) * whole_hazard + rho * (whole_hazard - first_hazard + tied_hazard)
        matrix[:, :buckets] = np.expm1(-hazard)
    if model in ("rate_damage", "bounded_benefit_rate") and parameters["nu"]:
        nu = parameters["nu"]
        first_rate = np.divide(early, alpha * total, out=np.zeros_like(total), where=total > 0)
        second_rate = np.divide(late, (1.0 - alpha) * total, out=np.zeros_like(total), where=total > 0)
        origin = float(mariner.harm(np.array(0.0), threshold))
        first_harm = mariner.harm(early, threshold)
        whole_harm = base[:, buckets:]
        matrix[:, buckets:] = (
            origin + first_rate**nu * (first_harm - origin) + second_rate**nu * (whole_harm - first_harm)
        )
    if model == "contrast_magnitude":
        fraction = np.divide(early, total, out=np.full_like(total, alpha), where=total > 0)
        magnitude = np.mean((fraction - alpha) ** 2, axis=1) / (alpha * (1.0 - alpha))
        offset = parameters["theta"] * magnitude

    tied = temporal.tied_rows(early, late, alpha)
    matrix[tied] = base[tied]
    offset[tied] = 0.0
    return matrix, offset


@dataclass(frozen=True)
class MechanismFit:
    model: str
    alpha: float
    shape: dict[str, float]
    ridge: float
    kappa: float
    flat_profile: bool
    aggregate_inner_rmse: float
    conditional_inner_rmse: float
    parameters: dict[str, float]
    head: mariner.Head
    selection_profile: tuple[dict[str, Any], ...]

    def predict(self, early: np.ndarray, late: np.ndarray) -> np.ndarray:
        matrix, offset = design_and_offset(early, late, self.alpha, self.shape, self.model, self.parameters)
        return temporal.predict_head(self.head, matrix, offset)

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "alpha": self.alpha,
            "shape": self.shape,
            "ridge": self.ridge,
            "kappa": self.kappa,
            "flat_profile": self.flat_profile,
            "aggregate_inner_rmse": self.aggregate_inner_rmse,
            "conditional_inner_rmse": self.conditional_inner_rmse,
            "parameters": self.parameters,
            "head": {
                "intercept": self.head.intercept,
                "coefficients": self.head.coefficients.tolist(),
                "floor": self.head.floor,
            },
            "selection_profile": list(self.selection_profile),
            "calibration_contract": "asymmetric training median; no tied anchor or repeat noise",
            "selection_contract": "prior native aggregate search then conditional kinetics; inner heads refit",
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> MechanismFit:
        head = value["head"]
        return cls(
            model=value["model"],
            alpha=float(value["alpha"]),
            shape={key: float(number) for key, number in value["shape"].items()},
            ridge=float(value["ridge"]),
            kappa=float(value["kappa"]),
            flat_profile=bool(value["flat_profile"]),
            aggregate_inner_rmse=float(value["aggregate_inner_rmse"]),
            conditional_inner_rmse=float(value["conditional_inner_rmse"]),
            parameters={key: float(number) for key, number in value["parameters"].items()},
            head=mariner.Head(float(head["intercept"]), np.asarray(head["coefficients"]), float(head["floor"])),
            selection_profile=tuple(value["selection_profile"]),
        )


def fit_extension(
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    aggregate: temporal.TemporalFit,
    model: str,
) -> MechanismFit:
    """Fit a conditional mechanism using asymmetric outcomes and fresh inner heads."""
    if early.ndim != 2 or early.shape != late.shape or len(early) != len(observed) or len(labels) != len(observed):
        raise ValueError("Training exposures, outcomes and folds must align")
    if np.any(early < 0) or np.any(late < 0) or not np.isfinite(early + late).all():
        raise ValueError("Training exposures must be finite and nonnegative")
    if not np.isfinite(observed).all() or temporal.tied_rows(early, late, aggregate.alpha).any():
        raise ValueError("Fitting requires finite asymmetric outcomes; tied policies are evaluation-only")
    if aggregate.model != "aggregate_mariner":
        raise ValueError("Conditional selection requires the native aggregate fit")
    splits = temporal.inner_splits(labels)
    grid = parameter_grid(model)
    profile = []
    best = None
    with threadpool_limits(limits=1):
        for index, parameters in enumerate(grid):
            matrix, offset = design_and_offset(early, late, aggregate.alpha, aggregate.shape, model, parameters)
            squared_error = 0.0
            count = 0
            for train, held in splits:
                head = temporal.fit_head_with_offset(
                    matrix[train], offset[train], observed[train], aggregate.ridge, aggregate.kappa
                )
                predicted = temporal.predict_head(head, matrix[held], offset[held])
                squared_error += float(np.sum(np.square(predicted - observed[held])))
                count += len(held)
            score = math.sqrt(squared_error / count)
            profile.append({"parameters": parameters, "conditional_inner_rmse": score})
            if best is None or (score, index) < best:
                best = score, index
        assert best is not None
        if not math.isfinite(best[0]):
            raise ValueError(f"No finite conditional fit for {model}")
        parameters = grid[best[1]]
        matrix, offset = design_and_offset(early, late, aggregate.alpha, aggregate.shape, model, parameters)
        head = temporal.fit_head_with_offset(matrix, offset, observed, aggregate.ridge, aggregate.kappa)
    return MechanismFit(
        model=model,
        alpha=aggregate.alpha,
        shape=aggregate.shape,
        ridge=aggregate.ridge,
        kappa=aggregate.kappa,
        flat_profile=aggregate.flat_profile,
        aggregate_inner_rmse=aggregate.aggregate_inner_rmse,
        conditional_inner_rmse=best[0],
        parameters=parameters,
        head=head,
        selection_profile=tuple(profile),
    )


def refine_ridge_gamma(
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    aggregate: temporal.TemporalFit,
) -> temporal.TemporalFit:
    """Rescore native ridge choices at their fitted floor, holding shape fixed."""
    if aggregate.model != "aggregate_mariner":
        raise ValueError("Ridge/floor refinement requires the native aggregate fit")
    if temporal.tied_rows(early, late, aggregate.alpha).any():
        raise ValueError("Tied policies are evaluation-only")
    splits = temporal.inner_splits(labels)
    matrix = mariner.design_matrix(early + late, aggregate.shape)
    anchor = mariner.Anchor(float("nan"), 0.0)
    ridge_grid = sorted(mariner.RIDGE_GRID, key=lambda value: value != aggregate.ridge)
    profile = []
    best = None
    with threadpool_limits(limits=1):
        for index, ridge in enumerate(ridge_grid):
            kappa, flat, score = mariner.search_kappa(matrix, observed, ridge, anchor, splits)
            profile.append({"ridge": ridge, "kappa": kappa, "flat_profile": flat, "conditional_inner_rmse": score})
            if best is None or (score, index) < best:
                best = score, index
        assert best is not None
        selected = profile[best[1]]
        head = mariner.fit_head(
            matrix, observed, selected["ridge"], mariner.FloorSpec(float("nan"), 0.0, selected["kappa"])
        )
    return replace(
        aggregate,
        ridge=selected["ridge"],
        kappa=selected["kappa"],
        flat_profile=selected["flat_profile"],
        aggregate_inner_rmse=best[0],
        conditional_inner_rmse=best[0],
        head=head,
        selection_profile=tuple(profile),
    )
