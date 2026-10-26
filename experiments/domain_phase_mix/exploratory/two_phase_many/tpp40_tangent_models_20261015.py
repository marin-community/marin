# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "threadpoolctl"]
# ///
"""Matched original/tangent odds laws using the frozen J2 search algorithm.

The frozen source is loaded into a private module instance. Only that instance's
feature and kinetic-grid functions change; the canonical module stays intact.
Returned fits embed the canonical serializable record, never the dynamic class.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_joint_estimation_20261014 as frozen

mariner = frozen.mariner
TestGrid = frozen.TestGrid
MODEL_NAMES = ("aggregate_mariner", "bounded_benefit_odds", "tangent_benefit_odds")
STRENGTHS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


def parameter_grid(model: str) -> tuple[dict[str, float], ...]:
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown matched odds family: {model}")
    return ({},) if model == "aggregate_mariner" else tuple({"kappa": strength} for strength in STRENGTHS)


def odds_shift(benefit: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Apply a log-odds shift while preserving native floating-point endpoints."""
    result = benefit.copy()
    interior = (benefit > 0) & (benefit < 1)
    value = benefit[interior]
    result[interior] = expit(np.log(value) - np.log1p(-value) + shift[interior])
    return result


def design_and_offset(
    early: np.ndarray,
    late: np.ndarray,
    alpha: float,
    shape: dict[str, float],
    model: str,
    parameters: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return native harm and an original or first-order benefit odds shift."""
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown matched odds family: {model}")
    total = early + late
    base = mariner.design_matrix(total, shape)
    offset = np.zeros(len(total))
    if model == "aggregate_mariner" or parameters["kappa"] == 0:
        return base, offset
    width = total.shape[1]
    rate, power = shape["rate"], shape["power"]
    if model == "bounded_benefit_odds":
        departure = mariner.benefit(alpha * total, rate, power) - mariner.benefit(early, rate, power)
    else:
        q = (alpha * rate * total) ** power
        fraction = np.divide(early, alpha * total, out=np.ones_like(total), where=total > 0)
        departure = power * q * np.exp(-q) * (1.0 - fraction)
    matrix = base.copy()
    matrix[:, :width] = -odds_shift(-base[:, :width], parameters["kappa"] * departure)
    tied = frozen.temporal.tied_rows(early, late, alpha)
    matrix[tied] = base[tied]
    return matrix, offset


def isolated_backend() -> ModuleType:
    """Load the unchanged J2 estimator with locally bound feature/grid functions."""
    name = f"{__name__}._frozen_joint"
    spec = importlib.util.spec_from_file_location(name, Path(frozen.__file__))
    assert spec is not None and spec.loader is not None
    backend = importlib.util.module_from_spec(spec)
    sys.modules[name] = backend
    spec.loader.exec_module(backend)
    backend.design_and_offset = design_and_offset
    backend.parameter_grid = parameter_grid
    return backend


_BACKEND = isolated_backend()


@dataclass(frozen=True)
class JointFit:
    """Pickle-safe fitted record with prediction under this module's law."""

    selected: frozen.JointFit

    @property
    def model(self) -> str:
        return self.selected.model

    @property
    def alpha(self) -> float:
        return self.selected.alpha

    @property
    def shape(self) -> dict[str, float]:
        return self.selected.shape

    @property
    def parameters(self) -> dict[str, float]:
        return self.selected.parameters

    @property
    def ridge(self) -> float:
        return self.selected.ridge

    @property
    def kappa(self) -> float | None:
        return self.selected.kappa

    @property
    def known_floor(self) -> float | None:
        return self.selected.known_floor

    @property
    def inner_rmse(self) -> float:
        return self.selected.inner_rmse

    @property
    def head(self) -> mariner.Head:
        return self.selected.head

    @property
    def selection_profile(self) -> tuple[dict[str, Any], ...]:
        return self.selected.selection_profile

    @property
    def search_space(self) -> dict[str, Any]:
        return self.selected.search_space

    def predict(self, early: np.ndarray, late: np.ndarray) -> np.ndarray:
        matrix, offset = design_and_offset(early, late, self.alpha, self.shape, self.model, self.parameters)
        return frozen.temporal.predict_head(self.head, matrix, offset)

    def to_json(self) -> dict[str, Any]:
        return self.selected.to_json()

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> JointFit:
        return cls(frozen.JointFit.from_json(payload))


def fit_joint(
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    model: str,
    *,
    shapes: tuple[dict[str, float], ...] | None = None,
    test_grid: frozen.TestGrid | None = None,
    matched_null_fit: JointFit | None = None,
) -> JointFit:
    """Run the frozen J2 schedule; explicit shapes/test_grid are diagnostic overrides."""
    result = _BACKEND.fit_joint(
        early,
        late,
        observed,
        labels,
        alpha,
        model,
        shapes=shapes,
        test_grid=test_grid,
        matched_null_fit=None if matched_null_fit is None else matched_null_fit.selected,
    )
    return JointFit.from_json(result.to_json())


def fit_known_floor(
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    model: str,
    *,
    floor: float,
    shapes: tuple[dict[str, float], ...] | None = None,
    test_grid: frozen.TestGrid | None = None,
    matched_null_fit: JointFit | None = None,
) -> JointFit:
    """Run the same discrete search with a supplied diagnostic absolute floor."""
    result = _BACKEND.fit_known_floor(
        early,
        late,
        observed,
        labels,
        alpha,
        model,
        floor=floor,
        shapes=shapes,
        test_grid=test_grid,
        matched_null_fit=None if matched_null_fit is None else matched_null_fit.selected,
    )
    return JointFit.from_json(result.to_json())


def strength_profiles(
    fit: JointFit,
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    query_early: np.ndarray,
    query_late: np.ndarray,
) -> tuple[dict[str, Any], ...]:
    """Replay each strength's best searched candidate with local OOF/full heads.

    These profiles inherit J2's common conditional gamma schedule; they are not
    separately gamma-refined estimators and must not select models on query data.
    """
    results = []
    with threadpool_limits(limits=1):
        for parameter_index, parameters in enumerate(fit.search_space["parameters"]):
            candidates = [
                (index, trial)
                for index, trial in enumerate(fit.selection_profile)
                if trial["parameter_index"] == parameter_index and trial["inner_rmse"] is not None
            ]
            index, trial = min(
                candidates,
                key=lambda item: (
                    item[1]["inner_rmse"],
                    item[1]["shape_index"],
                    item[1]["ridge_index"],
                    0.0 if item[1]["gamma"] is None else item[1]["gamma"],
                    item[0],
                ),
            )
            shape = fit.search_space["shapes"][trial["shape_index"]]
            ridge = fit.search_space["ridges"][trial["ridge_index"]]
            gamma = trial["gamma"]
            matrix, offset = design_and_offset(early, late, fit.alpha, shape, fit.model, parameters)
            inner = frozen.candidate_oof(matrix, offset, observed, labels, ridge, gamma, known_floor=fit.known_floor)
            score = float(np.sqrt(np.mean(np.square(inner - observed))))
            np.testing.assert_allclose(score, trial["inner_rmse"], rtol=1e-12, atol=1e-14)
            head = frozen.fit_fixed_head(matrix, offset, observed, ridge, gamma, fit.known_floor)
            candidate = JointFit(
                frozen.JointFit(
                    fit.model,
                    fit.alpha,
                    dict(shape),
                    dict(parameters),
                    ridge,
                    gamma,
                    fit.known_floor,
                    score,
                    head,
                    (dict(trial),),
                    {
                        "selected_trial_index": 0,
                        "parent_trial_index": index,
                        "parent_parameter_index": parameter_index,
                        "profile_scope": (
                            "best candidate already visited by common J2 schedule; no separate gamma refinement"
                        ),
                    },
                )
            )
            results.append(
                {
                    "strength": parameters.get("kappa", 0.0),
                    "parent_trial_index": index,
                    "selected_trial": trial,
                    "inner_rmse": score,
                    "model": candidate.to_json(),
                    "inner_predictions": inner.tolist(),
                    "query_predictions": candidate.predict(query_early, query_late).tolist(),
                    "scope": "diagnostic conditional profile; query labels are not inputs",
                }
            )
    return tuple(results)
