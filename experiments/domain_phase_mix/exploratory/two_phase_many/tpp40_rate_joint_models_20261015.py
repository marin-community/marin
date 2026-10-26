# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "threadpoolctl"]
# ///
"""Frozen J2 search plus a final union of fully refined nested controls.

The canonical estimator and stage-one laws remain unchanged. A private J2
module uses the qualified rate/odds features. The combined model additionally
rescans the complete selected aggregate, benefit-only and rate-only candidates
after its ordinary J2 schedule, refitting every inner and full amplitude head.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass, replace
from types import ModuleType
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_joint_estimation_20261014 as frozen
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_rate_reopening_20261015 as law

MODEL_NAMES = law.FAMILIES
TestGrid = frozen.TestGrid
mariner = frozen.mariner


def parameter_grid(model: str) -> tuple[dict[str, float], ...]:
    return law.parameter_grid(model)


def design_and_offset(
    early: np.ndarray, late: np.ndarray, alpha: float, shape: dict[str, float], model: str, parameters: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    if model not in MODEL_NAMES:
        raise ValueError(f"Unsupported rate comparison: {model}")
    return law.design(early, late, alpha, shape, parameters), np.zeros(len(early))


def isolated_backend() -> ModuleType:
    """Bind laws locally without modifying the canonical J2 module."""
    name = f"{__name__}._frozen_joint"
    specification = importlib.util.spec_from_file_location(name, frozen.__file__)
    assert specification is not None and specification.loader is not None
    backend = importlib.util.module_from_spec(specification)
    sys.modules[name] = backend
    specification.loader.exec_module(backend)
    backend.design_and_offset = design_and_offset
    backend.parameter_grid = parameter_grid
    return backend


_BACKEND = isolated_backend()


@dataclass(frozen=True)
class JointFit:
    """Canonical fitted record with prediction under the qualified rate law."""

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
    def head(self) -> mariner.Head:
        return self.selected.head

    @property
    def inner_rmse(self) -> float:
        return self.selected.inner_rmse

    @property
    def selection_profile(self) -> tuple[dict[str, Any], ...]:
        return self.selected.selection_profile

    @property
    def search_space(self) -> dict[str, Any]:
        return self.selected.search_space

    def predict(self, early: np.ndarray, late: np.ndarray) -> np.ndarray:
        return law.prediction(early, late, self.alpha, self.shape, self.parameters, self.head)

    def to_json(self) -> dict[str, Any]:
        payload = self.selected.to_json()
        payload["selection_contract"] = (
            "Frozen J2 schedule, then fresh-CV final candidate union with supplied nested controls"
        )
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> JointFit:
        return cls(frozen.JointFit.from_json(payload))


def trial_key(item: tuple[int, dict[str, Any]]) -> tuple[float, int, int, int, float, int]:
    index, trial = item
    return (
        trial["inner_rmse"],
        trial["parameter_index"],
        trial["shape_index"],
        trial["ridge_index"],
        trial["gamma"],
        index,
    )


def retain_controls(
    fit: JointFit,
    controls: tuple[JointFit, ...],
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
) -> JointFit:
    """Union complete control candidates with fresh training-only CV and head fits."""
    if not controls:
        return fit
    profile = list(fit.selection_profile)
    search_space = dict(fit.search_space)
    retained: list[dict[str, Any]] = []
    full_heads: dict[int, mariner.Head] = {}
    allowed = set(MODEL_NAMES[:3]) if fit.model == "bounded_benefit_rate" else {"aggregate_mariner"}
    with threadpool_limits(limits=1):
        for control in controls:
            if control.model not in allowed or control.alpha != fit.alpha or control.known_floor is not None:
                raise ValueError("Controls must be nested learned-floor estimators on the same phase geometry")
            si = search_space["shapes"].index(control.shape)
            pi = search_space["parameters"].index(control.parameters)
            ri = search_space["ridges"].index(control.ridge)
            matrix, offset = design_and_offset(early, late, fit.alpha, control.shape, fit.model, control.parameters)
            inner = frozen.candidate_oof(matrix, offset, observed, labels, control.ridge, control.kappa)
            score = float(np.sqrt(np.mean((inner - observed) ** 2)))
            if not math.isfinite(score):
                raise ValueError("The supplied nested control produced nonfinite fresh CV predictions")
            head = frozen.fit_fixed_head(matrix, offset, observed, control.ridge, control.kappa, None)
            index = len(profile)
            profile.append(
                {
                    "stage": "retained_control_final_union",
                    "shape_index": si,
                    "parameter_index": pi,
                    "ridge_index": ri,
                    "gamma": control.kappa,
                    "inner_rmse": score,
                    "cached": False,
                    "refinement_index": None,
                    "control_model": control.model,
                }
            )
            full_heads[index] = head
            retained.append(
                {
                    "source_model": control.model,
                    "source_inner_rmse": control.inner_rmse,
                    "fresh_inner_rmse": score,
                    "trial_index": index,
                    "source_selected_from_failed_refinement": control.search_space["selected_from_failed_refinement"],
                    "shape": control.shape,
                    "parameters": control.parameters,
                    "ridge": control.ridge,
                    "gamma": control.kappa,
                    "inner_predictions": inner.tolist(),
                    "head": {
                        "intercept": head.intercept,
                        "coefficients": head.coefficients.tolist(),
                        "floor": head.floor,
                    },
                }
            )
    selected, trial = min(((i, t) for i, t in enumerate(profile) if t["inner_rmse"] is not None), key=trial_key)
    inherited_failed = fit.search_space["selected_from_failed_refinement"]
    if selected in full_heads:
        inherited_failed = next(
            item["source_selected_from_failed_refinement"] for item in retained if item["trial_index"] == selected
        )
    search_space.update(
        {
            "selected_trial_index": selected,
            "unique_cv_evaluations": fit.search_space["unique_cv_evaluations"] + len(retained),
            "selected_from_failed_refinement": inherited_failed,
            "retained_control_candidates": retained,
            "control_union_scope": "Final candidate union after unchanged J2 schedule; no additional gamma refinement",
        }
    )
    return JointFit(
        replace(
            fit.selected,
            shape=search_space["shapes"][trial["shape_index"]],
            parameters=search_space["parameters"][trial["parameter_index"]],
            ridge=search_space["ridges"][trial["ridge_index"]],
            kappa=trial["gamma"],
            inner_rmse=trial["inner_rmse"],
            head=full_heads[selected] if selected in full_heads else fit.head,
            selection_profile=tuple(profile),
            search_space=search_space,
        )
    )


def fit_joint(
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    model: str,
    *,
    controls: tuple[JointFit, ...] = (),
    shapes: tuple[dict[str, float], ...] | None = None,
    test_grid: TestGrid | None = None,
) -> JointFit:
    """Fit the unchanged J2 schedule and retain the complete nested controls.

    Explicit shapes/test_grid are only diagnostic overrides. Production callers
    use all native shapes/ridges/gammas and the registered 1/10/4/40 kinetic grids.
    """
    aggregates = [item for item in controls if item.model == "aggregate_mariner"]
    if len(aggregates) > 1:
        raise ValueError("Provide at most one matched aggregate control")
    result = _BACKEND.fit_joint(
        early,
        late,
        observed,
        labels,
        alpha,
        model,
        shapes=shapes,
        test_grid=test_grid,
        matched_null_fit=None if not aggregates else aggregates[0].selected,
    )
    fit = JointFit.from_json(result.to_json())
    return retain_controls(fit, controls, early, late, observed, labels)
