# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "threadpoolctl"]
# ///
"""Effective-harm J2 adapter retaining the complete refined native competitor.

Private instances reuse the frozen rate adapter's final control union and its
canonical J2 backend. Only local feature/grid bindings change. Public records
embed canonical J2 fits, so spawned workers never pickle dynamic classes.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, replace
from types import ModuleType
from typing import Any

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_effective_harm_readout_20261015 as law
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_joint_estimation_20261014 as frozen
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_rate_joint_models_20261015 as control_union

MODEL_NAMES = law.MODEL_NAMES
STRENGTHS = law.STRENGTHS
TestGrid = frozen.TestGrid
mariner = frozen.mariner
parameter_grid = law.parameter_grid
design_and_offset = law.design_and_offset


def isolated_adapter() -> ModuleType:
    """Bind the frozen control-union procedure and J2 to this law privately."""
    name = f"{__name__}._control_union"
    specification = importlib.util.spec_from_file_location(name, control_union.__file__)
    assert specification is not None and specification.loader is not None
    adapter = importlib.util.module_from_spec(specification)
    sys.modules[name] = adapter
    specification.loader.exec_module(adapter)
    adapter.MODEL_NAMES = MODEL_NAMES
    adapter.parameter_grid = parameter_grid
    adapter.design_and_offset = design_and_offset
    adapter._BACKEND.parameter_grid = parameter_grid
    adapter._BACKEND.design_and_offset = design_and_offset
    return adapter


_ADAPTER = isolated_adapter()


@dataclass(frozen=True)
class JointFit:
    """Serializable canonical record with the frozen effective-harm prediction law."""

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
        matrix, offset = design_and_offset(early, late, self.alpha, self.shape, self.model, self.parameters)
        return frozen.temporal.predict_head(self.head, matrix, offset)

    def to_json(self) -> dict[str, Any]:
        payload = self.selected.to_json()
        payload["selection_contract"] = (
            "Frozen J2 plus fresh-CV final union with the complete native aggregate; "
            "native {} maps to lambda=0 only within the temporal search"
        )
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> JointFit:
        return cls(frozen.JointFit.from_json(payload))


def union_control(aggregate: JointFit) -> Any:
    """Represent native parameters in the temporal menu without changing its fit."""
    if aggregate.model != "aggregate_mariner" or aggregate.parameters != {} or aggregate.known_floor is not None:
        raise ValueError("The complete learned-floor native aggregate with {} parameters is required")
    return _ADAPTER.JointFit(replace(aggregate.selected, parameters={"lambda": 0.0}))


def retain_aggregate(
    fit: JointFit,
    aggregate: JointFit,
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
) -> JointFit:
    """Freshly score/refit the complete aggregate and retain all source records."""
    if fit.model != "late_refresh_effective_harm":
        raise ValueError("Final aggregate union applies to the effective-harm family")
    result = _ADAPTER.retain_controls(
        _ADAPTER.JointFit(fit.selected), (union_control(aggregate),), early, late, observed, labels
    )
    search_space = dict(result.search_space)
    search_space["complete_aggregate_source"] = aggregate.to_json()
    return JointFit(replace(result.selected, search_space=search_space))


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
    """Run frozen J2 and require the complete aggregate for temporal fitting.

    Shapes/test_grid are toy diagnostic overrides only. Scientific callers use
    all native shapes/ridges/gammas and the unchanged seven-point lambda grid.
    """
    if model not in MODEL_NAMES or len(controls) != (model == "late_refresh_effective_harm"):
        raise ValueError("Fit aggregate without controls, then effective-harm with its one complete aggregate")
    control = None if not controls else union_control(controls[0])
    result = _ADAPTER._BACKEND.fit_joint(
        early,
        late,
        observed,
        labels,
        alpha,
        model,
        shapes=shapes,
        test_grid=test_grid,
        matched_null_fit=None if control is None else control.selected,
    )
    fit = JointFit.from_json(result.to_json())
    if not controls:
        return fit
    return retain_aggregate(fit, controls[0], early, late, observed, labels)
