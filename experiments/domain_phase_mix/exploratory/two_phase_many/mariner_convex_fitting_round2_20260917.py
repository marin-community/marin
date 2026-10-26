# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///

"""Isolated train-only fitting changes for three fixed MARINER response forms.

The selection-aware score is a component-level proxy for decision quality, not
the aggregate objective's regret. All final coefficients remain nonnegative;
the fitting changes preserve each convex candidate's uncapped mixture geometry.
"""

import dataclasses
import hashlib
import math
from enum import StrEnum
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_convex_candidates_20260917 as equations,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as models,
)

MODULE_SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
FAMILY_IDS = ("cv17_mariner", "cv17_smooth_d1", "cv17_huber_d8")
PENALTY_BOUNDS = (0.25, 4.0)
SELECTION_WEIGHT = 0.5


class Procedure(StrEnum):
    CANONICAL = "canonical"
    BOUNDED_RMS = "bounded_rms"
    SELECTION = "selection"
    JOINT = "joint"


@dataclasses.dataclass(frozen=True)
class CandidateSpec:
    model_id: str
    family_id: str
    procedure: Procedure


def candidate_specs() -> tuple[CandidateSpec, ...]:
    return tuple(
        CandidateSpec(f"cf2_{family.removeprefix('cv17_')}__{procedure}", family, procedure)
        for family in FAMILY_IDS
        for procedure in (Procedure.BOUNDED_RMS, Procedure.SELECTION, Procedure.JOINT)
    )


def training_penalty(design: models.Design, procedure: Procedure) -> np.ndarray:
    """Return coefficient penalties from these training rows alone."""
    if procedure is not Procedure.BOUNDED_RMS:
        return design.ridge
    return design.ridge * np.clip(np.mean(design.values**2, axis=0), *PENALTY_BOUNDS)


def fit_head(
    design: models.Design, response: np.ndarray, ridge: float, spec: models.HeadSpec, procedure: Procedure
) -> models.FittedHead:
    if spec.scale_columns:
        raise ValueError("This screen requires canonical unscaled head coordinates")
    return models.fit_head(
        models.Design(design.values, training_penalty(design, procedure), design.names), response, ridge, spec
    )


@dataclasses.dataclass(frozen=True)
class CvScore:
    objective: float
    rmse: float
    regret: float


def prediction_score(observations: list[np.ndarray], predictions: list[np.ndarray], procedure: Procedure) -> CvScore:
    """Score pooled error and fold-menu decisions without using outer test data."""
    if any(not np.isfinite(prediction).all() for prediction in predictions):
        return CvScore(float("inf"), float("inf"), float("inf"))
    residuals = np.concatenate([p - y for y, p in zip(observations, predictions, strict=True)])
    rmse = float(np.sqrt(np.mean(residuals**2)))
    regret = float(np.mean([y[int(np.argmin(p))] - y.min() for y, p in zip(observations, predictions, strict=True)]))
    objective = rmse + (SELECTION_WEIGHT * regret if procedure is Procedure.SELECTION else 0.0)
    return CvScore(objective, rmse, regret)


def cross_validate(
    design: models.Design,
    response: np.ndarray,
    ridge: float,
    spec: models.HeadSpec,
    inner: models.InnerFolds,
    procedure: Procedure,
) -> CvScore:
    observations, predictions = [], []
    for train, validation in inner:
        head = fit_head(
            models.Design(design.values[train], design.ridge, design.names), response[train], ridge, spec, procedure
        )
        predictions.append(models.predict_head(head, design.values[validation], spec))
        observations.append(response[validation])
    return prediction_score(observations, predictions, procedure)


@dataclasses.dataclass(frozen=True)
class FittingModel:
    """Canonical finite-grid/floor search with an explicit training procedure."""

    reference: models.FittedFloorModel
    procedure: Procedure
    source_hash: str = MODULE_SOURCE_HASH

    @property
    def model_id(self) -> str:
        return self.reference.model_id

    @property
    def base(self) -> models.GridModel:
        grid = self.reference.base
        if not isinstance(grid, models.GridModel):
            raise TypeError("Expected a shared finite shape grid")
        return grid

    def head_spec(self, shape: models.Shape, kappa: float) -> models.HeadSpec:
        return self.reference._spec(shape, models.LinkKind.KAPPA_FLOOR, kappa)

    def fit(
        self, features: models.Features, response: np.ndarray, train: np.ndarray, inner: models.InnerFolds, seed: int
    ) -> models.Fitted:
        if self.procedure in (Procedure.CANONICAL, Procedure.JOINT):
            return self.reference.fit(features, response, train, inner, seed)
        grid = self.base
        shapes = grid.candidate_shapes(features)
        if grid.refine or grid.link_candidates or len(shapes) > grid.screen_top:
            raise ValueError("This isolated screen requires the canonical exhaustive single-link grid")
        table = np.full((len(shapes), len(grid.ridge_grid)), np.inf)
        best = (float("inf"), 0, 0)
        for shape_index, shape in enumerate(shapes):
            design = grid.design(features, shape)
            for ridge_index, ridge in enumerate(grid.ridge_grid):
                score = cross_validate(design, response, ridge, grid.head_for(shape), inner, self.procedure)
                table[shape_index, ridge_index] = score.objective
                best = min(best, (score.objective, shape_index, ridge_index))
        if not math.isfinite(best[0]):
            raise ValueError("No finite inner-CV grid candidate")
        shape = dict(shapes[best[1]])
        ridge = float(grid.ridge_grid[best[2]])
        design = grid.design(features, shape)

        def objective(log_kappa: float) -> float:
            score = cross_validate(
                design, response, ridge, self.head_spec(shape, math.exp(log_kappa)), inner, self.procedure
            ).objective
            return score if math.isfinite(score) else models.INFINITE_CV_PENALTY

        low, high = self.reference.kappa_bounds
        search = minimize_scalar(
            objective,
            bounds=(math.log(low), math.log(high)),
            method="bounded",
            options={"maxiter": self.reference.search_evaluations, "xatol": 0.02},
        )
        kappa = float(math.exp(search.x))
        evaluations = int(search.nfev)
        if self.reference.flat_profile_kappa is not None and math.log(kappa) >= (
            1 - models.FLAT_PROFILE_FRACTION
        ) * math.log(high):
            kappa = float(self.reference.flat_profile_kappa)
            evaluations += 1
        spec = self.head_spec(shape, kappa)
        score = cross_validate(design, response, ridge, spec, inner, self.procedure)
        head = fit_head(
            models.Design(design.values[train], design.ridge, design.names), response[train], ridge, spec, self.procedure
        )
        return models.Fitted(
            shape,
            ridge,
            head,
            {
                "inner_cv_objective": score.objective,
                "inner_cv_rmse": score.rmse,
                "inner_cv_regret": score.regret,
                "selection_inner_cv_objective": best[0],
                "criterion": str(self.procedure),
                "candidates": table.size,
                "link": str(spec.link),
                "kappa": head.kappa,
                "floor": head.floor,
                "converged": bool(head.converged),
                "kappa_search_evaluations": evaluations,
                "columns": design.values.shape[1],
                "effective_rank": models.effective_rank(design.values[train]),
                "fitted_dof": head.active + 1 + grid.shape_dof + 1,
                "nonlinear_dof": grid.shape_dof + 1,
            },
            cv_table=table,
        )

    def predict(self, fitted: models.Fitted, features: models.Features, rows: np.ndarray) -> np.ndarray:
        return self.reference.predict(fitted, features, rows)

    def nonlinear_dof(self, features: models.Features) -> int:
        return self.reference.nonlinear_dof(features)


def build_model(spec: CandidateSpec, features: models.Features) -> FittingModel:
    equation = next(item for item in equations.candidate_specs() if item.model_id == spec.family_id)
    model = equations.build_model(equation, features)
    if not isinstance(model.base, models.GridModel):
        raise TypeError("Expected shared grid")
    if model.candidate_links != (models.LinkKind.KAPPA_FLOOR,) or model.kappa_rule != "argmin":
        raise ValueError("Frozen floor procedure changed")
    grid = dataclasses.replace(model.base, model_id=spec.model_id)
    reference = dataclasses.replace(model, base=grid, joint_rounds=int(spec.procedure is Procedure.JOINT))
    return FittingModel(reference, spec.procedure)
