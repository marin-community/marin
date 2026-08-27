# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare, fit, and evaluate transfer GPs over data mixtures."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, TypedDict

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll_torch
from botorch.models import SingleTaskGP
from botorch.optim.core import OptimizationResult
from gpytorch.kernels import Kernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

from experiments.datakit.mixprior.campaign import Campaign
from experiments.datakit.mixprior.data import Swarm

FIT_STEP_LIMIT = 10_000
FIT_PROGRESS_INTERVAL = 250
logger = logging.getLogger(__name__)


class ModelMetadata(TypedDict):
    kind: str
    device: str
    details: dict[str, Any]


class FeatureMap(Protocol):
    def __call__(self, swarm: Swarm, weights: np.ndarray) -> np.ndarray: ...


class PredictiveMoments(NamedTuple):
    mean: np.ndarray
    latent_variance: np.ndarray


class MixturePredictor(Protocol):
    def predict(self, swarm: Swarm, weights: np.ndarray) -> PredictiveMoments: ...


class BotorchMixturePredictor(MixturePredictor, Protocol):
    @property
    def model_metadata(self) -> ModelMetadata: ...

    @property
    def botorch_model(self) -> SingleTaskGP: ...

    def candidate_tensor(self, swarm: Swarm, weights: np.ndarray) -> torch.Tensor: ...


class SwarmTrainingRows(NamedTuple):
    swarm_id: str
    features: np.ndarray
    objective_values: np.ndarray
    objective_variances: np.ndarray


class SwarmOutcomeScale(NamedTuple):
    mean: float
    scale: float


class TrainingData(NamedTuple):
    features: np.ndarray
    standardized_objective_values: np.ndarray
    standardized_objective_variances: np.ndarray
    swarm_indices: dict[str, int]
    observation_counts: dict[str, int]
    outcome_scales: dict[str, SwarmOutcomeScale]


class MapInitialization(NamedTuple):
    name: str
    seed: int
    initialize: Callable[[SingleTaskGP, int], None]


class MapFitSummary(NamedTuple):
    name: str
    seed: int
    marginal_log_likelihood: float
    optimizer_objective: float
    optimizer_status: str
    fit_steps: int
    elapsed: float
    selected: bool


class MapFit(NamedTuple):
    model: SingleTaskGP
    restarts: tuple[MapFitSummary, ...]
    elapsed: float


@dataclass(frozen=True)
class FittedSwarmGP:
    _model: SingleTaskGP
    feature_map: FeatureMap
    swarm_indices: dict[str, int]
    outcome_scales: dict[str, SwarmOutcomeScale]
    model_metadata: ModelMetadata

    @property
    def device(self) -> torch.device:
        return self._model.train_inputs[0].device

    @property
    def botorch_model(self) -> SingleTaskGP:
        return self._model

    def candidate_tensor(self, swarm: Swarm, weights: np.ndarray) -> torch.Tensor:
        if swarm.swarm_id not in self.swarm_indices:
            raise ValueError(f"Swarm {swarm.swarm_id!r} was not included when this model was fit")
        features = self.feature_map(swarm, weights)
        swarm_column = np.full((len(features), 1), self.swarm_indices[swarm.swarm_id], dtype=np.float64)
        return torch.as_tensor(
            np.concatenate([features, swarm_column], axis=1),
            dtype=torch.double,
            device=self.device,
        )

    def predict(self, swarm: Swarm, weights: np.ndarray) -> PredictiveMoments:
        with torch.no_grad():
            posterior = self._model.posterior(self.candidate_tensor(swarm, weights))
        scale = self.outcome_scales[swarm.swarm_id]
        return PredictiveMoments(
            mean=scale.mean + scale.scale * posterior.mean.detach().cpu().numpy().reshape(-1),
            latent_variance=scale.scale**2 * posterior.variance.detach().cpu().numpy().reshape(-1),
        )


class SameSwarmKernel(Kernel):
    """Unit covariance for rows from the same swarm and zero otherwise."""

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: object,
    ) -> torch.Tensor:
        if last_dim_is_batch:
            raise ValueError("Same-swarm covariance does not batch over input dimensions")
        if diag:
            return (x1[..., 0] == x2[..., 0]).to(x1.dtype)
        return (x1[..., :, None, 0] == x2[..., None, :, 0]).to(x1.dtype)


def lognormal_prior_with_mode(
    value: torch.Tensor | float,
    log_sd: float,
    like: torch.Tensor,
) -> LogNormalPrior:
    mode = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    return LogNormalPrior(torch.log(mode) + log_sd**2, torch.full_like(mode, log_sd))


def prepare_training_data(campaign: Campaign, feature_map: FeatureMap) -> TrainingData:
    rows = []
    for swarm in (campaign.target, *campaign.sources):
        observations = campaign.objective.observations(swarm)
        rows.append(
            SwarmTrainingRows(
                swarm_id=swarm.swarm_id,
                features=feature_map(swarm, swarm.data.weights),
                objective_values=observations.values,
                objective_variances=observations.variances,
            )
        )
    return assemble_training_data(rows)


def assemble_training_data(rows: list[SwarmTrainingRows]) -> TrainingData:
    """Standardize outcomes within each swarm and append its integer index."""
    swarm_indices = {row.swarm_id: index for index, row in enumerate(rows)}
    if len(swarm_indices) != len(rows):
        raise ValueError("Swarm training row IDs must be unique")
    train_X = []
    train_Y = []
    train_Yvar = []
    observation_counts = {}
    outcome_scales = {}
    for row in rows:
        values = np.asarray(row.objective_values, dtype=np.float64)
        variances = np.asarray(row.objective_variances, dtype=np.float64)
        if values.ndim != 1 or variances.ndim != 1:
            raise ValueError(f"Objectives for swarm {row.swarm_id!r} must be vectors")
        if len(values) < 2:
            raise ValueError(f"Swarm {row.swarm_id!r} needs at least two observations")
        if values.shape != variances.shape or len(row.features) != len(values):
            raise ValueError(f"Training arrays for swarm {row.swarm_id!r} do not align")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(variances)) or np.any(variances <= 0):
            raise ValueError(f"Swarm {row.swarm_id!r} has invalid objective values or variances")
        scale = float(np.std(values))
        if scale <= 0:
            raise ValueError(f"Swarm {row.swarm_id!r} needs non-constant objective values")
        mean = float(np.mean(values))
        index = swarm_indices[row.swarm_id]
        swarm_column = np.full((len(values), 1), index, dtype=np.float64)
        train_X.append(np.concatenate([row.features, swarm_column], axis=1))
        train_Y.append(((values - mean) / scale)[:, None])
        train_Yvar.append((variances / scale**2)[:, None])
        observation_counts[row.swarm_id] = len(values)
        outcome_scales[row.swarm_id] = SwarmOutcomeScale(mean, scale)

    return TrainingData(
        features=np.concatenate(train_X),
        standardized_objective_values=np.concatenate(train_Y),
        standardized_objective_variances=np.concatenate(train_Yvar),
        swarm_indices=swarm_indices,
        observation_counts=observation_counts,
        outcome_scales=outcome_scales,
    )


def draw_parameters_from_priors(model: SingleTaskGP, seed: int) -> None:
    torch.manual_seed(seed)
    for _, module, prior, _, setting_closure in model.named_priors():
        if setting_closure is None:
            raise ValueError("Every GP prior must define how to initialize its parameter")
        setting_closure(module, prior.sample())


def fit_map_restarts(
    build_model: Callable[[], SingleTaskGP],
    initializations: tuple[MapInitialization, ...],
) -> MapFit:
    """Fit declared MAP starts and retain the model with the highest MLL."""
    if not initializations:
        raise ValueError("At least one MAP initialization is required")
    started_all = time.monotonic()
    summaries = []
    best_model = None
    best_mll = -float("inf")
    best_index = -1
    for index, initialization in enumerate(initializations):
        model = build_model()
        initialization.initialize(model, initialization.seed)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        started = time.monotonic()
        result = fit_gpytorch_mll_torch(mll, step_limit=FIT_STEP_LIMIT, callback=_log_fit_progress)
        if result.step >= FIT_STEP_LIMIT:
            raise RuntimeError(f"GP MAP fit did not converge after {FIT_STEP_LIMIT} steps")
        elapsed = time.monotonic() - started
        model.train()
        mll.train()
        with torch.no_grad():
            value = float(mll(model(*model.train_inputs), model.train_targets).detach().cpu())
        model.eval()
        summaries.append(
            MapFitSummary(
                initialization.name,
                initialization.seed,
                value,
                float(result.fval),
                result.status.name,
                result.step,
                elapsed,
                False,
            )
        )
        logger.info(
            "GP restart %s seed %d completed in %.1fs after %d steps with MLL %.6f",
            initialization.name,
            initialization.seed,
            elapsed,
            result.step,
            value,
        )
        if value > best_mll:
            best_model = model
            best_mll = value
            best_index = index
    if best_model is None:
        raise AssertionError("MAP fitting produced no model")
    summaries[best_index] = summaries[best_index]._replace(selected=True)
    return MapFit(best_model, tuple(summaries), time.monotonic() - started_all)


def _log_fit_progress(_parameters: dict[str, torch.Tensor], result: OptimizationResult) -> None:
    if result.step % FIT_PROGRESS_INTERVAL == 0:
        logger.info("GP fit step %s/%s: objective %.6f", result.step, FIT_STEP_LIMIT, result.fval)
