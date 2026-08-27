# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare transfer-GP training data and fit compiled JAX objectives."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, NamedTuple, Protocol, TypedDict

import jax
import numpy as np
from jax.scipy.optimize import minimize

from experiments.datakit.mixprior.campaign import Campaign
from experiments.datakit.mixprior.data import Swarm

FIT_STEP_LIMIT = 1_000
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


class JaxMixturePredictor(MixturePredictor, Protocol):
    @property
    def model_metadata(self) -> ModelMetadata: ...

    def noisy_expected_improvement(
        self,
        swarm: Swarm,
        weights: np.ndarray,
        reference_weights: np.ndarray,
        *,
        sample_count: int,
        seed: int,
    ) -> np.ndarray: ...


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
    parameters: np.ndarray


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
    parameters: np.ndarray
    restarts: tuple[MapFitSummary, ...]
    elapsed: float


def default_device() -> jax.Device:
    devices = jax.devices()
    return next((device for device in devices if device.platform == "gpu"), devices[0])


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
        train_Y.append((values - mean) / scale)
        train_Yvar.append(variances / scale**2)
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


@jax.enable_x64()
def fit_map_restarts(
    objective: Callable[[jax.Array], jax.Array],
    initializations: tuple[MapInitialization, ...],
    device: jax.Device,
) -> MapFit:
    """Fit fixed MAP starts with JAX BFGS and retain the lowest objective."""
    if not initializations:
        raise ValueError("At least one MAP initialization is required")
    compiled_objective = jax.jit(objective)
    started_all = time.monotonic()
    summaries = []
    candidates = []
    for initialization in initializations:
        started = time.monotonic()
        result = minimize(
            compiled_objective,
            jax.device_put(initialization.parameters, device),
            method="BFGS",
            tol=1e-7,
            options={"maxiter": FIT_STEP_LIMIT},
        )
        objective_value = float(result.fun)
        elapsed = time.monotonic() - started
        if np.isfinite(objective_value):
            candidates.append((objective_value, np.asarray(result.x)))
        summaries.append(
            MapFitSummary(
                initialization.name,
                initialization.seed,
                -objective_value,
                objective_value,
                f"status={int(result.status)}, success={bool(result.success)}",
                int(result.nit),
                elapsed,
                False,
            )
        )
        logger.info(
            "GP restart %s seed %d completed in %.1fs after %d steps with MLL %.6f",
            initialization.name,
            initialization.seed,
            elapsed,
            int(result.nit),
            -objective_value,
        )
    if not candidates:
        raise RuntimeError("Every GP MAP restart produced a non-finite objective")
    best_objective, best_parameters = min(candidates, key=lambda candidate: candidate[0])
    best_index = next(index for index, summary in enumerate(summaries) if summary.optimizer_objective == best_objective)
    summaries[best_index] = summaries[best_index]._replace(selected=True)
    return MapFit(best_parameters, tuple(summaries), time.monotonic() - started_all)
