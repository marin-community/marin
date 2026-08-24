# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit the shared-content transfer GP."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.generators.torch.botorch_modular.surrogate import Surrogate, SurrogateSpec
from ax.generators.torch.botorch_modular.utils import ModelConfig
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.datasets import SupervisedDataset
from gpytorch.kernels import IndexKernel, RBFKernel, ScaleKernel

from experiments.datakit.mixprior.campaign import Campaign
from experiments.datakit.mixprior.objective import objective_observations

OBJECTIVE_NAME = "negative_hinge_loss"
RAV_GAMMA_FACTOR = 0.25
logger = logging.getLogger(__name__)


def curriculum_features(
    weights: np.ndarray,
    component_content: np.ndarray,
    phase_token_fractions: np.ndarray,
) -> np.ndarray:
    """Map phase mixtures to square-root content distributions."""
    weights = np.asarray(weights, dtype=np.float64)
    component_content = np.asarray(component_content, dtype=np.float64)
    phase_token_fractions = np.asarray(phase_token_fractions, dtype=np.float64)
    if weights.ndim != 3 or component_content.ndim != 2:
        raise ValueError("Weights and component content must be rank-3 and rank-2")
    if weights.shape[2] != component_content.shape[0]:
        raise ValueError("Mixture weights and component content do not align")
    if phase_token_fractions.shape != (weights.shape[1],):
        raise ValueError("Phase token fractions must match the curriculum phases")
    if np.any(weights < 0) or np.any(component_content < 0):
        raise ValueError("Mixture weights and component content must be non-negative")
    if not np.allclose(weights.sum(axis=2), 1.0):
        raise ValueError("Every phase must be a simplex vector")
    if not np.allclose(component_content.sum(axis=1), 1.0):
        raise ValueError("Every component-content row must be a distribution")
    if np.any(phase_token_fractions <= 0) or not np.isclose(phase_token_fractions.sum(), 1.0):
        raise ValueError("Phase token fractions must be a positive simplex vector")

    phase_content = weights @ component_content
    rooted = np.sqrt(phase_content) * np.sqrt(phase_token_fractions)[None, :, None]
    return rooted.reshape(len(weights), -1)


def squared_hellinger(features: np.ndarray) -> np.ndarray:
    """Return pairwise phase-weighted squared Hellinger distances."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("Curriculum features must be a matrix")
    squared_norm = np.sum(np.square(features), axis=1)
    squared_distance = squared_norm[:, None] + squared_norm[None, :] - 2 * features @ features.T
    return np.maximum(squared_distance, 0.0) / 2


def rav_lengthscale(reference_features: np.ndarray, gamma_factor: float = RAV_GAMMA_FACTOR) -> float:
    """Calibrate Rav's fixed RBF lengthscale from observed curricula."""
    if gamma_factor <= 0:
        raise ValueError("Gamma factor must be positive")
    distances = squared_hellinger(reference_features)
    upper = distances[np.triu_indices(len(distances), k=1)]
    positive = upper[upper > np.finfo(np.float64).eps]
    if not len(positive):
        raise ValueError("Kernel calibration needs two distinct curricula")
    return float(np.sqrt(np.median(positive) / gamma_factor))


class SharedSwarmHellingerGP(SingleTaskGP):
    """A shared objective function plus a swarm-specific deviation.

    The final input column is an integer swarm index. All preceding columns are
    the two-phase square-root content representation.
    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        train_Yvar: torch.Tensor,
        num_swarms: int,
        lengthscale: float,
        outcome_transform=None,
    ) -> None:
        swarm_indices = train_X[:, -1]
        if not torch.all(swarm_indices == swarm_indices.round()):
            raise ValueError("Swarm indices must be integers")
        if torch.any(swarm_indices < 0) or torch.any(swarm_indices >= num_swarms):
            raise ValueError("Swarm indices are outside the configured range")
        content_dims = tuple(range(train_X.shape[-1] - 1))
        swarm_dim = train_X.shape[-1] - 1
        shared = ScaleKernel(
            rav_rbf_kernel(
                lengthscale,
                device=train_X.device,
                active_dims=content_dims,
            )
        )
        residual_content = rav_rbf_kernel(
            lengthscale,
            device=train_X.device,
            active_dims=content_dims,
        )
        residual = residual_content * IndexKernel(
            num_tasks=num_swarms,
            rank=0,
            active_dims=(swarm_dim,),
        )
        covariance = (shared + residual).to(dtype=train_X.dtype, device=train_X.device)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            covar_module=covariance,
            outcome_transform=outcome_transform,
        )

    @classmethod
    def construct_inputs(  # pyrefly: ignore [bad-override]
        cls,
        training_data: SupervisedDataset,
        num_swarms: int,
        lengthscale: float,
        categorical_features: list[int] | None = None,
    ) -> dict:
        expected = [training_data.X.shape[-1] - 1]
        if categorical_features != expected:
            raise ValueError("The final GP input must be the categorical swarm index")
        return {
            **super().construct_inputs(training_data),
            "num_swarms": num_swarms,
            "lengthscale": lengthscale,
        }


class SwarmTrainingRows(NamedTuple):
    swarm_id: str
    features: np.ndarray
    objective_values: np.ndarray
    objective_variances: np.ndarray


class TransferTrainingData(NamedTuple):
    features: np.ndarray
    objective_values: np.ndarray
    objective_variances: np.ndarray
    feature_names: list[str]
    swarm_indices: dict[str, int]
    target_swarm_index: int
    observation_counts: dict[str, int]


class HellingerTransferData(NamedTuple):
    training: TransferTrainingData
    kernel_reference_features: np.ndarray


class PredictiveMoments(NamedTuple):
    mean: np.ndarray
    variance: np.ndarray


class TransferPredictor(Protocol):
    """Prediction boundary shared by acquisition and interpretation."""

    target_swarm_index: int

    def predict(self, features: np.ndarray, swarm_index: int) -> PredictiveMoments: ...


@dataclass(frozen=True)
class AxTransferFit:
    _surrogate: Surrogate
    swarm_indices: dict[str, int]
    target_swarm_index: int
    observation_counts: dict[str, int]
    lengthscale: float
    elapsed: float

    @property
    def device(self) -> torch.device:
        return self._surrogate.device

    def predict(self, features: np.ndarray, swarm_index: int) -> PredictiveMoments:
        candidates = self._candidate_tensor(features, swarm_index)
        with torch.no_grad():
            mean, covariance = self._surrogate.predict(candidates)
        marginal_variance = np.diagonal(covariance.detach().cpu().numpy(), axis1=-2, axis2=-1)
        return PredictiveMoments(
            mean=mean.detach().cpu().numpy().reshape(-1),
            variance=marginal_variance.reshape(-1),
        )

    def model_state(self) -> dict:
        return {
            "model_class": type(self._surrogate.model).__name__,
            "state_dict": self._surrogate.model.state_dict(),
        }

    def _candidate_tensor(self, features: np.ndarray, swarm_index: int) -> torch.Tensor:
        return torch.as_tensor(
            np.concatenate(
                [
                    features,
                    np.full((len(features), 1), swarm_index, dtype=np.float64),
                ],
                axis=1,
            ),
            dtype=torch.double,
            device=self.device,
        )


def rav_rbf_kernel(
    lengthscale: float,
    *,
    device: torch.device | None = None,
    active_dims: tuple[int, ...] | None = None,
) -> RBFKernel:
    kernel = RBFKernel(active_dims=active_dims).to(dtype=torch.double)
    if device is not None:
        kernel = kernel.to(device=device)
    kernel.raw_lengthscale.requires_grad_(False)
    kernel.lengthscale = lengthscale
    return kernel


def prepare_hellinger_transfer_data(campaign: Campaign) -> HellingerTransferData:
    """Project campaign observations into Rav's shared Hellinger space."""
    swarms = [campaign.target, *campaign.sources]
    if len({swarm.content_basis_id for swarm in swarms}) != 1:
        raise ValueError("All campaign swarms must share one content basis")
    rows = []
    features_by_swarm = {}
    for swarm in swarms:
        objective_values, objective_variances = objective_observations(
            swarm,
            campaign.objective,
            campaign.objective_metrics,
            campaign.observation_sd,
        )
        features = curriculum_features(
            swarm.data.weights,
            swarm.content_matrix,
            swarm.phase_budgets / swarm.phase_budgets.sum(),
        )
        features_by_swarm[swarm.swarm_id] = features
        rows.append(
            SwarmTrainingRows(
                swarm_id=swarm.swarm_id,
                features=features,
                objective_values=objective_values,
                objective_variances=objective_variances,
            )
        )
    return HellingerTransferData(
        training=assemble_transfer_data(rows, target_swarm=campaign.target.swarm_id),
        kernel_reference_features=features_by_swarm[campaign.kernel_reference_swarm],
    )


def assemble_transfer_data(rows: list[SwarmTrainingRows], *, target_swarm: str) -> TransferTrainingData:
    """Combine already-computed per-swarm features into GP training arrays."""
    swarm_indices = {row.swarm_id: index for index, row in enumerate(rows)}
    if len(swarm_indices) != len(rows):
        raise ValueError("Swarm training row IDs must be unique")
    train_X = []
    train_Y = []
    train_Yvar = []
    observation_counts = {}
    for row in rows:
        swarm_index = swarm_indices[row.swarm_id]
        train_X.append(
            np.concatenate(
                [
                    row.features,
                    np.full((len(row.objective_values), 1), swarm_index, dtype=np.float64),
                ],
                axis=1,
            )
        )
        train_Y.append(row.objective_values[:, None])
        train_Yvar.append(row.objective_variances[:, None])
        observation_counts[row.swarm_id] = len(row.objective_values)

    X = np.concatenate(train_X)
    return TransferTrainingData(
        features=X,
        objective_values=np.concatenate(train_Y),
        objective_variances=np.concatenate(train_Yvar),
        feature_names=[f"feature_{index}" for index in range(X.shape[1] - 1)] + ["swarm"],
        swarm_indices=swarm_indices,
        target_swarm_index=swarm_indices[target_swarm],
        observation_counts=observation_counts,
    )


def fit_additive_hellinger_model(data: HellingerTransferData, device: torch.device) -> AxTransferFit:
    training = data.training
    X = torch.as_tensor(training.features, dtype=torch.double, device=device)
    Y = torch.as_tensor(training.objective_values, dtype=torch.double, device=device)
    Yvar = torch.as_tensor(training.objective_variances, dtype=torch.double, device=device)
    lengthscale = rav_lengthscale(data.kernel_reference_features)
    logger.info(
        "Fitting %s observations across %s swarms on %s",
        len(X),
        len(training.swarm_indices),
        device,
    )
    dataset = SupervisedDataset(
        X=X,
        Y=Y,
        Yvar=Yvar,
        feature_names=training.feature_names,
        outcome_names=[OBJECTIVE_NAME],
    )
    search_space_digest = SearchSpaceDigest(
        feature_names=training.feature_names,
        bounds=[(0.0, 1.0)] * (X.shape[1] - 1) + [(0.0, float(len(training.swarm_indices) - 1))],
        categorical_features=[X.shape[1] - 1],
        discrete_choices={X.shape[1] - 1: list(range(len(training.swarm_indices)))},
        task_features=[X.shape[1] - 1],
        target_values={X.shape[1] - 1: training.target_swarm_index},
    )
    surrogate = Surrogate(
        SurrogateSpec(
            model_configs=[
                ModelConfig(
                    botorch_model_class=SharedSwarmHellingerGP,
                    model_options={
                        "num_swarms": len(training.swarm_indices),
                        "lengthscale": lengthscale,
                    },
                    input_transform_classes=None,
                    outcome_transform_classes=[Standardize],
                )
            ]
        )
    )
    started = time.monotonic()
    surrogate.fit([dataset], search_space_digest)
    elapsed = time.monotonic() - started
    logger.info("Transfer GP fit completed in %.1f seconds", elapsed)
    return AxTransferFit(
        _surrogate=surrogate,
        swarm_indices=training.swarm_indices,
        target_swarm_index=training.target_swarm_index,
        observation_counts=training.observation_counts,
        lengthscale=lengthscale,
        elapsed=elapsed,
    )
