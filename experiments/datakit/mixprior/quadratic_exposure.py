# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quadratic exposure prior and phase-linked content covariance."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import Positive
from gpytorch.kernels import AdditiveKernel, Kernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean, Mean
from gpytorch.priors import LogNormalPrior, NormalPrior
from linear_operator.operators import LinearOperator

from experiments.datakit.mixprior.campaign import Campaign
from experiments.datakit.mixprior.data import PHASE_COUNT, Swarm
from experiments.datakit.mixprior.surrogate import (
    FittedSwarmGP,
    MapFit,
    MapInitialization,
    ModelMetadata,
    SameSwarmKernel,
    TrainingData,
    draw_parameters_from_priors,
    fit_map_restarts,
    lognormal_prior_with_mode,
    prepare_training_data,
)

MATERN_NU = 2.5
LENGTHSCALE_PRIOR_LOG_SD = 1.0
HARM_CURVATURE_INITIAL = (0.01, 0.015)
RESIDUAL_OUTPUTSCALE_INITIAL = 0.25


def keep_initialization(_model: SingleTaskGP, _seed: int) -> None:
    pass


class QuadraticExposureLayout(NamedTuple):
    phase_exposure_content: tuple[slice, ...]
    quadratic_exposure: slice
    feature_count: int


def quadratic_exposure_layout(content_dim: int) -> QuadraticExposureLayout:
    phase_width = 2 * content_dim
    phase_exposure_content = tuple(slice(phase * phase_width, (phase + 1) * phase_width) for phase in range(PHASE_COUNT))
    offset = PHASE_COUNT * phase_width
    quadratic_exposure = slice(offset, offset + PHASE_COUNT)
    return QuadraticExposureLayout(phase_exposure_content, quadratic_exposure, quadratic_exposure.stop)


def quadratic_exposure_features(swarm: Swarm, weights: np.ndarray) -> np.ndarray:
    """Map phase mixtures to content-weighted first and second epoch moments."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 3 or weights.shape[1] != PHASE_COUNT:
        raise ValueError(f"Weights must have candidate, {PHASE_COUNT} phase, and component axes")
    if weights.shape[2] != len(swarm.data.available_tokens):
        raise ValueError("Mixture weights and swarm components do not align")
    if np.any(weights < 0) or not np.allclose(weights.sum(axis=2), 1.0):
        raise ValueError("Every phase must be a non-negative simplex vector")

    token_share = swarm.data.available_tokens / swarm.data.available_tokens.sum()
    exposure = weights * swarm.exposure_multipliers[None]
    first_moment = token_share[None, None, :] * exposure
    second_moment = first_moment * exposure
    phase_content = np.concatenate(
        [first_moment @ swarm.content_matrix, second_moment @ swarm.content_matrix],
        axis=2,
    )
    return np.concatenate(
        [
            phase_content.reshape(len(weights), -1),
            second_moment.sum(axis=2),
        ],
        axis=1,
    )


class QuadraticExposurePenaltyMean(Mean):
    """Soft prior against concentrating a phase on repeatedly sampled data."""

    def __init__(self, layout: QuadraticExposureLayout, like: torch.Tensor) -> None:
        super().__init__()
        self.layout = layout
        self.constant = ConstantMean()
        self.register_parameter("raw_harm_curvature", torch.nn.Parameter(like.new_zeros(PHASE_COUNT)))
        self.register_constraint("raw_harm_curvature", Positive())
        self.harm_curvature = torch.as_tensor(HARM_CURVATURE_INITIAL, dtype=like.dtype, device=like.device)
        self.register_prior(
            "harm_curvature_prior",
            lognormal_prior_with_mode(torch.as_tensor(HARM_CURVATURE_INITIAL), 1.5, like),
            lambda module: module.harm_curvature,
            lambda module, value: setattr(module, "harm_curvature", value),
        )

    @property
    def harm_curvature(self) -> torch.Tensor:
        return self.raw_harm_curvature_constraint.transform(self.raw_harm_curvature)

    @harm_curvature.setter
    def harm_curvature(self, value: torch.Tensor | float) -> None:
        constrained = torch.as_tensor(
            value,
            dtype=self.raw_harm_curvature.dtype,
            device=self.raw_harm_curvature.device,
        )
        self.initialize(raw_harm_curvature=self.raw_harm_curvature_constraint.inverse_transform(constrained))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quadratic = x[..., self.layout.quadratic_exposure]
        return self.constant.forward(x) - (self.harm_curvature * quadratic).sum(dim=-1)


def draw_quadratic_initialization(model: SingleTaskGP, seed: int) -> None:
    draw_parameters_from_priors(model, seed)
    mean = model.mean_module
    if not isinstance(mean, QuadraticExposurePenaltyMean):
        raise TypeError("Quadratic exposure GP must use QuadraticExposurePenaltyMean")
    mean.constant.initialize(constant=torch.randn_like(mean.constant.constant))


MAP_INITIALIZATIONS = (
    MapInitialization("prior_mode", 0, keep_initialization),
    MapInitialization("prior_draw", 11, draw_quadratic_initialization),
    MapInitialization("prior_draw", 22, draw_quadratic_initialization),
)


class PhaseContentKernel(Kernel):
    """Share a content response across phases with learned phase covariance."""

    def __init__(
        self,
        phase_exposure_content: tuple[slice, ...],
        initial_lengthscale: float,
        like: torch.Tensor,
    ) -> None:
        super().__init__()
        self.phase_exposure_content = phase_exposure_content
        phase_count = len(phase_exposure_content)
        self.content_kernel = MaternKernel(
            nu=MATERN_NU,
            lengthscale_prior=LogNormalPrior(
                like.new_tensor(math.log(initial_lengthscale)),
                like.new_tensor(LENGTHSCALE_PRIOR_LOG_SD),
            ),
        )
        self.content_kernel.lengthscale = initial_lengthscale
        self.phase_factor = torch.nn.Parameter(like.new_ones(phase_count, 1))
        self.register_prior(
            "phase_factor_prior",
            NormalPrior(like.new_tensor(1.0), like.new_tensor(1.0)),
            "phase_factor",
        )
        self.register_parameter("raw_phase_diagonal", torch.nn.Parameter(like.new_zeros(phase_count)))
        self.register_constraint("raw_phase_diagonal", Positive())
        self.phase_diagonal = 0.25

    @property
    def phase_diagonal(self) -> torch.Tensor:
        return self.raw_phase_diagonal_constraint.transform(self.raw_phase_diagonal)

    @phase_diagonal.setter
    def phase_diagonal(self, value: torch.Tensor | float) -> None:
        constrained = torch.as_tensor(
            value,
            dtype=self.raw_phase_diagonal.dtype,
            device=self.raw_phase_diagonal.device,
        )
        self.initialize(raw_phase_diagonal=self.raw_phase_diagonal_constraint.inverse_transform(constrained))

    @property
    def phase_covariance(self) -> torch.Tensor:
        return self.phase_factor @ self.phase_factor.transpose(-1, -2) + torch.diag(self.phase_diagonal)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: object,
    ) -> LinearOperator | torch.Tensor:
        if last_dim_is_batch:
            raise ValueError("Phase-content covariance does not batch over input dimensions")
        result = None
        for first, first_slice in enumerate(self.phase_exposure_content):
            for second, second_slice in enumerate(self.phase_exposure_content):
                value = self.phase_covariance[first, second] * self.content_kernel(
                    x1[..., first_slice],
                    x2[..., second_slice],
                    diag=diag,
                    **params,
                )
                result = value if result is None else result + value
        if result is None:
            raise AssertionError("Phase-content covariance requires at least one phase")
        return result


def moment_lengthscale(features: np.ndarray, layout: QuadraticExposureLayout) -> float:
    phase_features = np.concatenate([features[:, phase] for phase in layout.phase_exposure_content])
    squared_norm = np.square(phase_features).sum(axis=1)
    squared_distance = squared_norm[:, None] + squared_norm[None, :] - 2 * phase_features @ phase_features.T
    upper = squared_distance[np.triu_indices(len(squared_distance), k=1)]
    positive = upper[upper > np.finfo(np.float64).eps]
    if not len(positive):
        raise ValueError("Kernel calibration needs two distinct exposure profiles")
    return float(np.sqrt(np.median(positive)))


def quadratic_exposure_covariance(
    layout: QuadraticExposureLayout,
    initial_lengthscale: float,
    like: torch.Tensor,
) -> Kernel:
    shared = PhaseContentKernel(layout.phase_exposure_content, initial_lengthscale, like)
    phase_dims = tuple(range(layout.phase_exposure_content[0].start, layout.phase_exposure_content[-1].stop))
    residual_response = MaternKernel(
        nu=MATERN_NU,
        active_dims=phase_dims,
        lengthscale_prior=lognormal_prior_with_mode(initial_lengthscale, LENGTHSCALE_PRIOR_LOG_SD, like),
    )
    residual_response.lengthscale = initial_lengthscale
    same_swarm = SameSwarmKernel(active_dims=(layout.feature_count,)).to(dtype=like.dtype, device=like.device)
    residual = ScaleKernel(
        residual_response * same_swarm,
        outputscale_prior=lognormal_prior_with_mode(RESIDUAL_OUTPUTSCALE_INITIAL, 1.0, like),
    )
    residual.outputscale = RESIDUAL_OUTPUTSCALE_INITIAL
    return (shared + residual).to(dtype=like.dtype, device=like.device)


def quadratic_exposure_gp(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    train_Yvar: torch.Tensor,
    *,
    content_dim: int,
    num_swarms: int,
    initial_lengthscale: float,
) -> SingleTaskGP:
    layout = quadratic_exposure_layout(content_dim)
    if train_X.shape[-1] != layout.feature_count + 1:
        raise ValueError(f"Expected {layout.feature_count + 1} GP input columns, got {train_X.shape[-1]}")
    swarm_indices = train_X[:, layout.feature_count]
    if not torch.all(swarm_indices == swarm_indices.round()):
        raise ValueError("Swarm indices must be integers")
    if torch.any(swarm_indices < 0) or torch.any(swarm_indices >= num_swarms):
        raise ValueError("Swarm indices are outside the configured range")
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        mean_module=QuadraticExposurePenaltyMean(layout, train_X),
        covar_module=quadratic_exposure_covariance(layout, initial_lengthscale, train_X),
        outcome_transform=Standardize(m=1),
    )
    return model.to(dtype=train_X.dtype, device=train_X.device)


def quadratic_model_metadata(
    campaign: Campaign,
    training: TrainingData,
    model: SingleTaskGP,
    fit: MapFit,
) -> ModelMetadata:
    layout = quadratic_exposure_layout(campaign.target.content_matrix.shape[1])
    initial_lengthscale = moment_lengthscale(training.features[:, : layout.feature_count], layout)
    mean = model.mean_module
    covariance = model.covar_module
    if not isinstance(mean, QuadraticExposurePenaltyMean):
        raise TypeError("Quadratic exposure GP must use QuadraticExposurePenaltyMean")
    if not isinstance(covariance, AdditiveKernel) or not isinstance(covariance.kernels[0], PhaseContentKernel):
        raise TypeError("Quadratic exposure GP must use shared phase and same-swarm covariance")
    shared = covariance.kernels[0]
    return {
        "kind": "quadratic_exposure_transfer_gp",
        "device": str(model.train_inputs[0].device),
        "details": {
            "mean": "learned phase-specific quadratic penalty on token-mass-weighted epochs",
            "harm_curvature_initial": list(HARM_CURVATURE_INITIAL),
            "harm_curvature": mean.harm_curvature.detach().cpu().tolist(),
            "covariance": "phase-linked content response plus same-swarm Matern-5/2 residual",
            "initial_lengthscale": initial_lengthscale,
            "content_lengthscale": float(shared.content_kernel.lengthscale.detach().cpu().reshape(-1)[0]),
            "phase_covariance": shared.phase_covariance.detach().cpu().tolist(),
            "outcome_transform": "per-swarm affine standardization followed by BoTorch Standardize",
            "hyperparameter_inference": "highest-MLL converged MAP fit from three fixed starts",
            "map_restarts": [summary._asdict() for summary in fit.restarts],
            "observation_counts": training.observation_counts,
            "fit_seconds": fit.elapsed,
        },
    }


def fit_quadratic_exposure_model(campaign: Campaign, device: torch.device) -> FittedSwarmGP:
    training = prepare_training_data(campaign, quadratic_exposure_features)
    X = torch.as_tensor(training.features, dtype=torch.double, device=device)
    Y = torch.as_tensor(training.standardized_objective_values, dtype=torch.double, device=device)
    Yvar = torch.as_tensor(training.standardized_objective_variances, dtype=torch.double, device=device)
    layout = quadratic_exposure_layout(campaign.target.content_matrix.shape[1])
    initial_lengthscale = moment_lengthscale(training.features[:, : layout.feature_count], layout)
    fit = fit_map_restarts(
        lambda: quadratic_exposure_gp(
            X,
            Y,
            Yvar,
            content_dim=campaign.target.content_matrix.shape[1],
            num_swarms=len(training.swarm_indices),
            initial_lengthscale=initial_lengthscale,
        ),
        MAP_INITIALIZATIONS,
    )
    return FittedSwarmGP(
        _model=fit.model,
        feature_map=quadratic_exposure_features,
        swarm_indices=training.swarm_indices,
        outcome_scales=training.outcome_scales,
        model_metadata=quadratic_model_metadata(campaign, training, fit.model, fit),
    )
