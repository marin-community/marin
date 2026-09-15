# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compiled JAX GP with quadratic epoch exposure and phase-linked content."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_solve, solve_triangular

from experiments.datakit.mixprior.campaign import Campaign
from experiments.datakit.mixprior.data import PHASE_COUNT, Swarm
from experiments.datakit.mixprior.surrogate import (
    MapFit,
    MapInitialization,
    ModelMetadata,
    PredictiveMoments,
    SwarmOutcomeScale,
    TrainingData,
    fit_map_restarts,
    prepare_training_data,
)

HARM_CURVATURE_INITIAL = (0.01, 0.015)
PHASE_DIAGONAL_INITIAL = 0.25
RESIDUAL_OUTPUTSCALE_INITIAL = 0.25
LENGTHSCALE_PRIOR_LOG_SD = 1.0
HARM_CURVATURE_PRIOR_LOG_SD = 1.5
GP_JITTER = 1e-6
POSITIVE_FLOOR = 1e-8

CONSTANT_INDEX = 0
HARM_CURVATURE_SLICE = slice(1, 3)
CONTENT_LENGTHSCALE_INDEX = 3
PHASE_FACTOR_SLICE = slice(4, 6)
PHASE_DIAGONAL_SLICE = slice(6, 8)
RESIDUAL_LENGTHSCALE_INDEX = 8
RESIDUAL_OUTPUTSCALE_INDEX = 9
PARAMETER_COUNT = 10
logger = logging.getLogger(__name__)


class QuadraticExposureLayout(NamedTuple):
    phase_exposure_content: tuple[slice, ...]
    quadratic_exposure: slice
    feature_count: int


class QuadraticParameters(NamedTuple):
    constant: jax.Array
    harm_curvature: jax.Array
    content_lengthscale: jax.Array
    phase_factor: jax.Array
    phase_diagonal: jax.Array
    residual_lengthscale: jax.Array
    residual_outputscale: jax.Array


class TrainingConditioning(NamedTuple):
    cholesky: jax.Array
    residual: jax.Array
    alpha: jax.Array


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


def _positive(raw: jax.Array) -> jax.Array:
    return jax.nn.softplus(raw) + POSITIVE_FLOOR


def _inverse_positive(value: np.ndarray | float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64) - POSITIVE_FLOOR
    return np.log(np.expm1(value))


def quadratic_parameter_values(parameters: jax.Array) -> QuadraticParameters:
    return QuadraticParameters(
        parameters[CONSTANT_INDEX],
        _positive(parameters[HARM_CURVATURE_SLICE]),
        _positive(parameters[CONTENT_LENGTHSCALE_INDEX]),
        parameters[PHASE_FACTOR_SLICE],
        _positive(parameters[PHASE_DIAGONAL_SLICE]),
        _positive(parameters[RESIDUAL_LENGTHSCALE_INDEX]),
        _positive(parameters[RESIDUAL_OUTPUTSCALE_INDEX]),
    )


def initial_parameters(initial_lengthscale: float) -> np.ndarray:
    parameters = np.zeros(PARAMETER_COUNT, dtype=np.float64)
    parameters[HARM_CURVATURE_SLICE] = _inverse_positive(np.asarray(HARM_CURVATURE_INITIAL))
    parameters[CONTENT_LENGTHSCALE_INDEX] = _inverse_positive(initial_lengthscale)
    parameters[PHASE_FACTOR_SLICE] = 1.0
    parameters[PHASE_DIAGONAL_SLICE] = _inverse_positive(PHASE_DIAGONAL_INITIAL)
    parameters[RESIDUAL_LENGTHSCALE_INDEX] = _inverse_positive(initial_lengthscale)
    parameters[RESIDUAL_OUTPUTSCALE_INDEX] = _inverse_positive(RESIDUAL_OUTPUTSCALE_INITIAL)
    return parameters


def draw_initial_parameters(initial_lengthscale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parameters = initial_parameters(initial_lengthscale)
    parameters[CONSTANT_INDEX] = rng.normal()
    parameters[HARM_CURVATURE_SLICE] = _inverse_positive(
        rng.lognormal(
            np.log(np.asarray(HARM_CURVATURE_INITIAL)) + HARM_CURVATURE_PRIOR_LOG_SD**2,
            HARM_CURVATURE_PRIOR_LOG_SD,
        )
    )
    parameters[CONTENT_LENGTHSCALE_INDEX] = _inverse_positive(
        rng.lognormal(
            math.log(initial_lengthscale),
            LENGTHSCALE_PRIOR_LOG_SD,
        )
    )
    parameters[PHASE_FACTOR_SLICE] = rng.normal(1.0, 1.0, PHASE_COUNT)
    parameters[RESIDUAL_LENGTHSCALE_INDEX] = _inverse_positive(
        rng.lognormal(
            math.log(initial_lengthscale) + LENGTHSCALE_PRIOR_LOG_SD**2,
            LENGTHSCALE_PRIOR_LOG_SD,
        )
    )
    parameters[RESIDUAL_OUTPUTSCALE_INDEX] = _inverse_positive(
        rng.lognormal(
            math.log(RESIDUAL_OUTPUTSCALE_INITIAL) + LENGTHSCALE_PRIOR_LOG_SD**2,
            LENGTHSCALE_PRIOR_LOG_SD,
        )
    )
    return parameters


def quadratic_initializations(initial_lengthscale: float) -> tuple[MapInitialization, ...]:
    return (
        MapInitialization("prior_mode", 0, initial_parameters(initial_lengthscale)),
        MapInitialization("prior_draw", 11, draw_initial_parameters(initial_lengthscale, 11)),
        MapInitialization("prior_draw", 22, draw_initial_parameters(initial_lengthscale, 22)),
    )


def quadratic_exposure_mean(parameters: jax.Array, features: jax.Array) -> jax.Array:
    values = quadratic_parameter_values(parameters)
    feature_count = features.shape[-1] - 1
    quadratic = features[..., feature_count - PHASE_COUNT : feature_count]
    return values.constant - quadratic @ values.harm_curvature


def _squared_distance(first: jax.Array, second: jax.Array) -> jax.Array:
    squared = jnp.sum(jnp.square(first), axis=1)[:, None]
    squared += jnp.sum(jnp.square(second), axis=1)[None, :]
    squared -= 2.0 * first @ second.T
    return jnp.maximum(squared, 0.0)


def _matern52(first: jax.Array, second: jax.Array, lengthscale: jax.Array) -> jax.Array:
    scaled_distance = jnp.sqrt(5.0 * _squared_distance(first, second) + 1e-30) / lengthscale
    return (1.0 + scaled_distance + jnp.square(scaled_distance) / 3.0) * jnp.exp(-scaled_distance)


def quadratic_exposure_covariance(parameters: jax.Array, first: jax.Array, second: jax.Array) -> jax.Array:
    values = quadratic_parameter_values(parameters)
    feature_count = first.shape[-1] - 1
    phase_width = (feature_count - PHASE_COUNT) // PHASE_COUNT
    phase_covariance = jnp.outer(values.phase_factor, values.phase_factor)
    phase_covariance += jnp.diag(values.phase_diagonal)

    shared = jnp.zeros((len(first), len(second)), dtype=first.dtype)
    for first_phase in range(PHASE_COUNT):
        first_slice = slice(first_phase * phase_width, (first_phase + 1) * phase_width)
        for second_phase in range(PHASE_COUNT):
            second_slice = slice(second_phase * phase_width, (second_phase + 1) * phase_width)
            shared += phase_covariance[first_phase, second_phase] * _matern52(
                first[:, first_slice],
                second[:, second_slice],
                values.content_lengthscale,
            )

    response_slice = slice(0, PHASE_COUNT * phase_width)
    same_swarm = first[:, feature_count, None] == second[None, :, feature_count]
    residual = values.residual_outputscale * _matern52(
        first[:, response_slice],
        second[:, response_slice],
        values.residual_lengthscale,
    )
    return shared + residual * same_swarm


def _lognormal_log_probability(value: jax.Array, log_mean: jax.Array, log_sd: float) -> jax.Array:
    standardized = (jnp.log(value) - log_mean) / log_sd
    return -jnp.log(value * log_sd * math.sqrt(2.0 * math.pi)) - 0.5 * jnp.square(standardized)


def quadratic_log_prior(parameters: jax.Array, initial_lengthscale: float) -> jax.Array:
    values = quadratic_parameter_values(parameters)
    log_probability = jnp.sum(
        _lognormal_log_probability(
            values.harm_curvature,
            jnp.log(jnp.asarray(HARM_CURVATURE_INITIAL)) + HARM_CURVATURE_PRIOR_LOG_SD**2,
            HARM_CURVATURE_PRIOR_LOG_SD,
        )
    )
    log_probability += _lognormal_log_probability(
        values.content_lengthscale,
        jnp.log(jnp.asarray(initial_lengthscale)),
        LENGTHSCALE_PRIOR_LOG_SD,
    )
    log_probability += jnp.sum(-0.5 * jnp.square(values.phase_factor - 1.0))
    log_probability += _lognormal_log_probability(
        values.residual_lengthscale,
        jnp.log(jnp.asarray(initial_lengthscale)) + LENGTHSCALE_PRIOR_LOG_SD**2,
        LENGTHSCALE_PRIOR_LOG_SD,
    )
    log_probability += _lognormal_log_probability(
        values.residual_outputscale,
        jnp.log(jnp.asarray(RESIDUAL_OUTPUTSCALE_INITIAL)) + LENGTHSCALE_PRIOR_LOG_SD**2,
        LENGTHSCALE_PRIOR_LOG_SD,
    )
    return log_probability


def negative_log_posterior(
    parameters: jax.Array,
    train_X: jax.Array,
    train_Y: jax.Array,
    train_Yvar: jax.Array,
    initial_lengthscale: float,
) -> jax.Array:
    conditioning = _training_conditioning(parameters, train_X, train_Y, train_Yvar)
    negative_log_likelihood = 0.5 * conditioning.residual @ conditioning.alpha
    negative_log_likelihood += jnp.sum(jnp.log(jnp.diag(conditioning.cholesky)))
    negative_log_likelihood += 0.5 * len(train_X) * math.log(2.0 * math.pi)
    return negative_log_likelihood - quadratic_log_prior(parameters, initial_lengthscale)


def _training_conditioning(
    parameters: jax.Array,
    train_X: jax.Array,
    train_Y: jax.Array,
    train_Yvar: jax.Array,
) -> TrainingConditioning:
    covariance = quadratic_exposure_covariance(parameters, train_X, train_X)
    covariance += jnp.diag(train_Yvar + GP_JITTER)
    cholesky = jnp.linalg.cholesky(covariance)
    residual = train_Y - quadratic_exposure_mean(parameters, train_X)
    return TrainingConditioning(cholesky, residual, cho_solve((cholesky, True), residual))


@jax.jit
def _posterior_marginals(
    parameters: jax.Array,
    train_X: jax.Array,
    cholesky: jax.Array,
    alpha: jax.Array,
    test_X: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    cross_covariance = quadratic_exposure_covariance(parameters, test_X, train_X)
    mean = quadratic_exposure_mean(parameters, test_X) + cross_covariance @ alpha
    solved = solve_triangular(cholesky, cross_covariance.T, lower=True)
    prior_variance = jnp.diag(quadratic_exposure_covariance(parameters, test_X, test_X))
    variance = prior_variance - jnp.sum(jnp.square(solved), axis=0)
    return mean, jnp.maximum(variance, 0.0)


@partial(jax.jit, static_argnames=("sample_count",))
def _noisy_expected_improvement(
    parameters: jax.Array,
    train_X: jax.Array,
    cholesky: jax.Array,
    alpha: jax.Array,
    reference_X: jax.Array,
    candidate_X: jax.Array,
    key: jax.Array,
    *,
    sample_count: int,
) -> jax.Array:
    reference_train = quadratic_exposure_covariance(parameters, reference_X, train_X)
    candidate_train = quadratic_exposure_covariance(parameters, candidate_X, train_X)
    solved_reference = solve_triangular(cholesky, reference_train.T, lower=True)
    solved_candidate = solve_triangular(cholesky, candidate_train.T, lower=True)

    reference_mean = quadratic_exposure_mean(parameters, reference_X) + reference_train @ alpha
    candidate_mean = quadratic_exposure_mean(parameters, candidate_X) + candidate_train @ alpha
    reference_covariance = quadratic_exposure_covariance(parameters, reference_X, reference_X)
    reference_covariance -= solved_reference.T @ solved_reference
    candidate_reference = quadratic_exposure_covariance(parameters, candidate_X, reference_X)
    candidate_reference -= solved_candidate.T @ solved_reference
    candidate_variance = jnp.diag(quadratic_exposure_covariance(parameters, candidate_X, candidate_X))
    candidate_variance -= jnp.sum(jnp.square(solved_candidate), axis=0)

    candidate_count = len(candidate_X)
    reference_count = len(reference_X)
    joint_mean = jnp.concatenate(
        [
            jnp.broadcast_to(reference_mean, (candidate_count, reference_count)),
            candidate_mean[:, None],
        ],
        axis=1,
    )
    joint_covariance = jnp.zeros(
        (candidate_count, reference_count + 1, reference_count + 1),
        dtype=train_X.dtype,
    )
    joint_covariance = joint_covariance.at[:, :reference_count, :reference_count].set(reference_covariance)
    joint_covariance = joint_covariance.at[:, -1, :reference_count].set(candidate_reference)
    joint_covariance = joint_covariance.at[:, :reference_count, -1].set(candidate_reference)
    joint_covariance = joint_covariance.at[:, -1, -1].set(candidate_variance)
    joint_covariance += GP_JITTER * jnp.eye(reference_count + 1)[None]
    joint_cholesky = jnp.linalg.cholesky(joint_covariance)
    standard_normal = jax.random.normal(
        key,
        (candidate_count, sample_count, reference_count + 1),
        dtype=train_X.dtype,
    )
    samples = joint_mean[:, None, :] + jnp.einsum("cij,csj->csi", joint_cholesky, standard_normal)
    incumbent = jnp.max(samples[:, :, :reference_count], axis=2)
    return jnp.mean(jnp.maximum(samples[:, :, -1] - incumbent, 0.0), axis=1)


def moment_lengthscale(features: np.ndarray, layout: QuadraticExposureLayout) -> float:
    phase_features = np.concatenate([features[:, phase] for phase in layout.phase_exposure_content])
    squared_norm = np.square(phase_features).sum(axis=1)
    squared_distance = squared_norm[:, None] + squared_norm[None, :] - 2 * phase_features @ phase_features.T
    upper = squared_distance[np.triu_indices(len(squared_distance), k=1)]
    positive = upper[upper > np.finfo(np.float64).eps]
    if not len(positive):
        raise ValueError("Kernel calibration needs two distinct exposure profiles")
    return float(np.sqrt(np.median(positive)))


@dataclass(frozen=True)
class FittedQuadraticExposureGP:
    parameters: jax.Array
    train_X: jax.Array
    cholesky: jax.Array
    alpha: jax.Array
    swarm_indices: dict[str, int]
    outcome_scales: dict[str, SwarmOutcomeScale]
    model_metadata: ModelMetadata

    @jax.enable_x64()
    def candidate_array(self, swarm: Swarm, weights: np.ndarray) -> jax.Array:
        if swarm.swarm_id not in self.swarm_indices:
            raise ValueError(f"Swarm {swarm.swarm_id!r} was not included when this model was fit")
        features = quadratic_exposure_features(swarm, weights)
        swarm_column = np.full((len(features), 1), self.swarm_indices[swarm.swarm_id], dtype=np.float64)
        values = np.concatenate([features, swarm_column], axis=1)
        return jax.device_put(values, self.parameters.device)

    @jax.enable_x64()
    def predict(self, swarm: Swarm, weights: np.ndarray) -> PredictiveMoments:
        mean, variance = _posterior_marginals(
            self.parameters,
            self.train_X,
            self.cholesky,
            self.alpha,
            self.candidate_array(swarm, weights),
        )
        scale = self.outcome_scales[swarm.swarm_id]
        return PredictiveMoments(
            mean=scale.mean + scale.scale * np.asarray(mean),
            latent_variance=scale.scale**2 * np.asarray(variance),
        )

    @jax.enable_x64()
    def noisy_expected_improvement(
        self,
        swarm: Swarm,
        weights: np.ndarray,
        reference_weights: np.ndarray,
        *,
        sample_count: int,
        seed: int,
    ) -> np.ndarray:
        values = _noisy_expected_improvement(
            self.parameters,
            self.train_X,
            self.cholesky,
            self.alpha,
            self.candidate_array(swarm, reference_weights),
            self.candidate_array(swarm, weights),
            jax.random.key(seed),
            sample_count=sample_count,
        )
        return self.outcome_scales[swarm.swarm_id].scale * np.asarray(values)


def quadratic_model_metadata(
    training: TrainingData,
    parameters: jax.Array,
    initial_lengthscale: float,
    fit: MapFit,
    device: jax.Device,
) -> ModelMetadata:
    values = quadratic_parameter_values(parameters)
    phase_factor = values.phase_factor
    phase_covariance = jnp.outer(phase_factor, phase_factor) + jnp.diag(values.phase_diagonal)
    return {
        "kind": "quadratic_exposure_transfer_gp",
        "device": str(device),
        "details": {
            "backend": "compiled vanilla JAX",
            "mean": "learned phase-specific quadratic penalty on token-mass-weighted epochs",
            "harm_curvature_initial": list(HARM_CURVATURE_INITIAL),
            "harm_curvature": np.asarray(values.harm_curvature).tolist(),
            "covariance": "phase-linked content response plus same-swarm Matern-5/2 residual",
            "initial_lengthscale": initial_lengthscale,
            "content_lengthscale": float(values.content_lengthscale),
            "phase_covariance": np.asarray(phase_covariance).tolist(),
            "outcome_transform": "per-swarm affine standardization",
            "hyperparameter_inference": "lowest-negative-log-posterior JAX BFGS fit from three fixed starts",
            "map_restarts": [summary._asdict() for summary in fit.restarts],
            "observation_counts": training.observation_counts,
            "fit_seconds": fit.elapsed,
        },
    }


@jax.enable_x64()
def fit_quadratic_exposure_model(campaign: Campaign, device: jax.Device) -> FittedQuadraticExposureGP:
    started = time.monotonic()
    swarms = (campaign.target, *campaign.sources)
    logger.info(
        "Preparing GP data for target %s from %d swarms on %s",
        campaign.target.swarm_id,
        len(swarms),
        device,
    )
    training = prepare_training_data(campaign, quadratic_exposure_features)
    train_X = jax.device_put(training.features, device)
    train_Y = jax.device_put(training.standardized_objective_values, device)
    train_Yvar = jax.device_put(training.standardized_objective_variances, device)
    layout = quadratic_exposure_layout(campaign.target.content_matrix.shape[1])
    initial_lengthscale = moment_lengthscale(training.features[:, : layout.feature_count], layout)
    logger.info(
        "Prepared %d observations with %d features; initial lengthscale %.6f",
        len(training.features),
        training.features.shape[1] - 1,
        initial_lengthscale,
    )
    objective = partial(
        negative_log_posterior,
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        initial_lengthscale=initial_lengthscale,
    )
    fit = fit_map_restarts(objective, quadratic_initializations(initial_lengthscale), device)
    parameters = jax.device_put(fit.parameters, device)
    conditioning = jax.jit(_training_conditioning)(
        parameters,
        train_X,
        train_Y,
        train_Yvar,
    )
    jax.block_until_ready(conditioning.cholesky)
    logger.info("GP fit and conditioning completed in %.1fs", time.monotonic() - started)
    return FittedQuadraticExposureGP(
        parameters=parameters,
        train_X=train_X,
        cholesky=conditioning.cholesky,
        alpha=conditioning.alpha,
        swarm_indices=training.swarm_indices,
        outcome_scales=training.outcome_scales,
        model_metadata=quadratic_model_metadata(
            training,
            parameters,
            initial_lengthscale,
            fit,
            device,
        ),
    )
