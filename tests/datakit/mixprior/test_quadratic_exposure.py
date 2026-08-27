# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from experiments.datakit.mixprior.campaign import Campaign
from experiments.datakit.mixprior.quadratic_exposure import (
    GP_JITTER,
    HARM_CURVATURE_INITIAL,
    fit_quadratic_exposure_model,
    initial_parameters,
    negative_log_posterior,
    quadratic_exposure_covariance,
    quadratic_exposure_features,
    quadratic_exposure_layout,
    quadratic_exposure_mean,
)


def test_quadratic_exposure_features_match_epoch_moments(tiny_campaign: Campaign) -> None:
    swarm = tiny_campaign.target
    weights = swarm.data.weights
    exposure = weights * swarm.exposure_multipliers[None]
    token_share = swarm.data.available_tokens / swarm.data.available_tokens.sum()
    first = token_share[None, None, :] * exposure
    second = first * exposure
    expected_phase_content = np.concatenate(
        [first @ swarm.content_matrix, second @ swarm.content_matrix],
        axis=2,
    ).reshape(len(weights), -1)

    features = quadratic_exposure_features(swarm, weights)
    layout = quadratic_exposure_layout(swarm.content_matrix.shape[1])

    assert np.allclose(features[:, : layout.quadratic_exposure.start], expected_phase_content)
    assert np.allclose(features[:, layout.quadratic_exposure], second.sum(axis=2))


def test_quadratic_exposure_features_ignore_component_order(tiny_campaign: Campaign) -> None:
    swarm = tiny_campaign.target
    permutation = np.asarray([1, 0])
    permuted = replace(
        swarm,
        data=swarm.data._replace(
            mixture_components=[swarm.data.mixture_components[index] for index in permutation],
            component_metadata=[swarm.data.component_metadata[index] for index in permutation],
            available_tokens=swarm.data.available_tokens[permutation],
            weights=swarm.data.weights[:, :, permutation],
        ),
        content_matrix=swarm.content_matrix[permutation],
    )

    assert np.allclose(
        quadratic_exposure_features(swarm, swarm.data.weights),
        quadratic_exposure_features(permuted, permuted.data.weights),
    )


def test_quadratic_mean_uses_epoch_exposure() -> None:
    layout = quadratic_exposure_layout(content_dim=1)
    inputs = np.zeros((3, layout.feature_count + 1), dtype=np.float64)
    inputs[:, layout.quadratic_exposure.start] = [0.0, 1.0, 4.0]

    values = np.asarray(quadratic_exposure_mean(jnp.asarray(initial_parameters(1.0)), jnp.asarray(inputs)))
    expected = -HARM_CURVATURE_INITIAL[0] * np.asarray([0.0, 1.0, 4.0])

    assert np.allclose(values, expected)


def test_quadratic_covariance_is_psd_with_finite_gradients(tiny_campaign: Campaign) -> None:
    features = quadratic_exposure_features(tiny_campaign.target, tiny_campaign.target.data.weights)
    inputs = jnp.asarray(np.concatenate([features, np.zeros((len(features), 1))], axis=1))
    parameters = jnp.asarray(initial_parameters(1.0))

    gram = quadratic_exposure_covariance(parameters, inputs, inputs)
    gradient = jax.grad(lambda values: jnp.square(quadratic_exposure_covariance(parameters, values, values)).sum())(
        inputs
    )

    assert np.allclose(gram, gram.T, atol=1e-12, rtol=0.0)
    assert np.linalg.eigvalsh(np.asarray(gram)).min() >= -1e-10
    assert np.isfinite(gradient).all()


def test_negative_log_posterior_compiles_to_same_value(tiny_campaign: Campaign) -> None:
    target_features = quadratic_exposure_features(tiny_campaign.target, tiny_campaign.target.data.weights)
    source_features = quadratic_exposure_features(tiny_campaign.sources[0], tiny_campaign.sources[0].data.weights)
    inputs = jnp.asarray(
        np.concatenate(
            [
                np.concatenate([target_features, np.zeros((2, 1))], axis=1),
                np.concatenate([source_features, np.ones((2, 1))], axis=1),
            ]
        )
    )
    values = jnp.asarray([-1.0, 1.0, -1.0, 1.0])
    variances = jnp.full(4, 0.1)
    parameters = jnp.asarray(initial_parameters(1.0))

    eager = negative_log_posterior(parameters, inputs, values, variances, 1.0)
    compiled = jax.jit(negative_log_posterior, static_argnums=4)(parameters, inputs, values, variances, 1.0)

    assert np.isfinite(eager)
    assert np.allclose(eager, compiled, atol=GP_JITTER, rtol=0.0)


def test_fitted_model_predicts_and_acquires(tiny_campaign: Campaign) -> None:
    model = fit_quadratic_exposure_model(tiny_campaign, jax.devices("cpu")[0])
    weights = tiny_campaign.target.data.weights

    moments = model.predict(tiny_campaign.target, weights)
    improvement = model.noisy_expected_improvement(
        tiny_campaign.target,
        weights,
        weights,
        sample_count=32,
        seed=7,
    )

    assert moments.mean.shape == (2,)
    assert moments.latent_variance.shape == (2,)
    assert np.isfinite(moments.mean).all()
    assert np.all(moments.latent_variance >= 0)
    assert improvement.shape == (2,)
    assert np.all(improvement >= 0)
