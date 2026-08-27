# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.datakit.mixprior.campaign import Campaign
from experiments.datakit.mixprior.quadratic_exposure import (
    QuadraticExposurePenaltyMean,
    quadratic_exposure_features,
    quadratic_exposure_gp,
    quadratic_exposure_layout,
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


def test_quadratic_penalty_grows_with_epoch_concentration() -> None:
    layout = quadratic_exposure_layout(content_dim=1)
    mean = QuadraticExposurePenaltyMean(layout, torch.empty((), dtype=torch.double))
    mean.harm_curvature = torch.tensor([0.2, 1e-9])
    inputs = torch.zeros((3, layout.feature_count + 1), dtype=torch.double)
    inputs[:, layout.quadratic_exposure.start] = torch.tensor([0.0, 1.0, 4.0])

    values = mean(inputs)

    assert values[0] > values[1] > values[2]
    assert values[0] - values[1] < values[1] - values[2]


def test_quadratic_gp_covariance_is_psd_with_finite_gradients(tiny_campaign: Campaign) -> None:
    features = quadratic_exposure_features(tiny_campaign.target, tiny_campaign.target.data.weights)
    X = torch.as_tensor(
        np.concatenate([features, np.zeros((len(features), 1))], axis=1),
        dtype=torch.double,
    ).requires_grad_()
    model = quadratic_exposure_gp(
        train_X=X.detach(),
        train_Y=torch.tensor([[-1.0], [1.0]], dtype=torch.double),
        train_Yvar=torch.full((2, 1), 0.1, dtype=torch.double),
        content_dim=2,
        num_swarms=1,
        initial_lengthscale=1.0,
    )

    gram = model.covar_module(X).to_dense()
    (gradient,) = torch.autograd.grad(gram.square().sum(), X)

    assert torch.allclose(gram, gram.T, atol=1e-12, rtol=0.0)
    assert torch.linalg.eigvalsh(gram.detach()).min() >= -1e-10
    assert torch.isfinite(gradient).all()


def test_quadratic_gp_rejects_unknown_swarm_index(tiny_campaign: Campaign) -> None:
    features = quadratic_exposure_features(tiny_campaign.target, tiny_campaign.target.data.weights)
    X = torch.as_tensor(
        np.concatenate([features, np.full((len(features), 1), 2.0)], axis=1),
        dtype=torch.double,
    )

    with pytest.raises(ValueError, match="outside"):
        quadratic_exposure_gp(
            train_X=X,
            train_Y=torch.tensor([[-1.0], [1.0]], dtype=torch.double),
            train_Yvar=torch.full((2, 1), 0.1, dtype=torch.double),
            content_dim=2,
            num_swarms=1,
            initial_lengthscale=1.0,
        )
