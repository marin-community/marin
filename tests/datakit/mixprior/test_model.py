# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from experiments.datakit.mixprior.model import (
    SharedSwarmHellingerGP,
    SwarmTrainingRows,
    assemble_transfer_data,
    curriculum_features,
    rav_lengthscale,
    rav_rbf_kernel,
)


def test_fixed_rbf_matches_phase_weighted_hellinger() -> None:
    content = np.asarray([[0.8, 0.2], [0.1, 0.9]])
    weights = np.asarray(
        [
            [[0.8, 0.2], [0.5, 0.5]],
            [[0.3, 0.7], [0.9, 0.1]],
            [[0.6, 0.4], [0.2, 0.8]],
        ]
    )
    composition = weights @ content
    phase_token_fractions = np.asarray([0.8, 0.2])
    affinity = np.einsum("ipk,jpk->ijp", np.sqrt(composition), np.sqrt(composition))
    distances = np.sum((1.0 - affinity) * phase_token_fractions, axis=-1)
    gamma = 0.25 / np.median(distances[np.triu_indices(3, k=1)])
    expected = np.exp(-gamma * distances)

    features = curriculum_features(weights, content, phase_token_fractions)
    kernel = rav_rbf_kernel(rav_lengthscale(features))
    actual = kernel(torch.as_tensor(features, dtype=torch.double)).to_dense().detach()

    assert np.allclose(actual.numpy(), expected, atol=1e-10)


def test_swarm_kernel_adds_residual_only_within_same_swarm() -> None:
    X = torch.tensor(
        [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 1.0]],
        dtype=torch.double,
    )
    model = SharedSwarmHellingerGP(
        train_X=X,
        train_Y=torch.zeros((3, 1), dtype=torch.double),
        train_Yvar=torch.full((3, 1), 0.1, dtype=torch.double),
        num_swarms=2,
        lengthscale=1.0,
    )
    covariance = model.covar_module(X).to_dense().detach()

    assert covariance[0, 1] > covariance[0, 2]
    assert covariance[0, 0] == pytest.approx(covariance[0, 1])


def test_transfer_assembly_accepts_precomputed_geometry() -> None:
    target = SwarmTrainingRows(
        swarm_id="target",
        features=np.asarray([[0.1, 0.2], [0.3, 0.4]]),
        objective_values=np.asarray([1.0, 2.0]),
        objective_variances=np.asarray([0.1, 0.2]),
    )
    source = SwarmTrainingRows(
        swarm_id="source",
        features=np.asarray([[0.5, 0.6]]),
        objective_values=np.asarray([3.0]),
        objective_variances=np.asarray([0.3]),
    )

    data = assemble_transfer_data([target, source], target_swarm="target")

    assert np.array_equal(
        data.features,
        np.asarray(
            [
                [0.1, 0.2, 0.0],
                [0.3, 0.4, 0.0],
                [0.5, 0.6, 1.0],
            ]
        ),
    )
    assert np.array_equal(data.objective_values[:, 0], [1.0, 2.0, 3.0])
    assert np.array_equal(data.objective_variances[:, 0], [0.1, 0.2, 0.3])
    assert data.swarm_indices == {"target": 0, "source": 1}
