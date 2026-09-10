# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from experiments.datakit.mixprior.acquisition import fit_additive
from experiments.datakit.mixprior.objective import Objective


def test_metric_cell_alignment_and_acquisition_gradient(data):
    outcomes = np.column_stack([data.weights[:, 0, 0] - data.weights[:, 1, 2], data.weights[:, 1, 1] - 0.3])
    data = replace(data, outcomes=outcomes)
    objective = Objective(np.array([0, 1]), np.array([True, False]), np.zeros(2), np.ones(2), np.eye(2) * 0.01, 0.0)
    counts = np.ones(len(outcomes))
    device = jax.devices("cpu")[0]
    model = fit_additive(data, objective, counts, device)
    logits = jnp.array([[[0.3, -0.2, 0.7], [0.1, 0.2, -0.3]]])

    def acquisition(point):
        return model.acquisition(jax.nn.softmax(point, axis=-1))[0]

    gradient = np.asarray(jax.grad(acquisition)(logits))
    finite = np.zeros(logits.shape)
    for index in np.ndindex(logits.shape):
        direction = np.zeros(logits.shape)
        direction[index] = 1e-5
        finite[index] = (acquisition(logits + direction) - acquisition(logits - direction)) / 2e-5
    np.testing.assert_allclose(gradient, finite, atol=1e-7, rtol=1e-5)
    order = np.array([2, 0, 1])
    other = replace(
        data,
        weights=data.weights[:, :, order],
        quality=data.quality[order],
        available_tokens=data.available_tokens[order],
        components=[data.components[i] for i in order],
        domains=[data.domains[i] for i in order],
    )
    reordered = fit_additive(other, objective, counts, device)
    probes = np.asarray(jax.nn.softmax(logits, axis=-1))
    np.testing.assert_allclose(
        reordered.acquisition(probes[:, :, order]), model.acquisition(probes), atol=1e-8, rtol=1e-8
    )
