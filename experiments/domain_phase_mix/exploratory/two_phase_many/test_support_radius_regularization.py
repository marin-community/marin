# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many.support_radius_regularization import (
    build_support_geometry,
    logits_to_weights,
    support_distance,
    support_distance_and_gradient,
    weights_to_logits,
)


def _geometry():
    train = np.asarray(
        [
            [[0.8, 0.2, 0.0], [0.4, 0.3, 0.3]],
            [[0.2, 0.7, 0.1], [0.1, 0.2, 0.7]],
            [[0.4, 0.4, 0.2], [0.5, 0.2, 0.3]],
        ]
    )
    basis = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
        ]
    )
    return build_support_geometry(train, basis, np.asarray([0.8, 0.2]))


def test_logit_round_trip() -> None:
    weights = np.asarray([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    recovered = logits_to_weights(weights_to_logits(weights), 3)
    np.testing.assert_allclose(recovered, weights, atol=1e-12)


def test_support_distance_gradient_matches_finite_difference() -> None:
    geometry = _geometry()
    weights = np.asarray([[0.35, 0.25, 0.4], [0.25, 0.5, 0.25]])
    logits = weights_to_logits(weights)
    value, gradient, nearest = support_distance_and_gradient(logits, geometry)
    direct, direct_nearest = support_distance(weights, geometry)
    assert nearest == direct_nearest
    assert np.isclose(value, direct)

    epsilon = 1e-6
    numerical = np.empty_like(logits)
    for index in range(len(logits)):
        offset = np.zeros_like(logits)
        offset[index] = epsilon
        plus, _, plus_nearest = support_distance_and_gradient(logits + offset, geometry)
        minus, _, minus_nearest = support_distance_and_gradient(logits - offset, geometry)
        assert plus_nearest == nearest == minus_nearest
        numerical[index] = (plus - minus) / (2.0 * epsilon)
    np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-6)
