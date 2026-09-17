# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.datakit.mixprior.search import MIXTURE_DENOMINATOR, feasible_pool, quantize, search, search_cooldown


class QuadraticObjective:
    def acquisition(self, weights):
        optimum = np.array([[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]])
        return -np.square(weights - optimum).sum(axis=(1, 2))


def test_search_finds_high_scores_on_the_feasible_unobserved_lattice(data):
    model = QuadraticObjective()
    selected = search(
        model, data.available_tokens, data.phase_budgets, data.weights, pool_size=512, batch_size=3, seed=7, max_epochs=3
    )
    np.testing.assert_allclose(selected.sum(axis=-1), 1, atol=1e-12)
    np.testing.assert_array_equal(selected * MIXTURE_DENOMINATOR, np.rint(selected * MIXTURE_DENOMINATOR))
    assert np.all(selected >= 0)
    assert np.max((selected * data.exposure).sum(axis=1)) <= 3
    assert len(np.unique(selected.reshape(3, -1), axis=0)) == 3
    assert np.all(model.acquisition(selected) > -0.05)
    reserved = np.concatenate([data.weights, selected])
    following = search(model, data.available_tokens, data.phase_budgets, reserved, pool_size=512, seed=7, max_epochs=3)
    assert not any(np.allclose(following[0], row, atol=1e-12, rtol=0) for row in reserved)


def test_quantization_allocates_remainder_without_losing_probability_mass():
    weights = np.array([[[0.1, 0.2, 0.7], [1 / 3, 1 / 3, 1 / 3]]])
    rounded = quantize(weights)
    np.testing.assert_allclose(rounded.sum(axis=-1), 1, atol=1e-12)
    assert np.max(np.abs(rounded - weights)) <= 1 / MIXTURE_DENOMINATOR


def test_pool_excludes_observed_mixtures_after_training_lattice_rounding():
    observed = np.array([[[0.1, 0.3, 0.6], [0.1, 0.3, 0.6]]])
    rounded = np.array([[[4915, 14746, 29491], [4915, 14746, 29491]]]) / MIXTURE_DENOMINATOR
    unseen = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    pool = feasible_pool(np.concatenate([rounded, unseen]), observed, np.ones((2, 3)), max_epochs=2)
    np.testing.assert_array_equal(pool, unseen)


def test_cooldown_search_preserves_main_and_accounts_for_consumed_tokens(data):
    anchor = quantize(np.array([[0.6, 0.3, 0.1], [0.3, 0.3, 0.4]]))
    selected = search_cooldown(
        QuadraticObjective(),
        anchor,
        np.array([10.0, 20.0, 30.0]),
        np.array([8.0, 10.0, 10.0]),
        10.0,
        data.weights,
        pool_size=1024,
        batch_size=3,
        seed=7,
        max_epochs=1.0,
        radius=0.3,
    )
    np.testing.assert_array_equal(selected[:, 0], np.broadcast_to(anchor[0], (3, 3)))
    assert np.all(selected[:, 1, 0] <= 0.2)
    assert np.max(np.abs(selected[:, 1] - anchor[1]).sum(-1) / 2) <= 0.3
    assert np.all(QuadraticObjective().acquisition(selected) > QuadraticObjective().acquisition(anchor[None])[0])
    assert np.max((np.array([8.0, 10.0, 10.0]) + 10 * selected[:, 1]) / np.array([10.0, 20.0, 30.0])) <= 1
    following = search_cooldown(
        QuadraticObjective(),
        anchor,
        np.array([10.0, 20.0, 30.0]),
        np.array([8.0, 10.0, 10.0]),
        10.0,
        np.concatenate([data.weights, selected]),
        pool_size=1024,
        batch_size=3,
        seed=7,
        max_epochs=1.0,
        radius=0.3,
    )
    assert not any(np.array_equal(a, b) for a in selected for b in following)
