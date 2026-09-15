# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.datakit.mixprior.campaign import Campaign
from experiments.datakit.mixprior.data import Swarm
from experiments.datakit.mixprior.search import (
    MIXTURE_DENOMINATOR,
    CandidatePoolInputs,
    acquire_noisy_expected_improvement,
    acquire_posterior_mean,
    quantize_mixtures,
    sample_candidate_pool,
    sample_candidate_pool_union,
    validate_candidate_pool,
)
from experiments.datakit.mixprior.surrogate import PredictiveMoments, prepare_training_data


class LinearPredictor:
    def predict(self, _swarm: Swarm, weights: np.ndarray) -> PredictiveMoments:
        return PredictiveMoments(mean=weights[:, 0, 0], latent_variance=np.full(len(weights), 0.1))


class AcquisitionPredictor:
    def __init__(self) -> None:
        self.reference_weights = np.empty((0, 2, 2))

    def noisy_expected_improvement(
        self,
        _swarm: Swarm,
        weights: np.ndarray,
        reference_weights: np.ndarray,
        *,
        sample_count: int,
        seed: int,
    ) -> np.ndarray:
        self.reference_weights = reference_weights
        return weights[:, 0, 0]


def test_transfer_data_uses_caller_feature_map(tiny_campaign: Campaign) -> None:
    def flatten_weights(_swarm: Swarm, weights: np.ndarray) -> np.ndarray:
        return weights.reshape(len(weights), -1)

    training = prepare_training_data(tiny_campaign, flatten_weights)

    target_rows = training.features[: len(tiny_campaign.target.data.weights), :-1]
    assert np.array_equal(target_rows, tiny_campaign.target.data.weights.reshape(2, -1))


def test_candidate_pool_is_quantized_unique_and_excludes_observations(
    tiny_campaign: Campaign,
) -> None:
    target = tiny_campaign.target
    proportional = target.data.available_tokens / target.data.available_tokens.sum()
    proportional = np.broadcast_to(proportional, target.data.weights.shape[1:])
    space = CandidatePoolInputs(
        center_designs=np.concatenate([proportional[None], target.data.weights]),
        availability_proportional_design=proportional,
        observed_weights=target.data.weights,
    )

    pool = sample_candidate_pool(space, size=128, seed=17)

    assert np.allclose(pool.sum(axis=-1), 1.0)
    assert np.allclose(pool * MIXTURE_DENOMINATOR, np.round(pool * MIXTURE_DENOMINATOR))
    assert len(np.unique(np.round(pool.reshape(len(pool), -1), 12), axis=0)) == len(pool)
    for observed in quantize_mixtures(target.data.weights):
        assert not np.any(np.all(np.isclose(pool, observed, atol=1e-12), axis=(1, 2)))


def test_small_candidate_pool_contains_global_simplex_draws() -> None:
    local_center = np.asarray([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    space = CandidatePoolInputs(
        center_designs=local_center[None],
        availability_proportional_design=local_center,
        observed_weights=local_center[None],
    )

    pool = sample_candidate_pool(space, size=128, seed=17)

    assert np.any(pool[:, :, 2] > 0)


def test_candidate_pool_union_combines_independent_draws(tiny_campaign: Campaign) -> None:
    space = CandidatePoolInputs(
        center_designs=tiny_campaign.target.data.weights,
        availability_proportional_design=np.full_like(tiny_campaign.target.data.weights[0], 0.5),
        observed_weights=tiny_campaign.target.data.weights,
    )

    first = sample_candidate_pool(space, size=32, seed=11)
    second = sample_candidate_pool(space, size=32, seed=22)
    combined = sample_candidate_pool_union(space, size_per_seed=32, seeds=(11, 22))

    combined_rows = {row.tobytes() for row in np.round(combined.reshape(len(combined), -1), 12)}
    assert all(row.tobytes() in combined_rows for row in np.round(first.reshape(len(first), -1), 12))
    assert all(row.tobytes() in combined_rows for row in np.round(second.reshape(len(second), -1), 12))


def test_candidate_pool_union_requires_a_seed(tiny_campaign: Campaign) -> None:
    space = CandidatePoolInputs(
        center_designs=tiny_campaign.target.data.weights,
        availability_proportional_design=np.full_like(tiny_campaign.target.data.weights[0], 0.5),
        observed_weights=tiny_campaign.target.data.weights,
    )

    with pytest.raises(ValueError, match="At least one"):
        sample_candidate_pool_union(space, size_per_seed=32, seeds=())

    with pytest.raises(ValueError, match="distinct"):
        sample_candidate_pool_union(space, size_per_seed=32, seeds=(11, 11))


def test_quantize_mixtures_requires_phase_simplexes() -> None:
    with pytest.raises(ValueError, match="simplex"):
        quantize_mixtures(np.asarray([[[0.2, 0.2], [0.5, 0.5]]]))


def test_candidate_pool_rejects_off_lattice_weights() -> None:
    with pytest.raises(ValueError, match="multiples"):
        validate_candidate_pool(np.asarray([[[0.7, 0.3], [0.6, 0.4]]]), (2, 2))


def test_candidate_pool_rejects_wrong_campaign_shape() -> None:
    with pytest.raises(ValueError, match="phases and components"):
        validate_candidate_pool(np.asarray([[[0.5, 0.5]]]), (2, 2))


def test_posterior_mean_selects_best_pool_row(tiny_campaign: Campaign) -> None:
    model = LinearPredictor()
    pool = np.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )
    acquired = acquire_posterior_mean(model, tiny_campaign.target, pool)

    assert acquired.pool_index == 0


def test_noisy_expected_improvement_uses_observed_and_pending_rows(tiny_campaign: Campaign) -> None:
    model = AcquisitionPredictor()
    pool = np.asarray(
        [
            [[0.25, 0.75], [0.25, 0.75]],
            [[0.75, 0.25], [0.75, 0.25]],
        ]
    )

    pending = pool[:1]
    acquired = acquire_noisy_expected_improvement(model, tiny_campaign.target, pool, seed=7, pending_weights=pending)

    assert acquired.pool_index == 1
    assert np.array_equal(
        model.reference_weights,
        np.concatenate([tiny_campaign.target.data.weights, pending]),
    )
