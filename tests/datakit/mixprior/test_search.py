# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import torch

from experiments.datakit.mixprior.campaign import Campaign
from experiments.datakit.mixprior.model import fit_additive_hellinger_model, prepare_hellinger_transfer_data
from experiments.datakit.mixprior.search import (
    LognormalPoolInputs,
    acquire_posterior_mean,
    prepare_candidate_features,
    sample_lognormal_pool,
)


def test_lognormal_pool_is_unique_feasible_and_excludes_observations(
    tiny_campaign: Campaign,
) -> None:
    campaign = replace(
        tiny_campaign,
        target=replace(
            tiny_campaign.target,
            phase_budgets=np.asarray([6.0, 6.0]),
            provenance=replace(tiny_campaign.target.provenance, simulated_training_tokens=12),
        ),
        max_cumulative_epochs=8.0,
    )
    target = campaign.target
    proportional = target.data.available_tokens / target.data.available_tokens.sum()
    proportional = np.broadcast_to(proportional, target.data.weights.shape[1:])
    space = LognormalPoolInputs(
        center_designs=np.concatenate([proportional[None], target.data.weights]),
        availability_proportional_design=proportional,
        observed_weights=target.data.weights,
        exposure_multipliers=target.exposure_multipliers,
        max_cumulative_epochs=campaign.max_cumulative_epochs,
    )

    pool = sample_lognormal_pool(space, size=128, seed=17)

    assert np.allclose(pool.sum(axis=-1), 1.0)
    epochs = (pool * campaign.target.exposure_multipliers[None]).sum(axis=1)
    assert np.max(epochs) <= 8.0
    assert len(np.unique(np.round(pool.reshape(len(pool), -1), 12), axis=0)) == len(pool)
    for observed in target.data.weights:
        assert not np.any(np.all(np.isclose(pool, observed, atol=1e-12), axis=(1, 2)))


def test_posterior_mean_selects_best_pool_row(tiny_campaign: Campaign) -> None:
    torch.manual_seed(5)
    model = fit_additive_hellinger_model(prepare_hellinger_transfer_data(tiny_campaign), torch.device("cpu"))
    pool = np.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )
    candidates = prepare_candidate_features(tiny_campaign.target, pool)

    acquired = acquire_posterior_mean(model, candidates.features, model.target_swarm_index)

    assert acquired.pool_index == 0
