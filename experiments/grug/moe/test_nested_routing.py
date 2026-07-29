# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from jax import numpy as jnp

from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import training_expert_eligibility


def _model_config(*, nested_count: int, nested_layer_fraction: float) -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=128,
        hidden_dim=64,
        intermediate_dim=32,
        shared_expert_intermediate_dim=64,
        num_experts=256,
        num_experts_per_token=4,
        nested_expert_counts=(nested_count,),
        nested_batch_fraction=0.25,
        nested_layer_fraction=nested_layer_fraction,
        num_layers=8,
        num_heads=1,
        num_kv_heads=1,
        max_seq_len=16,
    )


def test_all_layer_schedule_restricts_one_quarter_of_sequences() -> None:
    eligibility = np.asarray(
        training_expert_eligibility(
            _model_config(nested_count=128, nested_layer_fraction=1.0),
            batch_size=32,
            step=jnp.asarray(3),
        )
    )

    assert eligibility.shape == (32, 256)
    eligible_counts = eligibility.sum(axis=-1)
    assert np.count_nonzero(eligible_counts == 128) == 8
    assert np.count_nonzero(eligible_counts == 256) == 24
    np.testing.assert_array_equal(eligibility[eligible_counts == 128, :128], True)
    np.testing.assert_array_equal(eligibility[eligible_counts == 128, 128:], False)


def test_layerwise_schedule_rotates_two_restricted_layers_evenly() -> None:
    config = _model_config(nested_count=16, nested_layer_fraction=0.25)
    restricted_by_step = []

    for step in range(4):
        eligibility = np.asarray(training_expert_eligibility(config, batch_size=32, step=jnp.asarray(step)))
        assert eligibility.shape == (8, 32, 256)
        eligible_counts = eligibility.sum(axis=-1)
        restricted = eligible_counts == 16
        nested_rows = np.any(restricted, axis=0)

        assert np.count_nonzero(nested_rows) == 8
        np.testing.assert_array_equal(restricted[:, nested_rows].sum(axis=0), 2)
        np.testing.assert_array_equal(eligibility[:, ~nested_rows, :], True)
        np.testing.assert_array_equal(eligibility[:, nested_rows, :16], True)
        restricted_by_step.append(restricted)

    restrictions_per_layer = np.concatenate(restricted_by_step, axis=1).sum(axis=1)
    np.testing.assert_array_equal(restrictions_per_layer, np.full(8, 8))
