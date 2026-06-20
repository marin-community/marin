# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from levanter.grug.attention import AttentionMask, ThdSegmentMetadata
from levanter.grug.grug_moe import GroupedMoEExpertMlp, MoEExpertMlp
from levanter.utils.activation import ActivationFunctionEnum

from experiments.grug.moe.model import _layer_attention_masks


def test_layer_attention_masks_use_direct_short_window_and_full_long_mask():
    metadata = ThdSegmentMetadata(
        segment_lengths=jnp.array([[4, 4]], dtype=jnp.int32),
        num_segments=jnp.array([2], dtype=jnp.int32),
    )
    mask = AttentionMask(is_causal=True, thd_segment_metadata=metadata)

    short_mask, long_mask = _layer_attention_masks(mask, sliding_window=8)

    assert short_mask.sliding_window == 8
    assert long_mask.sliding_window is None
    assert short_mask.thd_segment_metadata is metadata
    assert long_mask.thd_segment_metadata is metadata
    assert short_mask.segment_ids is None
    assert long_mask.segment_ids is None


def test_grouped_expert_mlp_from_layers_matches_individual_layers():
    key = jax.random.PRNGKey(0)
    layer_keys = jax.random.split(key, 2)
    layers = tuple(
        MoEExpertMlp(
            w_gate_up=jax.random.normal(layer_key, (4, 6, 10), dtype=jnp.float32),
            w_down=jax.random.normal(layer_key, (4, 5, 6), dtype=jnp.float32),
            implementation="scatter",
            activation=ActivationFunctionEnum.silu,
            capacity_factor=1.0,
            remat_mode="none",
        )
        for layer_key in layer_keys
    )
    grouped = GroupedMoEExpertMlp.from_layers(layers)
    x = jax.random.normal(jax.random.PRNGKey(1), (7, 6), dtype=jnp.float32)
    selected_experts = jnp.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [0, 2],
            [1, 3],
            [2, 0],
        ],
        dtype=jnp.int32,
    )
    combine_weights = jnp.array(
        [
            [0.7, 0.3],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.8, 0.2],
            [0.2, 0.8],
        ],
        dtype=jnp.float32,
    )

    expected = jnp.stack([layer(x, selected_experts, combine_weights) for layer in layers], axis=0)
    actual = grouped(
        jnp.stack([x, x], axis=0),
        jnp.stack([selected_experts, selected_experts], axis=0),
        jnp.stack([combine_weights, combine_weights], axis=0),
    )

    assert grouped.valid_group_size == len(layers)
    assert actual.shape == expected.shape
    assert jnp.max(jnp.abs(actual - expected)) < 1e-4


def test_grouped_expert_mlp_from_layers_rejects_mixed_backends():
    layer = MoEExpertMlp(
        w_gate_up=jnp.zeros((2, 3, 8), dtype=jnp.float32),
        w_down=jnp.zeros((2, 4, 3), dtype=jnp.float32),
        implementation="scatter",
        activation=ActivationFunctionEnum.silu,
        capacity_factor=1.0,
        remat_mode="none",
    )
    mismatched = MoEExpertMlp(
        w_gate_up=jnp.zeros((2, 3, 8), dtype=jnp.float32),
        w_down=jnp.zeros((2, 4, 3), dtype=jnp.float32),
        implementation="ring",
        activation=ActivationFunctionEnum.silu,
        capacity_factor=1.0,
        remat_mode="none",
    )

    with pytest.raises(ValueError, match="same implementation"):
        GroupedMoEExpertMlp.from_layers((layer, mismatched))
