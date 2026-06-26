# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the flat-Levanter-state-dict -> nested-nnx-state conversion used to sync weights into vLLM.

vLLM's Pallas kernels require attention head dimensions padded to a multiple of 128, and grouped
query-attention heads flattened into a single head axis. These tests pin that reshape/pad contract.
"""

import jax.numpy as jnp
import numpy as np
from marin.rl.weight_utils import levanter_state_dict_to_nnx_state_on_cpu


def _self_attn(state):
    return state["model"]["layers"]["0"]["self_attn"]


def test_grouped_query_weights_are_flattened_and_padded_to_128():
    state_dict = {
        # q_proj weight: (kv_heads, group, head_size, embed) -> (kv_heads*group, head_size->128, embed)
        "model.layers.0.self_attn.q_proj.weight": jnp.ones((2, 3, 64, 16)),
        # k/v projections arrive already flattened: (heads, head_size, embed) -> pad head_size to 128
        "model.layers.0.self_attn.k_proj.weight": jnp.ones((2, 64, 16)),
        "model.layers.0.self_attn.v_proj.weight": jnp.ones((2, 64, 16)),
        # o_proj weight: (embed, heads, head_size) -> pad head_size (axis 2) to 128
        "model.layers.0.self_attn.o_proj.weight": jnp.ones((16, 2, 64)),
    }

    attn = _self_attn(levanter_state_dict_to_nnx_state_on_cpu(state_dict))

    assert attn["q_proj"].value.shape == (6, 128, 16)
    assert attn["k_proj"].value.shape == (2, 128, 16)
    assert attn["v_proj"].value.shape == (2, 128, 16)
    assert attn["o_proj"].value.shape == (16, 2, 128)


def test_attention_bias_is_flattened_and_renamed():
    state_dict = {
        # q_proj bias: (kv_heads, group, head_size) -> (kv_heads*group, head_size->128)
        "model.layers.0.self_attn.q_proj.bias": jnp.ones((2, 3, 64)),
    }

    attn = _self_attn(levanter_state_dict_to_nnx_state_on_cpu(state_dict))

    # The trailing "bias" token is folded into the leaf name rather than becoming a nesting level.
    assert attn["q_proj_bias"].value.shape == (6, 128)


def test_head_dim_already_multiple_of_128_is_unchanged():
    state_dict = {"model.layers.0.self_attn.k_proj.weight": jnp.ones((2, 128, 16))}
    attn = _self_attn(levanter_state_dict_to_nnx_state_on_cpu(state_dict))
    assert attn["k_proj"].value.shape == (2, 128, 16)


def test_non_attention_params_pass_through_unmodified():
    state_dict = {
        "model.embed_tokens.weight": jnp.arange(10 * 16, dtype=jnp.float32).reshape(10, 16),
        # numpy inputs (as delivered over the wire) are converted to jax arrays.
        "lm_head.weight": np.ones((8, 16), dtype=np.float32),
    }

    state = levanter_state_dict_to_nnx_state_on_cpu(state_dict)

    embed = state["model"]["embed_tokens"].value
    assert embed.shape == (10, 16)
    np.testing.assert_array_equal(np.asarray(embed), np.arange(10 * 16).reshape(10, 16))
    assert state["lm_head"].value.shape == (8, 16)
