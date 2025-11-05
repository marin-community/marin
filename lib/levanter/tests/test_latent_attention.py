# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.random as jrandom
import jax.numpy as jnp
import pytest

import haliax as hax
from haliax import Axis

from levanter.layers.attention import (
    AttentionMask,
    MultiHeadLatentAttention,
    MultiHeadLatentAttentionConfig,
)
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig


@pytest.mark.parametrize("q_lora_rank", [None, 2])
def test_multihead_latent_attention_shapes(q_lora_rank):
    Batch = Axis("batch", 2)
    Pos = Axis("position", 8)
    Embed = Axis("embed", 32)

    config = MultiHeadLatentAttentionConfig(
        Embed=Embed,
        num_heads=2,
        kv_lora_rank=Embed.size // 16,
        q_lora_rank=q_lora_rank,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=8,
        rope=DefaultRotaryEmbeddingsConfig(),
    )

    attn = MultiHeadLatentAttention.init(config, key=jrandom.PRNGKey(0))
    x = hax.random.normal(jrandom.PRNGKey(1), (Batch, Pos, Embed))
    mask = AttentionMask.causal()
    out = attn(x, mask, key=jrandom.PRNGKey(2))

    assert out.axes == x.axes
    assert not jnp.isnan(out.array).any()
