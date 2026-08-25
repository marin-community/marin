# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
from jax import random

from levanter.models.whisper import WhisperConfig, WhisperLayer


def test_decoder_layer_cross_attention_is_independently_initialized():
    """A decoder layer's cross-attention must not be a copy of its self-attention."""
    config = WhisperConfig(
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
    )
    layer = WhisperLayer.init(
        config.DecoderHeads,
        config.DecoderHeadSize,
        config.DecoderMlp,
        config,
        has_cross=True,
        key=random.PRNGKey(0),
    )

    assert layer.encoder_attn is not None
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        self_weight = getattr(layer.self_attn, name).weight.array
        cross_weight = getattr(layer.encoder_attn, name).weight.array
        assert not jnp.array_equal(self_weight, cross_weight), f"{name} shared between self- and cross-attention"
