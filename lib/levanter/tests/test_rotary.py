# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import random

import haliax as hax

from levanter.layers.rotary import _rotate_half as levanter_rotate_half
from test_llama import _get_llama_config
from test_utils import skip_if_no_torch


@skip_if_no_torch
@pytest.mark.parametrize("test_seq_len", [64, 128, 256])
def test_apply_rotary_pos_emb(test_seq_len):
    import torch
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as hf_apply_rotary_pos_emb
    from transformers.models.llama.modeling_llama import rotate_half as hf_rotate_half

    def assert_equal_out(hax_out, torch_out: torch.Tensor):
        assert np.isclose(
            torch_out.numpy(), np.array(hax_out.array), rtol=1e-2, atol=1e-2
        ).all(), f"{torch_out} != {hax_out}"

    def named_array_to_tensor(named_array):
        return torch.from_numpy(np.array(named_array.array))

    _llama_config = _get_llama_config()
    Pos = _llama_config.max_Pos
    Heads = _llama_config.attention_config().Heads
    HeadSize = _llama_config.attention_config().HeadSize
    Batch = hax.Axis("batch", 3)

    # note here we switch Heads and Pos for the shape of the output tensors
    q = hax.random.normal(random.PRNGKey(0), (Batch, Pos, Heads, HeadSize))
    k = hax.random.normal(random.PRNGKey(1), (Batch, Pos, Heads, HeadSize))

    # Check the output of _rotate_half() from levanter and hf
    levanter_out_rf_q = levanter_rotate_half(q, HeadSize)
    levanter_out_rf_k = levanter_rotate_half(k, HeadSize)

    q_tensor = named_array_to_tensor(q).transpose(1, 2)  # needed for HF
    k_tensor = named_array_to_tensor(k).transpose(1, 2)
    hf_out_rf_q = hf_rotate_half(q_tensor).transpose(1, 2)  # re-transpose to match levanter
    hf_out_rf_k = hf_rotate_half(k_tensor).transpose(1, 2)

    assert_equal_out(levanter_out_rf_q, hf_out_rf_q)
    assert_equal_out(levanter_out_rf_k, hf_out_rf_k)

    rot_config = _llama_config.rope
    rot = rot_config.build(HeadSize)
    hf_rot = LlamaRotaryEmbedding(_llama_config.to_hf_config(1000))

    position_ids = hax.arange(Pos)

    lev_rot_q = rot(q, position_ids)
    lev_rot_k = rot(k, position_ids)

    # actually apply the rotary embeddings
    hf_cos, hf_sin = hf_rot(q_tensor, named_array_to_tensor(position_ids).reshape(1, -1))
    hf_rot_q, hf_rot_k = hf_apply_rotary_pos_emb(q_tensor, k_tensor, hf_cos[:, :], hf_sin[:, :])
    hf_rot_q = hf_rot_q.transpose(1, 2)
    hf_rot_k = hf_rot_k.transpose(1, 2)

    assert_equal_out(lev_rot_q, hf_rot_q)
    assert_equal_out(lev_rot_k, hf_rot_k)


def test_yarn_rotary_embedding():
    """Test that YarnRotaryEmbeddings can be created and used."""
    from jax import random

    import haliax as hax

    from levanter.layers.rotary import YarnRotaryEmbeddingsConfig

    # Test configuration
    HeadSize = hax.Axis("HeadSize", 64)
    Pos = hax.Axis("Pos", 128)
    Batch = hax.Axis("batch", 2)
    Heads = hax.Axis("Heads", 8)

    # Create Yarn config
    yarn_config = YarnRotaryEmbeddingsConfig(
        theta=10000.0, factor=2.0, beta_fast=32.0, beta_slow=1.0, original_max_position_embeddings=2048, mscale=1.0
    )

    # Build the rotary embeddings
    rope = yarn_config.build(HeadSize)

    # Create test inputs
    q = hax.random.normal(random.PRNGKey(0), (Batch, Pos, Heads, HeadSize))
    position_ids = hax.arange(Pos)

    # Apply rotary embeddings
    q_rotated = rope(q, position_ids)

    # Basic assertions
    assert q_rotated.shape == q.shape
    assert q_rotated.axes == q.axes

    # Test that the output is different from input (rotary embeddings should modify the input)
    assert not hax.all(hax.isclose(q, q_rotated, rtol=1e-6, atol=1e-6))

    # Test HF config conversion
    theta, config = yarn_config.to_hf_config()
    assert theta == 10000.0
    assert config["rope_type"] == "yarn"
    assert config["factor"] == 2.0
    assert config["beta_fast"] == 32.0
    assert config["beta_slow"] == 1.0
    assert config["original_max_position_embeddings"] == 2048
    assert config["mscale"] == 1.0

    # Test creating from HF config
    yarn_config_from_hf = YarnRotaryEmbeddingsConfig.make_from_hf_config(theta, config)
    assert yarn_config_from_hf.theta == yarn_config.theta
    assert yarn_config_from_hf.factor == yarn_config.factor
    assert yarn_config_from_hf.beta_fast == yarn_config.beta_fast
    assert yarn_config_from_hf.beta_slow == yarn_config.beta_slow
    assert yarn_config_from_hf.original_max_position_embeddings == yarn_config.original_max_position_embeddings
    assert yarn_config_from_hf.mscale == yarn_config.mscale


@skip_if_no_torch
@pytest.mark.parametrize("factor", [1.0, 2.0, 4.0, 8.0])
def test_yarn_rotary_embedding_vs_hf(factor):
    """Test that YARN rotary embeddings match HuggingFace implementation."""
    import torch
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFLlamaRotaryEmbedding
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    from levanter.layers.rotary import YarnRotaryEmbeddingsConfig

    head_dim = 64
    seq_len = 32
    original_max_position_embeddings = 16  # Scaled down for testing
    theta = 500000.0

    HeadSize = hax.Axis("HeadSize", head_dim)
    Pos = hax.Axis("Pos", seq_len)
    Heads = hax.Axis("Heads", 4)
    Batch = hax.Axis("batch", 2)

    # Create YARN config
    yarn_config = YarnRotaryEmbeddingsConfig(
        theta=theta,
        factor=factor,
        beta_fast=32.0,
        beta_slow=1.0,
        original_max_position_embeddings=original_max_position_embeddings,
        mscale=1.0,
    )

    # Build Levanter YARN
    rope = yarn_config.build(HeadSize)

    # Create test inputs
    q = hax.random.normal(random.PRNGKey(0), (Batch, Pos, Heads, HeadSize))
    position_ids = hax.arange(Pos)

    # Apply Levanter YARN
    q_rotated = rope(q, position_ids)

    # Create HF config with YARN rope_scaling
    rope_theta, rope_scaling = yarn_config.to_hf_config()
    hf_config = LlamaConfig(
        hidden_size=head_dim * Heads.size,
        num_attention_heads=Heads.size,
        head_dim=head_dim,
        max_position_embeddings=seq_len * int(factor),
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
    )

    # Build HF rotary embeddings
    hf_rope = HFLlamaRotaryEmbedding(config=hf_config)

    # Apply HF rotary embeddings
    # HF expects shape (batch, heads, seq, head_dim), so transpose
    q_torch = torch.from_numpy(np.array(q.array)).transpose(1, 2)
    position_ids_torch = torch.arange(seq_len).reshape(1, -1)

    hf_cos, hf_sin = hf_rope(q_torch, position_ids_torch)
    hf_q_rotated, _ = apply_rotary_pos_emb(q_torch, q_torch, hf_cos, hf_sin)
    hf_q_rotated = hf_q_rotated.transpose(1, 2)  # Back to (batch, seq, heads, head_dim)

    # Compare
    np.testing.assert_allclose(
        np.array(q_rotated.array),
        hf_q_rotated.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"YARN rotary embeddings don't match HF for factor={factor}",
    )
