# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax.numpy as jnp
from jax import random

import haliax as hax

from levanter.layers.attention import AttentionMask
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel


def test_llama_pause_tokens_forward():
    """Test that pause tokens work with expansion_factor > 1"""
    config = LlamaConfig(
        max_seq_len=64,
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        num_kv_heads=2,
        expansion_factor=3,  # 1 real token + 2 pause tokens
        pause_token_id=128072,
        pause_aggregate_method="last",
        gradient_checkpointing=False,
    )
    
    Batch = hax.Axis("batch", 2)
    Vocab = hax.Axis("vocab", 1000)
    Pos = hax.Axis("position", 16)  # Original sequence length
    
    # Create input_ids
    input_ids = hax.random.randint(random.PRNGKey(0), (Batch, Pos), 0, Vocab.size)
    mask = AttentionMask.causal()
    
    # Initialize model
    model = LlamaLMHeadModel.init(Vocab=Vocab, config=config, key=random.PRNGKey(0))
    
    # Forward pass
    out = model(input_ids, attn_mask=mask, pos_ids=None)
    
    # Check output shape: should be [batch, original_pos, vocab]
    # The expansion happens internally, but output should be collapsed back
    assert out.array.shape == (Batch.size, Pos.size, Vocab.size)
    assert out.axes[0].name == "batch"
    assert out.axes[1].name == "position"
    assert out.axes[1].size == Pos.size  # Should be original position size
    assert out.axes[2].name == "vocab"


def test_llama_pause_tokens_mean_aggregation():
    """Test pause tokens with mean aggregation"""
    config = LlamaConfig(
        max_seq_len=64,
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        num_kv_heads=2,
        expansion_factor=2,  # 1 real token + 1 pause token
        pause_token_id=128072,
        pause_aggregate_method="mean",
        gradient_checkpointing=False,
    )
    
    Batch = hax.Axis("batch", 1)
    Vocab = hax.Axis("vocab", 500)
    Pos = hax.Axis("position", 8)
    
    input_ids = hax.random.randint(random.PRNGKey(1), (Batch, Pos), 0, Vocab.size)
    mask = AttentionMask.causal()
    
    model = LlamaLMHeadModel.init(Vocab=Vocab, config=config, key=random.PRNGKey(1))
    out = model(input_ids, attn_mask=mask, pos_ids=None)
    
    # Check output shape
    assert out.array.shape == (Batch.size, Pos.size, Vocab.size)


def test_llama_pause_tokens_no_expansion():
    """Test that expansion_factor=1 works normally (no pause tokens)"""
    config = LlamaConfig(
        max_seq_len=64,
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        num_kv_heads=2,
        expansion_factor=1,  # No pause tokens
        gradient_checkpointing=False,
    )
    
    Batch = hax.Axis("batch", 2)
    Vocab = hax.Axis("vocab", 1000)
    Pos = hax.Axis("position", 16)
    
    input_ids = hax.random.randint(random.PRNGKey(2), (Batch, Pos), 0, Vocab.size)
    mask = AttentionMask.causal()
    
    model = LlamaLMHeadModel.init(Vocab=Vocab, config=config, key=random.PRNGKey(2))
    out = model(input_ids, attn_mask=mask, pos_ids=None)
    
    # Should work normally without pause tokens
    assert out.array.shape == (Batch.size, Pos.size, Vocab.size)


def test_llama_pause_tokens_pos_ids_assertion():
    """Test that pos_ids must be None when expansion_factor > 1"""
    config = LlamaConfig(
        max_seq_len=64,
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        num_kv_heads=2,
        expansion_factor=3,
        gradient_checkpointing=False,
    )
    
    Batch = hax.Axis("batch", 1)
    Vocab = hax.Axis("vocab", 500)
    Pos = hax.Axis("position", 8)
    
    input_ids = hax.random.randint(random.PRNGKey(3), (Batch, Pos), 0, Vocab.size)
    pos_ids = hax.arange(Pos, dtype=jnp.int32)
    mask = AttentionMask.causal()
    
    model = LlamaLMHeadModel.init(Vocab=Vocab, config=config, key=random.PRNGKey(3))
    
    # Should raise assertion error
    with pytest.raises(AssertionError, match="pos_ids must be None when expansion_factor > 1"):
        model(input_ids, attn_mask=mask, pos_ids=pos_ids)


@pytest.mark.parametrize("expansion_factor", [2, 3, 4])
def test_llama_pause_tokens_various_expansion_factors(expansion_factor):
    """Test pause tokens with various expansion factors"""
    config = LlamaConfig(
        max_seq_len=64,
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        num_kv_heads=2,
        expansion_factor=expansion_factor,
        pause_token_id=128072,
        pause_aggregate_method="last",
        gradient_checkpointing=False,
    )
    
    Batch = hax.Axis("batch", 1)
    Vocab = hax.Axis("vocab", 500)
    Pos = hax.Axis("position", 8)
    
    input_ids = hax.random.randint(random.PRNGKey(4), (Batch, Pos), 0, Vocab.size)
    mask = AttentionMask.causal()
    
    model = LlamaLMHeadModel.init(Vocab=Vocab, config=config, key=random.PRNGKey(4))
    out = model(input_ids, attn_mask=mask, pos_ids=None)
    
    # Output should always be original shape
    assert out.array.shape == (Batch.size, Pos.size, Vocab.size)
