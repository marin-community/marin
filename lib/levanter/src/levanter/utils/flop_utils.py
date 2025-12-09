# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0


def lm_flops_per_token(
    hidden_dim: int,
    intermediate_dim: int,
    num_layers: int,
    num_kv_heads: int,
    num_heads: int,
    seq_len: int,
    vocab_size: int,
    glu: bool,
    num_experts: int = 1,
    num_shared_experts: int = 0,
    num_experts_per_tok: int = 1,
):
    head_dim = hidden_dim / num_heads
    mlp = 2 * (3 if glu else 2) * hidden_dim * intermediate_dim * (num_experts_per_tok + num_shared_experts)
    if num_experts > 1:
        mlp += 2 * hidden_dim * num_experts  # router layer
    qkv_proj = 2 * hidden_dim * (num_heads * head_dim + 2 * num_kv_heads * head_dim)
    dense_proj = 2 * hidden_dim * hidden_dim
    # The following are across the whole sequence
    # assume full attention map like megatron-lm
    key_query_logits = 2 * seq_len**2 * num_heads * head_dim
    mask = 3 * seq_len * seq_len * num_heads
    mask_value = 2 * seq_len * seq_len * head_dim * num_heads
    seq_flops = key_query_logits + mask + mask_value
    # so we divide by the sequence length to get the per-token flops
    attn = seq_flops / seq_len
    lm_head = 2 * hidden_dim * vocab_size
    return num_layers * (mlp + qkv_proj + dense_proj + attn) + lm_head
