# Copyright The Levanter Authors
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
    shared_intermediate_dim: int | None = None,
    head_dim: int | None = None,
):
    # head_dim defaults to the standard MHA sizing (hidden_dim / num_heads); pass it explicitly for
    # models that decouple attention width from hidden_dim (e.g. num_heads * head_dim != hidden_dim).
    resolved_head_dim = hidden_dim // num_heads if head_dim is None else head_dim
    attn_dim = num_heads * resolved_head_dim
    shared_intermediate_dim = intermediate_dim if shared_intermediate_dim is None else shared_intermediate_dim
    routed_mlp = 2 * (3 if glu else 2) * hidden_dim * intermediate_dim * num_experts_per_tok
    shared_mlp = 2 * (3 if glu else 2) * hidden_dim * shared_intermediate_dim * num_shared_experts
    mlp = routed_mlp + shared_mlp
    if num_experts > 1:
        mlp += 2 * hidden_dim * num_experts  # router layer
    qkv_proj = 2 * hidden_dim * (num_heads * resolved_head_dim + 2 * num_kv_heads * resolved_head_dim)
    # output projection maps the concatenated head outputs (num_heads * head_dim) back to hidden_dim
    dense_proj = 2 * attn_dim * hidden_dim
    # The following are across the whole sequence
    # assume full attention map like megatron-lm
    key_query_logits = 2 * seq_len**2 * num_heads * resolved_head_dim
    mask = 3 * seq_len * seq_len * num_heads
    mask_value = 2 * seq_len * seq_len * resolved_head_dim * num_heads
    seq_flops = key_query_logits + mask + mask_value
    # so we divide by the sequence length to get the per-token flops
    attn = seq_flops / seq_len
    lm_head = 2 * hidden_dim * vocab_size
    return num_layers * (mlp + qkv_proj + dense_proj + attn) + lm_head
