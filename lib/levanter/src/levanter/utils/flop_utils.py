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
    sliding_window: int | None = None,
    global_every: int | None = None,
    local_kv_heads: int | None = None,
    global_kv_heads: int | None = None,
):
    """Analytic forward FLOPs per token.

    ``sliding_window`` + ``global_every`` model interleaved local/global attention: every
    ``global_every``-th layer runs full attention over ``seq_len`` while the rest attend only a
    ``sliding_window`` span. ``local_kv_heads`` / ``global_kv_heads`` give those two layer classes
    different KV-head counts (heterogeneous GQA). Left at their defaults (``None``) every layer is full
    attention with ``num_kv_heads``, matching the original megatron-lm estimate exactly.
    """
    head_dim = hidden_dim / num_heads
    shared_intermediate_dim = intermediate_dim if shared_intermediate_dim is None else shared_intermediate_dim
    routed_mlp = 2 * (3 if glu else 2) * hidden_dim * intermediate_dim * num_experts_per_tok
    shared_mlp = 2 * (3 if glu else 2) * hidden_dim * shared_intermediate_dim * num_shared_experts
    mlp = routed_mlp + shared_mlp
    if num_experts > 1:
        mlp += 2 * hidden_dim * num_experts  # router layer
    dense_proj = 2 * hidden_dim * hidden_dim

    def _qkv_proj(kv_heads: int) -> float:
        return 2 * hidden_dim * (num_heads * head_dim + 2 * kv_heads * head_dim)

    def _attn_per_token(attn_span: int) -> float:
        # key-query logits + mask + mask*value over ``attn_span`` keys per query, per the megatron-lm
        # estimate, divided back to per-token. With attn_span == seq_len this is the full-attention map.
        seq_flops = (2 * seq_len * attn_span * num_heads * head_dim) + (3 * seq_len * attn_span * num_heads)
        seq_flops += 2 * seq_len * attn_span * head_dim * num_heads
        return seq_flops / seq_len

    if sliding_window is not None and global_every is not None and 0 < sliding_window < seq_len:
        num_global_layers = num_layers // global_every
        num_local_layers = num_layers - num_global_layers
        local_kv = local_kv_heads if local_kv_heads is not None else num_kv_heads
        global_kv = global_kv_heads if global_kv_heads is not None else num_kv_heads
        attn_total = num_global_layers * (_qkv_proj(global_kv) + _attn_per_token(seq_len))
        attn_total += num_local_layers * (_qkv_proj(local_kv) + _attn_per_token(sliding_window))
    else:
        attn_total = num_layers * (_qkv_proj(num_kv_heads) + _attn_per_token(seq_len))

    lm_head = 2 * hidden_dim * vocab_size
    return num_layers * (mlp + dense_proj) + attn_total + lm_head
