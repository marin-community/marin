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
    num_full_attention_layers: int | None = None,
):
    """Analytic forward FLOPs per token for an LM.

    Attention is counted as full causal across ``seq_len`` on every layer by
    default. If ``sliding_window`` is set, layers using windowed attention only
    pay ``min(seq_len, sliding_window)`` for the attention "sequence" -- the
    per-token attention FLOPs scale with the effective attention span, not the
    full sequence. Pass ``num_full_attention_layers`` to split the model
    between windowed and full-attention layers (e.g. a hybrid where every 4th
    layer is full, the rest are sliding window); the remaining
    ``num_layers - num_full_attention_layers`` layers use windowed attention.
    """
    head_dim = hidden_dim / num_heads
    shared_intermediate_dim = intermediate_dim if shared_intermediate_dim is None else shared_intermediate_dim
    routed_mlp = 2 * (3 if glu else 2) * hidden_dim * intermediate_dim * num_experts_per_tok
    shared_mlp = 2 * (3 if glu else 2) * hidden_dim * shared_intermediate_dim * num_shared_experts
    mlp = routed_mlp + shared_mlp
    if num_experts > 1:
        mlp += 2 * hidden_dim * num_experts  # router layer
    qkv_proj = 2 * hidden_dim * (num_heads * head_dim + 2 * num_kv_heads * head_dim)
    dense_proj = 2 * hidden_dim * hidden_dim

    def _attn_per_token(effective_seq: int) -> float:
        # Full-sequence FLOPs at this effective span, then normalise per token
        # (assumes Megatron-style causal attention accounting).
        key_query_logits = 2 * effective_seq**2 * num_heads * head_dim
        mask = 3 * effective_seq * effective_seq * num_heads
        mask_value = 2 * effective_seq * effective_seq * head_dim * num_heads
        return (key_query_logits + mask + mask_value) / effective_seq

    if sliding_window is None:
        n_full = num_layers
        n_window = 0
        effective_window_seq = seq_len
    else:
        n_full = num_full_attention_layers if num_full_attention_layers is not None else 0
        if n_full < 0 or n_full > num_layers:
            raise ValueError(f"num_full_attention_layers ({n_full}) must be in [0, {num_layers}]")
        n_window = num_layers - n_full
        effective_window_seq = min(seq_len, sliding_window)

    attn_full = _attn_per_token(seq_len) if n_full else 0.0
    attn_window = _attn_per_token(effective_window_seq) if n_window else 0.0
    per_layer_dense = mlp + qkv_proj + dense_proj
    lm_head = 2 * hidden_dim * vocab_size
    return num_layers * per_layer_dense + n_full * attn_full + n_window * attn_window + lm_head
