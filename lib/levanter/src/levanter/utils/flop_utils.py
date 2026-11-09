# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionFlops:
    """Forward FLOPs per token of the attention maps (scores, mask and value mixing), summed over layers."""

    noncausal: float
    """The count ``lm_flops_per_token`` includes: every query attends to its layer's whole span."""
    causal: float
    """Only the keys a causal mask keeps: ``min(i + 1, span)`` for the query at position ``i``."""


def _full_attention_layer_count(
    num_layers: int,
    seq_len: int,
    sliding_window: int | None,
    global_every: int | None,
    num_full_attention_layers: int | None,
) -> int | None:
    """Layers that attend over all of ``seq_len`` when the rest use ``sliding_window``; None if every layer does."""
    if num_full_attention_layers is not None and not 0 <= num_full_attention_layers <= num_layers:
        raise ValueError(f"num_full_attention_layers ({num_full_attention_layers}) must be in [0, {num_layers}]")
    if num_full_attention_layers is None and global_every is not None:
        num_full_attention_layers = num_layers // global_every
    if sliding_window is not None and num_full_attention_layers is not None and 0 < sliding_window < seq_len:
        return num_full_attention_layers
    return None


def _attention_map_flops(num_query_key_pairs: int, num_heads: int, head_dim: float) -> float:
    # key-query logits + mask + mask*value over every query-key pair, per the megatron-lm estimate.
    flops = (2 * num_query_key_pairs * num_heads * head_dim) + (3 * num_query_key_pairs * num_heads)
    return flops + 2 * num_query_key_pairs * head_dim * num_heads


def _causal_query_key_pairs(seq_len: int, attn_span: int) -> int:
    """Sum over query positions ``i`` of ``min(i + 1, attn_span)``."""
    return attn_span * (attn_span + 1) // 2 + (seq_len - attn_span) * attn_span


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
    num_full_attention_layers: int | None = None,
):
    """Analytic forward FLOPs per token.

    ``sliding_window`` + ``global_every`` model interleaved local/global attention: every
    ``global_every``-th layer runs full attention over ``seq_len`` while the rest attend only a
    ``sliding_window`` span. ``num_full_attention_layers`` gives that count directly for schedules
    that are not a plain stride, such as every fourth layer plus the last. ``local_kv_heads`` /
    ``global_kv_heads`` give those two layer classes different KV-head counts (heterogeneous GQA).
    Left at their defaults (``None``) every layer is full attention with ``num_kv_heads``, matching
    the original megatron-lm estimate exactly.
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
        # The attention map over ``attn_span`` keys per query, divided back to per-token.
        return _attention_map_flops(seq_len * attn_span, num_heads, head_dim) / seq_len

    num_global_layers = _full_attention_layer_count(
        num_layers, seq_len, sliding_window, global_every, num_full_attention_layers
    )
    if num_global_layers is not None:
        assert sliding_window is not None
        num_local_layers = num_layers - num_global_layers
        local_kv = local_kv_heads if local_kv_heads is not None else num_kv_heads
        global_kv = global_kv_heads if global_kv_heads is not None else num_kv_heads
        attn_total = num_global_layers * (_qkv_proj(global_kv) + _attn_per_token(seq_len))
        attn_total += num_local_layers * (_qkv_proj(local_kv) + _attn_per_token(sliding_window))
    else:
        attn_total = num_layers * (_qkv_proj(num_kv_heads) + _attn_per_token(seq_len))

    lm_head = 2 * hidden_dim * vocab_size
    return num_layers * (mlp + dense_proj) + attn_total + lm_head


def lm_attention_flops_per_token(
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    seq_len: int,
    sliding_window: int | None = None,
    global_every: int | None = None,
    num_full_attention_layers: int | None = None,
) -> AttentionFlops:
    """Forward attention-map FLOPs per token: the part of ``lm_flops_per_token`` that depends on the mask.

    Takes the same shape and layer-schedule arguments as ``lm_flops_per_token``. The QKV and output projections
    are not included. Subtracting ``noncausal`` from ``lm_flops_per_token`` and adding ``causal`` gives the count
    with causal masking charged exactly.
    """
    head_dim = hidden_dim / num_heads
    num_global_layers = _full_attention_layer_count(
        num_layers, seq_len, sliding_window, global_every, num_full_attention_layers
    )
    if num_global_layers is None:
        spans = [(num_layers, seq_len)]
    else:
        assert sliding_window is not None
        spans = [(num_global_layers, seq_len), (num_layers - num_global_layers, sliding_window)]
    noncausal = sum(n * _attention_map_flops(seq_len * span, num_heads, head_dim) for n, span in spans)
    causal = sum(
        n * _attention_map_flops(_causal_query_key_pairs(seq_len, span), num_heads, head_dim) for n, span in spans
    )
    return AttentionFlops(noncausal=noncausal / seq_len, causal=causal / seq_len)
