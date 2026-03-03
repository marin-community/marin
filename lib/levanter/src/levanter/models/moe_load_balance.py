# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def quantile_balancing_bias_target(
    router_logits: Array,
    selection_bias: Array,
    *,
    top_k: int,
    num_iterations: int = 5,
) -> tuple[Array, Array, Array, Array]:
    """Compute a Quantile Balancing (QB) target for the routing selection bias.

    This follows the alternating quantile scheme from MoE Odyssey:
      1) alpha_i <- Quantile(s_i + b, 1 - k / n) over experts
      2) beta_j  <- Quantile(s - alpha, 1 - k / n) over tokens
      3) b <- -beta

    where:
      - s has shape [tokens, experts]
      - b has shape [experts]
      - k is top-k experts per token
      - n is number of experts

    Args:
      router_logits: `[tokens, experts]` pre-bias router scores.
      selection_bias: `[experts]` additive bias used for current-step selection.
      top_k: Number of experts selected per token.
      num_iterations: Number of alternating quantile updates to run.

    Returns:
      `(target_bias, expert_counts, target_count, quantile_level)`:
      - `target_bias`: `[experts]` additive bias target for selection logits.
      - `expert_counts`: `[experts]` binary top-k assignment counts under current bias.
      - `target_count`: scalar desired assignment count per expert (`tokens * k / experts`).
      - `quantile_level`: scalar `1 - k / experts` used by both quantile steps.
    """
    if router_logits.ndim != 2:
        raise ValueError(f"router_logits must be rank-2 [tokens, experts], got {router_logits.shape}")
    if selection_bias.ndim != 1:
        raise ValueError(f"selection_bias must be rank-1 [experts], got {selection_bias.shape}")

    tokens, experts = router_logits.shape
    if selection_bias.shape[0] != experts:
        raise ValueError(f"selection_bias has size {selection_bias.shape[0]} but router_logits has experts={experts}")
    if not (1 <= top_k <= experts):
        raise ValueError(f"top_k must satisfy 1 <= top_k <= experts, got top_k={top_k}, experts={experts}")
    if num_iterations < 1:
        raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")

    scores = router_logits.astype(jnp.float32)
    bias = selection_bias.astype(jnp.float32)
    quantile_level_float = 1.0 - top_k / float(experts)

    for _ in range(num_iterations):
        adjusted = scores + bias[None, :]
        alpha = jnp.quantile(adjusted, quantile_level_float, axis=1, keepdims=True)
        beta = jnp.quantile(scores - alpha, quantile_level_float, axis=0, keepdims=False)
        bias = -beta

    current_adjusted = scores + selection_bias.astype(jnp.float32)[None, :]
    topk_idx = jax.lax.top_k(current_adjusted, top_k)[1].astype(jnp.int32)
    expert_counts = jnp.bincount(topk_idx.reshape(-1), length=experts).astype(jnp.float32)
    target_count = jnp.asarray(tokens * top_k / float(experts), dtype=jnp.float32)
    quantile_level = jnp.asarray(quantile_level_float, dtype=jnp.float32)

    return bias.astype(selection_bias.dtype), expert_counts, target_count, quantile_level
