# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def equilibrium_bias_delta_from_topk(
    adjusted_router_logits: Array,
    topk_idx: Array,
    topk_weights: Array,
    *,
    unit_capacity: float = 1.0,
) -> tuple[Array, Array, Array, Array]:
    """Compute one quantile-balancing bias update from top-k routing.

    This implements the single-step column update from Quantile Balancing:
      - build per-token threshold from the K-th largest adjusted logit,
      - compute per-expert overload residuals,
      - compute target-aware quantiles from current expert loads,
      - return per-expert bias deltas.

    Args:
      adjusted_router_logits: `[tokens, experts]` logits after adding selection bias.
      topk_idx: `[tokens, k]` selected expert indices.
      topk_weights: `[tokens, k]` routing weights for selected experts.
      unit_capacity: Upper-bound offset in the clipped allocation residual term.

    Returns:
      A tuple `(delta, expert_weighted_load, target_load, quantile_prob)` where:
      - `delta`: `[experts]` bias update to subtract (`b <- b - delta`),
      - `expert_weighted_load`: `[experts]` current weighted load,
      - `target_load`: scalar target load per expert (`tokens / experts`),
      - `quantile_prob`: `[experts]` quantile level used per expert.
    """
    if adjusted_router_logits.ndim != 2:
        raise ValueError(
            f"adjusted_router_logits must be rank-2 [tokens, experts], got {adjusted_router_logits.shape}"
        )
    if topk_idx.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("topk_idx and topk_weights must be rank-2 [tokens, k]")
    if topk_idx.shape != topk_weights.shape:
        raise ValueError(f"topk_idx shape {topk_idx.shape} must match topk_weights shape {topk_weights.shape}")

    tokens, experts = adjusted_router_logits.shape
    if topk_idx.shape[0] != tokens:
        raise ValueError("topk tensors must have same token dimension as adjusted_router_logits")

    k = topk_idx.shape[1]
    logits = adjusted_router_logits.astype(jnp.float32)
    idx = topk_idx.astype(jnp.int32)
    weights = topk_weights.astype(jnp.float32)

    kth = jax.lax.top_k(logits, k)[0][:, -1]
    residual = jnp.maximum(logits - kth[:, None] - unit_capacity, 0.0)

    flat_idx = idx.reshape(-1)
    flat_weights = weights.reshape(-1)
    expert_weighted_load = jnp.bincount(flat_idx, weights=flat_weights, length=experts)

    tokens_f = jnp.asarray(tokens, dtype=jnp.float32)
    target_load = tokens_f / float(experts)
    quantile_prob = jnp.clip(1.0 - (expert_weighted_load - target_load) / tokens_f, 0.0, 1.0)

    sorted_residual = jnp.sort(residual, axis=0)
    quantile_pos = jnp.floor(quantile_prob * float(tokens - 1)).astype(jnp.int32)
    quantile_pos = jnp.clip(quantile_pos, 0, tokens - 1)
    expert_ids = jnp.arange(experts, dtype=jnp.int32)
    delta = sorted_residual[quantile_pos, expert_ids]

    return delta.astype(adjusted_router_logits.dtype), expert_weighted_load, target_load, quantile_prob
