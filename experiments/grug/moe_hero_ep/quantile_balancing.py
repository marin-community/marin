# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quantile-balancing targets for the EP64 MoE hero experiment."""

from enum import StrEnum

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class QuantileBalancingMethod(StrEnum):
    """Available quantile-balancing estimators."""

    LOCAL_EXACT = "local_exact"
    GLOBAL_HISTOGRAM = "global_histogram"


def _histogram_bias_target(
    histogram: Int[Array, "E B"],
    current_bias: Float[Array, " E"],
    *,
    top_k: int,
) -> Float[Array, " E"]:
    num_experts, num_bins = histogram.shape
    global_tokens = jnp.sum(histogram[0], dtype=jnp.int32)
    target_rank = (global_tokens * top_k + num_experts - 1) // num_experts

    cumulative = jnp.cumsum(histogram, axis=-1, dtype=jnp.int32)
    selected_bin = jnp.argmax(cumulative >= target_rank, axis=-1).astype(jnp.int32)
    previous_bin = jnp.maximum(selected_bin - 1, 0)
    count_before = jnp.take_along_axis(cumulative, previous_bin[:, None], axis=-1)[:, 0]
    count_before = jnp.where(selected_bin > 0, count_before, 0)
    selected_count = jnp.take_along_axis(histogram, selected_bin[:, None], axis=-1)[:, 0]

    lower_bound = jnp.min(current_bias) - 1.0
    upper_bound = jnp.max(current_bias) + 1.0
    bin_width = (upper_bound - lower_bound) / num_bins
    fraction = (target_rank - count_before).astype(jnp.float32) / jnp.maximum(selected_count, 1)
    bias = lower_bound + (selected_bin.astype(jnp.float32) + fraction) * bin_width
    return bias - jnp.mean(bias)


def histogram_quantile_bias(
    required_bias: Float[Array, "T E"],
    current_bias: Float[Array, " E"],
    *,
    top_k: int,
    num_bins: int,
    reduce_axes: tuple[str, ...],
) -> Float[Array, " E"]:
    """Estimate the pooled QB target with one integer histogram reduction."""
    if required_bias.ndim != 2:
        raise ValueError(f"required_bias must have shape [tokens, experts], got {required_bias.shape}")
    expected_bias_shape = (required_bias.shape[1],)
    if current_bias.shape != expected_bias_shape:
        raise ValueError(f"current_bias must have shape {expected_bias_shape}, got {current_bias.shape}")
    if top_k <= 0 or top_k > required_bias.shape[1]:
        raise ValueError(f"top_k must be in [1, {required_bias.shape[1]}], got {top_k}")
    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2, got {num_bins}")

    required_bias = required_bias.astype(jnp.float32)
    current_bias = jax.lax.stop_gradient(current_bias.astype(jnp.float32))
    lower_bound = jnp.min(current_bias) - 1.0
    upper_bound = jnp.max(current_bias) + 1.0
    bin_width = (upper_bound - lower_bound) / num_bins
    bin_indices = jnp.floor((required_bias - lower_bound) / bin_width).astype(jnp.int32)
    bin_indices = jnp.clip(bin_indices, 0, num_bins - 1)

    num_experts = required_bias.shape[1]
    expert_indices = jnp.broadcast_to(jnp.arange(num_experts, dtype=jnp.int32), bin_indices.shape)
    flat_indices = expert_indices.reshape(-1) * num_bins + bin_indices.reshape(-1)
    histogram = jnp.bincount(flat_indices, length=num_experts * num_bins).reshape(num_experts, num_bins)
    if reduce_axes:
        histogram = jax.lax.psum(histogram, axis_name=reduce_axes)

    bias = _histogram_bias_target(histogram, current_bias, top_k=top_k)
    return jax.lax.stop_gradient(bias)
