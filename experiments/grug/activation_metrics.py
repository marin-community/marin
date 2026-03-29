# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared activation metric helpers for grug variants.

Computes per-block activation histograms and scalars gated by the existing
``compute_watch`` static JIT argument, so the fast path compiles a separate
kernel with zero overhead.
"""

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from levanter.tracker.histogram import Histogram

_DEFAULT_NUM_BINS = 31


def _sharding_safe_histogram(array: jax.Array, num_bins: int = _DEFAULT_NUM_BINS) -> Histogram:
    """Build a Histogram from a flat array without ``jnp.searchsorted``.

    ``Histogram.from_array`` relies on ``jnp.histogram`` → ``searchsorted``,
    which triggers sharding mismatches under an abstract mesh when the input
    is batch-sharded.  This implementation uses comparison-based binning that
    broadcasts cleanly across sharded axes.
    """
    array = array.ravel().astype(jnp.float32)
    min_val = array.min()
    max_val = array.max()
    num = array.size
    sum_val = array.sum()
    sum_squares = jnp.sum(array**2)

    edges = jnp.linspace(min_val, max_val, num_bins + 1)
    left = edges[:-1, None]  # (num_bins, 1)
    right = edges[1:, None]  # (num_bins, 1)
    a_row = array[None, :]  # (1, N)
    in_bin = (a_row >= left) & (a_row < right)
    counts = in_bin.sum(axis=1).astype(jnp.float32)
    # Last bin is inclusive on the right edge.
    counts = counts.at[-1].add(jnp.sum(array == max_val).astype(jnp.float32))
    return Histogram(
        min=min_val,
        max=max_val,
        num=num,
        sum=sum_val,
        sum_squares=sum_squares,
        bucket_limits=edges,
        bucket_counts=counts,
    )


def tokenwise_l2_norm(x: Float[Array, "B S D"]) -> Float[Array, "B S"]:
    """L2 norm per token (reduce over hidden dim), in float32."""
    return jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32)), axis=-1))


def block_activation_metrics(
    attn_in: Float[Array, "B S D"],
    attn_out: Float[Array, "B S D"],
    mlp_in: Float[Array, "B S D"],
    mlp_out: Float[Array, "B S D"],
    block_out: Float[Array, "B S D"],
    max_abs_attn_logit: jax.Array,
) -> dict[str, jax.Array | Histogram]:
    """Compute per-block activation metrics as histograms and scalars.

    All histogram inputs are tokenwise L2 norms (shape [B, S]), reducing from
    full activation tensors inside the block to avoid materializing [L, B, S, D]
    after a scan.
    """
    attn_in_norms = tokenwise_l2_norm(attn_in)
    attn_out_norms = tokenwise_l2_norm(attn_out)
    mlp_in_norms = tokenwise_l2_norm(mlp_in)
    mlp_out_norms = tokenwise_l2_norm(mlp_out)
    block_out_norms = tokenwise_l2_norm(block_out)

    return {
        "attn_in": _sharding_safe_histogram(attn_in_norms),
        "attn_out": _sharding_safe_histogram(attn_out_norms),
        "mlp_out": _sharding_safe_histogram(mlp_out_norms),
        "block_out": _sharding_safe_histogram(block_out_norms),
        "attn_out_to_in_ratio": _sharding_safe_histogram(
            attn_out_norms / jnp.maximum(attn_in_norms, 1e-8)
        ),
        "mlp_out_to_in_ratio": _sharding_safe_histogram(
            mlp_out_norms / jnp.maximum(mlp_in_norms, 1e-8)
        ),
        "max_abs_attn_logit": max_abs_attn_logit,
    }


def max_abs_attn_logit_from_qk(
    q: Float[Array, "B Q H D"],
    k: Float[Array, "B K H D"],
    head_dim: int,
) -> jax.Array:
    """Compute max |attention logit| by materializing [B, H, Q, K] transiently.

    Only called on watch steps, so the extra memory is acceptable.
    """
    scale = 1.0 / math.sqrt(head_dim)
    logits = jnp.einsum(
        "bqhd,bkhd->bhqk",
        (q * scale).astype(jnp.float32),
        k.astype(jnp.float32),
    )
    return jnp.max(jnp.abs(logits))


def flatten_per_layer_metrics(
    per_layer_metrics: list[dict[str, jax.Array | Histogram]],
    prefix: str = "activations",
) -> dict[str, jax.Array | Histogram]:
    """Flatten per-layer metric dicts into tracker-ready keys.

    Converts a list of per-layer dicts into a flat dict keyed by
    ``{prefix}/layer_{i}/{metric_name}``.
    """
    out: dict[str, jax.Array | Histogram] = {}
    for i, layer_metrics in enumerate(per_layer_metrics):
        for key, value in layer_metrics.items():
            out[f"{prefix}/layer_{i}/{key}"] = value
    return out


def unstack_activation_metrics(
    stacked: dict[str, jax.Array | Histogram],
    num_layers: int,
    prefix: str = "activations",
) -> dict[str, jax.Array | Histogram]:
    """Unstack vmapped/scanned activation metrics into per-layer tracker keys.

    If metrics were produced by ``jax.vmap`` or ``jax.lax.scan`` over layers,
    each value has an extra leading dimension of size ``num_layers``.  This
    helper splits that dimension into individual per-layer entries.

    For ``Histogram`` values, each scalar field (min, max, sum, …) and each
    array field (bucket_limits, bucket_counts) is expected to have the layer
    dimension as axis 0.
    """
    out: dict[str, jax.Array | Histogram] = {}
    for key, value in stacked.items():
        if isinstance(value, Histogram):
            for i in range(num_layers):
                out[f"{prefix}/layer_{i}/{key}"] = Histogram(
                    min=value.min[i],
                    max=value.max[i],
                    num=value.num[i] if isinstance(value.num, jax.Array) else value.num,
                    sum=value.sum[i],
                    sum_squares=value.sum_squares[i],
                    bucket_limits=value.bucket_limits[i] if value.bucket_limits.ndim > 1 else value.bucket_limits,
                    bucket_counts=value.bucket_counts[i],
                )
        elif isinstance(value, jax.Array) and value.ndim > 0 and value.shape[0] == num_layers:
            for i in range(num_layers):
                out[f"{prefix}/layer_{i}/{key}"] = value[i]
        else:
            out[f"{prefix}/{key}"] = value
    return out
