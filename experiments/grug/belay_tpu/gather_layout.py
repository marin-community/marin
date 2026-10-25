# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare representations for the wide row gathers in local-token MoE."""

import jax
import jax.numpy as jnp


def packed_gather(rows, indices, *, barrier=False):
    """Gather pairs of BF16 features as one 32-bit word, preserving all bits."""
    if rows.ndim != 2 or rows.dtype != jnp.bfloat16 or rows.shape[1] % 2:
        raise ValueError("Packed gather requires a BF16 matrix with even feature width")
    words = jax.lax.bitcast_convert_type(rows.reshape(rows.shape[0], -1, 2), jnp.int32)
    if barrier:
        words = jax.lax.optimization_barrier(words)
    gathered = words[indices]
    return jax.lax.bitcast_convert_type(gathered, rows.dtype).reshape(indices.shape[0], rows.shape[1])


def fp32_gather(rows, indices):
    """Force an FP32 intermediate to measure the cost of a different layout."""
    expanded = jax.lax.optimization_barrier(rows.astype(jnp.float32))
    return expanded[indices].astype(rows.dtype)
