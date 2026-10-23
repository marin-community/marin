# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MLA partial key offset (``mla_key_offset``): half of each head's key channels come from the previous token."""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Float, Int


def _shifted_channels(heads: int, head_dim: int) -> jax.Array:
    """``[H, D]`` mask of the key channels taken from the previous token: the first half of every head."""
    counts = jnp.asarray((head_dim // 2,) * heads)
    return jnp.arange(head_dim)[None, :] < counts[:, None]


def previous_token(x: jax.Array, segment_ids: Int[Array, "B S"] | None) -> jax.Array:
    """``x`` shifted one token later along axis 1, zero at document starts."""
    prev = jnp.pad(x[:, :-1], ((0, 0), (1, 0)) + ((0, 0),) * (x.ndim - 2))
    if segment_ids is None:
        return prev
    starts = jnp.pad(segment_ids[:, 1:] != segment_ids[:, :-1], ((0, 0), (1, 0)), constant_values=True)
    return jnp.where(starts.reshape(starts.shape + (1,) * (x.ndim - 2)), 0, prev)


def partial_key_offset(k: Float[Array, "B S H D"], segment_ids: Int[Array, "B S"] | None, spec: P | None) -> jax.Array:
    """Replace the first half of each head's key channels with the previous token's."""
    shifted = _shifted_channels(k.shape[2], k.shape[3])
    return reshard(jnp.where(shifted, previous_token(k, segment_ids).astype(k.dtype), k), spec)
