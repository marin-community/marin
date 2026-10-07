# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MLA token-shift features: the partial key offset, the appended shared match score and the value lookahead."""

import math
from enum import StrEnum

import jax
import jax.numpy as jnp
from haliax.jax_utils import named_call
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Float, Int

# The attention kernel's head-dim granule.
_KERNEL_HEAD_DIM_GRANULE = 32


class MatchAppend(StrEnum):
    """How ``mla_shared_match_dim`` enters the MLA scores."""

    OFF = "off"
    """``m`` is every head's shifted half, for the key (shifted) and query (unshifted), with ``mla_key_offset``:
    per head a learned scale (M = half the head) or a ``match_head`` map (M > half)."""
    SCALE = "scale"
    """Appended to full un-shifted heads: each head adds ``g_h <m_i, m_{j-1}> / sqrt(M)``, ``g_h`` a learned scalar."""
    TOKEN = "token"
    """As ``SCALE`` with ``g_h(x_i) = g_h * 2 sigmoid(x_i . w_h)`` (``w`` zero-init): each query sets how much it
    matches."""


def shifted_channels(heads: int, head_dim: int, counts: tuple[int, ...]) -> jax.Array:
    """``[H, D]`` mask of the key channels taken from the previous token (the first ``counts[h]``; half by default)."""
    counts_arr = jnp.asarray(counts if counts else (head_dim // 2,) * heads)
    if counts_arr.shape[0] != heads:
        raise ValueError(f"mla_key_offset_channels needs one entry per head ({heads}), got {counts}")
    return jnp.arange(head_dim)[None, :] < counts_arr[:, None]


def previous_token(x: jax.Array, segment_ids: Int[Array, "B S"] | None) -> jax.Array:
    """``x`` shifted one token later along axis 1, zero at document starts."""
    prev = jnp.pad(x[:, :-1], ((0, 0), (1, 0)) + ((0, 0),) * (x.ndim - 2))
    if segment_ids is None:
        return prev
    starts = jnp.pad(segment_ids[:, 1:] != segment_ids[:, :-1], ((0, 0), (1, 0)), constant_values=True)
    return jnp.where(starts.reshape(starts.shape + (1,) * (x.ndim - 2)), 0, prev)


def partial_key_offset(
    k: Float[Array, "B S H D"], segment_ids: Int[Array, "B S"] | None, spec: P | None, counts: tuple[int, ...] = ()
) -> jax.Array:
    """Replace each head's leading key channels (``shifted_channels``) with the previous token's."""
    shifted = shifted_channels(k.shape[2], k.shape[3], counts)
    return reshard(jnp.where(shifted, previous_token(k, segment_ids).astype(k.dtype), k), spec)


@named_call
def append_shared_match(
    q: Float[Array, "B S N H"],
    k: Float[Array, "B S N H"],
    v: Float[Array, "B S N H"],
    match: Float[Array, "B S M"],
    gate: jax.Array,
    segment_ids: Int[Array, "B S"] | None,
    head_spec: P,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Extend every head by the ``M`` match channels so the kernel's ``q'.k' / sqrt(H + M)`` is
    ``q.k / sqrt(H) + gate <match_i, match_{j-1}> / sqrt(M)``; ``v`` is zero-padded (slice the output to ``H``).

    ``gate`` is ``[N]`` or ``[B, S, N]``; ``head_spec`` is the ``[B, S, N, *]`` spec of the extended arrays."""
    head_dim, match_dim = q.shape[-1], match.shape[-1]
    padded = head_dim + match_dim
    if padded % _KERNEL_HEAD_DIM_GRANULE:
        raise ValueError(f"head_dim + mla_shared_match_dim must be a multiple of {_KERNEL_HEAD_DIM_GRANULE}")
    heads = q.shape[2]
    q_tail = gate[..., :, None].astype(jnp.float32) * match[..., None, :].astype(jnp.float32)
    q_tail = reshard(jnp.broadcast_to(q_tail * math.sqrt(padded / match_dim), (*q.shape[:-1], match_dim)), head_spec)
    prev = previous_token(match, segment_ids)
    k_tail = reshard(jnp.broadcast_to(prev[..., None, :], (*k.shape[:2], heads, match_dim)), head_spec)
    q = jnp.concatenate([reshard(q * math.sqrt(padded / head_dim), head_spec), q_tail.astype(q.dtype)], axis=-1)
    k = jnp.concatenate([reshard(k, head_spec), k_tail.astype(k.dtype)], axis=-1)
    return q, k, reshard(jnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, match_dim))), head_spec)


def next_token(x: jax.Array, segment_ids: Int[Array, "B S"] | None) -> jax.Array:
    """``x`` shifted one token earlier along axis 1, zero at document ends."""
    nxt = jnp.pad(x[:, 1:], ((0, 0), (0, 1)) + ((0, 0),) * (x.ndim - 2))
    if segment_ids is None:
        return nxt
    ends = jnp.pad(segment_ids[:, 1:] != segment_ids[:, :-1], ((0, 0), (0, 1)), constant_values=True)
    return jnp.where(ends.reshape(ends.shape + (1,) * (x.ndim - 2)), 0, nxt)


@named_call
def value_lookahead(
    k: Float[Array, "B S N D"],
    v: Float[Array, "B S N E"],
    ahead: Float[Array, "N E"],
    segment_ids: Int[Array, "B S"] | None,
    k_null: Float[Array, "N D"] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Each key token's value also reads the next token's, ``v'_t = v_t + ahead * v_{t+1}``; keys and values then
    move one slot later (zero at document starts), so under the causal mask query ``i`` sees tokens ``t <= i - 1``
    (never itself, so ``v_{t+1}`` never leaks the future) and every document's first slot is an empty sink: zero
    value, and key ``k_null`` (zero if None)."""
    spec_k, spec_v = jax.typeof(k).sharding.spec, jax.typeof(v).sharding.spec
    mixed = v + ahead.astype(v.dtype) * next_token(v, segment_ids)
    shifted_k = previous_token(k, segment_ids)
    if k_null is not None:
        first = jnp.zeros(k.shape[:2], bool).at[:, 0].set(True)
        if segment_ids is not None:
            first = first | jnp.pad(segment_ids[:, 1:] != segment_ids[:, :-1], ((0, 0), (1, 0)))
        shifted_k = jnp.where(first[..., None, None], k_null.astype(k.dtype), shifted_k)
    return (
        reshard(shifted_k, spec_k),
        reshard(previous_token(mixed, segment_ids).astype(v.dtype), spec_v),
    )
