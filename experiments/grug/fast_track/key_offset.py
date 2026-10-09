# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MLA token-shift features: the partial key offset, the appended shared match score and the value lookahead."""


import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Float, Int


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


def shift_tokens(x: jax.Array, segment_ids: Int[Array, "B S"] | None, n: int) -> jax.Array:
    """``x`` moved ``n`` tokens later along axis 1 (``n < 0``: earlier), zero where the source token is outside the
    sequence or in another document."""
    if n == 0:
        return x
    pad = ((0, 0), (n, 0)) if n > 0 else ((0, 0), (0, -n))
    src = x[:, :-n] if n > 0 else x[:, -n:]
    out = jnp.pad(src, pad + ((0, 0),) * (x.ndim - 2))
    seq = x.shape[1]
    valid = jnp.broadcast_to((jnp.arange(seq) - n >= 0) & (jnp.arange(seq) - n < seq), x.shape[:2])
    if segment_ids is not None:
        moved = jnp.pad(segment_ids[:, :-n] if n > 0 else segment_ids[:, -n:], pad, constant_values=-1)
        valid = valid & (moved == segment_ids)
    return jnp.where(valid.reshape(valid.shape + (1,) * (x.ndim - 2)), out, 0)


def parse_lag_groups(spec: tuple[str, ...], head_dim: int) -> tuple[tuple[int, int, int], ...]:
    """``"count:q_lag:k_lag"`` channel groups (``mla_lag_groups``), checked to tile the head."""
    groups = tuple(tuple(int(p) for p in item.split(":")) for item in spec)
    if any(len(g) != 3 or g[0] <= 0 or g[1] < 0 or g[2] < 0 for g in groups):
        raise ValueError(f"mla_lag_groups entries are count:q_lag:k_lag with count > 0, lags >= 0; got {spec}")
    if sum(g[0] for g in groups) != head_dim:
        raise ValueError(f"mla_lag_groups counts must sum to the head dim {head_dim}; got {spec}")
    return groups  # pyrefly: ignore[bad-return]


def _lag_channels(x: jax.Array, segment_ids: Int[Array, "B S"] | None, lags: list[int]) -> jax.Array:
    """Channel ``c`` of every head taken from ``lags[c]`` tokens back (zero across document starts)."""
    lag_of = jnp.asarray(lags)
    out = x
    for lag in sorted(set(lags) - {0}):
        out = jnp.where(lag_of == lag, shift_tokens(x, segment_ids, lag).astype(x.dtype), out)
    return reshard(out, jax.typeof(x).sharding.spec)


def apply_lag_groups(
    q: Float[Array, "B S N D"],
    k: Float[Array, "B S N D"],
    segment_ids: Int[Array, "B S"] | None,
    groups: tuple[tuple[int, int, int], ...],
) -> tuple[jax.Array, jax.Array]:
    """Each channel group ``(count, a, b)`` scores the query's token ``i - a`` against the key's token ``t - b``
    (``mla_lag_groups``; the key offset is ``(64, 0, 1), (64, 0, 0)``). Values are never shifted: only past tokens
    of the query are read, so nothing leaks."""
    q_lags = [a for count, a, _ in groups for _ in range(count)]
    k_lags = [b for count, _, b in groups for _ in range(count)]
    return _lag_channels(q, segment_ids, q_lags), _lag_channels(k, segment_ids, k_lags)
