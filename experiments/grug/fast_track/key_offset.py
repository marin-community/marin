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
    own_weight: Float[Array, "N E"] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Each key token's value also reads the next token's, ``v'_t = own_weight * v_t + ahead * v_{t+1}``
    (``own_weight`` 1 if None); keys and values then
    move one slot later (zero at document starts), so under the causal mask query ``i`` sees tokens ``t <= i - 1``
    (never itself, so ``v_{t+1}`` never leaks the future) and every document's first slot is an empty sink: zero
    value, and key ``k_null`` (zero if None)."""
    spec_k, spec_v = jax.typeof(k).sharding.spec, jax.typeof(v).sharding.spec
    own = v if own_weight is None else own_weight.astype(v.dtype) * v
    mixed = own + ahead.astype(v.dtype) * next_token(v, segment_ids)
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


@named_call
def future_heads(
    qf: Float[Array, "B S F D"],
    k: Float[Array, "B S N D"],
    v: Float[Array, "B S N E"],
    segment_ids: Int[Array, "B S"] | None,
    window: int,
) -> tuple[jax.Array, jax.Array]:
    """The last ``F`` heads remember each token's future: token ``t`` attends over its own document's next ``window``
    tokens with query ``qf_t`` (the head's keys / values, plus a zero null slot) to ``f_t``; those heads store
    ``v_t + f_t`` and their keys / values move ``window`` slots later, so query ``i`` reads only tokens
    ``t <= i - window`` there (``f_t`` covers tokens ``<= i``: no leakage)."""
    spec_k, spec_v = jax.typeof(k).sharding.spec, jax.typeof(v).sharding.spec
    heads = qf.shape[2]
    kf, vf = k[:, :, -heads:].astype(jnp.float32), v[:, :, -heads:].astype(jnp.float32)
    qf = qf.astype(jnp.float32) / math.sqrt(qf.shape[-1])
    ahead_k = [shift_tokens(kf, segment_ids, -o) for o in range(1, window + 1)]
    ahead_v = [shift_tokens(vf, segment_ids, -o) for o in range(1, window + 1)]
    valid = [shift_tokens(jnp.ones(k.shape[:2], bool), segment_ids, -o) for o in range(1, window + 1)]
    scores = [jnp.where(m[..., None], jnp.sum(qf * ko, axis=-1), -jnp.inf) for ko, m in zip(ahead_k, valid, strict=True)]
    logits = jnp.stack([*scores, jnp.zeros_like(scores[0])], axis=-1)  # [B, S, F, window + 1]; last: the null slot
    probs = jax.nn.softmax(logits, axis=-1)
    future = sum(probs[..., o, None] * vo for o, vo in enumerate(ahead_v))
    k_out = jnp.concatenate([k[:, :, :-heads], shift_tokens(k[:, :, -heads:], segment_ids, window)], axis=2)
    v_mem = (vf + future).astype(v.dtype)
    v_out = jnp.concatenate([v[:, :, :-heads], shift_tokens(v_mem, segment_ids, window)], axis=2)
    return reshard(k_out, spec_k), reshard(v_out, spec_v)


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
