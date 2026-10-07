# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local-window + learned sparse-index attention for the MLA layers, as a dense masked emulation.

Each query ``i`` attends to its own document's last ``window`` keys, the document's first key (the sink) and the
followers ``m + 1`` of the ``topk`` earlier positions ``m`` whose index vectors best match its own
(``score = idx_i . idx_m / sqrt(R)``, a tied low-rank similarity). Selected keys also get ``lam_h * score`` on their
logit, which is how the index learns. The scores are computed densely in query chunks (rematerialised), so this
measures quality, not speed.
"""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.sharding import auto_axes
from jaxtyping import Array, Float, Int
from levanter.grug.attention._inkling_relpos import REL_BIAS_BLOCK, rel_extent_of_band
from levanter.kernels.pallas.autotune_utils import named_sharding_of

_QUERY_CHUNK = 256


def _document_starts(segment_ids: Int[Array, "B S"]) -> jax.Array:
    """Each token's document start position ``[B, S]``."""
    seq = segment_ids.shape[1]
    idx = jnp.arange(seq, dtype=jnp.int32)
    starts = jnp.pad(segment_ids[:, 1:] != segment_ids[:, :-1], ((0, 0), (1, 0)), constant_values=True)
    return jax.lax.cummax(jnp.where(starts, idx, 0), axis=1)


def _sparse_math(q, k, v, rel_bias, segment_ids, index, lam, *, window: int, topk: int):
    batch, seq, heads, head_dim = q.shape
    chunk = min(_QUERY_CHUNK, seq)
    if seq % chunk:
        raise ValueError(f"sequence length {seq} must be a multiple of {chunk}")
    scale = 1.0 / math.sqrt(head_dim)
    starts = _document_starts(segment_ids)
    keys = jnp.arange(seq)
    rel_extent = None if rel_bias is None else rel_extent_of_band(rel_bias)
    index = index.astype(jnp.float32) / math.sqrt(index.shape[-1])

    @jax.checkpoint
    def one_chunk(c0):
        qi = c0 + jnp.arange(chunk)  # [C]
        qc = jax.lax.dynamic_slice_in_dim(q, c0, chunk, axis=1).astype(jnp.float32)
        logits = jnp.einsum("bqhd,bkhd->bhqk", qc * scale, k.astype(jnp.float32))  # [B, H, C, S]
        if rel_bias is not None:
            band = jax.lax.dynamic_slice_in_dim(rel_bias, c0, chunk, axis=2).astype(jnp.float32)
            column = keys[None, :] - (qi[:, None] // REL_BIAS_BLOCK) * REL_BIAS_BLOCK + rel_extent
            in_band = (column >= 0) & (column < band.shape[-1])
            gathered = jnp.take_along_axis(
                band, jnp.broadcast_to(jnp.clip(column, 0, band.shape[-1] - 1), band.shape[:2] + column.shape), axis=-1
            )
            logits = logits + jnp.where(in_band, gathered, 0.0)
        seg_q = jax.lax.dynamic_slice_in_dim(segment_ids, c0, chunk, axis=1)  # [B, C]
        start_q = jax.lax.dynamic_slice_in_dim(starts, c0, chunk, axis=1)
        same = seg_q[:, :, None] == segment_ids[:, None, :]  # [B, C, S]
        dist = qi[None, :, None] - keys[None, None, :]
        allowed = same & (dist >= 0) & ((dist < window) | (keys[None, None, :] == start_q[:, :, None]))
        bias = jnp.zeros(allowed.shape, jnp.float32)
        if topk:
            idx_q = jax.lax.dynamic_slice_in_dim(index, c0, chunk, axis=1)
            score = jnp.einsum("bqr,bkr->bqk", idx_q, index)  # [B, C, S] against candidate m
            # Candidate m's follower m + 1 must be visible (m + 1 <= i) and outside the local window.
            cand = same & (dist >= 1) & (dist - 1 >= window)
            masked = jnp.where(cand, score, -jnp.inf)
            kth = jax.lax.top_k(masked, topk)[0][..., -1:]
            chosen = cand & (masked >= kth)  # kth = -inf (fewer than topk candidates): every candidate
            # Move candidate m's selection and score onto its follower key m + 1.
            chosen = jnp.pad(chosen[..., :-1], ((0, 0), (0, 0), (1, 0)))
            score = jnp.pad(score[..., :-1], ((0, 0), (0, 0), (1, 0)))
            allowed = allowed | chosen
            bias = jnp.where(chosen, score, 0.0)
        logits = logits + bias[:, None] * lam.astype(jnp.float32)[None, :, None, None]
        logits = jnp.where(allowed[:, None], logits, -1e9)
        weights = jax.nn.softmax(logits, axis=-1)
        return jnp.einsum("bhqk,bkhd->bqhd", weights, v.astype(jnp.float32)).astype(v.dtype)

    chunks = jax.lax.map(one_chunk, jnp.arange(0, seq, chunk))  # [n, B, C, H, D]
    return jnp.moveaxis(chunks, 0, 1).reshape(batch, seq, heads, v.shape[-1])


def sparse_index_attention(
    q: Float[Array, "B S H D"],
    k: Float[Array, "B S H D"],
    v: Float[Array, "B S H E"],
    rel_bias: Float[Array, "B H S W"] | None,
    segment_ids: Int[Array, "B S"] | None,
    index: Float[Array, "B S R"],
    lam: Float[Array, " H"],
    *,
    window: int,
    topk: int,
) -> Float[Array, "B S H E"]:
    """Attention over each query's local ``window``, its document start and its ``topk`` index-matched followers."""
    if segment_ids is None:
        segment_ids = jnp.zeros(q.shape[:2], jnp.int32)
    math_fn = lambda *a: _sparse_math(*a, window=window, topk=topk)  # noqa: E731
    out_sharding = named_sharding_of(q)
    if out_sharding is None:
        return math_fn(q, k, v, rel_bias, segment_ids, index, lam)
    # pyrefly: ignore[bad-assignment]  # auto_axes's decorator overload erases the wrapped signature
    wrapped: Callable[..., jax.Array] = auto_axes(math_fn, out_sharding=out_sharding)
    return wrapped(q, k, v, rel_bias, segment_ids, index, lam)
