# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local-window + learned sparse-index attention for the MLA layers, as a dense masked emulation.

Each query ``i`` attends to its own document's last ``window`` keys, the document's first key (the sink) and the
followers ``m + 1`` of the ``topk`` earlier positions ``m`` whose index vectors best match its own
(``score = idx_i . idx_m / sqrt(R)``, a tied low-rank similarity). Selected keys also get ``lam_h * score`` on their
logit, which is how the index learns. With a ``Lightning`` indexer (DeepSeek-V3.2's DSA) the selection is instead
the ``topk`` keys with the best ``I(i, s) = sum_h w_h(x_i) relu(q_h(x_i) . k(x_s))``, trained only by an injected
``KL(main attention || softmax(I))`` gradient. The scores are computed densely in query chunks (rematerialised), so
this measures quality, not speed.
"""

import math
from collections.abc import Callable
from enum import StrEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import auto_axes
from jaxtyping import Array, Float, Int
from levanter.grug.attention._inkling_relpos import REL_BIAS_BLOCK, rel_extent_of_band
from levanter.kernels.pallas.autotune_utils import named_sharding_of

_QUERY_CHUNK = 256


class SparseIndex(StrEnum):
    """How ``mla_sparse_topk`` picks the far keys."""

    FOLLOW = "follow"
    """The followers of the best tied low-rank matches, scored onto their logits."""
    LIGHTNING = "lightning"
    """DSA's lightning indexer over the keys themselves, trained by KL to the dense attention."""
    BLOCK = "block"
    """NSA-style: the ``topk`` key blocks whose compressed (block-mean) keys the heads attend to most."""


class Lightning(NamedTuple):
    """DSA lightning-indexer inputs (computed from a stop-gradient attention input)."""

    q: Float[Array, "B S G R"]
    k: Float[Array, "B S R"]
    w: Float[Array, "B S G"]
    kl_weight: float


@jax.custom_vjp
def _inject_kl(out: jax.Array, index_logits: jax.Array, target: jax.Array, scale: jax.Array) -> jax.Array:
    """Identity on ``out``; backward gives ``index_logits`` the gradient of ``scale * KL(target || softmax)``."""
    del index_logits, target, scale
    return out


def _inject_kl_fwd(out, index_logits, target, scale):
    return out, (index_logits, target, scale)


def _inject_kl_bwd(res, g):
    index_logits, target, scale = res
    grad = scale * (jax.nn.softmax(index_logits, axis=-1) - target)
    return g, grad.astype(index_logits.dtype), jnp.zeros_like(target), jnp.zeros_like(scale)


_inject_kl.defvjp(_inject_kl_fwd, _inject_kl_bwd)


def _document_starts(segment_ids: Int[Array, "B S"]) -> jax.Array:
    """Each token's document start position ``[B, S]``."""
    seq = segment_ids.shape[1]
    idx = jnp.arange(seq, dtype=jnp.int32)
    starts = jnp.pad(segment_ids[:, 1:] != segment_ids[:, :-1], ((0, 0), (1, 0)), constant_values=True)
    return jax.lax.cummax(jnp.where(starts, idx, 0), axis=1)


def _sparse_math(
    q,
    k,
    v,
    rel_bias,
    segment_ids,
    index,
    lam,
    lightning,
    *,
    window: int,
    topk: int,
    select_match: bool,
    block: int,
    compress: bool,
    block_select: bool,
):
    batch, seq, heads, head_dim = q.shape
    chunk = min(_QUERY_CHUNK, seq)
    if seq % chunk:
        raise ValueError(f"sequence length {seq} must be a multiple of {chunk}")
    scale = 1.0 / math.sqrt(head_dim)
    starts = _document_starts(segment_ids)
    keys = jnp.arange(seq)
    rel_extent = None if rel_bias is None else rel_extent_of_band(rel_bias)
    index = index.astype(jnp.float32) / math.sqrt(index.shape[-1])
    if compress or block_select:
        if seq % block:
            raise ValueError(f"sequence length {seq} must be a multiple of the block {block}")
        blocks = seq // block
        kc = k.astype(jnp.float32).reshape(batch, blocks, block, heads, -1).mean(axis=2)  # [B, nb, H, D]
        vc = v.astype(jnp.float32).reshape(batch, blocks, block, heads, -1).mean(axis=2)
        seg_blocks = segment_ids.reshape(batch, blocks, block)
        # A block is usable only when it lies inside one document; its last token sets when it is visible.
        whole = seg_blocks[..., 0] == seg_blocks[..., -1]
        block_end = jnp.arange(blocks) * block + block - 1

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
        local = same & (dist >= 0) & ((dist < window) | (keys[None, None, :] == start_q[:, :, None]))
        allowed = local[:, None]  # [B, G, C, S], G = 1 or one per head
        bias = jnp.zeros(allowed.shape, jnp.float32)
        index_logits = None
        if lightning is not None:
            lq = jax.lax.dynamic_slice_in_dim(lightning.q, c0, chunk, axis=1).astype(jnp.float32)
            lw = jax.lax.dynamic_slice_in_dim(lightning.w, c0, chunk, axis=1).astype(jnp.float32)
            per_head = jax.nn.relu(jnp.einsum("bqgr,bkr->bqgk", lq, lightning.k.astype(jnp.float32)))
            visible = same & (dist >= 0)
            index_logits = jnp.where(visible, jnp.einsum("bqgk,bqg->bqk", per_head, lw), -1e9)
            kth = jax.lax.top_k(index_logits, topk)[0][..., -1:]
            allowed = allowed | (visible & (index_logits >= kth))[:, None]
        elif topk and not block_select:
            idx_q = jax.lax.dynamic_slice_in_dim(index, c0, chunk, axis=1)
            score = jnp.einsum("bqgr,bkgr->bgqk", idx_q, index)  # [B, G, C, S] against candidate m
            # Candidate m's follower m + 1 must be visible (m + 1 <= i) and outside the local window.
            cand = (same & (dist >= 1) & (dist - 1 >= window))[:, None]
            masked = jnp.where(cand, score, -jnp.inf)
            kth = jax.lax.top_k(masked, topk)[0][..., -1:]
            matched = cand & (masked >= kth)  # kth = -inf (fewer than topk candidates): every candidate
            # Move candidate m's selection and score onto its follower key m + 1.
            follow = jnp.pad(matched[..., :-1], ((0, 0), (0, 0), (0, 0), (1, 0)))
            bias = jnp.where(follow, jnp.pad(score[..., :-1], ((0, 0), (0, 0), (0, 0), (1, 0))), 0.0)
            allowed = allowed | follow
            if select_match:  # also the matched token m itself
                bias = bias + jnp.where(matched & ~follow, score, 0.0)
                allowed = allowed | matched
        if compress or block_select:
            c_logits = jnp.einsum("bqhd,bnhd->bhqn", qc * scale, kc)  # [B, H, C, nb]
            # Fully past (so never the query's own, partly future block) and in the query's document.
            c_ok = whole[:, None, :] & (seg_blocks[:, None, :, 0] == seg_q[:, :, None])
            c_ok = c_ok & (block_end[None, None, :] < qi[None, :, None])  # [B, C, nb]
            c_logits = jnp.where(c_ok[:, None], c_logits, -1e9)
            if block_select:
                importance = jnp.mean(jax.nn.softmax(c_logits, axis=-1), axis=1)  # [B, C, nb]
                importance = jnp.where(c_ok, importance, -1.0)
                kth = jax.lax.top_k(importance, topk)[0][..., -1:]
                picked = c_ok & (importance >= kth)
                tokens = jnp.repeat(picked, block, axis=-1)  # [B, C, S]
                allowed = allowed | (tokens & same & (dist >= 0))[:, None]
        logits = logits + bias * lam.astype(jnp.float32)[None, :, None, None]
        dense = logits
        logits = jnp.where(allowed, logits, -1e9)
        if compress:
            joint = jax.nn.softmax(jnp.concatenate([logits, c_logits], axis=-1), axis=-1)
            weights, c_weights = joint[..., :seq], joint[..., seq:]
            out = jnp.einsum("bhqk,bkhd->bqhd", weights, v.astype(jnp.float32))
            out = (out + jnp.einsum("bhqn,bnhd->bqhd", c_weights, vc)).astype(v.dtype)
        else:
            weights = jax.nn.softmax(logits, axis=-1)
            out = jnp.einsum("bhqk,bkhd->bqhd", weights, v.astype(jnp.float32)).astype(v.dtype)
        if index_logits is None:
            return out
        # DSA target: the dense causal attention, summed over heads and renormalised.
        visible = same & (dist >= 0)
        target = jnp.mean(jax.nn.softmax(jnp.where(visible[:, None], dense, -1e9), axis=-1), axis=1)
        kl_scale = jnp.asarray(lightning.kl_weight / (batch * seq), jnp.float32)
        return _inject_kl(out, index_logits, jax.lax.stop_gradient(target), kl_scale)

    chunks = jax.lax.map(one_chunk, jnp.arange(0, seq, chunk))  # [n, B, C, H, D]
    return jnp.moveaxis(chunks, 0, 1).reshape(batch, seq, heads, v.shape[-1])


def sparse_index_attention(
    q: Float[Array, "B S H D"],
    k: Float[Array, "B S H D"],
    v: Float[Array, "B S H E"],
    rel_bias: Float[Array, "B H S W"] | None,
    segment_ids: Int[Array, "B S"] | None,
    index: Float[Array, "B S G R"],
    lam: Float[Array, " H"],
    lightning: Lightning | None = None,
    *,
    window: int,
    topk: int,
    select_match: bool = False,
    block: int = 32,
    compress: bool = False,
    block_select: bool = False,
) -> Float[Array, "B S H E"]:
    """Attention over each query's local ``window``, its document start and its ``topk`` index-matched followers."""
    if segment_ids is None:
        segment_ids = jnp.zeros(q.shape[:2], jnp.int32)
    math_fn = lambda *a: _sparse_math(  # noqa: E731
        *a,
        window=window,
        topk=topk,
        select_match=select_match,
        block=block,
        compress=compress,
        block_select=block_select,
    )
    out_sharding = named_sharding_of(q)
    if out_sharding is None:
        return math_fn(q, k, v, rel_bias, segment_ids, index, lam, lightning)
    # pyrefly: ignore[bad-assignment]  # auto_axes's decorator overload erases the wrapped signature
    wrapped: Callable[..., jax.Array] = auto_axes(math_fn, out_sharding=out_sharding)
    return wrapped(q, k, v, rel_bias, segment_ids, index, lam, lightning)
