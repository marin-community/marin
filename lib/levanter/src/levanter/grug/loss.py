# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Blockwise linear softmax cross-entropy for grug.

This is the "large vocab friendly" alternative to materializing full logits
`hidden @ lm_head` with shape (batch, seq, vocab).

Design notes:
  - Works on plain `jax.Array` inputs (grug core doesn't use NamedArray).
  - Computes `logsumexp` over vocab in blocks to reduce peak memory.
  - Computes the correct-class logit via gather+dot (O(N*H)), avoiding a full
    logits materialization.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, get_abstract_mesh, reshard


def linear_softmax_cross_entropy_loss_and_logz(
    hidden: jax.Array,
    lm_head: jax.Array,
    labels: jax.Array,
    *,
    block_size: int | None = None,
    dtype: jnp.dtype = jnp.float32,
    logit_soft_cap: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute per-position cross entropy without materializing full logits.

    Args:
        hidden: Array with shape (..., hidden_dim).
        lm_head: Array with shape (hidden_dim, vocab_size).
        labels: Integer array with shape (...,).
        block_size: Vocab block size for logsumexp.
        dtype: Accumulator dtype for logsumexp.
        logit_soft_cap: Optional tanh soft cap for logits (applied before exp).

    Returns:
        (loss, logz) each with shape labels.shape.
    """
    if block_size is None:
        block_size = lm_head.shape[1]

    hidden_dim = hidden.shape[-1]
    if lm_head.ndim != 2:
        raise ValueError(f"lm_head must be 2D (hidden_dim, vocab), got shape={lm_head.shape}")
    if lm_head.shape[0] != hidden_dim:
        raise ValueError(f"hidden_dim mismatch: hidden={hidden_dim}, lm_head={lm_head.shape[0]}")

    vocab_size = lm_head.shape[1]
    nblocks = (vocab_size + block_size - 1) // block_size
    vocab_padded = nblocks * block_size
    if vocab_padded != vocab_size:
        # We pad the vocab dimension so we can `dynamic_slice_in_dim(..., slice_size=block_size)`
        # for every block. Use `lax.pad` (rather than concatenating a freshly-created zeros array)
        # so sharding stays consistent under explicit meshes.
        lm_head = jax.lax.pad(
            lm_head,
            jnp.array(0, dtype=lm_head.dtype),
            padding_config=((0, 0, 0), (0, vocab_padded - vocab_size, 0)),
        )
    flat_hidden = hidden.reshape((-1, hidden_dim)).astype(dtype)
    flat_labels = labels.reshape((-1,)).astype(jnp.int32)

    # correct logits: dot(hidden, lm_head[:, label])
    # We take rows from lm_head.T (shape [vocab, hidden]) so the gather output is [N, hidden].
    # Under explicit meshes, gather output sharding is otherwise ambiguous, so we supply out_sharding.
    # Some JAX versions require `out_sharding=` for this gather, and (in some versions) the
    # value must be a PartitionSpec rather than a full Sharding object.
    out_sharding = None
    labels_sharding = getattr(flat_labels, "sharding", None)
    if isinstance(labels_sharding, NamedSharding) and isinstance(labels_sharding.spec, P):
        first_dim = labels_sharding.spec[0] if labels_sharding.spec else None
        out_sharding = P(first_dim, None)
    else:
        # In some tracing contexts we don't get a NamedSharding instance back, but the mesh is
        # still explicit. Default to the common "data-parallel batch" mapping used in Levanter.
        mesh = get_abstract_mesh()
        if mesh is not None and not mesh.empty:
            batch_axes = tuple(ax for ax in ("replica_dcn", "replica", "data") if ax in mesh.shape)
            if batch_axes:
                out_sharding = P(batch_axes, None)

    w_y = lm_head.T.at[flat_labels].get(out_sharding=out_sharding).astype(dtype)
    hidden_sharding = getattr(flat_hidden, "sharding", None)
    if isinstance(hidden_sharding, NamedSharding) and isinstance(hidden_sharding.spec, P):
        w_y = reshard(w_y, hidden_sharding.spec)

    logit_y = jnp.sum(flat_hidden * w_y, axis=-1)

    neg_inf = jnp.array(-jnp.inf, dtype=dtype)
    # Match the sharding of the computed per-example logits so the loop carry types are stable.
    m0 = jnp.full_like(logit_y, neg_inf)
    s0 = jnp.zeros_like(logit_y, dtype=dtype)

    neg_inf_logits = jnp.array(-jnp.inf, dtype=dtype)

    def _body(i: int, state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        m, s = state
        start = i * block_size
        w_block = jax.lax.dynamic_slice_in_dim(lm_head, start_index=start, slice_size=block_size, axis=1).astype(dtype)
        logits = flat_hidden @ w_block
        valid = jnp.arange(block_size) < (vocab_size - start)
        logits = jnp.where(valid[None, :], logits, neg_inf_logits)
        if logit_soft_cap is not None:
            logits = jnp.tanh(logits / logit_soft_cap) * logit_soft_cap
        block_max = jnp.max(logits, axis=-1)
        new_m = jnp.maximum(m, block_max)
        s = s * jnp.exp(m - new_m) + jnp.sum(jnp.exp(logits - new_m[:, None]), axis=-1)
        return new_m, s

    m, s = jax.lax.fori_loop(0, nblocks, _body, (m0, s0))
    logz = m + jnp.log(s)
    loss = logz - logit_y

    return loss.reshape(labels.shape), logz.reshape(labels.shape)


def next_token_linear_softmax_cross_entropy(
    token_ids: jax.Array,
    hidden: jax.Array,
    lm_head: jax.Array,
    *,
    block_size: int,
    reduction: Literal["mean", "sum", "none"] = "mean",
    dtype: jnp.dtype = jnp.float32,
    logsumexp_weight: float | None = None,
    logit_soft_cap: float | None = None,
) -> jax.Array:
    """Next-token loss using blockwise logits.

    This matches the common "predict token t+1 from hidden at t" loss.
    The last position is ignored.
    """
    if token_ids.ndim < 1:
        raise ValueError("token_ids must have at least 1 dimension")

    # Shift tokens left: label[t] = token_ids[t+1]
    labels = jnp.concatenate([token_ids[..., 1:], token_ids[..., :1] * 0], axis=-1)
    weight = jnp.concatenate(
        [
            jnp.ones(token_ids.shape[:-1] + (token_ids.shape[-1] - 1,), dtype=dtype),
            jnp.zeros(token_ids.shape[:-1] + (1,), dtype=dtype),
        ],
        axis=-1,
    )

    loss, logz = linear_softmax_cross_entropy_loss_and_logz(
        hidden,
        lm_head,
        labels,
        block_size=block_size,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
    )
    loss = loss * weight

    if logsumexp_weight is not None and logsumexp_weight != 0.0:
        loss = loss + logsumexp_weight * (logz**2) * weight

    if reduction == "none":
        return loss
    if reduction == "sum":
        return jnp.sum(loss)
    if reduction == "mean":
        denom = jnp.sum(weight)
        return jnp.sum(loss) / jnp.maximum(denom, jnp.array(1.0, dtype=dtype))

    raise ValueError(f"Unknown reduction: {reduction}")


__all__ = [
    "linear_softmax_cross_entropy_loss_and_logz",
    "next_token_linear_softmax_cross_entropy",
]
