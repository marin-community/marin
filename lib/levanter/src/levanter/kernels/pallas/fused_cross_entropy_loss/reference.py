# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def _apply_logit_soft_cap(logits: Float[Array, "B V"], logit_soft_cap: Optional[float]) -> Float[Array, "B V"]:
    if logit_soft_cap is None:
        return logits
    return jnp.tanh(logits / logit_soft_cap) * logit_soft_cap


def linear_softmax_cross_entropy_loss_reference(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    *,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
) -> tuple[Float[Array, "B"], Float[Array, "B"]]:
    """Reference loss + logsumexp for linear softmax cross-entropy.

    Args:
        x: [B, H] input activations.
        labels: [B] integer labels.
        w: [H, V] projection weights.
        dtype: Optional dtype for logits/softmax.
        logit_soft_cap: Optional tanh soft cap for logits.

    Returns:
        loss: [B] per-example cross-entropy loss.
        lse: [B] logsumexp of logits.
    """

    logits = x @ w
    if dtype is not None:
        logits = logits.astype(dtype)

    logits = _apply_logit_soft_cap(logits, logit_soft_cap)
    lse = jax.nn.logsumexp(logits, axis=-1)
    label_logits = logits[jnp.arange(logits.shape[0]), labels]
    loss = lse - label_logits
    return loss, lse


def linear_softmax_cross_entropy_loss_streaming(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    *,
    block_size: int,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
) -> tuple[Float[Array, "B"], Float[Array, "B"]]:
    """Streaming reference loss + logsumexp without materializing logits."""

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    b_dim = x.shape[0]
    v_dim = w.shape[1]
    out_dtype = jnp.dtype(dtype) if dtype is not None else x.dtype

    pad = (-v_dim) % block_size
    if pad:
        w = jnp.pad(w, ((0, 0), (0, pad)), mode="constant", constant_values=0)
    v_padded = v_dim + pad
    num_blocks = v_padded // block_size

    logsumexp_init = jnp.full((b_dim,), -jnp.inf, dtype=out_dtype)
    label_logit_init = jnp.full((b_dim,), -jnp.inf, dtype=out_dtype)

    def body(block_idx, state):
        logsumexp, label_logit = state
        start = block_idx * block_size

        w_block = jax.lax.dynamic_slice(w, (0, start), (w.shape[0], block_size))
        logits = jax.lax.dot_general(
            x,
            w_block,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        if dtype is not None:
            logits = logits.astype(dtype)
        logits = _apply_logit_soft_cap(logits, logit_soft_cap)

        valid = (start + jnp.arange(block_size)) < v_dim
        logits = jnp.where(valid, logits, -jnp.inf)

        block_lse = jax.nn.logsumexp(logits, axis=-1)
        logsumexp = jnp.logaddexp(logsumexp, block_lse)

        in_block = (labels >= start) & (labels < start + block_size)
        label_idx = labels - start
        safe_idx = jnp.where(in_block, label_idx, 0)
        block_label_logit = logits[jnp.arange(b_dim), safe_idx]
        label_logit = jnp.where(in_block, block_label_logit, label_logit)
        return logsumexp, label_logit

    logsumexp, label_logit = jax.lax.fori_loop(0, num_blocks, body, (logsumexp_init, label_logit_init))
    loss = logsumexp - label_logit
    return loss, logsumexp
