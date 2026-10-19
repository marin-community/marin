# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Weighted-sum linear cross-entropy that computes its gradients in the forward pass.

The streaming XLA path runs one logits GEMM in the forward and recomputes the logits in the
backward before its two gradient GEMMs. When the loss is a weighted sum, every row's cotangent is
the scalar output cotangent times that row's weight, so the forward can finish ``d/dx`` and ``d/dw``
while each block of logits is live, as Liger-Kernel's fused linear cross-entropy does. Value and
gradient then take 3 GEMMs, and the backward only scales the saved gradients.

The saved gradients stay live from the forward to the backward: ``[B, H]`` in ``x``'s dtype and
``[H, V]`` in ``w``'s dtype. Each scan step holds a ``[token_block_size, V]`` float32 logits block.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from .reference import _cross_entropy_exp, _cross_entropy_logsumexp


def _pad_rows(a: jax.Array, padding: int) -> jax.Array:
    return jnp.pad(a, ((0, padding),) + ((0, 0),) * (a.ndim - 1)) if padding else a


def _row_blocks(a: jax.Array, token_block_size: int) -> jax.Array:
    return a.reshape((a.shape[0] // token_block_size, token_block_size) + a.shape[1:])


def _block_logits(x_block, w, dtype, precision) -> jax.Array:
    logits = jax.lax.dot_general(
        x_block, w, (((1,), (0,)), ((), ())), precision=precision, preferred_element_type=jnp.float32
    )
    if dtype is not None:
        logits = logits.astype(dtype)
    return logits.astype(jnp.float32)


def _block_loss(logits, labels_block, logsumexp_weight):
    lse = _cross_entropy_logsumexp(logits)
    label_logits = jnp.take_along_axis(logits, labels_block[:, None], axis=1)[:, 0]
    return lse, lse - label_logits + logsumexp_weight * lse**2


def _blocked(x, labels, weight, token_block_size):
    """Pad rows to a multiple of the block with zero weight, which drops them from the sum and grads."""
    padding = (-x.shape[0]) % token_block_size
    x, labels, weight = (_pad_rows(a, padding) for a in (x, labels, weight))
    return tuple(_row_blocks(a, token_block_size) for a in (x, labels, weight))


def _weighted_sum_primal(x, labels, w, weight, logsumexp_weight, token_block_size, dtype, precision):
    def body(total, block):
        x_block, labels_block, weight_block = block
        _, loss = _block_loss(_block_logits(x_block, w, dtype, precision), labels_block, logsumexp_weight)
        return total + jnp.sum(loss * weight_block.astype(jnp.float32)), None

    total, _ = jax.lax.scan(body, jnp.zeros((), jnp.float32), _blocked(x, labels, weight, token_block_size))
    return total


def _weighted_sum_fwd(x, labels, w, weight, logsumexp_weight, token_block_size, dtype, precision):
    rows, hidden_dim = x.shape
    gemm_dtype = jnp.result_type(x.dtype, w.dtype)

    def body(grad_w, block):
        x_block, labels_block, weight_block = block
        logits = _block_logits(x_block, w, dtype, precision)
        lse, loss = _block_loss(logits, labels_block, logsumexp_weight)
        weight_f32 = weight_block.astype(jnp.float32)
        # d(weight * loss) / d logits = weight * (softmax * (1 + 2 z lse) - onehot(label)).
        probs = _cross_entropy_exp(logits - lse[:, None])
        label_hit = jnp.arange(logits.shape[1], dtype=jnp.int32)[None, :] == labels_block[:, None]
        dlogits = probs * (weight_f32 * (1.0 + 2.0 * logsumexp_weight * lse))[:, None]
        dlogits = (dlogits - jnp.where(label_hit, weight_f32[:, None], 0.0)).astype(gemm_dtype)
        grad_x = jax.lax.dot_general(
            dlogits, w, (((1,), (1,)), ((), ())), precision=precision, preferred_element_type=jnp.float32
        )
        grad_w = grad_w + jax.lax.dot_general(
            x_block, dlogits, (((0,), (0,)), ((), ())), precision=precision, preferred_element_type=jnp.float32
        )
        return grad_w, (loss, grad_x.astype(x.dtype))

    grad_w, (loss, grad_x) = jax.lax.scan(
        body, jnp.zeros(w.shape, jnp.float32), _blocked(x, labels, weight, token_block_size)
    )
    loss = loss.reshape(-1)[:rows]
    grad_x = grad_x.reshape(-1, hidden_dim)[:rows]
    total = jnp.sum(loss * weight.astype(jnp.float32))
    return total, (grad_x, grad_w.astype(w.dtype), loss.astype(weight.dtype))


def _weighted_sum_bwd(logsumexp_weight, token_block_size, dtype, precision, residuals, g):
    del logsumexp_weight, token_block_size, dtype, precision
    grad_x, grad_w, loss = residuals
    g = g.astype(jnp.float32)
    labels_cotangent = np.zeros(loss.shape, dtype=jax.dtypes.float0)
    return (
        (g * grad_x).astype(grad_x.dtype),
        labels_cotangent,
        (g * grad_w).astype(grad_w.dtype),
        (g * loss).astype(loss.dtype),
    )


_weighted_sum = jax.custom_vjp(_weighted_sum_primal, nondiff_argnums=(4, 5, 6, 7))
_weighted_sum.defvjp(_weighted_sum_fwd, _weighted_sum_bwd)


def linear_softmax_cross_entropy_weighted_sum(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    weight: Float[Array, "B"],
    *,
    logsumexp_weight: Optional[float],
    token_block_size: int,
    dtype: Optional[jnp.dtype] = jnp.float32,
    precision: jax.lax.PrecisionLike = None,
) -> Float[Array, ""]:
    """``sum_i weight_i * (lse_i - logits_i[labels_i] + logsumexp_weight * lse_i**2)``.

    Differentiating it runs the forward with gradients: 3 GEMMs per row block, against 4 for the
    streaming ``xla_fast_bwd`` path. Undifferentiated calls compute the loss alone.

    Args:
        x: [B, H] input activations.
        labels: [B] integer labels.
        w: [H, V] projection weights.
        weight: [B] per-row weights; zero drops a row.
        logsumexp_weight: Weight of the z-loss term, or ``None`` for none.
        token_block_size: Rows per scan step. Rows are padded with zero weight to a multiple of it.
        dtype: Dtype the logits are rounded to before the float32 softmax math.
        precision: Matmul precision for the three GEMMs.
    """
    if token_block_size <= 0:
        raise ValueError(f"token_block_size must be positive, got {token_block_size}.")
    return _weighted_sum(
        x,
        labels.astype(jnp.int32),
        w,
        weight,
        0.0 if logsumexp_weight is None else float(logsumexp_weight),
        token_block_size,
        dtype,
        precision,
    )
