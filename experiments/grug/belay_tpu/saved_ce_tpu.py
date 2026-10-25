# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Save Ckpt 1's FP32 logit tiles while retaining the source fast-backward math."""

import functools

import jax
import jax.numpy as jnp
from levanter.kernels.pallas.fused_cross_entropy_loss.reference import (
    _cross_entropy_exp,
    _cross_entropy_logaddexp,
    _cross_entropy_logsumexp,
    logit_soft_cap_and_deriv,
)


def _forward(x, labels, w, block_size, precision, logit_soft_cap):
    rows, hidden = x.shape
    vocab = w.shape[1]
    if vocab % block_size:
        raise ValueError("Saved Ckpt 1 cross-entropy requires complete vocabulary tiles")

    def body(state, index):
        lse, label_logit = state
        start = index * block_size
        weights = jax.lax.dynamic_slice(w, (0, start), (hidden, block_size))
        logits = jax.lax.dot_general(
            x, weights, (((1,), (0,)), ((), ())), precision=precision, preferred_element_type=jnp.float32
        )
        raw_logits = logits
        if logit_soft_cap is not None:
            logits, _ = logit_soft_cap_and_deriv(logits, logit_soft_cap)
        lse = _cross_entropy_logaddexp(lse, _cross_entropy_logsumexp(logits))
        in_block = (labels >= start) & (labels < start + block_size)
        safe_index = jnp.where(in_block, labels - start, 0)
        block_label = logits[jnp.arange(rows), safe_index]
        label_logit = jnp.where(in_block, block_label, label_logit)
        return (lse, label_logit), raw_logits

    initial = jnp.full((rows,), -jnp.inf, jnp.float32)
    (lse, label_logit), logits = jax.lax.scan(body, (initial, initial), jnp.arange(vocab // block_size))
    return (lse - label_logit, lse), (x, labels, w, lse, logits)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def saved_linear_cross_entropy(x, labels, w, block_size, precision=None, logit_soft_cap=None):
    """Compute FP32 loss/LSE with saved logits and source BF16 gradient rounding.

    This candidate supports the Ckpt 1 loss-head contract: complete vocabulary
    tiles, optional scalar soft cap, and one batch tile. It does not select a
    model backend. Raw logits are retained; backward repeats the elementwise
    soft cap in the source expression to preserve its floating-point rounding.
    """
    return _forward(x, labels, w, block_size, precision, logit_soft_cap)[0]


def _backward(block_size, precision, logit_soft_cap, residuals, cotangents):
    x, labels, w, lse, logits = residuals
    g_loss, g_lse = cotangents
    g_loss = g_loss.astype(jnp.float32)
    g_softmax = g_loss + g_lse.astype(jnp.float32)
    gemm_dtype = jnp.result_type(x.dtype, w.dtype)
    offsets = jnp.arange(block_size, dtype=jnp.int32)

    def body(grad_x, index):
        start = index * block_size
        weights = jax.lax.dynamic_slice(w, (0, start), (w.shape[0], block_size))
        tile = logits[index]
        cap_derivative = None
        if logit_soft_cap is not None:
            tile, cap_derivative = logit_soft_cap_and_deriv(tile, logit_soft_cap)
        probs = _cross_entropy_exp(tile - lse[:, None])
        label_hit = offsets[None, :] == (labels - start)[:, None]
        delta = probs * g_softmax[:, None] - jnp.where(label_hit, g_loss[:, None], 0)
        if cap_derivative is not None:
            delta = delta * cap_derivative
        delta = delta.astype(gemm_dtype)
        grad_x = grad_x + jax.lax.dot_general(
            delta, weights, (((1,), (1,)), ((), ())), precision=precision, preferred_element_type=jnp.float32
        )
        grad_w = jax.lax.dot_general(
            x, delta, (((0,), (0,)), ((), ())), precision=precision, preferred_element_type=jnp.float32
        )
        return grad_x, grad_w.astype(w.dtype)

    grad_x, grad_w = jax.lax.scan(body, jnp.zeros(x.shape, jnp.float32), jnp.arange(logits.shape[0]))
    grad_w = grad_w.transpose(1, 0, 2).reshape(w.shape)
    return grad_x.astype(x.dtype), None, grad_w


saved_linear_cross_entropy.defvjp(_forward, _backward)
