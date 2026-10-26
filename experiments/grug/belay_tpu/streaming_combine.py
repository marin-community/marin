# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Accumulate expert routes directly into token outputs without a full restore."""

from functools import partial

import jax
import jax.numpy as jnp


def sum_routes(rows, positions, weights=None, *, implementation="unrolled"):
    """Sum [tokens, topk] expert-row indices with FP32 accumulation.

    The scan variant keeps one token-sized accumulator. Unrolled variants expose
    each route to XLA fusion; the flat variant also changes the gather indexing.
    Inputs are fully device-local, as in the enclosing local expert shard_map.
    """
    tokens, topk = positions.shape
    width = rows.shape[1]
    if implementation not in ("unrolled", "scan", "flat"):
        raise ValueError("Unknown route accumulation implementation")

    def accumulate(slot, total):
        indices = positions[:, slot]
        if implementation == "flat":
            selected = rows.reshape(-1)[indices[:, None] * width + jnp.arange(width)[None, :]]
        else:
            selected = rows[indices]
        values = selected.astype(jnp.float32)
        if weights is not None:
            values = values * weights[:, slot, None]
        return total + values

    result = jnp.zeros((tokens, width), jnp.float32)
    if implementation == "scan":
        result = jax.lax.fori_loop(0, topk, accumulate, result)
    else:
        for slot in range(topk):
            result = accumulate(slot, result)
    return result.astype(rows.dtype)


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def combine_streaming(rows, order, inverse, weights, implementation="unrolled"):
    """Combine routed expert outputs and form pullbacks in expert row order."""
    del order
    return sum_routes(rows, inverse.reshape(weights.shape), weights, implementation=implementation)


def _forward(rows, order, inverse, weights, implementation):
    return combine_streaming(rows, order, inverse, weights, implementation), (rows, order, inverse, weights)


def _backward(implementation, residuals, gradient):
    del implementation
    rows, order, inverse, weights = residuals
    token_gradient = gradient[order // weights.shape[1]].astype(jnp.float32)
    row_weights = weights.reshape(-1)[order]
    d_rows = (token_gradient * row_weights[:, None]).astype(rows.dtype)
    d_weights = (token_gradient * rows.astype(jnp.float32)).sum(axis=-1)
    return d_rows, None, None, d_weights[inverse].reshape(weights.shape).astype(weights.dtype)


combine_streaming.defvjp(_forward, _backward)


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def dispatch_streaming(x, order, inverse, topk):
    """Dispatch token rows and sum their input gradients without a full restore."""
    del inverse
    return x[order // topk]


def _dispatch_forward(x, order, inverse, topk):
    return dispatch_streaming(x, order, inverse, topk), inverse


def _dispatch_backward(topk, inverse, gradient):
    result = sum_routes(gradient, inverse.reshape(-1, topk), implementation="scan")
    return result, None, None


dispatch_streaming.defvjp(_dispatch_forward, _dispatch_backward)
