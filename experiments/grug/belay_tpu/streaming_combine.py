# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Accumulate expert routes directly into token outputs without a full restore."""

from functools import partial

import jax
import jax.numpy as jnp

from experiments.grug.belay_tpu.routing_tpu import restore_assignment_scalars_sort


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


def sum_routes_blocked(rows, positions, weights=None, *, tile_tokens=1024):
    """Reduce all routes within bounded token tiles without a global accumulator."""
    tokens, topk = positions.shape
    tile = min(tokens, tile_tokens)
    padding = (-tokens) % tile
    indices = jnp.pad(positions, ((0, padding), (0, 0))).reshape(-1, tile, topk)
    coefficients = None if weights is None else jnp.pad(weights, ((0, padding), (0, 0))).reshape(-1, tile, topk)

    def reduce_tile(inputs):
        index, coefficient = inputs
        values = rows[index].astype(jnp.float32)
        if coefficient is not None:
            values = values * coefficient[..., None].astype(jnp.float32)
        return values.sum(axis=1).astype(rows.dtype)

    outputs = jax.lax.map(reduce_tile, (indices, coefficients))
    return outputs.reshape(-1, rows.shape[1])[:tokens]


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def combine_streaming(
    rows, order, inverse, weights, implementation="unrolled", scalar_permutation="gather", token_tile=0
):
    """Combine routed expert outputs and form pullbacks in expert row order."""
    if scalar_permutation not in ("gather", "sort"):
        raise ValueError("Unknown scalar permutation implementation")
    del order
    if token_tile:
        return sum_routes_blocked(rows, inverse.reshape(weights.shape), weights, tile_tokens=token_tile)
    return sum_routes(rows, inverse.reshape(weights.shape), weights, implementation=implementation)


def _forward(rows, order, inverse, weights, implementation, scalar_permutation, token_tile):
    output = combine_streaming(rows, order, inverse, weights, implementation, scalar_permutation, token_tile)
    return output, (rows, order, inverse, weights)


def _backward(implementation, scalar_permutation, token_tile, residuals, gradient):
    del implementation, token_tile
    rows, order, inverse, weights = residuals
    token_gradient = gradient[order // weights.shape[1]].astype(jnp.float32)
    row_weights = (
        restore_assignment_scalars_sort(weights.reshape(-1), inverse)
        if scalar_permutation == "sort"
        else weights.reshape(-1)[order]
    )
    d_rows = (token_gradient * row_weights[:, None]).astype(rows.dtype)
    d_weights = (token_gradient * rows.astype(jnp.float32)).sum(axis=-1)
    d_weights = restore_assignment_scalars_sort(d_weights, order) if scalar_permutation == "sort" else d_weights[inverse]
    return d_rows, None, None, d_weights.reshape(weights.shape).astype(weights.dtype)


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
