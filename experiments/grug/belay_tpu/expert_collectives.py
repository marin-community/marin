# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expert-gradient reductions that return only the owning device's experts."""

import jax
import jax.numpy as jnp


def butterfly_reduce_scatter(gradient):
    """Sum FP32 gradients by recursive halving on a power-of-two expert mesh."""
    peers = jax.lax.axis_size("expert")
    if peers & (peers - 1) or gradient.shape[0] % peers:
        raise ValueError("Mesh must be a power of two and divide the expert count")
    rank = jax.lax.axis_index("expert")
    reduced = gradient
    stride = peers // 2
    while stride:
        half = reduced.shape[0] // 2
        side = (rank // stride) % 2
        keep = jax.lax.dynamic_slice_in_dim(reduced, side * half, half, axis=0)
        send = jax.lax.dynamic_slice_in_dim(reduced, (1 - side) * half, half, axis=0)
        received = jax.lax.ppermute(send, "expert", [(i, i ^ stride) for i in range(peers)])
        reduced = keep + received
        stride //= 2
    return reduced


def all_to_all_reduce_scatter(gradient):
    """Exchange expert shards, then sum FP32 contributions at their owner."""
    peers = jax.lax.axis_size("expert")
    received = jax.lax.all_to_all(gradient, "expert", split_axis=0, concat_axis=0, tiled=True)
    contributions = received.reshape(peers, gradient.shape[0] // peers, *gradient.shape[1:])
    return jnp.sum(contributions, axis=0)
