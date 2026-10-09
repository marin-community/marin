# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-query reads of a KDA layer's delta-rule state, fed to a small MLP that writes to the residual stream.

Each of the layer's ``H`` state heads is read with ``m`` learned, token-independent queries ``a_{h,i}``:
``r_t = [a_{h,i}^T S_{h,t}]`` (``H * m * Dv`` features, the same questions asked of the memory at every position).
The reads reuse the KDA kernel: query head ``h * m + i`` gets head ``h``'s decay, write strength and erase gate,
so it sees exactly head ``h``'s state (the kernel recomputes that state per query head, so ``m`` reads cost about
``m`` extra passes of the layer). ``out = relu(rms(r) W_in)^2 W_out`` with ``W_out`` zero-init: silent at init."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random


class StateRead(eqx.Module):
    """``queries`` ``[H m, Dk]`` (2-D so a stacked leaf stays 3-D); ``state_read_in`` ``[H m Dv, hidden]``;
    ``state_read_out`` ``[hidden, D]``. Trained in the ``adam`` group (``.state_read.``)."""

    queries: jax.Array
    state_read_in: jax.Array
    state_read_out: jax.Array


def state_read_init(key: jax.Array, heads: int, reads: int, dk: int, dv: int, hidden: int, d: int) -> StateRead:
    kq, kin = random.split(key)
    width = heads * reads * dv
    return StateRead(
        queries=random.normal(kq, (heads * reads, dk)) / jnp.sqrt(dk),
        state_read_in=random.normal(kin, (width, hidden)) / jnp.sqrt(width),
        state_read_out=jnp.zeros((hidden, d), jnp.float32),
    )


def repeat_heads(x: jax.Array, reads: int, axis: int) -> jax.Array:
    """Each head's slice repeated ``reads`` times in place (query head ``h * m + i`` -> state head ``h``)."""
    return jnp.repeat(x, reads, axis=axis, out_sharding=jax.typeof(x).sharding)


def state_read_features(p: StateRead, reads_out: jax.Array) -> jax.Array:
    """``[B, S, H m, Dv]`` kernel outputs -> the MLP's ``[B, S, D]`` contribution."""
    b, s = reads_out.shape[:2]
    flat = reads_out.reshape(b, s, -1).astype(jnp.float32)
    flat = flat * jax.lax.rsqrt(jnp.mean(jnp.square(flat), axis=-1, keepdims=True) + 1e-6)
    hidden = jnp.square(jax.nn.relu(flat @ p.state_read_in.astype(jnp.float32)))
    return hidden @ p.state_read_out.astype(jnp.float32)
