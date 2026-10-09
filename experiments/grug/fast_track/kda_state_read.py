# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-query reads of a KDA layer's delta-rule state, fed to a small MLP that writes to the residual stream.

Each of the layer's ``H`` state heads is read with ``m`` learned, token-independent queries ``a_{h,i}``:
``r_t = [a_{h,i}^T S_{h,t}]`` (``H * m * Dv`` features, the same questions asked of the memory at every position).
The reads reuse the KDA kernel: query head ``h * m + i`` gets head ``h``'s decay, write strength and erase gate,
so it sees exactly head ``h``'s state (the kernel recomputes that state per query head, so ``m`` reads cost about
``m`` extra passes of the layer).

``out = gate * relu(rms(r) W_in)^2 W_out``: ``W_in`` / ``W_out`` are ordinary MuonH matrices and the scalar
``gate`` (Adam) starts at 0, so the readout is silent at init (a zero-init ``W_out`` under Adam injects a large
write on the first steps)."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random


class StateRead(eqx.Module):
    """``queries`` ``[H m, Dk]`` and ``gate`` ``[1]`` (Adam); ``state_read_in`` ``[H m Dv, hidden]`` and
    ``state_read_out`` ``[hidden, D]`` (MuonH)."""

    queries: jax.Array
    gate: jax.Array
    state_read_in: jax.Array
    state_read_out: jax.Array


def state_read_init(
    key: jax.Array, heads: int, reads: int, dk: int, dv: int, hidden: int, d: int, std: float
) -> StateRead:
    kq, kin, kout = random.split(key, 3)
    width = heads * reads * dv
    return StateRead(
        queries=random.normal(kq, (heads * reads, dk)) / jnp.sqrt(dk),
        gate=jnp.zeros((1,), jnp.float32),
        state_read_in=random.normal(kin, (width, hidden)) * std,
        state_read_out=random.normal(kout, (hidden, d)) * std,
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
    return p.gate.astype(jnp.float32) * (hidden @ p.state_read_out.astype(jnp.float32))
