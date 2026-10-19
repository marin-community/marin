# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Block AttnRes phase 1 (the gates' weighted mixes of the completed blocks) as Pallas (Triton) kernels.

The scanned AttnRes layers keep a fixed-shape ``[N, T, D]`` history whose not-yet-filled slots carry zero
weight. XLA's fusions read every slot, and the forward re-reads the whole history once per gate. Here each
token tile reads only the ``num_valid`` filled slots (always a prefix), once for all gates:

- ``block_mix``: ``mixed[g] = sum_{n < num_valid} weights[n, g] * sources[n]`` (float32 sum in source order,
  one rounding to the sources' dtype), the same values as the dense sum over every slot.
- its VJP: ``d_weights[n, g] = <d_mixed[g], sources[n]>`` for the filled slots (zero for the rest, whose weights
  are exactly zero) and ``d_sources[n] = sum_g weights[n, g] * d_mixed[g]`` (in plain JAX; XLA drops it when the
  sources' cotangent is unused, as in the scan's per-layer recompute where the sources are held constant).
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt


class Tiles(NamedTuple):
    """A kernel's token and feature tile (each capped at its dimension) and Triton launch parameters."""

    token_block: int
    feature_block: int
    num_warps: int
    num_stages: int


MIX_TILES = Tiles(16, 128, 8, 1)
DOT_TILES = Tiles(8, 128, 8, 2)


def block_mix_reference(sources: jax.Array, weights: jax.Array) -> jax.Array:
    """``[G, T, D]``: ``sum_n weights[n, g][:, None] * sources[n]`` in float32 over every slot, in source order."""
    mixed = weights[0][..., None] * sources[0][None].astype(jnp.float32)
    for n in range(1, sources.shape[0]):
        mixed = mixed + weights[n][..., None] * sources[n][None].astype(jnp.float32)
    return mixed.astype(sources.dtype)


def _blocks(t: int, d: int, token_block: int, feature_block: int) -> tuple[int, int]:
    """Token and feature tiles (each at most the dimension, which it must divide)."""
    tb, fb = min(token_block, t), min(feature_block, d)
    if t % tb or d % fb:
        raise ValueError(f"[T={t}, D={d}] needs T % {tb} == 0 and D % {fb} == 0")
    return tb, fb


def _mix_kernel(num_valid_ref, sources_ref, weights_ref, out_ref, *, num_gates: int):
    num_valid = num_valid_ref[0]

    def body(n, acc):
        source = sources_ref[n].astype(jnp.float32)  # [TB, FB]
        return tuple(acc[g] + weights_ref[n, g][:, None] * source for g in range(num_gates))

    zero = jnp.zeros(out_ref.shape[1:], jnp.float32)
    acc = jax.lax.fori_loop(0, num_valid, body, tuple(zero for _ in range(num_gates)))
    for g in range(num_gates):
        out_ref[g] = acc[g].astype(out_ref.dtype)


def block_mix_local(
    sources: jax.Array, weights: jax.Array, num_valid: jax.Array, *, tiles: Tiles = MIX_TILES, interpret: bool = False
) -> jax.Array:
    """``block_mix`` on one device's tokens: ``sources [N, T, D]``, float32 ``weights [N, G, T]`` that are zero
    on slots ``>= num_valid`` (int32 scalar) -> ``[G, T, D]`` in the sources' dtype."""
    n, t, d = sources.shape
    g = weights.shape[1]
    if weights.shape != (n, g, t) or weights.dtype != jnp.float32:
        raise ValueError(f"weights {weights.shape} {weights.dtype} must be float32 [{n}, G, {t}]")
    tb, fb = _blocks(t, d, tiles.token_block, tiles.feature_block)
    itemsize = sources.dtype.itemsize
    return pl.pallas_call(
        functools.partial(_mix_kernel, num_gates=g),
        grid=(t // tb, d // fb),
        in_specs=[
            pl.BlockSpec((1,), lambda i, j: (0,)),
            pl.BlockSpec((n, tb, fb), lambda i, j: (0, i, j)),
            pl.BlockSpec((n, g, tb), lambda i, j: (0, 0, i)),
        ],
        out_specs=pl.BlockSpec((g, tb, fb), lambda i, j: (0, i, j)),
        out_shape=jax.ShapeDtypeStruct((g, t, d), sources.dtype),
        compiler_params=plt.CompilerParams(num_warps=tiles.num_warps, num_stages=tiles.num_stages),
        # Upper bound (every slot filled): read the history and weights once, write G mixes.
        cost_estimate=pl.CostEstimate(
            flops=2 * n * g * t * d,
            transcendentals=0,
            bytes_accessed=n * t * d * itemsize + n * g * t * 4 + g * t * d * itemsize,
        ),
        interpret=interpret,
        name="attn_res_block_mix",
    )(jnp.reshape(num_valid, (1,)).astype(jnp.int32), sources, weights)


def _dot_kernel(num_valid_ref, sources_ref, d_mixed_ref, out_ref, *, num_gates: int, num_chunks: int):
    n = pl.program_id(0)
    fb = sources_ref.shape[-1] // num_chunks

    @pl.when(n < num_valid_ref[0])
    def _():
        def body(k, acc):
            cols = pl.ds(k * fb, fb)
            source = sources_ref[:, cols].astype(jnp.float32)  # [TB, FB]
            return tuple(
                acc[g] + jnp.sum(d_mixed_ref[g, :, cols].astype(jnp.float32) * source, axis=1) for g in range(num_gates)
            )

        zero = jnp.zeros(out_ref.shape[1:], jnp.float32)
        acc = jax.lax.fori_loop(0, num_chunks, body, tuple(zero for _ in range(num_gates)))
        for g in range(num_gates):
            out_ref[g] = acc[g]

    @pl.when(n >= num_valid_ref[0])
    def _():
        out_ref[...] = jnp.zeros(out_ref.shape, out_ref.dtype)


def block_mix_weight_grads_local(
    sources: jax.Array, d_mixed: jax.Array, num_valid: jax.Array, *, tiles: Tiles = DOT_TILES, interpret: bool = False
) -> jax.Array:
    """``[N, G, T]`` float32: ``<d_mixed[g, t], sources[n, t]>`` for ``n < num_valid``, zero for the other slots.

    ``sources [N, T, D]``, ``d_mixed [G, T, D]`` (any float dtype, read as float32)."""
    n, t, d = sources.shape
    g = d_mixed.shape[0]
    if d_mixed.shape != (g, t, d):
        raise ValueError(f"d_mixed {d_mixed.shape} must be [G, {t}, {d}]")
    tb, fb = _blocks(t, d, tiles.token_block, tiles.feature_block)
    return pl.pallas_call(
        functools.partial(_dot_kernel, num_gates=g, num_chunks=d // fb),
        # Slots fastest: a token tile's d_mixed rows stay in L2 across the slots that read them.
        grid=(n, t // tb),
        in_specs=[
            pl.BlockSpec((1,), lambda s, i: (0,)),
            pl.BlockSpec((None, tb, d), lambda s, i: (s, i, 0)),
            pl.BlockSpec((g, tb, d), lambda s, i: (0, i, 0)),
        ],
        out_specs=pl.BlockSpec((None, g, tb), lambda s, i: (s, 0, i)),
        out_shape=jax.ShapeDtypeStruct((n, g, t), jnp.float32),
        compiler_params=plt.CompilerParams(num_warps=tiles.num_warps, num_stages=tiles.num_stages),
        cost_estimate=pl.CostEstimate(
            flops=2 * n * g * t * d,
            transcendentals=0,
            bytes_accessed=n * t * d * sources.dtype.itemsize + g * t * d * d_mixed.dtype.itemsize + n * g * t * 4,
        ),
        interpret=interpret,
        name="attn_res_block_mix_dw",
    )(jnp.reshape(num_valid, (1,)).astype(jnp.int32), sources, d_mixed)
