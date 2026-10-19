# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The AttnRes backward's per-layer source-cotangent update as one Pallas (Triton) kernel.

Every layer adds ``sum_g weights_g[n] * d_mix_g`` to all ``N`` history sources' cotangents. XLA's fusion writes
the ``[N, T, D]`` result element by element and re-reads each token's ``d_mix_g`` row once per source (``N`` times
the minimum traffic). The kernel runs the sources as the fastest grid axis, so a token tile's ``d_mix`` rows stay
in L2 across the ``N`` programs that read them, and updates the cotangent stack in place. Same math (fp32 sum,
one rounding to the stack's dtype). Slots ``>= num_valid`` (history slots no block has filled yet, zero weight in
every gate) are skipped: their cotangents stay as they are, which is what adding ``0 * d_mix`` gives.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from experiments.grug.fast_track.attn_res_block_mix import Tiles

SOURCE_GRAD_TILES = Tiles(16, 128, 8, 1)


def _kernel(*refs, num_gates: int):
    num_valid_ref = refs[0]
    d_sources_ref = refs[1]
    weight_refs = refs[2 : 2 + num_gates]
    mix_refs = refs[2 + num_gates : 2 + 2 * num_gates]
    out_ref = refs[-1]

    @pl.when(pl.program_id(0) < num_valid_ref[0])
    def _():
        acc = d_sources_ref[...].astype(jnp.float32)
        for weight_ref, mix_ref in zip(weight_refs, mix_refs, strict=True):
            acc = acc + weight_ref[...][:, None] * mix_ref[...]
        out_ref[...] = acc.astype(out_ref.dtype)


def accumulate_source_grads_local(
    d_sources: jax.Array,
    weights: tuple[jax.Array, ...],
    d_mixes: tuple[jax.Array, ...],
    num_valid: jax.Array,
    *,
    tiles: Tiles = SOURCE_GRAD_TILES,
    interpret: bool = False,
) -> jax.Array:
    """``d_sources + sum_g weights[g][..., None] * d_mixes[g][None]`` on one device's tokens.

    ``d_sources`` is ``[N, T, D]``, each ``weights[g]`` ``[N, T]`` (zero on slots ``>= num_valid``, an int32
    scalar) and each ``d_mixes[g]`` ``[T, D]``."""
    n, t, d = d_sources.shape
    tb, fb = min(tiles.token_block, t), min(tiles.feature_block, d)
    if t % tb or d % fb:
        raise ValueError(f"source cotangents [{n}, {t}, {d}] need T % {tb} == 0 and D % {fb} == 0")
    num_gates = len(weights)
    grid = (n, t // tb, d // fb)
    return pl.pallas_call(
        functools.partial(_kernel, num_gates=num_gates),
        grid=grid,
        in_specs=[
            pl.BlockSpec((1,), lambda s, i, j: (0,)),
            pl.BlockSpec((None, tb, fb), lambda s, i, j: (s, i, j)),
            *[pl.BlockSpec((None, tb), lambda s, i, j: (s, i)) for _ in range(num_gates)],
            *[pl.BlockSpec((tb, fb), lambda s, i, j: (i, j)) for _ in range(num_gates)],
        ],
        out_specs=pl.BlockSpec((None, tb, fb), lambda s, i, j: (s, i, j)),
        out_shape=jax.ShapeDtypeStruct(d_sources.shape, d_sources.dtype),
        input_output_aliases={1: 0},
        compiler_params=plt.CompilerParams(num_warps=tiles.num_warps, num_stages=tiles.num_stages),
        interpret=interpret,
        name="attn_res_source_grads",
    )(jnp.reshape(num_valid, (1,)).astype(jnp.int32), d_sources, *weights, *d_mixes)
