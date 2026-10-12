# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The AttnRes backward's per-layer source-cotangent update as one Pallas (Triton) kernel.

Every layer adds ``sum_g weights_g[n] * d_mix_g`` to all ``N`` history sources' cotangents. XLA's fusion writes
the ``[N, T, D]`` result element by element and re-reads each token's ``d_mix_g`` row once per source (``N`` times
the minimum traffic). The kernel runs the sources as the fastest grid axis, so a token tile's ``d_mix`` rows stay
in L2 across the ``N`` programs that read them, and updates the cotangent stack in place. Same math (fp32 sum,
one rounding to the stack's dtype).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

TOKEN_BLOCK = 16
FEATURE_BLOCK = 256


def _kernel(*refs, num_gates: int):
    d_sources_ref = refs[0]
    weight_refs = refs[1 : 1 + num_gates]
    mix_refs = refs[1 + num_gates : 1 + 2 * num_gates]
    out_ref = refs[-1]
    acc = d_sources_ref[...].astype(jnp.float32)
    for weight_ref, mix_ref in zip(weight_refs, mix_refs, strict=True):
        acc = acc + weight_ref[...][:, None] * mix_ref[...]
    out_ref[...] = acc.astype(out_ref.dtype)


def accumulate_source_grads_local(
    d_sources: jax.Array,
    weights: tuple[jax.Array, ...],
    d_mixes: tuple[jax.Array, ...],
    *,
    interpret: bool = False,
) -> jax.Array:
    """``d_sources + sum_g weights[g][..., None] * d_mixes[g][None]`` on one device's tokens.

    ``d_sources`` is ``[N, T, D]``, each ``weights[g]`` ``[N, T]`` and each ``d_mixes[g]`` ``[T, D]``."""
    n, t, d = d_sources.shape
    if t % TOKEN_BLOCK or d % FEATURE_BLOCK:
        raise ValueError(f"source cotangents [{n}, {t}, {d}] need T % {TOKEN_BLOCK} == 0 and D % {FEATURE_BLOCK} == 0")
    num_gates = len(weights)
    grid = (n, t // TOKEN_BLOCK, d // FEATURE_BLOCK)
    return pl.pallas_call(
        functools.partial(_kernel, num_gates=num_gates),
        grid=grid,
        in_specs=[
            pl.BlockSpec((None, TOKEN_BLOCK, FEATURE_BLOCK), lambda s, i, j: (s, i, j)),
            *[pl.BlockSpec((None, TOKEN_BLOCK), lambda s, i, j: (s, i)) for _ in range(num_gates)],
            *[pl.BlockSpec((TOKEN_BLOCK, FEATURE_BLOCK), lambda s, i, j: (i, j)) for _ in range(num_gates)],
        ],
        out_specs=pl.BlockSpec((None, TOKEN_BLOCK, FEATURE_BLOCK), lambda s, i, j: (s, i, j)),
        out_shape=jax.ShapeDtypeStruct(d_sources.shape, d_sources.dtype),
        input_output_aliases={0: 0},
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
        interpret=interpret,
        name="attn_res_source_grads",
    )(d_sources, *weights, *d_mixes)
