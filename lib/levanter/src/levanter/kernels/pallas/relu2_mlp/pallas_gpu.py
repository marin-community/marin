# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Pallas Triton kernels for the ungated ReLU² expert MLP (modded-nanogpt's ``linear_relu_square``).

The unfused expert MLP writes the ``[E, R, N]`` pre-activation, re-reads it for ``relu(.)^2``, and in the
backward re-reads it again to form ``d pre``. These kernels fold both elementwise steps into GEMM
epilogues:

- ``relu2_up``: ``post = relu(x @ w_up)^2``. Only ``post`` reaches HBM; ``pre`` stays in registers.
- ``relu2_dpre``: ``d pre = (g @ w_down^T) * 2 sqrt(post)``, since ``relu(pre) = sqrt(post)``. The
  ``d post`` GEMM output never reaches HBM either.

Both are batched over the expert axis (grid ``(E, R / bm, N / bn)``) with an in-kernel loop over the
contraction dimension. Triton needs power-of-2 tiles, while our contraction and intermediate widths
(e.g. 384, 640) are not, so the refs are whole arrays and each program loads power-of-2 sub-tiles at
computed offsets. ``R`` is padded up to ``bm`` by the caller.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jaxtyping import Array, Float


@dataclass(frozen=True, slots=True)
class BlockSizes:
    bm: int = 64
    bn: int = 64
    bk: int = 64
    num_warps: int = 4
    num_stages: int = 3

    @classmethod
    def get_default(cls) -> "BlockSizes":
        return cls()


def _check(dim: int, block: int, name: str) -> None:
    if dim % block:
        raise ValueError(f"relu2_mlp: {name}={dim} must be a multiple of its block size {block}")


def _up_kernel(x_ref, w_ref, o_ref, *, bm: int, bn: int, bk: int, k: int):
    e, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    rows, cols = pl.ds(i * bm, bm), pl.ds(j * bn, bn)

    def body(kk, acc):
        xs = x_ref[e, rows, pl.ds(kk * bk, bk)]
        ws = w_ref[e, pl.ds(kk * bk, bk), cols]
        return acc + pl.dot(xs, ws)

    acc = jax.lax.fori_loop(0, k // bk, body, jnp.zeros((bm, bn), jnp.float32))
    relu = jnp.maximum(acc, 0.0)
    o_ref[e, rows, cols] = (relu * relu).astype(o_ref.dtype)


def _dpre_kernel(g_ref, w_ref, p_ref, o_ref, *, bm: int, bn: int, bk: int, m: int):
    e, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    rows, cols = pl.ds(i * bm, bm), pl.ds(j * bn, bn)

    def body(kk, acc):
        gs = g_ref[e, rows, pl.ds(kk * bk, bk)]
        ws = w_ref[e, cols, pl.ds(kk * bk, bk)]
        return acc + pl.dot(gs, ws, trans_b=True)

    acc = jax.lax.fori_loop(0, m // bk, body, jnp.zeros((bm, bn), jnp.float32))
    post = p_ref[e, rows, cols].astype(jnp.float32)
    o_ref[e, rows, cols] = (acc * 2.0 * jnp.sqrt(post)).astype(o_ref.dtype)


def _params(bs: BlockSizes) -> plgpu.CompilerParams:
    return plgpu.CompilerParams(num_warps=bs.num_warps, num_stages=bs.num_stages)


@partial(jax.jit, static_argnames=("block_sizes", "interpret"))
def relu2_up_pallas(
    x: Float[Array, "E R K"],
    w_up: Float[Array, "E K N"],
    *,
    block_sizes: BlockSizes = BlockSizes(),
    interpret: bool = False,
) -> Float[Array, "E R N"]:
    e, r, k = x.shape
    n = w_up.shape[-1]
    bs = block_sizes
    _check(r, bs.bm, "R")
    _check(n, bs.bn, "N")
    _check(k, bs.bk, "K")
    return pl.pallas_call(
        partial(_up_kernel, bm=bs.bm, bn=bs.bn, bk=bs.bk, k=k),
        out_shape=jax.ShapeDtypeStruct((e, r, n), x.dtype),
        grid=(e, r // bs.bm, n // bs.bn),
        compiler_params=_params(bs),
        cost_estimate=pl.CostEstimate(
            flops=2 * e * r * k * n,
            transcendentals=0,
            bytes_accessed=(x.size + w_up.size + e * r * n) * x.dtype.itemsize,
        ),
        interpret=interpret,
    )(x, w_up)


@partial(jax.jit, static_argnames=("block_sizes", "interpret"))
def relu2_dpre_pallas(
    g: Float[Array, "E R M"],
    w_down: Float[Array, "E N M"],
    post: Float[Array, "E R N"],
    *,
    block_sizes: BlockSizes = BlockSizes(),
    interpret: bool = False,
) -> Float[Array, "E R N"]:
    e, r, m = g.shape
    n = w_down.shape[1]
    bs = block_sizes
    _check(r, bs.bm, "R")
    _check(n, bs.bn, "N")
    _check(m, bs.bk, "M")
    return pl.pallas_call(
        partial(_dpre_kernel, bm=bs.bm, bn=bs.bn, bk=bs.bk, m=m),
        out_shape=jax.ShapeDtypeStruct((e, r, n), g.dtype),
        grid=(e, r // bs.bm, n // bs.bn),
        compiler_params=_params(bs),
        cost_estimate=pl.CostEstimate(
            flops=2 * e * r * m * n,
            transcendentals=e * r * n,
            bytes_accessed=(g.size + w_down.size + 2 * e * r * n) * g.dtype.itemsize,
        ),
        interpret=interpret,
    )(g, w_down, post)
