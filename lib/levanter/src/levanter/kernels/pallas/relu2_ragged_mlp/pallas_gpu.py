# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Pallas Triton grouped (ragged) GEMMs for the ungated ReLU² expert MLP.

Rows of ``x`` are sorted by group; group ``g`` owns rows ``[lo_g, lo_g + group_sizes[g])``. Two kernels:

- ``gmm``: ``out[r] = epilogue(a[r] @ b[g(r)])`` over a flat grid of group-aligned row tiles. A tile never
  straddles two groups, so the grid is ``cdiv(M, bm) + G + 1`` tiles (an upper bound; surplus tiles exit
  at once) rather than haliax's ``cdiv(M, bm) * G``. Rows past ``sum(group_sizes)`` form one extra
  "padding group" whose output is written as zeros without running the contraction. Row loads and stores
  are masked to the tile's group, so the last tile never reads or writes past the buffer. Epilogues:
  none, ``relu(.)^2``, and ``(.) * 2 sqrt(post)`` (the ReLU² backward, since ``relu(pre) = sqrt(post)``).
- ``tgmm``: ``out[g] = a_g^T @ b_g``, the weight gradient, with the group's rows split ``splits`` ways into
  fp32 partial sums so a few dozen groups still fill the GPU.

Triton needs power-of-2 tiles, so refs are whole arrays and each program loads power-of-2 sub-tiles at
computed offsets; the column and contraction widths must be multiples of their blocks.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jaxtyping import Array, Float, Int


@dataclass(frozen=True, slots=True)
class GmmBlockSizes:
    """Defaults from an H100 sweep at the d512 ragged-EP chunk (M=301466, G=24, K=256, N=384, bf16)."""

    bm: int = 128
    bn: int = 128
    bk: int = 64
    num_warps: int = 8
    num_stages: int = 3


@dataclass(frozen=True, slots=True)
class TgmmBlockSizes:
    """Defaults from the same H100 sweep (fused fwd+bwd 1.59 ms vs 2.69 ms for haliax ragged_dot + ReLU²)."""

    bm: int = 128
    bn: int = 128
    bk: int = 32
    splits: int = 2
    num_warps: int = 4
    num_stages: int = 3


@dataclass(frozen=True, slots=True)
class BlockSizes:
    """Tiles for the row-grouped GEMMs (up, down, d pre, dx) and the weight-gradient GEMMs.

    Defaults start from the batched ``relu2_mlp`` H100 sweep (bm=bn=bk=64); sweep with
    ``scratch_lc1/bench_relu2_ragged_kernel.py``.
    """

    gmm: GmmBlockSizes = field(default_factory=GmmBlockSizes)
    tgmm: TgmmBlockSizes = field(default_factory=TgmmBlockSizes)

    @classmethod
    def get_default(cls) -> "BlockSizes":
        return cls()


class Epilogue(IntEnum):
    NONE = 0
    RELU2 = 1
    RELU2_DPRE = 2


def _check(dim: int, block: int, name: str) -> None:
    if dim % block:
        raise ValueError(f"relu2_ragged_mlp: {name}={dim} must be a multiple of its block size {block}")


def _tile_metadata(
    group_sizes: Int[Array, "G"], rows: int, bm: int
) -> tuple[Int[Array, "T"], Int[Array, "T"], Int[Array, "T"], int]:
    """Group, first row and end row of each group-aligned row tile.

    Group ``G`` is the padding group, rows ``[sum(group_sizes), rows)``. Tiles past the last real tile get
    ``start == end`` and do nothing.
    """
    num_groups = group_sizes.shape[0]
    sizes = jnp.concatenate([group_sizes.astype(jnp.int32), jnp.zeros((1,), jnp.int32)])
    sizes = sizes.at[-1].set(rows - jnp.sum(sizes))
    row_starts = jnp.cumulative_sum(sizes, include_initial=True)[:-1]
    tiles = (sizes + bm - 1) // bm
    tile_ends = jnp.cumulative_sum(tiles)
    num_tiles = pl.cdiv(rows, bm) + num_groups + 1
    t = jnp.arange(num_tiles, dtype=jnp.int32)
    group = jnp.minimum(jnp.searchsorted(tile_ends, t, side="right"), num_groups).astype(jnp.int32)
    first_tile = tile_ends[group] - tiles[group]
    start = row_starts[group] + (t - first_tile) * bm
    end = row_starts[group] + sizes[group]
    live = t < tile_ends[-1]
    return group, jnp.where(live, start, 0), jnp.where(live, end, 0), num_tiles


def _gmm_kernel(
    group_ref,
    start_ref,
    end_ref,
    a_ref,
    b_ref,
    *rest,
    bm: int,
    bn: int,
    bk: int,
    contraction: int,
    num_groups: int,
    trans_b: bool,
    epilogue: Epilogue,
):
    if epilogue == Epilogue.RELU2_DPRE:
        post_ref, o_ref = rest
    else:
        (o_ref,) = rest
    group, start, end = group_ref[()], start_ref[()], end_ref[()]
    # The interpreter only substitutes ``program_id`` outside control flow.
    cols = pl.ds(pl.program_id(1) * bn, bn)

    @pl.when(start < end)
    def _compute():
        rows = pl.ds(start, bm)
        row_mask = (start + jnp.arange(bm) < end)[:, None]
        is_padding = group == num_groups
        weight_group = jnp.minimum(group, num_groups - 1)

        def body(kk, acc):
            span_k = pl.ds(kk * bk, bk)
            a = plgpu.load(a_ref.at[rows, span_k], mask=row_mask, other=0.0)
            if trans_b:
                b = plgpu.load(b_ref.at[weight_group, cols, span_k])
            else:
                b = plgpu.load(b_ref.at[weight_group, span_k, cols])
            return acc + plgpu.dot(a, b, trans_b=trans_b)

        # Triton's scf.for needs bounds of one type; a weakly typed where() would not lower to int32.
        steps = jnp.where(is_padding, jnp.int32(0), jnp.int32(contraction // bk))
        acc = jax.lax.fori_loop(jnp.int32(0), steps, body, jnp.zeros((bm, bn), jnp.float32))
        if epilogue == Epilogue.RELU2:
            relu = jnp.maximum(acc, 0.0)
            acc = relu * relu
        elif epilogue == Epilogue.RELU2_DPRE:
            post = plgpu.load(post_ref.at[rows, cols], mask=row_mask, other=0.0).astype(jnp.float32)
            acc = acc * 2.0 * jnp.sqrt(post)
        plgpu.store(o_ref.at[rows, cols], acc.astype(o_ref.dtype), mask=row_mask)


@partial(jax.jit, static_argnames=("trans_b", "epilogue", "block_sizes", "interpret"))
def gmm(
    a: Float[Array, "M Kc"],
    b: Float[Array, "G Kc N"] | Float[Array, "G N Kc"],
    group_sizes: Int[Array, "G"],
    post: Float[Array, "M N"] | None = None,
    *,
    trans_b: bool,
    epilogue: Epilogue,
    block_sizes: GmmBlockSizes,
    interpret: bool = False,
) -> Float[Array, "M N"]:
    """``epilogue(a[r] @ b[g(r)])`` (``b[g(r)]^T`` when ``trans_b``); rows past ``sum(group_sizes)`` are zero."""
    rows, contraction = a.shape
    num_groups = b.shape[0]
    n = b.shape[1] if trans_b else b.shape[2]
    bs = block_sizes
    _check(n, bs.bn, "N")
    _check(contraction, bs.bk, "K")
    if (post is None) != (epilogue != Epilogue.RELU2_DPRE):
        raise ValueError("gmm: post is required by, and only by, the RELU2_DPRE epilogue")
    group, start, end, num_tiles = _tile_metadata(group_sizes, rows, bs.bm)
    tile_spec = pl.BlockSpec((None,), lambda t, j: (t,))
    operands = [group, start, end, a, b] + ([] if post is None else [post])
    extra_bytes = 0 if post is None else post.size * post.dtype.itemsize
    return pl.pallas_call(
        partial(
            _gmm_kernel,
            bm=bs.bm,
            bn=bs.bn,
            bk=bs.bk,
            contraction=contraction,
            num_groups=num_groups,
            trans_b=trans_b,
            epilogue=epilogue,
        ),
        out_shape=jax.ShapeDtypeStruct((rows, n), a.dtype),
        grid=(num_tiles, n // bs.bn),
        in_specs=[tile_spec] * 3 + [pl.no_block_spec] * (len(operands) - 3),
        out_specs=pl.no_block_spec,
        compiler_params=plgpu.CompilerParams(num_warps=bs.num_warps, num_stages=bs.num_stages),
        cost_estimate=pl.CostEstimate(
            flops=2 * rows * contraction * n,
            transcendentals=rows * n if epilogue == Epilogue.RELU2_DPRE else 0,
            bytes_accessed=(a.size + b.size + rows * n) * a.dtype.itemsize + extra_bytes,
        ),
        interpret=interpret,
    )(*operands)


def _tgmm_kernel(lo_ref, hi_ref, a_ref, b_ref, o_ref, *, bm: int, bn: int, bk: int, splits: int):
    lo, hi = lo_ref[()], hi_ref[()]
    split = pl.program_id(1)
    split_rows = pl.cdiv(pl.cdiv(hi - lo, splits), bk) * bk
    row_lo = lo + split * split_rows
    row_hi = jnp.minimum(row_lo + split_rows, hi)
    a_cols = pl.ds(pl.program_id(2) * bm, bm)
    b_cols = pl.ds(pl.program_id(3) * bn, bn)

    def body(i, acc):
        row0 = row_lo + i * bk
        rows = pl.ds(row0, bk)
        mask = (row0 + jnp.arange(bk) < row_hi)[:, None]
        a = plgpu.load(a_ref.at[rows, a_cols], mask=mask, other=0.0)
        b = plgpu.load(b_ref.at[rows, b_cols], mask=mask, other=0.0)
        return acc + plgpu.dot(a, b, trans_a=True)

    steps = pl.cdiv(jnp.maximum(row_hi - row_lo, 0), bk).astype(jnp.int32)
    acc = jax.lax.fori_loop(jnp.int32(0), steps, body, jnp.zeros((bm, bn), jnp.float32))
    o_ref[...] = acc


@partial(jax.jit, static_argnames=("block_sizes", "interpret"))
def tgmm(
    a: Float[Array, "M Ka"],
    b: Float[Array, "M Nb"],
    group_sizes: Int[Array, "G"],
    *,
    block_sizes: TgmmBlockSizes,
    interpret: bool = False,
) -> Float[Array, "G Ka Nb"]:
    """``a_g^T @ b_g`` per group, in ``a.dtype`` (fp32 accumulation); rows past ``sum(group_sizes)`` are unused."""
    _, ka = a.shape
    nb = b.shape[1]
    num_groups = group_sizes.shape[0]
    bs = block_sizes
    _check(ka, bs.bm, "Ka")
    _check(nb, bs.bn, "Nb")
    cum = jnp.cumulative_sum(group_sizes.astype(jnp.int32), include_initial=True)
    group_spec = pl.BlockSpec((None,), lambda g, s, i, j: (g,))
    partial_sums = pl.pallas_call(
        partial(_tgmm_kernel, bm=bs.bm, bn=bs.bn, bk=bs.bk, splits=bs.splits),
        out_shape=jax.ShapeDtypeStruct((num_groups, bs.splits, ka, nb), jnp.float32),
        grid=(num_groups, bs.splits, ka // bs.bm, nb // bs.bn),
        in_specs=[group_spec, group_spec, pl.no_block_spec, pl.no_block_spec],
        out_specs=pl.BlockSpec((None, None, bs.bm, bs.bn), lambda g, s, i, j: (g, s, i, j)),
        compiler_params=plgpu.CompilerParams(num_warps=bs.num_warps, num_stages=bs.num_stages),
        cost_estimate=pl.CostEstimate(
            flops=2 * a.shape[0] * ka * nb,
            transcendentals=0,
            bytes_accessed=(a.size + b.size) * a.dtype.itemsize + num_groups * bs.splits * ka * nb * 4,
        ),
        interpret=interpret,
    )(cum[:-1], cum[1:], a, b)
    return jnp.sum(partial_sums, axis=1).astype(a.dtype)
