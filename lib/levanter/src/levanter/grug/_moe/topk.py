# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Top-k indices over the last axis of a narrow ``[T, N]`` float32 array in one Pallas kernel.

XLA:GPU uses its own top-k kernel only for rows of at least 1,024 elements (``TopkSpecializer``).
Narrower rows, such as a router's logits over a few hundred experts, fall back to a full sort of
every row followed by a slice: 1.2 ms per call at the hero's ``[65536, 384]`` on GB200.

Here a program holds a block of rows in registers, as one power-of-two tile or, when it pads less,
three equal power-of-two tiles (a 384-wide row needs no padding), and takes ``k`` successive maxima.
Pallas-Triton arrays must have power-of-two shapes, hence the tiles. Each step reduces twice: the
largest key, then the lowest column holding it. The key is the float32 bit pattern with the
magnitude bits of negative values flipped, the total order XLA's sort comparator uses, so ``+0``
ranks above ``-0`` and a positive NaN above ``+inf``. The indices and their order are therefore
those of ``jax.lax.top_k`` for every input, including ties, signed zeros and NaNs. Rows of 1,024 or
more go to ``jax.lax.top_k``, which XLA then runs on its own kernel.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton
from jaxtyping import Array, Float, Int

from levanter.grug._moe.availability import gpu_device_present

# XLA's own top-k kernel takes rows of at least this many entries.
_XLA_TOPK_MIN_ROW = 1024
# Best measured on GB200 at [65536, 384], k = 9: 0.107 ms against 0.859 ms for the XLA sort.
_BLOCK_ROWS = 1
_NUM_WARPS = 2
# Unrolled, the k steps run about 8% faster than under ``fori_loop`` at k = 9, but compile time
# grows faster than k: 26 s at k = 64.
_MAX_UNROLLED_K = 16
# Dead entries enter the maximum as the smallest int32, which a negative NaN's key can also be.
_INT32_MIN = -(2**31)


def _top_k_rows_kernel(bits_ref, index_ref, *, rows: int, n: int, width: int, tiles: int):
    block_rows, k = index_ref.shape
    row = pl.program_id(0) * block_rows + jax.lax.broadcasted_iota(jnp.int32, (block_rows, 1), 0)
    row_ok = row < rows
    # A taken or absent entry is not live. Liveness is tracked apart from the key because a
    # negative NaN's key is the smallest int32, the value a sentinel would need.
    keys, lives, cols = [], [], []
    for tile in range(tiles):
        col = tile * width + jax.lax.broadcasted_iota(jnp.int32, (block_rows, width), 1)
        present = row_ok & (col < n)
        bits = pltriton.load(bits_ref.at[:, pl.ds(tile * width, width)], mask=present, other=0)
        keys.append(jnp.where(bits < 0, bits ^ 0x7FFFFFFF, bits))
        lives.append(present)
        cols.append(col)

    def step(j, lives):
        best = functools.reduce(jnp.maximum, [jnp.where(live, key, _INT32_MIN) for key, live in zip(keys, lives)])
        best = jnp.max(best, axis=1, keepdims=True)
        column = functools.reduce(
            jnp.minimum, [jnp.where(live & (key == best), col, n) for key, live, col in zip(keys, lives, cols)]
        )
        column = jnp.min(column, axis=1, keepdims=True)
        pltriton.store(index_ref.at[:, pl.ds(j, 1)], column, mask=row_ok)
        return tuple(live & (col != column) for live, col in zip(lives, cols))

    lives = tuple(lives)
    if k > _MAX_UNROLLED_K:
        jax.lax.fori_loop(0, k, step, lives)
    else:
        for j in range(k):
            lives = step(j, lives)


@functools.cache
def _tile_layout(n: int) -> tuple[int, int]:
    """(width, tiles): one power-of-two tile, or three when they cover ``n`` with fewer columns.

    Two or four equal power-of-two tiles never pad less than one tile, so those are the only cases.
    """
    one = pl.next_power_of_2(n)
    third = pl.next_power_of_2(pl.cdiv(n, 3))
    return (third, 3) if 3 * third < one else (one, 1)


def top_k_indices(x: Float[Array, "T N"], k: int, *, interpret: bool = False) -> Int[Array, "T K"]:
    """Indices of the ``k`` largest entries of each row, in ``jax.lax.top_k``'s order.

    Runs the Pallas kernel for float32 rows of fewer than 1,024 entries on GPU, and
    ``jax.lax.top_k`` otherwise. The kernel has no partitioning rule, so under a sharded mesh call
    it inside ``shard_map`` with ``check_vma=False``: a Pallas call's output has no varying-axes type. ``interpret=True`` runs the kernel body through the Pallas interpreter
    on any backend, for tests.
    """
    if x.ndim != 2:
        raise ValueError(f"expected a [T, N] array, got shape {x.shape}")
    rows, n = x.shape
    if not 0 < k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    if not (interpret or gpu_device_present()) or x.dtype != jnp.float32 or n >= _XLA_TOPK_MIN_ROW:
        return jax.lax.top_k(x, k)[1]
    width, tiles = _tile_layout(n)
    # XLA compiles the int32 view to a bitcast. An integer input carries no gradient, so autodiff
    # never asks the kernel for a JVP rule.
    bits = jax.lax.bitcast_convert_type(x, jnp.int32)
    return pl.pallas_call(
        functools.partial(_top_k_rows_kernel, rows=rows, n=n, width=width, tiles=tiles),
        out_shape=jax.ShapeDtypeStruct((rows, k), jnp.int32),
        grid=(pl.cdiv(rows, _BLOCK_ROWS),),
        # The block spans every tile; the kernel masks off columns past ``n``.
        in_specs=[pl.BlockSpec((_BLOCK_ROWS, width * tiles), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((_BLOCK_ROWS, k), lambda i: (i, 0)),
        compiler_params=pltriton.CompilerParams(num_warps=_NUM_WARPS, num_stages=1),
        interpret=interpret,
        # Two passes over the row per step, one compare each.
        cost_estimate=pl.CostEstimate(
            flops=2 * k * rows * width * tiles,
            transcendentals=0,
            bytes_accessed=bits.size * bits.dtype.itemsize + rows * k * 4,
        ),
        name="top_k_rows",
    )(bits)
