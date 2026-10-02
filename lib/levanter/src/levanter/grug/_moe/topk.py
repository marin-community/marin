# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Top-k indices over the last axis of a narrow ``[T, N]`` float32 array in one Triton kernel.

XLA:GPU uses its own top-k kernel only for rows of at least 1,024 elements (``TopkSpecializer``).
Narrower rows, such as a router's logits over a few hundred experts, fall back to a full sort of
every row followed by a slice: 1.2 ms per call at the hero's ``[65536, 384]`` on GB200.

Here a program holds a block of rows in registers, as one power-of-two tile or, when it pads less,
three equal power-of-two tiles (a 384-wide row needs no padding), and takes ``k`` successive maxima.
Each
step reduces twice: the largest key, then the lowest column holding it. The key is the float32 bit
pattern with the magnitude bits of negative values flipped, the total order XLA's sort comparator
uses, so ``+0`` ranks above ``-0`` and a positive NaN above ``+inf``. The indices and their order
are therefore those of ``jax.lax.top_k`` on GPU for every input, including ties, signed zeros and
NaNs. Rows of 1,024 or more go to ``jax.lax.top_k``, which XLA then runs on its own kernel.
"""

import functools

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from levanter.grug._moe.availability import gpu_device_present

try:
    import jax_triton as jt
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    jt = None
    triton = None
    tl = None

# XLA's own top-k kernel takes rows of at least this many entries.
_XLA_TOPK_MIN_ROW = 1024
# Best measured on GB200 at [65536, 384], k = 9: 0.102 ms against 0.835 ms for the XLA sort.
_BLOCK_ROWS = 4
_NUM_WARPS = 4


if triton is not None and tl is not None:

    @triton.jit
    def _load_tile(
        x_ptr, row, row_mask, n: tl.constexpr, width: tl.constexpr, tile: tl.constexpr, tiles: tl.constexpr
    ):
        col = tile * width + tl.arange(0, width)
        present = row_mask[:, None] & (col < n)[None, :] & (tile < tiles)
        bits = tl.load(x_ptr + row[:, None].to(tl.int64) * n + col[None, :], mask=present, other=0.0)
        bits = bits.to(tl.int32, bitcast=True)
        key = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
        return key, present, col[None, :] + tl.zeros(key.shape, tl.int32)

    @triton.jit
    def _live_keys(key, live):
        return tl.where(live, key, -2147483648)

    @triton.jit
    def _top_k_rows_kernel(
        x_ptr,  # (T, N) float32
        index_ptr,  # (T, K) int32
        rows: tl.constexpr,
        n: tl.constexpr,
        width: tl.constexpr,
        tiles: tl.constexpr,
        k: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        row_mask = row < rows
        # A taken or absent entry is not live. Liveness is tracked apart from the key because a
        # negative NaN's key is the smallest int32, the value a sentinel would need.
        key0, live0, col0 = _load_tile(x_ptr, row, row_mask, n, width, 0, tiles)
        key1, live1, col1 = _load_tile(x_ptr, row, row_mask, n, width, 1, tiles)
        key2, live2, col2 = _load_tile(x_ptr, row, row_mask, n, width, 2, tiles)
        for j in tl.static_range(k):
            best = _live_keys(key0, live0)
            if tiles == 3:
                best = tl.maximum(tl.maximum(best, _live_keys(key1, live1)), _live_keys(key2, live2))
            best = tl.max(best, axis=1)[:, None]
            column = tl.where(live0 & (key0 == best), col0, n)
            if tiles == 3:
                column = tl.minimum(column, tl.where(live1 & (key1 == best), col1, n))
                column = tl.minimum(column, tl.where(live2 & (key2 == best), col2, n))
            column = tl.min(column, axis=1)
            tl.store(index_ptr + row.to(tl.int64) * k + j, column, mask=row_mask)
            taken = column[:, None]
            live0 = live0 & (col0 != taken)
            live1 = live1 & (col1 != taken)
            live2 = live2 & (col2 != taken)

else:
    _top_k_rows_kernel = None


@functools.cache
def _tile_layout(n: int) -> tuple[int, int]:
    """(width, tiles): one power-of-two tile, or three when they cover ``n`` with fewer columns.

    Two or four equal power-of-two tiles never pad less than one tile, so those are the only cases.
    """
    one = triton.next_power_of_2(n)
    third = triton.next_power_of_2(triton.cdiv(n, 3))
    return (third, 3) if 3 * third < one else (one, 1)


def top_k_rows_available() -> bool:
    """Whether the Triton top-k kernel can run in this process."""
    return jt is not None and _top_k_rows_kernel is not None and gpu_device_present()


def top_k_indices(x: Float[Array, "T N"], k: int) -> Int[Array, "T K"]:
    """Indices of the ``k`` largest entries of each row, in ``jax.lax.top_k``'s order.

    Runs the Triton kernel for float32 rows of fewer than 1,024 entries on GPU, and
    ``jax.lax.top_k`` otherwise. The kernel has no partitioning rule, so under a sharded mesh call it inside
    ``shard_map``.
    """
    if x.ndim != 2:
        raise ValueError(f"expected a [T, N] array, got shape {x.shape}")
    rows, n = x.shape
    if not 0 < k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    if not top_k_rows_available() or x.dtype != jnp.float32 or n >= _XLA_TOPK_MIN_ROW:
        return jax.lax.top_k(x, k)[1]
    width, tiles = _tile_layout(n)
    # The indices are integers, so no gradient flows through them; stopping it here keeps autodiff
    # from asking the Triton call for a JVP rule it does not have.
    return jt.triton_call(
        jax.lax.stop_gradient(x),
        kernel=_top_k_rows_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, k), jnp.int32),
        grid=(triton.cdiv(rows, _BLOCK_ROWS),),
        num_warps=_NUM_WARPS,
        num_stages=1,
        rows=rows,
        n=n,
        width=width,
        tiles=tiles,
        k=k,
        block_rows=_BLOCK_ROWS,
    )
