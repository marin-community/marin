# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Top-k indices over the last axis of a narrow ``[T, N]`` float32 array in one Triton kernel.

XLA:GPU uses its own top-k kernel only for rows of at least 1,024 elements (``TopkSpecializer``).
Narrower rows, such as a router's logits over a few hundred experts, fall back to a full sort of
every row followed by a slice: 1.2 ms per call at the hero's ``[65536, 384]`` on GB200.

Here one warp holds one row in registers, as one power-of-two tile or, when it pads less, three
equal power-of-two tiles (a 384-wide row needs no padding), and takes ``k`` successive maxima. Each
step reduces twice: the largest key, then the lowest column holding it. The key is the float32 bit
pattern with the magnitude bits of negative values flipped, the total order XLA's sort comparator
uses, so ``+0`` ranks above ``-0`` and a positive NaN above ``+inf``. The indices and their order are
therefore those of ``jax.lax.top_k`` for every input, including ties, signed zeros and NaNs. Rows of
1,024 or more go to ``jax.lax.top_k``, which XLA then runs on its own kernel.

A taken or absent entry gets the smallest key and the column ``n``, which the lowest-column
reduction never returns. The column marks the entry dead because a negative NaN such as 0xFFFFFFFF
also has the smallest key: with a key sentinel alone, a taken entry could tie with that NaN.

A warp per row keeps both reductions inside the warp. Triton 3.6 reduces across warps through
shared memory twice, with three barriers per reduction, so a row split over two warps runs slower.
"""

import functools

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

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
_NUM_WARPS = 1
# Unrolled, the k steps run about 6% faster than a loop at k = 9 on GB200, but compile time grows
# with k: 36 s at k = 384, against 0.2 s for the loop.
_MAX_UNROLLED_K = 16


if triton is not None and tl is not None:

    @triton.jit
    def _load_tile(row_ptr, n: tl.constexpr, width: tl.constexpr, tile: tl.constexpr):
        col = tile * width + tl.arange(0, width)
        present = col < n
        bits = tl.load(row_ptr + col, mask=present, other=0)
        key = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
        return tl.where(present, key, -2147483648), tl.where(present, col, n)

    @triton.jit
    def _take(key, col, column, n: tl.constexpr):
        taken = col == column
        return tl.where(taken, -2147483648, key), tl.where(taken, n, col)

    @triton.jit
    def _select(out_ptr, j, key0, col0, key1, col1, key2, col2, n: tl.constexpr, tiles: tl.constexpr):
        best = key0
        if tiles == 3:
            best = tl.maximum(tl.maximum(best, key1), key2)
        best = tl.max(best, axis=0)
        column = tl.where(key0 == best, col0, n)
        if tiles == 3:
            column = tl.minimum(column, tl.where(key1 == best, col1, n))
            column = tl.minimum(column, tl.where(key2 == best, col2, n))
        column = tl.min(column, axis=0)
        tl.store(out_ptr + j, column)
        key0, col0 = _take(key0, col0, column, n)
        if tiles == 3:
            key1, col1 = _take(key1, col1, column, n)
            key2, col2 = _take(key2, col2, column, n)
        return key0, col0, key1, col1, key2, col2

    @triton.jit
    def _top_k_rows_kernel(
        bits_ptr,  # (T, N) int32 view of float32
        index_ptr,  # (T, K) int32
        n: tl.constexpr,
        width: tl.constexpr,
        tiles: tl.constexpr,
        k: tl.constexpr,
        unroll: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        row_ptr = bits_ptr + row * n
        out_ptr = index_ptr + row * k
        key0, col0 = _load_tile(row_ptr, n, width, 0)
        key1, col1, key2, col2 = key0, col0, key0, col0
        if tiles == 3:
            key1, col1 = _load_tile(row_ptr, n, width, 1)
            key2, col2 = _load_tile(row_ptr, n, width, 2)
        if unroll:
            for j in tl.static_range(k):
                key0, col0, key1, col1, key2, col2 = _select(out_ptr, j, key0, col0, key1, col1, key2, col2, n, tiles)
        else:
            for j in range(k):
                key0, col0, key1, col1, key2, col2 = _select(out_ptr, j, key0, col0, key1, col1, key2, col2, n, tiles)

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


def top_k_indices(x: Float[Array, "T N"], k: int) -> Int[Array, "T K"]:
    """Indices of the ``k`` largest entries of each row, in ``jax.lax.top_k``'s order.

    Runs the Triton kernel for float32 rows of fewer than 1,024 entries on GPU, and
    ``jax.lax.top_k`` otherwise. The kernel has no partitioning rule, so under a sharded mesh call
    it inside ``shard_map``.
    """
    if x.ndim != 2:
        raise ValueError(f"expected a [T, N] array, got shape {x.shape}")
    rows, n = x.shape
    if not 0 < k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    # The default device rather than jax.default_backend(): a test that stubs the backend still runs on
    # CPU devices, which cannot launch the kernel.
    on_gpu = jax.devices()[0].platform == "gpu"
    if jt is None or not on_gpu or x.dtype != jnp.float32 or n >= _XLA_TOPK_MIN_ROW:
        return jax.lax.top_k(x, k)[1]
    width, tiles = _tile_layout(n)
    # XLA compiles the int32 view to a bitcast. An integer input carries no gradient, so autodiff
    # never asks the kernel for a JVP rule.
    bits = jax.lax.bitcast_convert_type(x, jnp.int32)
    return jt.triton_call(
        bits,
        kernel=_top_k_rows_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, k), jnp.int32),
        grid=(rows,),
        num_warps=_NUM_WARPS,
        num_stages=1,
        n=n,
        width=width,
        tiles=tiles,
        k=k,
        unroll=k <= _MAX_UNROLLED_K,
    )
