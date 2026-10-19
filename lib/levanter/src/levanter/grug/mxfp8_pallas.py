# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-pass MXFP8 quantization as Pallas (Triton) kernels.

XLA lowers ``mx_quantize_rows`` / ``mx_quantize_cols`` (``levanter.grug.mxfp8``) as an amax reduction and a separate
scaling pass, each reading the input, and rematerializes the reduction in the backward recompute. These kernels read
each tile once and write the e4m3 values and the biased e8m0 exponents, with the same rounding (bit-identical).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from levanter.grug.mxfp8 import (
    MX_BLOCK,
    mx_dequantize_rows,
    mx_quantize_cols,
    mx_quantize_row_blocks,
    mx_quantize_rows,
)

_E4M3_MAX = 448.0
ROW_TILE = 16
"""Rows per program of the row quantizer; each program covers up to ``MX_BLOCK * MAX_COL_BLOCKS`` columns. Tuned on
GB200 (``scratch_lc1/mxq_tune.py``): 16 x 512 runs 688128 x 3072 in 1.83 ms vs 2.14 ms at 64 x 128."""
MAX_COL_BLOCKS = 16


def _col_blocks(k: int) -> int | None:
    """The most 32-column blocks per program (at most ``MAX_COL_BLOCKS``) that tile ``k``; None if none do."""
    if k % MX_BLOCK:
        return None
    blocks = k // MX_BLOCK
    return next((c for c in (16, 8, 4, 2, 1) if c <= MAX_COL_BLOCKS and blocks % c == 0), None)


def _exponent(amax):
    bits = jax.lax.bitcast_convert_type(amax / _E4M3_MAX, jnp.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).astype(jnp.int32)
    return jnp.where(amax > 0, jnp.clip(exponent, 1, 254), 127)


def _rows_kernel(x_ref, values_ref, exponents_ref, *, col_blocks: int):
    for j in range(col_blocks):
        block = x_ref[:, j * MX_BLOCK : (j + 1) * MX_BLOCK].astype(jnp.float32)  # [R, 32]
        exponent = _exponent(jnp.max(jnp.abs(block), axis=1))  # [R]
        scale = jax.lax.bitcast_convert_type(exponent << 23, jnp.float32)
        values_ref[:, j * MX_BLOCK : (j + 1) * MX_BLOCK] = (block / scale[:, None]).astype(values_ref.dtype)
        exponents_ref[:, j] = exponent.astype(exponents_ref.dtype)


@functools.partial(jax.jit, static_argnames=("interpret",))
def mx_quantize_rows_pallas(x: jax.Array, *, interpret: bool = False) -> tuple[jax.Array, jax.Array]:
    """``mx_quantize_rows`` of a local ``x[M, K]`` in one pass: e4m3 ``[M, K]`` and uint8 exponents ``[M, K / 32]``."""
    m, k = x.shape
    col_blocks = _col_blocks(k)
    if m % ROW_TILE or col_blocks is None:
        raise ValueError(f"mx_quantize_rows_pallas needs M % {ROW_TILE} == 0 and K % {MX_BLOCK} == 0, got {x.shape}")
    cols = MX_BLOCK * col_blocks
    return pl.pallas_call(
        functools.partial(_rows_kernel, col_blocks=col_blocks),
        grid=(m // ROW_TILE, k // cols),
        in_specs=[pl.BlockSpec((ROW_TILE, cols), lambda i, j: (i, j))],
        out_specs=[
            pl.BlockSpec((ROW_TILE, cols), lambda i, j: (i, j)),
            pl.BlockSpec((ROW_TILE, col_blocks), lambda i, j: (i, j)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((m, k), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((m, k // MX_BLOCK), jnp.uint8),
        ],
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
        interpret=interpret,
        name="mx_quantize_rows",
    )(x)


COL_TILE = 128
"""Columns per program of the column quantizer (32 rows each)."""


def _cols_kernel(x_ref, values_ref, exponents_ref):
    block = x_ref[...].astype(jnp.float32)  # [32, C]
    exponent = _exponent(jnp.max(jnp.abs(block), axis=0))  # [C]
    scale = jax.lax.bitcast_convert_type(exponent << 23, jnp.float32)
    values_ref[...] = (block / scale[None, :]).astype(values_ref.dtype)
    exponents_ref[0, :] = exponent.astype(exponents_ref.dtype)


@functools.partial(jax.jit, static_argnames=("interpret",))
def mx_quantize_cols_pallas(x: jax.Array, *, interpret: bool = False) -> tuple[jax.Array, jax.Array]:
    """``mx_quantize_cols`` of a local ``x[T, K]`` in one pass: e4m3 ``[T, K]`` and uint8 exponents ``[T / 32, K]``."""
    t, k = x.shape
    if t % MX_BLOCK or k % COL_TILE:
        raise ValueError(f"mx_quantize_cols_pallas needs T % {MX_BLOCK} == 0 and K % {COL_TILE} == 0, got {x.shape}")
    return pl.pallas_call(
        _cols_kernel,
        grid=(t // MX_BLOCK, k // COL_TILE),
        in_specs=[pl.BlockSpec((MX_BLOCK, COL_TILE), lambda i, j: (i, j))],
        out_specs=[
            pl.BlockSpec((MX_BLOCK, COL_TILE), lambda i, j: (i, j)),
            pl.BlockSpec((1, COL_TILE), lambda i, j: (i, j)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((t, k), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((t // MX_BLOCK, k), jnp.uint8),
        ],
        compiler_params=plt.CompilerParams(num_warps=2, num_stages=1),
        interpret=interpret,
        name="mx_quantize_cols",
    )(x)


# Triton computes block offsets in int32: a call must stay under 2^31 elements. Larger tensors take XLA's quantizer
# (slicing them into chunks for the kernel costs more than XLA's own pass).
_MAX_CALL_ELEMENTS = 2**31 - 1


def quantize_rows(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``mx_quantize_rows`` of a local 2-D ``x``: the single-pass kernel when it fits, else XLA's."""
    m, k = x.shape
    if m % ROW_TILE or _col_blocks(k) is None or x.size > _MAX_CALL_ELEMENTS:
        return mx_quantize_rows(x)
    return mx_quantize_rows_pallas(x)


def quantize_cols(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``mx_quantize_cols`` of a local 2-D ``x``: the single-pass kernel when it fits, else XLA's."""
    t, k = x.shape
    if t % MX_BLOCK or k % COL_TILE or x.size > _MAX_CALL_ELEMENTS:
        return mx_quantize_cols(x)
    return mx_quantize_cols_pallas(x)


def _rows_cols_kernel(x_ref, row_values_ref, row_exponents_ref, col_values_ref, col_exponents_ref):
    block = x_ref[...].astype(jnp.float32)  # [32, C]
    col_exponent = _exponent(jnp.max(jnp.abs(block), axis=0))  # [C]
    col_scale = jax.lax.bitcast_convert_type(col_exponent << 23, jnp.float32)
    col_values_ref[...] = (block / col_scale[None, :]).astype(col_values_ref.dtype)
    col_exponents_ref[0, :] = col_exponent.astype(col_exponents_ref.dtype)
    for j in range(COL_TILE // MX_BLOCK):
        part = x_ref[:, j * MX_BLOCK : (j + 1) * MX_BLOCK].astype(jnp.float32)  # [32, 32], from cache
        row_exponent = _exponent(jnp.max(jnp.abs(part), axis=1))  # [32]
        row_scale = jax.lax.bitcast_convert_type(row_exponent << 23, jnp.float32)
        row_values_ref[:, j * MX_BLOCK : (j + 1) * MX_BLOCK] = (part / row_scale[:, None]).astype(row_values_ref.dtype)
        row_exponents_ref[:, j] = row_exponent.astype(row_exponents_ref.dtype)


@functools.partial(jax.jit, static_argnames=("interpret",))
def mx_quantize_rows_cols_pallas(x: jax.Array, *, interpret: bool = False):
    """``(mx_quantize_rows(x), mx_quantize_cols(x))`` of a local ``x[T, K]`` reading ``x`` once."""
    t, k = x.shape
    if t % MX_BLOCK or k % COL_TILE:
        raise ValueError(
            f"mx_quantize_rows_cols_pallas needs T % {MX_BLOCK} == 0 and K % {COL_TILE} == 0, got {x.shape}"
        )
    row_values, row_exponents, col_values, col_exponents = pl.pallas_call(
        _rows_cols_kernel,
        grid=(t // MX_BLOCK, k // COL_TILE),
        in_specs=[pl.BlockSpec((MX_BLOCK, COL_TILE), lambda i, j: (i, j))],
        out_specs=[
            pl.BlockSpec((MX_BLOCK, COL_TILE), lambda i, j: (i, j)),
            pl.BlockSpec((MX_BLOCK, COL_TILE // MX_BLOCK), lambda i, j: (i, j)),
            pl.BlockSpec((MX_BLOCK, COL_TILE), lambda i, j: (i, j)),
            pl.BlockSpec((1, COL_TILE), lambda i, j: (i, j)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((t, k), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((t, k // MX_BLOCK), jnp.uint8),
            jax.ShapeDtypeStruct((t, k), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((t // MX_BLOCK, k), jnp.uint8),
        ],
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
        interpret=interpret,
        name="mx_quantize_rows_cols",
    )(x)
    return (row_values, row_exponents), (col_values, col_exponents)


def quantize_rows_cols(x: jax.Array):
    """``(quantize_rows(x), quantize_cols(x))``, in one pass over ``x`` when the fused kernel fits."""
    t, k = x.shape
    if t % MX_BLOCK or k % COL_TILE or x.size > _MAX_CALL_ELEMENTS:
        return quantize_rows(x), quantize_cols(x)
    return mx_quantize_rows_cols_pallas(x)


def _row_blocks_kernel(starts_ref, counts_ref, *refs, rows: int, dequantize: bool, with_rows: bool):
    """One row block x one ``COL_TILE`` column slice of ``mx_quantize_row_blocks``. ``dequantize``: the input is e4m3
    values plus per-row exponents ``[T, K / 32]`` (an MXFP8-along-rows tensor), rescaled here in f32. ``with_rows``:
    also write the block's rows quantized along the columns (``mx_quantize_rows``), re-reading the tile per 32
    columns from cache. Tuned on GB200 (301466 x 6144): 32 x 128 tiles with two
    warps run the token quantizer at ~4.4 TB/s; 32 x 32 tiles and 3-D reshapes for the row half are 2-3x slower."""
    if dequantize:
        x_ref, x_exponents_ref, *refs = refs
    else:
        x_ref, *refs = refs
    values_ref, exponents_ref, *row_refs = refs
    j = pl.program_id(0)
    c = pl.program_id(1)
    sub_blocks = COL_TILE // MX_BLOCK
    cols = pl.ds(c * COL_TILE, COL_TILE)
    start = starts_ref[j]
    count = counts_ref[j]
    # Read a full 32-row window that stays in bounds and mask the block's rows inside it: interpret mode clamps an
    # out-of-bounds dynamic slice instead of masking it.
    window = jnp.minimum(start, rows - MX_BLOCK)
    window_rows = pl.ds(window, MX_BLOCK)
    offset = window + jnp.arange(MX_BLOCK, dtype=jnp.int32)
    row_mask = (offset >= start) & (offset < start + count)
    mask = row_mask[:, None]
    block = plt.load(x_ref.at[window_rows, cols], mask=mask).astype(jnp.float32)  # [32, C]
    if dequantize:
        x_exponents = plt.load(x_exponents_ref.at[window_rows, pl.ds(c * sub_blocks, sub_blocks)], mask=mask)
        x_scale = jax.lax.bitcast_convert_type(x_exponents.astype(jnp.int32) << 23, jnp.float32)  # [32, C / 32]
        block = block * jnp.broadcast_to(x_scale[:, :, None], (MX_BLOCK, sub_blocks, MX_BLOCK)).reshape(block.shape)
    block = jnp.where(mask, block, 0.0)
    exponent = _exponent(jnp.max(jnp.abs(block), axis=0))  # [C]
    scale = jax.lax.bitcast_convert_type(exponent << 23, jnp.float32)
    plt.store(values_ref.at[window_rows, cols], (block / scale[None, :]).astype(values_ref.dtype), mask=mask)
    exponents_ref[0, :] = exponent.astype(exponents_ref.dtype)
    if not with_rows:
        return
    row_values_ref, row_exponents_ref = row_refs
    for sub in range(sub_blocks):
        sub_cols = pl.ds(c * COL_TILE + sub * MX_BLOCK, MX_BLOCK)
        part = jnp.where(mask, plt.load(x_ref.at[window_rows, sub_cols], mask=mask).astype(jnp.float32), 0.0)
        row_exponent = _exponent(jnp.max(jnp.abs(part), axis=1))  # [32]
        row_scale = jax.lax.bitcast_convert_type(row_exponent << 23, jnp.float32)
        row_values = (part / row_scale[:, None]).astype(row_values_ref.dtype)
        plt.store(row_values_ref.at[window_rows, sub_cols], row_values, mask=mask)
        plt.store(
            row_exponents_ref.at[window_rows, c * sub_blocks + sub],
            row_exponent.astype(row_exponents_ref.dtype),
            mask=row_mask,
        )


@functools.partial(jax.jit, static_argnames=("with_rows", "interpret"))
def _row_blocks_call(x, x_exponents, block_starts, block_rows, *, with_rows: bool, interpret: bool):
    t, k = x.shape
    num_blocks = block_starts.shape[0]
    if t < MX_BLOCK or k % COL_TILE:
        raise ValueError(f"the row-block quantizer needs T >= {MX_BLOCK} and K % {COL_TILE} == 0, got {x.shape}")
    sources = (x,) if x_exponents is None else (x, x_exponents)
    out_shape = [
        jax.ShapeDtypeStruct((t, k), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((num_blocks, k), jnp.uint8),
    ]
    out_specs = [pl.no_block_spec, pl.BlockSpec((1, COL_TILE), lambda j, c: (j, c))]
    if with_rows:
        out_shape += [
            jax.ShapeDtypeStruct((t, k), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((t, k // MX_BLOCK), jnp.uint8),
        ]
        out_specs += [pl.no_block_spec, pl.no_block_spec]
    return pl.pallas_call(
        functools.partial(_row_blocks_kernel, rows=t, dequantize=x_exponents is not None, with_rows=with_rows),
        grid=(num_blocks, k // COL_TILE),
        in_specs=[pl.no_block_spec] * (2 + len(sources)),
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=plt.CompilerParams(num_warps=2, num_stages=1),
        interpret=interpret,
        name="mx_quantize_row_blocks",
    )(block_starts.astype(jnp.int32), block_rows.astype(jnp.int32), *sources)


def mx_quantize_row_blocks_pallas(
    x: jax.Array, block_starts: jax.Array, block_rows: jax.Array, *, interpret: bool = False
) -> tuple[jax.Array, jax.Array]:
    """``mx_quantize_row_blocks`` in one pass: each program quantizes one row block of one ``COL_TILE`` column
    slice. ``x`` and the values are addressed whole, so rows in no block are never written."""
    values, exponents = _row_blocks_call(x, None, block_starts, block_rows, with_rows=False, interpret=interpret)
    return values, exponents


def mx_requantize_row_blocks_pallas(
    values: jax.Array, exponents: jax.Array, block_starts: jax.Array, block_rows: jax.Array, *, interpret=False
) -> tuple[jax.Array, jax.Array]:
    """``mx_quantize_row_blocks`` of ``mx_dequantize_rows(values, exponents)`` without materializing it."""
    out = _row_blocks_call(values, exponents, block_starts, block_rows, with_rows=False, interpret=interpret)
    return out[0], out[1]


def mx_quantize_rows_and_row_blocks_pallas(
    x: jax.Array, block_starts: jax.Array, block_rows: jax.Array, *, interpret: bool = False
):
    """``(mx_quantize_rows(x), mx_quantize_row_blocks(x, ...))`` reading ``x`` once; rows in no block get
    unspecified row values and exponents."""
    values, exponents, row_values, row_exponents = _row_blocks_call(
        x, None, block_starts, block_rows, with_rows=True, interpret=interpret
    )
    return (row_values, row_exponents), (values, exponents)


def _row_blocks_fit(x: jax.Array) -> bool:
    t, k = x.shape
    return t >= MX_BLOCK and k % COL_TILE == 0 and x.size <= _MAX_CALL_ELEMENTS


def quantize_row_blocks(x: jax.Array, block_starts: jax.Array, block_rows: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``mx_quantize_row_blocks`` of a local 2-D ``x``: the single-pass kernel when it fits, else XLA's."""
    if not _row_blocks_fit(x):
        return mx_quantize_row_blocks(x, block_starts, block_rows)
    return mx_quantize_row_blocks_pallas(x, block_starts, block_rows)


def requantize_row_blocks(values, exponents, block_starts, block_rows) -> tuple[jax.Array, jax.Array]:
    """``quantize_row_blocks`` of an MXFP8-along-rows tensor (e4m3 ``values``, per-row ``exponents``)."""
    if not _row_blocks_fit(values):
        return mx_quantize_row_blocks(mx_dequantize_rows(values, exponents, jnp.float32), block_starts, block_rows)
    return mx_requantize_row_blocks_pallas(values, exponents, block_starts, block_rows)


def quantize_rows_and_row_blocks(x: jax.Array, block_starts: jax.Array, block_rows: jax.Array):
    """``(quantize_rows(x), quantize_row_blocks(x, ...))``, in one pass over ``x`` when the fused kernel fits."""
    if not _row_blocks_fit(x) or x.shape[0] % ROW_TILE:
        return quantize_rows(x), mx_quantize_row_blocks(x, block_starts, block_rows)
    return mx_quantize_rows_and_row_blocks_pallas(x, block_starts, block_rows)
