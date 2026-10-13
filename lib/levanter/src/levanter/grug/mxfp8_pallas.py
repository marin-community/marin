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

from levanter.grug.mxfp8 import MX_BLOCK, mx_quantize_cols, mx_quantize_rows

_E4M3_MAX = 448.0
ROW_TILE = 64
"""Rows per program of the row quantizer; each program covers ``MX_BLOCK * COL_BLOCKS`` columns."""
COL_BLOCKS = 4


def _exponent(amax):
    bits = jax.lax.bitcast_convert_type(amax / _E4M3_MAX, jnp.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).astype(jnp.int32)
    return jnp.where(amax > 0, jnp.clip(exponent, 1, 254), 127)


def _rows_kernel(x_ref, values_ref, exponents_ref):
    for j in range(COL_BLOCKS):
        block = x_ref[:, j * MX_BLOCK : (j + 1) * MX_BLOCK].astype(jnp.float32)  # [R, 32]
        exponent = _exponent(jnp.max(jnp.abs(block), axis=1))  # [R]
        scale = jax.lax.bitcast_convert_type(exponent << 23, jnp.float32)
        values_ref[:, j * MX_BLOCK : (j + 1) * MX_BLOCK] = (block / scale[:, None]).astype(values_ref.dtype)
        exponents_ref[:, j] = exponent.astype(exponents_ref.dtype)


@functools.partial(jax.jit, static_argnames=("interpret",))
def mx_quantize_rows_pallas(x: jax.Array, *, interpret: bool = False) -> tuple[jax.Array, jax.Array]:
    """``mx_quantize_rows`` of a local ``x[M, K]`` in one pass: e4m3 ``[M, K]`` and uint8 exponents ``[M, K / 32]``."""
    m, k = x.shape
    cols = MX_BLOCK * COL_BLOCKS
    if m % ROW_TILE or k % cols:
        raise ValueError(f"mx_quantize_rows_pallas needs M % {ROW_TILE} == 0 and K % {cols} == 0, got {x.shape}")
    return pl.pallas_call(
        _rows_kernel,
        grid=(m // ROW_TILE, k // cols),
        in_specs=[pl.BlockSpec((ROW_TILE, cols), lambda i, j: (i, j))],
        out_specs=[
            pl.BlockSpec((ROW_TILE, cols), lambda i, j: (i, j)),
            pl.BlockSpec((ROW_TILE, COL_BLOCKS), lambda i, j: (i, j)),
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
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
        interpret=interpret,
        name="mx_quantize_cols",
    )(x)


# Triton computes block offsets in int32: a call must stay under 2^31 elements.
_MAX_CALL_ELEMENTS = 2**31 - 1


def _row_chunks(rows: int, cols: int, row_multiple: int) -> list[tuple[int, int]]:
    per_chunk = max(row_multiple, _MAX_CALL_ELEMENTS // cols // row_multiple * row_multiple)
    return [(start, min(start + per_chunk, rows)) for start in range(0, rows, per_chunk)]


def quantize_rows(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``mx_quantize_rows`` of a local 2-D ``x``: the single-pass kernel when its tiling fits, else XLA's."""
    m, k = x.shape
    if m % ROW_TILE or k % (MX_BLOCK * COL_BLOCKS):
        return mx_quantize_rows(x)
    parts = [mx_quantize_rows_pallas(x[a:b]) for a, b in _row_chunks(m, k, ROW_TILE)]
    if len(parts) == 1:
        return parts[0]
    return jnp.concatenate([p[0] for p in parts]), jnp.concatenate([p[1] for p in parts])


def quantize_cols(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``mx_quantize_cols`` of a local 2-D ``x``: the single-pass kernel when its tiling fits, else XLA's."""
    t, k = x.shape
    if t % MX_BLOCK or k % COL_TILE:
        return mx_quantize_cols(x)
    parts = [mx_quantize_cols_pallas(x[a:b]) for a, b in _row_chunks(t, k, MX_BLOCK)]
    if len(parts) == 1:
        return parts[0]
    return jnp.concatenate([p[0] for p in parts]), jnp.concatenate([p[1] for p in parts])
