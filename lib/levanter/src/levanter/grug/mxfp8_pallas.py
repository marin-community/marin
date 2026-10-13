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
