# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Numerics and determinism of the fused MXFP8 expert kernels (SM100).

Each kernel leg is checked against per-expert f32 matmuls on the dequantized operands it actually
consumes, so the comparison isolates the kernel from quantization noise. Quantized outputs are
checked after dequantizing both the kernel's bytes and scales and an independent reference
quantization of the f32 reference result.
"""

import functools
import importlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug._moe.mxfp8.layout import GROUP_PAD

_SF_VEC = 32
_E4M3_MAX = 448.0
# The fused epilogues scale by amax times 1/448 rounded to f32; multiplying (not dividing) matches them.
_RCP_E4M3_MAX = np.float32(1.0 / 448.0)
_GATE_UP_BLOCK = 32
_HIGHEST = jax.lax.Precision.HIGHEST
# experiments/grug/moe_hero_ep/train.py: XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG, as the hero compiles.
_jit_without_command_buffers = functools.partial(jax.jit, compiler_options={"xla_gpu_enable_command_buffer": ""})

# Research gates (c3cb334f8e:experiments/grug/moe/standalone/test_mxfp8_op_gpu.py).
_GEMM_TOLERANCE = 1e-3
_QUANTIZED_TOLERANCE = 2e-3
_DPROB_TOLERANCE = 2e-2


class _Problem(NamedTuple):
    hidden: int
    intermediate: int
    # Token count per expert before padding; zero marks an expert that receives no tokens.
    group_sizes: tuple[int, ...]


# One expert gets no tokens; the intermediate sizes give 2 and 3 dSwiGLU column tiles.
_SMALL_PROBLEMS = [
    _Problem(hidden=256, intermediate=512, group_sizes=(300, 0, 1000, 213)),
    _Problem(hidden=512, intermediate=768, group_sizes=(700, 512, 0)),
]
# One hero expert chunk: H = I = 3072, 3 experts, 302,336 padded rows.
_HERO_PROBLEM = _Problem(hidden=3072, intermediate=3072, group_sizes=(140_000, 100_000, 62_000))
_PROBLEMS = [
    *(pytest.param(problem, id=f"h{problem.hidden}-i{problem.intermediate}") for problem in _SMALL_PROBLEMS),
    pytest.param(_HERO_PROBLEM, id="hero-chunk", marks=pytest.mark.slow),
]
# The hero chunk's 303,104 padded rows with skewed routing and an expert that gets no tokens.
_SKEWED_HERO_PROBLEM = _Problem(hidden=3072, intermediate=3072, group_sizes=(182_000, 0, 121_000))
_FORWARD_ONLY_PROBLEMS = [
    *(pytest.param(problem, id=f"h{problem.hidden}-i{problem.intermediate}") for problem in _SMALL_PROBLEMS),
    pytest.param(_SKEWED_HERO_PROBLEM, id="hero-chunk-skewed", marks=pytest.mark.slow),
]


def _require_sm100():
    if jax.default_backend() != "gpu":
        pytest.skip("fused MXFP8 kernels require an SM100 GPU")
    if int(float(jax.devices("gpu")[0].compute_capability)) != 10:
        pytest.skip("fused MXFP8 kernels require SM100")
    pytest.importorskip("cutlass")
    pytest.importorskip("quack")
    return importlib.import_module("levanter.grug._moe.mxfp8.adapter")


def _padded_sizes(problem: _Problem) -> tuple[int, ...]:
    return tuple(-(-size // GROUP_PAD) * GROUP_PAD for size in problem.group_sizes)


# --------------------------------------------------------------------------- #
# reference MXFP8 quantization and SM100 scale-factor layouts
# --------------------------------------------------------------------------- #


def _e8m0_round_up(x):
    """Exponent byte of the smallest power of two >= x, for finite x >= 0 (byte 0 is 2^-127)."""
    bits = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    exponent = bits >> 23
    mantissa = bits & 0x7FFFFF
    round_up = (mantissa > 0) & (exponent != 0xFE) & ~((exponent == 0) & (mantissa <= 0x400000))
    return jnp.where(round_up, exponent + 1, exponent).astype(jnp.uint8)


def _e8m0_decode(scale):
    exponent = scale.astype(jnp.uint32)
    bits = jnp.where(exponent == 0, jnp.uint32(0x00400000), exponent << 23)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def _quantize(x, axis):
    """MXFP8 along ``axis`` in blocks of 32: (float8_e4m3fn values, uint8 e8m0 scales)."""
    moved = jnp.moveaxis(x.astype(jnp.float32), axis, -1)
    blocks = moved.reshape(*moved.shape[:-1], moved.shape[-1] // _SF_VEC, _SF_VEC)
    scale = _e8m0_round_up(jnp.max(jnp.abs(blocks), axis=-1) * _RCP_E4M3_MAX)
    q = jnp.clip(blocks / _e8m0_decode(scale)[..., None], -_E4M3_MAX, _E4M3_MAX).astype(jnp.float8_e4m3fn)
    return jnp.moveaxis(q.reshape(moved.shape), -1, axis), jnp.moveaxis(scale, -1, axis)


def _dequantize(q, scale, axis):
    moved = jnp.moveaxis(q.astype(jnp.float32), axis, -1)
    blocks = moved.reshape(*moved.shape[:-1], moved.shape[-1] // _SF_VEC, _SF_VEC)
    values = blocks * _e8m0_decode(jnp.moveaxis(scale, axis, -1))[..., None]
    return jnp.moveaxis(values.reshape(moved.shape), -1, axis)


def _swizzle(scale):
    """(rows, cols) uint8 scales -> flat bytes in the SM100 32x4x4 atom order (128x4 tiles, row-major)."""
    rows, cols = scale.shape
    assert rows % 128 == 0 and cols % 4 == 0, scale.shape
    tiles = scale.reshape(rows // 128, 128, cols // 4, 4).transpose(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).swapaxes(1, 2).reshape(-1)


def _unswizzle(flat, rows, cols):
    tiles = jax.lax.bitcast_convert_type(flat.reshape(-1), jnp.uint8)
    tiles = tiles.reshape(-1, 32, 4, 4).swapaxes(1, 2).reshape(rows // 128, cols // 4, 128, 4)
    return tiles.transpose(0, 2, 1, 3).reshape(rows, cols)


def _as_e8m0(raw):
    return jax.lax.bitcast_convert_type(raw, jnp.float8_e8m0fnu)


def _column_scales(scale, padded_sizes):
    """(T/32, F) token-axis scales -> per-expert swizzled (F, T/32 slice) chunks, as an (F, T/32) array."""
    by_feature = scale.T
    chunks, start = [], 0
    for size in padded_sizes:
        chunks.append(_swizzle(by_feature[:, start : start + size // _SF_VEC]))
        start += size // _SF_VEC
    return _as_e8m0(jnp.concatenate(chunks).reshape(by_feature.shape))


def _unswizzle_columns(flat, features, padded_sizes):
    """Inverse of :func:`_column_scales`: kernel column-orientation scales -> (T/32, F) uint8."""
    flat = flat.reshape(-1)
    columns, start = [], 0
    for size in padded_sizes:
        count = features * size // _SF_VEC
        columns.append(_unswizzle(flat[start : start + count], features, size // _SF_VEC))
        start += count
    return jnp.concatenate(columns, axis=1).T


def _weight_scales(scale):
    """(E, N, K/32) weight scales -> one swizzled block per expert."""
    return _as_e8m0(jnp.stack([_swizzle(scale[e]) for e in range(scale.shape[0])]))


# --------------------------------------------------------------------------- #
# operands, kernel legs, and the f32 reference
# --------------------------------------------------------------------------- #


class _Inputs(NamedTuple):
    x: jax.Array  # (M, H) bf16, padding rows zero
    g: jax.Array  # (M, H) bf16 output cotangent, padding rows zero
    w13i: jax.Array  # (E, 2I, H) bf16, gate/up interleaved in 32-column blocks
    w2: jax.Array  # (E, I, H) bf16


class _KernelOperands(NamedTuple):
    x: jax.Array
    x_sf: jax.Array
    x_col: jax.Array
    x_col_sf: jax.Array
    g: jax.Array
    g_sf: jax.Array
    g_col: jax.Array
    g_col_sf: jax.Array
    w13i: jax.Array
    w13i_sf: jax.Array
    w2f: jax.Array
    w2f_sf: jax.Array
    w2: jax.Array
    w2_sf: jax.Array
    w13dg: jax.Array
    w13dg_sf: jax.Array


def _make_inputs(problem: _Problem, seed: int) -> _Inputs:
    padded = _padded_sizes(problem)
    valid = np.concatenate([np.arange(p) < s for s, p in zip(problem.group_sizes, padded, strict=True)])
    valid = jnp.asarray(valid)[:, None]
    experts, hidden, intermediate = len(padded), problem.hidden, problem.intermediate
    kx, kg, k13, k2 = jax.random.split(jax.random.key(seed), 4)
    rows = sum(padded)
    x = jnp.where(valid, jax.random.normal(kx, (rows, hidden), jnp.bfloat16), 0)
    g = jnp.where(valid, jax.random.normal(kg, (rows, hidden), jnp.bfloat16), 0)
    w13i = (jax.random.normal(k13, (experts, 2 * intermediate, hidden), jnp.float32) / hidden**0.5).astype(
        jnp.bfloat16
    )
    w2 = (jax.random.normal(k2, (experts, intermediate, hidden), jnp.float32) / intermediate**0.5).astype(jnp.bfloat16)
    return _Inputs(x=x, g=g, w13i=w13i, w2=w2)


def _quantize_operands(inputs: _Inputs, padded_sizes: tuple[int, ...]) -> _KernelOperands:
    def rows(t):
        q, scale = _quantize(t, axis=1)
        return q, _as_e8m0(_swizzle(scale))

    def columns(t):
        q, scale = _quantize(t, axis=0)
        return q, _column_scales(scale, padded_sizes)

    def weight(w):
        q, scale = _quantize(w, axis=2)
        return q, _weight_scales(scale)

    return _KernelOperands(
        *rows(inputs.x),
        *columns(inputs.x),
        *rows(inputs.g),
        *columns(inputs.g),
        *weight(inputs.w13i),
        *weight(jnp.swapaxes(inputs.w2, 1, 2)),
        *weight(inputs.w2),
        *weight(jnp.swapaxes(inputs.w13i, 1, 2)),
    )


class _Outputs(NamedTuple):
    c13: jax.Array
    h: jax.Array
    h_col: jax.Array
    sfh_row: jax.Array
    sfh_col: jax.Array
    y: jax.Array
    dc: jax.Array
    dc_col: jax.Array
    sfdc_row: jax.Array
    sfdc_col: jax.Array
    dprob: jax.Array
    dx: jax.Array
    dw13i: jax.Array
    dw2: jax.Array


def _run_legs(adapter, padded_sizes: tuple[int, ...], ops: _KernelOperands) -> _Outputs:
    offs = jnp.asarray(np.cumsum(padded_sizes), jnp.int32)
    n2 = ops.w13i.shape[1]
    c13, h, h_col, sfh_row, sfh_col = adapter.mxfp8_fused_swiglu(ops.x, ops.x_sf, ops.w13i, ops.w13i_sf, offs)
    y = adapter.mxfp8_fused_gemm(h, sfh_row, ops.w2f, ops.w2f_sf, offs)
    dc, dc_col, sfdc_row, sfdc_col, dprob = adapter.mxfp8_fused_dswiglu(ops.g, ops.g_sf, ops.w2, ops.w2_sf, c13, offs)
    dx = adapter.mxfp8_fused_gemm(dc, sfdc_row, ops.w13dg, ops.w13dg_sf, offs)
    dw13i = adapter.mxfp8_fused_wgrad(ops.x_col, ops.x_col_sf, dc_col, sfdc_col.reshape(n2, -1), offs)
    dw2 = adapter.mxfp8_fused_wgrad(h_col, sfh_col.reshape(n2 // 2, -1), ops.g_col, ops.g_col_sf, offs)
    return _Outputs(c13, h, h_col, sfh_row, sfh_col, y, dc, dc_col, sfdc_row, sfdc_col, dprob, dx, dw13i, dw2)


def _gate_up(c):
    blocks = c.reshape(*c.shape[:-1], -1, 2, _GATE_UP_BLOCK)
    return blocks[..., 0, :].reshape(*c.shape[:-1], -1), blocks[..., 1, :].reshape(*c.shape[:-1], -1)


def _swiglu(c):
    gate, up = _gate_up(c)
    return jax.nn.silu(gate) * up


def _grouped_matmul(a, b, padded_sizes):
    """Per expert ``a[rows_e] @ b[e].T`` in f32: a (M, K), b (E, N, K) -> (M, N)."""
    outputs, start = [], 0
    for expert, size in enumerate(padded_sizes):
        outputs.append(jnp.matmul(a[start : start + size], b[expert].T, precision=_HIGHEST))
        start += size
    return jnp.concatenate(outputs)


def _grouped_wgrad(a, b, padded_sizes):
    """Per expert ``a[rows_e].T @ b[rows_e]`` in f32: a (M, K), b (M, N) -> (E, K, N)."""
    outputs, start = [], 0
    for size in padded_sizes:
        outputs.append(jnp.matmul(a[start : start + size].T, b[start : start + size], precision=_HIGHEST))
        start += size
    return jnp.stack(outputs)


def _relative_error(actual, expected):
    actual = actual.astype(jnp.float32)
    expected = expected.astype(jnp.float32)
    return jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)


def _dequantize_rows(q, sf):
    rows, features = q.shape
    return _dequantize(q, _unswizzle(sf, rows, features // _SF_VEC), axis=1)


def _dequantize_columns(q, sf, padded_sizes):
    return _dequantize(q, _unswizzle_columns(sf, q.shape[1], padded_sizes), axis=0)


def _dequantize_weight(q, sf):
    experts, n, k = q.shape
    scale = jnp.stack([_unswizzle(sf[e], n, k // _SF_VEC) for e in range(experts)])
    return _dequantize(q, scale, axis=2)


# The reference is built in three separately jitted stages. Within one XLA computation the
# excess-precision rewrite may drop an f32 -> fp8/bf16 -> f32 round trip, which would compare the
# kernels against an unquantized or unrounded reference. Values crossing a jit boundary are
# materialized in their dtype, so every quantize or bf16 rounding happens in its own stage and
# every dequantization reads materialized bytes.


class _References(NamedTuple):
    """f32 reference of each kernel output, computed from the operands that kernel consumed."""

    c13: jax.Array
    h: jax.Array
    y: jax.Array
    dc: jax.Array
    dprob: jax.Array
    dx: jax.Array
    dw13i: jax.Array
    dw2: jax.Array


def _references(padded_sizes: tuple[int, ...], ops: _KernelOperands, out: _Outputs) -> _References:
    x = _dequantize_rows(ops.x, ops.x_sf)
    g = _dequantize_rows(ops.g, ops.g_sf)
    x_col = _dequantize_columns(ops.x_col, ops.x_col_sf, padded_sizes)
    g_col = _dequantize_columns(ops.g_col, ops.g_col_sf, padded_sizes)
    w13i = _dequantize_weight(ops.w13i, ops.w13i_sf)
    w2f = _dequantize_weight(ops.w2f, ops.w2f_sf)
    w2 = _dequantize_weight(ops.w2, ops.w2_sf)
    w13dg = _dequantize_weight(ops.w13dg, ops.w13dg_sf)
    # Downstream legs consume the kernels' own quantized intermediates.
    h = _dequantize_rows(out.h, out.sfh_row)
    h_col = _dequantize_columns(out.h_col, out.sfh_col, padded_sizes)
    dc = _dequantize_rows(out.dc, out.sfdc_row)
    dc_col = _dequantize_columns(out.dc_col, out.sfdc_col, padded_sizes)

    c13_ref = _grouped_matmul(x, w13i, padded_sizes)
    dh = _grouped_matmul(g, w2, padded_sizes)
    # The dSwiGLU epilogue reads the bf16 pre-activation the forward kernel stored.
    c13 = out.c13.astype(jnp.float32)
    _, swiglu_pullback = jax.vjp(_swiglu, c13)
    (dc_ref,) = swiglu_pullback(dh)
    return _References(
        c13=c13_ref,
        h=_swiglu(c13_ref),
        y=_grouped_matmul(h, w2f, padded_sizes),
        dc=dc_ref,
        dprob=jnp.sum(_swiglu(c13) * dh, axis=-1),
        dx=_grouped_matmul(dc, w13dg, padded_sizes),
        dw13i=_grouped_wgrad(x_col, dc_col, padded_sizes),
        dw2=_grouped_wgrad(h_col, g_col, padded_sizes),
    )


class _RoundedReferences(NamedTuple):
    """The references in the output's precision: bf16 for GEMM outputs, MXFP8 (values, scales) otherwise."""

    c13: jax.Array
    y: jax.Array
    dx: jax.Array
    dw13i: jax.Array
    dw2: jax.Array
    h_rows: tuple[jax.Array, jax.Array]
    h_cols: tuple[jax.Array, jax.Array]
    dc_rows: tuple[jax.Array, jax.Array]
    dc_cols: tuple[jax.Array, jax.Array]
    dprob: jax.Array


def _round_references(ref: _References) -> _RoundedReferences:
    return _RoundedReferences(
        c13=ref.c13.astype(jnp.bfloat16),
        y=ref.y.astype(jnp.bfloat16),
        dx=ref.dx.astype(jnp.bfloat16),
        dw13i=ref.dw13i.astype(jnp.bfloat16),
        dw2=ref.dw2.astype(jnp.bfloat16),
        h_rows=_quantize(ref.h, axis=1),
        h_cols=_quantize(ref.h, axis=0),
        dc_rows=_quantize(ref.dc, axis=1),
        dc_cols=_quantize(ref.dc, axis=0),
        dprob=ref.dprob,
    )


def _leg_errors(padded_sizes: tuple[int, ...], out: _Outputs, ref: _RoundedReferences) -> dict[str, jax.Array]:
    """Relative Frobenius error of every kernel output against its rounded reference."""
    return {
        "c13": _relative_error(out.c13, ref.c13),
        "y": _relative_error(out.y, ref.y),
        "dx": _relative_error(out.dx, ref.dx),
        "dw13i": _relative_error(out.dw13i, ref.dw13i),
        "dw2": _relative_error(out.dw2, ref.dw2),
        "h": _relative_error(_dequantize_rows(out.h, out.sfh_row), _dequantize(*ref.h_rows, axis=1)),
        "h_col": _relative_error(
            _dequantize_columns(out.h_col, out.sfh_col, padded_sizes), _dequantize(*ref.h_cols, axis=0)
        ),
        "dc": _relative_error(_dequantize_rows(out.dc, out.sfdc_row), _dequantize(*ref.dc_rows, axis=1)),
        "dc_col": _relative_error(
            _dequantize_columns(out.dc_col, out.sfdc_col, padded_sizes), _dequantize(*ref.dc_cols, axis=0)
        ),
        "dprob": _relative_error(out.dprob, ref.dprob),
    }


_TOLERANCES = {
    "c13": _GEMM_TOLERANCE,
    "y": _GEMM_TOLERANCE,
    "dx": _GEMM_TOLERANCE,
    "h": _QUANTIZED_TOLERANCE,
    "h_col": _QUANTIZED_TOLERANCE,
    "dc": _QUANTIZED_TOLERANCE,
    "dc_col": _QUANTIZED_TOLERANCE,
    "dw13i": _QUANTIZED_TOLERANCE,
    "dw2": _QUANTIZED_TOLERANCE,
    "dprob": _DPROB_TOLERANCE,
}


def _bits(a):
    unsigned = {1: jnp.uint8, 2: jnp.uint16, 4: jnp.uint32}[a.dtype.itemsize]
    return jax.lax.bitcast_convert_type(a, unsigned)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _PROBLEMS)
def test_fused_mxfp8_legs_match_dequantized_f32_reference(problem):
    adapter = _require_sm100()
    padded = _padded_sizes(problem)
    operands = jax.jit(functools.partial(_quantize_operands, padded_sizes=padded))(_make_inputs(problem, seed=0))
    out = jax.jit(functools.partial(_run_legs, adapter, padded))(operands)
    references = jax.jit(functools.partial(_references, padded))(operands, out)
    rounded = jax.jit(_round_references)(references)
    del references
    errors = jax.jit(functools.partial(_leg_errors, padded))(out, rounded)
    errors = {name: float(value) for name, value in errors.items()}
    print(f"{problem}: " + ", ".join(f"{name}={value:.3e}" for name, value in errors.items()))

    failures = {name: value for name, value in errors.items() if not value < _TOLERANCES[name]}
    assert not failures, f"legs over tolerance: {failures} (tolerances {_TOLERANCES})"

    for expert, size in enumerate(problem.group_sizes):
        if size == 0:
            assert not np.any(np.asarray(out.dw13i[expert])), f"expert {expert} has no tokens but a nonzero dw13"
            assert not np.any(np.asarray(out.dw2[expert])), f"expert {expert} has no tokens but a nonzero dw2"


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _PROBLEMS)
def test_fused_mxfp8_legs_are_bitwise_identical_across_reruns(problem):
    adapter = _require_sm100()
    padded = _padded_sizes(problem)
    operands = jax.jit(functools.partial(_quantize_operands, padded_sizes=padded))(_make_inputs(problem, seed=1))
    run = jax.jit(functools.partial(_run_legs, adapter, padded))
    first = run(operands)
    for rerun in range(2):
        again = run(operands)
        identical = jax.tree.map(lambda a, b: bool(jnp.array_equal(_bits(a), _bits(b))), first, again)
        differing = [name for name, same in identical._asdict().items() if not same]
        assert not differing, f"rerun {rerun + 1} changed {differing}"


def _swiglu_operands(inputs: _Inputs) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The gate/up GEMM's operands: row-quantized tokens and weights, with their swizzled scales."""
    x, x_scale = _quantize(inputs.x, axis=1)
    w13i, w13i_scale = _quantize(inputs.w13i, axis=2)
    return x, _as_e8m0(_swizzle(x_scale)), w13i, _weight_scales(w13i_scale)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _FORWARD_ONLY_PROBLEMS)
def test_forward_only_swiglu_writes_the_full_kernels_h_and_row_scales_bitwise(problem):
    adapter = _require_sm100()
    padded = _padded_sizes(problem)
    operands = jax.jit(_swiglu_operands)(_make_inputs(problem, seed=2))
    offs = jnp.asarray(np.cumsum(padded), jnp.int32)

    _c13, h, _h_col, sfh_row, _sfh_col = _jit_without_command_buffers(adapter.mxfp8_fused_swiglu)(*operands, offs)
    h_forward, sfh_row_forward = _jit_without_command_buffers(adapter.mxfp8_fused_swiglu_forward)(*operands, offs)

    assert bool(jnp.array_equal(_bits(h_forward), _bits(h))), "h differs from the full kernel's"
    assert bool(jnp.array_equal(_bits(sfh_row_forward), _bits(sfh_row))), "sfh_row differs from the full kernel's"
