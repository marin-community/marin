# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Numerics and determinism of the cudnn-frontend fused MXFP8 ReLU^2 expert kernels (SM100).

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
from levanter.grug._moe.mxfp8.quantize import WeightMajor

_SF_VEC = 32
_E4M3_MAX = 448.0
# The fused epilogues scale by amax times 1/448 rounded to f32; multiplying (not dividing) matches them.
_RCP_E4M3_MAX = np.float32(1.0 / 448.0)
_HIGHEST = jax.lax.Precision.HIGHEST

# Research gates (c3cb334f8e:experiments/grug/moe/standalone/test_mxfp8_op_gpu.py).
_GEMM_TOLERANCE = 1e-3
_QUANTIZED_TOLERANCE = 2e-3
_DPROB_TOLERANCE = 2e-2


class _Problem(NamedTuple):
    hidden: int
    intermediate: int
    # Token count per expert before padding; zero marks an expert that receives no tokens.
    group_sizes: tuple[int, ...]


_SMALL_PROBLEMS = [
    # One expert gets no tokens; two ReLU^2 column tiles, so two row-dot slots.
    _Problem(hidden=256, intermediate=512, group_sizes=(300, 0, 1000, 213)),
    # The last expert gets no tokens; H and I end inside a 256-wide N tile of every GEMM.
    _Problem(hidden=384, intermediate=640, group_sizes=(700, 512, 0)),
]
# One expert chunk at the Belay hero's scale: H = 3072, I = 4608, 3 experts.
_HERO_PROBLEM = _Problem(hidden=3072, intermediate=4608, group_sizes=(150_000, 110_000, 84_000))
_PROBLEMS = [
    *(pytest.param(problem, id=f"h{problem.hidden}-i{problem.intermediate}") for problem in _SMALL_PROBLEMS),
    pytest.param(_HERO_PROBLEM, id="hero-chunk", marks=pytest.mark.slow),
]


def _require_sm100():
    if jax.default_backend() != "gpu":
        pytest.skip("fused MXFP8 kernels require an SM100 GPU")
    if int(float(jax.devices("gpu")[0].compute_capability)) != 10:
        pytest.skip("fused MXFP8 kernels require SM100")
    pytest.importorskip("cutlass")
    pytest.importorskip("cudnn")
    pytest.importorskip("quack")
    return (
        importlib.import_module("levanter.grug._moe.mxfp8.adapter"),
        importlib.import_module("levanter.grug._moe.mxfp8.adapter_srelu"),
    )


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


def _k_major_weight(w):
    """An (E, N, K) weight quantized along K, with its scales."""
    q, scale = _quantize(w, axis=2)
    return q, _weight_scales(scale)


def _n_major_weight(w):
    """An (E, K, N) weight quantized along K in place, with its scales over (N, K/32)."""
    q, scale = _quantize(w, axis=1)
    return q, _weight_scales(jnp.swapaxes(scale, 1, 2))


# --------------------------------------------------------------------------- #
# operands, kernel legs, and the f32 reference
# --------------------------------------------------------------------------- #


class _Inputs(NamedTuple):
    x: jax.Array  # (M, H) bf16, padding rows zero
    g: jax.Array  # (M, H) bf16 output cotangent, padding rows zero
    w_up: jax.Array  # (E, H, I) bf16
    w_down: jax.Array  # (E, I, H) bf16


class _KernelOperands(NamedTuple):
    x: jax.Array
    x_sf: jax.Array
    x_col: jax.Array
    x_col_sf: jax.Array
    g: jax.Array
    g_sf: jax.Array
    g_col: jax.Array
    g_col_sf: jax.Array
    # Every weight in its stored layout: w_up (E, H, I) and w_down (E, I, H).
    w_up_n: jax.Array  # quantized along H, read N-major by the up GEMM
    w_up_n_sf: jax.Array
    w_down_n: jax.Array  # quantized along I, read N-major by the down GEMM
    w_down_n_sf: jax.Array
    w_down_k: jax.Array  # quantized along H, read K-major by the down data gradient
    w_down_k_sf: jax.Array
    w_up_k: jax.Array  # quantized along I, read K-major by the up data gradient
    w_up_k_sf: jax.Array


def _make_inputs(problem: _Problem, seed: int) -> _Inputs:
    padded = _padded_sizes(problem)
    valid = np.concatenate([np.arange(p) < s for s, p in zip(problem.group_sizes, padded, strict=True)])
    valid = jnp.asarray(valid)[:, None]
    experts, hidden, intermediate = len(padded), problem.hidden, problem.intermediate
    kx, kg, kup, kdown = jax.random.split(jax.random.key(seed), 4)
    rows = sum(padded)
    x = jnp.where(valid, jax.random.normal(kx, (rows, hidden), jnp.bfloat16), 0)
    g = jnp.where(valid, jax.random.normal(kg, (rows, hidden), jnp.bfloat16), 0)
    w_up = (jax.random.normal(kup, (experts, hidden, intermediate), jnp.float32) / hidden**0.5).astype(jnp.bfloat16)
    w_down = (jax.random.normal(kdown, (experts, intermediate, hidden), jnp.float32) / intermediate**0.5).astype(
        jnp.bfloat16
    )
    return _Inputs(x=x, g=g, w_up=w_up, w_down=w_down)


def _quantize_operands(inputs: _Inputs, padded_sizes: tuple[int, ...]) -> _KernelOperands:
    def rows(t):
        q, scale = _quantize(t, axis=1)
        return q, _as_e8m0(_swizzle(scale))

    def columns(t):
        q, scale = _quantize(t, axis=0)
        return q, _column_scales(scale, padded_sizes)

    return _KernelOperands(
        *rows(inputs.x),
        *columns(inputs.x),
        *rows(inputs.g),
        *columns(inputs.g),
        *_n_major_weight(inputs.w_up),
        *_n_major_weight(inputs.w_down),
        *_k_major_weight(inputs.w_down),
        *_k_major_weight(inputs.w_up),
    )


class _Outputs(NamedTuple):
    c: jax.Array
    h: jax.Array
    sfh_row: jax.Array
    y: jax.Array
    dc: jax.Array
    sfdc_row: jax.Array
    dc_col: jax.Array
    sfdc_col: jax.Array
    h_col: jax.Array
    sfh_col: jax.Array
    dprob: jax.Array
    dx: jax.Array
    dw_up: jax.Array
    dw_down: jax.Array


def _run_legs(adapters, padded_sizes: tuple[int, ...], ops: _KernelOperands) -> _Outputs:
    adapter, adapter_srelu = adapters
    offs = jnp.asarray(np.cumsum(padded_sizes), jnp.int32)
    intermediate = ops.w_up_n.shape[2]
    c, h, sfh_row = adapter_srelu.mxfp8_fused_srelu(ops.x, ops.x_sf, ops.w_up_n, ops.w_up_n_sf, offs)
    y = adapter.mxfp8_fused_gemm(h, sfh_row, ops.w_down_n, ops.w_down_n_sf, offs, b_major=WeightMajor.N)
    dc, sfdc_row, dc_col, sfdc_col, h_col, sfh_col, dprob = adapter_srelu.mxfp8_fused_dsrelu(
        ops.g, ops.g_sf, ops.w_down_k, ops.w_down_k_sf, c, offs
    )
    dx = adapter.mxfp8_fused_gemm(dc, sfdc_row, ops.w_up_k, ops.w_up_k_sf, offs, b_major=WeightMajor.K)
    dw_up = adapter.mxfp8_fused_wgrad(ops.x_col, ops.x_col_sf, dc_col, sfdc_col.reshape(intermediate, -1), offs)
    dw_down = adapter.mxfp8_fused_wgrad(h_col, sfh_col.reshape(intermediate, -1), ops.g_col, ops.g_col_sf, offs)
    return _Outputs(c, h, sfh_row, y, dc, sfdc_row, dc_col, sfdc_col, h_col, sfh_col, dprob, dx, dw_up, dw_down)


def _grouped_matmul(a, b, padded_sizes):
    """Per expert ``a[rows_e] @ b[e]`` in f32: a (M, K), b (E, K, N) -> (M, N)."""
    outputs, start = [], 0
    for expert, size in enumerate(padded_sizes):
        outputs.append(jnp.matmul(a[start : start + size], b[expert], precision=_HIGHEST))
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


def _dequantize_k_major_weight(q, sf):
    experts, n, k = q.shape
    scale = jnp.stack([_unswizzle(sf[e], n, k // _SF_VEC) for e in range(experts)])
    return _dequantize(q, scale, axis=2)


def _dequantize_n_major_weight(q, sf):
    experts, k, n = q.shape
    scale = jnp.stack([_unswizzle(sf[e], n, k // _SF_VEC) for e in range(experts)])
    return _dequantize(q, jnp.swapaxes(scale, 1, 2), axis=1)


# The reference is built in three separately jitted stages. Within one XLA computation the
# excess-precision rewrite may drop an f32 -> fp8/bf16 -> f32 round trip, which would compare the
# kernels against an unquantized or unrounded reference. Values crossing a jit boundary are
# materialized in their dtype, so every quantize or bf16 rounding happens in its own stage and
# every dequantization reads materialized bytes.


class _References(NamedTuple):
    """f32 reference of each kernel output, computed from the operands that kernel consumed."""

    c: jax.Array
    h: jax.Array
    y: jax.Array
    dc: jax.Array
    h_regenerated: jax.Array
    dprob: jax.Array
    dx: jax.Array
    dw_up: jax.Array
    dw_down: jax.Array


def _references(padded_sizes: tuple[int, ...], ops: _KernelOperands, out: _Outputs) -> _References:
    x = _dequantize_rows(ops.x, ops.x_sf)
    g = _dequantize_rows(ops.g, ops.g_sf)
    x_col = _dequantize_columns(ops.x_col, ops.x_col_sf, padded_sizes)
    g_col = _dequantize_columns(ops.g_col, ops.g_col_sf, padded_sizes)
    w_up_n = _dequantize_n_major_weight(ops.w_up_n, ops.w_up_n_sf)  # (E, H, I)
    w_down_n = _dequantize_n_major_weight(ops.w_down_n, ops.w_down_n_sf)  # (E, I, H)
    w_down_k = _dequantize_k_major_weight(ops.w_down_k, ops.w_down_k_sf)  # (E, I, H)
    w_up_k = _dequantize_k_major_weight(ops.w_up_k, ops.w_up_k_sf)  # (E, H, I)
    # Downstream legs consume the kernels' own quantized intermediates.
    h = _dequantize_rows(out.h, out.sfh_row)
    dc = _dequantize_rows(out.dc, out.sfdc_row)
    dc_col = _dequantize_columns(out.dc_col, out.sfdc_col, padded_sizes)
    h_col = _dequantize_columns(out.h_col, out.sfh_col, padded_sizes)

    c_ref = _grouped_matmul(x, w_up_n, padded_sizes)
    dh = _grouped_matmul(g, jnp.swapaxes(w_down_k, 1, 2), padded_sizes)
    # The ReLU^2 backward epilogue reads the bf16 pre-activation the forward kernel stored.
    relu_c = jax.nn.relu(out.c.astype(jnp.float32))
    return _References(
        c=c_ref,
        h=jnp.square(jax.nn.relu(c_ref)),
        y=_grouped_matmul(h, w_down_n, padded_sizes),
        dc=2 * relu_c * dh,
        h_regenerated=jnp.square(relu_c),
        dprob=jnp.sum(jnp.square(relu_c) * dh, axis=-1),
        dx=_grouped_matmul(dc, jnp.swapaxes(w_up_k, 1, 2), padded_sizes),
        dw_up=_grouped_wgrad(x_col, dc_col, padded_sizes),
        dw_down=_grouped_wgrad(h_col, g_col, padded_sizes),
    )


class _RoundedReferences(NamedTuple):
    """The references in the output's precision: bf16 for GEMM outputs, MXFP8 (values, scales) otherwise."""

    c: jax.Array
    y: jax.Array
    dx: jax.Array
    dw_up: jax.Array
    dw_down: jax.Array
    h_rows: tuple[jax.Array, jax.Array]
    h_cols: tuple[jax.Array, jax.Array]
    dc_rows: tuple[jax.Array, jax.Array]
    dc_cols: tuple[jax.Array, jax.Array]
    dprob: jax.Array


def _round_references(ref: _References) -> _RoundedReferences:
    return _RoundedReferences(
        c=ref.c.astype(jnp.bfloat16),
        y=ref.y.astype(jnp.bfloat16),
        dx=ref.dx.astype(jnp.bfloat16),
        dw_up=ref.dw_up.astype(jnp.bfloat16),
        dw_down=ref.dw_down.astype(jnp.bfloat16),
        h_rows=_quantize(ref.h, axis=1),
        h_cols=_quantize(ref.h_regenerated, axis=0),
        dc_rows=_quantize(ref.dc, axis=1),
        dc_cols=_quantize(ref.dc, axis=0),
        dprob=ref.dprob,
    )


def _leg_errors(padded_sizes: tuple[int, ...], out: _Outputs, ref: _RoundedReferences) -> dict[str, jax.Array]:
    """Relative Frobenius error of every kernel output against its rounded reference."""
    return {
        "c": _relative_error(out.c, ref.c),
        "y": _relative_error(out.y, ref.y),
        "dx": _relative_error(out.dx, ref.dx),
        "dw_up": _relative_error(out.dw_up, ref.dw_up),
        "dw_down": _relative_error(out.dw_down, ref.dw_down),
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
    "c": _GEMM_TOLERANCE,
    "y": _GEMM_TOLERANCE,
    "dx": _GEMM_TOLERANCE,
    "h": _QUANTIZED_TOLERANCE,
    "h_col": _QUANTIZED_TOLERANCE,
    "dc": _QUANTIZED_TOLERANCE,
    "dc_col": _QUANTIZED_TOLERANCE,
    "dw_up": _QUANTIZED_TOLERANCE,
    "dw_down": _QUANTIZED_TOLERANCE,
    "dprob": _DPROB_TOLERANCE,
}


def _bits(a):
    unsigned = {1: jnp.uint8, 2: jnp.uint16, 4: jnp.uint32}[a.dtype.itemsize]
    return jax.lax.bitcast_convert_type(a, unsigned)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _PROBLEMS)
def test_fused_mxfp8_relu2_legs_match_dequantized_f32_reference(problem):
    adapters = _require_sm100()
    padded = _padded_sizes(problem)
    operands = jax.jit(functools.partial(_quantize_operands, padded_sizes=padded))(_make_inputs(problem, seed=0))
    out = jax.jit(functools.partial(_run_legs, adapters, padded))(operands)
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
            assert not np.any(np.asarray(out.dw_up[expert])), f"expert {expert} has no tokens but a nonzero dw_up"
            assert not np.any(np.asarray(out.dw_down[expert])), f"expert {expert} has no tokens but a nonzero dw_down"


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _PROBLEMS)
def test_fused_mxfp8_relu2_legs_are_bitwise_identical_across_reruns(problem):
    adapters = _require_sm100()
    padded = _padded_sizes(problem)
    operands = jax.jit(functools.partial(_quantize_operands, padded_sizes=padded))(_make_inputs(problem, seed=1))
    run = jax.jit(functools.partial(_run_legs, adapters, padded))
    first = run(operands)
    for rerun in range(2):
        again = run(operands)
        identical = jax.tree.map(lambda a, b: bool(jnp.array_equal(_bits(a), _bits(b))), first, again)
        differing = [name for name, same in identical._asdict().items() if not same]
        assert not differing, f"rerun {rerun + 1} changed {differing}"


def _gemm_operands_both_ways(inputs: _Inputs):
    """``x`` row-quantized, and ``w_up`` quantized along H both in place and as its (E, I, H) transpose."""
    x, x_scale = _quantize(inputs.x, axis=1)
    w_up, w_up_sf = _n_major_weight(inputs.w_up)
    return x, _as_e8m0(_swizzle(x_scale)), w_up, jnp.swapaxes(w_up, 1, 2), w_up_sf


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _PROBLEMS)
def test_fused_mxfp8_gemm_reads_n_major_weights_as_their_k_major_transpose_bitwise(problem):
    # The expert MLP's forward GEMMs read their weights N-major so that no weight is transposed;
    # they must compute exactly what the K-major weight-transpose layout computes.
    adapter, _ = _require_sm100()
    offs = jnp.asarray(np.cumsum(_padded_sizes(problem)), jnp.int32)
    x, x_sf, w_n_major, w_k_major, w_sf = jax.jit(_gemm_operands_both_ways)(_make_inputs(problem, seed=2))

    def gemm(b_major):
        return jax.jit(functools.partial(adapter.mxfp8_fused_gemm, b_major=b_major))

    n_major = gemm(WeightMajor.N)(x, x_sf, w_n_major, w_sf, offs)
    k_major = gemm(WeightMajor.K)(x, x_sf, w_k_major, w_sf, offs)

    assert bool(jnp.array_equal(_bits(n_major), _bits(k_major)))
