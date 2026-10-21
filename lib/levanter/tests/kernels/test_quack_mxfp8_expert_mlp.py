# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""The QuACK MXFP8 gated SwiGLU expert MLP against its simulated reference (``mx_swiglu_experts``, checked on any
backend) and the exact MLP (SM100 only)."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug.mxfp8 import (
    _ragged_wgrad,
    mx_dequantize_rows,
    mx_pack_rows,
    mx_quantize_dequantize_groups,
    mx_quantize_row_blocks,
    mx_quantize_rows,
    mx_swiglu_experts,
)
from levanter.grug.mxfp8_pallas import (
    mx_quantize_row_blocks_pallas,
    mx_quantize_rows_and_row_blocks_pallas,
    mx_requantize_row_blocks_pallas,
)

# Three experts with the middle one empty; group boundaries off the 128-row scale tiles.
_GROUP_SIZES = (97, 0, 259)
_NUM_TOKENS = sum(_GROUP_SIZES)
_HIDDEN = 256
_WIDTH = 384


def _require_sm100():
    if jax.default_backend() != "gpu":
        pytest.skip("QuACK MXFP8 GEMMs require an SM100 GPU")
    if float(jax.devices("gpu")[0].compute_capability) < 10.0:
        pytest.skip("QuACK MXFP8 GEMMs require SM100")
    pytest.importorskip("quack")


def _normalized_errors(actual, expected) -> tuple[float, float]:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    error = np.abs(actual - expected) / max(float(np.max(np.abs(expected))), 1e-6)
    return float(np.max(error)), float(np.mean(error))


def _inputs(tail_rows: int):
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.normal(0, 1.0, (_NUM_TOKENS + tail_rows, _HIDDEN)), jnp.bfloat16)
    w13_il = jnp.asarray(rng.normal(0, _HIDDEN**-0.5, (3, _HIDDEN, 2 * _WIDTH)), jnp.bfloat16)
    w_down = jnp.asarray(rng.normal(0, _WIDTH**-0.5, (3, _WIDTH, _HIDDEN)), jnp.bfloat16)
    dy = jnp.asarray(rng.normal(0, 1.0, (_NUM_TOKENS + tail_rows, _HIDDEN)), jnp.bfloat16)
    return x, w13_il, w_down, dy


def _kernel_run(x, w13_il, w_down, dy):
    mx = importlib.import_module("levanter.grug._moe.quack_mxfp8")
    sonic = importlib.import_module("levanter.grug._moe.sonic_cute")
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(_GROUP_SIZES)]), jnp.int32)
    y, residuals = mx.swiglu_expert_mlp_mxfp8_fwd(x, w13_il, w_down, cu)
    dx, row_dot, wgrad_args = mx.swiglu_expert_mlp_mxfp8_backward_input(residuals, dy)
    dw13, dw_down = sonic._swiglu_expert_mlp_quack_wgrads(*wgrad_args)
    return y, dx, dw13, dw_down, row_dot


def _reference_run(reference, x, w13_il, w_down, dy):
    group_sizes = jnp.asarray(_GROUP_SIZES, jnp.int32)
    y, pullback = jax.vjp(lambda a, b, c: reference(a, b, c, group_sizes), x, w13_il, w_down)
    return (y, *pullback(dy.astype(y.dtype)))


def _exact_swiglu_experts(x, w13_il, w_down, group_sizes):
    as_f32 = lambda a: a.astype(jnp.float32)  # noqa: E731
    gu = jax.lax.ragged_dot(as_f32(x), as_f32(w13_il), group_sizes)
    return jax.lax.ragged_dot(jax.nn.silu(gu[:, 0::2]) * gu[:, 1::2], as_f32(w_down), group_sizes)


def test_simulated_reference_stays_within_mxfp8_error_and_skips_empty_experts():
    kx, kw, kd, kg = jax.random.split(jax.random.PRNGKey(4), 4)
    group_sizes = jnp.asarray([40, 0, 88], jnp.int32)
    x = jax.random.normal(kx, (128, 64), jnp.float32)
    w13_il = jax.random.normal(kw, (3, 64, 192), jnp.float32) / 8
    w_down = jax.random.normal(kd, (3, 96, 64), jnp.float32) / 10
    dy = jax.random.normal(kg, (128, 64), jnp.float32)

    y, pullback = jax.vjp(lambda a, b, c: mx_swiglu_experts(a, b, c, group_sizes), x, w13_il, w_down)
    exact_y, exact_pullback = jax.vjp(lambda a, b, c: _exact_swiglu_experts(a, b, c, group_sizes), x, w13_il, w_down)
    for got, want in zip((y, *pullback(dy)), (exact_y, *exact_pullback(dy)), strict=True):
        assert 0 < float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want)) < 0.08
    # The empty expert gets no weight gradient.
    _, dw13, dw_down = pullback(dy)
    assert not np.any(np.asarray(dw13[1])) and not np.any(np.asarray(dw_down[1]))


@pytest.mark.parametrize("dispatch", ["bf16", "mxfp8_bytes"])
@pytest.mark.parametrize("tail_rows", [0, 133])
def test_swiglu_mxfp8_expert_mlp_matches_simulated_reference(dispatch, tail_rows):
    _require_sm100()
    x, w13_il, w_down, dy = _inputs(tail_rows)
    rows = mx_pack_rows(x) if dispatch == "mxfp8_bytes" else x
    # Unused capacity must not affect active outputs or any expert weight gradient.
    if tail_rows:
        nan_tail = jnp.zeros((_NUM_TOKENS + tail_rows, 1), jnp.bfloat16).at[_NUM_TOKENS:].set(jnp.nan)
        dy = dy + nan_tail
        if dispatch == "bf16":
            rows = rows + nan_tail
    y, dx, dw13, dw_down, row_dot = jax.jit(_kernel_run)(rows, w13_il, w_down, dy)
    active = slice(0, _NUM_TOKENS)
    ref_y, ref_dx, ref_dw13, ref_dw_down = jax.jit(lambda *a: _reference_run(mx_swiglu_experts, *a))(
        x[active], w13_il, w_down, dy[active]
    )
    # Same quantized operands on both sides: what differs is accumulation order and the bf16 rounding of
    # intermediates, which can flip a few e4m3 roundings (and, through ``d gu``, bf16 roundings of ``dw13``).
    for name, got, want in [
        ("y", y[active], ref_y),
        ("dx", dx[active], ref_dx),
        ("dw13", dw13, ref_dw13),
        ("dw_down", dw_down, ref_dw_down),
    ]:
        max_error, mean_error = _normalized_errors(got, want)
        assert max_error <= 5e-2 and mean_error <= 4e-3, f"{name}: max {max_error:.2e} mean {mean_error:.2e}"
    # The row dot is <h, dh> with dh from the quantized dy and W_down, so it matches <y, dy> to MXFP8 error only
    # (as the ungated path's epilogue row dot does).
    expected_dot = jnp.sum(ref_y.astype(jnp.float32) * dy[active].astype(jnp.float32), axis=-1)
    max_error, mean_error = _normalized_errors(row_dot[active], expected_dot)
    assert max_error <= 8e-2 and mean_error <= 1.5e-2, f"row_dot: max {max_error:.2e} mean {mean_error:.2e}"


def test_swiglu_mxfp8_expert_mlp_stays_within_mxfp8_error_of_the_exact_mlp():
    _require_sm100()
    x, w13_il, w_down, dy = _inputs(0)
    actual = jax.jit(_kernel_run)(x, w13_il, w_down, dy)[:4]
    expected = jax.jit(lambda *a: _reference_run(_exact_swiglu_experts, *a))(x, w13_il, w_down, dy)
    for name, got, want in zip(("y", "dx", "dw13", "dw_down"), actual, expected, strict=True):
        got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
        relative = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        max_error, mean_error = _normalized_errors(got, want)
        print(f"{name}: relative L2 {relative:.3e}, max normalized {max_error:.3e}, mean normalized {mean_error:.3e}")
        # e4m3 keeps 3 mantissa bits (~3% relative rounding per element); chained GEMMs stay well under 10%.
        assert 0 < relative < 0.1, f"{name}: relative L2 error {relative:.3e}"


def _exact_relu2_experts(x, w_up, w_down, group_sizes):
    as_f32 = lambda a: a.astype(jnp.float32)  # noqa: E731
    pre = jax.lax.ragged_dot(as_f32(x), as_f32(w_up), group_sizes)
    return jax.lax.ragged_dot(jnp.square(jax.nn.relu(pre)), as_f32(w_down), group_sizes)


def _relu2_kernel_run(x, w_up, w_down, dy, *, mxfp8: bool):
    mx = importlib.import_module("levanter.grug._moe.quack_mxfp8")
    sonic = importlib.import_module("levanter.grug._moe.sonic_cute")
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(_GROUP_SIZES)]), jnp.int32)
    if not mxfp8:
        y, residuals = sonic._relu2_expert_mlp_quack_fwd(x, w_up, w_down, cu)
        dx, dw_up, dw_down, row_dot = sonic._relu2_expert_mlp_quack_backward(residuals, dy)
        return y, dx, dw_up, dw_down, row_dot
    y, residuals = mx.relu2_expert_mlp_mxfp8_fwd(x, w_up, w_down, cu)
    dx, row_dot, wgrad_args = mx.relu2_expert_mlp_mxfp8_backward_input(residuals, dy)
    dw_up, dw_down = sonic._relu2_expert_mlp_quack_wgrads(*wgrad_args)
    return y, dx, dw_up, dw_down, row_dot


@pytest.mark.parametrize(("mxfp8", "max_relative"), [(False, 2e-2), (True, 0.1)], ids=["bf16", "mxfp8"])
def test_relu2_expert_mlp_matches_the_exact_mlp(mxfp8, max_relative):
    _require_sm100()
    x, w13_il, w_down, dy = _inputs(0)
    w_up = w13_il[..., :_WIDTH]
    actual = jax.jit(lambda *a: _relu2_kernel_run(*a, mxfp8=mxfp8))(x, w_up, w_down, dy)
    expected = jax.jit(lambda *a: _reference_run(_exact_relu2_experts, *a))(x, w_up, w_down, dy)
    for name, got, want in zip(("y", "dx", "dw_up", "dw_down"), actual[:4], expected, strict=True):
        got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
        relative = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        assert relative < max_relative, f"{name}: relative L2 error {relative:.3e}"
    expected_dot = np.sum(np.asarray(expected[0], np.float32) * np.asarray(dy, np.float32), axis=-1)
    got_dot = np.asarray(actual[4], np.float32)
    assert np.linalg.norm(got_dot - expected_dot) / np.linalg.norm(expected_dot) < max_relative


def test_mx_linear_stays_within_mxfp8_error_of_the_exact_matmul():
    _require_sm100()
    mx = importlib.import_module("levanter.grug._moe.quack_mxfp8")
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.normal(0, 1.0, (2, 192, 256)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(0, 256**-0.5, (256, 384)), jnp.float32)
    dy = jnp.asarray(rng.normal(0, 1.0, (2, 192, 384)), jnp.bfloat16)
    mesh = jax.make_mesh((1,), ("data",), axis_types=(jax.sharding.AxisType.Explicit,))

    def run(fn):
        y, pullback = jax.vjp(fn, x, w)
        return (y, *pullback(dy.astype(y.dtype)))

    with jax.set_mesh(mesh):
        actual = jax.jit(lambda: run(lambda a, b: mx.mx_linear(a, b)))()
        expected = jax.jit(
            lambda: run(lambda a, b: jnp.einsum("...k,kn->...n", a.astype(jnp.float32), b).astype(a.dtype))
        )()
    for name, got, want in zip(("y", "dx", "dw"), actual, expected, strict=True):
        got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
        relative = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        assert 0 < relative < 0.1, f"{name}: relative L2 error {relative:.3e}"


def test_row_block_quantizer_kernel_matches_xla():
    rng = np.random.default_rng(3)
    rows = 100
    # Per-row magnitudes spread over many binades, so every block picks its own scale.
    x = jnp.asarray(rng.normal(size=(rows, 256)) * np.exp2(rng.integers(-6, 6, (rows, 1))), jnp.bfloat16)
    # Full, partial, single-row and empty blocks, including a partial block ending at the last row.
    starts = jnp.asarray([0, 32, 64, 96, 97, 100, 100], jnp.int32)
    counts = jnp.asarray([32, 32, 32, 1, 3, 0, 0], jnp.int32)
    want_values, want_exponents = mx_quantize_row_blocks(x, starts, counts)
    values, exponents = mx_quantize_row_blocks_pallas(x, starts, counts, interpret=True)
    np.testing.assert_array_equal(np.asarray(exponents), np.asarray(want_exponents))
    np.testing.assert_array_equal(np.asarray(values, np.float32), np.asarray(want_values, np.float32))
    assert np.all(np.asarray(exponents[5:]) == 127)
    # Fused with the row quantization, and requantized from MXFP8 rows.
    (row_values, row_exponents), (values, exponents) = mx_quantize_rows_and_row_blocks_pallas(
        x, starts, counts, interpret=True
    )
    want_row_values, want_row_exponents = mx_quantize_rows(x)
    np.testing.assert_array_equal(np.asarray(row_values, np.float32), np.asarray(want_row_values, np.float32))
    np.testing.assert_array_equal(np.asarray(row_exponents), np.asarray(want_row_exponents))
    np.testing.assert_array_equal(np.asarray(values, np.float32), np.asarray(want_values, np.float32))
    np.testing.assert_array_equal(np.asarray(exponents), np.asarray(want_exponents))
    values, exponents = mx_requantize_row_blocks_pallas(
        want_row_values, want_row_exponents, starts, counts, interpret=True
    )
    dequantized = mx_dequantize_rows(want_row_values, want_row_exponents, jnp.float32)
    want_values, want_exponents = mx_quantize_row_blocks(dequantized, starts, counts)
    np.testing.assert_array_equal(np.asarray(values, np.float32), np.asarray(want_values, np.float32))
    np.testing.assert_array_equal(np.asarray(exponents), np.asarray(want_exponents))


def _simulated_wgrad(a, g, group_sizes):
    """Per-group ``a^T g`` of the operands rounded to MXFP8 along the token axis per group, accumulated in f32."""
    qa, qg = mx_quantize_dequantize_groups(a, group_sizes), mx_quantize_dequantize_groups(g, group_sizes)
    return _ragged_wgrad(
        qa.astype(jnp.float32), qg, jnp.zeros((group_sizes.shape[0], a.shape[1], g.shape[1])), group_sizes
    )


def _exact_wgrad(a, g, group_sizes):
    return _ragged_wgrad(
        a.astype(jnp.float32), g, jnp.zeros((group_sizes.shape[0], a.shape[1], g.shape[1])), group_sizes
    )


def _relative_l2(got, want) -> float:
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


@pytest.mark.parametrize("tail_rows", [0, 133])
def test_mxfp8_grouped_wgrad_matches_simulated_and_exact_wgrads(tail_rows):
    _require_sm100()
    mx = importlib.import_module("levanter.grug._moe.quack_mxfp8")
    rng = np.random.default_rng(11)
    rows = _NUM_TOKENS + tail_rows
    group_sizes = jnp.asarray(_GROUP_SIZES, jnp.int32)
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(_GROUP_SIZES)]), jnp.int32)
    scale = np.exp2(rng.integers(-4, 4, (rows, 1)))
    a = jnp.asarray(rng.normal(size=(rows, _HIDDEN)) * scale, jnp.bfloat16)
    g = jnp.asarray(rng.normal(size=(rows, 2 * _WIDTH)), jnp.bfloat16)
    if tail_rows:
        # Unused capacity must not reach any block scale or gradient.
        a = a.at[_NUM_TOKENS:].set(jnp.nan)
        g = g.at[_NUM_TOKENS:].set(jnp.inf)
    got = jax.jit(mx.mxfp8_grouped_wgrad)(a, g, cu)
    active = slice(0, _NUM_TOKENS)
    simulated = jax.jit(_simulated_wgrad)(a[active], g[active], group_sizes)
    exact = jax.jit(_exact_wgrad)(a[active], g[active], group_sizes)
    assert got.shape == (3, _HIDDEN, 2 * _WIDTH) and got.dtype == jnp.bfloat16
    assert not np.any(np.asarray(got[1], np.float32)), "the empty expert must get a zero gradient"
    # Same quantized operands: only the accumulation order and the final bf16 rounding differ.
    max_error, mean_error = _normalized_errors(got, simulated)
    assert max_error <= 1e-2 and mean_error <= 1e-3, f"vs simulated: max {max_error:.2e} mean {mean_error:.2e}"
    relative = _relative_l2(got, exact)
    max_error, mean_error = _normalized_errors(got, exact)
    print(f"wgrad vs exact: relative L2 {relative:.3e}, max normalized {max_error:.3e}, mean {mean_error:.3e}")
    assert 0 < relative < 0.06, f"vs exact: relative L2 {relative:.3e}"


@pytest.mark.parametrize("dispatch", ["bf16", "mxfp8_bytes"])
@pytest.mark.parametrize("gated", [True, False], ids=["swiglu", "relu2"])
def test_mxfp8_expert_backward_wgrads_stay_within_mxfp8_error_of_the_exact_mlp(gated, dispatch):
    _require_sm100()
    mx = importlib.import_module("levanter.grug._moe.quack_mxfp8")
    sonic = importlib.import_module("levanter.grug._moe.sonic_cute")
    tail_rows = 133
    x, w13_il, w_down, dy = _inputs(tail_rows)
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(_GROUP_SIZES)]), jnp.int32)
    rows = mx_pack_rows(x) if dispatch == "mxfp8_bytes" else x
    # Unused capacity must not reach any block scale or gradient.
    nan_tail = jnp.zeros((_NUM_TOKENS + tail_rows, 1), jnp.bfloat16).at[_NUM_TOKENS:].set(jnp.nan)
    dy = dy + nan_tail
    if dispatch == "bf16":
        rows = rows + nan_tail
    fwd, wgrad_fwd, backward_input, backward, bf16_wgrads = (
        (
            mx.swiglu_expert_mlp_mxfp8_fwd,
            mx.swiglu_expert_mlp_mxfp8_wgrad_fwd,
            mx.swiglu_expert_mlp_mxfp8_backward_input,
            mx.swiglu_expert_mlp_mxfp8_backward,
            sonic._swiglu_expert_mlp_quack_wgrads,
        )
        if gated
        else (
            mx.relu2_expert_mlp_mxfp8_fwd,
            mx.relu2_expert_mlp_mxfp8_wgrad_fwd,
            mx.relu2_expert_mlp_mxfp8_backward_input,
            mx.relu2_expert_mlp_mxfp8_backward,
            sonic._relu2_expert_mlp_quack_wgrads,
        )
    )

    def run(x, w_up, w_down, dy):
        y, residuals = fwd(x, w_up, w_down, cu)
        wgrad_y, wgrad_residuals = wgrad_fwd(x, w_up, w_down, cu)
        dx, row_dot, wgrad_args = backward_input(residuals, dy)
        return (
            (y, wgrad_y),
            backward(wgrad_residuals, dy),
            (dx, row_dot),
            bf16_wgrads(*wgrad_args),
            mx.expert_mlp_mxfp8_wgrads(*wgrad_args),
        )

    w_up = w13_il if gated else w13_il[..., :_WIDTH]
    reference = _exact_swiglu_experts if gated else _exact_relu2_experts
    (y, wgrad_y), fused, (dx, row_dot), bf16_wgrads, unfused_wgrads = jax.jit(run)(rows, w_up, w_down, dy)
    active = slice(0, _NUM_TOKENS)
    np.testing.assert_array_equal(np.asarray(wgrad_y[active], np.float32), np.asarray(y[active], np.float32))
    _, _, exact_dw_up, exact_dw_down = jax.jit(lambda *a: _reference_run(reference, *a))(
        x[active], w_up, w_down, dy[active]
    )
    # The fused backward's input gradient is the activation-only path's: its row quantization is the same.
    np.testing.assert_array_equal(np.asarray(fused[0][active], np.float32), np.asarray(dx[active], np.float32))
    np.testing.assert_array_equal(np.asarray(fused[3][active]), np.asarray(row_dot[active]))
    for index, name, want in ((1, "dw_up", exact_dw_up), (2, "dw_down", exact_dw_down)):
        got, bf16 = fused[index], bf16_wgrads[index - 1]
        relative, bf16_relative = _relative_l2(got, want), _relative_l2(bf16, want)
        max_error, mean_error = _normalized_errors(got, want)
        print(
            f"{'swiglu' if gated else 'relu2'} {dispatch} {name}: mxfp8 wgrad relative L2 {relative:.3e} "
            f"(bf16 wgrad {bf16_relative:.3e}), max normalized {max_error:.3e}, mean {mean_error:.3e}"
        )
        assert 0 < relative < 0.1, f"{name}: relative L2 error {relative:.3e}"
        assert not np.any(np.asarray(got[1], np.float32)), f"{name}: the empty expert must get a zero gradient"
        # The unfused weight gradients quantize the same bf16 operands, except ReLU^2's ``d pre``, which the fused
        # path requantizes from its MXFP8 rows and the unfused one from their bf16 dequantization: equal values.
        max_error, _ = _normalized_errors(unfused_wgrads[index - 1], got)
        assert max_error <= 1e-2, f"{name}: fused vs unfused max normalized {max_error:.2e}"


@pytest.mark.parametrize("dispatch", ["bf16", "mxfp8_bytes"])
def test_relu2_mxfp8_split_forward_matches_the_full_forward_bitwise(dispatch):
    """``ExpertForward.LEAN``/``RESIDUALS``: the output-only and residuals-only forwards give the full forward's
    output, residuals and backward bit for bit on the active rows."""
    _require_sm100()
    mx = importlib.import_module("levanter.grug._moe.quack_mxfp8")
    tail_rows = 133
    x, w13_il, w_down, dy = _inputs(tail_rows)
    w_up = w13_il[..., :_WIDTH]
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(_GROUP_SIZES)]), jnp.int32)
    rows = mx_pack_rows(x) if dispatch == "mxfp8_bytes" else x

    def run(x, w_up, w_down, dy):
        y, residuals = mx.relu2_expert_mlp_mxfp8_wgrad_fwd(x, w_up, w_down, cu)
        lean_y, _ = mx.relu2_expert_mlp_mxfp8_wgrad_output(x, w_up, w_down, cu)
        lean_residuals = mx.relu2_expert_mlp_mxfp8_wgrad_residuals(x, w_up, w_down, cu)
        return (
            (y, lean_y),
            mx.relu2_expert_mlp_mxfp8_backward(residuals, dy),
            mx.relu2_expert_mlp_mxfp8_backward(lean_residuals, dy),
            residuals[3],
            lean_residuals[3],
        )

    (y, lean_y), full_grads, lean_grads, pre, lean_pre = jax.jit(run)(rows, w_up, w_down, dy)
    active = slice(0, _NUM_TOKENS)
    np.testing.assert_array_equal(np.asarray(lean_y[active], np.float32), np.asarray(y[active], np.float32))
    np.testing.assert_array_equal(np.asarray(lean_pre[active], np.float32), np.asarray(pre[active], np.float32))
    dx, dw_up, dw_down, row_dot = full_grads
    lean_dx, lean_dw_up, lean_dw_down, lean_row_dot = lean_grads
    np.testing.assert_array_equal(np.asarray(lean_dx[active], np.float32), np.asarray(dx[active], np.float32))
    np.testing.assert_array_equal(np.asarray(lean_row_dot[active]), np.asarray(row_dot[active]))
    np.testing.assert_array_equal(np.asarray(lean_dw_up, np.float32), np.asarray(dw_up, np.float32))
    np.testing.assert_array_equal(np.asarray(lean_dw_down, np.float32), np.asarray(dw_down, np.float32))
