# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""The QuACK MXFP8 gated SwiGLU expert MLP against its simulated reference (``mx_swiglu_experts``, checked on any
backend) and the exact MLP (SM100 only)."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug.mxfp8 import mx_pack_rows, mx_swiglu_experts

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
