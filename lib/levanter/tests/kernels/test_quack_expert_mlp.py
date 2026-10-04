# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Numerical coverage for the QuACK epilogue API used by expert training and Muon."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug._moe.shared_swiglu import select_shared_swiglu_mlp

_NUM_TOKENS = 224
_EXPERT_SPLIT = 97
_CU_SEQLENS = (0, _EXPERT_SPLIT, _EXPERT_SPLIT, _NUM_TOKENS)


def _require_sm100():
    if jax.default_backend() != "gpu":
        pytest.skip("QuACK expert GEMMs require an SM100 GPU")
    if float(jax.devices("gpu")[0].compute_capability) < 10.0:
        pytest.skip("QuACK expert GEMMs require SM100")
    pytest.importorskip("quack")


def _assert_bfloat16_close(actual, expected):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    scale = max(float(np.max(np.abs(expected))), 1e-6)
    normalized_absolute_error = np.abs(actual - expected) / scale
    max_error = float(np.max(normalized_absolute_error))
    mean_error = float(np.mean(normalized_absolute_error))
    assert max_error <= 2e-2, f"max normalized absolute error: {max_error}"
    # Half a bfloat16 ULP at unit scale is about 4e-3; leave a small margin for chained operations.
    assert mean_error <= 5e-3, f"mean normalized absolute error: {mean_error}"


@pytest.mark.parametrize("use_clc", [False, True])
def test_gated_grouped_gemm_keeps_preact_and_interleaved_swiglu(use_clc):
    _require_sm100()
    kernels = importlib.import_module("levanter.grug._moe.quack_moe_cute")
    rng = np.random.default_rng(42)
    x = jnp.asarray(rng.normal(0, 0.2, (_NUM_TOKENS, 64)), dtype=jnp.bfloat16)
    w = jnp.asarray(rng.normal(0, 0.2, (3, 64, 128)), dtype=jnp.bfloat16)
    cu = jnp.asarray(_CU_SEQLENS, dtype=jnp.int32)
    preact, postact = jax.jit(
        lambda a, b: kernels.quack_gated_grouped_gemm(a, b, cu, return_preact=True, use_clc_persistence=use_clc)
    )(x, w)
    expected = jnp.concatenate(
        [
            x[:_EXPERT_SPLIT].astype(jnp.float32) @ w[0].astype(jnp.float32),
            x[_EXPERT_SPLIT:].astype(jnp.float32) @ w[2].astype(jnp.float32),
        ]
    )
    _assert_bfloat16_close(preact, expected)
    _assert_bfloat16_close(postact, jax.nn.silu(expected[:, 0::2]) * expected[:, 1::2])


@pytest.mark.parametrize("tail_rows", [0, 127])
def test_expert_mlp_forward_and_all_gradients_match_reference(tail_rows):
    _require_sm100()
    sonic = importlib.import_module("levanter.grug._moe.sonic_cute")
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.normal(0, 0.2, (_NUM_TOKENS, 64)), dtype=jnp.bfloat16)
    w13 = jnp.asarray(rng.normal(0, 0.2, (3, 64, 128)), dtype=jnp.bfloat16)
    w2 = jnp.asarray(rng.normal(0, 0.2, (3, 64, 64)), dtype=jnp.bfloat16)
    dy = jnp.asarray(rng.normal(0, 0.2, (_NUM_TOKENS, 64)), dtype=jnp.bfloat16)
    # Unused capacity must not affect active outputs or any expert weight gradient.
    x = jnp.pad(x, ((0, tail_rows), (0, 0)), constant_values=jnp.nan)
    dy = jnp.pad(dy, ((0, tail_rows), (0, 0)), constant_values=jnp.nan)
    cu = jnp.asarray(_CU_SEQLENS, dtype=jnp.int32)

    def reference(a, b, c):
        outputs = []
        for expert, start, stop in [(0, 0, _EXPERT_SPLIT), (2, _EXPERT_SPLIT, _NUM_TOKENS)]:
            gu = a[start:stop] @ b[expert]
            h = jax.nn.silu(gu[:, 0::2]) * gu[:, 1::2]
            outputs.append(h @ c[expert])
        return jnp.concatenate(outputs)

    @jax.jit
    def run(a, b, c, dy):
        y, residuals = sonic._expert_mlp_quack_wgrad_fwd(a, b, c, cu)
        dx, dw13, dw2, _row_dot = sonic._expert_mlp_quack_wgrad_backward(residuals, dy)
        return y, (dx, dw13, dw2)

    actual, actual_gradients = run(x, w13, w2, dy)
    expected, expected_pullback = jax.vjp(jax.jit(reference), x, w13, w2)
    _assert_bfloat16_close(actual[:_NUM_TOKENS], expected)
    expected_gradients = expected_pullback(dy[:_NUM_TOKENS])
    _assert_bfloat16_close(actual_gradients[0][:_NUM_TOKENS], expected_gradients[0][:_NUM_TOKENS])
    for got, want in zip(actual_gradients[1:], expected_gradients[1:], strict=True):
        _assert_bfloat16_close(got, want)


@pytest.mark.parametrize("tail_rows", [0, 32])
# The row dot sums one partial per 128-column tile of dh; 320 columns end in a partial third tile.
@pytest.mark.parametrize("intermediate", [64, 320])
def test_dswiglu_gemm_matches_the_swiglu_backward_of_dh(tail_rows, intermediate):
    _require_sm100()
    kernels = importlib.import_module("levanter.grug._moe.quack_moe_cute")
    rng = np.random.default_rng(13)
    dy = jnp.asarray(rng.normal(0, 0.2, (_NUM_TOKENS, 64)), dtype=jnp.bfloat16)
    gate_up = jnp.asarray(rng.normal(0, 1.0, (_NUM_TOKENS, 2 * intermediate)), dtype=jnp.bfloat16)
    w2 = jnp.asarray(rng.normal(0, 0.2, (3, intermediate, 64)), dtype=jnp.bfloat16)
    # Rows past the last group are unspecified input and must not reach the active rows.
    dy = jnp.pad(dy, ((0, tail_rows), (0, 0)), constant_values=jnp.nan)
    gate_up = jnp.pad(gate_up, ((0, tail_rows), (0, 0)), constant_values=jnp.nan)
    cu = jnp.asarray(_CU_SEQLENS, dtype=jnp.int32)

    d_gate_up, row_dot = jax.jit(lambda dy, gu: kernels.quack_grouped_dswiglu_gemm(dy, w2, gu, cu))(dy, gate_up)

    rows = []
    for expert, start, stop in [(0, 0, _EXPERT_SPLIT), (2, _EXPERT_SPLIT, _NUM_TOKENS)]:
        rows.append(dy[start:stop].astype(jnp.float32) @ w2[expert].astype(jnp.float32).T)
    dh = jnp.concatenate(rows)
    gate = gate_up[:_NUM_TOKENS, 0::2].astype(jnp.float32)
    up = gate_up[:_NUM_TOKENS, 1::2].astype(jnp.float32)
    h, pullback = jax.vjp(lambda g, u: jax.nn.silu(g) * u, gate, up)
    d_gate, d_up = pullback(dh)
    expected = jnp.stack([d_gate, d_up], axis=-1).reshape(_NUM_TOKENS, 2 * intermediate)
    _assert_bfloat16_close(d_gate_up[:_NUM_TOKENS], expected)
    _assert_bfloat16_close(row_dot[:_NUM_TOKENS], jnp.sum(h * dh, axis=-1))


# The production dh GEMM tiles 256 columns; 320 columns end in a partial second tile.
@pytest.mark.parametrize("intermediate", [64, 320])
def test_expert_mlp_backward_row_dot_is_the_output_scale_gradient(intermediate):
    _require_sm100()
    sonic = importlib.import_module("levanter.grug._moe.sonic_cute")
    rng = np.random.default_rng(11)
    x = jnp.asarray(rng.normal(0, 0.2, (_NUM_TOKENS, 64)), dtype=jnp.bfloat16)
    w13 = jnp.asarray(rng.normal(0, 0.2, (3, 64, 2 * intermediate)), dtype=jnp.bfloat16)
    w2 = jnp.asarray(rng.normal(0, 0.2, (3, intermediate, 64)), dtype=jnp.bfloat16)
    dy = jnp.asarray(rng.normal(0, 0.2, (_NUM_TOKENS, 64)), dtype=jnp.bfloat16)
    cu = jnp.asarray(_CU_SEQLENS, dtype=jnp.int32)

    @jax.jit
    def run(x, w13, w2, dy):
        y, residuals = sonic._expert_mlp_quack_wgrad_fwd(x, w13, w2, cu)
        return y, sonic._expert_mlp_quack_wgrad_backward(residuals, dy)[3]

    y, row_dot = run(x, w13, w2, dy)

    # The row dot is d/ds of <s * y, dy> for a per-row scale s, without reading y.
    expected = np.sum(np.asarray(y, np.float32) * np.asarray(dy, np.float32), axis=-1)
    _assert_bfloat16_close(row_dot, expected)


def test_gated_grouped_gemm_without_preact_returns_the_same_swiglu():
    _require_sm100()
    kernels = importlib.import_module("levanter.grug._moe.quack_moe_cute")
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.normal(0, 0.2, (_NUM_TOKENS, 64)), dtype=jnp.bfloat16)
    w = jnp.asarray(rng.normal(0, 0.2, (3, 64, 128)), dtype=jnp.bfloat16)
    cu = jnp.asarray(_CU_SEQLENS, dtype=jnp.int32)
    _preact, with_preact = jax.jit(lambda a, b: kernels.quack_gated_grouped_gemm(a, b, cu, return_preact=True))(x, w)
    without_preact = jax.jit(lambda a, b: kernels.quack_gated_grouped_gemm(a, b, cu))(x, w)
    np.testing.assert_array_equal(np.asarray(without_preact), np.asarray(with_preact))


def _shared_mlp_reference(x, w_gate, w_up, w_down):
    f32 = jnp.float32
    gate = x.astype(f32) @ w_gate.astype(f32)
    up = x.astype(f32) @ w_up.astype(f32)
    return (jax.nn.silu(gate) * up) @ w_down.astype(f32)


def _shared_operands(seed):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.normal(0, 1.0, (_NUM_TOKENS, 64)), dtype=jnp.bfloat16)
    w_gate = jnp.asarray(rng.normal(0, 0.125, (64, 96)), dtype=jnp.bfloat16)
    w_up = jnp.asarray(rng.normal(0, 0.125, (64, 96)), dtype=jnp.bfloat16)
    w_down = jnp.asarray(rng.normal(0, 0.1, (96, 64)), dtype=jnp.bfloat16)
    dy = jnp.asarray(rng.normal(0, 1.0, (_NUM_TOKENS, 64)), dtype=jnp.bfloat16)
    return x, w_gate, w_up, w_down, dy


def test_shared_gate_up_and_swiglu_down_match_the_swiglu_mlp_and_its_gradients():
    _require_sm100()
    sonic = importlib.import_module("levanter.grug._moe.sonic_cute")
    x, w_gate, w_up, w_down, dy = _shared_operands(17)

    def fused(x, w_gate, w_up, w_down):
        preact, h = sonic.shared_gate_up(x, w_gate, w_up)
        return sonic.shared_swiglu_down(preact, h, w_down)

    actual, actual_pullback = jax.vjp(jax.jit(fused), x, w_gate, w_up, w_down)
    expected, expected_pullback = jax.vjp(_shared_mlp_reference, x, w_gate, w_up, w_down)
    _assert_bfloat16_close(actual, expected)
    for got, want in zip(actual_pullback(dy), expected_pullback(dy.astype(jnp.float32)), strict=True):
        _assert_bfloat16_close(got, want)


def test_shared_gate_up_takes_a_gradient_on_its_swiglu_output():
    _require_sm100()
    sonic = importlib.import_module("levanter.grug._moe.sonic_cute")
    x, w_gate, w_up, _w_down, _dy = _shared_operands(19)
    probe = jnp.asarray(np.random.default_rng(23).normal(0, 1.0, (_NUM_TOKENS, 96)), dtype=jnp.float32)

    # A consumer that reads both outputs: the gradient reaches the pre-activations both directly
    # and through h, which the backward has to fold through the SwiGLU backward itself.
    def loss(x, w_gate, w_up):
        preact, h = sonic.shared_gate_up(x, w_gate, w_up)
        return jnp.sum(h.astype(jnp.float32) * probe) + jnp.sum(preact[:, 0::2].astype(jnp.float32) * probe)

    def reference_loss(x, w_gate, w_up):
        f32 = jnp.float32
        gate = x.astype(f32) @ w_gate.astype(f32)
        up = x.astype(f32) @ w_up.astype(f32)
        return jnp.sum(jax.nn.silu(gate) * up * probe) + jnp.sum(gate * probe)

    actual = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(x, w_gate, w_up)
    expected = jax.grad(reference_loss, argnums=(0, 1, 2))(x, w_gate, w_up)
    for got, want in zip(actual, expected, strict=True):
        _assert_bfloat16_close(got, want)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32], ids=["bfloat16", "float32"])
def test_selected_shared_swiglu_mlp_matches_the_reference_in_each_dtype(dtype):
    # QuACK's gated epilogue takes only 16-bit outputs, so float32 operands must select the einsums.
    _require_sm100()
    x, w_gate, w_up, w_down, dy = (operand.astype(dtype) for operand in _shared_operands(29))
    mlp = select_shared_swiglu_mlp(jax.nn.silu, dtype)

    def run(x, w_gate, w_up, w_down):
        return mlp.down(mlp.gate_up(x, w_gate, w_up), w_down)

    actual, actual_pullback = jax.vjp(jax.jit(run), x, w_gate, w_up, w_down)
    expected, expected_pullback = jax.vjp(_shared_mlp_reference, x, w_gate, w_up, w_down)
    _assert_bfloat16_close(actual, expected)
    for got, want in zip(actual_pullback(dy), expected_pullback(dy.astype(jnp.float32)), strict=True):
        _assert_bfloat16_close(got, want)


def test_muon_symmetric_gemm_matches_gram_matrix():
    _require_sm100()
    kernels = importlib.import_module("levanter.grug._moe.quack_symmetric_cute")
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(0, 0.2, (3, 256, 128)), dtype=jnp.bfloat16)
    got = jax.jit(kernels.quack_symmetric_gemm)(x)
    expected = x.astype(jnp.float32) @ jnp.swapaxes(x.astype(jnp.float32), -1, -2)
    _assert_bfloat16_close(got, expected)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(jnp.swapaxes(got, -1, -2)))
