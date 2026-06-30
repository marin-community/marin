# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import importlib
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from haliax.nn import ragged_dot
from haliax.quantization import Fp8RaggedDotOp, apply_updates, partition_for_grad_overwrite


def _ragged_inputs(dtype=jnp.float32):
    lhs = jrandom.normal(jrandom.PRNGKey(0), (8, 16), dtype=dtype)
    rhs = jrandom.normal(jrandom.PRNGKey(1), (2, 16, 12), dtype=dtype)
    group_sizes = jnp.array([3, 5], dtype=jnp.int32)
    return lhs, rhs, group_sizes


def _relative_frobenius(got, want):
    return np.linalg.norm(np.asarray(got - want)) / np.linalg.norm(np.asarray(want))


def test_fp8_ragged_dot_forward_matches_bf16_reference():
    lhs, rhs, group_sizes = _ragged_inputs(jnp.bfloat16)
    fp8_op = Fp8RaggedDotOp.init(amax_history_length=16)

    fp8_out = ragged_dot(lhs, rhs, group_sizes, implementation="xla", fp8_dot=fp8_op)
    ref_out = ragged_dot(lhs, rhs, group_sizes, implementation="xla")

    assert fp8_out.shape == ref_out.shape
    assert fp8_out.dtype == ref_out.dtype
    assert _relative_frobenius(fp8_out.astype(jnp.float32), ref_out.astype(jnp.float32)) < 0.12


def test_fp8_ragged_dot_grads_match_reference():
    lhs, rhs, group_sizes = _ragged_inputs(jnp.float32)
    fp8_op = Fp8RaggedDotOp.init(amax_history_length=16, compute_dtype=jnp.float32)

    def fp8_loss(lhs, rhs):
        out = ragged_dot(lhs, rhs, group_sizes, implementation="xla", fp8_dot=fp8_op)
        return jnp.sum(out.astype(jnp.float32) ** 2)

    def ref_loss(lhs, rhs):
        out = ragged_dot(lhs, rhs, group_sizes, implementation="xla")
        return jnp.sum(out.astype(jnp.float32) ** 2)

    fp8_grads = jax.grad(fp8_loss, argnums=(0, 1))(lhs, rhs)
    ref_grads = jax.grad(ref_loss, argnums=(0, 1))(lhs, rhs)

    assert _relative_frobenius(fp8_grads[0], ref_grads[0]) < 0.2
    assert _relative_frobenius(fp8_grads[1], ref_grads[1]) < 0.2


def test_fp8_ragged_dot_updates_delayed_scaling_state():
    lhs, rhs, group_sizes = _ragged_inputs(jnp.float32)
    fp8_op = Fp8RaggedDotOp.init(amax_history_length=16, compute_dtype=jnp.float32)

    def loss(op):
        out = ragged_dot(lhs, rhs, group_sizes, implementation="xla", fp8_dot=op)
        return jnp.sum(out.astype(jnp.float32))

    def apply_gradients(op):
        grads = eqx.filter_grad(loss)(op)
        overwrites, grads = partition_for_grad_overwrite(grads)
        return apply_updates(op, grads, overwrites)

    updated = apply_gradients(fp8_op)
    updated_again = apply_gradients(updated)

    expected_lhs_amax = jnp.max(jnp.abs(lhs))
    expected_rhs_amax = jnp.max(jnp.abs(rhs))
    np.testing.assert_allclose(updated.input_amax_history[0], expected_lhs_amax, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(updated.kernel_amax_history[0], expected_rhs_amax, rtol=1e-5, atol=1e-5)
    assert updated.output_grad_amax_history[0] > 0
    assert not np.allclose(np.asarray(updated_again.input_scale), np.asarray(updated.input_scale))


def test_fp8_ragged_dot_contracts_e4m3_forward_and_e5m2_backward(monkeypatch):
    ragged_dot_module = importlib.import_module("haliax.nn.ragged_dot")

    lhs, rhs, group_sizes = _ragged_inputs(jnp.float32)
    fp8_op = Fp8RaggedDotOp.init(amax_history_length=16, compute_dtype=jnp.float32)
    calls = []

    def fake_pallas_call(
        lhs,
        rhs,
        group_sizes,
        ragged_dot_dimension_numbers=ragged_dot_module._DEFAULT_DIM_NUMS,
        *,
        out_dtype=None,
        max_group_size=None,
        block_m=None,
        block_n=None,
        block_k=None,
    ):
        del max_group_size, block_m, block_n, block_k
        calls.append((lhs.dtype, rhs.dtype, ragged_dot_dimension_numbers, out_dtype))
        out = jax.lax.ragged_dot_general(
            lhs=lhs.astype(jnp.float32),
            rhs=rhs.astype(jnp.float32),
            group_sizes=group_sizes,
            ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
        )
        if out_dtype is not None:
            out = out.astype(out_dtype)
        return out

    monkeypatch.setattr(ragged_dot_module, "_triton_pallas_call", fake_pallas_call)

    def loss(lhs, rhs):
        out = ragged_dot(lhs, rhs, group_sizes, implementation="triton", fp8_dot=fp8_op)
        return jnp.sum(out.astype(jnp.float32) ** 2)

    jax.grad(loss, argnums=(0, 1))(lhs, rhs)

    assert calls[0][:2] == (jnp.float8_e4m3fn, jnp.float8_e4m3fn)
    assert any(jnp.float8_e5m2 in call[:2] for call in calls[1:])
    assert {call[2] for call in calls} == {
        ragged_dot_module._DEFAULT_DIM_NUMS,
        ragged_dot_module._DLHS_DIM_NUMS,
        ragged_dot_module._DRHS_DIM_NUMS,
    }


def test_fp8_nonuniform_groups_route_through_ragged_triton_layouts(monkeypatch):
    ragged_dot_module = importlib.import_module("haliax.nn.ragged_dot")
    lhs = jrandom.normal(jrandom.PRNGKey(0), (512, 16), dtype=jnp.float32)
    rhs = jrandom.normal(jrandom.PRNGKey(1), (2, 16, 12), dtype=jnp.float32)
    group_sizes = jnp.array([255, 257], dtype=jnp.int32)
    fp8_op = Fp8RaggedDotOp.init(amax_history_length=16, compute_dtype=jnp.float32)
    calls = []

    def fake_pallas_call(
        lhs,
        rhs,
        group_sizes,
        ragged_dot_dimension_numbers=ragged_dot_module._DEFAULT_DIM_NUMS,
        *,
        out_dtype=None,
        max_group_size=None,
        block_m=None,
        block_n=None,
        block_k=None,
    ):
        del max_group_size, block_m, block_n, block_k
        calls.append((lhs.dtype, rhs.dtype, ragged_dot_dimension_numbers))
        out = jax.lax.ragged_dot_general(
            lhs=lhs.astype(jnp.float32),
            rhs=rhs.astype(jnp.float32),
            group_sizes=group_sizes,
            ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
        )
        if out_dtype is not None:
            out = out.astype(out_dtype)
        return out

    monkeypatch.setattr(ragged_dot_module, "_triton_pallas_call", fake_pallas_call)

    def loss(lhs, rhs):
        out = ragged_dot(lhs, rhs, group_sizes, implementation="triton", fp8_dot=fp8_op, max_group_size=256)
        return jnp.sum(out.astype(jnp.float32) ** 2)

    jax.grad(loss, argnums=(0, 1))(lhs, rhs)

    assert calls[0] == (jnp.float8_e4m3fn, jnp.float8_e4m3fn, ragged_dot_module._DEFAULT_DIM_NUMS)
    assert calls[1] == (jnp.float8_e5m2, jnp.float8_e5m2, ragged_dot_module._DLHS_DIM_NUMS)
    assert calls[2] == (jnp.float8_e5m2, jnp.float8_e5m2, ragged_dot_module._DRHS_DIM_NUMS)


def test_fp8_ragged_dot_can_be_stacked_by_equinox():
    axis = 3
    fp8_op = Fp8RaggedDotOp.init(amax_history_length=16)
    stacked = jax.vmap(lambda _: fp8_op)(jnp.arange(axis))

    assert isinstance(stacked, Fp8RaggedDotOp)
    assert stacked.input_scale.shape == (axis, 1)
    assert stacked.input_amax_history.shape == (axis, 16)
