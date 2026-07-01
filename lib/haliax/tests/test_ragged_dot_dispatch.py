# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from haliax.nn import ragged_dot
from haliax.quantization import Fp8RaggedDotOp, partition_for_grad_overwrite

ragged_dot_module = importlib.import_module("haliax.nn.ragged_dot")

gpu_only = pytest.mark.skipif(jax.default_backend() != "gpu", reason="fp8 wgmma only lowers on GPU")


def _inputs():
    lhs = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    rhs = jnp.arange(2 * 4 * 5, dtype=jnp.float32).reshape(2, 4, 5)
    group_sizes = jnp.array([2, 1], dtype=jnp.int32)
    return lhs, rhs, group_sizes


def test_ragged_dot_platform_default_is_close_to_xla_call():
    lhs, rhs, group_sizes = _inputs()

    default_out = ragged_dot(lhs, rhs, group_sizes, implementation="auto")
    xla_out = ragged_dot(lhs, rhs, group_sizes, implementation="xla")

    assert jnp.allclose(default_out, xla_out, rtol=1e-5, atol=1e-5)


def test_triton_kernel_traces_with_jax_0_9_pallas_memory_api_on_cpu_interpreter():
    if not ragged_dot_module._has_pallas_triton:
        pytest.skip("Pallas Triton backend is not available")

    lhs = jnp.arange(4, dtype=jnp.float32).reshape(2, 2)
    rhs = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    lo = jnp.array(0, dtype=jnp.int32)
    hi = jnp.array(lhs.shape[0], dtype=jnp.int32)

    pallas_call = ragged_dot_module.pl.pallas_call(
        lambda a, b, lo, hi, out: ragged_dot_module._triton_ragged_dot_kernel(
            a, b, lo, hi, out, block_m=lhs.shape[0], block_k=lhs.shape[1]
        ),
        out_shape=jax.ShapeDtypeStruct((lhs.shape[0], rhs.shape[1]), lhs.dtype),
        in_specs=[
            ragged_dot_module.pl.no_block_spec,
            ragged_dot_module.pl.no_block_spec,
            ragged_dot_module.pl.no_block_spec,
            ragged_dot_module.pl.no_block_spec,
        ],
        out_specs=ragged_dot_module.pl.no_block_spec,
        grid=(1,),
        interpret=True,
    )

    assert jnp.allclose(pallas_call(lhs, rhs, lo, hi), lhs @ rhs, rtol=1e-5, atol=1e-5)


def test_ragged_dot_gpu_auto_uses_triton_when_available(monkeypatch):
    lhs, rhs, group_sizes = _inputs()
    expected = jnp.full((lhs.shape[0], rhs.shape[2]), 17.0, dtype=lhs.dtype)
    monkeypatch.setattr(ragged_dot_module.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(ragged_dot_module, "_has_pallas_triton", True)

    def triton_result(lhs, rhs, group_sizes):
        return jnp.full((lhs.shape[0], rhs.shape[2]), expected[0, 0], dtype=lhs.dtype)

    monkeypatch.setattr(ragged_dot_module, "_ragged_dot_triton_impl", triton_result)

    auto_out = ragged_dot(lhs, rhs, group_sizes, implementation="auto")

    assert jnp.array_equal(auto_out, expected)


def test_triton_default_block_sizes_use_blackwell_n_tile(monkeypatch):
    monkeypatch.setattr(ragged_dot_module, "_is_blackwell_gpu_backend", lambda: True)

    assert ragged_dot_module._triton_default_block_sizes(32768, 5120, 5120) == (128, 256, 32)


def test_triton_default_block_sizes_keep_non_blackwell_n_tile(monkeypatch):
    monkeypatch.setattr(ragged_dot_module, "_is_blackwell_gpu_backend", lambda: False)

    assert ragged_dot_module._triton_default_block_sizes(32768, 5120, 5120) == (128, 128, 32)


def test_triton_custom_vjp_routes_backward_through_triton_layouts(monkeypatch):
    lhs, rhs, group_sizes = _inputs()
    calls = []

    def fake_triton_pallas_call(
        lhs,
        rhs,
        group_sizes,
        ragged_dot_dimension_numbers=ragged_dot_module._DEFAULT_DIM_NUMS,
    ):
        calls.append(ragged_dot_dimension_numbers)
        return jax.lax.ragged_dot_general(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
        )

    monkeypatch.setattr(ragged_dot_module, "_has_pallas_triton", True)
    monkeypatch.setattr(ragged_dot_module, "_triton_pallas_call", fake_triton_pallas_call)

    def triton_loss(lhs, rhs):
        return jnp.sum(ragged_dot_module._ragged_dot_triton_impl(lhs, rhs, group_sizes))

    def xla_loss(lhs, rhs):
        return jnp.sum(ragged_dot(lhs, rhs, group_sizes, implementation="xla"))

    triton_value, triton_grads = jax.value_and_grad(triton_loss, argnums=(0, 1))(lhs, rhs)
    xla_value, xla_grads = jax.value_and_grad(xla_loss, argnums=(0, 1))(lhs, rhs)

    assert jnp.allclose(triton_value, xla_value, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(triton_grads[0], xla_grads[0], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(triton_grads[1], xla_grads[1], rtol=1e-5, atol=1e-5)
    assert calls == [
        ragged_dot_module._DEFAULT_DIM_NUMS,
        ragged_dot_module._DLHS_DIM_NUMS,
        ragged_dot_module._DRHS_DIM_NUMS,
    ]


# ---------------------------------------------------------------------------
# FP8 op dispatch tests (routing, op=None reference, state plumbing)
# ---------------------------------------------------------------------------


def _fp8_inputs(T=64, K=128, E=4, N=128, seed=0):
    # FP8-wgmma-friendly dims (K, N multiples of 128 so the FP8 kernel's block tiles
    # and bf16 output-store swizzle align) driving both the CPU XLA reference and the
    # H100 FP8 op with the same non-uniform inputs.
    rng = np.random.default_rng(seed)
    lhs = jnp.asarray(rng.standard_normal((T, K)) * 0.1, jnp.bfloat16)
    rhs = jnp.asarray(rng.standard_normal((E, K, N)) * 0.1, jnp.bfloat16)
    group_sizes = jnp.asarray(rng.multinomial(T, np.ones(E) / E), jnp.int32)  # non-uniform, sums to T
    return lhs, rhs, group_sizes


def test_op_none_matches_xla_reference():
    """ragged_dot(op=None) default path computes a correct grouped matmul.

    Compares against jax.lax.ragged_dot_general (the canonical XLA reference)
    on a non-uniform group distribution to prove adding op= left the default
    path producing correct results.
    """
    lhs, rhs, gs = _fp8_inputs()
    out = ragged_dot(lhs, rhs, gs, op=None)
    ref = jax.lax.ragged_dot_general(
        lhs=lhs,
        rhs=rhs,
        group_sizes=gs,
        ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
            dot_dimension_numbers=(((1,), (1,)), ((), ())),
            lhs_ragged_dimensions=(0,),
            rhs_group_dimensions=(0,),
        ),
    )
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=2e-2, atol=2e-2)


@gpu_only
def test_op_routes_to_op_and_runs_end_to_end():
    # The FP8 op runs a genuine e4m3 wgmma forward, which only lowers on the H100.
    # Routing through op= executes that FP8 path end-to-end and matches the bf16
    # reference to the FP8 forward tolerance.
    lhs, rhs, gs = _fp8_inputs()
    op = Fp8RaggedDotOp.init()
    out = ragged_dot(lhs, rhs, gs, op=op)
    ref = ragged_dot(lhs, rhs, gs, op=None)
    assert out.shape == ref.shape
    out, ref = np.asarray(out, np.float32), np.asarray(ref, np.float32)
    assert np.linalg.norm(out - ref) / (np.linalg.norm(ref) + 1e-12) < 5e-2


def test_op_state_partitions_as_overwrite():
    """Fp8RaggedDotOp is OverwriteWithGradient: its scale/amax_history arrays
    go entirely to the overwrite partition and stay out of the optimizer gradient.

    Verifies the structural plumbing required for delayed-scaling state
    management via partition_for_grad_overwrite / apply_updates.
    """
    op = Fp8RaggedDotOp.init(amax_history_length=4)
    regular_param = jnp.array([1.0, 2.0])

    # Partition a tree containing both the op and a regular parameter.
    tree = {"op": op, "weight": regular_param}
    overwrites, grads = partition_for_grad_overwrite(tree)

    # The op (OverwriteWithGradient) lands in overwrites, not in grads.
    assert isinstance(overwrites["op"], Fp8RaggedDotOp)
    assert grads["op"] is None

    # Regular parameters are not overwritten; they stay in the grad/optimizer partition.
    assert overwrites["weight"] is None
    np.testing.assert_array_equal(np.asarray(grads["weight"]), np.asarray(regular_param))
