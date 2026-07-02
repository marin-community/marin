# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

gpu_only = pytest.mark.skipif(jax.default_backend() != "gpu", reason="CuTe fp8 GEMM only lowers on GPU")


def _rel_fro(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)


def _nonuniform_fp8(T, K, E, N, seed=0):
    rng = np.random.default_rng(seed)
    lhs = jnp.asarray(rng.standard_normal((T, K)) * 0.1, jnp.bfloat16)
    rhs = jnp.asarray(rng.standard_normal((E, K, N)) * 0.1, jnp.bfloat16)
    parts = rng.multinomial(T, np.ones(E) / E)
    return lhs, rhs, jnp.asarray(parts, jnp.int32)


@gpu_only
def test_cute_ragged_dot_matches_reference():
    from haliax._src.ragged_dot_cute import cute_ragged_dot  # noqa: PLC0415

    lhs, rhs, gs = _nonuniform_fp8(512, 256, 8, 256)
    E4M3 = jnp.float8_e4m3fn
    a = (lhs / jnp.max(jnp.abs(lhs)) * 448.0).astype(E4M3)  # [T,K] E4M3
    b = (jnp.swapaxes(rhs, 1, 2) / jnp.max(jnp.abs(rhs)) * 448.0).astype(E4M3)  # [E,N,K] E4M3
    out_scale = jnp.ones((1,), jnp.float32)  # dequant handled in the test's ref
    out = cute_ragged_dot(a, b, gs, out_dtype=jnp.bfloat16, out_scale=out_scale)
    # Reference: dequantize operands (no scaling here) and contract in f32, same layout.
    ref = jax.lax.ragged_dot(a.astype(jnp.float32), jnp.swapaxes(b, 1, 2).astype(jnp.float32), gs)
    assert _rel_fro(out, ref) < 5e-2


@gpu_only
def test_cute_ragged_dot_scale_divide():
    """Verify the epilogue DIVIDES (not multiplies) the accumulator by out_scale."""
    from haliax._src.ragged_dot_cute import cute_ragged_dot  # noqa: PLC0415

    lhs, rhs, gs = _nonuniform_fp8(512, 256, 8, 256, seed=1)
    E4M3 = jnp.float8_e4m3fn
    a = (lhs / jnp.max(jnp.abs(lhs)) * 448.0).astype(E4M3)
    b = (jnp.swapaxes(rhs, 1, 2) / jnp.max(jnp.abs(rhs)) * 448.0).astype(E4M3)
    scale_value = 2.0
    out_scale = jnp.full((1,), scale_value, jnp.float32)
    out_scaled = cute_ragged_dot(a, b, gs, out_dtype=jnp.bfloat16, out_scale=out_scale)
    # Reference: unscaled contract, then divide by scale_value.
    ref_unscaled = jax.lax.ragged_dot(a.astype(jnp.float32), jnp.swapaxes(b, 1, 2).astype(jnp.float32), gs)
    ref_scaled = ref_unscaled / scale_value
    # The divide path must match; a multiply would produce ref * scale^2 and fail.
    assert _rel_fro(out_scaled, ref_scaled) < 5e-2


def test_cute_ragged_dot_guard_non_conforming_shapes():
    """Guard raises ValueError for N/K shapes that don't meet tile alignment."""
    import importlib  # noqa: PLC0415

    # Patch cute_available to return True so we reach the guard without a GPU.
    ragged_dot_cute = importlib.import_module("haliax._src.ragged_dot_cute")
    original = ragged_dot_cute.cute_available

    def _fake_available():
        return True

    ragged_dot_cute.cute_available = _fake_available
    try:
        a = jnp.zeros((4, 128), jnp.float8_e4m3fn)
        gs = jnp.array([4], jnp.int32)
        # Bad N (not divisible by 256).
        b_bad_n = jnp.zeros((1, 100, 128), jnp.float8_e4m3fn)
        with pytest.raises(ValueError, match="N=100"):
            ragged_dot_cute.cute_ragged_dot(a, b_bad_n, gs, out_dtype=jnp.bfloat16, out_scale=jnp.ones((1,)))
        # Bad K (not divisible by 128).
        b_bad_k = jnp.zeros((1, 256, 64), jnp.float8_e4m3fn)
        with pytest.raises(ValueError, match="K=64"):
            ragged_dot_cute.cute_ragged_dot(a[:, :64], b_bad_k, gs, out_dtype=jnp.bfloat16, out_scale=jnp.ones((1,)))
    finally:
        ragged_dot_cute.cute_available = original


# ---------------------------------------------------------------------------
# FP8 forward + dgrad (E5M2 x E4M3), bf16 wgrad
#
# Shapes: N=256 (tile_n=256), K=256 so both the forward (b=[E,N,K]) and dgrad
# (b=[E,K,N]) passes satisfy the CuTe guard (N%256==0 and K%128==0 for each).
# ---------------------------------------------------------------------------


@gpu_only
def test_fp8_forward_rel_fro():
    """FP8 ragged_dot output should be within 5% of bf16 reference."""
    from haliax.nn.ragged_dot import ragged_dot  # noqa: PLC0415
    from haliax.quantization import Fp8RaggedDotOp  # noqa: PLC0415

    lhs, rhs, gs = _nonuniform_fp8(512, 256, 4, 256)
    op = Fp8RaggedDotOp.init(rev_dtype=jnp.float8_e5m2)
    out = ragged_dot(lhs, rhs, gs, op=op)
    ref = ragged_dot(lhs, rhs, gs, op=None)
    assert _rel_fro(out, ref) < 5e-2


@gpu_only
def test_fp8_dgrad_and_bf16_wgrad():
    """dlhs from FP8 dgrad (E5M2 x E4M3) within 8% of bf16; drhs (bf16 wgrad) < 0.1%."""
    from haliax.nn.ragged_dot import ragged_dot  # noqa: PLC0415
    from haliax.quantization import Fp8RaggedDotOp  # noqa: PLC0415

    lhs, rhs, gs = _nonuniform_fp8(512, 256, 4, 256)
    op = Fp8RaggedDotOp.init(rev_dtype=jnp.float8_e5m2)

    def loss(l, r, o):
        return ragged_dot(l, r, gs, op=o).astype(jnp.float32).sum()

    g_lhs_fp8, g_rhs_fp8 = jax.grad(lambda l, r: loss(l, r, op), argnums=(0, 1))(lhs, rhs)
    g_lhs_ref, g_rhs_ref = jax.grad(lambda l, r: loss(l, r, None), argnums=(0, 1))(lhs, rhs)
    assert _rel_fro(g_lhs_fp8, g_lhs_ref) < 8e-2  # fp8 dgrad (E5M2 x E4M3)
    assert _rel_fro(g_rhs_fp8, g_rhs_ref) < 1e-3  # bf16-exact wgrad


@gpu_only
def test_fp8_nonunit_scale_regression():
    """Regression for the output-scale inversion under a NON-unit applied scale.

    The CuTe epilogue DIVIDES the accumulator by ``out_scale``, while the raw fp8
    product ``q_a·q_b = (a/s_a)·(b/s_b) = (a·b)/(s_a·s_b)`` recovers ``a·b`` by
    MULTIPLYING by ``s_a·s_b``. So ``out_scale`` must be ``1/(s_a·s_b)`` (fwd) and
    ``1/(s_grad·s_rhs)`` (dgrad). With init scales=1 a single step applies scale 1
    and the inversion is invisible. Here we prime the amax histories so the applied
    scales are amax/448 ≪ 1: the buggy product-scale then over-scales by
    ``1/(s·s)^2`` and blows up the error, while the reciprocal fix stays in tol.
    """
    import dataclasses  # noqa: PLC0415

    from haliax.nn.ragged_dot import ragged_dot  # noqa: PLC0415
    from haliax.quantization import Fp8RaggedDotOp  # noqa: PLC0415

    lhs, rhs, gs = _nonuniform_fp8(512, 256, 4, 256)
    op = Fp8RaggedDotOp.init(amax_history_length=4, rev_dtype=jnp.float8_e5m2)

    # Prime the delayed-scaling state with the true per-tensor amax so the applied
    # forward scales are amax/448 ≪ 1 (a realistic post-warmup scale), and prime the
    # output-grad amax (loss=sum -> cotangent all-ones, amax 1) so the dgrad scale is
    # non-unit too. This forces both scale paths off their identity value.
    op = dataclasses.replace(
        op,
        input_amax_history=jnp.full_like(op.input_amax_history, jnp.max(jnp.abs(lhs)).astype(jnp.float32)),
        kernel_amax_history=jnp.full_like(op.kernel_amax_history, jnp.max(jnp.abs(rhs)).astype(jnp.float32)),
        output_grad_amax_history=jnp.ones_like(op.output_grad_amax_history),
    )

    def loss(l, r, o):
        return ragged_dot(l, r, gs, op=o).astype(jnp.float32).sum()

    out = ragged_dot(lhs, rhs, gs, op=op)
    ref = ragged_dot(lhs, rhs, gs, op=None)
    assert _rel_fro(out, ref) < 5e-2  # forward: reciprocal out_scale

    g_lhs_fp8 = jax.grad(lambda l: loss(l, rhs, op))(lhs)
    g_lhs_ref = jax.grad(lambda l: loss(l, rhs, None))(lhs)
    assert _rel_fro(g_lhs_fp8, g_lhs_ref) < 8e-2  # dgrad: reciprocal dgrad_scale


@gpu_only
def test_fp8_output_grad_amax_history_updates():
    """The output-grad amax history must roll on the backward (delayed scaling)."""
    from haliax.nn.ragged_dot import ragged_dot  # noqa: PLC0415
    from haliax.quantization import Fp8RaggedDotOp  # noqa: PLC0415

    lhs, rhs, gs = _nonuniform_fp8(512, 256, 4, 256)
    op = Fp8RaggedDotOp.init(amax_history_length=4, rev_dtype=jnp.float8_e5m2)

    def loss(o):
        return ragged_dot(lhs, rhs, gs, op=o).astype(jnp.float32).sum()

    grads = jax.grad(loss)(op)
    # OverwriteWithGradient returns the new state as the "gradient" of the op fields.
    assert not jnp.allclose(grads.output_grad_amax_history, op.output_grad_amax_history)
