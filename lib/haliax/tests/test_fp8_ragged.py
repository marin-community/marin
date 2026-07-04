# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from haliax.nn.ragged_dot import ragged_dot
from haliax.quantization import Fp8RaggedDotOp


def _on_hopper_gpu() -> bool:
    if jax.default_backend() != "gpu":
        return False
    return jax.devices()[0].compute_capability.startswith("9.")


# Mirrors fp8_scaled_ragged_dot's own precondition: the Mosaic wgmma kernels are
# sm_90a-specific, so on any non-Hopper GPU these tests must skip, not error.
hopper_only = pytest.mark.skipif(not _on_hopper_gpu(), reason="fp8 ragged wgmma kernels require Hopper (SM90)")


def _reference_ragged_dot(lhs, rhs, group_sizes):
    """Independent f32 reference: per-token expert gather + einsum.

    Deliberately shares no code with the in-repo Triton/Mosaic ragged kernels or
    with ``jax.lax.ragged_dot_general``, so a bug shared by the production
    kernels cannot cancel out of the comparison. Differentiable, so ``jax.grad``
    of it is the gradient reference as well.
    """
    ids = jnp.repeat(jnp.arange(group_sizes.shape[0]), group_sizes, total_repeat_length=lhs.shape[0])
    return jnp.einsum("tk,tkn->tn", lhs.astype(jnp.float32), rhs.astype(jnp.float32)[ids])


def _assert_close_fp8(actual, ref, rel_fro_tol, max_dev_tol, what):
    """FP8-vs-reference parity with an aggregate and a pointwise metric.

    The relative Frobenius norm bounds the overall quantization error; the max
    pointwise deviation (scaled by the reference's max magnitude) catches
    localized corruption -- e.g. a few bad tokens at a group boundary -- that a
    whole-tensor norm averages away.
    """
    a = np.asarray(actual, np.float32)
    r = np.asarray(ref, np.float32)
    rel_fro = np.linalg.norm(a - r) / (np.linalg.norm(r) + 1e-12)
    max_dev = np.max(np.abs(a - r)) / (np.max(np.abs(r)) + 1e-12)
    assert rel_fro < rel_fro_tol, f"{what}: relative Frobenius error {rel_fro:.4f} >= {rel_fro_tol}"
    assert max_dev < max_dev_tol, f"{what}: max pointwise deviation {max_dev:.4f} >= {max_dev_tol} of ref max"


def _nonuniform(T, K, E, N, seed=0):
    rng = np.random.default_rng(seed)
    lhs = jnp.asarray(rng.standard_normal((T, K)) * 0.1, jnp.bfloat16)
    rhs = jnp.asarray(rng.standard_normal((E, K, N)) * 0.1, jnp.bfloat16)
    # genuine non-uniform groups summing to T
    parts = rng.multinomial(T, np.ones(E) / E)
    return lhs, rhs, jnp.asarray(parts, jnp.int32)


@hopper_only
def test_fp8_forward_parity_vs_reference():
    lhs, rhs, gs = _nonuniform(128, 128, 4, 128)
    op = Fp8RaggedDotOp.init()
    out = ragged_dot(lhs, rhs, gs, op=op)
    ref = _reference_ragged_dot(lhs, rhs, gs)
    _assert_close_fp8(out, ref, rel_fro_tol=5e-2, max_dev_tol=0.15, what="FP8 forward")


@hopper_only
def test_fp8_forward_parity_with_empty_groups():
    # Zero-token experts: boundary case for the ragged group iteration. The
    # empty experts' weights must not contaminate neighboring groups' outputs.
    lhs, rhs, _ = _nonuniform(128, 128, 4, 128, seed=5)
    gs = jnp.asarray([0, 64, 0, 64], jnp.int32)
    op = Fp8RaggedDotOp.init()
    out = ragged_dot(lhs, rhs, gs, op=op)
    ref = _reference_ragged_dot(lhs, rhs, gs)
    _assert_close_fp8(out, ref, rel_fro_tol=5e-2, max_dev_tol=0.15, what="FP8 forward (empty groups)")


@hopper_only
def test_fp8_grads_parity_vs_reference():
    # The backward runs the exact bf16 Triton kernels on the saved residuals, so
    # both gradients match the f32 reference within bf16 rounding; the loose
    # FP8 tolerance keeps the assertion stable as the backward moves to FP8
    # GEMMs. The groups are non-uniform, including a small 5-token expert.
    lhs, rhs, _ = _nonuniform(128, 128, 4, 128, seed=3)
    gs = jnp.asarray([13, 5, 47, 63], jnp.int32)  # non-uniform, sums to T=128
    op = Fp8RaggedDotOp.init()

    def loss(l, r):
        return ragged_dot(l, r, gs, op=op).astype(jnp.float32).sum()

    def ref_loss(l, r):
        return _reference_ragged_dot(l, r, gs).sum()

    g_lhs_fp8, g_rhs_fp8 = jax.grad(loss, argnums=(0, 1))(lhs, rhs)
    g_lhs_ref, g_rhs_ref = jax.grad(ref_loss, argnums=(0, 1))(lhs, rhs)

    _assert_close_fp8(g_lhs_fp8, g_lhs_ref, rel_fro_tol=6e-2, max_dev_tol=0.15, what="grad_lhs")
    _assert_close_fp8(g_rhs_fp8, g_rhs_ref, rel_fro_tol=6e-2, max_dev_tol=0.15, what="grad_rhs")
