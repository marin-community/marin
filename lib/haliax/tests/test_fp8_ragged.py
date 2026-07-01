# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from haliax.nn.ragged_dot import ragged_dot
from haliax.quantization import Fp8RaggedDotOp, apply_updates, partition_for_grad_overwrite

gpu_only = pytest.mark.skipif(jax.default_backend() != "gpu", reason="fp8 wgmma only lowers on GPU")


def _rel_fro(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)


def _nonuniform(T, K, E, N, seed=0):
    # N is a multiple of 128 so the bf16 output store's TMA descriptor is
    # swizzle-aligned in the Mosaic ragged wgmma kernel.
    rng = np.random.default_rng(seed)
    lhs = jnp.asarray(rng.standard_normal((T, K)) * 0.1, jnp.bfloat16)
    rhs = jnp.asarray(rng.standard_normal((E, K, N)) * 0.1, jnp.bfloat16)
    # genuine non-uniform groups summing to T
    parts = rng.multinomial(T, np.ones(E) / E)
    return lhs, rhs, jnp.asarray(parts, jnp.int32)


@gpu_only
def test_fp8_forward_rel_fro():
    lhs, rhs, gs = _nonuniform(64, 128, 4, 128)
    op = Fp8RaggedDotOp.init()
    out = ragged_dot(lhs, rhs, gs, op=op)
    ref = ragged_dot(lhs, rhs, gs, op=None)
    assert _rel_fro(out, ref) < 5e-2


@gpu_only
def test_fp8_backward_matches_bf16():
    # The backward is bf16-exact: it differentiates the reference bf16
    # ragged_dot, so the operand gradients match bf16 ragged_dot's to roundoff.
    lhs, rhs, gs = _nonuniform(64, 128, 4, 128)
    op = Fp8RaggedDotOp.init()

    def loss(l, r, o):
        return ragged_dot(l, r, gs, op=o).astype(jnp.float32).sum()

    g_lhs_fp8, g_rhs_fp8 = jax.grad(lambda l, r: loss(l, r, op), argnums=(0, 1))(lhs, rhs)
    g_lhs_ref, g_rhs_ref = jax.grad(lambda l, r: loss(l, r, None), argnums=(0, 1))(lhs, rhs)

    assert _rel_fro(g_lhs_fp8, g_lhs_ref) < 1e-3
    assert _rel_fro(g_rhs_fp8, g_rhs_ref) < 1e-3


@gpu_only
def test_output_grad_scale_preserved_across_steps():
    """output_grad_scale must stay ≈ 1.0 (preserved), not collapse to 0.

    Before the fix, fp8_scaled_ragged_dot deleted grad_scale/grad_amax_history
    unused; JAX returned a zero cotangent for them, and apply_updates overwrote
    output_grad_scale with 0 on every step — a silent state-corruption hazard.
    After the fix, quantized_ragged_dot threads them as differentiable args and
    _qrd_bwd returns identity cotangents, so apply_updates leaves the value unchanged
    until the FP8-backward commit replaces them with updated scale/history.
    """
    lhs, rhs, gs = _nonuniform(64, 128, 4, 128)
    op = Fp8RaggedDotOp.init()

    def loss(op_, l, r):
        return ragged_dot(l, r, gs, op=op_).astype(jnp.float32).sum()

    def step(op_):
        grads = eqx.filter_grad(loss)(op_, lhs, rhs)
        overwrites, non_overwrites = partition_for_grad_overwrite(grads)
        updates = jax.tree_util.tree_map(lambda g: jnp.zeros_like(g), non_overwrites)
        return apply_updates(op_, updates, overwrites)

    op1 = step(op)
    op2 = step(op1)

    # output_grad_scale must be preserved at 1.0, not zeroed by apply_updates
    np.testing.assert_allclose(
        np.asarray(op2.output_grad_scale),
        np.ones_like(np.asarray(op2.output_grad_scale)),
        atol=1e-6,
        err_msg="output_grad_scale was zeroed after apply_updates (state corruption)",
    )
    # both input_scale and kernel_scale must have updated via in_q (non-trivially different from
    # 1.0 once the amax history accumulates non-zero values from the actual operand magnitudes);
    # require BOTH to move so a regression in either operand's in_q path is caught
    assert not np.allclose(np.asarray(op2.input_scale), 1.0, atol=1e-6) and not np.allclose(
        np.asarray(op2.kernel_scale), 1.0, atol=1e-6
    ), "both input_scale and kernel_scale should have updated from 1.0 via in_q delayed scaling"
