# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from haliax.nn.ragged_dot import ragged_dot
from haliax.quantization import Fp8RaggedDotOp, _jax_supports_mixed_fp8_wgmma, apply_updates, partition_for_grad_overwrite


def _on_hopper_gpu() -> bool:
    if jax.default_backend() != "gpu":
        return False
    return jax.devices()[0].compute_capability.startswith("9.")


# Mirrors fp8_scaled_ragged_dot's own precondition: the Mosaic wgmma kernels are
# sm_90a-specific, so on any non-Hopper GPU these tests must skip, not error.
hopper_only = pytest.mark.skipif(not _on_hopper_gpu(), reason="fp8 ragged wgmma kernels require Hopper (SM90)")

# The default mixed e5m2 x e4m3 backward lowers only on jax >= 0.11.0
# (jax-ml/jax#38859); Fp8RaggedDotOp.init with defaults fails fast on older jax.
mixed_wgmma = pytest.mark.skipif(
    not _jax_supports_mixed_fp8_wgmma(), reason="mixed e5m2 x e4m3 wgmma needs jax >= 0.11.0 (jax-ml/jax#38859)"
)


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
    # T, K, N are multiples of 128: fp8_scaled_ragged_dot's alignment contract
    # (wgrad token tile, dgrad/forward TMA swizzle).
    rng = np.random.default_rng(seed)
    lhs = jnp.asarray(rng.standard_normal((T, K)) * 0.1, jnp.bfloat16)
    rhs = jnp.asarray(rng.standard_normal((E, K, N)) * 0.1, jnp.bfloat16)
    # genuine non-uniform groups summing to T
    parts = rng.multinomial(T, np.ones(E) / E)
    return lhs, rhs, jnp.asarray(parts, jnp.int32)


@hopper_only
def test_fp8_forward_parity_vs_reference():
    lhs, rhs, gs = _nonuniform(128, 128, 4, 128)
    # rev_dtype only affects the backward; uniform e4m3 keeps the forward
    # parity tests runnable on jax 0.10.x Hopper pods.
    op = Fp8RaggedDotOp.init(rev_dtype=jnp.float8_e4m3fn)
    out = ragged_dot(lhs, rhs, gs, op=op)
    ref = _reference_ragged_dot(lhs, rhs, gs)
    _assert_close_fp8(out, ref, rel_fro_tol=5e-2, max_dev_tol=0.15, what="FP8 forward")


@hopper_only
def test_fp8_forward_parity_with_empty_groups():
    # Zero-token experts: boundary case for the ragged group iteration. The
    # empty experts' weights must not contaminate neighboring groups' outputs.
    lhs, rhs, _ = _nonuniform(128, 128, 4, 128, seed=5)
    gs = jnp.asarray([0, 64, 0, 64], jnp.int32)
    op = Fp8RaggedDotOp.init(rev_dtype=jnp.float8_e4m3fn)  # forward-only; see above
    out = ragged_dot(lhs, rhs, gs, op=op)
    ref = _reference_ragged_dot(lhs, rhs, gs)
    _assert_close_fp8(out, ref, rel_fro_tol=5e-2, max_dev_tol=0.15, what="FP8 forward (empty groups)")


@hopper_only
@mixed_wgmma
def test_fp8_grads_parity_vs_reference():
    # Both gradients run as FP8 GEMMs and are approximate vs the f32 reference
    # grad, held to the 6e-2 FP8 tolerance. T is a multiple of 128 (the wgrad's
    # token-dim TMA tile); the groups are non-uniform, including a small
    # 5-token expert.
    lhs, rhs, _ = _nonuniform(128, 128, 4, 128, seed=3)
    gs = jnp.asarray([13, 5, 47, 63], jnp.int32)  # non-uniform, sums to T=128
    op = Fp8RaggedDotOp.init()  # rev_dtype defaults to e5m2 (mixed backward)

    def loss(l, r):
        return ragged_dot(l, r, gs, op=op).astype(jnp.float32).sum()

    def ref_loss(l, r):
        return _reference_ragged_dot(l, r, gs).sum()

    g_lhs_fp8, g_rhs_fp8 = jax.grad(loss, argnums=(0, 1))(lhs, rhs)
    g_lhs_ref, g_rhs_ref = jax.grad(ref_loss, argnums=(0, 1))(lhs, rhs)

    _assert_close_fp8(g_lhs_fp8, g_lhs_ref, rel_fro_tol=6e-2, max_dev_tol=0.15, what="FP8 dgrad grad_lhs")
    _assert_close_fp8(g_rhs_fp8, g_rhs_ref, rel_fro_tol=6e-2, max_dev_tol=0.15, what="FP8 wgrad grad_rhs")


@hopper_only
@mixed_wgmma
def test_fp8_grad_rhs_parity_incl_boundaries():
    # The weight gradient (grad_rhs) contracts the ragged token dim in block_k=128
    # tiles via mgpu_dwgrad. This exercises the f16-upcast group-boundary mask:
    # the token groups are chosen so boundaries fall mid-tile and one group
    # straddles the 128-token block boundary (start 100, end 180). T is a multiple
    # of 128 (the wgrad's contracting-dim TMA tile), K and N multiples of 128
    # (the dgrad/forward alignment).
    lhs, rhs, _ = _nonuniform(256, 128, 4, 128, seed=4)
    gs = jnp.asarray([60, 40, 80, 76], jnp.int32)  # boundaries 60, 100, 180 (mid-block_k)
    assert int(gs.sum()) == 256
    op = Fp8RaggedDotOp.init()

    def loss(w):
        return ragged_dot(lhs, w, gs, op=op).astype(jnp.float32).sum()

    def ref_loss(w):
        return _reference_ragged_dot(lhs, w, gs).sum()

    g_rhs_fp8 = jax.grad(loss)(rhs)
    g_rhs_ref = jax.grad(ref_loss)(rhs)

    _assert_close_fp8(
        g_rhs_fp8, g_rhs_ref, rel_fro_tol=6e-2, max_dev_tol=0.15, what="FP8 wgrad grad_rhs at mid-tile boundaries"
    )


@hopper_only
@mixed_wgmma
def test_output_grad_scale_updates_across_steps():
    """The backward's delayed scaling updates output_grad_scale from gradient magnitudes.

    The backward returns the rolled grad_scale/grad_amax_history as
    OverwriteWithGradient cotangents, so after steps with non-zero gradients all
    three scales move away from 1.0 and none are zeroed.
    """
    lhs, rhs, gs = _nonuniform(128, 128, 4, 128)
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

    # output_grad_scale must not be zeroed (the delayed-scaling state stays valid).
    assert not np.allclose(np.asarray(op2.output_grad_scale), 0.0), "output_grad_scale was zeroed (state corruption)"
    # input_scale, kernel_scale, and output_grad_scale must all have moved away from
    # 1.0 — the input/kernel via in_q_ct, the output gradient via the FP8 dgrad's delayed
    # scaling. Require all three so a regression in any delayed-scaling path is caught.
    assert (
        not np.allclose(np.asarray(op2.input_scale), 1.0, atol=1e-6)
        and not np.allclose(np.asarray(op2.kernel_scale), 1.0, atol=1e-6)
        and not np.allclose(np.asarray(op2.output_grad_scale), 1.0, atol=1e-6)
    ), "input_scale, kernel_scale, and output_grad_scale should all update from 1.0 via delayed scaling"


@pytest.mark.parametrize(
    "t,k,n,match",
    [
        (192, 128, 128, "T=192"),  # multiple of 64 (forward tile) but not of 128 (wgrad tile)
        (128, 192, 128, "K=192"),
        (128, 128, 192, "N=192"),
    ],
)
def test_fp8_misaligned_shape_fails_fast(t, k, n, match):
    """Misaligned shapes raise ValueError at call time, naming the offending dim.

    Regression: the op= path bypasses ragged_dot's 512-row padding, and the
    backward's alignment constraints are stricter than the forward's, so without
    up-front validation a forward-only run would succeed and the first jax.grad
    would crash at trace time. Runs on all backends: the shape contract is
    checked before the Hopper device check.
    """
    lhs = jnp.zeros((t, k), jnp.bfloat16)
    rhs = jnp.zeros((4, k, n), jnp.bfloat16)
    gs = jnp.asarray([t // 4] * 4, jnp.int32)
    # Uniform rev_dtype so the op constructs on jax 0.10.x (incl. CPU CI); the
    # shape contract under test is independent of the backward dtype recipe.
    with pytest.raises(ValueError, match=match):
        ragged_dot(lhs, rhs, gs, op=Fp8RaggedDotOp.init(rev_dtype=jnp.float8_e4m3fn))
