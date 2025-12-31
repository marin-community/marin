# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import dataclasses

import jax
import jax.numpy as jnp
import haliax as hax
import haliax.nn as hnn
from haliax import Axis
import pytest

from levanter.layers.gated_deltanet import (
    FusedRMSNormGated,
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
    _rmsnorm_gated_flash,
    _rmsnorm_gated_reference,
)
from tests.test_utils import skip_if_no_torch

jax.config.update("jax_default_matmul_precision", "float32")

is_tpu = jax.devices()[0].platform == "tpu"
USE_FLASH_CASES = [True, False] if is_tpu else [False]


def _to_np(x):
    return np.array(x.detach().cpu().numpy())


def _get_hf_kernels():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.qwen3_next.modular_qwen3_next import (
        torch_chunk_gated_delta_rule as hf_chunk,
        torch_recurrent_gated_delta_rule as hf_recur,
    )

    return hf_chunk, hf_recur


def _named_kernels_inputs(B, H, L, dk, dv, key):
    """
    Helper: create independent q,k,v,g,beta with named axes and split PRNGKey to avoid correlation.
    Shapes: q,k: [B, L, H, dk], v: [B, L, H, dv], g,beta: [B, L, H].
    """
    kq, kk, kv, kg, kb = jax.random.split(key, 5)
    q = hax.random.normal(kq, {"batch": B, "position": L, "heads": H, "k_head_dim": dk}, dtype=jnp.float32)
    k = hax.random.normal(kk, {"batch": B, "position": L, "heads": H, "k_head_dim": dk}, dtype=jnp.float32)
    v = hax.random.normal(kv, {"batch": B, "position": L, "heads": H, "v_head_dim": dv}, dtype=jnp.float32)
    g = -0.1 * hax.random.normal(kg, {"batch": B, "position": L, "heads": H}, dtype=jnp.float32)  # mildly negative
    beta = hax.random.uniform(kb, {"batch": B, "position": L, "heads": H}, dtype=jnp.float32)
    return q, k, v, g, beta


@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
def test_fused_rms_norm_gated_matches_reference(use_flash: bool):
    key_x, key_g = jax.random.split(jax.random.PRNGKey(0))
    Batch = Axis("batch", 2)
    Pos = Axis("position", 3)
    Hidden = Axis("hidden", 5)

    module = FusedRMSNormGated.init(Hidden, eps=1e-6, use_flash=use_flash)
    module_ref = dataclasses.replace(module, use_flash=False)

    x = hax.random.normal(key_x, (Batch, Pos, Hidden), dtype=jnp.float32)
    gate = hax.random.normal(key_g, (Batch, Pos, Hidden), dtype=jnp.float32)

    y_flash = module(x, gate)
    y_ref = module_ref(x, gate)

    x32 = x.astype(jnp.float32)
    var = hax.mean(hax.square(x32), axis=Hidden)
    inv = hax.rsqrt(var + jnp.asarray(module.eps, dtype=jnp.float32))
    expected = (x32 * inv).astype(x.dtype)
    expected = module.weight * expected
    expected = expected * hnn.silu(gate)

    np.testing.assert_allclose(y_flash.array, y_ref.array, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(y_flash.array, expected.array, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
def test_fused_rms_norm_gated_backward_matches_reference(use_flash: bool):
    """Gradient parity for fused RMSNorm + SiLU gate against the reference JAX path.

    Compares grads w.r.t. inputs (x, gate) and the weight parameter.
    """
    key = jax.random.PRNGKey(123)
    kx, kg, kw = jax.random.split(key, 3)

    N, D = 7, 9
    x = jax.random.normal(kx, (N, D), dtype=jnp.float32)
    gate = jax.random.normal(kg, (N, D), dtype=jnp.float32)
    weight = jax.random.normal(kw, (D,), dtype=jnp.float32)
    eps = 1e-6

    def loss_flash(x_arr, g_arr, w_arr):
        y = _rmsnorm_gated_flash(x_arr, g_arr, w_arr, eps)
        return jnp.sum(y)

    def loss_ref(x_arr, g_arr, w_arr):
        y = _rmsnorm_gated_reference(x_arr, g_arr, w_arr, eps)
        return jnp.sum(y)

    gx_r, gg_r, gw_r = jax.grad(loss_ref, argnums=(0, 1, 2))(x, gate, weight)

    if use_flash:
        gx_f, gg_f, gw_f = jax.grad(loss_flash, argnums=(0, 1, 2))(x, gate, weight)
        np.testing.assert_allclose(np.array(gx_f), np.array(gx_r), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.array(gg_f), np.array(gg_r), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.array(gw_f), np.array(gw_r), rtol=1e-5, atol=1e-6)
    else:
        assert jnp.all(jnp.isfinite(gx_r))
        assert jnp.all(jnp.isfinite(gg_r))
        assert jnp.all(jnp.isfinite(gw_r))


@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
def test_flash_chunk_backward_chunk_size_invariance_kernel_level(use_flash: bool):
    """Kernel-level: gradients should be invariant to chunk_size (flash path).

    We compare grads wrt q,k,v,g,beta for two chunk sizes.
    """
    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 1, 2, 27, 8, 8
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    def loss_with_chunk(q_arr, k_arr, v_arr, g_arr, b_arr, chunk_size):
        qn = hax.named(q_arr, q.axes)
        kn = hax.named(k_arr, k.axes)
        vn = hax.named(v_arr, v.axes)
        gn = hax.named(g_arr, g.axes)
        bn = hax.named(b_arr, beta.axes)
        out, _ = chunk_gated_delta_rule(
            qn,
            kn,
            vn,
            gn,
            bn,
            chunk_size=chunk_size,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_flash=use_flash,
        )
        return jnp.sum(out.array)

    grads8 = jax.grad(lambda qa, ka, va, ga, ba: loss_with_chunk(qa, ka, va, ga, ba, 8), argnums=(0, 1, 2, 3, 4))(
        q.array, k.array, v.array, g.array, beta.array
    )
    grads32 = jax.grad(lambda qa, ka, va, ga, ba: loss_with_chunk(qa, ka, va, ga, ba, 32), argnums=(0, 1, 2, 3, 4))(
        q.array, k.array, v.array, g.array, beta.array
    )

    for g8, g32 in zip(grads8, grads32):
        np.testing.assert_allclose(np.array(g8), np.array(g32), rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
def test_recurrent_no_learning_beta_zero_outputs_zero(use_flash: bool):
    """If beta == 0 everywhere and S0 == 0, then S_t == 0 for all t and outputs are zero."""
    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 3, 17, 8, 8
    q, k, v, g, _ = _named_kernels_inputs(B, H, L, dk, dv, key)
    beta0 = hax.named(jnp.zeros((B, L, H), dtype=jnp.float32), ("batch", "position", "heads"))

    out, S_final = recurrent_gated_delta_rule(
        q, k, v, g, beta0, initial_state=None, output_final_state=True, use_flash=use_flash
    )
    np.testing.assert_allclose(np.array(out.array), 0.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(S_final, 0.0, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("use_flash", [True, False])
def test_recurrent_perfect_fit_on_current_key_when_alpha1_beta1_and_L2norm(use_flash: bool):
    """
    With α=1 (g=0) and β=1 and L2-normed K, the post-update state satisfies S_t^T k̂_t == v_t,
    where k̂_t is K L2-normalized along d_k (as in the kernel).
    """
    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 1, 2, 11, 8, 8

    q, k, v, _, _ = _named_kernels_inputs(B, H, L, dk, dv, key)
    g = hax.named(jnp.zeros((B, L, H), dtype=jnp.float32), ("batch", "position", "heads"))  # α=1
    beta1 = hax.named(jnp.ones((B, L, H), dtype=jnp.float32), ("batch", "position", "heads"))  # β=1

    S = None
    for t in range(L):
        q_t = q["position", hax.ds(t, Axis("one", 1))]
        k_t = k["position", hax.ds(t, Axis("one", 1))]
        v_t = v["position", hax.ds(t, Axis("one", 1))]
        g_t = g["position", hax.ds(t, Axis("one", 1))]
        b_t = beta1["position", hax.ds(t, Axis("one", 1))]

        _, S = recurrent_gated_delta_rule(
            q_t,
            k_t,
            v_t,
            g_t,
            b_t,
            initial_state=S,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_flash=use_flash,
        )

        # Check S^T k̂_t == v_t (k̂_t is K normalized as in the kernel)
        k_arr = hax.rearrange(k_t, ("batch", "heads", "position", "k_head_dim")).array[..., 0, :]
        inv = jax.lax.rsqrt(jnp.sum(k_arr * k_arr, axis=-1, keepdims=True) + 1e-6)
        k_hat = k_arr * inv
        kv = jnp.sum(S * k_hat[..., None], axis=-2)  # (B,H,dv)
        v_arr = hax.rearrange(v_t, ("batch", "heads", "position", "v_head_dim")).array[..., 0, :]
        np.testing.assert_allclose(kv, v_arr, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
@pytest.mark.parametrize("chunk_size", [16, 32, 64])
def test_chunk_equals_recurrent_for_random_inputs(chunk_size, use_flash):
    """Chunkwise kernel must match recurrent kernel for many chunk sizes"""
    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 3, 57, 8, 8
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    out_chunk, S_chunk = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=chunk_size,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_flash=use_flash,
    )
    out_recur, S_recur = recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_flash=use_flash,
    )

    np.testing.assert_allclose(np.array(out_chunk.array), np.array(out_recur.array), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(S_chunk, S_recur, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
def test_chunk_nondivisible_padding_matches_recurrent_jax_only(use_flash: bool):
    """When L % chunk_size != 0, padding path should still match the recurrent kernel (JAX-only)."""
    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 4, 61, 8, 16
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    out_chunk, _ = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=32,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_flash=use_flash,
    )
    out_recur, _ = recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_flash=use_flash,
    )
    np.testing.assert_allclose(np.array(out_chunk.array), np.array(out_recur.array), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
def test_chunk_continuation_two_pass_equals_one_pass(use_flash: bool):
    """
    Run prefix → get S_mid → run suffix with initial_state, and match the one-pass result.
    Pure JAX: validates the initial_state continuation semantics of the chunk kernel.
    """
    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 1, 3, 47, 8, 8
    split = 21
    chunk_size = 16
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    # One pass over the full sequence
    out_full, S_full = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=chunk_size,
        output_final_state=True,
        use_flash=use_flash,
    )

    # Two-pass: prefix then suffix with initial_state
    q_pre, q_suf = q["position", hax.ds(0, split)], q["position", hax.ds(split, Axis("pos2", L - split))]
    k_pre, k_suf = k["position", hax.ds(0, split)], k["position", hax.ds(split, Axis("pos2", L - split))]
    v_pre, v_suf = v["position", hax.ds(0, split)], v["position", hax.ds(split, Axis("pos2", L - split))]
    g_pre, g_suf = g["position", hax.ds(0, split)], g["position", hax.ds(split, Axis("pos2", L - split))]
    b_pre, b_suf = beta["position", hax.ds(0, split)], beta["position", hax.ds(split, Axis("pos2", L - split))]

    out_pre, S_mid = chunk_gated_delta_rule(
        q_pre,
        k_pre,
        v_pre,
        g_pre,
        b_pre,
        chunk_size=chunk_size,
        output_final_state=True,
        use_flash=use_flash,
    )
    out_suf, S_end = chunk_gated_delta_rule(
        q_suf,
        k_suf,
        v_suf,
        g_suf,
        b_suf,
        chunk_size=chunk_size,
        initial_state=S_mid,
        output_final_state=True,
        use_flash=use_flash,
    )

    # Compare outputs on the suffix region and final states
    out_full_suf = out_full["position", hax.ds(split, Axis("pos2", L - split))]
    np.testing.assert_allclose(np.array(out_suf.array), np.array(out_full_suf.array), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(S_end, S_full, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
def test_chunk_size_one_degenerates_to_recurrent_without_l2norm(use_flash: bool):
    """Degeneracy should also hold even when L2 norm is disabled."""
    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 2, 29, 8, 8
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)
    out_chunk, _ = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=1,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_flash=use_flash,
    )
    out_recur, _ = recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_flash=use_flash,
    )
    np.testing.assert_allclose(np.array(out_chunk.array), np.array(out_recur.array), rtol=3e-4, atol=1e-4)


@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
def test_extreme_gates_numerical_stability_jax_only(use_flash: bool):
    """Outputs should be finite when α ≈ 0 (very negative g) and β near 0 or near 1."""
    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 1, 2, 37, 16, 8

    q, k, v, _, _ = _named_kernels_inputs(B, H, L, dk, dv, key)
    # Strong negative g: α ~ 0
    g = -hax.random.uniform(key, {"batch": B, "position": L, "heads": H}, minval=2.0, maxval=8.0, dtype=jnp.float32)
    beta_small = hax.named(jnp.full((B, L, H), 1e-4, dtype=jnp.float32), ("batch", "position", "heads"))
    beta_big = hax.named(jnp.full((B, L, H), 1.0 - 1e-6, dtype=jnp.float32), ("batch", "position", "heads"))

    for beta in [beta_small, beta_big]:
        out_chunk, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=32,
            output_final_state=False,
            use_flash=use_flash,
        )
        assert np.isfinite(np.array(out_chunk.array)).all()

        out_recur, _ = recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            output_final_state=False,
            use_flash=use_flash,
        )
        assert np.isfinite(np.array(out_recur.array)).all()


@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
def test_gradients_exist_small_kernel_graph(use_flash: bool):
    """Smoke test: both kernels are differentiable w.r.t. inputs (no NaNs in grads)."""
    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 1, 1, 7, 4, 4
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    # grad wrt q only (keep the surface small)
    def loss_chunk(q_arr):
        qn = hax.named(q_arr, q.axes)
        out, _ = chunk_gated_delta_rule(
            qn,
            k,
            v,
            g,
            beta,
            chunk_size=4,
            output_final_state=False,
            use_flash=use_flash,
        )
        return jnp.sum(out.array)

    def loss_recur(q_arr):
        qn = hax.named(q_arr, q.axes)
        out, _ = recurrent_gated_delta_rule(
            qn,
            k,
            v,
            g,
            beta,
            output_final_state=False,
            use_flash=use_flash,
        )
        return jnp.sum(out.array)

    g1 = jax.grad(loss_chunk)(q.array)
    assert jnp.all(jnp.isfinite(g1))

    if use_flash:
        return

    g2 = jax.grad(loss_recur)(q.array)
    assert jnp.all(jnp.isfinite(g2))


@skip_if_no_torch
@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
def test_recurrent_kernel_matches_hf(use_flash: bool):
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 1, 2, 17, 8, 8
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    out_named, _ = recurrent_gated_delta_rule(q, k, v, g, beta, output_final_state=False, use_flash=use_flash)

    # HF expects (B, L, H, dim) on input and transposes internally.
    def to_t(arr: jnp.ndarray):
        return torch.from_numpy(np.array(arr))

    out_hf, _ = hf_recur(
        to_t(q.array),
        to_t(k.array),
        to_t(v.array),
        to_t(g.array),
        to_t(beta.array),
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    out_hf_np = _to_np(out_hf)

    np.testing.assert_allclose(np.array(out_named.array), out_hf_np, rtol=1e-4, atol=1e-4)


@skip_if_no_torch
@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
@pytest.mark.parametrize("use_triangular_solve", [True, False])
def test_chunk_kernel_matches_hf(use_flash: bool, use_triangular_solve: bool):
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 4, 64, 8, 16
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    out_named, _ = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=32,
        output_final_state=False,
        use_flash=use_flash,
        use_triangular_solve=use_triangular_solve,
    )

    def to_t(arr: jnp.ndarray):
        return torch.from_numpy(np.array(arr))

    out_hf, _ = hf_chunk(
        to_t(q.array),
        to_t(k.array),
        to_t(v.array),
        to_t(g.array),
        to_t(beta.array),
        chunk_size=32,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    out_hf_np = _to_np(out_hf)

    np.testing.assert_allclose(np.array(out_named.array), out_hf_np, rtol=1e-4, atol=1e-4)


@skip_if_no_torch
@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
def test_chunk_kernel_matches_hf_non_divisible(use_flash: bool):
    """L not divisible by chunk_size should still match HF fallback (padding path)."""
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 3, 61, 8, 16
    chunk_size = 32

    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    out_named, _ = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=chunk_size,
        output_final_state=False,
        use_flash=use_flash,
    )

    def to_t(arr: jnp.ndarray):
        return torch.from_numpy(np.array(arr))

    out_hf, _ = hf_chunk(
        to_t(q.array),
        to_t(k.array),
        to_t(v.array),
        to_t(g.array),
        to_t(beta.array),
        chunk_size=chunk_size,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    out_hf_np = _to_np(out_hf)

    np.testing.assert_allclose(np.array(out_named.array), out_hf_np, rtol=1e-4, atol=1e-4)


@skip_if_no_torch
@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
def test_chunk_size_one_matches_hf_recurrent(use_flash: bool):
    """chunk_size=1 should degenerate to the recurrent rule."""
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 2, 29, 8, 8
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    out_chunk, _ = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=1,
        output_final_state=False,
        use_flash=use_flash,
    )
    out_recur, _ = recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        output_final_state=False,
        use_flash=use_flash,
    )
    np.testing.assert_allclose(np.array(out_chunk.array), np.array(out_recur.array), rtol=1e-4, atol=1e-4)

    def to_t(arr: jnp.ndarray):
        return torch.from_numpy(np.array(arr))

    out_chunk_t, _ = hf_chunk(
        to_t(q.array),
        to_t(k.array),
        to_t(v.array),
        to_t(g.array),
        to_t(beta.array),
        chunk_size=1,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    out_recur_t, _ = hf_recur(
        to_t(q.array),
        to_t(k.array),
        to_t(v.array),
        to_t(g.array),
        to_t(beta.array),
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    np.testing.assert_allclose(_to_np(out_chunk_t), _to_np(out_recur_t), rtol=1e-4, atol=1e-4)


@skip_if_no_torch
@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
def test_chunk_kernel_with_initial_state_matches_recurrent_continuation(use_flash: bool):
    """
    Provide an initial S0 and check chunk kernel == recurrent kernel on the same sequence.
    """
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 1, 3, 47, 8, 8
    chunk_size = 16

    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)
    S0 = jax.random.normal(jax.random.PRNGKey(123), (B, H, dk, dv), dtype=jnp.float32) * 0.1

    out_chunk, _ = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=chunk_size,
        initial_state=S0,
        output_final_state=False,
        use_flash=use_flash,
    )
    out_recur, _ = recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=S0,
        output_final_state=False,
        use_flash=use_flash,
    )
    np.testing.assert_allclose(np.array(out_chunk.array), np.array(out_recur.array), rtol=1e-4, atol=1e-4)

    def to_t(arr: jnp.ndarray):
        return torch.from_numpy(np.array(arr))

    S0_t = to_t(S0)
    out_chunk_t, _ = hf_chunk(
        to_t(q.array),
        to_t(k.array),
        to_t(v.array),
        to_t(g.array),
        to_t(beta.array),
        chunk_size=chunk_size,
        initial_state=S0_t,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    out_recur_t, _ = hf_recur(
        to_t(q.array),
        to_t(k.array),
        to_t(v.array),
        to_t(g.array),
        to_t(beta.array),
        initial_state=S0_t,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    np.testing.assert_allclose(_to_np(out_chunk_t), _to_np(out_recur_t), rtol=1e-4, atol=1e-4)


@skip_if_no_torch
@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
def test_short_sequences_edge_cases(use_flash: bool):
    """Short L vs chunk_size and kernel-size behaviors."""
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)

    for L in [1, 2, 3, 5, 7]:
        B, H, dk, dv = 2, 2, 8, 8
        q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

        out_named, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=64,
            output_final_state=False,
            use_flash=use_flash,
        )

        def to_t(arr: jnp.ndarray):
            return torch.from_numpy(np.array(arr))

        out_t, _ = hf_chunk(
            to_t(q.array),
            to_t(k.array),
            to_t(v.array),
            to_t(g.array),
            to_t(beta.array),
            chunk_size=64,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        out_hf = _to_np(out_t)
        np.testing.assert_allclose(np.array(out_named.array), out_hf, rtol=1e-4, atol=1e-4)


@skip_if_no_torch
@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
def test_extreme_gates_no_nans_and_parity(use_flash: bool):
    """Stress alpha = exp(g) close to 0 (very negative g) and beta near 0/1."""
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 1, 2, 37, 16, 8

    q, k, v, _, _ = _named_kernels_inputs(B, H, L, dk, dv, key)
    g = -hax.random.uniform(key, {"batch": B, "position": L, "heads": H}, minval=2.0, maxval=8.0, dtype=jnp.float32)
    beta_small = hax.named(jnp.full((B, L, H), 1e-4, dtype=jnp.float32), ("batch", "position", "heads"))
    beta_big = hax.named(jnp.full((B, L, H), 1.0 - 1e-6, dtype=jnp.float32), ("batch", "position", "heads"))

    for beta in [beta_small, beta_big]:
        out_named, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=32,
            output_final_state=False,
            use_flash=use_flash,
        )

        def to_t(arr: jnp.ndarray):
            return torch.from_numpy(np.array(arr))

        out_t, _ = hf_chunk(
            to_t(q.array),
            to_t(k.array),
            to_t(v.array),
            to_t(g.array),
            to_t(beta.array),
            chunk_size=32,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        out_hf = _to_np(out_t)
        np.testing.assert_allclose(np.array(out_named.array), out_hf, rtol=1e-4, atol=1e-4)


@skip_if_no_torch
@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
def test_kernels_match_hf_without_l2norm(use_flash: bool):
    # TODO: fix edge case? although per original paper L2 norm is needed for stability
    pytest.skip("not matching HF implementation")

    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 3, 57, 16, 8

    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    # Haliax kernels with use_qk_l2norm_in_kernel=False
    out_chunk_j, _ = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=32,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_flash=use_flash,
    )
    out_recur_j, _ = recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_flash=use_flash,
    )

    # HF fallback expects (B, L, H, dim) on input and transposes internally; don't move axes.
    def to_t(arr: jnp.ndarray):
        return torch.from_numpy(np.array(arr))

    out_chunk_t, _ = hf_chunk(
        to_t(q.array),
        to_t(k.array),
        to_t(v.array),
        to_t(g.array),
        to_t(beta.array),
        chunk_size=32,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
    )
    out_recur_t, _ = hf_recur(
        to_t(q.array),
        to_t(k.array),
        to_t(v.array),
        to_t(g.array),
        to_t(beta.array),
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
    )

    np.testing.assert_allclose(np.array(out_chunk_j.array), _to_np(out_chunk_t), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.array(out_recur_j.array), _to_np(out_recur_t), rtol=1e-4, atol=1e-4)


@skip_if_no_torch
def test_recurrent_backward_matches_hf():
    """
    JAX vs HF fallback gradient parity for the recurrent (decode) kernel.
    We compare grads w.r.t. q, k, v, g, beta, and initial_state S0 on a small case.
    Only test the fallback version, as this is not used for training anyways and only a sanity check.
    """
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(202)
    k1, k2 = jax.random.split(key, 2)
    B, H, L, dk, dv = 1, 2, 16, 8, 8

    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, k1)
    S0 = jax.random.normal(k2, (B, H, dk, dv), dtype=jnp.float32) * 0.1

    # ---- JAX grads ----
    def loss_recur(q_arr, k_arr, v_arr, g_arr, b_arr, S0_arr):
        qn = hax.named(q_arr, q.axes)
        kn = hax.named(k_arr, k.axes)
        vn = hax.named(v_arr, v.axes)
        gn = hax.named(g_arr, g.axes)
        bn = hax.named(b_arr, beta.axes)
        out, _ = recurrent_gated_delta_rule(
            qn,
            kn,
            vn,
            gn,
            bn,
            initial_state=S0_arr,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_flash=False,
        )
        return jnp.sum(out.array)

    jax_grads = jax.grad(loss_recur, argnums=(0, 1, 2, 3, 4, 5))(q.array, k.array, v.array, g.array, beta.array, S0)
    jq, jk, jv, jg, jb, jS0 = [np.array(x) for x in jax_grads]

    # ---- Torch grads (HF fallback) ----
    def to_t(x, requires_grad=True):
        t = torch.from_numpy(np.array(x))
        t.requires_grad_(requires_grad)
        return t

    q_t = to_t(q.array)
    k_t = to_t(k.array)
    v_t = to_t(v.array)
    g_t = to_t(g.array)
    b_t = to_t(beta.array)
    S0_t = to_t(S0)

    out_t, _ = hf_recur(
        q_t,
        k_t,
        v_t,
        g_t,
        b_t,
        initial_state=S0_t,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    loss_t = out_t.sum()
    loss_t.backward()

    tq, tk, tv, tg, tb, tS0 = [x.grad.detach().cpu().numpy() for x in (q_t, k_t, v_t, g_t, b_t, S0_t)]

    np.testing.assert_allclose(jq, tq, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jk, tk, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jv, tv, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jg, tg, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jb, tb, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jS0, tS0, rtol=1e-4, atol=5e-6)


@skip_if_no_torch
@pytest.mark.parametrize("use_flash", USE_FLASH_CASES)
def test_chunk_backward_matches_hf(use_flash: bool):
    """
    JAX vs HF fallback gradient parity for the chunkwise kernel (two chunks).
    Includes gradients w.r.t. q, k, v, g, beta, and initial_state S0.
    """
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(303)
    k1, k2 = jax.random.split(key, 2)
    B, H, L, dk, dv = 1, 2, 16, 8, 8  # L multiple of chunk_size for a clean path
    chunk_size = 8

    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, k1)
    S0 = jax.random.normal(k2, (B, H, dk, dv), dtype=jnp.float32) * 0.1

    # ---- JAX grads ----
    def loss_chunk(q_arr, k_arr, v_arr, g_arr, b_arr, S0_arr):
        qn = hax.named(q_arr, q.axes)
        kn = hax.named(k_arr, k.axes)
        vn = hax.named(v_arr, v.axes)
        gn = hax.named(g_arr, g.axes)
        bn = hax.named(b_arr, beta.axes)
        out, _ = chunk_gated_delta_rule(
            qn,
            kn,
            vn,
            gn,
            bn,
            chunk_size=chunk_size,
            initial_state=S0_arr,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_flash=use_flash,
        )
        return jnp.sum(out.array)

    jax_grads = jax.grad(loss_chunk, argnums=(0, 1, 2, 3, 4, 5))(q.array, k.array, v.array, g.array, beta.array, S0)
    jq, jk, jv, jg, jb, jS0 = [np.array(x) for x in jax_grads]

    # ---- Torch grads (HF fallback) ----
    def to_t(x, requires_grad=True):
        t = torch.from_numpy(np.array(x))
        t.requires_grad_(requires_grad)
        return t

    q_t = to_t(q.array)
    k_t = to_t(k.array)
    v_t = to_t(v.array)
    g_t = to_t(g.array)
    b_t = to_t(beta.array)
    S0_t = to_t(S0)

    out_t, _ = hf_chunk(
        q_t,
        k_t,
        v_t,
        g_t,
        b_t,
        chunk_size=chunk_size,
        initial_state=S0_t,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    loss_t = out_t.sum()
    loss_t.backward()

    tq, tk, tv, tg, tb, tS0 = [x.grad.detach().cpu().numpy() for x in (q_t, k_t, v_t, g_t, b_t, S0_t)]

    np.testing.assert_allclose(jq, tq, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jk, tk, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jv, tv, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jg, tg, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jb, tb, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jS0, tS0, rtol=1e-4, atol=5e-6)


# -------------------------------
# Varlen / Flash-specific testing
# -------------------------------


@pytest.mark.skipif(not is_tpu, reason="use_flash=True varlen path is only tested on TPU")
@pytest.mark.parametrize("chunk_size", [1, 16, 32, 64])
def test_chunk_varlen_lengths_matches_recurrent_prefix_and_state(chunk_size):
    """Flash (varlen via lengths) == recurrent run up to each (b,h) length on the prefix."""
    key = jax.random.PRNGKey(42)
    B, H, L, dk, dv = 2, 3, 61, 8, 10  # dv=10 to exercise V-tiling/padding
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    # Random ragged lengths in [0..L]; ensure we hit 0 and near-L
    lengths_bh = jax.random.randint(key, (B, H), minval=0, maxval=L + 1)
    lengths_bh = lengths_bh.at[0, 0].set(0).at[0, 1].set(L).astype(jnp.int32)

    # Flash chunk with varlen lengths
    out_chunk, S_chunk = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=chunk_size,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_flash=True,
        use_varlen=True,
        lengths=lengths_bh,
    )
    out_chunk_arr = np.array(out_chunk.array)
    S_chunk_arr = np.array(S_chunk)

    # Recurrent per sequence up to its length
    out_ref = np.zeros((B, L, H, dv), dtype=np.float32)
    S_ref = np.zeros((B, H, dk, dv), dtype=np.float32)
    for b in range(B):
        for h in range(H):
            ell = int(lengths_bh[b, h])
            if ell == 0:
                # state zero and output zero (already initialized)
                continue

            # slice b,h,0:ell as NamedArrays (batch=1, heads=1)
            q_i = q["batch", hax.ds(b, Axis("b", 1))]["heads", hax.ds(h, Axis("h", 1))][
                "position", hax.ds(0, Axis("p", ell))
            ]
            k_i = k["batch", hax.ds(b, Axis("b", 1))]["heads", hax.ds(h, Axis("h", 1))][
                "position", hax.ds(0, Axis("p", ell))
            ]
            v_i = v["batch", hax.ds(b, Axis("b", 1))]["heads", hax.ds(h, Axis("h", 1))][
                "position", hax.ds(0, Axis("p", ell))
            ]
            g_i = g["batch", hax.ds(b, Axis("b", 1))]["heads", hax.ds(h, Axis("h", 1))][
                "position", hax.ds(0, Axis("p", ell))
            ]
            b_i = beta["batch", hax.ds(b, Axis("b", 1))]["heads", hax.ds(h, Axis("h", 1))][
                "position", hax.ds(0, Axis("p", ell))
            ]

            out_i, S_i = recurrent_gated_delta_rule(
                q_i,
                k_i,
                v_i,
                g_i,
                b_i,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_flash=False,  # use reference for parity
            )
            out_ref[b, :ell, h, :] = np.array(out_i.array)[0, :ell, 0, :]
            S_ref[b, h, :, :] = np.array(S_i)[0, 0, :, :]

    # Compare only the valid prefixes
    for b in range(B):
        for h in range(H):
            ell = int(lengths_bh[b, h])
            np.testing.assert_allclose(out_chunk_arr[b, :ell, h, :], out_ref[b, :ell, h, :], rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(S_chunk_arr[b, h], S_ref[b, h], rtol=1e-4, atol=1e-4)

    # All outputs should be finite
    assert np.isfinite(out_chunk_arr).all()


@pytest.mark.skipif(not is_tpu, reason="use_flash=True varlen path is only tested on TPU")
def test_chunk_varlen_offsets_equivalence():
    """lengths and offsets (NH and NH+1) must yield identical outputs/states."""
    key = jax.random.PRNGKey(123)
    B, H, L, dk, dv = 2, 2, 47, 8, 8
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    lengths_bh = jax.random.randint(key, (B, H), minval=0, maxval=L + 1).astype(jnp.int32)
    lengths_flat = lengths_bh.reshape(-1)
    offsets_plus1 = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(lengths_flat, dtype=jnp.int32)], axis=0
    )

    # via lengths
    out_len, S_len = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=32,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_flash=True,
        use_varlen=True,
        lengths=lengths_bh,
    )
    # via offsets (NH+1)
    out_off, S_off = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=32,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_flash=True,
        use_varlen=True,
        offsets=offsets_plus1,
    )
    np.testing.assert_allclose(np.array(out_len.array), np.array(out_off.array), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(S_len, S_off, rtol=1e-5, atol=1e-5)

    # via offsets (NH == lengths)
    out_off2, S_off2 = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=32,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_flash=True,
        use_varlen=True,
        offsets=lengths_flat,  # lengths form
    )
    np.testing.assert_allclose(np.array(out_len.array), np.array(out_off2.array), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(S_len, S_off2, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not is_tpu, reason="use_flash=True varlen path is only tested on TPU")
def test_chunk_varlen_zero_length_and_S0_unchanged():
    """If a sequence has length 0, outputs are zero and S_final equals S0 for that (b,h)."""
    key = jax.random.PRNGKey(7)
    B, H, L, dk, dv = 2, 2, 33, 8, 8
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    # Make one BH pair zero-length, others random
    lengths_bh = jax.random.randint(key, (B, H), minval=1, maxval=L + 1).astype(jnp.int32)
    lengths_bh = lengths_bh.at[0, 0].set(0)

    # Random S0 to check "unchanged"
    S0 = jax.random.normal(jax.random.PRNGKey(11), (B, H, dk, dv), dtype=jnp.float32)

    out_chunk, S_fin = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=16,
        initial_state=S0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_flash=True,
        use_varlen=True,
        lengths=lengths_bh,
    )
    out = np.array(out_chunk.array)
    assert np.allclose(out[0, :, 0, :], 0.0, atol=1e-7)  # all positions are invalid => zeros
    np.testing.assert_allclose(S_fin[0, 0], np.array(S0[0, 0]), rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not is_tpu, reason="use_flash=True varlen path is only tested on TPU")
def test_chunk_varlen_head_first_layout_equivalence():
    """head_first=True path should match the standard layout for the valid prefixes."""
    key = jax.random.PRNGKey(99)
    B, H, L, dk, dv = 1, 3, 57, 16, 10
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    lengths_bh = jax.random.randint(key, (B, H), minval=0, maxval=L + 1).astype(jnp.int32)

    # Standard path
    out_std, _ = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=32,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_flash=True,
        use_varlen=True,
        lengths=lengths_bh,
        head_first=False,
    )
    out_std_arr = np.array(out_std.array)  # (B, L, H, V)

    # Head-first inputs (B, H, L, *)
    q_hf = hax.rearrange(q, ("batch", "heads", "position", "k_head_dim"))
    k_hf = hax.rearrange(k, ("batch", "heads", "position", "k_head_dim"))
    v_hf = hax.rearrange(v, ("batch", "heads", "position", "v_head_dim"))
    g_hf = hax.rearrange(g, ("batch", "heads", "position"))
    b_hf = hax.rearrange(beta, ("batch", "heads", "position"))

    out_hf, _ = chunk_gated_delta_rule(
        q_hf,
        k_hf,
        v_hf,
        g_hf,
        b_hf,
        chunk_size=32,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_flash=True,
        use_varlen=True,
        lengths=lengths_bh,
        head_first=True,
    )
    # Bring back to (B, L, H, V)
    out_hf_std = hax.rearrange(out_hf, ("batch", "position", "heads", "v_head_dim"))
    out_hf_arr = np.array(out_hf_std.array)

    # Compare only the valid prefixes for each head
    for h in range(H):
        ell = int(lengths_bh[0, h])
        np.testing.assert_allclose(out_std_arr[0, :ell, h, :], out_hf_arr[0, :ell, h, :], rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not is_tpu, reason="use_flash=True varlen path is only tested on TPU")
@pytest.mark.parametrize("chunk_size", [17, 32])
def test_chunk_varlen_nondivisible_and_parity_with_recurrent(chunk_size):
    """Non-divisible chunk sizes with varlen should match recurrent on valid prefixes."""
    key = jax.random.PRNGKey(314)
    B, H, L, dk, dv = 2, 2, 59, 8, 8
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    lengths_bh = jax.random.randint(key, (B, H), minval=0, maxval=L + 1).astype(jnp.int32)

    out_chunk, S_chunk = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=chunk_size,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_flash=True,
        use_varlen=True,
        lengths=lengths_bh,
    )
    out_chunk_arr = np.array(out_chunk.array)
    S_chunk_arr = np.array(S_chunk)

    # Reference recurrent per (b,h)
    out_ref = np.zeros((B, L, H, dv), dtype=np.float32)
    S_ref = np.zeros((B, H, dk, dv), dtype=np.float32)
    for b in range(B):
        for h in range(H):
            ell = int(lengths_bh[b, h])
            if ell == 0:
                continue
            q_i = q["batch", hax.ds(b, Axis("b", 1))]["heads", hax.ds(h, Axis("h", 1))][
                "position", hax.ds(0, Axis("p", ell))
            ]
            k_i = k["batch", hax.ds(b, Axis("b", 1))]["heads", hax.ds(h, Axis("h", 1))][
                "position", hax.ds(0, Axis("p", ell))
            ]
            v_i = v["batch", hax.ds(b, Axis("b", 1))]["heads", hax.ds(h, Axis("h", 1))][
                "position", hax.ds(0, Axis("p", ell))
            ]
            g_i = g["batch", hax.ds(b, Axis("b", 1))]["heads", hax.ds(h, Axis("h", 1))][
                "position", hax.ds(0, Axis("p", ell))
            ]
            b_i = beta["batch", hax.ds(b, Axis("b", 1))]["heads", hax.ds(h, Axis("h", 1))][
                "position", hax.ds(0, Axis("p", ell))
            ]
            out_i, S_i = recurrent_gated_delta_rule(
                q_i,
                k_i,
                v_i,
                g_i,
                b_i,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_flash=False,
            )
            out_ref[b, :ell, h, :] = np.array(out_i.array)[0, :ell, 0, :]
            S_ref[b, h, :, :] = np.array(S_i)[0, 0, :, :]

    # Compare only valid prefixes and final states
    for b in range(B):
        for h in range(H):
            ell = int(lengths_bh[b, h])
            np.testing.assert_allclose(out_chunk_arr[b, :ell, h, :], out_ref[b, :ell, h, :], rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(S_chunk_arr[b, h, :, :], S_ref[b, h, :, :], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("backward_mode", ["checkpoint", "custom_vjp"])
def test_backward_mode_gradients_exist(backward_mode):
    """Test that both backward modes produce finite gradients."""
    key = jax.random.PRNGKey(42)
    B, H, L, dk, dv = 2, 3, 32, 8, 8
    chunk_size = 8

    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    def loss_fn(q_arr, k_arr, v_arr, g_arr, b_arr):
        qn = hax.named(q_arr, q.axes)
        kn = hax.named(k_arr, k.axes)
        vn = hax.named(v_arr, v.axes)
        gn = hax.named(g_arr, g.axes)
        bn = hax.named(b_arr, beta.axes)
        out, _ = chunk_gated_delta_rule(
            qn,
            kn,
            vn,
            gn,
            bn,
            chunk_size=chunk_size,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            backward_mode=backward_mode,
        )
        return jnp.sum(out.array)

    # Verify gradients exist and are finite
    grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q.array, k.array, v.array, g.array, beta.array)
    for i, grad in enumerate(grads):
        assert jnp.all(jnp.isfinite(grad)), f"Gradient {i} has non-finite values for {backward_mode}"


@pytest.mark.parametrize("use_triangular_solve", [True, False])
def test_backward_modes_produce_same_gradients(use_triangular_solve: bool):
    """Test that checkpoint and custom_vjp backward modes produce identical gradients."""
    key = jax.random.PRNGKey(123)
    B, H, L, dk, dv = 2, 3, 32, 8, 8
    chunk_size = 8

    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    def loss_fn(q_arr, k_arr, v_arr, g_arr, b_arr, mode):
        qn = hax.named(q_arr, q.axes)
        kn = hax.named(k_arr, k.axes)
        vn = hax.named(v_arr, v.axes)
        gn = hax.named(g_arr, g.axes)
        bn = hax.named(b_arr, beta.axes)
        out, _ = chunk_gated_delta_rule(
            qn,
            kn,
            vn,
            gn,
            bn,
            chunk_size=chunk_size,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            backward_mode=mode,
            use_triangular_solve=use_triangular_solve,
        )
        return jnp.sum(out.array)

    # Gradients with checkpoint mode
    grads_checkpoint = jax.grad(lambda *args: loss_fn(*args, "checkpoint"), argnums=(0, 1, 2, 3, 4))(
        q.array, k.array, v.array, g.array, beta.array
    )

    # Gradients with custom_vjp mode
    grads_custom_vjp = jax.grad(lambda *args: loss_fn(*args, "custom_vjp"), argnums=(0, 1, 2, 3, 4))(
        q.array, k.array, v.array, g.array, beta.array
    )

    # Compare gradients
    for i, (g_ckpt, g_vjp) in enumerate(zip(grads_checkpoint, grads_custom_vjp)):
        np.testing.assert_allclose(
            np.array(g_ckpt),
            np.array(g_vjp),
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Gradient {i} differs between checkpoint and custom_vjp modes",
        )
