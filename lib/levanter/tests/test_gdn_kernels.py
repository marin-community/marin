# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import jax
import jax.numpy as jnp
import haliax as hax
from haliax import Axis
import pytest

from levanter.layers.gated_deltanet import chunk_gated_delta_rule, recurrent_gated_delta_rule
from tests.test_utils import skip_if_no_torch

jax.config.update("jax_default_matmul_precision", "float32")


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


def test_recurrent_no_learning_beta_zero_outputs_zero():
    """If beta == 0 everywhere and S0 == 0, then S_t == 0 for all t and outputs are zero."""
    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 3, 17, 8, 8
    q, k, v, g, _ = _named_kernels_inputs(B, H, L, dk, dv, key)
    beta0 = hax.named(jnp.zeros((B, L, H), dtype=jnp.float32), ("batch", "position", "heads"))

    out, S_final = recurrent_gated_delta_rule(q, k, v, g, beta0, initial_state=None, output_final_state=True)
    np.testing.assert_allclose(np.array(out.array), 0.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(S_final, 0.0, rtol=1e-6, atol=1e-6)


def test_recurrent_perfect_fit_on_current_key_when_alpha1_beta1_and_L2norm():
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
            q_t, k_t, v_t, g_t, b_t, initial_state=S, output_final_state=True, use_qk_l2norm_in_kernel=True
        )

        # Check S^T k̂_t == v_t (k̂_t is K normalized as in the kernel)
        k_arr = hax.rearrange(k_t, ("batch", "heads", "position", "k_head_dim")).array[..., 0, :]
        inv = jax.lax.rsqrt(jnp.sum(k_arr * k_arr, axis=-1, keepdims=True) + 1e-6)
        k_hat = k_arr * inv
        kv = jnp.sum(S * k_hat[..., None], axis=-2)  # (B,H,dv)
        v_arr = hax.rearrange(v_t, ("batch", "heads", "position", "v_head_dim")).array[..., 0, :]
        np.testing.assert_allclose(kv, v_arr, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 16, 32, 64])
def test_chunk_equals_recurrent_for_random_inputs(chunk_size):
    """Chunkwise kernel must match recurrent kernel for many chunk sizes (including 1)."""
    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 3, 57, 8, 8
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    out_chunk, S_chunk = chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=chunk_size, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    out_recur, S_recur = recurrent_gated_delta_rule(
        q, k, v, g, beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True
    )

    np.testing.assert_allclose(np.array(out_chunk.array), np.array(out_recur.array), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(S_chunk, S_recur, rtol=1e-4, atol=1e-4)


def test_chunk_nondivisible_padding_matches_recurrent_jax_only():
    """When L % chunk_size != 0, padding path should still match the recurrent kernel (JAX-only)."""
    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 4, 61, 8, 16
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    out_chunk, _ = chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=32, output_final_state=False, use_qk_l2norm_in_kernel=True
    )
    out_recur, _ = recurrent_gated_delta_rule(q, k, v, g, beta, output_final_state=False, use_qk_l2norm_in_kernel=True)
    np.testing.assert_allclose(np.array(out_chunk.array), np.array(out_recur.array), rtol=1e-4, atol=1e-4)


def test_chunk_continuation_two_pass_equals_one_pass():
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
    out_full, S_full = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk_size, output_final_state=True)

    # Two-pass: prefix then suffix with initial_state
    q_pre, q_suf = q["position", hax.ds(0, split)], q["position", hax.ds(split, Axis("pos2", L - split))]
    k_pre, k_suf = k["position", hax.ds(0, split)], k["position", hax.ds(split, Axis("pos2", L - split))]
    v_pre, v_suf = v["position", hax.ds(0, split)], v["position", hax.ds(split, Axis("pos2", L - split))]
    g_pre, g_suf = g["position", hax.ds(0, split)], g["position", hax.ds(split, Axis("pos2", L - split))]
    b_pre, b_suf = beta["position", hax.ds(0, split)], beta["position", hax.ds(split, Axis("pos2", L - split))]

    out_pre, S_mid = chunk_gated_delta_rule(
        q_pre, k_pre, v_pre, g_pre, b_pre, chunk_size=chunk_size, output_final_state=True
    )
    out_suf, S_end = chunk_gated_delta_rule(
        q_suf, k_suf, v_suf, g_suf, b_suf, chunk_size=chunk_size, initial_state=S_mid, output_final_state=True
    )

    # Compare outputs on the suffix region and final states
    out_full_suf = out_full["position", hax.ds(split, Axis("pos2", L - split))]
    np.testing.assert_allclose(np.array(out_suf.array), np.array(out_full_suf.array), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(S_end, S_full, rtol=1e-4, atol=1e-4)


def test_chunk_size_one_degenerates_to_recurrent_without_l2norm():
    """Degeneracy should also hold even when L2 norm is disabled."""
    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 2, 29, 8, 8
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)
    out_chunk, _ = chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=1, output_final_state=False, use_qk_l2norm_in_kernel=False
    )
    out_recur, _ = recurrent_gated_delta_rule(
        q, k, v, g, beta, output_final_state=False, use_qk_l2norm_in_kernel=False
    )
    np.testing.assert_allclose(np.array(out_chunk.array), np.array(out_recur.array), rtol=1e-4, atol=1e-4)


def test_extreme_gates_numerical_stability_jax_only():
    """Outputs should be finite when α ≈ 0 (very negative g) and β near 0 or near 1."""
    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 1, 2, 37, 16, 8

    q, k, v, _, _ = _named_kernels_inputs(B, H, L, dk, dv, key)
    # Strong negative g: α ~ 0
    g = -hax.random.uniform(key, {"batch": B, "position": L, "heads": H}, minval=2.0, maxval=8.0, dtype=jnp.float32)
    beta_small = hax.named(jnp.full((B, L, H), 1e-4, dtype=jnp.float32), ("batch", "position", "heads"))
    beta_big = hax.named(jnp.full((B, L, H), 1.0 - 1e-6, dtype=jnp.float32), ("batch", "position", "heads"))

    for beta in [beta_small, beta_big]:
        out_chunk, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=32, output_final_state=False)
        assert np.isfinite(np.array(out_chunk.array)).all()

        out_recur, _ = recurrent_gated_delta_rule(q, k, v, g, beta, output_final_state=False)
        assert np.isfinite(np.array(out_recur.array)).all()


def test_gradients_exist_small_kernel_graph():
    """Smoke test: both kernels are differentiable w.r.t. inputs (no NaNs in grads)."""
    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 1, 1, 7, 4, 4
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    # grad wrt q only (keep the surface small)
    def loss_chunk(q_arr):
        qn = hax.named(q_arr, q.axes)
        out, _ = chunk_gated_delta_rule(qn, k, v, g, beta, chunk_size=4, output_final_state=False)
        return jnp.sum(out.array)

    def loss_recur(q_arr):
        qn = hax.named(q_arr, q.axes)
        out, _ = recurrent_gated_delta_rule(qn, k, v, g, beta, output_final_state=False)
        return jnp.sum(out.array)

    g1 = jax.grad(loss_chunk)(q.array)
    g2 = jax.grad(loss_recur)(q.array)
    assert jnp.all(jnp.isfinite(g1))
    assert jnp.all(jnp.isfinite(g2))


@skip_if_no_torch
def test_recurrent_kernel_matches_hf():
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 1, 2, 17, 8, 8
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    out_named, _ = recurrent_gated_delta_rule(q, k, v, g, beta, output_final_state=False)

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
def test_chunk_kernel_matches_hf():
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 4, 64, 8, 16
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    out_named, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=32, output_final_state=False)

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
def test_chunk_kernel_matches_hf_non_divisible():
    """L not divisible by chunk_size should still match HF fallback (padding path)."""
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 3, 61, 8, 16
    chunk_size = 32

    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    out_named, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk_size, output_final_state=False)

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
def test_chunk_size_one_matches_hf_recurrent():
    """chunk_size=1 should degenerate to the recurrent rule."""
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 2, 29, 8, 8
    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    out_chunk, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=1, output_final_state=False)
    out_recur, _ = recurrent_gated_delta_rule(q, k, v, g, beta, output_final_state=False)
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
def test_chunk_kernel_with_initial_state_matches_recurrent_continuation():
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
        q, k, v, g, beta, chunk_size=chunk_size, initial_state=S0, output_final_state=False
    )
    out_recur, _ = recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=S0, output_final_state=False)
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
def test_short_sequences_edge_cases():
    """Short L vs chunk_size and kernel-size behaviors."""
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)

    for L in [1, 2, 3, 5, 7]:
        B, H, dk, dv = 2, 2, 8, 8
        q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

        out_named, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64, output_final_state=False)

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
def test_extreme_gates_no_nans_and_parity():
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
        out_named, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=32, output_final_state=False)

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
def test_kernels_match_hf_without_l2norm():
    # TODO: fix edge case? although per original paper L2 norm is needed for stability
    pytest.skip("not matching HF implementation")

    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 3, 57, 16, 8

    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    # Haliax kernels with use_qk_l2norm_in_kernel=False
    out_chunk_j, _ = chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=32, output_final_state=False, use_qk_l2norm_in_kernel=False
    )
    out_recur_j, _ = recurrent_gated_delta_rule(
        q, k, v, g, beta, output_final_state=False, use_qk_l2norm_in_kernel=False
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

    # Compare
    np.testing.assert_allclose(jq, tq, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jk, tk, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jv, tv, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jg, tg, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jb, tb, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jS0, tS0, rtol=1e-4, atol=5e-6)


@skip_if_no_torch
def test_chunk_backward_matches_hf():
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

    # Compare
    np.testing.assert_allclose(jq, tq, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jk, tk, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jv, tv, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jg, tg, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jb, tb, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(jS0, tS0, rtol=1e-4, atol=5e-6)
