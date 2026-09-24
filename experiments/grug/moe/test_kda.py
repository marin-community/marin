# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Numerical parity for the Kimi Delta Attention (KDA) kernels.

The chunked-parallel kernel is the MFU-friendly training path; its correctness is
anchored to the sequential recurrence (an independent, obviously-correct oracle),
and in the scalar-gate limit to levanter's HF-validated gated-delta-rule kernel.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.moe.kda import chunk_kda, recurrent_kda

jax.config.update("jax_default_matmul_precision", "float32")


def _inputs(batch, heads, length, dk, dv, *, seed=0):
    rng = np.random.RandomState(seed)
    q = jnp.asarray(rng.randn(batch, heads, length, dk), jnp.float32)
    k = jnp.asarray(rng.randn(batch, heads, length, dk), jnp.float32)
    v = jnp.asarray(rng.randn(batch, heads, length, dv), jnp.float32)
    # Per-channel log-decay g <= 0 (alpha = exp(g) in (0, 1]).
    g = -0.1 * jnp.abs(jnp.asarray(rng.randn(batch, heads, length, dk), jnp.float32))
    beta = jnp.asarray(rng.rand(batch, heads, length), jnp.float32)
    return q, k, v, g, beta


@pytest.mark.parametrize(
    ("length", "chunk_size"),
    [(64, 64), (128, 64), (57, 16), (61, 32), (29, 7), (48, 16)],
)
def test_chunk_kda_matches_recurrent(length, chunk_size):
    """Chunkwise-parallel KDA (exact fp32 GEMMs) equals the sequential recurrence."""
    q, k, v, g, beta = _inputs(2, 3, length, 16, 16)
    out_chunk, state_chunk = chunk_kda(q, k, v, g, beta, chunk_size=chunk_size, matmul_dtype=jnp.float32)
    out_recur, state_recur = recurrent_kda(q, k, v, g, beta)
    np.testing.assert_allclose(np.asarray(out_chunk), np.asarray(out_recur), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(state_chunk), np.asarray(state_recur), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("length", "chunk_size"), [(128, 64), (256, 64), (192, 32)])
def test_chunk_kda_bf16_matmuls_match_recurrent(length, chunk_size):
    """The default bf16 intra-chunk GEMMs stay close to the fp32 recurrence.

    bf16 has ~3 decimal digits of mantissa, so the chunk kernel tracks the fp32
    oracle to ~1e-2 relative -- the accuracy of the activations it feeds anyway."""
    q, k, v, g, beta = _inputs(2, 3, length, 32, 32)
    out_chunk, _ = chunk_kda(q, k, v, g, beta, chunk_size=chunk_size, matmul_dtype=jnp.bfloat16)
    out_recur, _ = recurrent_kda(q, k, v, g, beta)
    np.testing.assert_allclose(np.asarray(out_chunk), np.asarray(out_recur), rtol=2e-2, atol=2e-2)


def test_chunk_kda_initial_state_continuation():
    """A non-zero initial state carries through both kernels identically."""
    q, k, v, g, beta = _inputs(1, 2, 48, 16, 16, seed=3)
    s0 = jnp.asarray(np.random.RandomState(9).randn(1, 2, 16, 16) * 0.1, jnp.float32)
    out_chunk, _ = chunk_kda(q, k, v, g, beta, chunk_size=16, initial_state=s0, matmul_dtype=jnp.float32)
    out_recur, _ = recurrent_kda(q, k, v, g, beta, initial_state=s0)
    np.testing.assert_allclose(np.asarray(out_chunk), np.asarray(out_recur), rtol=1e-4, atol=1e-4)


def test_scalar_gate_limit_matches_levanter_gdn():
    """With g broadcast over channels (scalar per-head decay), KDA reduces to the
    Gated DeltaNet rule already validated against HF in levanter -- an independent
    oracle for the whole per-channel derivation."""
    haliax = pytest.importorskip("haliax")
    from levanter.layers.gated_deltanet import recurrent_gated_delta_rule  # noqa: PLC0415

    batch, heads, length, dk, dv = 2, 3, 40, 8, 8
    q, k, v, _, beta = _inputs(batch, heads, length, dk, dv, seed=5)
    g_scalar = -0.1 * jnp.abs(jnp.asarray(np.random.RandomState(7).randn(batch, heads, length), jnp.float32))
    g_perchannel = jnp.broadcast_to(g_scalar[..., None], (batch, heads, length, dk))

    out_kda, _ = recurrent_kda(q, k, v, g_perchannel, beta)

    def to_named(arr, dim_name):
        return haliax.named(jnp.moveaxis(arr, 1, 2), ("batch", "position", "heads", dim_name))

    out_lev, _ = recurrent_gated_delta_rule(
        to_named(q, "k_head_dim"),
        to_named(k, "k_head_dim"),
        to_named(v, "v_head_dim"),
        haliax.named(jnp.moveaxis(g_scalar, 1, 2), ("batch", "position", "heads")),
        haliax.named(jnp.moveaxis(beta, 1, 2), ("batch", "position", "heads")),
        use_qk_l2norm_in_kernel=True,
    )
    out_lev = jnp.moveaxis(out_lev.array, 2, 1)  # -> (batch, heads, length, dv)
    np.testing.assert_allclose(np.asarray(out_kda), np.asarray(out_lev), rtol=1e-4, atol=1e-4)


def test_chunk_kda_finite_under_strong_decay():
    """Strong per-channel decay (alpha ~ 0) and extreme beta stay finite (fp32)."""
    q, k, v, _, _ = _inputs(1, 2, 37, 16, 8, seed=1)
    g = -jnp.asarray(np.random.RandomState(2).uniform(2.0, 6.0, size=(1, 2, 37, 16)), jnp.float32)
    for beta_val in (1e-4, 1.0 - 1e-6):
        beta = jnp.full((1, 2, 37), beta_val, jnp.float32)
        out_chunk, _ = chunk_kda(q, k, v, g, beta, chunk_size=32)
        assert np.isfinite(np.asarray(out_chunk)).all()


def test_chunk_kda_gradients_finite():
    """The chunked kernel is differentiable w.r.t. its inputs without NaNs."""
    q, k, v, g, beta = _inputs(1, 1, 16, 8, 8, seed=4)

    def loss(q_arr, g_arr, b_arr):
        out, _ = chunk_kda(q_arr, k, v, g_arr, b_arr, chunk_size=8)
        return jnp.sum(out)

    grads = jax.grad(loss, argnums=(0, 1, 2))(q, g, beta)
    assert all(jnp.all(jnp.isfinite(grad)) for grad in grads)
