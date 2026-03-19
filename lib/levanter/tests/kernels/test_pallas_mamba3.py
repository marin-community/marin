# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp

from levanter.kernels.pallas.mamba3 import (
    mamba3_chunk_state,
    mamba3_chunk_state_reference_batched,
    mamba3_chunked_forward,
    mamba3_chunked_forward_from_transformed,
    mamba3_chunked_forward_reference_batched,
    mamba3_chunked_sequential_reference_batched,
    mamba3_direct_recurrence_reference_batched,
    mamba3_intra_chunk,
    mamba3_intra_chunk_reference_batched,
    prepare_mamba3_chunked_scales,
    prepare_mamba3_scales,
)
from levanter.kernels.pallas.ssd import intra_chunk_log_alpha_cumsum, local_log_alpha


def _sample_inputs(
    *,
    leading_shape: tuple[int, ...],
    chunk_size: int,
    state_dim: int,
    value_dim: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    key = jax.random.PRNGKey(0)
    k_dt, k_lam, k_a, k_b, k_c, k_x = jax.random.split(key, 6)
    dt = 0.01 + 0.1 * jax.random.uniform(k_dt, leading_shape + (chunk_size,), dtype=jnp.float32)
    lam = jax.random.uniform(k_lam, leading_shape + (chunk_size,), dtype=jnp.float32)
    a = -0.5 - jax.random.uniform(k_a, leading_shape + (chunk_size,), dtype=jnp.float32)
    b = jax.random.normal(k_b, leading_shape + (chunk_size, state_dim), dtype=jnp.float32)
    c = jax.random.normal(k_c, leading_shape + (chunk_size, state_dim), dtype=jnp.float32)
    x = jax.random.normal(k_x, leading_shape + (chunk_size, value_dim), dtype=jnp.float32)
    return dt, lam, a, b, c, x


def _sample_chunked_inputs(
    *,
    leading_shape: tuple[int, ...],
    num_chunks: int,
    chunk_size: int,
    state_dim: int,
    value_dim: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return _sample_inputs(
        leading_shape=leading_shape + (num_chunks,),
        chunk_size=chunk_size,
        state_dim=state_dim,
        value_dim=value_dim,
    )


def test_prepare_mamba3_scales_matches_shift_formula():
    dt = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    lam = jnp.array([[0.25, 0.5, 0.75]], dtype=jnp.float32)
    src_scale, out_correction = prepare_mamba3_scales(dt, lam)

    expected_out = jnp.array([[1.0, 0.75, 0.0]], dtype=jnp.float32)
    expected_src = jnp.array([[1.25, 1.75, 2.25]], dtype=jnp.float32)
    assert jnp.allclose(out_correction, expected_out, atol=0.0, rtol=0.0)
    assert jnp.allclose(src_scale, expected_src, atol=0.0, rtol=0.0)


def test_prepare_mamba3_chunked_scales_shift_across_chunk_boundary():
    dt = jnp.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=jnp.float32)
    lam = jnp.array([[[0.25, 0.5], [0.75, 0.0]]], dtype=jnp.float32)
    src_scale, out_correction = prepare_mamba3_chunked_scales(dt, lam)

    expected_out = jnp.array([[[1.0, 0.75], [4.0, 0.0]]], dtype=jnp.float32)
    expected_src = jnp.array([[[1.25, 1.75], [6.25, 0.0]]], dtype=jnp.float32)
    assert jnp.allclose(out_correction, expected_out, atol=0.0, rtol=0.0)
    assert jnp.allclose(src_scale, expected_src, atol=0.0, rtol=0.0)


def test_mamba3_intra_chunk_xla_matches_reference():
    dt, lam, a, b, c, x = _sample_inputs(leading_shape=(2, 3), chunk_size=8, state_dim=5, value_dim=7)
    src_scale, out_correction = prepare_mamba3_scales(dt, lam)
    a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
    y_ref = mamba3_intra_chunk_reference_batched(
        a_log_cumsum.reshape(6, 8),
        src_scale.reshape(6, 8),
        out_correction.reshape(6, 8),
        b.reshape(6, 8, 5),
        c.reshape(6, 8, 5),
        x.reshape(6, 8, 7),
    )
    y = mamba3_intra_chunk(a_log_cumsum, src_scale, out_correction, b, c, x, implementation="xla")
    assert jnp.allclose(y.reshape(6, 8, 7), y_ref, atol=1e-5, rtol=1e-5)


def test_mamba3_chunk_state_matches_reference_shape_and_values():
    dt, lam, a, b, _, x = _sample_inputs(leading_shape=(2,), chunk_size=8, state_dim=5, value_dim=7)
    src_scale, _ = prepare_mamba3_scales(dt, lam)
    a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
    chunk_state = mamba3_chunk_state(a_log_cumsum, src_scale, b, x)
    manual = mamba3_chunk_state_reference_batched(
        a_log_cumsum.reshape(2, 8),
        src_scale.reshape(2, 8),
        b.reshape(2, 8, 5),
        x.reshape(2, 8, 7),
    )
    assert chunk_state.shape == (2, 7, 5)
    assert jnp.allclose(chunk_state, manual, atol=1e-5, rtol=1e-5)


def test_g_rewrite_matches_direct_recurrence_across_chunks():
    dt, lam, a, b, c, x = _sample_chunked_inputs(
        leading_shape=(2,),
        num_chunks=3,
        chunk_size=8,
        state_dim=4,
        value_dim=6,
    )
    src_scale, out_correction = prepare_mamba3_chunked_scales(dt, lam)
    a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
    y_rewrite, final_state_rewrite = mamba3_chunked_sequential_reference_batched(
        a_log_cumsum.reshape(2, 3, 8),
        src_scale.reshape(2, 3, 8),
        out_correction.reshape(2, 3, 8),
        b.reshape(2, 3, 8, 4),
        c.reshape(2, 3, 8, 4),
        x.reshape(2, 3, 8, 6),
    )
    y_direct, final_state_direct = mamba3_direct_recurrence_reference_batched(
        dt.reshape(2, 3, 8),
        lam.reshape(2, 3, 8),
        a.reshape(2, 3, 8),
        b.reshape(2, 3, 8, 4),
        c.reshape(2, 3, 8, 4),
        x.reshape(2, 3, 8, 6),
    )
    assert jnp.allclose(y_rewrite, y_direct, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(final_state_rewrite, final_state_direct, atol=1e-5, rtol=1e-5)


def test_chunked_xla_matches_direct_recurrence():
    inputs = _sample_chunked_inputs(leading_shape=(2,), num_chunks=3, chunk_size=8, state_dim=4, value_dim=6)
    y_xla, final_state_xla = mamba3_chunked_forward(*inputs, implementation="xla")
    y_ref, final_state_ref = mamba3_chunked_forward(*inputs, implementation="reference")
    assert jnp.allclose(y_xla, y_ref, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(final_state_xla, final_state_ref, atol=1e-5, rtol=1e-5)


def test_chunked_xla_state_heavy_auto_path_matches_direct_recurrence():
    inputs = _sample_chunked_inputs(leading_shape=(2,), num_chunks=3, chunk_size=8, state_dim=8, value_dim=3)
    y_xla, final_state_xla = mamba3_chunked_forward(*inputs, implementation="xla")
    y_ref, final_state_ref = mamba3_chunked_forward(*inputs, implementation="reference")
    assert jnp.allclose(y_xla, y_ref, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(final_state_xla, final_state_ref, atol=1e-5, rtol=1e-5)


def test_transformed_chunked_reference_matches_sequential_rewrite():
    dt, lam, a, b, c, x = _sample_chunked_inputs(
        leading_shape=(2,),
        num_chunks=3,
        chunk_size=8,
        state_dim=4,
        value_dim=6,
    )
    src_scale, out_correction = prepare_mamba3_chunked_scales(dt, lam)
    a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
    y_shim, final_state_shim = mamba3_chunked_forward_reference_batched(
        a_log_cumsum.reshape(2, 3, 8),
        src_scale.reshape(2, 3, 8),
        out_correction.reshape(2, 3, 8),
        b.reshape(2, 3, 8, 4),
        c.reshape(2, 3, 8, 4),
        x.reshape(2, 3, 8, 6),
    )
    y_seq, final_state_seq = mamba3_chunked_sequential_reference_batched(
        a_log_cumsum.reshape(2, 3, 8),
        src_scale.reshape(2, 3, 8),
        out_correction.reshape(2, 3, 8),
        b.reshape(2, 3, 8, 4),
        c.reshape(2, 3, 8, 4),
        x.reshape(2, 3, 8, 6),
    )
    assert jnp.allclose(y_shim, y_seq, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(final_state_shim, final_state_seq, atol=1e-5, rtol=1e-5)


def test_mamba3_chunked_forward_from_transformed_matches_native_api():
    dt, lam, a, b, c, x = _sample_chunked_inputs(
        leading_shape=(2,),
        num_chunks=2,
        chunk_size=8,
        state_dim=4,
        value_dim=6,
    )
    src_scale, out_correction = prepare_mamba3_chunked_scales(dt, lam)
    a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
    y_native, state_native = mamba3_chunked_forward(dt, lam, a, b, c, x, implementation="xla")
    y_transformed, state_transformed = mamba3_chunked_forward_from_transformed(
        a_log_cumsum,
        src_scale,
        out_correction,
        b,
        c,
        x,
        implementation="xla",
    )
    assert jnp.allclose(y_native, y_transformed, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(state_native, state_transformed, atol=1e-5, rtol=1e-5)


def test_mamba3_chunked_grad_matches_reference():
    dt, lam, a, b, c, x = _sample_chunked_inputs(
        leading_shape=(),
        num_chunks=2,
        chunk_size=4,
        state_dim=3,
        value_dim=5,
    )

    def loss_ref(x_in: jax.Array) -> jax.Array:
        y, _ = mamba3_chunked_forward(dt, lam, a, b, c, x_in.reshape(2, 4, 5), implementation="reference")
        return jnp.sum(y)

    def loss_xla(x_in: jax.Array) -> jax.Array:
        y, _ = mamba3_chunked_forward(dt, lam, a, b, c, x_in.reshape(2, 4, 5), implementation="xla")
        return jnp.sum(y)

    g_ref = jax.grad(loss_ref)(x)
    g_xla = jax.grad(loss_xla)(x)
    assert jnp.allclose(g_ref, g_xla, atol=1e-5, rtol=1e-5)


def test_mamba3_xla_longer_stress_stays_finite():
    dt, lam, a, b, c, x = _sample_chunked_inputs(
        leading_shape=(1,),
        num_chunks=8,
        chunk_size=32,
        state_dim=16,
        value_dim=32,
    )
    dt = dt * 5.0
    y, final_state = mamba3_chunked_forward(
        dt.astype(jnp.bfloat16),
        lam.astype(jnp.bfloat16),
        a.astype(jnp.bfloat16),
        b.astype(jnp.bfloat16),
        c.astype(jnp.bfloat16),
        x.astype(jnp.bfloat16),
        implementation="xla",
    )
    assert jnp.all(jnp.isfinite(y.astype(jnp.float32)))
    assert jnp.all(jnp.isfinite(final_state.astype(jnp.float32)))


def test_mamba3_tpu_aligned_smoke_shape_compiles_under_jit():
    dt, lam, a, b, c, x = _sample_chunked_inputs(
        leading_shape=(1,),
        num_chunks=2,
        chunk_size=128,
        state_dim=64,
        value_dim=128,
    )
    fn = jax.jit(lambda *args: mamba3_chunked_forward(*args, implementation="xla"))
    y, state = fn(
        dt.astype(jnp.bfloat16),
        lam.astype(jnp.bfloat16),
        a.astype(jnp.bfloat16),
        b.astype(jnp.bfloat16),
        c.astype(jnp.bfloat16),
        x.astype(jnp.bfloat16),
    )
    assert y.shape == (1, 2, 128, 128)
    assert state.shape == (1, 128, 64)
