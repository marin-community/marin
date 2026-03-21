# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from levanter.kernels.pallas.mamba3 import (
    HybridModeConfig,
    mamba3_chunk_state,
    mamba3_chunk_state_reference_batched,
    mamba3_chunked_forward,
    mamba3_chunked_forward_from_transformed,
    mamba3_chunked_forward_reference_batched,
    mamba3_chunked_sequential_reference_batched,
    mamba3_direct_recurrence_reference_batched,
    mamba3_hybrid_chunked_forward,
    mamba3_hybrid_chunked_forward_from_transformed,
    mamba3_intra_chunk,
    mamba3_intra_chunk_reference_batched,
    mamba3_mimo_chunked_forward,
    mamba3_mimo_direct_recurrence_reference_batched,
    mamba3_tpu_default_chunk_size,
    prepare_mamba3_chunked_scales,
    prepare_mamba3_scales,
)
from levanter.kernels.pallas.mamba3.reference import (
    mamba3_mimo_chunked_forward_ranked_reference_batched,
    mamba3_mimo_chunked_sequential_ranked_reference_batched,
    mamba3_mimo_direct_recurrence_ranked_reference_batched,
    mamba3_mimo_rank_expand,
)
from levanter.kernels.pallas.mamba3.xla import mamba3_mimo_chunked_forward_ranked_xla_batched
from levanter.kernels.pallas.ssd import intra_chunk_log_alpha_cumsum, local_log_alpha
from tests.test_utils import skip_if_no_torch


MAMBA3_MIMO_RANKED_PARITY_ATOL = 1e-4
MAMBA3_MIMO_RANKED_PARITY_RTOL = 1e-4


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


def _sample_mimo_chunked_inputs(
    *,
    leading_shape: tuple[int, ...],
    num_chunks: int,
    chunk_size: int,
    state_dim: int,
    value_dim: int,
    rank: int,
) -> tuple[jax.Array, ...]:
    key = jax.random.PRNGKey(17)
    keys = jax.random.split(key, 11)
    token_shape = leading_shape + (num_chunks, chunk_size)
    dt = 0.01 + 0.1 * jax.random.uniform(keys[0], token_shape, dtype=jnp.float32)
    lam = jax.random.uniform(keys[1], token_shape, dtype=jnp.float32)
    a = -0.5 - jax.random.uniform(keys[2], token_shape, dtype=jnp.float32)
    b = jax.random.normal(keys[3], token_shape + (state_dim, rank), dtype=jnp.float32)
    c = jax.random.normal(keys[4], token_shape + (state_dim, rank), dtype=jnp.float32)
    x_base = jax.random.normal(keys[5], token_shape + (value_dim,), dtype=jnp.float32)
    z_base = jax.random.normal(keys[6], token_shape + (value_dim,), dtype=jnp.float32)
    group_shape = leading_shape if leading_shape else ()
    w_x = jax.random.normal(keys[7], group_shape + (value_dim, rank), dtype=jnp.float32)
    w_z = jax.random.normal(keys[8], group_shape + (value_dim, rank), dtype=jnp.float32)
    w_o = jax.random.normal(keys[9], group_shape + (value_dim, rank), dtype=jnp.float32)
    return dt, lam, a, b, c, x_base, z_base, w_x, w_z, w_o


def _torch_logit(x):
    import torch

    eps = torch.finfo(x.dtype).eps
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x) - torch.log1p(-x)


def _upstream_mamba3_siso_step_ref_torch(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles):
    # Ported from state-spaces/mamba tests/ops/triton/test_mamba3_siso.py at a76afbd.
    import torch
    import torch.nn.functional as F

    batch, seqlen, nheads, headdim_qk = Q.shape
    headdim_v = V.shape[-1]
    headdim_angles = Angles.shape[-1]
    angle_state = torch.zeros((batch, nheads, headdim_angles), dtype=torch.float32, device=Q.device)
    ssm_state = torch.zeros((batch, nheads, headdim_v, headdim_qk), dtype=torch.float32, device=Q.device)
    k_state = torch.zeros((batch, nheads, headdim_qk), dtype=Q.dtype, device=Q.device)
    v_state = torch.zeros((batch, nheads, headdim_v), dtype=V.dtype, device=V.device)
    outputs = []

    def apply_rotary_emb(tensor, cos, sin):
        tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
        tensor_0 = tensor_reshaped[..., 0]
        tensor_1 = tensor_reshaped[..., 1]
        if cos.shape[-1] < tensor_0.shape[-1]:
            pad_size = tensor_0.shape[-1] - cos.shape[-1]
            cos = F.pad(cos, (0, pad_size), value=1.0)
            sin = F.pad(sin, (0, pad_size), value=0.0)
        rotated_0 = tensor_0 * cos - tensor_1 * sin
        rotated_1 = tensor_0 * sin + tensor_1 * cos
        return torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)

    for idx in range(seqlen):
        q = Q[:, idx] + Q_bias.unsqueeze(0)
        k = K[:, idx] + K_bias.unsqueeze(0)
        v = V[:, idx]
        angle_state = angle_state + torch.tanh(Angles[:, idx]) * DT[:, :, idx].unsqueeze(-1) * math.pi
        cos_angles = torch.cos(angle_state)
        sin_angles = torch.sin(angle_state)
        q_rot = apply_rotary_emb(q, cos_angles, sin_angles)
        k_rot = apply_rotary_emb(k, cos_angles, sin_angles)
        trap = torch.sigmoid(Trap[:, :, idx])
        alpha = torch.exp(ADT[:, :, idx])
        beta = (1 - trap) * DT[:, :, idx] * alpha
        gamma = trap * DT[:, :, idx]
        ssm_state = alpha[..., None, None] * ssm_state
        ssm_state = ssm_state + beta[..., None, None] * (k_state.unsqueeze(-2) * v_state.unsqueeze(-1))
        ssm_state = ssm_state + gamma[..., None, None] * (k_rot.unsqueeze(-2) * v.unsqueeze(-1))
        outputs.append(torch.einsum("bhdD,bhD->bhd", ssm_state, q_rot.to(ssm_state.dtype)))
        k_state = k_rot
        v_state = v

    return torch.stack(outputs, dim=1), ssm_state


def _upstream_mamba3_mimo_step_ref_torch(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, MIMO_V, MIMO_O, Z, MIMO_Z):
    # Ported from state-spaces/mamba tests/ops/tilelang/test_mamba3_mimo.py at a76afbd.
    import torch
    import torch.nn.functional as F

    batch, seqlen, mimo_rank, nheads, headdim_qk = Q.shape
    headdim_v = V.shape[-1]
    headdim_angles = Angles.shape[-1]
    angle_state = torch.zeros((batch, nheads, headdim_angles), dtype=torch.float32, device=Q.device)
    ssm_state = torch.zeros((batch, nheads, headdim_v, headdim_qk), dtype=torch.float32, device=Q.device)
    k_state = torch.zeros((batch, nheads, mimo_rank, headdim_qk), dtype=Q.dtype, device=Q.device)
    v_state = torch.zeros((batch, nheads, mimo_rank, headdim_v), dtype=V.dtype, device=V.device)
    outputs = []

    def apply_rotary_emb(tensor, cos, sin):
        tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
        tensor_0 = tensor_reshaped[..., 0]
        tensor_1 = tensor_reshaped[..., 1]
        if cos.shape[-1] < tensor_0.shape[-1]:
            pad_size = tensor_0.shape[-1] - cos.shape[-1]
            cos = F.pad(cos, (0, pad_size), value=1.0)
            sin = F.pad(sin, (0, pad_size), value=0.0)
        rotated_0 = tensor_0 * cos - tensor_1 * sin
        rotated_1 = tensor_0 * sin + tensor_1 * cos
        return torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)

    q_bias = Q_bias.permute(1, 0, 2)
    k_bias = K_bias.permute(1, 0, 2)
    v_proj = torch.einsum("bthd,hrd->btrhd", V, MIMO_V)
    z_proj = torch.einsum("bthd,hrd->btrhd", Z, MIMO_Z)

    for idx in range(seqlen):
        q = (Q[:, idx] + q_bias.unsqueeze(0)).permute(0, 2, 1, 3)
        k = (K[:, idx] + k_bias.unsqueeze(0)).permute(0, 2, 1, 3)
        v = v_proj[:, idx].permute(0, 2, 1, 3)
        z = z_proj[:, idx].permute(0, 2, 1, 3)
        angle_state = angle_state + torch.tanh(Angles[:, idx]) * DT[:, :, idx].unsqueeze(-1) * math.pi
        cos_angles = torch.cos(angle_state).unsqueeze(2)
        sin_angles = torch.sin(angle_state).unsqueeze(2)
        q_rot = apply_rotary_emb(q, cos_angles, sin_angles)
        k_rot = apply_rotary_emb(k, cos_angles, sin_angles)
        trap = torch.sigmoid(Trap[:, :, idx])
        alpha = torch.exp(ADT[:, :, idx])
        beta = (1 - trap) * DT[:, :, idx] * alpha
        gamma = trap * DT[:, :, idx]
        prev_kv = torch.einsum("bhrd,bhrp->bhpd", k_state, v_state)
        curr_kv = torch.einsum("bhrd,bhrp->bhpd", k_rot, v)
        ssm_state = alpha[..., None, None] * ssm_state
        ssm_state = ssm_state + beta[..., None, None] * prev_kv
        ssm_state = ssm_state + gamma[..., None, None] * curr_kv
        out = torch.einsum("bhpd,bhrd->bhrp", ssm_state, q_rot.to(ssm_state.dtype))
        out = out * z * torch.sigmoid(z)
        outputs.append(torch.einsum("bhrp,hrp->bhp", out, MIMO_O))
        k_state = k_rot
        v_state = v

    return torch.stack(outputs, dim=1), ssm_state


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


def test_mamba3_hybrid_mode_config_resolves_current_tpu_defaults():
    assert HybridModeConfig(mode="siso").resolved_chunk_size() == 512
    assert HybridModeConfig(mode="mimo", mimo_rank=4).resolved_chunk_size() == 256
    assert mamba3_tpu_default_chunk_size("siso") == 512
    assert mamba3_tpu_default_chunk_size("mimo", mimo_rank=4) == 256


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


def test_chunked_xla_scan_fused_auto_path_preserves_carry_dtype():
    dt, lam, a, b, c, x = _sample_chunked_inputs(
        leading_shape=(2,),
        num_chunks=3,
        chunk_size=8,
        state_dim=4,
        value_dim=8,
    )
    bf16_inputs = tuple(arr.astype(jnp.bfloat16) for arr in (dt, lam, a, b, c, x))
    _, final_state = mamba3_chunked_forward(*bf16_inputs, implementation="xla")
    assert final_state.dtype == jnp.bfloat16


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


def test_mamba3_hybrid_siso_matches_siso_api():
    dt, lam, a, b, c, x = _sample_chunked_inputs(
        leading_shape=(2,),
        num_chunks=2,
        chunk_size=8,
        state_dim=4,
        value_dim=6,
    )
    y_hybrid, state_hybrid = mamba3_hybrid_chunked_forward(dt, lam, a, b, c, x, mode="siso", implementation="xla")
    y_siso, state_siso = mamba3_chunked_forward(dt, lam, a, b, c, x, implementation="xla")
    assert jnp.allclose(y_hybrid, y_siso, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(state_hybrid, state_siso, atol=1e-5, rtol=1e-5)


def test_mamba3_hybrid_siso_from_transformed_matches_siso_api():
    dt, lam, a, b, c, x = _sample_chunked_inputs(
        leading_shape=(2,),
        num_chunks=2,
        chunk_size=8,
        state_dim=4,
        value_dim=6,
    )
    src_scale, out_correction = prepare_mamba3_chunked_scales(dt, lam)
    a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
    y_hybrid, state_hybrid = mamba3_hybrid_chunked_forward_from_transformed(
        a_log_cumsum,
        src_scale,
        out_correction,
        b,
        c,
        x,
        mode="siso",
        implementation="xla",
    )
    y_siso, state_siso = mamba3_chunked_forward_from_transformed(
        a_log_cumsum,
        src_scale,
        out_correction,
        b,
        c,
        x,
        implementation="xla",
    )
    assert jnp.allclose(y_hybrid, y_siso, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(state_hybrid, state_siso, atol=1e-5, rtol=1e-5)


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


@skip_if_no_torch
def test_mamba3_siso_reference_matches_upstream_torch_step_reference():
    import torch

    dt, lam, a, b, c, x = _sample_chunked_inputs(
        leading_shape=(1,),
        num_chunks=2,
        chunk_size=4,
        state_dim=6,
        value_dim=5,
    )
    groups = dt.shape[0]
    seq_len = dt.shape[1] * dt.shape[2]
    dt_flat = dt.reshape(groups, seq_len)
    lam_flat = lam.reshape(groups, seq_len)
    a_flat = a.reshape(groups, seq_len)
    b_flat = b.reshape(groups, seq_len, b.shape[-1])
    c_flat = c.reshape(groups, seq_len, c.shape[-1])
    x_flat = x.reshape(groups, seq_len, x.shape[-1])

    y_ref, state_ref = mamba3_direct_recurrence_reference_batched(dt, lam, a, b, c, x)

    dtype = torch.float32
    q_t = torch.from_numpy(np.array(c_flat))[:, :, None, :].to(dtype)
    k_t = torch.from_numpy(np.array(b_flat))[:, :, None, :].to(dtype)
    v_t = torch.from_numpy(np.array(x_flat))[:, :, None, :].to(dtype)
    adt_t = torch.from_numpy(np.array(local_log_alpha(dt_flat, a_flat)))[:, None, :].to(dtype)
    dt_t = torch.from_numpy(np.array(dt_flat))[:, None, :].to(dtype)
    trap_t = _torch_logit(torch.from_numpy(np.array(lam_flat))[:, None, :].to(dtype))
    q_bias_t = torch.zeros((1, b.shape[-1]), dtype=dtype)
    k_bias_t = torch.zeros((1, b.shape[-1]), dtype=dtype)
    angles_t = torch.zeros((groups, seq_len, 1, b.shape[-1] // 2), dtype=dtype)

    y_upstream, state_upstream = _upstream_mamba3_siso_step_ref_torch(
        q_t,
        k_t,
        v_t,
        adt_t,
        dt_t,
        trap_t,
        q_bias_t,
        k_bias_t,
        angles_t,
    )

    np.testing.assert_allclose(
        np.array(y_ref).reshape(groups, seq_len, x.shape[-1]),
        y_upstream[:, :, 0, :].detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(state_ref),
        state_upstream[:, 0, :, :].detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


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


def test_mamba3_mimo_rank1_matches_siso_ranked_core():
    dt, lam, a, b, c, x_base, _, w_x, _, _ = _sample_mimo_chunked_inputs(
        leading_shape=(2,),
        num_chunks=3,
        chunk_size=8,
        state_dim=4,
        value_dim=6,
        rank=1,
    )
    src_scale, out_correction = prepare_mamba3_chunked_scales(dt, lam)
    a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
    x_ranked = mamba3_mimo_rank_expand(x_base, w_x)

    y_mimo, state_mimo = mamba3_mimo_chunked_forward_ranked_xla_batched(
        a_log_cumsum.reshape(2, 3, 8),
        src_scale.reshape(2, 3, 8),
        out_correction.reshape(2, 3, 8),
        b.reshape(2, 3, 8, 4, 1),
        c.reshape(2, 3, 8, 4, 1),
        x_ranked.reshape(2, 3, 8, 6, 1),
    )
    y_siso, state_siso = mamba3_chunked_forward(
        dt,
        lam,
        a,
        b[..., 0],
        c[..., 0],
        x_ranked[..., 0],
        implementation="xla",
    )
    assert jnp.allclose(y_mimo[..., 0], y_siso, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(state_mimo, state_siso, atol=1e-5, rtol=1e-5)


def test_mamba3_mimo_chunked_xla_matches_direct_ranked_recurrence():
    dt, lam, a, b, c, x_base, _, w_x, _, _ = _sample_mimo_chunked_inputs(
        leading_shape=(2,),
        num_chunks=3,
        chunk_size=8,
        state_dim=4,
        value_dim=5,
        rank=3,
    )
    src_scale, out_correction = prepare_mamba3_chunked_scales(dt, lam)
    a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
    x_ranked = mamba3_mimo_rank_expand(x_base, w_x)
    y_xla, state_xla = mamba3_mimo_chunked_forward_ranked_xla_batched(
        a_log_cumsum.reshape(2, 3, 8),
        src_scale.reshape(2, 3, 8),
        out_correction.reshape(2, 3, 8),
        b.reshape(2, 3, 8, 4, 3),
        c.reshape(2, 3, 8, 4, 3),
        x_ranked.reshape(2, 3, 8, 5, 3),
    )
    y_ref, state_ref = mamba3_mimo_direct_recurrence_ranked_reference_batched(
        dt.reshape(2, 3, 8),
        lam.reshape(2, 3, 8),
        a.reshape(2, 3, 8),
        b.reshape(2, 3, 8, 4, 3),
        c.reshape(2, 3, 8, 4, 3),
        x_ranked.reshape(2, 3, 8, 5, 3),
    )
    assert jnp.allclose(
        y_xla,
        y_ref,
        atol=MAMBA3_MIMO_RANKED_PARITY_ATOL,
        rtol=MAMBA3_MIMO_RANKED_PARITY_RTOL,
    )
    assert jnp.allclose(state_xla, state_ref, atol=1e-5, rtol=1e-5)


def test_mamba3_mimo_chunked_reference_matches_transformed_ranked_oracle():
    dt, lam, a, b, c, x_base, _, w_x, _, _ = _sample_mimo_chunked_inputs(
        leading_shape=(2,),
        num_chunks=3,
        chunk_size=8,
        state_dim=4,
        value_dim=5,
        rank=3,
    )
    src_scale, out_correction = prepare_mamba3_chunked_scales(dt, lam)
    a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
    x_ranked = mamba3_mimo_rank_expand(x_base, w_x)
    y_shim, state_shim = mamba3_mimo_chunked_forward_ranked_reference_batched(
        a_log_cumsum.reshape(2, 3, 8),
        src_scale.reshape(2, 3, 8),
        out_correction.reshape(2, 3, 8),
        b.reshape(2, 3, 8, 4, 3),
        c.reshape(2, 3, 8, 4, 3),
        x_ranked.reshape(2, 3, 8, 5, 3),
    )
    y_seq, state_seq = mamba3_mimo_chunked_sequential_ranked_reference_batched(
        a_log_cumsum.reshape(2, 3, 8),
        src_scale.reshape(2, 3, 8),
        out_correction.reshape(2, 3, 8),
        b.reshape(2, 3, 8, 4, 3),
        c.reshape(2, 3, 8, 4, 3),
        x_ranked.reshape(2, 3, 8, 5, 3),
    )
    assert jnp.allclose(
        y_shim,
        y_seq,
        atol=MAMBA3_MIMO_RANKED_PARITY_ATOL,
        rtol=MAMBA3_MIMO_RANKED_PARITY_RTOL,
    )
    assert jnp.allclose(state_shim, state_seq, atol=1e-5, rtol=1e-5)


def test_mamba3_mimo_ranked_matches_r_squared_siso_decomposition():
    dt, lam, a, b, c, x_base, _, w_x, _, _ = _sample_mimo_chunked_inputs(
        leading_shape=(1,),
        num_chunks=2,
        chunk_size=6,
        state_dim=3,
        value_dim=4,
        rank=3,
    )
    src_scale, out_correction = prepare_mamba3_chunked_scales(dt, lam)
    a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
    x_ranked = mamba3_mimo_rank_expand(x_base, w_x)
    y_mimo, _ = mamba3_mimo_chunked_forward_ranked_reference_batched(
        a_log_cumsum.reshape(1, 2, 6),
        src_scale.reshape(1, 2, 6),
        out_correction.reshape(1, 2, 6),
        b.reshape(1, 2, 6, 3, 3),
        c.reshape(1, 2, 6, 3, 3),
        x_ranked.reshape(1, 2, 6, 4, 3),
    )
    y_sum = jnp.zeros_like(y_mimo)
    for out_rank in range(3):
        partial = []
        for in_rank in range(3):
            y_pair, _ = mamba3_chunked_forward_reference_batched(
                a_log_cumsum.reshape(1, 2, 6),
                src_scale.reshape(1, 2, 6),
                out_correction.reshape(1, 2, 6),
                b[..., in_rank].reshape(1, 2, 6, 3),
                c[..., out_rank].reshape(1, 2, 6, 3),
                x_ranked[..., in_rank].reshape(1, 2, 6, 4),
            )
            partial.append(y_pair)
        y_sum = y_sum.at[..., out_rank].set(sum(partial))
    assert jnp.allclose(y_mimo, y_sum, atol=1e-5, rtol=1e-5)


def test_mamba3_mimo_public_api_matches_reference_and_has_siso_carry_shape():
    inputs = _sample_mimo_chunked_inputs(
        leading_shape=(2,),
        num_chunks=2,
        chunk_size=8,
        state_dim=4,
        value_dim=5,
        rank=3,
    )
    y_xla, state_xla = mamba3_mimo_chunked_forward(*inputs, implementation="xla")
    y_ref, state_ref = mamba3_mimo_chunked_forward(*inputs, implementation="reference")
    assert y_xla.shape == (2, 2, 8, 5)
    assert state_xla.shape == (2, 5, 4)
    assert jnp.allclose(y_xla, y_ref, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(state_xla, state_ref, atol=1e-5, rtol=1e-5)


def test_mamba3_hybrid_mimo_matches_mimo_api():
    inputs = _sample_mimo_chunked_inputs(
        leading_shape=(2,),
        num_chunks=2,
        chunk_size=8,
        state_dim=4,
        value_dim=5,
        rank=4,
    )
    y_hybrid, state_hybrid = mamba3_hybrid_chunked_forward(
        *inputs[:5],
        inputs[5],
        mode="mimo",
        z=inputs[6],
        w_x=inputs[7],
        w_z=inputs[8],
        w_o=inputs[9],
        implementation="xla",
    )
    y_mimo, state_mimo = mamba3_mimo_chunked_forward(*inputs, implementation="xla")
    assert jnp.allclose(y_hybrid, y_mimo, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(state_hybrid, state_mimo, atol=1e-5, rtol=1e-5)


def test_mamba3_hybrid_mimo_from_transformed_matches_mimo_api():
    inputs = _sample_mimo_chunked_inputs(
        leading_shape=(2,),
        num_chunks=2,
        chunk_size=8,
        state_dim=4,
        value_dim=5,
        rank=4,
    )
    dt, lam, a, b, c, x_base, z_base, w_x, w_z, w_o = inputs
    src_scale, out_correction = prepare_mamba3_chunked_scales(dt, lam)
    a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
    y_hybrid, state_hybrid = mamba3_hybrid_chunked_forward_from_transformed(
        a_log_cumsum,
        src_scale,
        out_correction,
        b,
        c,
        x_base,
        mode="mimo",
        z=z_base,
        w_x=w_x,
        w_z=w_z,
        w_o=w_o,
        implementation="xla",
    )
    y_mimo, state_mimo = mamba3_mimo_chunked_forward(
        dt,
        lam,
        a,
        b,
        c,
        x_base,
        z_base,
        w_x,
        w_z,
        w_o,
        implementation="xla",
    )
    assert jnp.allclose(y_hybrid, y_mimo, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(state_hybrid, state_mimo, atol=1e-5, rtol=1e-5)


def test_mamba3_mimo_grad_matches_reference():
    dt, lam, a, b, c, x_base, z_base, w_x, w_z, w_o = _sample_mimo_chunked_inputs(
        leading_shape=(),
        num_chunks=2,
        chunk_size=4,
        state_dim=3,
        value_dim=4,
        rank=2,
    )

    def loss_ref(x_in: jax.Array) -> jax.Array:
        y, _ = mamba3_mimo_chunked_forward(
            dt,
            lam,
            a,
            b,
            c,
            x_in.reshape(2, 4, 4),
            z_base,
            w_x,
            w_z,
            w_o,
            implementation="reference",
        )
        return jnp.sum(y)

    def loss_xla(x_in: jax.Array) -> jax.Array:
        y, _ = mamba3_mimo_chunked_forward(
            dt,
            lam,
            a,
            b,
            c,
            x_in.reshape(2, 4, 4),
            z_base,
            w_x,
            w_z,
            w_o,
            implementation="xla",
        )
        return jnp.sum(y)

    g_ref = jax.grad(loss_ref)(x_base)
    g_xla = jax.grad(loss_xla)(x_base)
    assert jnp.allclose(g_ref, g_xla, atol=1e-5, rtol=1e-5)


@skip_if_no_torch
def test_mamba3_mimo_reference_matches_upstream_torch_step_reference():
    import torch

    dt, lam, a, b, c, x_base, z_base, w_x, w_z, w_o = _sample_mimo_chunked_inputs(
        leading_shape=(),
        num_chunks=2,
        chunk_size=4,
        state_dim=6,
        value_dim=5,
        rank=4,
    )
    groups = 1
    seq_len = dt.shape[0] * dt.shape[1]
    dt_grouped = dt[None, ...]
    lam_grouped = lam[None, ...]
    a_grouped = a[None, ...]
    b_grouped = b[None, ...]
    c_grouped = c[None, ...]
    x_base_grouped = x_base[None, ...]
    z_base_grouped = z_base[None, ...]
    dt_flat = dt_grouped.reshape(groups, seq_len)
    lam_flat = lam_grouped.reshape(groups, seq_len)
    a_flat = a_grouped.reshape(groups, seq_len)
    b_flat = b_grouped.reshape(groups, seq_len, b.shape[-2], b.shape[-1])
    c_flat = c_grouped.reshape(groups, seq_len, c.shape[-2], c.shape[-1])
    x_base_flat = x_base_grouped.reshape(groups, seq_len, x_base.shape[-1])
    z_base_flat = z_base_grouped.reshape(groups, seq_len, z_base.shape[-1])

    y_ref, state_ref = mamba3_mimo_direct_recurrence_reference_batched(
        dt_grouped,
        lam_grouped,
        a_grouped,
        b_grouped,
        c_grouped,
        x_base_grouped,
        z_base_grouped,
        w_x,
        w_z,
        w_o,
    )

    dtype = torch.float32
    q_t = torch.from_numpy(np.array(c_flat).transpose(0, 1, 3, 2))[:, :, :, None, :].to(dtype)
    k_t = torch.from_numpy(np.array(b_flat).transpose(0, 1, 3, 2))[:, :, :, None, :].to(dtype)
    v_t = torch.from_numpy(np.array(x_base_flat))[:, :, None, :].to(dtype)
    z_t = torch.from_numpy(np.array(z_base_flat))[:, :, None, :].to(dtype)
    adt_t = torch.from_numpy(np.array(local_log_alpha(dt_flat, a_flat)))[:, None, :].to(dtype)
    dt_t = torch.from_numpy(np.array(dt_flat))[:, None, :].to(dtype)
    trap_t = _torch_logit(torch.from_numpy(np.array(lam_flat))[:, None, :].to(dtype))
    q_bias_t = torch.zeros((1, b.shape[-1], b.shape[-2]), dtype=dtype)
    k_bias_t = torch.zeros((1, b.shape[-1], b.shape[-2]), dtype=dtype)
    angles_t = torch.zeros((groups, seq_len, 1, b.shape[-2] // 2), dtype=dtype)
    mimo_v_t = torch.from_numpy(np.array(w_x).transpose(1, 0))[None, :, :].to(dtype)
    mimo_z_t = torch.from_numpy(np.array(w_z).transpose(1, 0))[None, :, :].to(dtype)
    mimo_o_t = torch.from_numpy(np.array(w_o).transpose(1, 0))[None, :, :].to(dtype)

    y_upstream, state_upstream = _upstream_mamba3_mimo_step_ref_torch(
        q_t,
        k_t,
        v_t,
        adt_t,
        dt_t,
        trap_t,
        q_bias_t,
        k_bias_t,
        angles_t,
        mimo_v_t,
        mimo_o_t,
        z_t,
        mimo_z_t,
    )

    np.testing.assert_allclose(
        np.array(y_ref).reshape(groups, seq_len, x_base.shape[-1]),
        y_upstream[:, :, 0, :].detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(state_ref),
        state_upstream[:, 0, :, :].detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_mamba3_mimo_tpu_aligned_smoke_shape_compiles_under_jit():
    inputs = _sample_mimo_chunked_inputs(
        leading_shape=(1,),
        num_chunks=2,
        chunk_size=128,
        state_dim=64,
        value_dim=128,
        rank=4,
    )
    fn = jax.jit(lambda *args: mamba3_mimo_chunked_forward(*args, implementation="xla"))
    y, state = fn(*[arg.astype(jnp.bfloat16) if i < 7 else arg.astype(jnp.bfloat16) for i, arg in enumerate(inputs)])
    assert y.shape == (1, 2, 128, 128)
    assert state.shape == (1, 128, 64)
