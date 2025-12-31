# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# based on:
# - https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py
# - the JAX implementation by Yu Sun and Leo Lee
# - Flash Linear Attention's Triton implementation: https://github.com/fla-org/flash-linear-attention

"""
This module implements the Gated DeltaNet: https://arxiv.org/abs/2412.06464.
It exposes:
  - recurrent_gated_delta_rule: the sequential (decode) rule
  - chunk_gated_delta_rule:    the chunkwise-parallel (prefill / train) rule
  - GatedDeltaNet:             a full layer that wraps projections, a small
                               depthwise causal conv over [Q|K|V], the kernels,
                               and the gated RMSNorm + output projection.

Core update (rectangular state S ∈ R^{d_k × d_v}):
  S_t = α_t S_{t-1} + β_t (v_t - S_{t-1} k_t) k_t^T
  o_t = S_t^T q_t

where:
  α_t = exp(g_t) ∈ (0,1)   (forget/decay gate, log-parameterized by g_t ≤ 0)
  β_t = σ(b_t) ∈ (0,1)     (learning-rate gate)

Follows the GDN implementation for Qwen3-Next. Notably most math is performed in fp32.
"""

from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import functools
import os
from typing import Optional, Tuple, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax._src.state.indexing import dslice

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray

_GDN_DBG_SHARDING = bool(int(os.environ.get("GDN_DEBUG_SHARDING", "0")))


def _dbg(tag: str, arr):
    if not _GDN_DBG_SHARDING:
        return
    try:
        jax.debug.inspect_array_sharding(arr, callback=lambda s: print(f"[GDN][internal] {tag}: {s}"))
    except Exception:
        pass


# ---------- small utilities ----------


def _l2norm(x: NamedArray, axis: hax.AxisSelector, eps: float = 1e-6) -> NamedArray:
    """L2-normalize x along a named axis.

    Args:
        x: NamedArray of any shape.
        axis: the single axis to normalize along (e.g., the head dimension Dk).
    """
    x32 = x.astype(jnp.float32)
    inv = hax.rsqrt(hax.sum(hax.square(x32), axis=axis) + jnp.asarray(eps, dtype=jnp.float32))
    return (x32 * inv).astype(x.dtype)


def _rmsnorm_gated_reference(
    x_2d: jnp.ndarray,
    gate_2d: jnp.ndarray,
    weight: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    """Fallback RMSNorm + SiLU gate."""

    x32 = x_2d.astype(jnp.float32)
    gate32 = gate_2d.astype(jnp.float32)
    weight32 = weight.astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + jnp.asarray(eps, dtype=jnp.float32))
    y32 = x32 * inv * weight32[None, :]
    gated32 = y32 * jax.nn.silu(gate32)
    return gated32.astype(x_2d.dtype)


def _fused_rmsnorm_gated_pallas(
    x_2d: jnp.ndarray,
    gate_2d: jnp.ndarray,
    weight: jnp.ndarray,
    eps: float,
    *,
    BM: int = 8,  # 8 is the minimum "nice" TPU multiple
) -> jnp.ndarray:
    n_rows, hidden_size = x_2d.shape

    # TPU blocked-indexing constraints: BM should be divisible by 8 unless it equals n_rows
    if BM % 8 != 0:
        raise ValueError("BM must be a multiple of 8 on TPU")

    # Pad only the OUTPUT shape so stores are always in-bounds.
    n_rows_pad = ((n_rows + BM - 1) // BM) * BM

    def kernel(x_ref, gate_ref, w_ref, out_ref):
        pid = pl.program_id(axis=0)  # block id along rows
        row_ids = pid * BM + jnp.arange(BM)

        # Cast boolean mask to float32 immediately
        row_mask_bool = row_ids < n_rows  # (BM,)
        mask_f32 = row_mask_bool.astype(jnp.float32)[:, None]  # (BM, 1)

        # Load the whole (BM, D) tile; OOB rows may be garbage => mask them out
        x = x_ref[:, :].astype(jnp.float32)
        g = gate_ref[:, :].astype(jnp.float32)

        x = x * mask_f32
        g = g * mask_f32

        w = w_ref[:].astype(jnp.float32)

        scale = 1.0 / hidden_size

        # RMS per row
        ss = jnp.sum(x * x, axis=-1, keepdims=True) * scale  # (BM, 1)
        inv = lax.rsqrt(ss + eps)  # (BM, 1)

        y = x * inv * w[None, :]  # (BM, D)
        out = y * jax.nn.silu(g)

        # Optional: keep padded rows deterministic
        out = out * mask_f32

        out_ref[:, :] = out.astype(out_ref.dtype)

    # TPU compiler params (optional)
    compiler_params = None
    try:
        from jax.experimental.pallas import tpu as pltpu

        compiler_params = pltpu.CompilerParams(dimension_semantics=["parallel"])
    except Exception:
        pass

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n_rows_pad, hidden_size), x_2d.dtype),
        grid=(n_rows_pad // BM,),
        in_specs=[
            # IMPORTANT: index_map returns BLOCK indices; do NOT multiply by BM.
            pl.BlockSpec((BM, hidden_size), lambda pid: (pid, 0)),
            pl.BlockSpec((BM, hidden_size), lambda pid: (pid, 0)),
            pl.BlockSpec((hidden_size,), lambda pid: (0,)),
        ],
        out_specs=pl.BlockSpec((BM, hidden_size), lambda pid: (pid, 0)),
        compiler_params=compiler_params,
    )(x_2d, gate_2d, weight)

    return out[:n_rows, :]


# ---------- depthwise conv: positional (lax) helpers with named wrappers ----------


def _causal_depthwise_conv1d_full(
    x_ncl: jnp.ndarray, w_ck: jnp.ndarray, bias_c: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """Depthwise 1D convolution with *causal* semantics (left padding).

    Shapes:
      x_ncl: (N, C, L)  - batch, channels, length
      w_ck:  (C, K)     - per-channel (depthwise) filter of length K
      bias:  (C,)       - optional per-channel bias
      return: (N, C, L)

    DimensionNumbers ("NCH","OIH","NCH") means:
    - lhs (x):    N=0, C=1, H=2
    - rhs (w):    O=0, I=1, H=2  (we inject a singleton I=1 for depthwise)
    - out:        N=0, C=1, H=2
    """
    in_dtype = x_ncl.dtype
    N, C, L = x_ncl.shape
    K = w_ck.shape[-1]
    # pad x on the left with K-1 zeros so that output length == L ("causal")
    x_pad = jnp.pad(x_ncl, ((0, 0), (0, 0), (K - 1, 0)))

    # Upcast both sides to float32 for conv
    x32 = x_pad.astype(jnp.float32)
    w32 = w_ck.astype(jnp.float32)
    w_oik = w32[:, None, :]  # (C, 1, K)

    y32 = lax.conv_general_dilated(
        lhs=x32,
        rhs=w_oik,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NCH", "OIH", "NCH"),
        feature_group_count=C,  # depthwise
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    _dbg("conv/full/y32", y32)

    if bias_c is not None:
        y32 = y32 + bias_c.astype(jnp.float32)[:, None]

    y32 = jax.nn.silu(y32)
    return y32.astype(in_dtype)


def _causal_depthwise_conv1d_update(
    x_ncl_1: jnp.ndarray,  # (N, C, 1)
    w_ck: jnp.ndarray,  # (C, K)
    bias_c: Optional[jnp.ndarray],
    prev_state_nck: jnp.ndarray,  # (N, C, K)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single-step streaming update for the causal depthwise conv.

    Args:
      x_ncl_1: (N, C, 1) current step input
      w_ck:    (C, K)    depthwise kernel
      bias_c:  (C,)      optional bias
      prev_state_nck: (N, C, K) left context state (the last K inputs)

    Returns:
      y: (N, C, 1)   the latest convolved sample
      new_state: (N, C, K) with the newest x appended on the right

    Used during decode to avoid re-convolving the entire history.
    """
    in_dtype = x_ncl_1.dtype

    x_hist = jnp.concatenate([prev_state_nck, x_ncl_1], axis=-1)
    x32 = x_hist.astype(jnp.float32)
    w32 = w_ck.astype(jnp.float32)

    y32_all = lax.conv_general_dilated(
        lhs=x32,
        rhs=w32[:, None, :],
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NCH", "OIH", "NCH"),
        feature_group_count=w32.shape[0],
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    y32 = y32_all[..., -1:]
    _dbg("conv/update/y32", y32)

    if bias_c is not None:
        y32 = y32 + bias_c.astype(jnp.float32)[:, None]

    y32 = jax.nn.silu(y32)
    new_state = jnp.concatenate([prev_state_nck[..., 1:], x_ncl_1], axis=-1)

    return y32.astype(in_dtype), new_state.astype(in_dtype)


# ---------- Fused Gated RMSNorm ----------


@jax.custom_vjp
def _rmsnorm_gated_flash(x_2d: jnp.ndarray, gate_2d: jnp.ndarray, weight: jnp.ndarray, eps: float) -> jnp.ndarray:
    """Forward: call the Pallas fused kernel (fallback to reference).
    Backward: analytic grads in pure JAX (no Pallas autodiff)."""
    return _fused_rmsnorm_gated_pallas(x_2d, gate_2d, weight, eps)


def _rmsnorm_gated_flash_fwd(x_2d, gate_2d, weight, eps):
    y = _rmsnorm_gated_flash(x_2d, gate_2d, weight, eps)
    # Keep inputs as residuals; recompute inv etc. in bwd
    return y, (x_2d, gate_2d, weight, jnp.asarray(eps, dtype=jnp.float32))


def _rmsnorm_gated_flash_bwd(res, dY):
    x_2d, gate_2d, weight, eps32 = res
    # Promote to fp32
    x = x_2d.astype(jnp.float32)
    gate = gate_2d.astype(jnp.float32)
    w = weight.astype(jnp.float32)
    dY = dY.astype(jnp.float32)

    # Row-wise RMSNorm stats
    # inv = 1 / sqrt(mean(x^2) + eps)
    D = x.shape[-1]
    inv = lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps32)  # [N,1]
    y_no_gate = (x * inv) * w  # [N,D]

    # SiLU and derivative
    sigma = jax.nn.sigmoid(gate)  # [N,D]
    silu = gate * sigma
    silu_prime = sigma * (1.0 + gate * (1.0 - sigma))  # [N,D]

    # Chain rule
    dy = dY * silu  # [N,D]
    # y_no_gate = (x*inv) * w = u * w, with u = x*inv
    du = dy * w  # [N,D]

    # dweight: sum over rows of dy * (x * inv)
    dW = jnp.sum(dy * (x * inv), axis=0)  # [D]

    # dx: u = x*inv  with inv = (mean(x^2)+eps)^(-1/2)
    # dL/dx = inv * du + (-inv^3 / D) * x * dot(du, x), where dot is along last dim
    dot = jnp.sum(du * x, axis=-1, keepdims=True)  # [N,1]
    dx = inv * du + (-(inv**3) / float(D)) * x * dot  # [N,D]

    # dgate: elementwise (no reduction)
    dGate = dY * y_no_gate * silu_prime  # [N,D]

    # Cast back to input dtypes
    return (dx.astype(x_2d.dtype), dGate.astype(gate_2d.dtype), dW.astype(weight.dtype), None)  # no grad for eps


_rmsnorm_gated_flash.defvjp(_rmsnorm_gated_flash_fwd, _rmsnorm_gated_flash_bwd)


class FusedRMSNormGated(eqx.Module):
    """RMSNorm(x) * SiLU(gate) using an optional fused Pallas kernel."""

    axis: Axis
    weight: NamedArray  # [axis]
    eps: float = eqx.field(default=1e-6, static=True)
    use_flash: bool = eqx.field(default=True, static=True)

    @staticmethod
    def init(axis: Axis, eps: float = 1e-6, *, use_flash: bool = True) -> "FusedRMSNormGated":
        return FusedRMSNormGated(axis=axis, weight=hax.ones(axis), eps=eps, use_flash=use_flash)

    def __call__(self, x: NamedArray, gate: NamedArray) -> NamedArray:
        if x.resolve_axis(self.axis.name) != gate.resolve_axis(self.axis.name):
            raise ValueError("x and gate must share the normalization axis")

        # Move target axis to the end to make the flattened 2D view contiguous
        other_axes = tuple(ax for ax in x.axes if ax.name != self.axis.name)
        permuted_axes = other_axes + (self.axis,)
        x_perm = hax.rearrange(x, permuted_axes)
        gate_perm = hax.rearrange(gate, permuted_axes)

        x_arr = x_perm.array.reshape(-1, self.axis.size)
        gate_arr = gate_perm.array.reshape(-1, self.axis.size)
        weight_arr = self.weight.array

        if self.use_flash:
            out_arr = _rmsnorm_gated_flash(x_arr, gate_arr, weight_arr, self.eps)
        else:
            out_arr = _rmsnorm_gated_reference(x_arr, gate_arr, weight_arr, self.eps)

        out_perm = out_arr.reshape(x_perm.array.shape)
        out_named = hax.named(out_perm, permuted_axes)
        return hax.rearrange(out_named, x.axes)


# ---------- Config ----------


@dataclass(frozen=True)
class GatedDeltaNetConfig:
    """Configuration for a GDN block (per layer).

    Head layout:
      - num_k_heads * head_k_dim = key_dim
      - num_v_heads * head_v_dim = value_dim
      - Keys/queries may have different head count/dim from values (rectangular S).

    Conv:
      - Small depthwise causal conv over concatenated channels [Q|K|V].
    """

    Embed: Axis
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int = 4
    rms_norm_eps: float = 1e-6

    @property
    def KHeads(self) -> Axis:
        return Axis("k_heads", self.num_k_heads)

    @property
    def VHeads(self) -> Axis:
        return Axis("v_heads", self.num_v_heads)

    @property
    def Heads(self) -> Axis:
        # expose VHeads as heads for tensor-parallel sharding
        return Axis("heads", self.num_v_heads)

    @property
    def KHeadDim(self) -> Axis:
        return Axis("k_head_dim", self.head_k_dim)

    @property
    def VHeadDim(self) -> Axis:
        return Axis("v_head_dim", self.head_v_dim)

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def mix_qkvz_axis(self) -> Axis:
        # [Q | K | V | Z]; the layer projects all at once
        return Axis("qkvz", self.key_dim * 2 + self.value_dim * 2)

    @property
    def ba_axis(self) -> Axis:
        # [b | a]; per value head: β = σ(b), g uses a via Mamba2-style discretization
        return Axis("ba", self.num_v_heads * 2)


# ---------- Kernels ----------


def _recurrent_gated_delta_rule_reference(
    query: NamedArray,  # [batch, position, heads, k_head_dim]
    key: NamedArray,  # [batch, position, heads, k_head_dim]
    value: NamedArray,  # [batch, position, heads, v_head_dim]
    g: NamedArray,  # [batch, position, heads] (log-decay; α = exp(g))
    beta: NamedArray,  # [batch, position, heads] (β ∈ (0,1))
    *,
    initial_state: Optional[jnp.ndarray] = None,  # (B, H, dk, dv)
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> Tuple[NamedArray, Optional[jnp.ndarray]]:
    """Sequential (decode) GDN kernel

    For each t:
      α_t = exp(g_t)
      kv_t = S_{t-1}^T k_t             # shape: [B, H, d_v]
      δ_t  = β_t * (v_t - kv_t)        # [B, H, d_v]
      S_t  = α_t S_{t-1} + k_t δ_t^T   # [B, H, d_k, d_v]
      o_t  = S_t^T q_t                 # [B, H, d_v] (readout)

    Args:
      query, key, value: NamedArray tensors with explicit [batch, position, heads, dim]
      g:   log-decay; α = exp(g) is the forget gate in (0,1)
      beta: learning-rate gate β in (0,1)
      initial_state: optional S_0 (B, H, d_k, d_v)
      output_final_state: whether to return S_T
      use_qk_l2norm_in_kernel: if True, L2-normalize Q,K and scale Q by 1/sqrt(d_k)

    Returns:
      outputs: [batch, position, heads, v_head_dim]
      final_state (optional): (B, H, d_k, d_v)
    """
    # ---- axes ----
    Batch = query.resolve_axis("batch")
    Pos = query.resolve_axis("position")
    Heads = query.resolve_axis("heads")
    Dk = query.resolve_axis("k_head_dim")
    Dv = value.resolve_axis("v_head_dim")

    # ---- promote & normalize ----
    q = query.astype(jnp.float32)
    k = key.astype(jnp.float32)
    v = value.astype(jnp.float32)
    b = beta.astype(jnp.float32)
    gg = g.astype(jnp.float32)

    if use_qk_l2norm_in_kernel:
        q = _l2norm(q, axis=Dk)
        k = _l2norm(k, axis=Dk)
    q = q * (Dk.size**-0.5)  # 1/sqrt(d_k) scaling

    # Prepare initial S
    B_, H_, L_, dk_, dv_ = Batch.size, Heads.size, Pos.size, Dk.size, Dv.size
    S0 = jnp.zeros((B_, H_, dk_, dv_), dtype=v.dtype) if initial_state is None else initial_state.astype(v.dtype)
    _dbg("recurrent/S0", S0)

    # Re-layout to positional major for lax.scan
    q_bhld = hax.rearrange(q, (Batch, Heads, Pos, Dk)).array  # (B,H,L,d_k)
    k_bhld = hax.rearrange(k, (Batch, Heads, Pos, Dk)).array
    v_bhld = hax.rearrange(v, (Batch, Heads, Pos, Dv)).array
    g_bhl = hax.rearrange(gg, (Batch, Heads, Pos)).array  # (B,H,L)
    b_bhl = hax.rearrange(b, (Batch, Heads, Pos)).array

    def step(S_prev_arr, xs_arr):
        # Unwrap per-step slices as NamedArrays for axis-safe math
        q_t_arr, k_t_arr, v_t_arr, g_t_arr, b_t_arr = xs_arr
        S_prev = hax.named(S_prev_arr, (Batch, Heads, Dk, Dv))
        q_t = hax.named(q_t_arr, (Batch, Heads, Dk))
        k_t = hax.named(k_t_arr, (Batch, Heads, Dk))
        v_t = hax.named(v_t_arr, (Batch, Heads, Dv))
        g_t = hax.named(g_t_arr, (Batch, Heads))
        b_t = hax.named(b_t_arr, (Batch, Heads))

        # Decay: S ← α_t S  (α_t = exp(g_t))
        decay = hax.exp(g_t).broadcast_axis(Dk).broadcast_axis(Dv)
        S_prev = S_prev * decay

        # Prediction kv_t = S^T k_t  (i.e., along Dk)
        kv = hax.dot(S_prev * k_t.broadcast_axis(Dv), axis=Dk)  # [B,H,Dv]

        # Rank-1 delta update and state write
        delta = (v_t - kv) * b_t.broadcast_axis(Dv)  # [B,H,Dv]
        S_new = S_prev + k_t.broadcast_axis(Dv) * delta.broadcast_axis(Dk)

        # Readout: o_t = S^T q_t
        y_t = hax.dot(S_new * q_t.broadcast_axis(Dv), axis=Dk)  # [B,H,Dv]
        return S_new.array, y_t.array

    S_final, out_seq = jax.lax.scan(
        step,
        S0,
        (
            jnp.moveaxis(q_bhld, 2, 0),  # time-major
            jnp.moveaxis(k_bhld, 2, 0),
            jnp.moveaxis(v_bhld, 2, 0),
            jnp.moveaxis(g_bhl, 2, 0),
            jnp.moveaxis(b_bhl, 2, 0),
        ),
        length=L_,
    )

    # Back to [B, Pos, H, Dv]
    out_bhlv = jnp.moveaxis(out_seq, 0, 2)  # (B,H,L,Dv)
    out_bhlv = hax.named(out_bhlv, (Batch, Heads, Pos, Dv))
    out_final = hax.rearrange(out_bhlv, (Batch, Pos, Heads, Dv))
    _dbg("recurrent/out", out_final.array)

    if output_final_state:
        return out_final, S_final
    else:
        return out_final, None


def _pick_bk_tile_for_decode(dk: int) -> int:
    # Simple heuristic; autotune later if desired.
    if dk >= 256:
        return 64
    elif dk >= 128:
        return 64
    else:
        return 32


def _pick_bv_tile_for_decode(dv: int) -> int:
    if dv >= 512:
        return 128
    elif dv >= 256:
        return 64
    else:
        return 32


def _pad_axis_right(arr: jnp.ndarray, axis: int, new_size: int) -> jnp.ndarray:
    """Right-pad `axis` to `new_size` without reshaping other axes."""
    axis = axis % arr.ndim
    cur = arr.shape[axis]
    if cur == new_size:
        return arr
    if cur > new_size:
        raise ValueError(f"Cannot pad axis {axis} from {cur} down to {new_size}")
    pad = new_size - cur
    pad_spec = [(0, 0)] * arr.ndim
    pad_spec[axis] = (0, pad)
    return jnp.pad(arr, pad_spec)


def _pad_kv_state_bh(state_bhkv: jnp.ndarray, K_pad: int, V_pad: int) -> jnp.ndarray:
    """Pad state [B,H,K,V] to [B,H,K_pad,V_pad] without flattening axes."""
    state = _pad_axis_right(state_bhkv, axis=-2, new_size=K_pad)
    state = _pad_axis_right(state, axis=-1, new_size=V_pad)
    return state


def _preserve_sharding_like(x: jnp.ndarray, like: jnp.ndarray) -> jnp.ndarray:
    """Best-effort: keep the sharding of `like` after pad/mask ops (avoids surprise reshard)."""
    try:
        return jax.lax.with_sharding_constraint(x, like.sharding)
    except Exception:
        return x


# --- TPU recurrent: BlockSpecs and kernel ---


def _in_specs_head_first_tpu_bh_out(B, H, T, K_pad, V_pad, is_beta_headwise):
    """Same inputs, but output `out` is [B,H,T,V_pad] (BH-major) to avoid NH reshapes."""

    def _bh(nh):
        b = nh // H
        h = nh - b * H
        return (b, h)

    in_specs = (
        pl.BlockSpec((1, 1, T, K_pad), lambda nh: (*_bh(nh), 0, 0)),  # q
        pl.BlockSpec((1, 1, T, K_pad), lambda nh: (*_bh(nh), 0, 0)),  # k
        pl.BlockSpec((1, 1, T, V_pad), lambda nh: (*_bh(nh), 0, 0)),  # v
        pl.BlockSpec((1, 1, T, 1), lambda nh: (*_bh(nh), 0, 0)),  # g as [B,H,T,1]
        (
            pl.BlockSpec((1, 1, T, 1), lambda nh: (*_bh(nh), 0, 0))  # β headwise [B,H,T,1]
            if is_beta_headwise
            else pl.BlockSpec((1, 1, T, V_pad), lambda nh: (*_bh(nh), 0, 0))  # β per-V
        ),
        pl.BlockSpec((1, 1, K_pad, V_pad), lambda nh: (*_bh(nh), 0, 0)),  # init
    )

    out_specs = (
        pl.BlockSpec((1, 1, T, V_pad), lambda nh: (*_bh(nh), 0, 0)),  # out [B,H,T,V_pad]
        pl.BlockSpec((1, 1, K_pad, V_pad), lambda nh: (*_bh(nh), 0, 0)),  # final [B,H,K,V]
    )
    return in_specs, out_specs


def _gdn_recurrent_fwd_kernel_tpu(
    q_ref,
    k_ref,
    v_ref,
    g_ref,
    beta_ref,
    init_ref,
    out_ref,
    final_ref,
    *,
    T,
    K_pad,
    V_pad,
    use_qk_l2norm,
    has_initial_state,
    is_beta_headwise,
    scale,
):
    """TPU kernel for GDN recurrence (decode). One (B,H) tile per program, full window in last two dims.

    Core step (rectangular state S ∈ ℝ^{K×V}):
      α = exp(g_t)                     # compute decay
      Sα = α · S                       # apply decay to the state
      kv = k_t^T Sα                    # retrieve old value
      yα = q_t^T Sα                    # output part 1 (readout from decayed state)
      δ  = (v_t - kv) * β_t            # compute delta (learning rate β_t)
      S  ← Sα + k_t @ δ                # update the state
      o_t = yα + (k_t · q_t) · δ       # output part 1 + part 2 (new memory)

      note that o_t = S^T @ q_t = (Sa + k_t @ δ)^T @ q_t = part 1 + part 2
    """
    # ---- Full-window local views (last-two dims are full → TPU-friendly) ----
    q_view = q_ref[dslice(0, 1), dslice(0, 1), dslice(0, T), dslice(0, K_pad)][0, 0]  # (T, K_pad)
    k_view = k_ref[dslice(0, 1), dslice(0, 1), dslice(0, T), dslice(0, K_pad)][0, 0]  # (T, K_pad)
    v_view = v_ref[dslice(0, 1), dslice(0, 1), dslice(0, T), dslice(0, V_pad)][0, 0]  # (T, V_pad)
    g_view = g_ref[dslice(0, 1), dslice(0, 1), dslice(0, T), dslice(0, 1)][0, 0, :, 0]  # (T,)

    if is_beta_headwise:
        beta_h = beta_ref[dslice(0, 1), dslice(0, 1), dslice(0, T), dslice(0, 1)][0, 0, :, 0]  # (T,)
    else:
        beta_h = beta_ref[dslice(0, 1), dslice(0, 1), dslice(0, T), dslice(0, V_pad)][0, 0]  # (T, V_pad)

    # ---- Initial state S ∈ ℝ^{K×V} (kept in fp32 for stability) ----
    S = (
        init_ref[dslice(0, 1), dslice(0, 1), dslice(0, K_pad), dslice(0, V_pad)][0, 0].astype(jnp.float32)
        if has_initial_state
        else jnp.zeros((K_pad, V_pad), dtype=jnp.float32)
    )

    # Output buffer (write once at the end). Keep in output dtype; math in fp32.
    out_tile = jnp.zeros((T, V_pad), dtype=out_ref.dtype)
    scale32 = jnp.asarray(scale, dtype=jnp.float32)

    # Integer iota (TPU needs int/index type for iota)
    tvec_i32 = jnp.arange(T, dtype=jnp.int32)

    # ---- Unrolled Python loop → indices are compile-time constants (no dynamic_slice) ----
    for i in range(T):
        # Static-indexed loads
        q_t = q_view[i].astype(jnp.float32)  # (K,)
        k_t = k_view[i].astype(jnp.float32)  # (K,)
        v_t = v_view[i].astype(jnp.float32)  # (V,)
        g_t = g_view[i].astype(jnp.float32)  # scalar

        # α and optional L2-normalization + scaling
        alpha = jnp.exp(g_t)
        if use_qk_l2norm:
            inv_q = jax.lax.rsqrt(jnp.sum(q_t * q_t) + 1e-6)
            inv_k = jax.lax.rsqrt(jnp.sum(k_t * k_t) + 1e-6)
            q_t = q_t * inv_q
            k_t = k_t * inv_k
        q_t = q_t * scale32

        # Compute Sα once and reuse it (saves multiplies and memory traffic)
        S_alpha = S * alpha

        # (1×K) @ (K×V) → (1×V) → (V,)
        kv = jnp.squeeze(jnp.matmul(k_t[None, :], S_alpha), axis=0)
        y_al = jnp.squeeze(jnp.matmul(q_t[None, :], S_alpha), axis=0)

        # Inner product for the residual term
        kq = jnp.sum(k_t * q_t)

        # δ = (v - kv) ⊙ β
        beta_t = (beta_h[i] if is_beta_headwise else beta_h[i, :]).astype(jnp.float32)
        delta = (v_t - kv) * beta_t  # (V,)

        # State update and output
        S = S_alpha + jnp.outer(k_t, delta)  # (K,V)
        y_t = (y_al + delta * kq).astype(out_ref.dtype)

        # Static write to out_tile using one‑hot row (no dynamic_update_slice)
        i_i32 = jnp.int32(i)
        row = (tvec_i32 == i_i32).astype(out_ref.dtype)[:, None]  # (T,1)
        out_tile = out_tile + row * y_t[None, :]

    # ---- Single contiguous stores ----
    out_ref[dslice(0, 1), dslice(0, 1), dslice(0, T), dslice(0, V_pad)] = out_tile[None, None, :, :]

    final_ref[dslice(0, 1), dslice(0, 1), dslice(0, K_pad), dslice(0, V_pad)] = S[None, None, :, :].astype(
        final_ref.dtype
    )


def _recurrent_gated_delta_rule_flash(
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    g: NamedArray,
    beta: NamedArray,
    *,
    initial_state: Optional[jnp.ndarray] = None,  # [B,H,dk,dv]
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    head_first: bool = False,  # True: inputs are [B,H,T,*]; False: [B,T,H,*]
    lengths: Optional[jnp.ndarray] = None,  # [B*H] or [B,H]
) -> Tuple[NamedArray, Optional[jnp.ndarray]]:
    Batch = query.resolve_axis("batch")
    Pos = query.resolve_axis("position")
    Heads = query.resolve_axis("heads")
    Dk = query.resolve_axis("k_head_dim")
    Dv = value.resolve_axis("v_head_dim")

    if Pos.size != 1:
        print(
            f"WARNING: You are calling the recurrent kernel with a position axis of size {Pos.size}. Use the chunkwise kernel for training / prefill."
        )

    B_, T_, H_, K_ = Batch.size, Pos.size, Heads.size, Dk.size
    V_ = Dv.size
    NH = B_ * H_

    # Ensure arrays match the declared layout (no-op if already matching)
    def _ensure_layout(x_named: NamedArray, layout: tuple[str, ...]) -> jnp.ndarray:
        have = tuple(ax.name for ax in x_named.axes)
        if have == layout:
            return x_named.array
        return hax.rearrange(x_named, layout).array

    is_tpu = jax.devices()[0].platform == "tpu"
    use_head_first = True if is_tpu else head_first

    if use_head_first:
        q_arr = _ensure_layout(query.astype(jnp.float32), (Batch.name, Heads.name, Pos.name, Dk.name))  # [B,H,T,K]
        k_arr = _ensure_layout(key.astype(jnp.float32), (Batch.name, Heads.name, Pos.name, Dk.name))
        v_arr = _ensure_layout(value.astype(jnp.float32), (Batch.name, Heads.name, Pos.name, Dv.name))
        g_arr = _ensure_layout(g.astype(jnp.float32), (Batch.name, Heads.name, Pos.name))
        beta_axis_names = tuple(ax.name for ax in beta.axes)
        is_beta_headwise = Dv.name not in beta_axis_names
        if is_beta_headwise:
            beta_arr = _ensure_layout(beta.astype(jnp.float32), (Batch.name, Heads.name, Pos.name))  # [B,H,T]
        else:
            beta_arr = _ensure_layout(
                beta.astype(jnp.float32), (Batch.name, Heads.name, Pos.name, Dv.name)
            )  # [B,H,T,V]
    else:
        q_arr = _ensure_layout(query.astype(jnp.float32), (Batch.name, Pos.name, Heads.name, Dk.name))  # [B,T,H,K]
        k_arr = _ensure_layout(key.astype(jnp.float32), (Batch.name, Pos.name, Heads.name, Dk.name))
        v_arr = _ensure_layout(value.astype(jnp.float32), (Batch.name, Pos.name, Heads.name, Dv.name))
        g_arr = _ensure_layout(g.astype(jnp.float32), (Batch.name, Pos.name, Heads.name))
        beta_axis_names = tuple(ax.name for ax in beta.axes)
        is_beta_headwise = Dv.name not in beta_axis_names
        if is_beta_headwise:
            beta_arr = _ensure_layout(beta.astype(jnp.float32), (Batch.name, Pos.name, Heads.name))  # [B,T,H]
        else:
            beta_arr = _ensure_layout(
                beta.astype(jnp.float32), (Batch.name, Pos.name, Heads.name, Dv.name)
            )  # [B,T,H,V]

    # Initial state: [B,H,K,V]
    if initial_state is None:
        init_arr = jnp.zeros((B_, H_, K_, V_), dtype=jnp.float32)
        has_initial = False
    else:
        init_in = initial_state.astype(jnp.float32)
        if init_in.shape == (B_, H_, K_, V_):
            init_arr = init_in
        elif init_in.ndim == 3 and init_in.shape[0] == NH:
            if is_tpu:
                raise ValueError(
                    "On TPU under pjit/sharding, pass initial_state as [B,H,dk,dv] "
                    "(not [BH,dk,dv]) to avoid reshape-induced reshard/all-to-all."
                )
            init_arr = init_in.reshape(B_, H_, K_, V_)
        else:
            init_arr = hax.rearrange(
                hax.named(init_in, (Batch, Heads, Dk, Dv)),
                (Batch.name, Heads.name, Dk.name, Dv.name),
            ).array
        has_initial = True

    # Varlen
    if lengths is None:
        lengths_flat = jnp.full((NH,), T_, dtype=jnp.int32)
    else:
        lf = lengths
        if lf.ndim == 2 and lf.shape == (B_, H_):
            lf = lf.reshape(NH)
        lengths_flat = lf.astype(jnp.int32)

    # 2D tiling & padding
    BK = _pick_bk_tile_for_decode(Dk.size)
    BV = _pick_bv_tile_for_decode(Dv.size)
    K_pad = int(((K_ + BK - 1) // BK) * BK)
    V_pad = int(((V_ + BV - 1) // BV) * BV)

    # 2D tiling & padding
    BK = _pick_bk_tile_for_decode(Dk.size)
    BV = _pick_bv_tile_for_decode(Dv.size)
    K_pad = int(((K_ + BK - 1) // BK) * BK)
    V_pad = int(((V_ + BV - 1) // BV) * BV)

    # ---- Axis-preserving padding (NO (B,H)->(BH) reshapes) ----
    q_pad = _pad_axis_right(q_arr, axis=-1, new_size=K_pad)
    k_pad = _pad_axis_right(k_arr, axis=-1, new_size=K_pad)
    v_pad = _pad_axis_right(v_arr, axis=-1, new_size=V_pad)

    init_kpad = _pad_kv_state_bh(init_arr, K_pad=K_pad, V_pad=V_pad)

    beta_pad = beta_arr if is_beta_headwise else _pad_axis_right(beta_arr, axis=-1, new_size=V_pad)

    # Optional: keep sharding stable after pad (helps under pjit)
    q_pad = _preserve_sharding_like(q_pad, q_arr)
    k_pad = _preserve_sharding_like(k_pad, k_arr)
    v_pad = _preserve_sharding_like(v_pad, v_arr)
    init_kpad = _preserve_sharding_like(init_kpad, init_arr)
    beta_pad = _preserve_sharding_like(beta_pad, beta_arr)

    # -------------------------
    # Pallas shapes (IMPORTANT)
    # -------------------------
    final_struct = jax.ShapeDtypeStruct((B_, H_, K_pad, V_pad), jnp.float32)

    # TPU: we want BH-major output to avoid NH<->BH reshapes
    out_struct = jax.ShapeDtypeStruct((B_, H_, T_, V_pad), value.dtype)

    # Expand g and headwise β to 4-D with a trailing singleton dim (TPU block rules)
    g_arg = g_arr[..., None]  # [B,H,T,1]
    beta_arg = beta_pad[..., None] if is_beta_headwise else beta_pad

    kernel_tpu = functools.partial(
        _gdn_recurrent_fwd_kernel_tpu,
        T=T_,
        K_pad=K_pad,
        V_pad=V_pad,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        has_initial_state=has_initial,
        is_beta_headwise=is_beta_headwise,
        scale=Dk.size**-0.5,
    )

    # try to parallelize NH across 2-core TPUs.
    compiler_params = None
    try:
        from jax.experimental.pallas import tpu as pltpu

        compiler_params = pltpu.CompilerParams(dimension_semantics=["parallel"])
    except Exception:
        pass

    # >>> Use the BH-out specs here (4-D output) <<<
    in_specs_tpu, out_specs_tpu = _in_specs_head_first_tpu_bh_out(B_, H_, T_, K_pad, V_pad, is_beta_headwise)

    out_pad_bhtv, final_pad = pl.pallas_call(
        kernel_tpu,
        out_shape=(out_struct, final_struct),
        grid=(NH,),
        in_specs=in_specs_tpu,
        out_specs=out_specs_tpu,
        compiler_params=compiler_params,
    )(q_pad, k_pad, v_pad, g_arg, beta_arg, init_kpad)

    # -------------------------
    # Trim + wrap outputs
    # -------------------------
    out_trim = out_pad_bhtv[:, :, :, :V_]  # (B,H,T,V)
    out_named = hax.named(out_trim, (Batch, Heads, Pos, Dv))
    out_final_named = hax.rearrange(out_named, (Batch, Pos, Heads, Dv))

    final_ret = None
    if output_final_state:
        final_ret = final_pad[:, :, :K_, :V_]  # already (B,H,K,V) after slicing

    return out_final_named, final_ret


def recurrent_gated_delta_rule(
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    g: NamedArray,
    beta: NamedArray,
    *,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_flash: bool = True,
) -> Tuple[NamedArray, Optional[jnp.ndarray]]:
    if use_flash:
        try:
            return _recurrent_gated_delta_rule_flash(
                query,
                key,
                value,
                g,
                beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                # optional toggles:
                head_first=False,
                lengths=None,
            )
        except Exception:
            if use_flash:
                raise
    return _recurrent_gated_delta_rule_reference(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


recurrent_gated_delta_rule.__doc__ = _recurrent_gated_delta_rule_reference.__doc__


# -----------------------------------------------------------------------------
# Chunkwise Gated Delta Rule (Flash path) with selectable backward:
# -----------------------------------------------------------------------------

_GDN_EXP_CLIP = 80.0
_GDN_TPU_MULT = 64


def _round_up_to(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _in_specs_chunk_step_tpu(B: int, H: int, Ct: int, K_pad: int, V_pad: int):
    """One program per (b,h)."""

    def _bh(nh):
        b = nh // H
        h = nh - b * H
        return (b, h)

    in_specs = (
        pl.BlockSpec((1, 1, Ct, K_pad), lambda nh: (*_bh(nh), 0, 0)),  # q
        pl.BlockSpec((1, 1, Ct, K_pad), lambda nh: (*_bh(nh), 0, 0)),  # k
        pl.BlockSpec((1, 1, Ct, V_pad), lambda nh: (*_bh(nh), 0, 0)),  # v
        pl.BlockSpec((1, 1, Ct, 1), lambda nh: (*_bh(nh), 0, 0)),  # g
        pl.BlockSpec((1, 1, Ct, 1), lambda nh: (*_bh(nh), 0, 0)),  # beta
        pl.BlockSpec((1, 1, K_pad, V_pad), lambda nh: (*_bh(nh), 0, 0)),  # Sprev
    )

    out_specs = (
        pl.BlockSpec((1, 1, Ct, V_pad), lambda nh: (*_bh(nh), 0, 0)),  # out
        pl.BlockSpec((1, 1, K_pad, V_pad), lambda nh: (*_bh(nh), 0, 0)),  # Snext
    )
    return in_specs, out_specs


def _gdn_chunk_step_fwd_kernel_tpu(
    q_ref,
    k_ref,
    v_ref,
    gcum_ref,
    b_ref,
    Sprev_ref,
    out_ref,
    Snext_ref,
    *,
    Ct: int,
    K_pad: int,
    V_pad: int,
):
    """Forward chunk step (TPU TC) using *precomputed* g_cum.

    NOTE: No arr[i] / arr[-1] indexing; no cumsum.
    """

    q = q_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, K_pad)].astype(jnp.float32).reshape(Ct, K_pad)
    k = k_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, K_pad)].astype(jnp.float32).reshape(Ct, K_pad)
    v = v_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, V_pad)].astype(jnp.float32).reshape(Ct, V_pad)
    g_cum = (
        gcum_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, 1)]
        .astype(jnp.float32)
        .reshape(
            Ct,
        )
    )
    beta = (
        b_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, 1)]
        .astype(jnp.float32)
        .reshape(
            Ct,
        )
    )
    S_prev = (
        Sprev_ref[dslice(0, 1), dslice(0, 1), dslice(0, K_pad), dslice(0, V_pad)]
        .astype(jnp.float32)
        .reshape(K_pad, V_pad)
    )

    idx = jnp.arange(Ct, dtype=jnp.int32)
    ii = idx[:, None]
    jj = idx[None, :]
    tril_strict = (ii > jj).astype(jnp.float32)
    causal_mask = (ii >= jj).astype(jnp.float32)

    v_beta = v * beta[:, None]
    k_beta = k * beta[:, None]

    g_cum_cl = jnp.clip(g_cum, -_GDN_EXP_CLIP, _GDN_EXP_CLIP)
    exp_g = jnp.exp(g_cum_cl)

    diff = g_cum[:, None] - g_cum[None, :]
    diff_cl = jnp.clip(diff, -_GDN_EXP_CLIP, _GDN_EXP_CLIP)
    exp_diff = jnp.exp(diff_cl)

    KKT = jnp.matmul(k_beta, k.T)
    A = -KKT * exp_diff * tril_strict

    def _solve_unit_lower(A_strict, rhs):
        R = rhs.shape[-1]
        X = jnp.zeros((Ct, R), dtype=jnp.float32)
        for i in range(Ct):
            mask_i = (idx == jnp.int32(i)).astype(jnp.float32)
            rhs_i = jnp.sum(rhs * mask_i[:, None], axis=0)
            A_i = jnp.sum(A_strict * mask_i[:, None], axis=0)
            contrib = jnp.sum(A_i[:, None] * X, axis=0)
            x_i = rhs_i + contrib
            X = X + mask_i[:, None] * x_i[None, :]
        return X

    v_pseudo = _solve_unit_lower(A, v_beta)
    k_scaled = k_beta * exp_g[:, None]
    k_cumdecay = _solve_unit_lower(A, k_scaled)

    QK = jnp.matmul(q, k.T)
    attn = QK * exp_diff * causal_mask

    v_prime = jnp.matmul(k_cumdecay, S_prev)
    v_new = v_pseudo - v_prime

    inter = jnp.matmul(q * exp_g[:, None], S_prev)
    intra = jnp.matmul(attn, v_new)
    out = inter + intra

    # g_tail = g_cum[-1] without indexing
    mask_last = (idx == jnp.int32(Ct - 1)).astype(jnp.float32)
    g_tail = jnp.sum(g_cum * mask_last)

    decay_tail = jnp.exp(jnp.clip(g_tail, -_GDN_EXP_CLIP, _GDN_EXP_CLIP))

    diff_tail = g_tail - g_cum
    diff_tail_cl = jnp.clip(diff_tail, -_GDN_EXP_CLIP, _GDN_EXP_CLIP)
    decay_w = jnp.exp(diff_tail_cl)

    k_w = k * decay_w[:, None]
    add = jnp.matmul(k_w.T, v_new)
    S_next = S_prev * decay_tail + add

    out_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, V_pad)] = out[None, None, :, :].astype(out_ref.dtype)
    Snext_ref[dslice(0, 1), dslice(0, 1), dslice(0, K_pad), dslice(0, V_pad)] = S_next[None, None, :, :].astype(
        Snext_ref.dtype
    )


def _gdn_chunk_step_fwd_pallas(
    q_bhck,
    k_bhck,
    v_bhcv,
    g_cum_bhc,
    b_bhc,
    S_prev_bhkv,
    *,
    Ct: int,
    K_pad: int,
    V_pad: int,
):
    B, H = q_bhck.shape[0], q_bhck.shape[1]
    NH = B * H

    gcum4 = g_cum_bhc[..., None]
    b4 = b_bhc[..., None]

    out_struct = jax.ShapeDtypeStruct((B, H, Ct, V_pad), jnp.float32)
    st_struct = jax.ShapeDtypeStruct((B, H, K_pad, V_pad), jnp.float32)

    kernel = functools.partial(_gdn_chunk_step_fwd_kernel_tpu, Ct=int(Ct), K_pad=int(K_pad), V_pad=int(V_pad))

    compiler_params = None
    try:
        from jax.experimental.pallas import tpu as pltpu

        compiler_params = pltpu.CompilerParams(dimension_semantics=["parallel"])
    except Exception:
        pass

    in_specs, out_specs = _in_specs_chunk_step_tpu(B, H, Ct, K_pad, V_pad)

    out, S_next = pl.pallas_call(
        kernel,
        out_shape=(out_struct, st_struct),
        grid=(NH,),
        in_specs=in_specs,
        out_specs=out_specs,
        compiler_params=compiler_params,
    )(q_bhck, k_bhck, v_bhcv, gcum4, b4, S_prev_bhkv)

    return out, S_next


# -----------------------------------------------------------------------------
# Chunkwise flash: TPU pallas forward + analytic backward (reverse scan)
# -----------------------------------------------------------------------------


def _chunk_gated_delta_rule_flash_pallas_impl(
    q_arr: jnp.ndarray,  # (B,H,L,dk)
    k_arr: jnp.ndarray,  # (B,H,L,dk)
    v_arr: jnp.ndarray,  # (B,H,L,dv)
    g_arr: jnp.ndarray,  # (B,H,L)
    b_arr: jnp.ndarray,  # (B,H,L)
    *,
    chunk_size: int,
    segment_size: int,
    initial_state: Optional[jnp.ndarray],
    use_triangular_solve: bool,
):
    B, H, L, dk = q_arr.shape
    dv = v_arr.shape[-1]
    C = int(chunk_size)

    # token pad to multiple of C
    pad_tok = (C - (L % C)) % C
    if pad_tok:
        q_arr = jnp.pad(q_arr, ((0, 0), (0, 0), (0, pad_tok), (0, 0)))
        k_arr = jnp.pad(k_arr, ((0, 0), (0, 0), (0, pad_tok), (0, 0)))
        v_arr = jnp.pad(v_arr, ((0, 0), (0, 0), (0, pad_tok), (0, 0)))
        g_arr = jnp.pad(g_arr, ((0, 0), (0, 0), (0, pad_tok)))
        b_arr = jnp.pad(b_arr, ((0, 0), (0, 0), (0, pad_tok)))

    L1 = L + pad_tok
    Nc = L1 // C

    # effective segment size (avoid padding to huge lengths when Nc small)
    seg = int(segment_size)
    seg = 1 if seg < 1 else seg
    seg = min(seg, Nc) if Nc > 0 else 1

    n_chunks_pad = _round_up_to(Nc, seg)
    pad_chunks = n_chunks_pad - Nc

    # reshape into chunks (B,H,Nc,C,*)
    q_c = q_arr.reshape(B, H, Nc, C, dk)
    k_c = k_arr.reshape(B, H, Nc, C, dk)
    v_c = v_arr.reshape(B, H, Nc, C, dv)
    g_c = g_arr.reshape(B, H, Nc, C)
    b_c = b_arr.reshape(B, H, Nc, C)

    # pad chunk axis with extra all-zero chunks
    if pad_chunks:
        q_c = jnp.pad(q_c, ((0, 0), (0, 0), (0, pad_chunks), (0, 0), (0, 0)))
        k_c = jnp.pad(k_c, ((0, 0), (0, 0), (0, pad_chunks), (0, 0), (0, 0)))
        v_c = jnp.pad(v_c, ((0, 0), (0, 0), (0, pad_chunks), (0, 0), (0, 0)))
        g_c = jnp.pad(g_c, ((0, 0), (0, 0), (0, pad_chunks), (0, 0)))
        b_c = jnp.pad(b_c, ((0, 0), (0, 0), (0, pad_chunks), (0, 0)))

    # pad feature dims for TPU
    K_pad = _round_up_to(dk, _GDN_TPU_MULT)
    V_pad = _round_up_to(dv, _GDN_TPU_MULT)

    q_c = _pad_axis_right(q_c, axis=-1, new_size=K_pad)
    k_c = _pad_axis_right(k_c, axis=-1, new_size=K_pad)
    v_c = _pad_axis_right(v_c, axis=-1, new_size=V_pad)

    # pad chunk-tile length Ct for TPU blocked indexing (Ct multiple of 8)
    Ct = _round_up_to(C, _GDN_TPU_MULT)
    if Ct != C:
        q_c = jnp.pad(q_c, ((0, 0), (0, 0), (0, 0), (0, Ct - C), (0, 0)))
        k_c = jnp.pad(k_c, ((0, 0), (0, 0), (0, 0), (0, Ct - C), (0, 0)))
        v_c = jnp.pad(v_c, ((0, 0), (0, 0), (0, 0), (0, Ct - C), (0, 0)))
        g_c = jnp.pad(g_c, ((0, 0), (0, 0), (0, 0), (0, Ct - C)))
        b_c = jnp.pad(b_c, ((0, 0), (0, 0), (0, 0), (0, Ct - C)))

    g_cum = jnp.cumsum(g_c, axis=-1)

    # init state
    if initial_state is None:
        S0 = jnp.zeros((B, H, K_pad, V_pad), dtype=jnp.float32)
    else:
        S0 = _pad_kv_state_bh(initial_state.astype(jnp.float32), K_pad=K_pad, V_pad=V_pad)

    # segment view: (B,H,n_segments,seg,Ct,*)
    n_segments = n_chunks_pad // seg
    q_s = q_c.reshape(B, H, n_segments, seg, Ct, K_pad)
    k_s = k_c.reshape(B, H, n_segments, seg, Ct, K_pad)
    v_s = v_c.reshape(B, H, n_segments, seg, Ct, V_pad)
    g_s = g_cum.reshape(B, H, n_segments, seg, Ct)
    b_s = b_c.reshape(B, H, n_segments, seg, Ct)

    # time-major over segments
    q_tm = jnp.moveaxis(q_s, 2, 0)
    k_tm = jnp.moveaxis(k_s, 2, 0)
    v_tm = jnp.moveaxis(v_s, 2, 0)
    g_tm = jnp.moveaxis(g_s, 2, 0)
    b_tm = jnp.moveaxis(b_s, 2, 0)

    def seg_body(S_carry, seg_inp):
        q_seg, k_seg, v_seg, g_seg, b_seg = seg_inp  # (B,H,seg,Ct,*)
        # time-major over chunks in segment
        q_ctm = jnp.moveaxis(q_seg, 2, 0)  # (seg,B,H,Ct,K)
        k_ctm = jnp.moveaxis(k_seg, 2, 0)
        v_ctm = jnp.moveaxis(v_seg, 2, 0)
        g_ctm = jnp.moveaxis(g_seg, 2, 0)  # (seg,B,H,Ct)
        b_ctm = jnp.moveaxis(b_seg, 2, 0)

        def chunk_body(S_prev, chunk_inp):
            q_i, k_i, v_i, g_i, b_i = chunk_inp  # (B,H,Ct,*)
            out_i, S_next = _gdn_chunk_step_fwd_pallas(
                q_i,
                k_i,
                v_i,
                g_i,
                b_i,
                S_prev,
                Ct=Ct,
                K_pad=K_pad,
                V_pad=V_pad,
            )
            # keep only first C real positions (drop Ct-C padded tail)
            return S_next, out_i[:, :, :C, :]

        S_end, out_chunks = lax.scan(chunk_body, S_carry, (q_ctm, k_ctm, v_ctm, g_ctm, b_ctm), length=seg)
        out_seg = jnp.moveaxis(out_chunks, 0, 2)  # (B,H,seg,C,V)
        return S_end, (out_seg, S_carry)  # also return seg start state

    S_final, (out_segs, seg_starts) = lax.scan(seg_body, S0, (q_tm, k_tm, v_tm, g_tm, b_tm), length=n_segments)

    # out_segs: (n_segments,B,H,seg,C,V) -> (B,H,n_chunks_pad,C,V)
    out_bhnscv = jnp.transpose(out_segs, (1, 2, 0, 3, 4, 5)).reshape(B, H, n_chunks_pad, C, V_pad)

    # drop padded chunks, flatten tokens, crop, crop dv
    out_bhncv = out_bhnscv[:, :, :Nc, :, :]  # (B,H,Nc,C,V)
    out_bhlv = out_bhncv.reshape(B, H, L1, V_pad)  # (B,H,L1,V)
    out = out_bhlv[:, :, :L, :dv]  # (B,H,L,dv)

    S_trim = S_final[:, :, :dk, :dv]  # (B,H,dk,dv)

    return out, S_trim, seg_starts  # seg_starts used for backward


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 8))
def _chunk_gated_delta_rule_flash_pallas(
    q_arr: jnp.ndarray,
    k_arr: jnp.ndarray,
    v_arr: jnp.ndarray,
    g_arr: jnp.ndarray,
    b_arr: jnp.ndarray,
    chunk_size: int,
    segment_size: int,
    initial_state: Optional[jnp.ndarray],
    use_triangular_solve: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    out, S, _ = _chunk_gated_delta_rule_flash_pallas_impl(
        q_arr,
        k_arr,
        v_arr,
        g_arr,
        b_arr,
        chunk_size=chunk_size,
        segment_size=segment_size,
        initial_state=initial_state,
        use_triangular_solve=use_triangular_solve,
    )
    return out, S


def _chunk_gated_delta_rule_flash_pallas_fwd(
    q_arr, k_arr, v_arr, g_arr, b_arr, chunk_size, segment_size, initial_state, use_triangular_solve
):
    out, S, seg_starts = _chunk_gated_delta_rule_flash_pallas_impl(
        q_arr,
        k_arr,
        v_arr,
        g_arr,
        b_arr,
        chunk_size=chunk_size,
        segment_size=segment_size,
        initial_state=initial_state,
        use_triangular_solve=use_triangular_solve,
    )
    # Save only what bwd needs
    residuals = (q_arr, k_arr, v_arr, g_arr, b_arr, seg_starts, initial_state)
    return (out, S), residuals


def _in_specs_chunk_state_tpu(B: int, H: int, Ct: int, K_pad: int, V_pad: int):
    def _bh(nh):
        b = nh // H
        h = nh - b * H
        return (b, h)

    in_specs = (
        pl.BlockSpec((1, 1, Ct, K_pad), lambda nh: (*_bh(nh), 0, 0)),  # k
        pl.BlockSpec((1, 1, Ct, V_pad), lambda nh: (*_bh(nh), 0, 0)),  # v
        pl.BlockSpec((1, 1, Ct, 1), lambda nh: (*_bh(nh), 0, 0)),  # g_cum
        pl.BlockSpec((1, 1, Ct, 1), lambda nh: (*_bh(nh), 0, 0)),  # beta
        pl.BlockSpec((1, 1, K_pad, V_pad), lambda nh: (*_bh(nh), 0, 0)),  # S_prev
    )
    out_specs = pl.BlockSpec((1, 1, K_pad, V_pad), lambda nh: (*_bh(nh), 0, 0))  # S_next
    return in_specs, out_specs


def _gdn_chunk_state_fwd_kernel_tpu(
    k_ref,
    v_ref,
    gcum_ref,
    b_ref,
    Sprev_ref,
    Snext_ref,
    *,
    Ct: int,
    K_pad: int,
    V_pad: int,
):
    k = k_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, K_pad)].astype(jnp.float32).reshape(Ct, K_pad)
    v = v_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, V_pad)].astype(jnp.float32).reshape(Ct, V_pad)
    g_cum = (
        gcum_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, 1)]
        .astype(jnp.float32)
        .reshape(
            Ct,
        )
    )
    beta = (
        b_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, 1)]
        .astype(jnp.float32)
        .reshape(
            Ct,
        )
    )
    S_prev = (
        Sprev_ref[dslice(0, 1), dslice(0, 1), dslice(0, K_pad), dslice(0, V_pad)]
        .astype(jnp.float32)
        .reshape(K_pad, V_pad)
    )

    idx = jnp.arange(Ct, dtype=jnp.int32)
    ii = idx[:, None]
    jj = idx[None, :]
    tril_strict = (ii > jj).astype(jnp.float32)

    v_beta = v * beta[:, None]
    k_beta = k * beta[:, None]

    g_cum_cl = jnp.clip(g_cum, -_GDN_EXP_CLIP, _GDN_EXP_CLIP)
    exp_g = jnp.exp(g_cum_cl)

    diff = g_cum[:, None] - g_cum[None, :]
    diff_cl = jnp.clip(diff, -_GDN_EXP_CLIP, _GDN_EXP_CLIP)
    exp_diff = jnp.exp(diff_cl)

    KKT = jnp.matmul(k_beta, k.T)
    A = -KKT * exp_diff * tril_strict

    def _solve_unit_lower(A_strict, rhs):
        R = rhs.shape[-1]
        X = jnp.zeros((Ct, R), dtype=jnp.float32)
        for i in range(Ct):
            mask_i = (idx == jnp.int32(i)).astype(jnp.float32)
            rhs_i = jnp.sum(rhs * mask_i[:, None], axis=0)
            A_i = jnp.sum(A_strict * mask_i[:, None], axis=0)
            contrib = jnp.sum(A_i[:, None] * X, axis=0)
            x_i = rhs_i + contrib
            X = X + mask_i[:, None] * x_i[None, :]
        return X

    v_pseudo = _solve_unit_lower(A, v_beta)
    k_scaled = k_beta * exp_g[:, None]
    k_cumdecay = _solve_unit_lower(A, k_scaled)

    v_prime = jnp.matmul(k_cumdecay, S_prev)
    v_new = v_pseudo - v_prime

    mask_last = (idx == jnp.int32(Ct - 1)).astype(jnp.float32)
    g_tail = jnp.sum(g_cum * mask_last)
    decay_tail = jnp.exp(jnp.clip(g_tail, -_GDN_EXP_CLIP, _GDN_EXP_CLIP))

    diff_tail = g_tail - g_cum
    diff_tail_cl = jnp.clip(diff_tail, -_GDN_EXP_CLIP, _GDN_EXP_CLIP)
    decay_w = jnp.exp(diff_tail_cl)

    add = jnp.matmul((k * decay_w[:, None]).T, v_new)
    S_next = S_prev * decay_tail + add

    Snext_ref[dslice(0, 1), dslice(0, 1), dslice(0, K_pad), dslice(0, V_pad)] = S_next[None, None, :, :].astype(
        Snext_ref.dtype
    )


def _gdn_chunk_state_update_pallas(
    k_bhck: jnp.ndarray,
    v_bhcv: jnp.ndarray,
    g_cum_bhc: jnp.ndarray,
    b_bhc: jnp.ndarray,
    S_prev_bhkv: jnp.ndarray,
    *,
    Ct: int,
    K_pad: int,
    V_pad: int,
) -> jnp.ndarray:
    B, H = k_bhck.shape[0], k_bhck.shape[1]
    NH = B * H

    gcum4 = g_cum_bhc[..., None]
    b4 = b_bhc[..., None]

    st_struct = jax.ShapeDtypeStruct((B, H, K_pad, V_pad), jnp.float32)

    kernel = functools.partial(_gdn_chunk_state_fwd_kernel_tpu, Ct=int(Ct), K_pad=int(K_pad), V_pad=int(V_pad))

    compiler_params = None
    try:
        from jax.experimental.pallas import tpu as pltpu

        compiler_params = pltpu.CompilerParams(dimension_semantics=["parallel"])
    except Exception:
        pass

    in_specs, out_specs = _in_specs_chunk_state_tpu(B, H, Ct, K_pad, V_pad)

    (S_next,) = pl.pallas_call(
        kernel,
        out_shape=(st_struct,),
        grid=(NH,),
        in_specs=in_specs,
        out_specs=(out_specs,),
        compiler_params=compiler_params,
    )(k_bhck, v_bhcv, gcum4, b4, S_prev_bhkv)

    return S_next


def _in_specs_chunk_bwd_tpu(B: int, H: int, Ct: int, K_pad: int, V_pad: int):
    def _bh(nh):
        b = nh // H
        h = nh - b * H
        return (b, h)

    in_specs = (
        pl.BlockSpec((1, 1, Ct, K_pad), lambda nh: (*_bh(nh), 0, 0)),  # q
        pl.BlockSpec((1, 1, Ct, K_pad), lambda nh: (*_bh(nh), 0, 0)),  # k
        pl.BlockSpec((1, 1, Ct, V_pad), lambda nh: (*_bh(nh), 0, 0)),  # v
        pl.BlockSpec((1, 1, Ct, 1), lambda nh: (*_bh(nh), 0, 0)),  # g_cum
        pl.BlockSpec((1, 1, Ct, 1), lambda nh: (*_bh(nh), 0, 0)),  # beta
        pl.BlockSpec((1, 1, K_pad, V_pad), lambda nh: (*_bh(nh), 0, 0)),  # S_prev
        pl.BlockSpec((1, 1, Ct, V_pad), lambda nh: (*_bh(nh), 0, 0)),  # d_out
        pl.BlockSpec((1, 1, K_pad, V_pad), lambda nh: (*_bh(nh), 0, 0)),  # dS_next
    )

    out_specs = (
        pl.BlockSpec((1, 1, Ct, K_pad), lambda nh: (*_bh(nh), 0, 0)),  # dq
        pl.BlockSpec((1, 1, Ct, K_pad), lambda nh: (*_bh(nh), 0, 0)),  # dk
        pl.BlockSpec((1, 1, Ct, V_pad), lambda nh: (*_bh(nh), 0, 0)),  # dv
        pl.BlockSpec((1, 1, Ct, 1), lambda nh: (*_bh(nh), 0, 0)),  # dg
        pl.BlockSpec((1, 1, Ct, 1), lambda nh: (*_bh(nh), 0, 0)),  # db
        pl.BlockSpec((1, 1, K_pad, V_pad), lambda nh: (*_bh(nh), 0, 0)),  # dS_prev
    )
    return in_specs, out_specs


def _gdn_chunk_step_bwd_kernel_tpu(
    q_ref,
    k_ref,
    v_ref,
    gcum_ref,
    b_ref,
    Sprev_ref,
    dOut_ref,
    dSnext_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    dg_ref,
    db_ref,
    dSprev_ref,
    *,
    Ct: int,
    K_pad: int,
    V_pad: int,
):
    # ---- loads ----
    q = q_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, K_pad)].astype(jnp.float32).reshape(Ct, K_pad)
    k = k_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, K_pad)].astype(jnp.float32).reshape(Ct, K_pad)
    v = v_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, V_pad)].astype(jnp.float32).reshape(Ct, V_pad)
    g_cum = (
        gcum_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, 1)]
        .astype(jnp.float32)
        .reshape(
            Ct,
        )
    )
    beta = (
        b_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, 1)]
        .astype(jnp.float32)
        .reshape(
            Ct,
        )
    )
    S_prev = (
        Sprev_ref[dslice(0, 1), dslice(0, 1), dslice(0, K_pad), dslice(0, V_pad)]
        .astype(jnp.float32)
        .reshape(K_pad, V_pad)
    )
    d_out = (
        dOut_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, V_pad)].astype(jnp.float32).reshape(Ct, V_pad)
    )
    dS_next = (
        dSnext_ref[dslice(0, 1), dslice(0, 1), dslice(0, K_pad), dslice(0, V_pad)]
        .astype(jnp.float32)
        .reshape(K_pad, V_pad)
    )

    idx = jnp.arange(Ct, dtype=jnp.int32)
    ii = idx[:, None]
    jj = idx[None, :]
    tril_strict = (ii > jj).astype(jnp.float32)
    causal_mask = (ii >= jj).astype(jnp.float32)

    # clip masks
    gcum_inrange = ((g_cum >= -_GDN_EXP_CLIP) & (g_cum <= _GDN_EXP_CLIP)).astype(jnp.float32)

    # exp(g_cum)
    g_cum_cl = jnp.clip(g_cum, -_GDN_EXP_CLIP, _GDN_EXP_CLIP)
    exp_g = jnp.exp(g_cum_cl)

    # diff + exp(diff)
    diff = g_cum[:, None] - g_cum[None, :]
    diff_inrange = ((diff >= -_GDN_EXP_CLIP) & (diff <= _GDN_EXP_CLIP)).astype(jnp.float32)
    diff_cl = jnp.clip(diff, -_GDN_EXP_CLIP, _GDN_EXP_CLIP)
    exp_diff = jnp.exp(diff_cl)

    # ---- forward recompute ----
    v_beta = v * beta[:, None]
    k_beta = k * beta[:, None]

    QK = jnp.matmul(q, k.T)
    attn = QK * exp_diff * causal_mask

    KKT = jnp.matmul(k_beta, k.T)
    A = -KKT * exp_diff * tril_strict

    def _solve_unit_lower(A_strict, rhs):
        R = rhs.shape[-1]
        X = jnp.zeros((Ct, R), dtype=jnp.float32)
        for i in range(Ct):
            mask_i = (idx == jnp.int32(i)).astype(jnp.float32)
            rhs_i = jnp.sum(rhs * mask_i[:, None], axis=0)
            A_i = jnp.sum(A_strict * mask_i[:, None], axis=0)
            contrib = jnp.sum(A_i[:, None] * X, axis=0)
            x_i = rhs_i + contrib
            X = X + mask_i[:, None] * x_i[None, :]
        return X

    def _solve_unit_upper_from_A(A_strict_lower, rhs):
        # solves (I - A)^T x = rhs  (unit upper)
        R = rhs.shape[-1]
        X = jnp.zeros((Ct, R), dtype=jnp.float32)
        for i in range(Ct - 1, -1, -1):
            mask_i = (idx == jnp.int32(i)).astype(jnp.float32)
            rhs_i = jnp.sum(rhs * mask_i[:, None], axis=0)
            # column i of A: A[:,i]
            mask_col = (idx == jnp.int32(i)).astype(jnp.float32)
            A_col_i = jnp.sum(A_strict_lower * mask_col[None, :], axis=1)  # (Ct,)
            contrib = jnp.sum(A_col_i[:, None] * X, axis=0)
            x_i = rhs_i + contrib
            X = X + mask_i[:, None] * x_i[None, :]
        return X

    v_pseudo = _solve_unit_lower(A, v_beta)
    k_scaled = k_beta * exp_g[:, None]
    k_cumdecay = _solve_unit_lower(A, k_scaled)

    v_prime = jnp.matmul(k_cumdecay, S_prev)
    v_new = v_pseudo - v_prime

    q_scaled = q * exp_g[:, None]
    inter = jnp.matmul(q_scaled, S_prev)
    intra = jnp.matmul(attn, v_new)
    # out = inter + intra

    mask_last = (idx == jnp.int32(Ct - 1)).astype(jnp.float32)
    g_tail = jnp.sum(g_cum * mask_last)
    gt_inrange = ((g_tail >= -_GDN_EXP_CLIP) & (g_tail <= _GDN_EXP_CLIP)).astype(jnp.float32)
    decay_tail = jnp.exp(jnp.clip(g_tail, -_GDN_EXP_CLIP, _GDN_EXP_CLIP))

    raw_decay_diff = g_tail - g_cum
    decay_diff_inrange = ((raw_decay_diff >= -_GDN_EXP_CLIP) & (raw_decay_diff <= _GDN_EXP_CLIP)).astype(jnp.float32)
    decay_diff_cl = jnp.clip(raw_decay_diff, -_GDN_EXP_CLIP, _GDN_EXP_CLIP)
    decay_w = jnp.exp(decay_diff_cl)
    k_w = k * decay_w[:, None]

    add = jnp.matmul(k_w.T, v_new)
    # S_next = S_prev * decay_tail + add

    # ---- backward ----
    d_inter = d_out
    d_intra = d_out

    # intra = attn @ v_new
    d_attn = jnp.matmul(d_intra, v_new.T)
    d_v_new = jnp.matmul(attn.T, d_intra)  # IMPORTANT: transpose (fixed)

    # inter = q_scaled @ S_prev
    d_q_scaled = jnp.matmul(d_inter, S_prev.T)
    dS_prev = jnp.matmul(q_scaled.T, d_inter)

    # S_next = S_prev*decay_tail + add
    dS_prev = dS_prev + dS_next * decay_tail
    d_decay_tail = jnp.sum(dS_next * S_prev)

    # add = k_w^T @ v_new
    d_add = dS_next
    d_k_w = jnp.matmul(v_new, d_add.T)
    d_v_new = d_v_new + jnp.matmul(k_w, d_add)

    d_k = d_k_w * decay_w[:, None]
    d_decay_w = jnp.sum(d_k_w * k, axis=-1)

    # decay_w = exp(clip(g_tail - g_cum))
    d_decay_diff = d_decay_w * decay_w
    d_decay_diff = d_decay_diff * decay_diff_inrange
    d_g_tail = jnp.sum(d_decay_diff)
    d_g_cum = -d_decay_diff

    # decay_tail = exp(clip(g_tail))
    d_g_tail = d_g_tail + d_decay_tail * decay_tail * gt_inrange
    d_g_cum = d_g_cum + mask_last * d_g_tail

    # v_new = v_pseudo - v_prime
    d_v_pseudo = d_v_new
    d_v_prime = -d_v_new

    # v_prime = k_cumdecay @ S_prev
    d_k_cumdecay = jnp.matmul(d_v_prime, S_prev.T)
    dS_prev = dS_prev + jnp.matmul(k_cumdecay.T, d_v_prime)

    # attn = QK * exp_diff * causal_mask
    d_QK = d_attn * (exp_diff * causal_mask)
    d_exp_diff = d_attn * QK * causal_mask

    # QK = q @ k^T
    d_q = jnp.matmul(d_QK, k)
    d_k = d_k + jnp.matmul(d_QK.T, q)

    # q_scaled = q * exp_g
    d_q = d_q + d_q_scaled * exp_g[:, None]
    d_exp_g = jnp.sum(d_q_scaled * q, axis=-1)

    # ---- backprop through solves ----
    tmp_v = _solve_unit_upper_from_A(A, d_v_pseudo)  # (I - A)^(-T) d
    d_v_beta = tmp_v
    dA = jnp.matmul(tmp_v, v_pseudo.T)

    tmp_k = _solve_unit_upper_from_A(A, d_k_cumdecay)
    d_k_scaled = tmp_k
    dA = dA + jnp.matmul(tmp_k, k_cumdecay.T)

    # k_scaled = k_beta * exp_g
    d_k_beta = d_k_scaled * exp_g[:, None]
    d_exp_g = d_exp_g + jnp.sum(d_k_scaled * k_beta, axis=-1)

    # A = -KKT * exp_diff * tril_strict
    dA = dA * tril_strict
    dKKT = -dA * exp_diff
    d_exp_diff = d_exp_diff + (-dA * KKT)

    # KKT = k_beta @ k^T
    d_k_beta = d_k_beta + jnp.matmul(dKKT, k)
    d_k = d_k + jnp.matmul(dKKT.T, k_beta)

    # exp_diff = exp(clip(diff))
    d_diff = (d_exp_diff * exp_diff) * diff_inrange
    d_g_cum = d_g_cum + jnp.sum(d_diff, axis=-1) - jnp.sum(d_diff, axis=-2)

    # exp_g = exp(clip(g_cum))
    d_g_cum = d_g_cum + d_exp_g * exp_g * gcum_inrange

    # g_cum = cumsum(g)  => d_g = reverse_cumsum(d_g_cum)
    d_g = jnp.zeros((Ct,), dtype=jnp.float32)
    acc = jnp.asarray(0.0, dtype=jnp.float32)
    for i in range(Ct - 1, -1, -1):
        mask_i = (idx == jnp.int32(i)).astype(jnp.float32)
        di = jnp.sum(d_g_cum * mask_i)
        acc = acc + di
        d_g = d_g + mask_i * acc

    # v_beta = v * beta
    d_v = d_v_beta * beta[:, None]
    d_beta = jnp.sum(d_v_beta * v, axis=-1)

    # k_beta = k * beta
    d_k = d_k + d_k_beta * beta[:, None]
    d_beta = d_beta + jnp.sum(d_k_beta * k, axis=-1)

    # ---- stores ----
    dq_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, K_pad)] = d_q[None, None, :, :].astype(dq_ref.dtype)
    dk_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, K_pad)] = d_k[None, None, :, :].astype(dk_ref.dtype)
    dv_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, V_pad)] = d_v[None, None, :, :].astype(dv_ref.dtype)
    dg_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, 1)] = d_g[None, None, :, None].astype(dg_ref.dtype)
    db_ref[dslice(0, 1), dslice(0, 1), dslice(0, Ct), dslice(0, 1)] = d_beta[None, None, :, None].astype(db_ref.dtype)
    dSprev_ref[dslice(0, 1), dslice(0, 1), dslice(0, K_pad), dslice(0, V_pad)] = dS_prev[None, None, :, :].astype(
        dSprev_ref.dtype
    )


def _gdn_chunk_step_bwd_pallas(
    q_bhck: jnp.ndarray,
    k_bhck: jnp.ndarray,
    v_bhcv: jnp.ndarray,
    g_cum_bhc: jnp.ndarray,
    b_bhc: jnp.ndarray,
    S_prev_bhkv: jnp.ndarray,
    d_out_bhcv: jnp.ndarray,
    dS_next_bhkv: jnp.ndarray,
    *,
    Ct: int,
    K_pad: int,
    V_pad: int,
):
    B, H = q_bhck.shape[0], q_bhck.shape[1]
    NH = B * H

    gcum4 = g_cum_bhc[..., None]
    b4 = b_bhc[..., None]

    out_shapes = (
        jax.ShapeDtypeStruct((B, H, Ct, K_pad), jnp.float32),  # dq
        jax.ShapeDtypeStruct((B, H, Ct, K_pad), jnp.float32),  # dk
        jax.ShapeDtypeStruct((B, H, Ct, V_pad), jnp.float32),  # dv
        jax.ShapeDtypeStruct((B, H, Ct, 1), jnp.float32),  # dg
        jax.ShapeDtypeStruct((B, H, Ct, 1), jnp.float32),  # db
        jax.ShapeDtypeStruct((B, H, K_pad, V_pad), jnp.float32),  # dS_prev
    )

    kernel = functools.partial(_gdn_chunk_step_bwd_kernel_tpu, Ct=int(Ct), K_pad=int(K_pad), V_pad=int(V_pad))

    compiler_params = None
    try:
        from jax.experimental.pallas import tpu as pltpu

        compiler_params = pltpu.CompilerParams(dimension_semantics=["parallel"])
    except Exception:
        pass

    in_specs, out_specs = _in_specs_chunk_bwd_tpu(B, H, Ct, K_pad, V_pad)

    return pl.pallas_call(
        kernel,
        out_shape=out_shapes,
        grid=(NH,),
        in_specs=in_specs,
        out_specs=out_specs,
        compiler_params=compiler_params,
    )(q_bhck, k_bhck, v_bhcv, gcum4, b4, S_prev_bhkv, d_out_bhcv, dS_next_bhkv)


def _chunk_gated_delta_rule_flash_pallas_bwd(chunk_size: int, seg: int, use_triangular_solve: bool, residuals, g_out):
    """Backward for flash chunk kernel (TPU).

    Robust to residual packing:
      - may include seg_starts (5D) and/or initial_state (4D or 3D) in either order.
    Uses:
      - g_cum precomputed outside Pallas
      - Pallas kernels for per-chunk state update and per-chunk backward
      - only scans over segments/chunks for the *state gradient* dependency
    """
    from jax.interpreters import ad

    # -----------------------
    # Unpack required primals
    # -----------------------
    if not isinstance(residuals, (tuple, list)) or len(residuals) < 5:
        raise ValueError("Unexpected residuals structure for flash chunk backward.")

    q_arr = residuals[0]
    k_arr = residuals[1]
    v_arr = residuals[2]
    g_arr = residuals[3]
    b_arr = residuals[4]
    tail = list(residuals[5:])

    # -----------------------
    # Unpack cotangents
    # -----------------------
    if isinstance(g_out, tuple):
        if len(g_out) >= 2:
            d_out, dS_final = g_out[0], g_out[1]
        else:
            d_out, dS_final = g_out[0], ad.Zero.from_value(0.0)
    else:
        d_out, dS_final = g_out, ad.Zero.from_value(0.0)

    # -----------------------
    # Shapes / static params
    # -----------------------
    B, H, L, dk_dim = q_arr.shape
    dv_dim = v_arr.shape[-1]
    C = int(chunk_size)
    seg = int(seg)

    TPU_MULT = 8

    def _round_up_to(x: int, m: int) -> int:
        return int(((int(x) + m - 1) // m) * m)

    Ct = _round_up_to(C, TPU_MULT)
    K_pad = _round_up_to(dk_dim, TPU_MULT)
    V_pad = _round_up_to(dv_dim, TPU_MULT)

    # Token padding -> chunk padding -> segment padding
    pad_tok = (C - (L % C)) % C
    Lt = L + pad_tok
    Nc = Lt // C
    if seg <= 0:
        seg = Nc if Nc > 0 else 1
    n_segments = (Nc + seg - 1) // seg
    n_chunks_pad = n_segments * seg
    L1 = n_chunks_pad * C  # length after chunk+segment padding

    # -----------------------
    # Parse optional residuals robustly
    # -----------------------
    seg_starts_res = None
    init_res = None

    for x in tail:
        if x is None:
            continue
        if hasattr(x, "ndim"):
            if x.ndim == 5:
                seg_starts_res = x
            elif x.ndim == 4 and x.shape[0] == B and x.shape[1] == H:
                init_res = x
            elif x.ndim == 3 and x.shape[0] == B * H:
                init_res = x.reshape(B, H, x.shape[1], x.shape[2])

    initial_state = init_res  # may be None

    # -----------------------
    # Materialize cotangents
    # -----------------------
    if isinstance(d_out, ad.Zero):
        d_out = jnp.zeros((B, H, L, dv_dim), dtype=jnp.float32)
    else:
        d_out = d_out.astype(jnp.float32)

    if isinstance(dS_final, ad.Zero) or dS_final is None:
        dS_final = jnp.zeros((B, H, dk_dim, dv_dim), dtype=jnp.float32)
    else:
        dS_final = dS_final.astype(jnp.float32)

    # -----------------------
    # Pad primals to (L1, K_pad/V_pad) and reshape to chunk/segment views
    # -----------------------
    q32 = q_arr.astype(jnp.float32)
    k32 = k_arr.astype(jnp.float32)
    v32 = v_arr.astype(jnp.float32)
    g32 = g_arr.astype(jnp.float32)
    b32 = b_arr.astype(jnp.float32)

    q_pad = jnp.pad(q32, ((0, 0), (0, 0), (0, L1 - L), (0, K_pad - dk_dim)))
    k_pad = jnp.pad(k32, ((0, 0), (0, 0), (0, L1 - L), (0, K_pad - dk_dim)))
    v_pad = jnp.pad(v32, ((0, 0), (0, 0), (0, L1 - L), (0, V_pad - dv_dim)))
    g_pad = jnp.pad(g32, ((0, 0), (0, 0), (0, L1 - L)))
    b_pad = jnp.pad(b32, ((0, 0), (0, 0), (0, L1 - L)))

    d_out_pad = jnp.pad(d_out, ((0, 0), (0, 0), (0, L1 - L), (0, V_pad - dv_dim)))

    q_c = q_pad.reshape(B, H, n_chunks_pad, C, K_pad)
    k_c = k_pad.reshape(B, H, n_chunks_pad, C, K_pad)
    v_c = v_pad.reshape(B, H, n_chunks_pad, C, V_pad)
    g_c = g_pad.reshape(B, H, n_chunks_pad, C)
    b_c = b_pad.reshape(B, H, n_chunks_pad, C)
    dO_c = d_out_pad.reshape(B, H, n_chunks_pad, C, V_pad)

    # Pad within-chunk axis from C -> Ct (TPU-friendly)
    if Ct != C:
        q_c = jnp.pad(q_c, ((0, 0), (0, 0), (0, 0), (0, Ct - C), (0, 0)))
        k_c = jnp.pad(k_c, ((0, 0), (0, 0), (0, 0), (0, Ct - C), (0, 0)))
        v_c = jnp.pad(v_c, ((0, 0), (0, 0), (0, 0), (0, Ct - C), (0, 0)))
        g_c = jnp.pad(g_c, ((0, 0), (0, 0), (0, 0), (0, Ct - C)))
        b_c = jnp.pad(b_c, ((0, 0), (0, 0), (0, 0), (0, Ct - C)))
        dO_c = jnp.pad(dO_c, ((0, 0), (0, 0), (0, 0), (0, Ct - C), (0, 0)))

    # Precompute g_cum outside Pallas: (B,H,n_chunks_pad,Ct)
    g_cum_c = jnp.cumsum(g_c, axis=-1)

    # Segment views:
    #   (n_segments, seg, B, H, Ct, feat) and (n_segments, seg, B, H, Ct)
    def _to_seg(x, has_feat: bool):
        if has_feat:
            x = x.reshape(B, H, n_segments, seg, Ct, x.shape[-1])
            return jnp.transpose(x, (2, 3, 0, 1, 4, 5))
        else:
            x = x.reshape(B, H, n_segments, seg, Ct)
            return jnp.transpose(x, (2, 3, 0, 1, 4))

    q_seg = _to_seg(q_c, True)
    k_seg = _to_seg(k_c, True)
    v_seg = _to_seg(v_c, True)
    gcum_seg = _to_seg(g_cum_c, False)
    b_seg = _to_seg(b_c, False)
    dO_seg = _to_seg(dO_c, True)

    # -----------------------
    # Initial / final state pads
    # -----------------------
    init_shape_for_return = None
    init_dtype_for_return = None
    init_is_padded = False

    if initial_state is None:
        S0_pad = jnp.zeros((B, H, K_pad, V_pad), dtype=jnp.float32)
    else:
        init_dtype_for_return = initial_state.dtype
        init_shape_for_return = initial_state.shape

        init32 = initial_state.astype(jnp.float32)
        if init32.shape[-2] == K_pad and init32.shape[-1] == V_pad:
            S0_pad = init32
            init_is_padded = True
        else:
            pad_k = K_pad - init32.shape[-2]
            pad_v = V_pad - init32.shape[-1]
            if pad_k < 0 or pad_v < 0:
                raise ValueError(f"initial_state larger than pads: {init32.shape} vs {(K_pad, V_pad)}")
            S0_pad = jnp.pad(init32, ((0, 0), (0, 0), (0, pad_k), (0, pad_v)))
            init_is_padded = False

    dS_final_pad = jnp.pad(dS_final, ((0, 0), (0, 0), (0, K_pad - dk_dim), (0, V_pad - dv_dim)))

    # -----------------------
    # Phase 1: compute segment-start states S_starts[s]
    # -----------------------
    use_stashed_seg_starts = False
    if seg_starts_res is not None and hasattr(seg_starts_res, "shape") and seg_starts_res.ndim == 5:
        if seg_starts_res.shape[0] == n_segments and seg_starts_res.shape[1] == B and seg_starts_res.shape[2] == H:
            use_stashed_seg_starts = True

    if use_stashed_seg_starts:
        S_starts = seg_starts_res.astype(jnp.float32)

        # >>> IMPORTANT FIX: reconcile stashed (K,V) with our (K_pad,V_pad) without negative pad <<<
        k_cur = int(S_starts.shape[-2])
        v_cur = int(S_starts.shape[-1])
        k_use = min(k_cur, K_pad)
        v_use = min(v_cur, V_pad)

        # truncate down if stashed is larger
        if k_use != k_cur or v_use != v_cur:
            S_starts = S_starts[..., :k_use, :v_use]

        # pad up if stashed is smaller
        pad_k = K_pad - k_use
        pad_v = V_pad - v_use
        if pad_k > 0 or pad_v > 0:
            S_starts = jnp.pad(S_starts, ((0, 0), (0, 0), (0, 0), (0, pad_k), (0, pad_v)))

    else:
        def _seg_state_fwd(S_in, seg_inps):
            k_s, v_s, gc_s, b_s = seg_inps

            def _chunk_state_body(S_prev, inp):
                k_i, v_i, gc_i, b_i = inp
                S_next = _gdn_chunk_state_update_pallas(
                    k_i, v_i, gc_i, b_i, S_prev,
                    Ct=Ct, K_pad=K_pad, V_pad=V_pad,
                )
                return S_next, S_prev

            S_out, _ = lax.scan(_chunk_state_body, S_in, (k_s, v_s, gc_s, b_s), length=seg)
            return S_out, S_in

        _, S_starts = lax.scan(
            _seg_state_fwd,
            S0_pad,
            (k_seg, v_seg, gcum_seg, b_seg),
            length=n_segments,
        )

    # -----------------------
    # Phase 2: reverse over segments, and within each segment reverse over chunks
    # -----------------------
    S_starts_rev = S_starts[::-1]
    q_seg_rev = q_seg[::-1]
    k_seg_rev = k_seg[::-1]
    v_seg_rev = v_seg[::-1]
    gcum_seg_rev = gcum_seg[::-1]
    b_seg_rev = b_seg[::-1]
    dO_seg_rev = dO_seg[::-1]

    def _seg_bwd_body(dS_after_seg, seg_inps):
        S_start, q_s, k_s, v_s, gc_s, b_s, dO_s = seg_inps

        def _chunk_state_body(S_prev, inp):
            k_i, v_i, gc_i, b_i = inp
            S_next = _gdn_chunk_state_update_pallas(
                k_i, v_i, gc_i, b_i, S_prev,
                Ct=Ct, K_pad=K_pad, V_pad=V_pad,
            )
            return S_next, S_prev

        _, S_prev_list = lax.scan(_chunk_state_body, S_start, (k_s, v_s, gc_s, b_s), length=seg)

        q_r = q_s[::-1]
        k_r = k_s[::-1]
        v_r = v_s[::-1]
        gc_r = gc_s[::-1]
        b_r = b_s[::-1]
        dO_r = dO_s[::-1]
        Sprev_r = S_prev_list[::-1]

        def _chunk_bwd_body(dS_next, inp):
            q_i, k_i, v_i, gc_i, b_i, S_prev_i, dO_i = inp
            dq_i, dk_i, dv_i, dg_i4, db_i4, dS_prev_i = _gdn_chunk_step_bwd_pallas(
                q_i, k_i, v_i, gc_i, b_i, S_prev_i, dO_i, dS_next,
                Ct=Ct, K_pad=K_pad, V_pad=V_pad,
            )
            dg_i = dg_i4.reshape(B, H, Ct)
            db_i = db_i4.reshape(B, H, Ct)
            return dS_prev_i, (dq_i, dk_i, dv_i, dg_i, db_i)

        dS_start_grad, (dq_r, dk_r, dv_r, dg_r, db_r) = lax.scan(
            _chunk_bwd_body,
            dS_after_seg,
            (q_r, k_r, v_r, gc_r, b_r, Sprev_r, dO_r),
            length=seg,
        )

        return dS_start_grad, (dq_r[::-1], dk_r[::-1], dv_r[::-1], dg_r[::-1], db_r[::-1])

    dS0_pad, (dq_segs_rev, dk_segs_rev, dv_segs_rev, dg_segs_rev, db_segs_rev) = lax.scan(
        _seg_bwd_body,
        dS_final_pad,
        (S_starts_rev, q_seg_rev, k_seg_rev, v_seg_rev, gcum_seg_rev, b_seg_rev, dO_seg_rev),
        length=n_segments,
    )

    dq_segs = dq_segs_rev[::-1]
    dk_segs = dk_segs_rev[::-1]
    dv_segs = dv_segs_rev[::-1]
    dg_segs = dg_segs_rev[::-1]
    db_segs = db_segs_rev[::-1]

    # -----------------------
    # Assemble back to (B,H,L,dim)
    # -----------------------
    def _seg_to_chunks_kt(x):
        x = jnp.transpose(x, (2, 3, 0, 1, 4, 5))
        return x.reshape(B, H, n_chunks_pad, Ct, x.shape[-1])

    def _seg_to_chunks_t(x):
        x = jnp.transpose(x, (2, 3, 0, 1, 4))
        return x.reshape(B, H, n_chunks_pad, Ct)

    dq_chunks = _seg_to_chunks_kt(dq_segs)
    dk_chunks = _seg_to_chunks_kt(dk_segs)
    dv_chunks = _seg_to_chunks_kt(dv_segs)
    dg_chunks = _seg_to_chunks_t(dg_segs)
    db_chunks = _seg_to_chunks_t(db_segs)

    dq_flat = dq_chunks[:, :, :, :C, :].reshape(B, H, L1, K_pad)[:, :, :L, :dk_dim]
    dk_flat = dk_chunks[:, :, :, :C, :].reshape(B, H, L1, K_pad)[:, :, :L, :dk_dim]
    dv_flat = dv_chunks[:, :, :, :C, :].reshape(B, H, L1, V_pad)[:, :, :L, :dv_dim]
    dg_flat = dg_chunks[:, :, :, :C].reshape(B, H, L1)[:, :, :L]
    db_flat = db_chunks[:, :, :, :C].reshape(B, H, L1)[:, :, :L]

    dq_ret = dq_flat.astype(q_arr.dtype)
    dk_ret = dk_flat.astype(k_arr.dtype)
    dv_ret = dv_flat.astype(v_arr.dtype)
    dg_ret = dg_flat.astype(g_arr.dtype)
    db_ret = db_flat.astype(b_arr.dtype)

    # initial_state grad: must match the primal initial_state shape if it was present
    if initial_state is None:
        dS0_ret = None
    else:
        if init_is_padded:
            dS0_ret = dS0_pad.astype(init_dtype_for_return)
        else:
            dk0 = init_shape_for_return[-2]
            dv0 = init_shape_for_return[-1]
            dS0_ret = dS0_pad[:, :, :dk0, :dv0].astype(init_dtype_for_return)

    return dq_ret, dk_ret, dv_ret, dg_ret, db_ret, dS0_ret


_chunk_gated_delta_rule_flash_pallas.defvjp(
    _chunk_gated_delta_rule_flash_pallas_fwd,
    _chunk_gated_delta_rule_flash_pallas_bwd,
)


def _apply_T_from_strict_A(A_strict: jnp.ndarray, rhs: jnp.ndarray, *, use_triangular_solve: bool) -> jnp.ndarray:
    """Compute (I - A_strict)^(-1) @ rhs for strictly-lower A_strict (diag must be 0).

    A_strict: [..., C, C] strictly lower triangular (diagonal is 0)
    rhs:      [..., C, R]  (R can be dv, dk, etc.)

    Returns:  [..., C, R]
    """
    if use_triangular_solve:
        # Solve (I - A) X = rhs.
        # If A is strictly lower, then (I - A) is lower-triangular with unit diagonal.
        #
        # triangular_solve with unit_diagonal=True treats diag as 1 and reads only off-diagonal.
        # Passing -A makes the effective matrix: I + (-A) = I - A.
        return lax.linalg.triangular_solve(
            -A_strict,
            rhs,
            left_side=True,
            lower=True,
            transpose_a=False,
            conjugate_a=False,
            unit_diagonal=True,
        )
    else:
        C = A_strict.shape[-1]
        T = _forward_subst(A_strict, C)  # explicit T = (I - A)^(-1)
        return jnp.matmul(T, rhs)


def _forward_subst(A: jnp.ndarray, C: int) -> jnp.ndarray:
    """Forward substitution to compute T = (I - A)^-1 for strictly lower-triangular A.

    Works for A of shape [..., C, C] with any leading batch dims.
    Returns T with the same shape/dtype as A (typically fp32).
    """
    eyeC = jnp.eye(C, dtype=A.dtype)
    T = jnp.broadcast_to(eyeC, A.shape)

    idx = jnp.arange(C)

    def body(i, T_acc):
        # A_row_i: [..., C]
        A_row_i = lax.dynamic_slice_in_dim(A, i, 1, axis=-2)
        A_row_i = jnp.squeeze(A_row_i, axis=-2)

        # Only j < i contribute
        mask = (idx < i).astype(A.dtype)
        A_row_masked = A_row_i * mask

        # contrib[k] = sum_j A[i,j] * T[j,k]
        contrib = jnp.einsum("...j,...jk->...k", A_row_masked, T_acc)

        # Update row i of T
        T_row_i = lax.dynamic_slice_in_dim(T_acc, i, 1, axis=-2)
        T_row_i = jnp.squeeze(T_row_i, axis=-2)
        new_row = T_row_i + contrib

        return lax.dynamic_update_slice_in_dim(T_acc, new_row[..., None, :], i, axis=-2)

    T = lax.fori_loop(1, C, body, T)
    return T


def _chunk_gated_delta_rule_fused(
    q_arr: jnp.ndarray,
    k_arr: jnp.ndarray,
    v_arr: jnp.ndarray,
    g_arr: jnp.ndarray,
    b_arr: jnp.ndarray,
    *,
    chunk_size: int,
    initial_state: Optional[jnp.ndarray],
    use_checkpoint: bool = True,
    use_triangular_solve: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Core fused implementation of chunkwise gated delta rule.

    This version fuses A/T computation into the scan, computing them on-the-fly
    for each chunk. Memory usage is O(B*H*C*C) instead of O(B*H*Nc*C*C).
    """
    B_, H_, L_, dk_ = q_arr.shape
    dv_ = v_arr.shape[-1]
    C = int(chunk_size)

    # Pad sequence to multiple of chunk_size
    pad = (C - (L_ % C)) % C
    Lt = L_ + pad
    Nc = Lt // C

    if pad > 0:
        q_arr = jnp.pad(q_arr, ((0, 0), (0, 0), (0, pad), (0, 0)))
        k_arr = jnp.pad(k_arr, ((0, 0), (0, 0), (0, pad), (0, 0)))
        v_arr = jnp.pad(v_arr, ((0, 0), (0, 0), (0, pad), (0, 0)))
        g_arr = jnp.pad(g_arr, ((0, 0), (0, 0), (0, pad)))
        b_arr = jnp.pad(b_arr, ((0, 0), (0, 0), (0, pad)))

    # Reshape to chunks: (B, H, Nc, C, D)
    q_c = q_arr.reshape(B_, H_, Nc, C, dk_)
    k_c = k_arr.reshape(B_, H_, Nc, C, dk_)
    v_c = v_arr.reshape(B_, H_, Nc, C, dv_)
    g_c = g_arr.reshape(B_, H_, Nc, C)
    b_c = b_arr.reshape(B_, H_, Nc, C)

    # Initial state
    S0 = (
        jnp.zeros((B_, H_, dk_, dv_), dtype=jnp.float32)
        if initial_state is None
        else initial_state.astype(jnp.float32)
    )

    # Pre-compute constant masks
    tril_mask_strict = jnp.tril(jnp.ones((C, C), dtype=jnp.float32), k=-1)  # strict lower
    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.float32))  # lower triangular incl diag

    def chunk_step_fused(S_prev, chunk_inputs):
        q_i, k_i, v_i, g_i, b_i = chunk_inputs
        # q_i,k_i: (B, H, C, dk), v_i: (B,H,C,dv), g_i,b_i: (B,H,C)

        # (1) Per-chunk quantities
        v_beta = v_i * b_i[..., None]  # (B,H,C,dv)
        k_beta = k_i * b_i[..., None]  # (B,H,C,dk)
        g_cum = jnp.cumsum(g_i, axis=-1)  # (B,H,C)

        # (2) Build A
        KKT = jnp.einsum("bhid,bhjd->bhij", k_beta, k_i)  # (B,H,C,C)
        g_i_exp = g_cum[..., :, None]  # (B,H,C,1)
        g_j_exp = g_cum[..., None, :]  # (B,H,1,C)
        diff_ij = g_i_exp - g_j_exp
        diff_ij_clamped = jnp.clip(diff_ij, -80.0, 80.0)
        decay_ij = jnp.exp(diff_ij_clamped)
        A = -KKT * decay_ij * tril_mask_strict  # strictly lower (B,H,C,C)

        # (3/4) Apply T = (I - A)^(-1) to RHS without explicitly building T (if enabled)
        # v_pseudo = T @ (βV)
        v_pseudo = _apply_T_from_strict_A(A, v_beta, use_triangular_solve=use_triangular_solve)

        # k_cumdecay = T @ (βK ⊙ exp(g_cum))
        g_cum_clamped = jnp.clip(g_cum, -80.0, 80.0)
        exp_g_cum = jnp.exp(g_cum_clamped)[..., None]  # (B,H,C,1)
        k_scaled = k_beta * exp_g_cum  # (B,H,C,dk)
        k_cumdecay = _apply_T_from_strict_A(A, k_scaled, use_triangular_solve=use_triangular_solve)

        # (5) Within-chunk decay-weighted attention
        decay_attn = jnp.exp(diff_ij_clamped) * causal_mask
        attn_i = jnp.einsum("bhid,bhjd->bhij", q_i, k_i) * decay_attn  # (B,H,C,C)

        # (6) Cross-chunk contribution
        v_prime = jnp.einsum("bhid,bhdm->bhim", k_cumdecay, S_prev)  # (B,H,C,dv)
        v_new = v_pseudo - v_prime

        # (7) Output
        q_scaled = q_i * jnp.exp(g_cum_clamped)[..., None]  # (B,H,C,dk)
        inter = jnp.einsum("bhid,bhdm->bhim", q_scaled, S_prev)  # (B,H,C,dv)
        intra = jnp.einsum("bhij,bhjm->bhim", attn_i, v_new)  # (B,H,C,dv)
        out_i = inter + intra

        # (8) Update state
        g_tail = g_cum[..., -1]  # (B,H)
        g_tail_clamped = jnp.clip(g_tail, -80.0, 80.0)
        decay_tail = jnp.exp(g_tail_clamped)[..., None, None]  # (B,H,1,1)

        decay_diff = jnp.clip(g_tail[..., None] - g_cum, -80.0, 80.0)  # (B,H,C)
        decay_weights = jnp.exp(decay_diff)[..., None]  # (B,H,C,1)

        add = jnp.einsum("bhid,bhim->bhdm", k_i * decay_weights, v_new)  # (B,H,dk,dv)
        S_new = S_prev * decay_tail + add
        return S_new, out_i

    # Time-major scan over chunks
    q_scan = jnp.moveaxis(q_c, 2, 0)  # (Nc,B,H,C,dk)
    k_scan = jnp.moveaxis(k_c, 2, 0)
    v_scan = jnp.moveaxis(v_c, 2, 0)
    g_scan = jnp.moveaxis(g_c, 2, 0)
    b_scan = jnp.moveaxis(b_c, 2, 0)

    step_fn = jax.checkpoint(chunk_step_fused) if use_checkpoint else chunk_step_fused

    S_final, out_chunks = lax.scan(step_fn, S0, (q_scan, k_scan, v_scan, g_scan, b_scan), length=Nc)

    # Reshape output back to (B,H,L,dv)
    out_arr = jnp.moveaxis(out_chunks, 0, 2)  # (B,H,Nc,C,dv)
    out_arr = out_arr.reshape(B_, H_, Lt, dv_)
    out_arr = out_arr[:, :, :L_, :]
    return out_arr, S_final


def chunk_gated_delta_rule(
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    g: NamedArray,
    beta: NamedArray,
    chunk_size: int = 64,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_flash: bool = True,
    *,
    head_first: bool = False,
    offsets: Optional[jnp.ndarray] = None,
    use_varlen: bool = False,
    lengths: Optional[jnp.ndarray] = None,
    backward_mode: "Literal['checkpoint','custom_vjp']" = "checkpoint",
    use_triangular_solve: bool = True,
) -> tuple[NamedArray, Optional[jnp.ndarray]]:
    """Top-level API for chunkwise gated delta rule."""
    Batch = query.resolve_axis("batch")
    Heads = query.resolve_axis("heads")
    Dk = query.resolve_axis("k_head_dim")
    Dv = value.resolve_axis("v_head_dim")

    pos_axis_names = [ax.name for ax in query.axes if ax.name not in ("batch", "heads", "k_head_dim")]
    assert len(pos_axis_names) == 1, f"Expected exactly one position axis, got {pos_axis_names}"
    Pos = query.resolve_axis(pos_axis_names[0])

    B_, H_, L_, dk_, _ = Batch.size, Heads.size, Pos.size, Dk.size, Dv.size
    NH = B_ * H_

    # Varlen: offsets -> lengths if needed
    if use_varlen:
        if lengths is None and offsets is not None:
            # offsets may be (NH,) lengths or (NH+1,) prefix sums
            if offsets.shape[0] == NH + 1:
                lengths_flat = offsets[1:] - offsets[:-1]
            else:
                lengths_flat = offsets
            lengths_bh = lengths_flat.reshape(B_, H_).astype(jnp.int32)
        elif lengths is None:
            lengths_bh = jnp.full((B_, H_), L_, dtype=jnp.int32)
        else:
            lengths_bh = lengths.astype(jnp.int32)
            if lengths_bh.ndim == 1:
                lengths_bh = lengths_bh.reshape(B_, H_)
    else:
        lengths_bh = jnp.full((B_, H_), L_, dtype=jnp.int32)

    def _to_bhtd(x: NamedArray, d_axis_name: str) -> jnp.ndarray:
        target_layout = (Batch.name, Heads.name, Pos.name, d_axis_name)
        have = tuple(ax.name for ax in x.axes)
        if have == target_layout:
            return x.array
        return hax.rearrange(x, target_layout).array

    def _to_bht(x: NamedArray) -> jnp.ndarray:
        target_layout = (Batch.name, Heads.name, Pos.name)
        have = tuple(ax.name for ax in x.axes)
        if have == target_layout:
            return x.array
        return hax.rearrange(x, target_layout).array

    # Promote + optional L2 norm + scale
    q32 = query.astype(jnp.float32)
    k32 = key.astype(jnp.float32)
    v32 = value.astype(jnp.float32)
    g32 = g.astype(jnp.float32)
    b32 = beta.astype(jnp.float32)

    if use_qk_l2norm_in_kernel:
        q32 = _l2norm(q32, axis=Dk)
        k32 = _l2norm(k32, axis=Dk)
    q32 = q32 * (dk_**-0.5)

    # Layout to (B,H,L,·)
    q_arr = _to_bhtd(q32, Dk.name)
    k_arr = _to_bhtd(k32, Dk.name)
    v_arr = _to_bhtd(v32, Dv.name)
    g_arr = _to_bht(g32)
    b_arr = _to_bht(b32)

    # Varlen masking: beyond length => q/k/v/b=0, g=0 (alpha=1)
    if use_varlen:
        t_idx = jnp.arange(L_, dtype=jnp.int32)[None, None, :]  # (1,1,L)
        valid = (t_idx < lengths_bh[:, :, None]).astype(jnp.float32)  # (B,H,L)
        q_arr = q_arr * valid[..., None]
        k_arr = k_arr * valid[..., None]
        v_arr = v_arr * valid[..., None]
        b_arr = b_arr * valid
        g_arr = g_arr * valid

    # Ensure chunk_size and use_triangular_solve are Python types (not tracers)
    chunk_size = int(chunk_size)
    use_triangular_solve = bool(use_triangular_solve)

    if backward_mode == "custom_vjp":
        seg = 1
    elif backward_mode == "checkpoint":
        seg = 8  # tune; min(seg, Nc) is applied internally
    else:
        raise ValueError(backward_mode)

    if use_flash:
        out_arr, S_final = _chunk_gated_delta_rule_flash_pallas(
            q_arr, k_arr, v_arr, g_arr, b_arr, int(chunk_size), int(seg), initial_state, True
        )
    else:
        # fallback: pure JAX forward (and JAX autodiff)
        out_arr, S_final = _chunk_gated_delta_rule_fused(
            q_arr,
            k_arr,
            v_arr,
            g_arr,
            b_arr,
            chunk_size=int(chunk_size),
            initial_state=initial_state,
            use_checkpoint=(backward_mode == "checkpoint"),
            use_triangular_solve=use_triangular_solve,
        )

    # Back to NamedArray (B, L, H, dv)
    out_named = hax.named(out_arr, (Batch, Heads, Pos, Dv))
    out_final = hax.rearrange(out_named, (Batch, Pos, Heads, Dv))

    return (out_final, S_final) if output_final_state else (out_final, None)


# ---------- Layer ----------


class GatedDeltaNet(eqx.Module):
    """Complete Gated DeltaNet layer (projections + conv + kernels + norm + out proj).

    Block structure (per token t):
      1) Linear projections → [Q | K | V | Z] and [b | a]
      2) Short depthwise causal Conv1D over concatenated [Q|K|V] channels
      3) Compute gates:
           β_t = σ(b_t)              (per V-head)
           g_t = -exp(A)·softplus(a_t + dt_bias)   (per V-head)
           α_t = exp(g_t)
      4) Core kernel:
           - prefill/train:  chunk_gated_delta_rule (chunkwise parallel, returns S_T)
           - decode:         recurrent_gated_delta_rule (sequential, updates S)
      5) Gated RMSNorm with Z:  RMSNorm(o) * SiLU(Z)
      6) Output projection back to model dim.

    Caching (inference):
      - conv_state: (N, Channels, K) running window for the causal depthwise conv
      - S_state:    (B, H, d_k, d_v) cross-chunk recurrent state for the delta rule

    Head layout:
      - If num_v_heads > num_k_heads, Q/K are repeated across V-head groups so each V-head
        has a corresponding Q,K.
    """

    config: GatedDeltaNetConfig = eqx.field(static=True)
    use_flash: bool = eqx.field(static=True)

    # projections
    in_proj_qkvz: hnn.Linear  # [Embed] -> [Q|K|V|Z]
    in_proj_ba: hnn.Linear  # [Embed] -> [b|a]

    # depthwise conv parameters over concatenated [Q|K|V] channels
    conv_weight: NamedArray  # [channels, conv_kernel]
    conv_bias: Optional[NamedArray]  # [channels] or None

    # discretization params per V head (Mamba2-style)
    A_log: NamedArray  # [Heads]
    dt_bias: NamedArray  # [Heads]

    # gated RMSNorm and output projection
    o_norm: FusedRMSNormGated
    out_proj: hnn.Linear  # [Heads, VHeadDim] -> [Embed]

    @staticmethod
    def init(config: GatedDeltaNetConfig, *, use_flash: bool = True, key) -> "GatedDeltaNet":
        """Initializer mirrors the HF defaults: no biases in projections/out_proj;
        A_log ~ log U(0,16), dt_bias = 1, small conv kernel."""
        k_qkvz, k_ba, k_conv, k_out = jax.random.split(key, 4)
        in_proj_qkvz = hnn.Linear.init(
            In=config.Embed,
            Out=config.mix_qkvz_axis,
            out_first=True,
            use_bias=False,
            key=k_qkvz,
        )
        in_proj_ba = hnn.Linear.init(
            In=config.Embed,
            Out=config.ba_axis,
            out_first=True,
            use_bias=False,
            key=k_ba,
        )

        # Depthwise conv over channels = 2*key_dim + value_dim
        C = config.key_dim * 2 + config.value_dim
        K = config.conv_kernel_size
        ConvChannels = Axis("channels", C)
        ConvKernel = Axis("conv_kernel", K)

        conv_w = jax.random.normal(k_conv, (C, K), dtype=jnp.float32) * (1.0 / jnp.sqrt(C * K))
        conv_weight = hax.named(conv_w, (ConvChannels, ConvKernel))
        conv_bias = None

        # GDN discretization parameters (per V-head)
        A_log = hax.named(
            jnp.log(jax.random.uniform(k_out, (config.Heads.size,), minval=1e-6, maxval=16.0, dtype=jnp.float32)),
            (config.Heads.name,),
        )
        dt_bias = hax.named(jnp.ones((config.Heads.size,), dtype=jnp.float32), (config.Heads.name,))

        o_norm = FusedRMSNormGated.init(config.VHeadDim, eps=config.rms_norm_eps, use_flash=use_flash)
        out_proj = hnn.Linear.init(
            In=(config.Heads, config.VHeadDim), Out=config.Embed, out_first=True, use_bias=False, key=k_out
        )
        return GatedDeltaNet(
            config=config,
            in_proj_qkvz=in_proj_qkvz,
            in_proj_ba=in_proj_ba,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            A_log=A_log,
            dt_bias=dt_bias,
            o_norm=o_norm,
            out_proj=out_proj,
            use_flash=use_flash,
        )

    def _fix_qkvz_ordering(
        self,
        mixed_qkvz: NamedArray,  # [B, Pos, qkvz=2*key_dim + 2*value_dim]
        mixed_ba: NamedArray,  # [B, Pos, 2*num_v_heads]
    ) -> Tuple[NamedArray, NamedArray, NamedArray, NamedArray, NamedArray, NamedArray]:
        """Split packed projections into per-head tensors and align head layout. (match HF version)

        Input shapes:
          mixed_qkvz: [B, Pos, 2*key_dim + 2*value_dim]  (Q|K|V|Z concatenated)
          mixed_ba:   [B, Pos, 2*num_v_heads]            (b|a per V-head)

        Returns:
          q: [B, Pos, KHeads, KHeadDim]
          k: [B, Pos, KHeads, KHeadDim]
          v: [B, Pos, VHeads, VHeadDim]
          z: [B, Pos, VHeads, VHeadDim]
          b: [B, Pos, VHeads]        (→ β via sigmoid)
          a: [B, Pos, VHeads]        (→ g via Mamba2-style discretization)
        """
        cfg = self.config
        ratio = cfg.num_v_heads // cfg.num_k_heads

        per_head = Axis("per_head", 2 * cfg.head_k_dim + 2 * ratio * cfg.head_v_dim)
        x = mixed_qkvz.unflatten_axis("qkvz", (cfg.KHeads, per_head))

        def sl(start, size):
            return hax.ds(start, size)

        # per-head order: [Q (dk)] [K (dk)] [V-chunk (ratio*dv)] [Z-chunk (ratio*dv)]
        q = x["per_head", sl(0, cfg.head_k_dim)].rename({"per_head": cfg.KHeadDim.name})
        k = x["per_head", sl(cfg.head_k_dim, cfg.head_k_dim)].rename({"per_head": cfg.KHeadDim.name})
        v_chunk = x["per_head", sl(2 * cfg.head_k_dim, ratio * cfg.head_v_dim)]
        z_chunk = x["per_head", sl(2 * cfg.head_k_dim + ratio * cfg.head_v_dim, ratio * cfg.head_v_dim)]

        # (KHeads, ratio*dv) → (VHeads, VHeadDim)
        v = v_chunk.unflatten_axis(
            v_chunk.resolve_axis("per_head"), (Axis("v_group", ratio), cfg.VHeadDim)
        ).flatten_axes(("k_heads", "v_group"), cfg.VHeads)
        z = z_chunk.unflatten_axis(
            z_chunk.resolve_axis("per_head"), (Axis("v_group", ratio), cfg.VHeadDim)
        ).flatten_axes(("k_heads", "v_group"), cfg.VHeads)

        # b | a are per V-head; shape path mirrors HF:
        per_ba = Axis("per_ba", 2 * ratio)
        ba = mixed_ba.unflatten_axis("ba", (cfg.KHeads, per_ba))
        b_chunk = ba["per_ba", hax.ds(0, ratio)]
        a_chunk = ba["per_ba", hax.ds(ratio, ratio)]
        b = b_chunk.flatten_axes(("k_heads", "per_ba"), cfg.VHeads)
        a = a_chunk.flatten_axes(("k_heads", "per_ba"), cfg.VHeads)

        return q, k, v, z, b, a

    def __call__(
        self,
        x: NamedArray,
        *,
        inference: bool = True,
        chunk_size: int = 64,
        attention_mask: Optional[NamedArray] = None,
        decode_state: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,  # (conv_state, S_state)
    ) -> Tuple[NamedArray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        """Run the full GDN token mixer.

        Args:
          x: [B, Pos, Embed]
          inference: if True, returns and expects state for streaming decode.
          chunk_size: chunk length for the parallel kernel (prefill/train).
          attention_mask: optional [B, Pos] (1 for real tokens, 0 for pad).
          decode_state: optional tuple (conv_state, S_state) for streaming decode:
              conv_state: (N, Channels, K)
              S_state:    (B, VHeads, d_k, d_v)

        Returns:
          y_out: [B, Pos, Embed]
          new_state (optional): (conv_state, S_state) if inference=True
        """
        cfg = self.config

        # Zero out padding tokens early so they don't affect conv or states.
        if attention_mask is not None:
            m3 = attention_mask.astype(x.dtype).broadcast_axis(cfg.Embed)
            x = x * m3

        _dbg("layer/in_x", x.array if hasattr(x, "array") else x)

        # 1) Project to [Q|K|V|Z] and [b|a]
        mixed_qkvz = self.in_proj_qkvz(x)  # [B, Pos, qkvz=2*key_dim + 2*value_dim]
        mixed_ba = self.in_proj_ba(x)  # [B, Pos, ba=2*num_v_heads]
        _dbg("layer/mixed_qkvz", mixed_qkvz.array if hasattr(mixed_qkvz, "array") else mixed_qkvz)
        _dbg("layer/mixed_ba", mixed_ba.array if hasattr(mixed_ba, "array") else mixed_ba)

        # 1b) Re-group like HF for parity (also used for conv channel ordering)
        q, k, v, z, b, a = self._fix_qkvz_ordering(mixed_qkvz, mixed_ba)

        # 2) Depthwise causal conv over concatenated [Q|K|V] channels
        #    HF orders channels as: [Q_flat | K_flat | V_flat] (no Z).
        q_ch = q.flatten_axes((cfg.KHeads, cfg.KHeadDim), Axis("channels", cfg.key_dim))
        k_ch = k.flatten_axes((cfg.KHeads, cfg.KHeadDim), Axis("channels", cfg.key_dim))
        v_ch = v.flatten_axes((cfg.VHeads, cfg.VHeadDim), Axis("channels", cfg.value_dim))
        qkv_ch = hax.concatenate("channels", [q_ch, k_ch, v_ch])  # [B, Pos, channels]
        qkv_ncl = hax.rearrange(qkv_ch, ("batch", "channels", "position")).array  # (N, C, L)
        _dbg("conv/in_ncl", qkv_ncl)

        S_state: Optional[jnp.ndarray] = None
        if decode_state is not None and x.axis_size("position") == 1:
            # Streaming decode: cheap single-step conv update + carry conv_state
            conv_state, S_state = decode_state
            K = self.conv_weight.resolve_axis("conv_kernel").size
            assert conv_state.shape[-1] == K
            _dbg("conv/state_in_decode", conv_state)
            y_ncl, new_conv_state = _causal_depthwise_conv1d_update(
                qkv_ncl,
                self.conv_weight.array,
                self.conv_bias.array if self.conv_bias is not None else None,
                conv_state,
            )
        else:
            # Prefill/train: full causal conv over the sequence
            y_ncl = _causal_depthwise_conv1d_full(
                qkv_ncl, self.conv_weight.array, self.conv_bias.array if self.conv_bias is not None else None
            )
            if inference:
                # cache the rightmost K samples of channels as the next conv_state
                K = self.conv_weight.resolve_axis("conv_kernel").size
                Lpos = x.axis_size("position")
                if Lpos >= K:
                    new_conv_state = qkv_ncl[..., -K:]
                else:
                    new_conv_state = jnp.pad(qkv_ncl, ((0, 0), (0, 0), (K - Lpos, 0)))
            else:
                new_conv_state = None
                S_state = None

        _dbg("conv/out_ncl", y_ncl)

        # Unpack [Q|K|V] after conv back to per-head tensors (mirror the same channel order)
        y_bpc = hax.rearrange(hax.named(y_ncl, ("batch", "channels", "position")), ("batch", "position", "channels"))
        q_y = y_bpc["channels", hax.ds(0, cfg.key_dim)]
        k_y = y_bpc["channels", hax.ds(cfg.key_dim, cfg.key_dim)]
        v_y = y_bpc["channels", hax.ds(2 * cfg.key_dim, cfg.value_dim)]
        q = q_y.unflatten_axis("channels", (cfg.KHeads, cfg.KHeadDim))
        k = k_y.unflatten_axis("channels", (cfg.KHeads, cfg.KHeadDim))
        v = v_y.unflatten_axis("channels", (cfg.VHeads, cfg.VHeadDim))

        # 3) Gates: β via sigmoid(b); α via g = -exp(A) * softplus(a + dt_bias), α=exp(g)
        # Map a, b to Heads axis to line up with TP and kernels.
        ratio = cfg.num_v_heads // cfg.num_k_heads
        if ratio > 1:
            VGroup = Axis("v_group", ratio)
            # Repeat Q,K to Heads (num_v_heads)
            q = q.broadcast_axis(VGroup).flatten_axes((cfg.KHeads, VGroup), cfg.Heads)
            k = k.broadcast_axis(VGroup).flatten_axes((cfg.KHeads, VGroup), cfg.Heads)
            # Map V/Z/B/A to Heads as well
            v_h = v.rename({cfg.VHeads.name: cfg.Heads.name})
            z_h = z.rename({cfg.VHeads.name: cfg.Heads.name})
            b_hparam = b.rename({cfg.VHeads.name: cfg.Heads.name})
            a_hparam = a.rename({cfg.VHeads.name: cfg.Heads.name})
        else:
            # 1:1 map KHeads -> Heads; and VHeads -> Heads for v/z/a/b
            q = q.rename({cfg.KHeads.name: cfg.Heads.name})
            k = k.rename({cfg.KHeads.name: cfg.Heads.name})
            v_h = v.rename({cfg.VHeads.name: cfg.Heads.name})
            z_h = z.rename({cfg.VHeads.name: cfg.Heads.name})
            b_hparam = b.rename({cfg.VHeads.name: cfg.Heads.name})
            a_hparam = a.rename({cfg.VHeads.name: cfg.Heads.name})

        beta = hnn.sigmoid(b_hparam)
        a32 = a_hparam.astype(jnp.float32)
        dt_bias_na = self.dt_bias.astype(jnp.float32)
        A_exp = hax.exp(self.A_log.astype(jnp.float32))
        g = -(A_exp * hnn.softplus(a32 + dt_bias_na)).astype(x.dtype)  # log-decay on Heads

        # 4) Kernels expect [batch, position, heads, dim] (axis name "heads")
        q_h = q.rename({cfg.Heads.name: "heads"})
        k_h = k.rename({cfg.Heads.name: "heads"})
        v_kern = v_h.rename({cfg.Heads.name: "heads"})
        g_h = g.rename({cfg.Heads.name: "heads"})
        b_h = beta.rename({cfg.Heads.name: "heads"})

        q_bphd = hax.rearrange(q_h, ("batch", "position", "heads", cfg.KHeadDim.name))
        k_bphd = hax.rearrange(k_h, ("batch", "position", "heads", cfg.KHeadDim.name))
        v_bphd = hax.rearrange(v_kern, ("batch", "position", "heads", cfg.VHeadDim.name))
        _dbg("kernel/q_bphd", q_bphd.array)
        _dbg("kernel/k_bphd", k_bphd.array)
        _dbg("kernel/v_bphd", v_bphd.array)

        # Choose the kernel:
        if decode_state is not None and x.axis_size("position") == 1 and S_state is not None:
            out_bphd, S_new = recurrent_gated_delta_rule(
                q_bphd,
                k_bphd,
                v_bphd,
                g_h,
                b_h,
                initial_state=S_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_flash=self.use_flash,
            )
        else:
            out_bphd, S_new = chunk_gated_delta_rule(
                q_bphd,
                k_bphd,
                v_bphd,
                g_h,
                b_h,
                chunk_size=chunk_size,
                initial_state=None,
                output_final_state=inference,
                use_qk_l2norm_in_kernel=True,
                use_flash=self.use_flash,
            )

        # Keep the kernel output on "heads" so TP can shard the out-projection.
        out = out_bphd  # [B, Pos, heads, VHeadDim]
        _dbg("kernel/out_bphd", out.array)

        # 5) Gated RMSNorm with Z (rename Z to "heads" to match)
        z_gate = z_h.rename({cfg.Heads.name: "heads"})
        y_norm = self.o_norm(out, gate=z_gate)

        # 6) Output projection back to model dimension (In=(Heads, VHeadDim) -> Out=Embed)
        y_out = self.out_proj(y_norm.astype(x.dtype))
        _dbg("layer/y_out", y_out.array)

        # State packing for streaming
        new_state: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
        if inference and (new_conv_state is not None) and (S_new is not None):
            new_state = (new_conv_state, S_new)
        return y_out, new_state

    def to_state_dict(self) -> dict[str, jnp.ndarray]:
        return {
            "in_proj_qkvz.weight": jnp.array(self.in_proj_qkvz.weight.array),
            "in_proj_ba.weight": jnp.array(self.in_proj_ba.weight.array),
            "conv_weight": jnp.array(self.conv_weight.array),
            "A_log": jnp.array(self.A_log.array),
            "dt_bias": jnp.array(self.dt_bias.array),
            "o_norm.weight": jnp.array(self.o_norm.weight.array),
            "out_proj.weight": jnp.array(self.out_proj.weight.array),
        }

    def load_state_dict(self, state: dict[str, jnp.ndarray]) -> "GatedDeltaNet":
        cfg = self.config

        def _assign_linear_weight(named_linear: hnn.Linear, np_weight: jnp.ndarray, out_axis: Axis, in_axis: Axis):
            w_named = hax.named(jnp.asarray(np_weight, dtype=jnp.float32), (out_axis.name, in_axis.name))
            return dataclasses.replace(named_linear, weight=w_named)

        new_in_proj_qkvz = _assign_linear_weight(
            self.in_proj_qkvz, state["in_proj_qkvz.weight"], cfg.mix_qkvz_axis, cfg.Embed
        )
        new_in_proj_ba = _assign_linear_weight(self.in_proj_ba, state["in_proj_ba.weight"], cfg.ba_axis, cfg.Embed)

        # Rebuild named conv axes
        ConvChannels = Axis("channels", cfg.key_dim * 2 + cfg.value_dim)
        ConvKernel = Axis("conv_kernel", cfg.conv_kernel_size)
        new_conv_weight = hax.named(jnp.asarray(state["conv_weight"], dtype=jnp.float32), (ConvChannels, ConvKernel))

        # Heads-based params
        new_A_log = hax.named(jnp.asarray(state["A_log"], dtype=jnp.float32), (cfg.Heads.name,))
        new_dt_bias = hax.named(jnp.asarray(state["dt_bias"], dtype=jnp.float32), (cfg.Heads.name,))
        new_o_norm = dataclasses.replace(
            self.o_norm, weight=hax.named(jnp.asarray(state["o_norm.weight"], dtype=jnp.float32), (cfg.VHeadDim.name,))
        )

        # out_proj.weight is (Embed, Heads, VHeadDim)
        out_w = jnp.asarray(state["out_proj.weight"], dtype=jnp.float32)
        new_out_proj = dataclasses.replace(
            self.out_proj, weight=hax.named(out_w, (cfg.Embed.name, cfg.Heads.name, cfg.VHeadDim.name))
        )

        return dataclasses.replace(
            self,
            in_proj_qkvz=new_in_proj_qkvz,
            in_proj_ba=new_in_proj_ba,
            conv_weight=new_conv_weight,
            A_log=new_A_log,
            dt_bias=new_dt_bias,
            o_norm=new_o_norm,
            out_proj=new_out_proj,
        )

    @classmethod
    def from_state_dict(
        cls,
        config: GatedDeltaNetConfig,
        state: dict[str, jnp.ndarray],
        use_flash: bool = True,
        *,
        key,
    ) -> "GatedDeltaNet":
        layer = cls.init(config, key=key, use_flash=use_flash)
        return layer.load_state_dict(state)
