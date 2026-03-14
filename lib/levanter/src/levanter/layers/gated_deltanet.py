# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# based on:
# - https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py
# - the JAX implementation by Yu Sun and Leo Lee

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
from typing import Optional, Tuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray


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
    N, C, L = x_ncl.shape
    K = w_ck.shape[-1]
    # pad x on the left with K-1 zeros so that output length == L ("causal")
    x_pad = jnp.pad(x_ncl, ((0, 0), (0, 0), (K - 1, 0)))
    w_oik = w_ck[:, None, :]  # (C, 1, K) → O=C, I=1, K
    y = lax.conv_general_dilated(
        lhs=x_pad,
        rhs=w_oik,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NCH", "OIH", "NCH"),
        feature_group_count=C,  # depthwise
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    if bias_c is not None:
        y = y + bias_c[:, None]
    y = jax.nn.silu(y)
    return y


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
    x_hist = jnp.concatenate([prev_state_nck, x_ncl_1], axis=-1)  # (N, C, K+1)
    y2 = lax.conv_general_dilated(
        lhs=x_hist,
        rhs=w_ck[:, None, :],
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NCH", "OIH", "NCH"),
        feature_group_count=x_hist.shape[1],
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    y = y2[..., -1:]  # (N, C, 1): the newest output sample
    if bias_c is not None:
        y = y + bias_c[:, None]
    y = jax.nn.silu(y)
    new_state = jnp.concatenate([prev_state_nck[..., 1:], x_ncl_1], axis=-1)
    return y, new_state


# ---------- Gated RMSNorm with external gate ----------


class GatedRmsNorm(eqx.Module):
    """RMSNorm(x) * SiLU(gate)"""

    axis: Axis
    weight: NamedArray  # [axis]
    eps: float = eqx.field(default=1e-6, static=True)

    @staticmethod
    def init(axis: Axis, eps: float = 1e-6) -> "GatedRmsNorm":
        return GatedRmsNorm(axis=axis, weight=hax.ones(axis), eps=eps)

    def __call__(self, x: NamedArray, gate: NamedArray) -> NamedArray:
        in_dtype = x.dtype
        x32 = x.astype(jnp.float32)
        var = hax.mean(hax.square(x32), axis=self.axis)
        inv = hax.rsqrt(var + jnp.asarray(self.eps, dtype=jnp.float32))
        y = (x32 * inv).astype(in_dtype)  # RMSNorm (from haliax/nn/normalization.py)
        y = self.weight * y  # learned scale
        gated = y * hnn.silu(gate)  # GDN's output gate
        return gated.astype(in_dtype)


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


# ---------- Triangular masks ----------


def _tri_upper_eq_mask(Ci: Axis, Cj: Axis) -> NamedArray:
    """Mask for i <= j (upper-triangular incl. diagonal) in (Ci, Cj) coordinates.

    Used to zero-out invalid contributions when building strictly lower-triangular
    in-chunk operators for the UT forward substitution.
    """
    ii = hax.arange(Ci)
    jj = hax.arange(Cj)
    I = ii.broadcast_axis(Cj)
    J = jj.broadcast_axis(Ci)
    return I <= J


def _diag_mask(Ci: Axis, Cj: Axis) -> NamedArray:
    ii = hax.arange(Ci)
    jj = hax.arange(Cj)
    I = ii.broadcast_axis(Cj)
    J = jj.broadcast_axis(Ci)
    return cast(NamedArray, I == J)


# ---------- Kernels ----------


def recurrent_gated_delta_rule(
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

    if output_final_state:
        return out_final, S_final
    else:
        return out_final, None


def chunk_gated_delta_rule(
    query: NamedArray,  # [batch, position, heads, k_head_dim]
    key: NamedArray,  # [batch, position, heads, k_head_dim]
    value: NamedArray,  # [batch, position, heads, v_head_dim]
    g: NamedArray,  # [batch, position, heads]  (log-decay; α=exp(g))
    beta: NamedArray,  # [batch, position, heads]  (β)
    *,
    chunk_size: int = 64,
    initial_state: Optional[jnp.ndarray] = None,  # (B,H,dk,dv)
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[NamedArray, Optional[jnp.ndarray]]:
    """Chunkwise-parallel GDN (DeltaNet UT/WY extended with decay).

    High-level sketch (per head):
      1) Split the length-L sequence into Nc = ceil(L/C) chunks of size C.
      2) Inside each chunk, form a strictly lower-triangular operator encoding
         the rank-1 updates and the *relative decays* between positions.
      3) Compute T = (I - A)^{-1} via *forward substitution*.
      4) Obtain "pseudo values" U = T (β V) and a decayed key summary K̂ = T (β K ⊙ exp(g)).
      5) Bridge chunks with the cross-chunk state S (decayed carry and innovation).
      6) Produce outputs by combining inter-chunk (from S) and intra-chunk terms.
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
    gg = g.astype(jnp.float32)
    b = beta.astype(jnp.float32)

    if use_qk_l2norm_in_kernel:
        q = _l2norm(q, axis=Dk)
        k = _l2norm(k, axis=Dk)
    q = q * (Dk.size**-0.5)

    # ---- pad to multiple of chunk_size ----
    L = Pos.size
    pad = (chunk_size - (L % chunk_size)) % chunk_size
    if pad > 0:
        q = hax.pad(q, {Pos: (0, pad)})
        k = hax.pad(k, {Pos: (0, pad)})
        v = hax.pad(v, {Pos: (0, pad)})
        b = hax.pad(b, {Pos: (0, pad)})
        gg = hax.pad(gg, {Pos: (0, pad)})

    PosPad = q.resolve_axis("position")
    Lt = PosPad.size
    Nc = Lt // chunk_size
    Chunks = Axis("chunks", Nc)
    C = Axis("chunk", chunk_size)

    # Helper to reshape [B, Lpad, H, d] → [B, Nc, C, H, d]
    def _chunk(x: NamedArray) -> NamedArray:
        return x.unflatten_axis(PosPad, (Chunks, C))

    q_c = _chunk(q)
    k_c = _chunk(k)
    v_c = _chunk(v)
    b_c = _chunk(b)
    g_c = _chunk(gg)

    v_beta = v_c * b_c.broadcast_axis(Dv)  # βV per position
    k_beta = k_c * b_c.broadcast_axis(Dk)  # βK per position

    # cumulative g in chunk (for relative decays)
    g_cum = hax.cumsum(g_c, axis=C)  # [B, Nc, C, H]

    # --- Build strictly lower-triangular A in (Ci, Cj) coordinates ---
    Ci = Axis("Ci", C.size)
    Cj = Axis("Cj", C.size)

    kb_ci = k_beta.rename({C.name: Ci.name})  # [B,Nc,Ci,H,Dk]
    k_cj = k_c.rename({C.name: Cj.name})  # [B,Nc,Cj,H,Dk]

    # Raw interactions scaled by β: -(βK) @ K^T  (per head)
    A_raw = -hax.dot(kb_ci, k_cj, axis=Dk)  # [B,Nc,Ci,Cj,H]

    # Relative decay between positions i and j inside the chunk:
    #   exp( g_cum[i] - g_cum[j] )  for i >= j, else 0
    gi = g_cum.rename({C.name: Ci.name})
    gj = g_cum.rename({C.name: Cj.name})
    diff = gi.broadcast_axis(Cj) - gj.broadcast_axis(Ci)
    # Avoid overflow/NaNs in the strict upper triangle by setting exp argument to -inf
    neg_inf = jnp.asarray(-jnp.inf, dtype=diff.dtype)
    diff = hax.where(_diag_mask(Ci, Cj), jnp.asarray(0.0, dtype=diff.dtype), diff)
    diff = hax.where(_tri_upper_eq_mask(Ci, Cj), neg_inf, diff)
    decay = hax.exp(diff)  # [B,Nc,Ci,Cj,H]

    # Zero out diagonal and strict upper triangle
    A = A_raw * decay
    A = hax.where(_tri_upper_eq_mask(Ci, Cj), jnp.asarray(0.0, dtype=A.dtype), A)

    # --- Forward substitution (UT transform) to get T = (I - A)^{-1} ---
    A_bhcc = hax.rearrange(A, (Batch, Heads, Chunks, Ci, Cj)).array
    eyeC = jnp.eye(C.size, dtype=A_bhcc.dtype)

    def body(i, attn):
        """Perform y[i] ← y[i] + sum_{j<i} y[i,j] * y[j,:]  (forward-subst)

        This loop computes the implicit lower-triangular transform so that
        'attn + I' acts like T above
        """
        row_i = lax.dynamic_slice_in_dim(attn, i, 1, axis=-2)  # (...,1,C)
        row_i = jnp.squeeze(row_i, axis=-2)  # (...,C)

        # Masks for the strict lower sub-block up to row i
        ar = jnp.arange(C.size, dtype=attn.dtype)
        m1 = (ar < i).astype(attn.dtype)  # vector mask
        m2 = ((ar[:, None] < i) & (ar[None, :] < i)).astype(attn.dtype)  # matrix mask

        row_pref = row_i * m1
        sub_pref = attn * m2
        incr = jnp.sum(row_pref[..., None] * sub_pref, axis=-2)
        new_row = jnp.expand_dims(row_i + incr, axis=-2)

        return lax.dynamic_update_slice_in_dim(attn, new_row, i, axis=-2)

    attn_low = lax.fori_loop(1, C.size, body, A_bhcc)
    T = attn_low + eyeC  # lower-triangular with ones on diagonal; acts like (I - A)^-1

    # --- Pseudo values and decayed key summaries (intra-chunk) ---
    # v_pseudo = T @ (β V)
    vbeta_bhccd = hax.rearrange(v_beta.rename({C.name: Cj.name}), (Batch, Heads, Chunks, Cj, Dv)).array
    v_pseudo = jnp.einsum("bhnij,bhnjd->bhnid", T, vbeta_bhccd)  # (B,H,Nc,C,Dv)

    # k_cumdecay = T @ (β K ⊙ exp(g_cum))
    kbeta_bhccd = hax.rearrange(k_beta.rename({C.name: Cj.name}), (Batch, Heads, Chunks, Cj, Dk)).array
    exp_g_bhcc = hax.rearrange(hax.exp(g_cum).rename({C.name: Cj.name}), (Batch, Heads, Chunks, Cj)).array
    k_cumdecay = jnp.einsum("bhnij,bhnjd->bhnid", T, kbeta_bhccd * exp_g_bhcc[..., None])  # (B,H,Nc,C,d_k)

    # --- Scan over chunks: bridge with cross-chunk S ---
    q_bhccd = hax.rearrange(q_c, (Batch, Heads, Chunks, C, Dk)).array
    k_bhccd = hax.rearrange(k_c, (Batch, Heads, Chunks, C, Dk)).array
    g_bhcc = hax.rearrange(g_cum, (Batch, Heads, Chunks, C)).array

    B_, H_, dk_, dv_ = Batch.size, Heads.size, Dk.size, Dv.size
    S = jnp.zeros((B_, H_, dk_, dv_), dtype=v.dtype) if initial_state is None else initial_state.astype(v.dtype)

    # Strict upper mask (i<j) to zero invalid future positions within a chunk
    mask_strict_upper = jnp.triu(jnp.ones((C.size, C.size), dtype=bool), k=1)

    def chunk_step(S_prev, inps):
        """Process one chunk i with in-chunk triangular ops + cross-chunk state S."""
        q_i, k_i, v_i, gcum_i, kcum_i = inps  # shapes: (B,H,C,dk/dv)
        # In-chunk relative decay mask for attention-like term with q
        diff = gcum_i[..., None] - gcum_i[..., None, :]  # (B,H,C,C)
        decay_i = jnp.exp(jnp.tril(diff))
        attn_i = jnp.einsum("bhid,bhjd->bhij", q_i, k_i) * decay_i
        attn_i = jnp.where(mask_strict_upper, 0.0, attn_i)  # strictly lower

        # Contribution predicted by previous cross-chunk state (remove it)
        v_prime = jnp.einsum("bhid,bhdm->bhim", kcum_i, S_prev)  # (B,H,C,dv)
        v_new = v_i - v_prime  # "innovation" within the chunk

        # Output: inter-chunk term (from decayed S) + in-chunk triangular mix
        qexp = q_i * jnp.exp(gcum_i)[..., None]
        inter = jnp.einsum("bhid,bhdm->bhim", qexp, S_prev)
        out_i = inter + jnp.einsum("bhij,bhjm->bhim", attn_i, v_new)

        # Update cross-chunk state S with the *tail* decay and innovations
        g_tail = gcum_i[..., -1]  # last position's cumulative g
        decay_tail = jnp.exp(g_tail)[..., None, None]  # α at the chunk tail
        decay_weights = jnp.exp((g_tail[..., None] - gcum_i))[..., None]  # exp(g_tail - g_pos)

        add = jnp.einsum("bhid,bhim->bhdm", k_i * decay_weights, v_new)
        S_new = S_prev * decay_tail + add
        return S_new, out_i

    S, out_chunks = jax.lax.scan(
        chunk_step,
        S,
        (
            jnp.moveaxis(q_bhccd, 2, 0),  # time-major over chunks
            jnp.moveaxis(k_bhccd, 2, 0),
            jnp.moveaxis(v_pseudo, 2, 0),
            jnp.moveaxis(g_bhcc, 2, 0),
            jnp.moveaxis(k_cumdecay, 2, 0),
        ),
        length=Nc,
    )

    # Back to [B, Pos, H, Dv], trimming padding if any
    out_bhcd = jnp.moveaxis(out_chunks, 0, 2)  # (B,H,Nc,C,Dv)
    out_bhcd = hax.named(out_bhcd, (Batch, Heads, Chunks, C, Dv))
    out_flat_bhPd = out_bhcd.flatten_axes((Chunks, C), PosPad)
    out_bhLd = out_flat_bhPd["position", hax.ds(0, L)]
    out_final = hax.rearrange(out_bhLd, (Batch, PosPad.name, Heads, Dv))

    return (out_final, S) if output_final_state else (out_final, None)


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

    # projections
    in_proj_qkvz: hnn.Linear  # [Embed] -> [Q|K|V|Z]
    in_proj_ba: hnn.Linear  # [Embed] -> [b|a]

    # depthwise conv parameters over concatenated [Q|K|V] channels
    conv_weight: jnp.ndarray
    conv_bias: Optional[jnp.ndarray]

    # discretization params per V head (Mamba2-style)
    A_log: jnp.ndarray
    dt_bias: jnp.ndarray

    # gated RMSNorm and output projection
    o_norm: GatedRmsNorm
    out_proj: hnn.Linear  # [VHeads, VHeadDim] -> [Embed]

    @staticmethod
    def init(config: GatedDeltaNetConfig, *, key) -> "GatedDeltaNet":
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
        conv_weight = jax.random.normal(k_conv, (C, K), dtype=jnp.float32) * (1.0 / jnp.sqrt(C * K))
        conv_bias = None

        # GDN discretization parameters (per V-head)
        A_log = jnp.log(jax.random.uniform(k_out, (config.num_v_heads,), minval=1e-6, maxval=16.0, dtype=jnp.float32))
        dt_bias = jnp.ones((config.num_v_heads,), dtype=jnp.float32)

        o_norm = GatedRmsNorm.init(config.VHeadDim, eps=config.rms_norm_eps)
        out_proj = hnn.Linear.init(
            In=(config.VHeads, config.VHeadDim), Out=config.Embed, out_first=True, use_bias=False, key=k_out
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
        )

    def _fix_qkvz_ordering(
        self,
        mixed_qkvz: NamedArray,  # [B, Pos, qkvz]
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

        # 1) Project to [Q|K|V|Z] and [b|a]
        mixed_qkvz = self.in_proj_qkvz(x)  # [B, Pos, qkvz=2*key_dim + 2*value_dim]
        mixed_ba = self.in_proj_ba(x)  # [B, Pos, ba=2*num_v_heads]

        # 1b) Re-group like HF for parity (also used for conv channel ordering)
        q, k, v, z, b, a = self._fix_qkvz_ordering(mixed_qkvz, mixed_ba)

        # 2) Depthwise causal conv over concatenated [Q|K|V] channels
        #    HF orders channels as: [Q_flat | K_flat | V_flat] (no Z).
        q_ch = q.flatten_axes((cfg.KHeads, cfg.KHeadDim), Axis("channels", cfg.key_dim))
        k_ch = k.flatten_axes((cfg.KHeads, cfg.KHeadDim), Axis("channels", cfg.key_dim))
        v_ch = v.flatten_axes((cfg.VHeads, cfg.VHeadDim), Axis("channels", cfg.value_dim))
        qkv_ch = hax.concatenate("channels", [q_ch, k_ch, v_ch])  # [B, Pos, channels]
        qkv_ncl = hax.rearrange(qkv_ch, ("batch", "channels", "position")).array  # (N, C, L)

        S_state: Optional[jnp.ndarray] = None
        if decode_state is not None and x.axis_size("position") == 1:
            # Streaming decode: cheap single-step conv update + carry conv_state
            conv_state, S_state = decode_state
            K = self.conv_weight.shape[-1]
            assert conv_state.shape[-1] == K
            y_ncl, new_conv_state = _causal_depthwise_conv1d_update(
                qkv_ncl, self.conv_weight, self.conv_bias, conv_state
            )
        else:
            # Prefill/train: full causal conv over the sequence
            y_ncl = _causal_depthwise_conv1d_full(qkv_ncl, self.conv_weight, self.conv_bias)
            if inference:
                # cache the rightmost K samples of channels as the next conv_state
                K = self.conv_weight.shape[-1]
                L = x.axis_size("position")
                if L >= K:
                    new_conv_state = qkv_ncl[..., -K:]
                else:
                    new_conv_state = jnp.pad(qkv_ncl, ((0, 0), (0, 0), (K - L, 0)))
            else:
                new_conv_state = None
                S_state = None

        # Unpack [Q|K|V] after conv back to per-head tensors (mirror the same channel order)
        y_bpc = hax.rearrange(hax.named(y_ncl, ("batch", "channels", "position")), ("batch", "position", "channels"))
        q_y = y_bpc["channels", hax.ds(0, cfg.key_dim)]
        k_y = y_bpc["channels", hax.ds(cfg.key_dim, cfg.key_dim)]
        v_y = y_bpc["channels", hax.ds(2 * cfg.key_dim, cfg.value_dim)]
        q = q_y.unflatten_axis("channels", (cfg.KHeads, cfg.KHeadDim))
        k = k_y.unflatten_axis("channels", (cfg.KHeads, cfg.KHeadDim))
        v = v_y.unflatten_axis("channels", (cfg.VHeads, cfg.VHeadDim))

        # 3) Gates: β via sigmoid(b); α via g = -exp(A) * softplus(a + dt_bias), α=exp(g)
        beta = hnn.sigmoid(b)
        a32 = a.astype(jnp.float32)
        dt_bias_na = hax.named(jnp.asarray(self.dt_bias, dtype=jnp.float32), cfg.VHeads)
        A_exp = hax.exp(hax.named(jnp.asarray(self.A_log, dtype=jnp.float32), cfg.VHeads))
        g = -(A_exp * hnn.softplus(a32 + dt_bias_na)).astype(x.dtype)  # log-decay

        # If we have more V-heads than K-heads, repeat Q,K across V-groups so
        # every V-head has its own (q,k) and thus its own rectangular S.
        ratio = cfg.num_v_heads // cfg.num_k_heads
        if ratio > 1:
            VGroup = Axis("v_group", ratio)
            q = q.broadcast_axis(VGroup).flatten_axes((cfg.KHeads, VGroup), cfg.VHeads)
            k = k.broadcast_axis(VGroup).flatten_axes((cfg.KHeads, VGroup), cfg.VHeads)
        else:
            q = q.rename({cfg.KHeads.name: cfg.VHeads.name})
            k = k.rename({cfg.KHeads.name: cfg.VHeads.name})

        # 4) Kernels expect [batch, position, heads, dim] with axis name "heads"
        q_h = q.rename({cfg.VHeads.name: "heads"})
        k_h = k.rename({cfg.VHeads.name: "heads"})
        v_h = v.rename({cfg.VHeads.name: "heads"})
        g_h = g.rename({cfg.VHeads.name: "heads"})
        b_h = beta.rename({cfg.VHeads.name: "heads"})

        q_bphd = hax.rearrange(q_h, ("batch", "position", "heads", cfg.KHeadDim.name))
        k_bphd = hax.rearrange(k_h, ("batch", "position", "heads", cfg.KHeadDim.name))
        v_bphd = hax.rearrange(v_h, ("batch", "position", "heads", cfg.VHeadDim.name))

        # Choose the kernel:
        #  - decode (1 token, with state): recurrent rule
        #  - else: chunkwise-parallel rule
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
            )
        else:
            out_bphd, S_new = chunk_gated_delta_rule(
                q_bphd,
                k_bphd,
                v_bphd,
                g_h,
                b_h,
                chunk_size=chunk_size,
                initial_state=None,  # could feed S_state if continuing
                output_final_state=inference,  # return S_T for caching if inference
                use_qk_l2norm_in_kernel=True,
            )

        # Back to [B, Pos, VHeads, VHeadDim]
        out = out_bphd.rename({"heads": cfg.VHeads.name})

        # 5) Gated RMSNorm with Z (GDN block’s output gate)
        y_norm = self.o_norm(out, gate=z)

        # 6) Output projection back to model dimension
        y_out = self.out_proj(y_norm.astype(x.dtype))

        # State packing for streaming
        new_state: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
        if inference and (new_conv_state is not None) and (S_new is not None):
            new_state = (new_conv_state, S_new)
        return y_out, new_state

    def to_state_dict(self) -> dict[str, jnp.ndarray]:
        return {
            "in_proj_qkvz.weight": jnp.array(self.in_proj_qkvz.weight.array),
            "in_proj_ba.weight": jnp.array(self.in_proj_ba.weight.array),
            "conv_weight": jnp.array(self.conv_weight),
            "A_log": jnp.array(self.A_log),
            "dt_bias": jnp.array(self.dt_bias),
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
        new_conv_weight = jnp.asarray(state["conv_weight"], dtype=jnp.float32)
        new_A_log = jnp.asarray(state["A_log"], dtype=jnp.float32)
        new_dt_bias = jnp.asarray(state["dt_bias"], dtype=jnp.float32)
        new_o_norm = dataclasses.replace(
            self.o_norm, weight=hax.named(jnp.asarray(state["o_norm.weight"], dtype=jnp.float32), (cfg.VHeadDim.name,))
        )
        out_w = jnp.asarray(state["out_proj.weight"], dtype=jnp.float32)  # (embed, v_heads, v_head_dim)
        new_out_proj = dataclasses.replace(
            self.out_proj, weight=hax.named(out_w, (cfg.Embed.name, cfg.VHeads.name, cfg.VHeadDim.name))
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
    def from_state_dict(cls, config: GatedDeltaNetConfig, state: dict[str, jnp.ndarray], *, key) -> "GatedDeltaNet":
        """
        Build a fresh layer from config + state dict.
        """
        layer = cls.init(config, key=key)
        return layer.load_state_dict(state)
