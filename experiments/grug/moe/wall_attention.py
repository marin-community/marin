# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Wall Attention (tilde-research/wall-attention-release), a TPU/JAX port.
#
# Softmax attention with a data-dependent, per-channel multiplicative decay folded into
# the QK inner product:
#
#     score_ij = scale * sum_k q_ik k_jk exp(P_ik - P_jk),  P_i = cumsum_{t<=i}(log_g_t)
#     logit_ij = score_ij + (C_i - C_j)        (optional FoX-style scalar gate, C = cumsum(log_g_scalar))
#     o_i = softmax_j(logit_ij) @ v            (with an attention-sink term in the denominator)
#
# with causal masking, an optional sliding window, and segment (document) masking. ``log_g = 0``
# recovers vanilla softmax attention. Replaces BOTH the sliding-window and full-causal attention
# in the Grug MoE baseline (short layers carry a window; long layers do not).
#
# The upstream repo ships a Triton (NVIDIA) kernel. The naive "rescale q,k by exp(+/-P)" form
# overflows over long sequences, so we use a chunked flash-style form with a per-query-block
# reference P_b: q' = q*exp(P_i - P_b) (exponent in [0, block-decay]) and k' = k*exp(P_b - P_j)
# (exponent <= 0; far-back keys underflow harmlessly to 0). This is exact and numerically stable.

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.numpy import einsum
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Float, PRNGKeyArray

_GATE_RANK: int = 16
_TAU: float = 16.0


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Array:
    return std * random.truncated_normal(key, -3, 3, shape)


def _rmsnorm_last(x: Array, eps: float = 1e-6) -> Array:
    """Non-parametric RMSNorm over the last (head) axis, in fp32."""
    x32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return x32 * jax.lax.rsqrt(var + eps)


def _batch_spec() -> P:
    return P(("data", "expert"))


def _wall_mask(i_idx: Array, j_idx: Array, window: int | None, seg_i: Array | None, seg_all: Array | None) -> Array:
    """Boolean allowed-mask [*, C, S]: causal, optional window, optional same-segment."""
    allowed = j_idx[None, :] <= i_idx[:, None]  # [C,S] causal
    if window is not None:
        allowed = allowed & ((i_idx[:, None] - j_idx[None, :]) < window)
    mask = allowed[None, None]  # [1,1,C,S]
    if seg_i is not None:
        same = seg_i[:, :, None] == seg_all[:, None, :]  # [B,C,S]
        mask = mask & same[:, None]  # [B,1,C,S]
    return mask


def wall_attention_chunk(
    q: Array,
    k: Array,
    v: Array,
    log_g: Array,
    chunk_size: int,
    scale: float,
    window: int | None = None,
    seg: Array | None = None,
    log_g_scalar: Array | None = None,
    sink: Array | None = None,
) -> Array:
    """Chunked flash-style wall attention. q,k,log_g [B,S,H,Dk]; v [B,S,H,Dv]. Returns [B,S,H,Dv].

    Numerically stable via per-query-block decay reference. Equivalent to the brute-force
    ``wall_attention_reference`` but O(S^2 d) compute with O(chunk*S) working memory.
    """
    b, s, h, _ = q.shape
    dv = v.shape[-1]
    c = chunk_size
    if s % c != 0:
        raise ValueError(f"sequence length {s} must be divisible by wall chunk_size {c}")
    nb = s // c

    cum_p = jnp.cumsum(log_g, axis=1)  # [B,S,H,Dk]
    cum_c = jnp.cumsum(log_g_scalar, axis=1) if log_g_scalar is not None else None  # [B,S,H]
    j_idx = jnp.arange(s)
    neg = jnp.finfo(jnp.float32).min

    def block(_, bi):
        r0 = bi * c
        qb = jax.lax.dynamic_slice_in_dim(q, r0, c, axis=1)  # [B,C,H,Dk]
        p_q = jax.lax.dynamic_slice_in_dim(cum_p, r0, c, axis=1)  # [B,C,H,Dk]
        p_b = p_q[:, -1]  # [B,H,Dk] block reference (last row)
        q_prime = qb * jnp.exp(p_q - p_b[:, None])  # exponent in [0, block-decay]
        # Valid keys (j <= block end) have exponent <= 0; future keys (j > block end) are causally
        # masked but their exponent is large-positive and would overflow exp() to +inf (then NaN in
        # the einsum) over long sequences. Clamp to <= 0: exact for valid keys, harmless for masked.
        k_prime = k * jnp.exp(jnp.minimum(p_b[:, None] - cum_p, 0.0))
        scores = einsum("bchd,bshd->bhcs", q_prime, k_prime) * scale  # [B,H,C,S]

        i_idx = r0 + jnp.arange(c)  # [C]
        if cum_c is not None:
            c_q = jax.lax.dynamic_slice_in_dim(cum_c, r0, c, axis=1)  # [B,C,H]
            scores = scores + c_q.transpose(0, 2, 1)[..., None] - cum_c.transpose(0, 2, 1)[:, :, None, :]

        seg_i = jax.lax.dynamic_slice_in_dim(seg, r0, c, axis=1) if seg is not None else None
        mask = _wall_mask(i_idx, j_idx, window, seg_i, seg)
        scores = jnp.where(mask, scores, neg)

        m_row = jnp.max(scores, axis=-1, keepdims=True)  # [B,H,C,1]
        if sink is not None:
            m_row = jnp.maximum(m_row, sink[None, :, None, None])
        p_w = jnp.exp(scores - m_row)
        den = jnp.sum(p_w, axis=-1, keepdims=True)
        if sink is not None:
            den = den + jnp.exp(sink[None, :, None, None] - m_row)
        w = p_w / den
        o_block = einsum("bhcs,bshd->bchd", w, v)  # [B,C,H,Dv]
        return None, o_block

    _, o_blocks = jax.lax.scan(block, None, jnp.arange(nb))  # [nb,B,C,H,Dv]
    return jnp.moveaxis(o_blocks, 0, 1).reshape(b, s, h, dv)


def wall_attention_reference(
    q: Array,
    k: Array,
    v: Array,
    log_g: Array,
    scale: float,
    window: int | None = None,
    seg: Array | None = None,
    log_g_scalar: Array | None = None,
    sink: Array | None = None,
) -> Array:
    """Brute-force wall attention (materializes the [B,S,S,H,Dk] decay; tests only, small S)."""
    s = q.shape[1]
    cum_p = jnp.cumsum(log_g, axis=1)  # [B,S,H,Dk]
    diff = cum_p[:, :, None] - cum_p[:, None]  # [B,S_i,S_j,H,Dk]
    scores = einsum("bihd,bjhd,bijhd->bhij", q, k, jnp.exp(diff)) * scale  # [B,H,S,S]
    if log_g_scalar is not None:
        cum_c = jnp.cumsum(log_g_scalar, axis=1).transpose(0, 2, 1)  # [B,H,S]
        scores = scores + cum_c[:, :, :, None] - cum_c[:, :, None, :]
    i_idx = jnp.arange(s)
    mask = _wall_mask(i_idx, i_idx, window, seg, seg if seg is not None else None)
    scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
    m_row = jnp.max(scores, axis=-1, keepdims=True)
    if sink is not None:
        m_row = jnp.maximum(m_row, sink[None, :, None, None])
    p_w = jnp.exp(scores - m_row)
    den = jnp.sum(p_w, axis=-1, keepdims=True)
    if sink is not None:
        den = den + jnp.exp(sink[None, :, None, None] - m_row)
    w = p_w / den
    return einsum("bhij,bjhd->bihd", w, v)


class WallAttention(eqx.Module):
    """Wall-attention sequence mixer, drop-in for ``CausalSelfAttention``.

    GQA layout matching the baseline (``num_heads`` query heads, ``num_kv_heads`` KV heads,
    ``head_dim``) so params match; a low-rank per-channel decay gate (per KV head, GQA-repeated),
    a learnable per-head attention sink, an optional FoX scalar gate, and QK RMSNorm. ``window``
    is set per layer (sliding window on short layers, ``None`` = full causal on long layers).
    """

    w_q: Float[Array, "D Nq_Dk"]
    w_k: Float[Array, "D Nkv_Dk"]
    w_v: Float[Array, "D Nkv_Dk"]
    w_o: Float[Array, "Nq_Dk D"]
    w_g1: Float[Array, "D R"]
    w_g2: Float[Array, "R Nkv_Dk"]
    w_gs: Array | None
    sink: Array | None
    num_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    window: int | None = eqx.field(static=True)
    use_scalar_gate: bool = eqx.field(static=True)
    use_sink: bool = eqx.field(static=True)
    tau: float = eqx.field(static=True)

    @staticmethod
    def init(
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        initializer_std: float,
        chunk_size: int = 64,
        window: int | None = None,
        use_scalar_gate: bool = False,
        use_sink: bool = True,
        *,
        key: PRNGKeyArray,
    ) -> "WallAttention":
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}")
        d = hidden_dim
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        kq, kk, kv, ko, kg1, kg2, kgs = random.split(key, 7)
        return WallAttention(
            w_q=reshard(_init_weight(kq, (d, q_dim), initializer_std), P("data", "model")),
            w_k=reshard(_init_weight(kk, (d, kv_dim), initializer_std), P("data", "model")),
            w_v=reshard(_init_weight(kv, (d, kv_dim), initializer_std), P("data", "model")),
            w_o=reshard(_init_weight(ko, (q_dim, d), initializer_std), P("model", "data")),
            w_g1=reshard(_init_weight(kg1, (d, _GATE_RANK), initializer_std), P("data", None)),
            w_g2=reshard(_init_weight(kg2, (_GATE_RANK, kv_dim), initializer_std), P(None, "model")),
            w_gs=(
                reshard(_init_weight(kgs, (d, num_heads), initializer_std), P("data", None)) if use_scalar_gate else None
            ),
            sink=reshard(jnp.zeros((num_heads,)), P(None)) if use_sink else None,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            chunk_size=chunk_size,
            window=window,
            use_scalar_gate=use_scalar_gate,
            use_sink=use_sink,
            tau=float(_TAU),
        )

    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask=None,  # AttentionMask; only segment_ids used (window/causal handled internally)
        use_pko: bool = False,  # unused
    ) -> Float[Array, "B S D"]:
        del use_pko
        seg = mask.segment_ids[0] if getattr(mask, "segment_ids", None) is not None else None
        n, m, hd = self.num_heads, self.num_kv_heads, self.head_dim
        b, s, _ = x.shape

        q = einsum("bsd,dk->bsk", x, self.w_q).reshape(b, s, n, hd).astype(jnp.float32)
        k = einsum("bsd,dk->bsk", x, self.w_k).reshape(b, s, m, hd).astype(jnp.float32)
        v = einsum("bsd,dk->bsk", x, self.w_v).reshape(b, s, m, hd).astype(jnp.float32)
        q = _rmsnorm_last(q)
        k = _rmsnorm_last(k)

        # Per-channel decay gate (low-rank, per KV head): log_g = logsigmoid(z)/tau <= 0.
        zg = einsum("bsr,rk->bsk", einsum("bsd,dr->bsr", x, self.w_g1), self.w_g2)
        log_g = (-jax.nn.softplus(-zg.astype(jnp.float32)) / self.tau).reshape(b, s, m, hd)

        if m != n:  # GQA: repeat KV/gate heads across each query group
            rep = n // m
            k = jnp.repeat(k, rep, axis=2)
            v = jnp.repeat(v, rep, axis=2)
            log_g = jnp.repeat(log_g, rep, axis=2)

        log_g_scalar = None
        if self.use_scalar_gate:
            zs = einsum("bsd,dn->bsn", x, self.w_gs).astype(jnp.float32)
            log_g_scalar = -jax.nn.softplus(-zs) / self.tau  # [B,S,Nq]

        seq_spec = P(("data", "expert"), None, None, None)
        q = reshard(q, seq_spec)
        k = reshard(k, seq_spec)
        v = reshard(v, seq_spec)
        log_g = reshard(log_g, seq_spec)
        if log_g_scalar is not None:
            log_g_scalar = reshard(log_g_scalar, P(("data", "expert"), None, None))
        if seg is not None:
            seg = reshard(seg, P(("data", "expert"), None))

        o = wall_attention_chunk(
            q, k, v, log_g, self.chunk_size, hd**-0.5, self.window, seg, log_g_scalar, self.sink
        )  # [B,S,Nq,hd]
        o = o.reshape(b, s, n * hd).astype(x.dtype)
        return einsum("bsk,kd->bsd", o, self.w_o, out_sharding=_batch_spec())


__all__ = ["WallAttention", "wall_attention_chunk", "wall_attention_reference"]
