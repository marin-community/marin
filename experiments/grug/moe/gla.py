# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Gated Linear Attention (GLA, arXiv 2312.06635), as a drop-in sequence mixer to
# replace sliding-window attention in the Grug MoE model.
#
# Per-head recurrence (S in R^{d'_k x d'_v}):
#     alpha_t = sigmoid(x_t W_a1 W_a2 + b_a) ** (1 / tau)   (data-dependent forget gate, in (0,1)^{d_k})
#     S_t = Diag(alpha_t) S_{t-1} + k_t^T v_t
#     o_t = q_t S_t
# with separate key/value dims d_k = d/2, d_v = d, multi-head, per-head RMSNorm on
# the head output, a Swish output gate r_t = Swish(x_t W_r), and output projection W_O.
#
# Training uses the chunkwise-parallel form (Sec. 3.3 of the paper): within each
# chunk we use the log-space parallel form (intra-chunk), and across chunks we run
# a recurrence on the chunk-summarized state (inter-chunk), giving the exact same
# result as the step recurrence but with matmuls. The recurrent form is provided
# as a reference for numerical testing. All gate/decay math is done in fp32.

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


def gla_recurrent(q: Array, k: Array, v: Array, log_alpha: Array, seg: Array | None = None) -> Array:
    """Reference GLA step recurrence (slow; for tests). All inputs [B, S, H, *], fp32.

    q, k, log_alpha: [B, S, H, Dk]; v: [B, S, H, Dv]. Returns o: [B, S, H, Dv].
    ``seg`` is an optional [B, S] integer segment id; when given, the recurrent state
    is reset to zero at each document boundary (``seg`` changes), so a token never
    attends across a document boundary.
    """
    b, _, h, dk = q.shape
    dv = v.shape[-1]

    def step(carry, xs):
        state, seg_prev = carry  # state [B,H,Dk,Dv], seg_prev [B]
        q_t, k_t, v_t, la_t, seg_t = xs  # [B,H,Dk]*3-ish, seg_t [B]
        if seg is not None:
            same = (seg_t == seg_prev).astype(state.dtype)[:, None, None, None]
            state = same * state
        state = jnp.exp(la_t)[..., None] * state + einsum("bhd,bhe->bhde", k_t, v_t)
        o_t = einsum("bhd,bhde->bhe", q_t, state)
        return (state, seg_t), o_t

    state0 = jnp.zeros((b, h, dk, dv), jnp.float32)
    seg0 = jnp.full((b,), -1, dtype=jnp.int32) if seg is None else jnp.moveaxis(seg, 1, 0)[0] - 1
    seg_seq = jnp.zeros((q.shape[1], b), jnp.int32) if seg is None else jnp.moveaxis(seg, 1, 0)
    xs = (jnp.moveaxis(q, 1, 0), jnp.moveaxis(k, 1, 0), jnp.moveaxis(v, 1, 0), jnp.moveaxis(log_alpha, 1, 0), seg_seq)
    _, out = jax.lax.scan(step, (state0, seg0), xs)
    return jnp.moveaxis(out, 0, 1)


def gla_chunk(q: Array, k: Array, v: Array, log_alpha: Array, chunk_size: int, seg: Array | None = None) -> Array:
    """Chunkwise-parallel GLA forward. Inputs [B, S, H, *] fp32; returns o [B, S, H, Dv].

    Equivalent to ``gla_recurrent`` but computed with matmuls over chunks of size C.
    ``seg`` is an optional [B, S] integer segment id; when given, document boundaries
    reset the recurrence so no token attends across a boundary (matches the segment
    masking applied to the softmax-attention layers). Documents are contiguous, so
    within a chunk the *live* document is the one containing the chunk's last token.
    """
    b, s, h, dk = q.shape
    dv = v.shape[-1]
    c = chunk_size
    if s % c != 0:
        raise ValueError(f"sequence length {s} must be divisible by gla chunk_size {c}")
    n = s // c

    def chunked(t, d):
        return t.reshape(b, n, c, h, d)

    qc, kc, vc, lac = chunked(q, dk), chunked(k, dk), chunked(v, dv), chunked(log_alpha, dk)

    # Inclusive within-chunk cumulative log-decay: Lam_j = log(b_{chunk_start+j} / b_{chunk_start}).
    lam = jnp.cumsum(lac, axis=2)  # [B,N,C,H,Dk]
    gamma = lam[:, :, -1]  # [B,N,H,Dk] total chunk log-decay
    exp_lam = jnp.exp(lam)
    exp_neg_lam = jnp.exp(-lam)

    # Segment structure (all derived from the [B,N,C] chunked segment ids).
    seg_eq = None  # [B,N,C,C] same-doc mask over (t, s) intra-chunk pairs
    seg_keep = None  # [B,N,C] tokens of the chunk's live (last) document, propagated forward
    carry_keep = None  # [B,N] whether the live doc spans into the previous chunk (else drop carry)
    inter_keep = None  # [B,N,C] whether token t's doc extends into a previous chunk
    if seg is not None:
        seg_c = seg.reshape(b, n, c)  # [B,N,C]
        seg_eq = seg_c[:, :, :, None] == seg_c[:, :, None, :]  # [B,N,C,C] (t == s)
        g_last = seg_c[:, :, -1]  # [B,N] live-doc id per chunk
        seg_keep = (seg_c == g_last[:, :, None]).astype(q.dtype)  # [B,N,C]
        # Previous chunk's last-token id; sentinel for chunk 0 (its state is zero anyway).
        # Shift-and-fill rather than concatenate to preserve the batch sharding of ``g_last``.
        g_prev_last = jnp.roll(g_last, 1, axis=1).at[:, 0].set(jnp.asarray(-1, seg_c.dtype))  # [B,N]
        carry_keep = (g_last == g_prev_last).astype(q.dtype)  # [B,N]
        inter_keep = (seg_c == g_prev_last[:, :, None]).astype(q.dtype)  # [B,N,C]

    # Intra-chunk (causal, incl. diagonal): P[t,s] = sum_k q_tk k_sk exp(Lam_tk - Lam_sk), t>=s.
    q_d = qc * exp_lam
    k_d = kc * exp_neg_lam
    scores = einsum("bnthd,bnshd->bnhts", q_d, k_d)  # [B,N,H,C,C], contract Dk
    idx = jnp.arange(c)
    causal = idx[:, None] >= idx[None, :]  # [C,C] (t >= s)
    if seg_eq is not None:
        mask = causal[None, None] & seg_eq  # [B,N,C,C]
        scores = jnp.where(mask[:, :, None], scores, 0.0)
    else:
        scores = jnp.where(causal, scores, 0.0)
    o_intra = einsum("bnhts,bnshd->bnthd", scores, vc)  # [B,N,C,H,Dv]

    # Inter-chunk recurrence on the chunk-summarized state S [B,H,Dk,Dv].
    gamma_decay = jnp.exp(gamma[:, :, None] - lam)  # Gamma_j = exp(gamma - Lam_j) [B,N,C,H,Dk]
    k_fac = kc * gamma_decay
    if seg_keep is not None:
        k_fac = k_fac * seg_keep[..., None, None]  # only the live doc propagates forward
    chunk_kv = einsum("bnshd,bnshe->bnhde", k_fac, vc)  # [B,N,H,Dk,Dv]
    gamma_exp = jnp.exp(gamma)  # [B,N,H,Dk]
    q_lam = qc * exp_lam  # [B,N,C,H,Dk]

    keep_seq = jnp.ones((n, b), q.dtype) if carry_keep is None else jnp.moveaxis(carry_keep, 1, 0)

    def step(state, xs):
        g_exp, ckv, qlam, ckeep = xs  # [B,H,Dk], [B,H,Dk,Dv], [B,C,H,Dk], [B]
        o_inter = einsum("bchd,bhde->bche", qlam, state)  # [B,C,H,Dv] uses state from previous chunks
        state_new = ckeep[:, None, None, None] * g_exp[..., None] * state + ckv
        return state_new, o_inter

    # state0 inherits the carry's sharding (batch-sharded, head/feature replicated) so the
    # lax.scan carry type is consistent across iterations.
    state0 = jnp.zeros_like(chunk_kv[:, 0])  # [B,H,Dk,Dv]
    xs = (jnp.moveaxis(gamma_exp, 1, 0), jnp.moveaxis(chunk_kv, 1, 0), jnp.moveaxis(q_lam, 1, 0), keep_seq)
    _, o_inter = jax.lax.scan(step, state0, xs)  # [N,B,C,H,Dv]
    o_inter = jnp.moveaxis(o_inter, 0, 1)  # [B,N,C,H,Dv]
    if inter_keep is not None:
        o_inter = o_inter * inter_keep[:, :, :, None, None]

    out = (o_intra + o_inter).reshape(b, s, h, dv)
    return out


class GatedLinearAttention(eqx.Module):
    """GLA sequence mixer (arXiv 2312.06635), drop-in for ``CausalSelfAttention``.

    Same residual interface ``(x, mask, use_pko) -> [B, S, D]``; only ``mask``'s
    ``segment_ids`` are used (to reset the recurrence at document boundaries), and
    ``use_pko`` is ignored (GLA is inherently causal).

    Uses GQA to match the softmax-attention parameter footprint: ``num_heads`` query
    heads and ``num_kv_heads`` key/value heads of width ``head_dim``, with the K/V/gate
    heads repeated across each query group (exactly like ``CausalSelfAttention``). The
    value head dim is ``round(head_dim * expand_v)``.
    """

    w_q: Float[Array, "D Nq_Dk"]
    w_k: Float[Array, "D Nkv_Dk"]
    w_v: Float[Array, "D Nkv_Dv"]
    w_a1: Float[Array, "D R"]
    w_a2: Float[Array, "R Nkv_Dk"]
    b_a: Array
    w_r: Float[Array, "D Nq_Dv"]
    w_o: Float[Array, "Nq_Dv D"]
    head_norm_weight: Float[Array, "Nq Dvh"]
    num_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    v_head_dim: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    tau: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    @staticmethod
    def init(
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        initializer_std: float,
        chunk_size: int = 64,
        expand_v: float = 1.0,
        *,
        key: PRNGKeyArray,
        eps: float = 1e-5,
    ) -> "GatedLinearAttention":
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}")
        d = hidden_dim
        v_head_dim = round(head_dim * expand_v)
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        qv_dim = num_heads * v_head_dim
        kv_v_dim = num_kv_heads * v_head_dim
        kq, kk, kv, ka1, ka2, kr, ko = random.split(key, 7)
        return GatedLinearAttention(
            w_q=reshard(_init_weight(kq, (d, q_dim), initializer_std), P("data", "model")),
            w_k=reshard(_init_weight(kk, (d, kv_dim), initializer_std), P("data", "model")),
            w_v=reshard(_init_weight(kv, (d, kv_v_dim), initializer_std), P("data", "model")),
            w_a1=reshard(_init_weight(ka1, (d, _GATE_RANK), initializer_std), P("data", None)),
            w_a2=reshard(_init_weight(ka2, (_GATE_RANK, kv_dim), initializer_std), P(None, "model")),
            b_a=reshard(jnp.zeros((kv_dim,)), P("model")),
            w_r=reshard(_init_weight(kr, (d, qv_dim), initializer_std), P("data", "model")),
            w_o=reshard(_init_weight(ko, (qv_dim, d), initializer_std), P("model", "data")),
            head_norm_weight=reshard(jnp.ones((num_heads, v_head_dim)), P(None, None)),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            chunk_size=chunk_size,
            tau=float(_TAU),
            eps=eps,
        )

    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask=None,  # AttentionMask; only its segment_ids are used (GLA is inherently causal)
        use_pko: bool = False,  # unused
    ) -> Float[Array, "B S D"]:
        del use_pko
        seg = mask.segment_ids[0] if getattr(mask, "segment_ids", None) is not None else None
        n, m = self.num_heads, self.num_kv_heads
        hd, hdv = self.head_dim, self.v_head_dim
        b, s, _ = x.shape

        q = einsum("bsd,dk->bsk", x, self.w_q).reshape(b, s, n, hd).astype(jnp.float32)
        k = einsum("bsd,dk->bsk", x, self.w_k).reshape(b, s, m, hd).astype(jnp.float32)
        v = einsum("bsd,dv->bsv", x, self.w_v).reshape(b, s, m, hdv).astype(jnp.float32)

        # QK RMSNorm over the head dim (deliberate addition over vanilla FLA GLA, which uses
        # identity q,k; matches the Grug softmax attention that RMS-norms q,k, and stabilizes
        # the q.k^T magnitudes feeding the linear-attention state).
        q = _rmsnorm_last(q)
        k = _rmsnorm_last(k)

        # Data-dependent forget gate (low-rank), per KV head: alpha = sigmoid(z)^(1/tau);
        # log alpha = -softplus(-z)/tau <= 0.
        z = einsum("bsd,dr->bsr", x, self.w_a1)
        z = einsum("bsr,rk->bsk", z, self.w_a2) + self.b_a
        log_alpha = (-jax.nn.softplus(-z.astype(jnp.float32)) / self.tau).reshape(b, s, m, hd)

        # GQA: repeat each KV/gate head across its query group (block repeat, head j -> q heads [j*r:(j+1)*r]).
        if m != n:
            rep = n // m
            k = jnp.repeat(k, rep, axis=2)
            v = jnp.repeat(v, rep, axis=2)
            log_alpha = jnp.repeat(log_alpha, rep, axis=2)
        h, dv_h = n, hdv

        # Run the per-head GLA recurrence with batch-only sharding (head/feature replicated):
        # the recurrent state is tiny, and this keeps the chunk lax.scan carry sharding consistent.
        seq_spec = P(("data", "expert"), None, None, None)
        q = reshard(q, seq_spec)
        k = reshard(k, seq_spec)
        v = reshard(v, seq_spec)
        log_alpha = reshard(log_alpha, seq_spec)
        if seg is not None:
            seg = reshard(seg, P(("data", "expert"), None))
        o = gla_chunk(q, k, v, log_alpha, self.chunk_size, seg)  # [B,S,H,Dvh] fp32

        # Per-head RMSNorm on the head output, then concat.
        var = jnp.mean(jnp.square(o), axis=-1, keepdims=True)
        o = o * jax.lax.rsqrt(var + self.eps) * self.head_norm_weight
        o = o.reshape(b, s, h * dv_h).astype(x.dtype)

        # Swish output gate, then output projection.
        r = jax.nn.silu(einsum("bsd,dv->bsv", x, self.w_r))
        o = r * o
        return einsum("bsv,vd->bsd", o, self.w_o, out_sharding=_batch_spec())


__all__ = ["GatedLinearAttention", "gla_chunk", "gla_recurrent"]
