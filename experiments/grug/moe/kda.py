# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kimi Delta Attention (KDA) kernels: the gated delta rule with a per-channel
forget gate, in the chunked-parallel form that keeps the linear-attention layers
matmul-bound (good MFU) instead of a token-by-token recurrence.

Two entry points, both operating on plain JAX arrays with arbitrary leading batch
dims ``(..., L, d)`` (in grug the leading dims are ``(batch, heads)``):

  - ``recurrent_kda``: the sequential reference (one token at a time). Correct but
    O(L) sequential steps -- used as the numerical oracle and for decode.
  - ``chunk_kda``: the chunkwise-parallel kernel used for training/prefill. Splits
    the sequence into chunks of ``chunk_size``; within a chunk everything is dense
    matmuls plus one unit-lower-triangular inverse (log-depth block doubling),
    across chunks a single recurrent state ``S`` is carried. This is the
    MFU-friendly form.

State ``S ∈ R^{d_k x d_v}`` maps keys to values; the read is ``o_t = S_t^T q_t`` and
the per-channel decay ``alpha_t = exp(g_t) ∈ (0,1]^{d_k}`` left-multiplies ``S`` on
the key axis. KDA is the per-channel generalization of scalar Gated DeltaNet
(``levanter.layers.gated_deltanet``); in the scalar limit (``g`` broadcast over
``d_k``) the two agree. All decay math is done in fp32.

References: Kimi Linear tech report (arXiv 2510.26692), Gated Delta Networks
(arXiv 2412.06464), Parallelizing Linear Transformers with the Delta Rule
(arXiv 2406.06484), and flash-linear-attention ``fla/ops/kda/naive.py``.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Int

from experiments.grug.moe.kda_prep_pallas import (
    DEFLATE_EXP_CAP,
    L2NORM_EPS,
    ChunkPrep,
    fused_chunk_prep,
)
from experiments.grug.moe.kda_state_pallas import chunk_state_pass

_L2NORM_EPS = L2NORM_EPS
# Cap the (single-sided) deflation exponent exp(-cumdecay) so extreme per-channel decay
# cannot overflow; see DEFLATE_EXP_CAP for when the chunked form stays exact.
_DEFLATE_EXP_CAP = DEFLATE_EXP_CAP


def _l2norm(x: jax.Array, eps: float = _L2NORM_EPS) -> jax.Array:
    """L2-normalize along the last axis (in fp32)."""
    x32 = x.astype(jnp.float32)
    inv = lax.rsqrt(jnp.sum(x32 * x32, axis=-1, keepdims=True) + eps)
    return x32 * inv


def _prepare_qk(q: jax.Array, k: jax.Array, use_qk_l2norm: bool) -> tuple[jax.Array, jax.Array]:
    """Optionally L2-normalize q,k, then scale q by 1/sqrt(d_k)."""
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    if use_qk_l2norm:
        q = _l2norm(q)
        k = _l2norm(k)
    q = q * (q.shape[-1] ** -0.5)
    return q, k


def doc_starts(segment_ids: jax.Array) -> jax.Array:
    """fp32 ``(..., L)``: 1 where a token starts a new document (its segment id differs
    from its predecessor's), else 0. Position 0 is never a start (it continues from the
    initial state)."""
    change = segment_ids[..., 1:] != segment_ids[..., :-1]
    return jnp.concatenate([jnp.zeros_like(change[..., :1]), change], axis=-1).astype(jnp.float32)


class _SegmentMasks(NamedTuple):
    """Per-chunk document masks (all from the within-chunk document index ``idx``, the
    count of document starts at or before each token of the chunk)."""

    same: jax.Array  # (..., n, C, C) bool: tokens r, i in the same document
    carry_in: jax.Array  # (..., n, C) fp32: token still in the document the chunk started in
    last_doc: jax.Array  # (..., n, C) fp32: token in the chunk's last document
    carry_through: jax.Array  # (..., n) fp32: no document starts inside the chunk


def _segment_masks(starts_c: jax.Array) -> _SegmentMasks:
    idx = jnp.cumsum(starts_c, axis=-1)
    last = idx[..., -1:]
    return _SegmentMasks(
        same=idx[..., :, None] == idx[..., None, :],
        carry_in=(idx == 0).astype(jnp.float32),
        last_doc=(idx == last).astype(jnp.float32),
        carry_through=(last[..., 0] == 0).astype(jnp.float32),
    )


def recurrent_kda(
    q: Float[Array, "... L Dk"],
    k: Float[Array, "... L Dk"],
    v: Float[Array, "... L Dv"],
    g: Float[Array, "... L Dk"],
    beta: Float[Array, "... L"],
    *,
    initial_state: jax.Array | None = None,
    use_qk_l2norm: bool = True,
    segment_ids: Int[Array, "... L"] | None = None,
) -> tuple[Float[Array, "... L Dv"], jax.Array]:
    """Sequential per-channel gated delta rule (reference / decode kernel).

    Args:
        q, k: ``(..., L, d_k)`` queries/keys.
        v: ``(..., L, d_v)`` values.
        g: ``(..., L, d_k)`` per-channel log-decay (``alpha = exp(g)``, ``g <= 0``).
        beta: ``(..., L)`` per-head write strength in ``(0, 1)``.
        initial_state: optional ``(..., d_k, d_v)`` starting state.
        use_qk_l2norm: L2-normalize q,k inside the kernel (and scale q by 1/sqrt(d_k)).
        segment_ids: optional ``(..., L)`` document ids of packed sequences; the state
            is reset to zero at every document start (a token whose id differs from
            its predecessor), so documents never see each other.

    Returns:
        ``(outputs (..., L, d_v), final_state (..., d_k, d_v))``.
    """
    q, k = _prepare_qk(q, k, use_qk_l2norm)
    v = v.astype(jnp.float32)
    g = g.astype(jnp.float32)
    beta = beta.astype(jnp.float32)

    lead = q.shape[:-2]
    dk, dv = q.shape[-1], v.shape[-1]
    if initial_state is None:
        state = jnp.zeros((*lead, dk, dv), dtype=jnp.float32)
    else:
        state = initial_state.astype(jnp.float32)

    # Move the length axis to the front for lax.scan.
    q_t = jnp.moveaxis(q, -2, 0)  # (L, ..., d_k)
    k_t = jnp.moveaxis(k, -2, 0)
    v_t = jnp.moveaxis(v, -2, 0)
    g_t = jnp.moveaxis(g, -2, 0)
    b_t = jnp.moveaxis(beta, -1, 0)  # (L, ...)
    if segment_ids is None:
        keep_t = jnp.ones(b_t.shape, jnp.float32)
    else:
        keep_t = jnp.moveaxis(1.0 - doc_starts(jnp.broadcast_to(segment_ids, beta.shape)), -1, 0)

    def step(s_prev: jax.Array, inp):
        q_i, k_i, v_i, g_i, b_i, keep_i = inp
        s_prev = s_prev * (jnp.exp(g_i) * keep_i[..., None])[..., :, None]  # Diag(alpha), 0 at doc starts
        kv = jnp.sum(k_i[..., :, None] * s_prev, axis=-2)  # S^T k  -> (..., d_v)
        delta = (v_i - kv) * b_i[..., None]
        s_new = s_prev + k_i[..., :, None] * delta[..., None, :]
        o_i = jnp.sum(q_i[..., :, None] * s_new, axis=-2)  # S^T q -> (..., d_v)
        return s_new, o_i

    state, out_t = lax.scan(step, state, (q_t, k_t, v_t, g_t, b_t, keep_t))
    out = jnp.moveaxis(out_t, 0, -2)  # (..., L, d_v)
    return out, state


def _unit_lower_triangular_inverse(
    a_strict_lower: jax.Array, chunk_size: int, matmul_dtype: jnp.dtype | None = None
) -> jax.Array:
    """Invert a unit-lower-triangular ``(I - A)`` given strictly-lower ``A``.

    Returns ``T = (I - A)^{-1}`` (unit lower-triangular) for ``a_strict_lower`` of shape
    ``(..., C, C)``, by recursive block doubling: with ``T_s`` the inverse of the ``s x s``
    diagonal blocks, the ``2s`` blocks are ``[[T11, 0], [T22 A21 T11, T22]]``, i.e.
    ``T_2s = T_s + T_s (A * M_s) T_s`` where ``M_s`` selects the lower-left ``s x s`` block of
    every ``2s`` block. ``2 * ceil(log2 C)`` batched ``C x C`` matmuls, all parallel.

    Every intermediate is a block of the true inverse, so it stays bounded when the keys in a
    chunk are nearly parallel (repeated tokens). The Neumann product ``(I+A)(I+A^2)(I+A^4)...``
    this replaces sums binomially large alternating terms there and overflowed to inf/NaN.

    ``matmul_dtype`` (e.g. ``bfloat16``) runs the products in that dtype while carrying ``T``
    in fp32; ``None`` keeps fp32.
    """
    out_dtype = a_strict_lower.dtype

    def mm(x: jax.Array, y: jax.Array) -> jax.Array:
        if matmul_dtype is None:
            return x @ y
        return (x.astype(matmul_dtype) @ y.astype(matmul_dtype)).astype(out_dtype)

    rows = jnp.arange(chunk_size)[:, None]
    cols = jnp.arange(chunk_size)[None, :]
    inv = jnp.broadcast_to(jnp.eye(chunk_size, dtype=out_dtype), a_strict_lower.shape)
    s = 1
    while s < chunk_size:
        lower_left = (rows // (2 * s) == cols // (2 * s)) & ((rows // s) % 2 == 1) & ((cols // s) % 2 == 0)
        inv = inv + mm(mm(inv, jnp.where(lower_left, a_strict_lower, 0.0)), inv)
        s *= 2
    return inv


def chunk_kda(
    q: Float[Array, "... L Dk"],
    k: Float[Array, "... L Dk"],
    v: Float[Array, "... L Dv"],
    g: Float[Array, "... L Dk"],
    beta: Float[Array, "... L"],
    *,
    chunk_size: int = 128,
    initial_state: jax.Array | None = None,
    use_qk_l2norm: bool = True,
    matmul_dtype: jnp.dtype | None = jnp.bfloat16,
    scan_unroll: int = 1,
    scan_impl: str = "parallel",
    prep_impl: str = "xla",
    segment_ids: Int[Array, "... L"] | None = None,
) -> tuple[Float[Array, "... L Dv"], jax.Array]:
    """Chunkwise-parallel per-channel gated delta rule (KDA train/prefill kernel).

    Does the intra-chunk work as dense matmuls plus one unit-lower-triangular inverse,
    and carries a single recurrent state ``S`` across chunks -- the matmul-bound form
    that keeps the layer efficient. See module docstring for the equations.

    Args mirror :func:`recurrent_kda`. ``chunk_size`` is the intra-chunk length C.
    With the default bf16 GEMMs, C=128 is the H100 sweet spot (fewer sequential
    scan steps, and the larger intra-chunk GEMMs are cheap on tensor cores); C=64
    is a lower-memory fallback (the reverse pass saves ~2x less CxC state). In fp32
    the optimum is C=64. ``matmul_dtype`` selects the dtype of the
    *intra-chunk* GEMM operands (delta-correction matrix, its block-doubling inverse, the
    pseudo-value/decayed-key products, and the intra-chunk attention); the fp32 cross-
    chunk state recurrence and all decay/cumsum math are always fp32. ``bfloat16``
    (the default) runs the intra-chunk GEMMs on Hopper/Blackwell tensor cores and is
    ~8% (fwd) / ~13% (fwd+bwd) faster on H100 at ~0.5% relative error vs the fp32
    reference; pass ``jnp.float32`` for the exact-fp32 path used as the test oracle.

    ``segment_ids`` (broadcastable to ``(..., L)``) packs several documents per row: the
    state is hard-reset at every document start (see :func:`recurrent_kda`), implemented
    with per-chunk document masks (no -inf gates). Not supported by ``scan_impl="sequential"``.

    ``prep_impl`` / ``scan_impl`` pick the implementation of the two stages. The fastest
    H100 configuration is ``prep_impl="pallas", scan_impl="pallas", chunk_size=64``: one
    fused Triton kernel for the whole intra-chunk prep (``kda_prep_pallas``) and one
    sequential on-chip state pass that also emits the outputs (``kda_state_pallas``),
    ~4x faster forward / ~2.7x faster forward+backward than the XLA default at
    B=4, H=8, L=8192, d=128 (see ``.agents/projects/kimi-delta-attention-h100.md``).
    The ``"pallas_interpret"`` variants run the same kernels in the Pallas interpreter
    (CPU tests). The XLA path (``"xla"`` / ``"parallel"``, or ``"sequential"``) stays the
    default so the function runs on any backend.
    """
    mm_dtype = matmul_dtype

    def mm(spec: str, a: jax.Array, b: jax.Array) -> jax.Array:
        """Einsum with operands cast to ``mm_dtype`` (fp32-accumulated), else fp32."""
        if mm_dtype is None:
            return jnp.einsum(spec, a, b)
        return jnp.einsum(spec, a.astype(mm_dtype), b.astype(mm_dtype)).astype(jnp.float32)

    # The fused kernel reads the inputs in their own dtype (bf16 activations stay bf16
    # in HBM) and applies the q/k L2-norm and query scale itself.
    if prep_impl not in ("pallas", "pallas_interpret"):
        q, k = _prepare_qk(q, k, use_qk_l2norm)
        v = v.astype(jnp.float32)
        g = g.astype(jnp.float32)
        beta = beta.astype(jnp.float32)

    lead = q.shape[:-2]
    orig_len = q.shape[-2]
    dk, dv = q.shape[-1], v.shape[-1]
    pad = (-orig_len) % chunk_size
    if pad:
        q = jnp.pad(q, [(0, 0)] * (q.ndim - 2) + [(0, pad), (0, 0)])
        k = jnp.pad(k, [(0, 0)] * (k.ndim - 2) + [(0, pad), (0, 0)])
        v = jnp.pad(v, [(0, 0)] * (v.ndim - 2) + [(0, pad), (0, 0)])
        g = jnp.pad(g, [(0, 0)] * (g.ndim - 2) + [(0, pad), (0, 0)])
        beta = jnp.pad(beta, [(0, 0)] * (beta.ndim - 1) + [(0, pad)])

    n_chunks = q.shape[-2] // chunk_size
    c = chunk_size
    starts, masks = None, None
    if segment_ids is not None:
        seg = jnp.broadcast_to(segment_ids, (*lead, orig_len))
        if pad:  # padding continues the last document
            seg = jnp.pad(seg, [(0, 0)] * (seg.ndim - 1) + [(0, pad)], mode="edge")
        starts = doc_starts(seg)
        masks = _segment_masks(starts.reshape(*lead, n_chunks, c))

    def to_chunks(x: jax.Array) -> jax.Array:
        # (..., n_chunks, C, d) with chunks on the front for lax.scan later.
        return x.reshape(*lead, n_chunks, c, x.shape[-1])

    qc = to_chunks(q)
    kc = to_chunks(k)
    vc = to_chunks(v)
    gc = to_chunks(g)
    bc = beta.reshape(*lead, n_chunks, c)

    if initial_state is None:
        state = jnp.zeros((*lead, dk, dv), dtype=jnp.float32)
    else:
        state = initial_state.astype(jnp.float32)

    if scan_impl == "sequential":
        if prep_impl != "xla":
            raise ValueError("scan_impl='sequential' only supports prep_impl='xla'")
        if segment_ids is not None:
            raise ValueError("scan_impl='sequential' does not support segment_ids")
        inter = _chunk_intermediates(qc, kc, vc, gc, bc, mm, mm_dtype, c, None)
        strict_upper = jnp.triu(jnp.ones((c, c), dtype=bool), k=1)
        out, state = _sequential_chunk_recurrence(
            inter.q_inflate,
            inter.k_deflate,
            kc,
            inter.g_cum,
            inter.v_pseudo,
            inter.k_cumdecay,
            state,
            strict_upper,
            mm,
            scan_unroll,
        )
    elif scan_impl in ("parallel", "pallas", "pallas_interpret"):
        if prep_impl == "xla":
            prep = _chunk_prep_xla(qc, kc, vc, gc, bc, mm, mm_dtype, c, masks)
        elif prep_impl in ("pallas", "pallas_interpret"):
            prep = _chunk_prep_pallas(
                q,
                k,
                v,
                g,
                beta,
                starts,
                c,
                mm_dtype,
                use_qk_l2norm=use_qk_l2norm,
                interpret=prep_impl == "pallas_interpret",
            )
        else:
            raise ValueError(f"prep_impl must be 'xla', 'pallas' or 'pallas_interpret', got {prep_impl!r}")
        if scan_impl == "parallel":
            out, state = _parallel_chunk_recurrence(prep, state, mm, lead, dk, dv)
        else:
            out, state = _pallas_chunk_recurrence(prep, state, interpret=scan_impl == "pallas_interpret")
    else:
        raise ValueError(
            f"scan_impl must be 'sequential', 'parallel', 'pallas' or 'pallas_interpret', got {scan_impl!r}"
        )

    out = out.reshape(*lead, n_chunks * c, dv)
    if pad:
        out = out[..., :orig_len, :]
    return out, state


def _sequential_chunk_recurrence(
    q_inflate, k_deflate, kc, g_cum, v_pseudo, k_cumdecay, state, strict_upper, mm, scan_unroll
):
    """Serial ``lax.scan`` over chunks. Correct, but its depth is L/C
    sequential steps -- latency-bound on GPU (see ``_parallel_chunk_recurrence``)."""

    def move_chunk_front(x: jax.Array) -> jax.Array:
        return jnp.moveaxis(x, -3, 0)

    scan_inputs = tuple(move_chunk_front(x) for x in (q_inflate, k_deflate, kc, g_cum, v_pseudo, k_cumdecay))

    def chunk_step(s_prev: jax.Array, inp):
        q_inf, k_def, k_i, gcum_i, v_ps, k_cd = inp
        attn = mm("...rd,...jd->...rj", q_inf, k_def)
        attn = jnp.where(strict_upper, 0.0, attn)
        # State-touching GEMMs stay fp32 (S carries the whole prefix -- precision matters).
        v_prime = jnp.einsum("...rd,...dm->...rm", k_cd, s_prev)
        v_new = v_ps - v_prime
        inter = jnp.einsum("...rd,...dm->...rm", q_inf, s_prev)
        out_i = inter + mm("...rj,...jm->...rm", attn, v_new)
        g_tail = gcum_i[..., -1, :]
        decay_tail = jnp.exp(g_tail)
        decay_weights = jnp.exp(g_tail[..., None, :] - gcum_i)
        add = jnp.einsum("...rd,...rm->...dm", k_i * decay_weights, v_new)
        s_new = s_prev * decay_tail[..., :, None] + add
        return s_new, out_i

    state, out_chunks = lax.scan(chunk_step, state, scan_inputs, unroll=scan_unroll)
    return jnp.moveaxis(out_chunks, 0, -3), state


class _ChunkIntermediates(NamedTuple):
    g_cum: jax.Array
    q_inflate: jax.Array
    k_deflate: jax.Array
    v_pseudo: jax.Array
    k_cumdecay: jax.Array


def _chunk_intermediates(qc, kc, vc, gc, bc, mm, mm_dtype, c, masks: _SegmentMasks | None) -> _ChunkIntermediates:
    """XLA intra-chunk prep shared by both recurrences: gates, ``A``, its inverse, U/W.

    With document masks the cumulative log-decay restarts at every document start (a
    masked within-document cumsum), so a document's decay factors never contain another
    document's gates: documents stay numerically independent, not just mathematically.

    With document ``masks``, ``A`` only couples tokens of the same document (so ``T`` is
    block-diagonal per document) and the incoming state reaches only the tokens still in
    the chunk's first document (``k_cumdecay`` rows of later documents are zero). Decays
    are only ever taken between tokens of one document, so no -inf gates are needed."""
    if masks is None:
        g_cum = jnp.cumsum(gc, axis=-2)  # (..., n, C, d_k) cumulative log-decay per channel
    else:
        seg_lower = (jnp.tril(jnp.ones((c, c), dtype=bool)) & masks.same).astype(jnp.float32)
        g_cum = jnp.einsum("...tj,...jd->...td", seg_lower, gc, precision=lax.Precision.HIGHEST)
    exp_g = jnp.exp(g_cum)  # <= 1 (g_cum <= 0): inflation factor, always safe
    exp_ng = jnp.exp(jnp.minimum(-g_cum, _DEFLATE_EXP_CAP))  # deflation factor, capped

    v_beta = vc * bc[..., None]
    k_beta = kc * bc[..., None]

    # Delta-correction matrix A[r,i] = -beta_r (k_r . k_i) exp(g_cum_r - g_cum_i),
    # folded via inflate/deflate so it is a plain matmul. Strictly lower triangular.
    k_beta_inflate = k_beta * exp_g
    k_deflate = kc * exp_ng
    a_raw = -mm("...rd,...id->...ri", k_beta_inflate, k_deflate)
    strict_lower = jnp.tril(jnp.ones((c, c), dtype=bool), k=-1)
    if masks is not None:
        strict_lower = strict_lower & masks.same
    a_raw = jnp.where(strict_lower, a_raw, 0.0)

    lead_n = a_raw.shape[:-2]
    a_bcc = a_raw.reshape(-1, c, c)
    t_mat = _unit_lower_triangular_inverse(a_bcc, c, matmul_dtype=mm_dtype).reshape(*lead_n, c, c)

    v_pseudo = mm("...rj,...jd->...rd", t_mat, v_beta)
    k_state = k_beta_inflate if masks is None else k_beta_inflate * masks.carry_in[..., None]
    k_cumdecay = mm("...rj,...jd->...rd", t_mat, k_state)
    return _ChunkIntermediates(g_cum, qc * exp_g, k_deflate, v_pseudo, k_cumdecay)


def _chunk_prep_xla(qc, kc, vc, gc, bc, mm, mm_dtype, c, masks: _SegmentMasks | None) -> ChunkPrep:
    """Reference (XLA) prep: intermediates plus the per-chunk attention, the keys
    decayed to the chunk end and the whole-chunk decay. ``fused_chunk_prep`` computes
    the same tuple in one kernel.

    With document ``masks``: the attention is within-document; ``q_inflate`` (only used
    against the incoming state) is zero past the chunk's first document; only the last
    document's keys feed the outgoing state; and the incoming state is dropped (decay 0)
    if a document starts inside the chunk."""
    inter = _chunk_intermediates(qc, kc, vc, gc, bc, mm, mm_dtype, c, masks)
    lower = jnp.tril(jnp.ones((c, c), dtype=bool))
    if masks is not None:
        lower = lower & masks.same
    attn = jnp.where(lower, mm("...rd,...jd->...rj", inter.q_inflate, inter.k_deflate), 0.0)

    g_tail = inter.g_cum[..., -1, :]  # (..., n, d_k)
    # Clamped: <= 0 on the rows that feed the state; earlier documents' rows (masked off
    # below) could otherwise overflow exp to inf and turn the 0 mask into NaN.
    kw = kc * jnp.exp(jnp.minimum(g_tail[..., None, :] - inter.g_cum, 0.0))  # (..., n, C, d_k)
    q_inflate, decay = inter.q_inflate, jnp.exp(g_tail)
    if masks is not None:
        q_inflate = q_inflate * masks.carry_in[..., None]
        kw = kw * masks.last_doc[..., None]
        decay = decay * masks.carry_through[..., None]
    return ChunkPrep(q_inflate, inter.k_cumdecay, inter.v_pseudo, attn, kw, decay)


def _heads_first_to_model_layout(x: jax.Array, trailing: int) -> jax.Array:
    """``(*lead, L[, d])`` -> ``(B, L, H[, d])`` with ``H = lead[-1]``, ``B = prod(lead[:-1])``."""
    lead = x.shape[: x.ndim - 1 - trailing]
    heads = lead[-1] if lead else 1
    x = x.reshape(-1, heads, *x.shape[len(lead) :])
    return jnp.swapaxes(x, 1, 2)


def _fused_prep(
    q, k, v, g, beta, c, mm_dtype, *, use_qk_l2norm: bool, interpret: bool, gate=None, starts=None
) -> ChunkPrep:
    """Fused Pallas prep on model-layout ``(B, L, H, d)`` inputs; per-chunk outputs ``(G, n, ...)``."""
    return fused_chunk_prep(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=c,
        mm_dtype=jnp.float32 if mm_dtype is None else mm_dtype,
        gate=gate,
        doc_starts=starts,
        use_qk_l2norm=use_qk_l2norm,
        interpret=interpret,
    )


def _chunk_prep_pallas(q, k, v, g, beta, starts, c, mm_dtype, *, use_qk_l2norm: bool, interpret: bool) -> ChunkPrep:
    """Fused Pallas prep from heads-first ``(*lead, L, d)`` inputs; outputs ``(*lead, n, ...)``.

    ``starts`` (document starts, ``(*lead, L)``) must be the same for every head; the
    kernel reads one ``(B, L)`` row per batch element."""
    lead = q.shape[:-2]
    if starts is not None:
        heads = lead[-1] if lead else 1
        starts = starts.reshape(-1, heads, starts.shape[-1])[:, 0]
    prep = _fused_prep(
        *(_heads_first_to_model_layout(x, 1) for x in (q, k, v, g)),
        _heads_first_to_model_layout(beta, 0),
        c,
        mm_dtype,
        use_qk_l2norm=use_qk_l2norm,
        interpret=interpret,
        starts=starts,
    )
    return ChunkPrep(*(x.reshape(*lead, *x.shape[1:]) for x in prep))


def _parallel_chunk_recurrence(prep: ChunkPrep, state, mm, lead, dk, dv):
    """Chunk-parallel inter-chunk recurrence via a log-depth associative scan.

    The cross-chunk state update is *linear* in ``S``:

        S_n = Diag(decay_tail_n) S_{n-1} + Kw_n^T (v_pseudo_n - k_cumdecay_n S_{n-1})
            = M_n S_{n-1} + C_n,   M_n = Diag(decay_tail_n) - Kw_n^T k_cumdecay_n,

    so instead of the L/C serial ``lax.scan`` we run ``lax.associative_scan`` over the
    affine transforms ``(M_n, C_n)`` (composition ``(M_r M_l, M_r C_l + C_r)``): depth
    drops from L/C to log2(L/C), and the per-chunk outputs then compute in one parallel
    batched pass. All state math is fp32 (stability); only the intra-chunk attention
    GEMM uses ``mm`` (bf16). This is the arXiv 2406.06484 chunk-parallel delta rule.
    """
    eye_dk = jnp.eye(dk, dtype=jnp.float32)
    m_mat = prep.decay[..., :, None] * eye_dk - jnp.einsum("...rd,...re->...de", prep.kw, prep.k_cumdecay)
    c_mat = jnp.einsum("...rd,...rm->...dm", prep.kw, prep.v_pseudo)  # (..., n, d_k, d_v)
    # Fold the initial state into chunk 0: S_0 = M_0 S_init + C_0 (a no-op when S_init=0).
    c0 = c_mat[..., 0, :, :] + jnp.einsum("...de,...ef->...df", m_mat[..., 0, :, :], state)
    c_mat = c_mat.at[..., 0, :, :].set(c0)

    def combine(left, right):
        m_l, c_l = left
        m_r, c_r = right
        m = jnp.einsum("...ij,...jk->...ik", m_r, m_l)
        c = jnp.einsum("...ij,...jm->...im", m_r, c_l) + c_r
        return m, c

    _, s_incl = lax.associative_scan(combine, (m_mat, c_mat), axis=-3)  # S_0..S_{n-1}

    # State entering each chunk: [S_init, S_0, ..., S_{n-2}].
    s_init = jnp.broadcast_to(state[..., None, :, :], (*lead, 1, dk, dv))
    s_prev = jnp.concatenate([s_init, s_incl[..., :-1, :, :]], axis=-3)
    final_state = s_incl[..., -1, :, :]

    # Per-chunk outputs, all parallel: o = q_inflate S_prev + tril(attn)(v_pseudo - k_cumdecay S_prev).
    v_new = prep.v_pseudo - jnp.einsum("...rd,...dm->...rm", prep.k_cumdecay, s_prev)
    inter = jnp.einsum("...rd,...dm->...rm", prep.q_inflate, s_prev)
    out = inter + mm("...rj,...jm->...rm", prep.attn, v_new)
    return out, final_state


def _pallas_chunk_recurrence(prep: ChunkPrep, state, *, interpret: bool):
    """Inter-chunk recurrence and chunk outputs as one sequential Pallas state pass
    (``kda_state_pallas``) from heads-first per-chunk tensors ``(*lead, n, ...)``."""
    lead = prep.kw.shape[:-3]
    heads = lead[-1] if lead else 1

    def flat(x, trailing):
        return x.reshape(-1, *x.shape[-trailing:])

    out, final_state = chunk_state_pass(
        flat(prep.q_inflate, 3),
        flat(prep.attn, 3),
        flat(prep.kw, 3),
        flat(prep.k_cumdecay, 3),
        flat(prep.v_pseudo, 3),
        flat(prep.decay, 2),
        state.reshape(-1, *state.shape[-2:]),
        num_heads=heads,
        interpret=interpret,
    )
    # (B, L, H, d_v) -> (*lead, n, C, d_v), the heads-first chunked layout of the other paths.
    n, c = prep.kw.shape[-3], prep.kw.shape[-2]
    out = jnp.swapaxes(out, 1, 2).reshape(*lead, n, c, out.shape[-1])
    return out, final_state.reshape(state.shape)


def kda_fused(
    q: Float[Array, "B L H Dk"],
    k: Float[Array, "B L M Dk"],
    v: Float[Array, "B L M Dv"],
    g: Float[Array, "B L H Dk"],
    beta: Float[Array, "B L H"],
    *,
    gate: tuple[Float[Array, "H Dk"], Float[Array, "H Dk"]] | None = None,
    segment_ids: Int[Array, "B L"] | None = None,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    interpret: bool = False,
    save_chunk_states: bool = False,
) -> Float[Array, "B L H Dv"]:
    """Fused-kernel KDA in the model's ``(batch, seq, heads, head_dim)`` layout (GPU).

    Same math as :func:`chunk_kda` with ``prep_impl="pallas", scan_impl="pallas"``, but
    the kernels read q/k/v/g/beta and write the output directly in the activation layout
    (no ``(B, H, L)`` transposes), and the output comes back in ``v.dtype``. Zero initial
    state; the final state is not returned. ``interpret`` runs the Pallas interpreter.

    Grouped-query k/v (``M`` heads dividing ``H``; query head h uses kv head
    ``h // (H/M)``) are read in place. With ``gate = (rate, bias)`` (each ``(H, d_k)``),
    ``g`` is the gate pre-activation and the log-decay ``rate * softplus(g + bias)`` is
    computed on-chip (Kimi Linear's gate with ``rate = -exp(A_log)``).

    ``segment_ids`` (``(B, L)``) packs documents: the state is hard-reset at every
    document start (see :func:`recurrent_kda`). ``save_chunk_states`` keeps the state pass's
    per-chunk states for the backward (see :func:`chunk_state_pass`).
    """
    b, length, heads, _ = q.shape
    pad = (-length) % chunk_size
    if pad:  # zero padding: k = 0 (no write) and g = 0 (no decay) leave the state unchanged
        q, k, v, g = (jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0))) for x in (q, k, v, g))
        beta = jnp.pad(beta, ((0, 0), (0, pad), (0, 0)))
    starts = None
    if segment_ids is not None:
        seg = jnp.pad(segment_ids, ((0, 0), (0, pad)), mode="edge") if pad else segment_ids
        starts = doc_starts(seg)
    prep = _fused_prep(
        q,
        k,
        v,
        g,
        beta,
        chunk_size,
        matmul_dtype,
        use_qk_l2norm=use_qk_l2norm,
        interpret=interpret,
        gate=gate,
        starts=starts,
    )
    state = jnp.zeros((b * heads, q.shape[-1], v.shape[-1]), jnp.float32)
    out, _ = chunk_state_pass(
        prep.q_inflate,
        prep.attn,
        prep.kw,
        prep.k_cumdecay,
        prep.v_pseudo,
        prep.decay,
        state,
        num_heads=heads,
        out_dtype=v.dtype,
        interpret=interpret,
        save_states=save_chunk_states,
    )
    return out[:, :length] if pad else out
