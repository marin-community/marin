# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local-window + learned sparse-index attention for the MLA layers, as a dense masked emulation.

Each query ``i`` attends to its own document's last ``window`` keys, the document's first key (the sink) and the
followers ``m + 1`` of the ``topk`` earlier positions ``m`` whose index vectors best match its own
(``score = idx_i . idx_m / sqrt(R)``, a tied low-rank similarity). Selected keys also get ``lam_h * score`` on their
logit, which is how the index learns. With a ``Lightning`` indexer (DeepSeek-V3.2's DSA) the selection is instead
the ``topk`` keys with the best ``I(i, s) = sum_h w_h(x_i) relu(q_h(x_i) . k(x_s))``, trained only by an injected
``KL(main attention || softmax(I))`` gradient. The scores are computed densely in query chunks (rematerialised), so
this measures quality, not speed.
"""

import math
from collections.abc import Callable
from enum import StrEnum
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import auto_axes, reshard
from jaxtyping import Array, Float, Int
from levanter.grug.attention._inkling_relpos import REL_BIAS_BLOCK, rel_extent_of_band
from levanter.kernels.pallas.autotune_utils import named_sharding_of

from experiments.grug.fast_track.key_offset import shift_tokens, shifted_channels

_QUERY_CHUNK = 256


class SparseIndex(StrEnum):
    """How ``mla_sparse_topk`` picks the far keys."""

    FOLLOW = "follow"
    """The followers of the best tied low-rank matches, scored onto their logits."""
    LIGHTNING = "lightning"
    """DSA's lightning indexer over the keys themselves, trained by KL to the dense attention."""
    BLOCK = "block"
    """NSA-style: the ``topk`` key blocks whose compressed (block-mean) keys the heads attend to most."""


class IndexSpec(NamedTuple):
    """Follow-index variants (static)."""

    untied: bool = False
    """Separate low-rank query and key projections (the index holds both halves) instead of a tied one."""
    anchor_match: bool = False
    """Attend to the matched key itself rather than its follower."""
    lookback: int = 0
    """> 0: candidates only this far back."""
    score_only: bool = False
    """Weight the picks by the low-rank score alone (no head q.k, Inkling bias or local window)."""


class Compose(NamedTuple):
    """Pre-softmax head composition: ``logit_h = sum_g mix[h, g] Z_g (+ u_h(x_i) sum_g w_g(x_i) Z_g)`` over the
    ``2H`` sub-head scores ``Z`` (each head's offset-channel score, then each head's current-channel score)."""

    mix: Float[Array, "H H2"]
    u: Float[Array, "B S H"] | None
    w: Float[Array, "B S H2"] | None
    offset_mask: jax.Array  # [H, D] channels whose key is the previous token's


class Gauss(eqx.Module):
    """Per-head Gaussian value kernel: shift ``mu`` (tokens, positive = later) and width ``softplus(rho) + 0.05``;
    ``qscale``: the tied query's per-head scale (``mla_gauss_tied``). Trained with Adam (``.gauss.``)."""

    mu: jax.Array
    rho: jax.Array
    qscale: jax.Array | None


class GaussValues(NamedTuple):
    """``[H, 2K + 1]`` kernel over value offsets ``-K..K``, the in-document smoothed values ``[B, S, H, E]`` and the
    kernel mass that stays in the document ``[B, S, H]``."""

    kernel: jax.Array
    smooth: jax.Array
    mass: jax.Array


def gauss_values(g: Gauss, v: jax.Array, segment_ids: jax.Array, reach: int) -> GaussValues:
    """Smooth each head's values along the sequence: ``smooth_j = sum_o G(o) v_{j+o}`` within the document."""
    offsets = jnp.arange(-reach, reach + 1, dtype=jnp.float32)
    sigma = jax.nn.softplus(g.rho.astype(jnp.float32)) + 0.05
    centred = offsets[None, :] - g.mu.astype(jnp.float32)[:, None]
    kernel = jax.nn.softmax(-jnp.square(centred) / (2 * sigma[:, None] ** 2), axis=-1)
    vf = v.astype(jnp.float32)
    ones = jnp.ones(v.shape[:3], jnp.float32)
    smooth = jnp.zeros_like(vf)
    mass = jnp.zeros(v.shape[:3], jnp.float32)
    for i, o in enumerate(range(-reach, reach + 1)):
        w = kernel[:, i]  # [H]
        smooth = smooth + w[None, None, :, None] * shift_tokens(vf, segment_ids, -o)
        mass = mass + w[None, None, :] * shift_tokens(ones, segment_ids, -o)
    return GaussValues(kernel, smooth, mass)


def _gauss_read(a, gv: GaussValues, v, segment_ids, qi, seg_q):
    """``sum_j a_ij smooth_j / sum_j a_ij mass_j`` without the kernel mass that lands after the query: for keys within
    the kernel's reach, subtract their spill onto tokens ``i + 1 .. i + K`` from the output and the normaliser."""
    batch, heads, chunk, seq = a.shape
    reach = (gv.kernel.shape[-1] - 1) // 2
    main = jnp.einsum("bhqk,bkhd->bqhd", a, gv.smooth)
    z = jnp.einsum("bhqk,bkh->bqh", a, gv.mass)
    m = jnp.arange(reach)
    back = qi[:, None] - m[None, :]  # [C, K]: key i - m
    a_band = jnp.take_along_axis(a, jnp.broadcast_to(jnp.clip(back, 0, seq - 1), (batch, heads, chunk, reach)), axis=-1)
    a_band = jnp.where(back >= 0, a_band, 0.0)
    u = jnp.arange(1, reach + 1)
    o = m[None, :] + u[:, None]  # [K(u), K(m)]: the offset that lands on token i + u
    spill_w = jnp.where(o <= reach, gv.kernel[:, jnp.clip(o + reach, 0, 2 * reach)], 0.0)  # [H, K, K]
    c = jnp.einsum("bhcm,hum->bhcu", a_band, spill_w)
    ahead = jnp.clip(qi[:, None] + u[None, :], 0, seq - 1)  # [C, K]
    same_doc = segment_ids[:, ahead] == seg_q[:, :, None]  # [B, C, K]
    ok = ((qi[:, None] + u[None, :]) < seq)[None] & same_doc
    c = c * ok[:, None]
    v_ahead = v.astype(jnp.float32)[:, ahead]  # [B, C, K, H, E]
    corr = jnp.einsum("bhcu,bcuhd->bchd", c, v_ahead)
    z = z - jnp.sum(c, axis=-1).transpose(0, 2, 1)
    return (main - corr) / (z[..., None] + 1e-6)


class DcmhaMode(NamedTuple):
    """Which DCMHA parts run (static)."""

    pre: bool = True
    post: bool = True
    dynamic: bool = True


class DcmhaWeights(NamedTuple):
    """DCMHA's per-token composition weights (``dcmha_weights``), one set per side (query ``q``, key ``k``):
    ``[B, S, 2 (pre/post), H, R]`` low-rank factors ``f1`` / ``f2`` and ``[B, S, 2, H]`` gates; plus the static
    ``[2, H, H]`` mixes."""

    static: Float[Array, "2 H H"]
    q1: jax.Array
    q2: jax.Array
    qg: jax.Array
    k1: jax.Array
    k2: jax.Array
    kg: jax.Array


def _dcmha_compose(a: jax.Array, dw: DcmhaWeights, stage: int, c0: int, chunk: int, dynamic: bool) -> jax.Array:
    """DCMHA ``Compose`` (arXiv 2405.08553 eq. 4) of one chunk's ``[B, H, C, S]`` scores or probabilities:
    ``a'_h = sum_g a_g (W_gh + sum_r q1_gr q2_hr + sum_r k1_gr k2_hr) + a_h (tanh qg_h + tanh kg_h)`` with the
    query-wise terms from query ``i`` and the key-wise terms from key ``j``."""
    take = lambda x: jax.lax.dynamic_slice_in_dim(x, c0, chunk, axis=1)[:, :, stage].astype(jnp.float32)  # noqa: E731
    q1, q2, qg = take(dw.q1), take(dw.q2), take(dw.qg)  # [B, C, H, R], [B, C, H]
    k1, k2, kg = (x[:, :, stage].astype(jnp.float32) for x in (dw.k1, dw.k2, dw.kg))  # [B, S, H, R], [B, S, H]
    out = jnp.einsum("bgqk,gh->bhqk", a, dw.static[stage].astype(jnp.float32))
    if not dynamic:
        return out
    out = out + jnp.einsum("bgqk,bqgr,bqhr->bhqk", a, q1, q2)
    out = out + jnp.einsum("bgqk,bkgr,bkhr->bhqk", a, k1, k2)
    gate = jnp.tanh(qg).transpose(0, 2, 1)[..., None] + jnp.tanh(kg).transpose(0, 2, 1)[:, :, None, :]
    return out + a * gate


class Lightning(NamedTuple):
    """DSA lightning-indexer inputs (computed from a stop-gradient attention input)."""

    q: Float[Array, "B S G R"]
    k: Float[Array, "B S R"]
    w: Float[Array, "B S G"]
    kl_weight: float


@jax.custom_vjp
def _inject_kl(out: jax.Array, index_logits: jax.Array, target: jax.Array, scale: jax.Array) -> jax.Array:
    """Identity on ``out``; backward gives ``index_logits`` the gradient of ``scale * KL(target || softmax)``."""
    del index_logits, target, scale
    return out


def _inject_kl_fwd(out, index_logits, target, scale):
    return out, (index_logits, target, scale)


def _inject_kl_bwd(res, g):
    index_logits, target, scale = res
    grad = scale * (jax.nn.softmax(index_logits, axis=-1) - target)
    return g, grad.astype(index_logits.dtype), jnp.zeros_like(target), jnp.zeros_like(scale)


_inject_kl.defvjp(_inject_kl_fwd, _inject_kl_bwd)


def _document_starts(segment_ids: Int[Array, "B S"]) -> jax.Array:
    """Each token's document start position ``[B, S]``."""
    seq = segment_ids.shape[1]
    idx = jnp.arange(seq, dtype=jnp.int32)
    starts = jnp.pad(segment_ids[:, 1:] != segment_ids[:, :-1], ((0, 0), (1, 0)), constant_values=True)
    return jax.lax.cummax(jnp.where(starts, idx, 0), axis=1)


def _sparse_math(
    q,
    k,
    v,
    rel_bias,
    segment_ids,
    index,
    lam,
    lightning,
    tap_weights,
    compose,
    dcmha,
    gauss,
    *,
    spec: "IndexSpec",
    tap_offsets: tuple[int, ...],
    dcmha_mode: "DcmhaMode",
    window: int,
    topk: int,
    dsa_topk: int,
    select_match: bool,
    block: int,
    compress: bool,
    block_select: bool,
):
    batch, seq, heads, head_dim = q.shape
    chunk = min(_QUERY_CHUNK, seq)
    if seq % chunk:
        raise ValueError(f"sequence length {seq} must be a multiple of {chunk}")
    scale = 1.0 / math.sqrt(head_dim)
    starts = _document_starts(segment_ids)
    keys = jnp.arange(seq)
    rel_extent = None if rel_bias is None else rel_extent_of_band(rel_bias)
    index = index.astype(jnp.float32) / math.sqrt(index.shape[-1])
    if compress or block_select:
        if seq % block:
            raise ValueError(f"sequence length {seq} must be a multiple of the block {block}")
        blocks = seq // block
        kc = k.astype(jnp.float32).reshape(batch, blocks, block, heads, -1).mean(axis=2)  # [B, nb, H, D]
        vc = v.astype(jnp.float32).reshape(batch, blocks, block, heads, -1).mean(axis=2)
        seg_blocks = segment_ids.reshape(batch, blocks, block)
        # A block is usable only when it lies inside one document; its last token sets when it is visible.
        whole = seg_blocks[..., 0] == seg_blocks[..., -1]
        block_end = jnp.arange(blocks) * block + block - 1

    @jax.checkpoint
    def one_chunk(c0):
        qi = c0 + jnp.arange(chunk)  # [C]
        qc = jax.lax.dynamic_slice_in_dim(q, c0, chunk, axis=1).astype(jnp.float32)
        if compose is None:
            logits = jnp.einsum("bqhd,bkhd->bhqk", qc * scale, k.astype(jnp.float32))  # [B, H, C, S]
        else:
            mask = compose.offset_mask.astype(jnp.float32)  # [H, D]
            kf = k.astype(jnp.float32)
            z_prev = jnp.einsum("bqhd,bkhd->bhqk", qc * scale * mask, kf)
            z_cur = jnp.einsum("bqhd,bkhd->bhqk", qc * scale * (1.0 - mask), kf)
            z = jnp.concatenate([z_prev, z_cur], axis=1)  # [B, 2H, C, S]
            logits = jnp.einsum("hg,bgqk->bhqk", compose.mix.astype(jnp.float32), z)
            if compose.u is not None and compose.w is not None:
                uc = jax.lax.dynamic_slice_in_dim(compose.u, c0, chunk, axis=1).astype(jnp.float32)
                wc = jax.lax.dynamic_slice_in_dim(compose.w, c0, chunk, axis=1).astype(jnp.float32)
                logits = logits + jnp.einsum("bqh,bqg,bgqk->bhqk", uc, wc, z)
        if rel_bias is not None:
            band = jax.lax.dynamic_slice_in_dim(rel_bias, c0, chunk, axis=2).astype(jnp.float32)
            column = keys[None, :] - (qi[:, None] // REL_BIAS_BLOCK) * REL_BIAS_BLOCK + rel_extent
            in_band = (column >= 0) & (column < band.shape[-1])
            gathered = jnp.take_along_axis(
                band, jnp.broadcast_to(jnp.clip(column, 0, band.shape[-1] - 1), band.shape[:2] + column.shape), axis=-1
            )
            logits = logits + jnp.where(in_band, gathered, 0.0)
        seg_q = jax.lax.dynamic_slice_in_dim(segment_ids, c0, chunk, axis=1)  # [B, C]
        start_q = jax.lax.dynamic_slice_in_dim(starts, c0, chunk, axis=1)
        same = seg_q[:, :, None] == segment_ids[:, None, :]  # [B, C, S]
        dist = qi[None, :, None] - keys[None, None, :]
        local = same & (dist >= 0) & ((dist < window) | (keys[None, None, :] == start_q[:, :, None]))
        allowed = local[:, None]  # [B, G, C, S], G = 1 or one per head
        bias = jnp.zeros(allowed.shape, jnp.float32)
        index_logits = None
        if lightning is not None:
            lq = jax.lax.dynamic_slice_in_dim(lightning.q, c0, chunk, axis=1).astype(jnp.float32)
            lw = jax.lax.dynamic_slice_in_dim(lightning.w, c0, chunk, axis=1).astype(jnp.float32)
            per_head = jax.nn.relu(jnp.einsum("bqgr,bkr->bqgk", lq, lightning.k.astype(jnp.float32)))
            visible = same & (dist >= 0)
            index_logits = jnp.where(visible, jnp.einsum("bqgk,bqg->bqk", per_head, lw), -1e9)
            kth = jax.lax.top_k(index_logits, dsa_topk)[0][..., -1:]
            allowed = allowed | (visible & (index_logits >= kth))[:, None]
        chosen = None
        if topk and not block_select:
            idx_q = jax.lax.dynamic_slice_in_dim(index, c0, chunk, axis=1)
            idx_k = index
            if spec.untied:  # separate low-rank query / key halves
                half = index.shape[-1] // 2
                idx_q, idx_k = idx_q[..., :half], index[..., half:]
            score = jnp.einsum("bqgr,bkgr->bgqk", idx_q, idx_k)  # [B, G, C, S] against candidate m
            if spec.anchor_match:
                # The candidate is the key itself: visible and outside the local window.
                cand = same & (dist >= 0) & (dist >= window)
            else:
                # Candidate m's follower m + 1 must be visible (m + 1 <= i) and outside the local window.
                cand = same & (dist >= 1) & (dist - 1 >= window)
            if spec.lookback:
                cand = cand & (dist < spec.lookback)
            cand = cand[:, None]
            masked = jnp.where(cand, score, -jnp.inf)
            kth = jax.lax.top_k(masked, topk)[0][..., -1:]
            matched = cand & (masked >= kth)  # kth = -inf (fewer than topk candidates): every candidate
            if spec.anchor_match:
                follow = matched
                bias = jnp.where(matched, score, 0.0)
            else:
                # Move candidate m's selection and score onto its follower key m + 1.
                follow = jnp.pad(matched[..., :-1], ((0, 0), (0, 0), (0, 0), (1, 0)))
                bias = jnp.where(follow, jnp.pad(score[..., :-1], ((0, 0), (0, 0), (0, 0), (1, 0))), 0.0)
            chosen = follow
            allowed = allowed | follow
            if select_match:  # also the matched token m itself
                bias = bias + jnp.where(matched & ~follow, score, 0.0)
                allowed = allowed | matched
        if compress or block_select:
            c_logits = jnp.einsum("bqhd,bnhd->bhqn", qc * scale, kc)  # [B, H, C, nb]
            # Fully past (so never the query's own, partly future block) and in the query's document.
            c_ok = whole[:, None, :] & (seg_blocks[:, None, :, 0] == seg_q[:, :, None])
            c_ok = c_ok & (block_end[None, None, :] < qi[None, :, None])  # [B, C, nb]
            c_logits = jnp.where(c_ok[:, None], c_logits, -1e9)
            if block_select:
                importance = jnp.mean(jax.nn.softmax(c_logits, axis=-1), axis=1)  # [B, C, nb]
                importance = jnp.where(c_ok, importance, -1.0)
                kth = jax.lax.top_k(importance, topk)[0][..., -1:]
                picked = c_ok & (importance >= kth)
                tokens = jnp.repeat(picked, block, axis=-1)  # [B, C, S]
                allowed = allowed | (tokens & same & (dist >= 0))[:, None]
        if spec.score_only and chosen is not None:
            # The picks are weighted by their low-rank score alone; the document start is a zero-logit null slot.
            sink = (keys[None, None, :] == start_q[:, :, None])[:, None]
            logits = jnp.where(chosen, bias * lam.astype(jnp.float32)[None, :, None, None], jnp.where(sink, 0.0, -1e9))
            allowed = chosen | sink
        else:
            logits = logits + bias * lam.astype(jnp.float32)[None, :, None, None]
        if dcmha is not None and dcmha_mode.pre:
            logits = _dcmha_compose(logits, dcmha, 0, c0, chunk, dcmha_mode.dynamic)
        dense = logits
        logits = jnp.where(allowed, logits, -1e9)
        if gauss is not None:
            out = _gauss_read(jax.nn.softmax(logits, axis=-1), gauss, v, segment_ids, qi, seg_q).astype(v.dtype)
        elif dcmha is not None and dcmha_mode.post:
            weights = _dcmha_compose(jax.nn.softmax(logits, axis=-1), dcmha, 1, c0, chunk, dcmha_mode.dynamic)
            out = jnp.einsum("bhqk,bkhd->bqhd", weights, v.astype(jnp.float32)).astype(v.dtype)
        elif compress:
            joint = jax.nn.softmax(jnp.concatenate([logits, c_logits], axis=-1), axis=-1)
            weights, c_weights = joint[..., :seq], joint[..., seq:]
            out = jnp.einsum("bhqk,bkhd->bqhd", weights, v.astype(jnp.float32))
            out = (out + jnp.einsum("bhqn,bnhd->bqhd", c_weights, vc)).astype(v.dtype)
        elif tap_weights is not None and chosen is not None:
            # Each pick reads a learned mix of the values around it: the picks' weights convolved along the key
            # axis by the per-head (or per-channel) taps, dropping any tap past the query or out of its document.
            weights = jax.nn.softmax(logits, axis=-1)
            far = jnp.where(chosen, weights, 0.0)
            vf = v.astype(jnp.float32)
            out = jnp.einsum("bhqk,bkhd->bqhd", weights - far, vf)
            reach = (same & (dist >= 0))[:, None]
            tap_w = tap_weights.astype(jnp.float32)  # [H, T] or [H, T, E]
            for t, offset in enumerate(tap_offsets):
                moved = far if offset == 0 else jnp.roll(far, offset, axis=-1)
                if offset > 0:
                    moved = moved.at[..., :offset].set(0.0)
                elif offset < 0:
                    moved = moved.at[..., offset:].set(0.0)
                read = jnp.einsum("bhqk,bkhd->bqhd", jnp.where(reach, moved, 0.0), vf)
                w_t = tap_w[:, t] if tap_w.ndim == 2 else tap_w[:, t, :]
                out = out + read * (w_t[None, None, :, None] if w_t.ndim == 1 else w_t[None, None])
            out = out.astype(v.dtype)
        else:
            weights = jax.nn.softmax(logits, axis=-1)
            out = jnp.einsum("bhqk,bkhd->bqhd", weights, v.astype(jnp.float32)).astype(v.dtype)
        if index_logits is None:
            return out
        # DSA target: the dense causal attention, summed over heads and renormalised.
        visible = same & (dist >= 0)
        target = jnp.mean(jax.nn.softmax(jnp.where(visible[:, None], dense, -1e9), axis=-1), axis=1)
        kl_scale = jnp.asarray(lightning.kl_weight / (batch * seq), jnp.float32)
        return _inject_kl(out, index_logits, jax.lax.stop_gradient(target), kl_scale)

    chunks = jax.lax.map(one_chunk, jnp.arange(0, seq, chunk))  # [n, B, C, H, D]
    return jnp.moveaxis(chunks, 0, 1).reshape(batch, seq, heads, v.shape[-1])


def sparse_index_attention(
    q: Float[Array, "B S H D"],
    k: Float[Array, "B S H D"],
    v: Float[Array, "B S H E"],
    rel_bias: Float[Array, "B H S W"] | None,
    segment_ids: Int[Array, "B S"] | None,
    index: Float[Array, "B S G R"],
    lam: Float[Array, " H"],
    lightning: Lightning | None = None,
    tap_weights: jax.Array | None = None,
    compose: Compose | None = None,
    dcmha: DcmhaWeights | None = None,
    gauss: GaussValues | None = None,
    *,
    window: int,
    topk: int,
    dsa_topk: int = 0,
    spec: IndexSpec = IndexSpec(),
    tap_offsets: tuple[int, ...] = (),
    dcmha_mode: DcmhaMode = DcmhaMode(),
    select_match: bool = False,
    block: int = 32,
    compress: bool = False,
    block_select: bool = False,
) -> Float[Array, "B S H E"]:
    """Attention over each query's local ``window``, its document start and its ``topk`` index-matched followers."""
    if segment_ids is None:
        segment_ids = jnp.zeros(q.shape[:2], jnp.int32)
    math_fn = lambda *a: _sparse_math(  # noqa: E731
        *a,
        window=window,
        topk=topk,
        dsa_topk=dsa_topk,
        spec=spec,
        tap_offsets=tap_offsets,
        dcmha_mode=dcmha_mode,
        select_match=select_match,
        block=block,
        compress=compress,
        block_select=block_select,
    )
    out_sharding = named_sharding_of(q)
    if out_sharding is None:
        return math_fn(q, k, v, rel_bias, segment_ids, index, lam, lightning, tap_weights, compose, dcmha, gauss)
    # pyrefly: ignore[bad-assignment]  # auto_axes's decorator overload erases the wrapped signature
    wrapped: Callable[..., jax.Array] = auto_axes(math_fn, out_sharding=out_sharding)
    return wrapped(q, k, v, rel_bias, segment_ids, index, lam, lightning, tap_weights, compose, dcmha, gauss)


def mla_sparse_attention(attn, x, q, k, v, rel_bias, segment_ids, rms_norm, project):
    """The MLA layer's sparse attention from its config and index parameters (``attn``: the MLA module)."""
    cfg = attn.cfg
    dsa_first = cfg.mla_sparse_index == SparseIndex.LIGHTNING
    idx = jnp.zeros((*x.shape[:2], 1, 1))
    if attn.w_sidx is not None:
        flat = project(cfg, x, attn.w_sidx)
        idx = flat.reshape(*flat.shape[:-1], -1, cfg.mla_sparse_index_dim * (2 if cfg.mla_sparse_untied else 1))
        if cfg.mla_sparse_untied:  # RMS-norm the query and key halves separately
            r = cfg.mla_sparse_index_dim
            idx = jnp.concatenate([rms_norm(idx[..., :r]), rms_norm(idx[..., r:])], axis=-1)
        else:
            idx = rms_norm(idx)
    lam = jnp.zeros((q.shape[2],)) if attn.sparse_lam is None else attn.sparse_lam
    light = None
    if attn.w_lq is not None and attn.w_lk is not None and attn.w_lw is not None:
        xs = jax.lax.stop_gradient(x)
        lq = project(cfg, xs, attn.w_lq)
        lq = lq.reshape(*lq.shape[:-1], cfg.mla_sparse_index_heads, -1)
        light = Lightning(lq, project(cfg, xs, attn.w_lk), project(cfg, xs, attn.w_lw), cfg.mla_sparse_kl_weight)
    compose = None
    if attn.logit_mix is not None:
        heads, head_dim = q.shape[2], q.shape[3]
        dynamic = attn.logit_mix_u is not None and attn.logit_mix_w is not None
        compose = Compose(
            attn.logit_mix,
            project(cfg, x, attn.logit_mix_u) if dynamic else None,
            project(cfg, x, attn.logit_mix_w) if dynamic else None,
            (
                shifted_channels(heads, head_dim, cfg.mla_key_offset_channels)
                if cfg.mla_key_offset
                else jnp.zeros((heads, head_dim), bool)
            ),
        )
    gv = None
    if attn.gauss is not None:
        seg = jnp.zeros(q.shape[:2], jnp.int32) if segment_ids is None else segment_ids
        gv = gauss_values(attn.gauss, v, seg, cfg.mla_gauss_reach)
        if attn.gauss.qscale is not None:  # tied: the query is the token's own key
            q = (k.astype(jnp.float32) * attn.gauss.qscale.astype(jnp.float32)[None, None, :, None]).astype(q.dtype)
    spec = IndexSpec(
        untied=cfg.mla_sparse_untied,
        anchor_match=cfg.mla_sparse_anchor_match,
        lookback=cfg.mla_sparse_lookback,
        score_only=cfg.mla_sparse_score_only,
    )
    return sparse_index_attention(
        q,
        k,
        v,
        rel_bias,
        segment_ids,
        idx,
        lam,
        light,
        attn.sparse_taps,
        compose,
        None if attn.dcmha is None else dcmha_weights(attn.dcmha, x, q.shape[2], cfg.mla_dcmha_rank),
        gv,
        window=cfg.mla_sparse_window,
        spec=spec,
        tap_offsets=cfg.mla_sparse_taps,
        dcmha_mode=DcmhaMode(
            pre="pre" in cfg.mla_dcmha_parts,
            post="post" in cfg.mla_dcmha_parts,
            dynamic="dynamic" in cfg.mla_dcmha_parts,
        ),
        topk=0 if dsa_first else cfg.mla_sparse_topk,
        dsa_topk=cfg.mla_sparse_topk if dsa_first else cfg.mla_sparse_dsa_topk,
        select_match=cfg.mla_sparse_select_match,
        block=cfg.mla_sparse_block,
        compress=cfg.mla_sparse_compress,
        block_select=cfg.mla_sparse_index == SparseIndex.BLOCK,
    )


_DCMHA_BOUND = 0.5
"""Largest change any static entry or second dynamic factor can make (``tanh``-bounded)."""


class Dcmha(eqx.Module):
    """DCMHA parameters: static ``[H, H]`` pre- and post-softmax mixes ``I + 0.5 tanh(delta)`` (``delta`` zero-init;
    separate leaves, since a stacked 4-D leaf would be sharded like an expert weight) and, per side, a
    ``GELU(x W_a) W_b`` network giving each token's low-rank factors and gates (the second factors and the gates
    are zero-init, so the layer starts as plain attention). Trained with Adam (``.dcmha.``)."""

    static_pre: jax.Array
    static_post: jax.Array
    q_a: jax.Array
    q_b1: jax.Array
    q_b2: jax.Array
    q_bg: jax.Array
    k_a: jax.Array
    k_b1: jax.Array
    k_b2: jax.Array
    k_bg: jax.Array


def dcmha_weights(p: Dcmha, x: jax.Array, heads: int, rank: int) -> DcmhaWeights:
    """Each token's DCMHA composition weights from its attention input ``x``."""
    xf = x.astype(jnp.float32)

    def side(a, b1, b2, bg):
        hidden = jax.nn.gelu(xf @ a)
        shape = (*x.shape[:2], 2, heads)
        # Bounded factors, so two stacked compositions cannot drive the attention output to overflow.
        return (
            jnp.tanh(hidden @ b1).reshape(*shape, rank),
            (_DCMHA_BOUND * jnp.tanh(hidden @ b2)).reshape(*shape, rank),
            (hidden @ bg).reshape(shape),
        )

    eye = jnp.eye(heads, dtype=jnp.float32)
    static = jnp.stack([eye + _DCMHA_BOUND * jnp.tanh(p.static_pre), eye + _DCMHA_BOUND * jnp.tanh(p.static_post)])
    return DcmhaWeights(static, *side(p.q_a, p.q_b1, p.q_b2, p.q_bg), *side(p.k_a, p.k_b1, p.k_b2, p.k_bg))


def mla_sparse_params(cfg, d: int, n: int, h: int, key: jax.Array, std: float, init_weight) -> dict:
    """The MLA module's sparse-index, value-tap, previous-token gate and logit-composition parameters (None when off)."""

    def matrix(i: int, width: int) -> jax.Array:
        return reshard(init_weight(random.fold_in(key, i), (d, width), std), P(None, None))

    follow = cfg.mla_sparse_topk > 0 and cfg.mla_sparse_index == SparseIndex.FOLLOW
    lightning = cfg.mla_sparse_dsa_topk > 0 or (
        cfg.mla_sparse_topk > 0 and cfg.mla_sparse_index == SparseIndex.LIGHTNING
    )
    groups = n if cfg.mla_sparse_index_per_head else 1
    r = cfg.mla_sparse_index_dim
    taps = None
    if cfg.mla_sparse_taps:
        shape = (n, len(cfg.mla_sparse_taps)) + ((h,) if cfg.mla_sparse_taps_per_channel else ())
        taps = jnp.zeros(shape).at[:, cfg.mla_sparse_taps.index(0)].set(1.0)
    return dict(
        w_sidx=matrix(0, groups * r * (2 if cfg.mla_sparse_untied else 1)) if follow else None,
        w_lq=matrix(1, cfg.mla_sparse_index_heads * r) if lightning else None,
        w_lk=matrix(2, r) if lightning else None,
        w_lw=matrix(3, cfg.mla_sparse_index_heads) if lightning else None,
        sparse_lam=jnp.ones((n,), jnp.float32) if follow else None,
        sparse_taps=taps,
        prev_gate=jnp.zeros((d, n), jnp.float32) if cfg.mla_prev_gate else None,
        logit_mix=(
            jnp.concatenate([jnp.eye(n), jnp.eye(n)], axis=1).astype(jnp.float32) if cfg.mla_logit_compose else None
        ),
        logit_mix_u=jnp.zeros((d, n), jnp.float32) if cfg.mla_logit_compose_dynamic else None,
        logit_mix_w=matrix(4, 2 * n) if cfg.mla_logit_compose_dynamic else None,
        gauss=_gauss_init(cfg, n) if cfg.mla_gauss else None,
        dcmha=(
            _dcmha_init(random.fold_in(key, 5), d, n, cfg.mla_dcmha_rank, cfg.mla_dcmha_hidden)
            if cfg.mla_dcmha
            else None
        ),
    )


def _gauss_init(cfg, heads: int) -> Gauss:
    mu = jnp.asarray(cfg.mla_gauss_mu_init if cfg.mla_gauss_mu_init else (0.0,) * heads, jnp.float32)
    if mu.shape != (heads,):
        raise ValueError(f"mla_gauss_mu_init needs one entry per head ({heads}), got {cfg.mla_gauss_mu_init}")
    rho = jnp.full((heads,), math.log(math.expm1(cfg.mla_gauss_sigma_init - 0.05)), jnp.float32)
    qscale = jnp.full((heads,), cfg.qk_mult, jnp.float32) if cfg.mla_gauss_tied else None
    return Gauss(mu=mu, rho=rho, qscale=qscale)


def _dcmha_init(key: jax.Array, d: int, heads: int, rank: int, hidden: int) -> Dcmha:
    ka, kb, kc, kd = random.split(key, 4)
    width = 2 * heads * rank  # pre- and post-softmax
    a = lambda k: random.normal(k, (d, hidden)) / math.sqrt(d)  # noqa: E731
    b1 = lambda k: random.normal(k, (hidden, width)) / math.sqrt(hidden)  # noqa: E731
    return Dcmha(
        static_pre=jnp.zeros((heads, heads), jnp.float32),
        static_post=jnp.zeros((heads, heads), jnp.float32),
        q_a=a(ka),
        q_b1=b1(kb),
        q_b2=jnp.zeros((hidden, width)),
        q_bg=jnp.zeros((hidden, 2 * heads)),
        k_a=a(kc),
        k_b1=b1(kd),
        k_b2=jnp.zeros((hidden, width)),
        k_bg=jnp.zeros((hidden, 2 * heads)),
    )
