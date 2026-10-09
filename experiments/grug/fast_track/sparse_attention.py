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

import functools
import math
from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import auto_axes, reshard
from jaxtyping import Array, Float, Int
from levanter.grug.attention._inkling_relpos import REL_BIAS_BLOCK, rel_extent_of_band
from levanter.kernels.pallas.autotune_utils import named_sharding_of

from experiments.grug.fast_track.key_offset import shift_tokens

_QUERY_CHUNK = 256


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
    """Per-head value kernel, a mixture of ``C`` Gaussians: shifts ``mu`` ``[H, C]`` (tokens, positive = later),
    widths ``softplus(rho) + 0.05`` and weights ``softmax(mix)``; ``qscale``: the tied query's per-head scale
    (``mla_gauss_tied``). ``taps`` ``[H, 1 or E, 2K + 1]``: free kernel logits replacing the Gaussian
    (``mla_gauss_taps``). Trained in the ``dcmha`` group (``.gauss.``)."""

    mu: jax.Array
    rho: jax.Array
    mix: jax.Array
    qscale: jax.Array | None
    taps: jax.Array | None = None


class GaussValues(NamedTuple):
    """``[H, 2K + 1]`` kernel over value offsets ``-K..K``, the in-document smoothed values ``[B, S, H, E]`` and the
    kernel mass that stays in the document ``[B, S, H]``."""

    kernel: jax.Array
    smooth: jax.Array
    mass: jax.Array


def _gauss_read(a, gv: GaussValues, v, segment_ids, qi, seg_q):
    """``sum_{t <= i} [sum_j a_ij G(t - j)] v_t`` over the query's document, normalised by its total weight. Keys at
    least ``K`` back use the smoothed values (their kernel lies entirely in the past); for the ``K`` nearest keys the
    weights on raw past tokens ``i - 2K + 1 .. i`` are built explicitly, so no future value ever enters a sum (a
    subtract-the-spill form leaks through rounding when the normaliser is small)."""
    batch, heads, chunk, seq = a.shape
    reach = (gv.kernel.shape[-1] - 1) // 2
    keys = jnp.arange(seq)
    far = (qi[:, None] - keys[None, :]) >= reach  # [C, S]
    a_far = jnp.where(far[None, None], a, 0.0)
    num = jnp.einsum("bhqk,bkhd->bqhd", a_far, gv.smooth)
    z = jnp.einsum("bhqk,bkh->bqh", a_far, gv.mass)
    m = jnp.arange(reach)
    back = qi[:, None] - m[None, :]  # [C, K]: near key i - m
    a_band = jnp.take_along_axis(a, jnp.broadcast_to(jnp.clip(back, 0, seq - 1), (batch, heads, chunk, reach)), axis=-1)
    a_band = jnp.where(back >= 0, a_band, 0.0)
    r = jnp.arange(2 * reach)  # token t = i - r
    o = m[None, :] - r[:, None]  # [2K(r), K(m)]: offset from key i - m to token i - r
    near_w = jnp.where(jnp.abs(o) <= reach, gv.kernel[:, jnp.clip(o + reach, 0, 2 * reach)], 0.0)  # [H, 2K, K]
    w = jnp.einsum("bhcm,hrm->bhcr", a_band, near_w)  # weight on token i - r
    tok = qi[:, None] - r[None, :]  # [C, 2K]
    tok_c = jnp.clip(tok, 0, seq - 1)
    ok = (tok >= 0)[None] & (segment_ids[:, tok_c] == seg_q[:, :, None])  # [B, C, 2K]
    w = w * ok[:, None]
    num = num + jnp.einsum("bhcr,bcrhd->bchd", w, v.astype(jnp.float32)[:, tok_c])
    z = z + jnp.sum(w, axis=-1).transpose(0, 2, 1)
    return num / (z[..., None] + 1e-6)


class DcmhaMode(NamedTuple):
    """Which DCMHA parts run (static)."""

    pre: bool = True
    post: bool = True
    terms: tuple[str, ...] = ("qw", "kw", "qg", "kg")
    """Dynamic terms: query-wise / key-wise low-rank mixing and query / key gates (empty: static only)."""


def _dcmha_terms(parts: tuple[str, ...]) -> tuple[str, ...]:
    """``dynamic`` turns on every dynamic term; otherwise the listed ``qw`` / ``kw`` / ``qg`` / ``kg``."""
    if "dynamic" in parts:
        return ("qw", "kw", "qg", "kg")
    return tuple(t for t in ("qw", "kw", "qg", "kg") if t in parts)


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


def _dcmha_compose(a: jax.Array, dw: DcmhaWeights, stage: int, c0: int, chunk: int, terms: tuple[str, ...]) -> jax.Array:
    """DCMHA ``Compose`` (arXiv 2405.08553 eq. 4) of one chunk's ``[B, H, C, S]`` scores or probabilities:
    ``a'_h = sum_g a_g (W_gh + sum_r q1_gr q2_hr + sum_r k1_gr k2_hr) + a_h (tanh qg_h + tanh kg_h)`` with the
    query-wise terms from query ``i`` and the key-wise terms from key ``j``."""
    take = lambda x: jax.lax.dynamic_slice_in_dim(x, c0, chunk, axis=1)[:, :, stage].astype(jnp.float32)  # noqa: E731
    q1, q2, qg = take(dw.q1), take(dw.q2), take(dw.qg)  # [B, C, H, R], [B, C, H]
    k1, k2, kg = (x[:, :, stage].astype(jnp.float32) for x in (dw.k1, dw.k2, dw.kg))  # [B, S, H, R], [B, S, H]
    out = jnp.einsum("bgqk,gh->bhqk", a, dw.static[stage].astype(jnp.float32))
    if "qw" in terms:
        out = out + jnp.einsum("bgqk,bqgr,bqhr->bhqk", a, q1, q2)
    if "kw" in terms:
        out = out + jnp.einsum("bgqk,bkgr,bkhr->bhqk", a, k1, k2)
    if "qg" in terms:
        out = out + a * jnp.tanh(qg).transpose(0, 2, 1)[..., None]
    if "kg" in terms:
        out = out + a * jnp.tanh(kg).transpose(0, 2, 1)[:, :, None, :]
    return out


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
    far_half: str = "",
    far_window: int = 128,
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
            if far_half:
                # Keys at least far_window back score on one half of the channels only (cur: first, prev: second).
                half = qc.shape[-1] // 2
                sl = slice(0, half) if far_half == "cur" else slice(half, None)
                far = jnp.einsum("bqhd,bkhd->bhqk", qc[..., sl] * scale, k.astype(jnp.float32)[..., sl])
                dist_ = qi[:, None] - keys[None, :]
                logits = jnp.where((dist_ >= far_window)[None, None], far, logits)
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
            index_logits = jnp.where(visible, jnp.einsum("bqgk,bqg->bqk", per_head, lw), -1000000000.0)
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
            c_logits = jnp.where(c_ok[:, None], c_logits, -1000000000.0)
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
            logits = jnp.where(
                chosen, bias * lam.astype(jnp.float32)[None, :, None, None], jnp.where(sink, 0.0, -1000000000.0)
            )
            allowed = chosen | sink
        else:
            logits = logits + bias * lam.astype(jnp.float32)[None, :, None, None]
        if dcmha is not None and dcmha_mode.pre:
            logits = _dcmha_compose(logits, dcmha, 0, c0, chunk, dcmha_mode.terms)
        dense = logits
        logits = jnp.where(allowed, logits, -1000000000.0)
        if gauss is not None:
            out = _gauss_read(jax.nn.softmax(logits, axis=-1), gauss, v, segment_ids, qi, seg_q).astype(v.dtype)
        elif dcmha is not None and dcmha_mode.post:
            weights = _dcmha_compose(jax.nn.softmax(logits, axis=-1), dcmha, 1, c0, chunk, dcmha_mode.terms)
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
        target = jnp.mean(jax.nn.softmax(jnp.where(visible[:, None], dense, -1000000000.0), axis=-1), axis=1)
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
    far_half: str = "",
    far_window: int = 128,
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
        far_half=far_half,
        far_window=far_window,
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
    idx = jnp.zeros((*x.shape[:2], 1, 1))
    lam = jnp.zeros((q.shape[2],))
    spec = IndexSpec(
        untied=(False),
        anchor_match=(False),
        lookback=(0),
        score_only=(False),
    )
    return sparse_index_attention(
        q,
        k,
        v,
        rel_bias,
        segment_ids,
        idx,
        lam,
        (None),
        None,
        (None),
        None,
        (None),
        window=(0),
        spec=spec,
        tap_offsets=(),
        far_half=(""),
        far_window=(128),
        dcmha_mode=DcmhaMode(
            pre=True,
            post=True,
            terms=_dcmha_terms(("pre", "post", "dynamic")),
        ),
        topk=(0),
        dsa_topk=(0),
        select_match=(False),
        block=(32),
        compress=(False),
        block_select=False,
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


class DcmhaPaper(eqx.Module):
    """DCMHA as released (``Caiyun-AI/DCFormer`` ``DynamicWeightProjection``): per branch (pre/post x query/key) a
    ``GELU(x W1)`` hidden layer of width K, factors ``w = hidden W2`` split into ``w1`` (RMS-normed over heads, no
    gain) and ``w2``, gates ``tanh(x Wd)``; static ``I + sw`` with ``sw`` absent, zero or Xavier-initialised."""

    dw1: jax.Array  # [D, 4, K]
    qkw: jax.Array  # [4, K, 2R, H]
    dd: jax.Array  # [D, 4H]
    sw: jax.Array | None  # [2, H, H]


def _dcmha_paper_weights(p: DcmhaPaper, x: jax.Array, heads: int, rank: int) -> DcmhaWeights:
    xf = x.astype(jnp.float32)
    hidden = jax.nn.gelu(jnp.einsum("btd,dck->btck", xf, p.dw1.astype(jnp.float32)))
    w = jnp.einsum("btck,ckjh->btcjh", hidden, p.qkw.astype(jnp.float32))  # [B, T, 4, 2R, H]
    w1, w2 = w[..., :rank, :], w[..., rank:, :]
    w1 = w1 * jax.lax.rsqrt(jnp.mean(jnp.square(w1), axis=-1, keepdims=True) + 1e-6)
    w1, w2 = jnp.swapaxes(w1, -1, -2), jnp.swapaxes(w2, -1, -2)  # [B, T, 4, H, R]
    dd = (xf @ p.dd.astype(jnp.float32)).reshape(*x.shape[:2], 4, heads)  # tanh is applied in the compose
    pre_q, pre_k, post_q, post_k = 0, 1, 2, 3
    stack = lambda t, a, b: jnp.stack([t[:, :, a], t[:, :, b]], axis=2)  # noqa: E731
    eye = jnp.eye(heads, dtype=jnp.float32)
    static = eye[None] + (0.0 if p.sw is None else p.sw.astype(jnp.float32))
    static = jnp.broadcast_to(static, (2, heads, heads))
    return DcmhaWeights(
        static,
        stack(w1, pre_q, post_q),
        stack(w2, pre_q, post_q),
        stack(dd, pre_q, post_q),
        stack(w1, pre_k, post_k),
        stack(w2, pre_k, post_k),
        stack(dd, pre_k, post_k),
    )


def dcmha_weights(p: "Dcmha | DcmhaPaper", x: jax.Array, heads: int, rank: int) -> DcmhaWeights:
    """Each token's DCMHA composition weights from its attention input ``x``."""
    if isinstance(p, DcmhaPaper):
        return _dcmha_paper_weights(p, x, heads, rank)
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


def gauss_kernel(g: Gauss, reach: int) -> jax.Array:
    """The ``[H, 1 or E, 2K + 1]`` value kernel over offsets ``-K..K``: the taps, else the Gaussian mixture."""
    if g.taps is not None:
        return jax.nn.softmax(g.taps.astype(jnp.float32), axis=-1)
    return _gaussian_kernel(g, reach)[:, None, :]


def _gaussian_kernel(g: Gauss, reach: int) -> jax.Array:
    offsets = jnp.arange(-reach, reach + 1, dtype=jnp.float32)
    sigma = jax.nn.softplus(g.rho.astype(jnp.float32)) + 0.05
    centred = offsets[None, None, :] - g.mu.astype(jnp.float32)[..., None]
    components = jax.nn.softmax(-jnp.square(centred) / (2 * sigma[..., None] ** 2), axis=-1)
    return jnp.einsum("hc,hco->ho", jax.nn.softmax(g.mix.astype(jnp.float32), axis=-1), components)


def gauss_attention_fast(q, k, v, rel_bias, segment_ids, g: Gauss, reach: int, attend_with_lse):
    """Gaussian value-kernel attention without the dense emulation. Each key ``j`` reads its smoothed value
    ``sum_o G(o) v_{j+o} / sum_o G(o)`` over the offsets visible to the query (in its document, ``j + o <= i``).
    Keys at least ``K`` back see their whole kernel: one FlashAttention call over keys and smoothed values moved
    ``K`` slots later (so the causal mask admits exactly ``j <= i - K``; the bias band rolls with them) returns
    their output and log-sum-exp. The ``K`` nearest keys are scored explicitly over a ``2K``-token window of raw
    values, and the two parts merge by log-sum-exp. No value after the query enters any sum.

    ``attend_with_lse(q, k, v, lower_bounds, valid, rel_bias) -> (out [B, S, H, E], lse [B, H, S])``."""
    seq = q.shape[1]
    kernel = gauss_kernel(g, reach)  # [H, 1 or E, 2K + 1]
    starts = _document_starts(segment_ids)  # [B, S]
    pos = jnp.arange(seq)
    vf = v.astype(jnp.float32)
    # Far keys: their kernel is clipped only by the document (and sequence) ends.
    ones = jnp.ones((*v.shape[:3], 1), jnp.float32)
    smooth = jnp.zeros_like(vf)
    mass = jnp.zeros((*v.shape[:3], kernel.shape[1]), jnp.float32)
    for t, o in enumerate(range(-reach, reach + 1)):
        w = kernel[None, None, :, :, t]  # [1, 1, H, 1 or E]
        smooth = smooth + w * shift_tokens(vf, segment_ids, -o)
        mass = mass + w * shift_tokens(ones, segment_ids, -o)
    smooth = (smooth / jnp.maximum(mass, 1e-9)).astype(v.dtype)  # a spike kernel leaves 0 mass at doc ends
    k_far = jnp.pad(k[:, : seq - reach], ((0, 0), (reach, 0), (0, 0), (0, 0)))
    v_far = jnp.pad(smooth[:, : seq - reach], ((0, 0), (reach, 0), (0, 0), (0, 0)))
    band = (
        None if rel_bias is None else jnp.pad(rel_bias[..., : rel_bias.shape[-1] - reach], ((0, 0),) * 3 + ((reach, 0),))
    )
    lower = starts + reach
    far_ok = pos[None, :] >= lower  # [B, S]
    out_far, lse_far = attend_with_lse(q, reshard_like(k_far, k), reshard_like(v_far, v), lower, far_ok, band)
    out_sharding = named_sharding_of(q)
    near = functools.partial(_gauss_near_merge, reach=reach)
    if out_sharding is None:
        return near(q, k, v, rel_bias, starts, kernel, far_ok, out_far, lse_far)
    # The near band's gathers and contractions run under Auto axes (explicit sharding cannot infer them).
    # pyrefly: ignore[bad-assignment]
    wrapped: Callable[..., jax.Array] = auto_axes(near, out_sharding=out_sharding)
    return wrapped(q, k, v, rel_bias, starts, kernel, far_ok, out_far, lse_far)


def _gauss_near_merge(q, k, v, rel_bias, starts, kernel, far_ok, out_far, lse_far, *, reach: int):
    """The ``K`` nearest keys of :func:`gauss_attention_fast`, merged with the far part by log-sum-exp."""
    batch, seq, heads, head_dim = q.shape
    pos = jnp.arange(seq)
    vf = v.astype(jnp.float32)
    lse_far = jnp.where(far_ok[:, None, :], lse_far.astype(jnp.float32), -jnp.inf)  # [B, H, S]
    # Near keys j = i - m (m < K): explicit scores, and smoothed values clipped at the query.
    m = jnp.arange(reach)
    near_idx = pos[:, None] - m[None, :]  # [S, K]
    near_ok = near_idx[None] >= starts[:, :, None]  # [B, S, K]
    k_near = k[:, jnp.clip(near_idx, 0, seq - 1)]  # [B, S, K, H, D]
    scores = jnp.einsum("bshd,bsmhd->bhsm", q.astype(jnp.float32), k_near.astype(jnp.float32)) / math.sqrt(head_dim)
    if rel_bias is not None:
        column = (pos % REL_BIAS_BLOCK)[:, None] + rel_extent_of_band(rel_bias) - m[None, :]  # [S, K]
        scores = scores + jnp.take_along_axis(
            rel_bias.astype(jnp.float32), jnp.broadcast_to(column, (batch, heads, seq, reach)), axis=-1
        )
    scores = jnp.where(near_ok[:, None], scores, -jnp.inf)
    r = jnp.arange(2 * reach)  # raw token i - r
    tok = pos[:, None] - r[None, :]
    tok_ok = tok[None] >= starts[:, :, None]  # [B, S, 2K] (always <= i)
    offset = m[:, None] - r[None, :]  # [K(m), 2K(r)]: offset from key i - m to token i - r
    in_kernel = (offset >= -reach) & (offset <= reach)
    # [H, C, K, 2K] with C = 1 (per-head kernel) or E (per-channel)
    w_base = jnp.where(in_kernel[None, None], kernel[:, :, jnp.clip(offset + reach, 0, 2 * reach)], 0.0)
    tok_okf = tok_ok.astype(jnp.float32)
    near_mass = jnp.einsum("hcmr,bsr->bhsmc", w_base, tok_okf)  # [B, H, S, K, C]
    top = jnp.maximum(lse_far, jnp.max(scores, axis=-1))  # [B, H, S]
    far_w = jnp.exp(lse_far - top)
    near_p = jnp.exp(scores - top[..., None])  # [B, H, S, K]
    coef = near_p[..., None] / jnp.maximum(near_mass, 1e-9)
    tok_w = jnp.einsum("bhsmc,hcmr->bhsrc", coef, w_base) * tok_okf[:, None, :, :, None]  # weight on token i - r
    v_win = vf[:, jnp.clip(tok, 0, seq - 1)]  # [B, S, 2K, H, E]
    if kernel.shape[1] == 1:
        near_num = jnp.einsum("bhsr,bsrhe->bshe", tok_w[..., 0], v_win)
    else:
        near_num = jnp.einsum("bhsre,bsrhe->bshe", tok_w, v_win)
    den = far_w + jnp.sum(near_p, axis=-1)  # [B, H, S]
    num = far_w.transpose(0, 2, 1)[..., None] * out_far.astype(jnp.float32) + near_num
    return (num / den.transpose(0, 2, 1)[..., None]).astype(v.dtype)


def reshard_like(x: jax.Array, like: jax.Array) -> jax.Array:
    spec = named_sharding_of(like)
    return x if spec is None else reshard(x, spec.spec)


PARALLAX_QUERY_BLOCK = 256


def parallax_attention(q, k, v, r, rel_bias, segment_ids):
    """Parallax (arXiv 2605.29157, ``parallax/reference.py``): softmax attention minus the attention-weighted
    key-value covariance probed by ``r``, ``o_i - Cov_{p_i}(v, k) r_i = sum_j p_ij v_j (1 + E_p[s_i]) - sum_j p_ij
    s_ij v_j`` with ``s_ij = r_i . k_j``. Dense emulation over query blocks (equal-step loss only, not speed): causal,
    in-document, with the banded Inkling bias on the softmax scores."""
    out_sharding = named_sharding_of(q)

    def compute(q, k, v, r, rel_bias, segment_ids):
        b, seq, h, d = q.shape
        out_dtype = v.dtype
        k = jnp.broadcast_to(k, (b, seq, h, k.shape[-1])).astype(jnp.float32)
        v = jnp.broadcast_to(v, (b, seq, h, v.shape[-1])).astype(jnp.float32)
        starts = _document_starts(segment_ids)
        keys = jnp.arange(seq)
        block = min(PARALLAX_QUERY_BLOCK, seq)

        @jax.checkpoint
        def one(i):
            rows = i * block + jnp.arange(block)
            qb = jax.lax.dynamic_slice_in_dim(q, i * block, block, axis=1).astype(jnp.float32)
            rb = jax.lax.dynamic_slice_in_dim(r, i * block, block, axis=1).astype(jnp.float32)
            s1 = jnp.einsum("bqhd,bkhd->bhqk", qb, k) / math.sqrt(d)
            if rel_bias is not None:
                band = jax.lax.dynamic_slice_in_dim(rel_bias, i * block, block, axis=2).astype(jnp.float32)
                column = (
                    keys[None, :] - (rows[:, None] // REL_BIAS_BLOCK) * REL_BIAS_BLOCK + rel_extent_of_band(rel_bias)
                )
                in_band = (column >= 0) & (column < band.shape[-1])
                idx = jnp.broadcast_to(jnp.clip(column, 0, band.shape[-1] - 1), (b, h, block, seq))
                s1 = s1 + jnp.where(in_band, jnp.take_along_axis(band, idx, axis=-1), 0.0)
            start = jax.lax.dynamic_slice_in_dim(starts, i * block, block, axis=1)  # [B, Q]
            ok = (keys[None, None, :] <= rows[None, :, None]) & (keys[None, None, :] >= start[:, :, None])
            p = jax.nn.softmax(jnp.where(ok[:, None], s1, -jnp.inf), axis=-1)
            p = jnp.where(ok[:, None], p, 0.0)
            ps = p * jnp.einsum("bqhd,bkhd->bhqk", rb, k)
            o1 = jnp.einsum("bhqk,bkhd->bqhd", p, v)
            o2 = jnp.einsum("bhqk,bkhd->bqhd", ps, v)
            mean_s = jnp.sum(ps, axis=-1).transpose(0, 2, 1)[..., None]  # [B, Q, H, 1]
            return o1 * (1.0 + mean_s) - o2

        out = jax.lax.map(one, jnp.arange(seq // block))  # [N, B, Q, H, E]
        return jnp.moveaxis(out, 0, 1).reshape(b, seq, h, -1).astype(out_dtype)

    if out_sharding is None:
        return compute(q, k, v, r, rel_bias, segment_ids)
    # pyrefly: ignore[bad-assignment]
    wrapped: Callable[..., jax.Array] = auto_axes(compute, out_sharding=out_sharding)
    return wrapped(q, k, v, r, rel_bias, segment_ids)
