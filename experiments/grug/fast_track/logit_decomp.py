# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-token decomposition of the final logits into the residual-stream head and the ``lm_head_extra_dim`` slice.

With ``lm_head_extra_dim`` the lm_head reads ``[final-normed stream (D) | normed extra slice (W)]``, so the
pre-cap logits split exactly into ``main = h W[:D]`` and ``extra = e W[D:]``. The stats ask what the extra term
does: raise the target (prediction), lower the stream head's strongest wrong candidates (suppression), move a few
tokens or many (sparsity), rescale the stream head's distribution (alignment / entropy), follow the unigram
frequencies, or push on tokens seen just before (copy suppression).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from levanter.grug.attention import AttentionMask

from experiments.grug.fast_track.model import Transformer, _batch_spec, _logit_cap

LOGIT_DECOMP_FIELDS = (
    "loss_full",  # next-token loss with both terms
    "loss_main",  # with the extra term removed
    "entropy_full",
    "entropy_main",
    "extra_target",  # extra logit of the target, centered over the vocab
    "extra_main_top10",  # mean centered extra logit on the stream head's top-10 wrong candidates
    "extra_recent",  # mean centered extra logit on the previous _RECENT tokens' ids (same document)
    "extra_num_gt1",  # vocab entries whose centered extra logit exceeds 1 in magnitude
    "extra_top100_energy",  # share of the centered extra term's squared norm in its top 100 entries
    "cos_extra_main",  # cosine of the centered extra and stream-head logits
    "cos_extra_unigram",  # cosine of the centered extra logits and centered unigram log-frequencies
    "extra_rms",  # RMS of the centered extra logits
    "main_rms",
    "valid",  # 1 where the position has a next token in the same document
)
_RECENT = 64
_TOP_COMPETITORS = 10


def _cap(logits: jax.Array, cap) -> jax.Array:
    if cap is None:
        return logits
    if isinstance(cap, tuple):
        a, b, c = cap
        return a * jax.nn.sigmoid((logits + b) / c)
    return jnp.tanh(logits / cap) * cap


def _cos(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.sum(a * b, axis=-1) / (jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + 1e-9)


def logit_decomp_stats(
    model: Transformer, ids: Int[Array, "B S"], segments: Int[Array, "B S"], unigram: Float[Array, " V"]
) -> Float[Array, "B S F"]:
    """``[B, S, len(LOGIT_DECOMP_FIELDS)]`` per-token stats (``segments`` -1 marks padding)."""
    cfg = model.config
    if not cfg.lm_head_extra_dim or cfg.lm_head_prototypes != 1:
        raise ValueError("logit_decomp needs lm_head_extra_dim and a single lm_head vector per token")
    hidden, _ = model(ids, mask=AttentionMask.causal().with_segment_ids(segments))
    head_in, head = model._lm_head_operands(hidden, ids)
    d = cfg.hidden_dim
    if head_in.shape[-1] != d + cfg.lm_head_extra_dim:
        raise ValueError("logit_decomp supports a head reading exactly the stream and the extra slice")
    head = head.astype(jnp.float32)
    head = jax.sharding.reshard(head, jax.sharding.PartitionSpec(None, None))
    spec = _batch_spec()
    main = jnp.einsum("bsd,dv->bsv", head_in[..., :d].astype(jnp.float32), head[:d], out_sharding=spec)
    extra = jnp.einsum("bsd,dv->bsv", head_in[..., d:].astype(jnp.float32), head[d:], out_sharding=spec)
    cap = _logit_cap(cfg)
    log_full = jax.nn.log_softmax(_cap(main + extra, cap), axis=-1)
    log_main = jax.nn.log_softmax(_cap(main, cap), axis=-1)

    labels = jnp.pad(ids[:, 1:], ((0, 0), (0, 1)))
    valid = (segments >= 0) & (jnp.pad(segments[:, 1:], ((0, 0), (0, 1)), constant_values=-1) == segments)
    pick = lambda x, idx: jnp.take_along_axis(x, idx[..., None], axis=-1)[..., 0]  # noqa: E731

    extra_c = extra - jnp.mean(extra, axis=-1, keepdims=True)
    main_c = main - jnp.mean(main, axis=-1, keepdims=True)
    uni_c = unigram - jnp.mean(unigram)
    wrong = jnp.where(jnp.arange(main.shape[-1]) == labels[..., None], -jnp.inf, main)
    _, competitors = jax.lax.top_k(wrong, min(_TOP_COMPETITORS, wrong.shape[-1] - 1))
    on_competitors = jnp.mean(jnp.take_along_axis(extra_c, competitors, axis=-1), axis=-1)

    # The previous _RECENT tokens of the same document (excluding the current token's own target).
    seq = ids.shape[1]
    lags = jnp.arange(1, _RECENT + 1)
    positions = jnp.arange(seq)[:, None] - lags[None, :] + 1  # include the current token itself (lag 0 -> +1 shift)
    in_range = positions >= 0
    positions = jnp.clip(positions, 0, seq - 1)
    recent_ids = jnp.take(ids, positions, axis=1)  # [B, S, R]
    same_doc = jnp.take(segments, positions, axis=1) == segments[..., None]
    recent_mask = (in_range[None] & same_doc).astype(jnp.float32)
    on_recent = jnp.sum(jnp.take_along_axis(extra_c, recent_ids, axis=-1) * recent_mask, axis=-1) / jnp.maximum(
        jnp.sum(recent_mask, axis=-1), 1.0
    )

    energy = jnp.square(extra_c)
    top_energy = jnp.sum(jax.lax.top_k(energy, min(100, energy.shape[-1]))[0], axis=-1) / (
        jnp.sum(energy, axis=-1) + 1e-12
    )
    fields = [
        -pick(log_full, labels),
        -pick(log_main, labels),
        -jnp.sum(jnp.exp(log_full) * log_full, axis=-1),
        -jnp.sum(jnp.exp(log_main) * log_main, axis=-1),
        pick(extra_c, labels),
        on_competitors,
        on_recent,
        jnp.sum((jnp.abs(extra_c) > 1.0).astype(jnp.float32), axis=-1),
        top_energy,
        _cos(extra_c, main_c),
        _cos(extra_c, jnp.broadcast_to(uni_c, extra_c.shape)),
        jnp.sqrt(jnp.mean(jnp.square(extra_c), axis=-1)),
        jnp.sqrt(jnp.mean(jnp.square(main_c), axis=-1)),
        valid.astype(jnp.float32),
    ]
    return jnp.stack([jax.sharding.reshard(f.astype(jnp.float32), spec) for f in fields], axis=-1)
