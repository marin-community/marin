# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Document n-gram count cache (``ngram_cache_mixture``) mixed into the output.

For position ``t`` with label ``y``, over earlier positions ``s < t`` of the same packed document: ``c1`` counts
``x_s = x_t`` and ``n1`` those with ``x_{s+1} = y``; ``c2`` counts ``(x_{s-1}, x_s) = (x_{t-1}, x_t)`` and ``n2``
those with ``x_{s+1} = y``. ``q_n = n_n / c_n``. The next-token distribution is
``p(y) = (1 - G) p_vocab(y) + open * (pi_1 q1(y) + pi_2 q2(y))`` with ``(pi_0, pi_1, pi_2) = softmax(0, l_1, l_2)``
over the orders with ``c_n > 0``, ``l_n = h . w_n + 8 b_n log2(1 + c_n) / 12 - 2`` (``h`` the lm_head input),
``G = open * (pi_1 + pi_2)`` and ``open = clip(8 o, 0, 1)``. ``w``, ``b`` and ``o`` start at zero, so the model
starts as the control.

The loss is ``ce - log((1 - G) + M exp(ce))``, with ``ce`` the fused kernel's per-token cross-entropy (with the
z-loss in training) and ``M = open * (pi_1 q1(label) + pi_2 q2(label))``. The counts are integer work on the token
ids: one stable sort per key (token, pair, triple) gives each position's rank within its (document, key) run, and
``c1 = rank_token[t]``, ``c2 = rank_pair[t]``, ``n1 = rank_pair[t + 1]``, ``n2 = rank_triple[t + 1]``.
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard

NGRAM_ORDERS = 2
_LOGIT_BIAS = -2.0
_COUNT_SCALE = 8.0 / 12.0
_OPEN_SCALE = 8.0
_MASKED = -1e30
# exp(ce) stays finite in float32 below this; a token this unlikely never occurs in practice.
_MAX_EXP_CE = 80.0


class NgramCacheHead(eqx.Module):
    ngram_cache_w: jax.Array  # [D, 2]
    ngram_cache_b: jax.Array  # [2]
    ngram_cache_open: jax.Array  # []

    @staticmethod
    def init(hidden_dim: int) -> "NgramCacheHead":
        return NgramCacheHead(
            ngram_cache_w=reshard(jnp.zeros((hidden_dim, NGRAM_ORDERS), jnp.float32), P(None, None)),
            ngram_cache_b=reshard(jnp.zeros((NGRAM_ORDERS,), jnp.float32), P(None)),
            ngram_cache_open=jnp.zeros((), jnp.float32),
        )


class NgramCounts(NamedTuple):
    """Per-position ``[B, S]`` int32 counts of earlier same-document matches (``c``) and of those followed by the
    label (``n``), for unigram (``1``) and bigram (``2``) contexts."""

    c1: jax.Array
    n1: jax.Array
    c2: jax.Array
    n2: jax.Array


def _run_ranks(segments: jax.Array, high: jax.Array, low: jax.Array) -> jax.Array:
    """One row: each position's count of earlier positions in its document with the same ``(high, low)`` key."""
    positions = jnp.arange(segments.shape[0], dtype=jnp.int32)
    high, low, order = jax.lax.sort((high, low, positions), num_keys=3)
    seg = segments[order]
    breaks = jnp.concatenate(
        [jnp.ones((1,), bool), (high[1:] != high[:-1]) | (low[1:] != low[:-1]) | (seg[1:] != seg[:-1])]
    )
    run_start = jax.lax.cummax(jnp.where(breaks, positions, 0))
    return jnp.zeros_like(positions).at[order].set(positions - run_start)


def _row_counts(tokens: jax.Array, segments: jax.Array, vocab_size: int) -> NgramCounts:
    positions = jnp.arange(tokens.shape[0], dtype=jnp.int32)
    starts = jnp.concatenate([jnp.ones((1,), bool), segments[1:] != segments[:-1]])
    # A pair or triple that would cross a document start gets a distinct negative key that never matches.
    unique = -1 - positions
    previous = jnp.concatenate([jnp.zeros((1,), tokens.dtype), tokens[:-1]])
    pair = jnp.where(starts, unique, previous * vocab_size + tokens)
    previous_pair = jnp.concatenate([jnp.zeros((1,), pair.dtype), pair[:-1]])
    previous_start = jnp.concatenate([jnp.ones((1,), bool), starts[:-1]])
    triple_high = jnp.where(starts | previous_start, unique, previous_pair)
    zeros = jnp.zeros_like(tokens)
    rank_token = _run_ranks(segments, zeros, tokens)
    rank_pair = _run_ranks(segments, zeros, pair)
    rank_triple = _run_ranks(segments, triple_high, tokens)

    def next_position(rank: jax.Array) -> jax.Array:
        return jnp.concatenate([rank[1:], jnp.zeros((1,), rank.dtype)])

    return NgramCounts(c1=rank_token, n1=next_position(rank_pair), c2=rank_pair, n2=next_position(rank_triple))


def document_ngram_counts(
    tokens: jax.Array, segments: jax.Array, vocab_size: int, batch_axes: tuple[str, ...]
) -> NgramCounts:
    """``NgramCounts`` for batch-sharded ``[B, S]`` token ids and document ids."""
    if vocab_size * vocab_size >= 2**31:
        raise ValueError(f"ngram_cache_mixture keys pairs as int32 x_(u-1) * vocab + x_u; vocab {vocab_size} overflows")
    spec = P(batch_axes, None)
    tokens, segments = reshard(tokens.astype(jnp.int32), spec), reshard(segments.astype(jnp.int32), spec)
    return jax.shard_map(
        jax.vmap(lambda t, s: _row_counts(t, s, vocab_size)),
        mesh=get_abstract_mesh(),
        in_specs=(spec, spec),
        out_specs=spec,
        check_vma=False,
    )(tokens, segments)


def ngram_cache_mixture_nll(
    head: NgramCacheHead, order_logits: jax.Array, counts: NgramCounts, ce: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-token mixture NLL, cache gate ``G`` and label mass ``M``, each ``[B, S]``. ``order_logits`` is ``h . w``
    ``[B, S, 2]`` and ``ce`` the vocabulary head's per-token loss."""
    head = jax.tree.map(lambda x: x.astype(jnp.float32), head)
    context = jnp.stack([counts.c1, counts.c2], -1).astype(jnp.float32)
    follow = jnp.stack([counts.n1, counts.n2], -1).astype(jnp.float32)
    valid = context > 0
    logits = order_logits.astype(jnp.float32) + _COUNT_SCALE * head.ngram_cache_b * jnp.log2(1.0 + context)
    logits = jnp.where(valid, logits + _LOGIT_BIAS, _MASKED)
    log_normalizer = jnp.logaddexp(0.0, jax.nn.logsumexp(logits, -1))
    pi = jnp.exp(logits - log_normalizer[..., None])
    open_gate = jnp.clip(_OPEN_SCALE * head.ngram_cache_open, 0.0, 1.0)
    gate = open_gate * jnp.sum(pi, -1)
    label_mass = open_gate * jnp.sum(pi * follow / jnp.maximum(context, 1.0), -1)
    ce = ce.astype(jnp.float32)
    nll = ce - jnp.log1p(label_mass * jnp.exp(jnp.minimum(ce, _MAX_EXP_CE)) - gate)
    return nll, gate, label_mass
