# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact-match pointer mixture (``pointer_mixture``) on the output.

Each position ``t`` gathers ``POINTER_SLOTS`` candidate next tokens from earlier in its packed document: for
``n = 1`` and ``n = 2``, the token ``x[s+1]`` after each of the ``MATCHES_PER_ORDER`` most recent positions
``s < t`` whose last ``n`` tokens equal the ``n`` tokens ending at ``t``. The next-token distribution is
``p(y) = (1 - G) p_vocab(y) + open * sum_k pi_k [c_k = y]`` with ``(pi_0, pi_1..pi_K) = softmax(0, l_1..l_K)``
over the candidates that exist, ``l_k = h . w_k + 8 b[k, a_k] - 2`` (``h`` the lm_head input, ``a_k`` the number
of other slots proposing the same token), ``G = open * sum_k pi_k`` and ``open = clip(8 o, 0, 1)``. ``w``, ``b``
and ``o`` start at zero, so the model starts as the control.

The loss is ``ce - log((1 - G) + M exp(ce))``, with ``ce`` the fused kernel's per-token cross-entropy (with the
z-loss in training) and ``M`` the pointer mass on the label. The matches are integer work on the token ids: one
batched sort of (document, key, position) per order gives each position's most recent earlier match, and three
more link hops give the older ones.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard

POINTER_SLOTS = 8
MATCHES_PER_ORDER = 4
_LOGIT_BIAS = -2.0
_AGREEMENT_SCALE = 8.0
_OPEN_SCALE = 8.0
_MASKED = -1e30
# exp(ce) stays finite in float32 below this; a token this unlikely never occurs in practice.
_MAX_EXP_CE = 80.0


class PointerHead(eqx.Module):
    pointer_w: jax.Array  # [D, K]
    pointer_b: jax.Array  # [K, K], indexed by slot and agreement count
    pointer_open: jax.Array  # []

    @staticmethod
    def init(hidden_dim: int) -> "PointerHead":
        return PointerHead(
            pointer_w=reshard(jnp.zeros((hidden_dim, POINTER_SLOTS), jnp.float32), P(None, None)),
            pointer_b=reshard(jnp.zeros((POINTER_SLOTS, POINTER_SLOTS), jnp.float32), P(None, None)),
            pointer_open=jnp.zeros((), jnp.float32),
        )


def _previous_match(segments: jax.Array, first: jax.Array, second: jax.Array) -> jax.Array:
    """One row: the most recent earlier position with the same ``(segment, first, second)``, or -1."""
    positions = jnp.arange(segments.shape[0], dtype=jnp.int32)
    seg, a, b, order = jax.lax.sort((segments, first, second, positions), num_keys=4)
    same = (seg[1:] == seg[:-1]) & (a[1:] == a[:-1]) & (b[1:] == b[:-1])
    previous = jnp.concatenate([jnp.full((1,), -1, jnp.int32), jnp.where(same, order[:-1], -1)])
    return jnp.zeros_like(positions).at[order].set(previous)


def _row_candidates(tokens: jax.Array, segments: jax.Array) -> jax.Array:
    """One row's ``[S, POINTER_SLOTS]`` candidate tokens (unigram matches, then bigram), -1 where none exists."""
    positions = jnp.arange(tokens.shape[0], dtype=jnp.int32)
    starts = jnp.concatenate([jnp.ones((1,), bool), segments[1:] != segments[:-1]])
    # A document start has no bigram: a distinct negative key per position never matches.
    previous_token = jnp.where(starts, -1 - positions, jnp.concatenate([jnp.zeros((1,), tokens.dtype), tokens[:-1]]))
    slots = []
    for first in (jnp.zeros_like(tokens), previous_token):
        link = _previous_match(segments, first, tokens)
        match = link
        for _ in range(MATCHES_PER_ORDER):
            slots.append(match)
            match = jnp.where(match >= 0, link[jnp.maximum(match, 0)], -1)
    matches = jnp.stack(slots, -1)
    return jnp.where(matches >= 0, tokens[matches + 1], -1)


def pointer_candidates(tokens: jax.Array, segments: jax.Array, batch_axes: tuple[str, ...]) -> jax.Array:
    """``[B, S, POINTER_SLOTS]`` candidate next tokens for batch-sharded ``[B, S]`` token ids and document ids."""
    spec = P(batch_axes, None)
    tokens, segments = reshard(tokens.astype(jnp.int32), spec), reshard(segments.astype(jnp.int32), spec)
    return jax.shard_map(
        jax.vmap(_row_candidates),
        mesh=get_abstract_mesh(),
        in_specs=(spec, spec),
        out_specs=P(batch_axes, None, None),
        check_vma=False,
    )(tokens, segments)


def pointer_mixture_nll(
    head: PointerHead, slot_logits: jax.Array, candidates: jax.Array, labels: jax.Array, ce: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-token mixture NLL, pointer gate ``G`` and label mass ``M``, each ``[B, S]``. ``slot_logits`` is
    ``h . w`` ``[B, S, K]`` and ``ce`` the vocabulary head's per-token loss."""
    head = jax.tree.map(lambda x: x.astype(jnp.float32), head)
    valid = candidates >= 0
    # a_k: the other slots proposing slot k's token (a missing slot's -1 matches only other missing slots).
    agreement = jnp.sum(candidates[..., :, None] == candidates[..., None, :], -1) - 1
    agreement_bias = jnp.sum(jax.nn.one_hot(agreement, POINTER_SLOTS) * head.pointer_b, -1)
    logits = slot_logits.astype(jnp.float32) + _AGREEMENT_SCALE * agreement_bias + _LOGIT_BIAS
    logits = jnp.where(valid, logits, _MASKED)
    log_normalizer = jnp.logaddexp(0.0, jax.nn.logsumexp(logits, -1))
    pi = jnp.exp(logits - log_normalizer[..., None])
    open_gate = jnp.clip(_OPEN_SCALE * head.pointer_open, 0.0, 1.0)
    gate = open_gate * jnp.sum(pi, -1)
    label_mass = open_gate * jnp.sum(jnp.where(candidates == labels[..., None], pi, 0.0), -1)
    ce = ce.astype(jnp.float32)
    nll = ce - jnp.log1p(label_mass * jnp.exp(jnp.minimum(ce, _MAX_EXP_CE)) - gate)
    return nll, gate, label_mass
