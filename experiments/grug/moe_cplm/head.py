# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Target-only Copy-Paste Language Model loss (arXiv:2610.19247, equations 1-7).

Query tiles bound score storage and are rematerialized in the backward pass.
Arithmetic remains quadratic in sequence length. These functions operate on
one shard with a complete vocabulary and context; shard the batch externally.
"""

import jax
import jax.numpy as jnp


def copy_log_probabilities(
    queries: jax.Array,
    keys: jax.Array,
    sink: jax.Array,
    tokens: jax.Array,
    labels: jax.Array,
    segments: jax.Array,
    *,
    query_tile: int,
) -> tuple[jax.Array, jax.Array]:
    """Return sink and target log probabilities, summing duplicate token mass.

    Queries and keys have shape (batch, sequence, hidden); sink has shape
    (hidden,). Integer tokens, next-token labels and document IDs have shape
    (batch, sequence). Copy positions include the observed query token itself,
    exclude future positions, and must belong to the same packed document.
    """
    batch, length, dim = queries.shape
    if query_tile <= 0 or length % query_tile:
        raise ValueError("query_tile must be positive and divide sequence length")
    positions = jnp.arange(length)
    queries = queries.astype(jnp.float32) / dim**0.5
    keys, sink = keys.astype(jnp.float32), sink.astype(jnp.float32)

    @jax.checkpoint
    def chunk(carry, index):
        start = index * query_tile
        query = jax.lax.dynamic_slice_in_dim(queries, start, query_tile, axis=1)
        segment = jax.lax.dynamic_slice_in_dim(segments, start, query_tile, axis=1)
        target = jax.lax.dynamic_slice_in_dim(labels, start, query_tile, axis=1)
        scores = jnp.einsum("bqd,bkd->bqk", query, keys)
        allowed = (positions[None, None, :] <= (start + jnp.arange(query_tile))[None, :, None]) & (
            segments[:, None, :] == segment[:, :, None]
        )
        scores = jnp.where(allowed, scores, -1e30)
        sink_score = jnp.einsum("bqd,d->bq", query, sink)
        normalizer = jnp.logaddexp(jax.nn.logsumexp(scores, axis=-1), sink_score)
        matches = allowed & (tokens[:, None, :] == target[:, :, None])
        # Finite sentinel avoids undefined gradients of an all-minus-inf
        # logsumexp when the target never occurs in the visible document.
        target_score = jax.nn.logsumexp(jnp.where(matches, scores, -1e30), axis=-1)
        return carry, (sink_score - normalizer, target_score - normalizer)

    _, (sink_logp, target_logp) = jax.lax.scan(chunk, None, jnp.arange(length // query_tile))
    return (
        sink_logp.transpose(1, 0, 2).reshape(batch, length),
        target_logp.transpose(1, 0, 2).reshape(batch, length),
    )


def mixture_nll(
    native_nll: jax.Array,
    log_partition: jax.Array,
    copy_logit: jax.Array,
    sink_logp: jax.Array,
    target_logp: jax.Array,
) -> jax.Array:
    """Mix the vocabulary head, copy target mass and sink fallback in log space.

    The copy gate competes with the whole vocabulary: sigmoid(copy_logit-log Z).
    All inputs and the result have shape (batch, sequence). Keep log Z attached
    to autodiff; detaching it changes the vocabulary-head training objective.
    """
    log_copy = -jax.nn.softplus(log_partition - copy_logit)
    log_native = -jax.nn.softplus(copy_logit - log_partition)
    native_mass = jnp.logaddexp(log_native, log_copy + sink_logp)
    return -jnp.logaddexp(native_mass - native_nll, log_copy + target_logp)


def copy_paste_nll(
    hidden: jax.Array,
    query_weight: jax.Array,
    key_weight: jax.Array,
    copy_weight: jax.Array,
    sink: jax.Array,
    tokens: jax.Array,
    labels: jax.Array,
    segments: jax.Array,
    native_nll: jax.Array,
    log_partition: jax.Array,
    *,
    query_tile: int,
) -> jax.Array:
    """Compute unreduced CPLM NLL from native vocabulary NLL and log partition.

    The head adds two (hidden, hidden) matrices and two (hidden,) vectors.
    It adds no position embeddings or query/key normalization. Loss weighting
    and auxiliary router objectives remain the caller's responsibility.
    """
    queries = jnp.einsum("bsd,dh->bsh", hidden, query_weight, preferred_element_type=jnp.float32)
    keys = jnp.einsum("bsd,dh->bsh", hidden, key_weight, preferred_element_type=jnp.float32)
    copy_logit = jnp.sum(hidden.astype(jnp.float32) * copy_weight.astype(jnp.float32), -1)
    sink_logp, target_logp = copy_log_probabilities(queries, keys, sink, tokens, labels, segments, query_tile=query_tile)
    return mixture_nll(native_nll, log_partition, copy_logit, sink_logp, target_logp)


