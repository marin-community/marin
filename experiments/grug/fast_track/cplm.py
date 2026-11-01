# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPLM Plus copy head (``cplm_plus``): a Copy-Paste Language Model mixture output head with predecessor keys and
case-class emissions, ported from PRs #10260 (CPLM, arXiv:2610.19247) and #10678 (Plus).

Each next-token distribution mixes the vocabulary softmax with a copy distribution over earlier positions of the
same packed document (including the current one) plus a sink:
``p = (1 - c + c * a_sink) * softmax(z) + c * sum_s a_s * e_s``, with copy gate
``c = sigmoid(g - logsumexp(z))`` competing with the whole vocabulary. Copy keys add the gated predecessor state
``k_s = h_s W_key + gain * (h_(s-1) W_prev)``. A source emits its own token with probability ``1 - alpha`` or a token
of its case class (``case_class_tables``), distributed by the vocabulary logits within the class, with probability
``alpha = sigmoid(h_t w_variant + b_variant)``.

The loss is target-only: query tiles bound the ``[B, tile, S]`` score storage and are rematerialized in the backward
pass; arithmetic is quadratic in sequence length. The vocabulary logits are uncapped and carry the recipe's z-loss
(``z_loss * logsumexp(z)^2``); only the copy-gate logit is soft-capped, inside this head.
"""

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from levanter.tokenizers import MarinTokenizer

_INIT_STD = 0.02
_COPY_KEY_SALT = 261019247
_PREV_KEY_SALT = 261019248
_VARIANT_BIAS_INIT = -3.0
_GATE_CAP = 10.0
_QUERY_TILE = 128
# Finite sentinel: an all-masked logsumexp stays differentiable when the target never occurs in the document.
_MASKED = -1e30


@dataclasses.dataclass(frozen=True)
class CaseTables:
    """``classes[v]`` is the smallest id in ``v``'s case class; ``members[v]`` lists the class in increasing id
    order, padded with -1 (so it always contains ``v``)."""

    classes: np.ndarray
    members: np.ndarray


_BOUND_TABLES: CaseTables | None = None


def case_key(text: str) -> str | None:
    """Class key of a decoded token, or None for a singleton: strip leading spaces, then leading underscores, and
    lowercase what remains if it is nonempty ASCII alphanumeric with a letter (``View``, `` view``, ``_VIEW``)."""
    rest = text.lstrip(" ").lstrip("_")
    if rest and rest.isascii() and rest.isalnum() and any(c.isalpha() for c in rest):
        return rest.lower()
    return None


def case_tables_from_pieces(pieces: list[str]) -> CaseTables:
    """Case classes of a vocabulary given each id's decoded text; the width is the largest class."""
    groups: dict[str, list[int]] = {}
    for index, piece in enumerate(pieces):
        key = case_key(piece)
        if key is not None:
            groups.setdefault(key, []).append(index)
    width = max((len(ids) for ids in groups.values()), default=1)
    classes = np.arange(len(pieces), dtype=np.int32)
    members = np.full((len(pieces), width), -1, dtype=np.int32)
    members[:, 0] = classes
    for ids in groups.values():
        for i in ids:
            classes[i] = ids[0]
            members[i, : len(ids)] = ids
    return CaseTables(classes, members)


def case_class_tables(tokenizer: MarinTokenizer, vocab_size: int) -> CaseTables:
    """Case classes of the first ``vocab_size`` ids of ``tokenizer``; depends only on its vocabulary."""
    special = set(tokenizer.all_special_ids)
    pieces = ["" if i in special else tokenizer.decode([i]) for i in range(vocab_size)]
    return case_tables_from_pieces(pieces)


def bind_case_tables(tables: CaseTables) -> None:
    """Set the case tables every ``cplm_plus`` loss in this process reads (training and evaluation alike)."""
    global _BOUND_TABLES
    _BOUND_TABLES = tables


def bound_case_tables() -> CaseTables:
    if _BOUND_TABLES is None:
        raise ValueError("cplm_plus needs bind_case_tables(case_class_tables(tokenizer, vocab_size)) first")
    return _BOUND_TABLES


class CopyHead(eqx.Module):
    query: jax.Array  # [D, D]
    key: jax.Array  # [D, D]
    prev_key: jax.Array  # [D, D], the predecessor-state key projection
    gate: jax.Array  # [D], the copy-gate logit
    sink: jax.Array  # [D]
    prev_gain: jax.Array  # [D], zero at init so the predecessor term starts off
    variant: jax.Array  # [D], the case-class emission gate
    variant_bias: jax.Array  # []

    @staticmethod
    def init(hidden_dim: int, *, key: jax.Array) -> "CopyHead":
        # Independent folded streams, so the backbone keeps its initialization.
        keys = jax.random.split(jax.random.fold_in(key, _COPY_KEY_SALT), 4)
        d = hidden_dim
        query, key_proj, gate, sink = (
            reshard(_INIT_STD * jax.random.normal(k, shape), P(*(None,) * len(shape)))
            for k, shape in zip(keys, ((d, d), (d, d), (d,), (d,)), strict=True)
        )
        prev_key = _INIT_STD * jax.random.normal(jax.random.fold_in(key, _PREV_KEY_SALT), (d, d))
        return CopyHead(
            query=query,
            key=key_proj,
            prev_key=reshard(prev_key, P(None, None)),
            gate=gate,
            sink=sink,
            prev_gain=reshard(jnp.zeros((d,)), P(None)),
            variant=reshard(jnp.zeros((d,)), P(None)),
            variant_bias=jnp.full((), _VARIANT_BIAS_INIT),
        )


def predecessor_states(hidden: jax.Array, segments: jax.Array) -> jax.Array:
    """``h_(s-1)``, zero at sequence and document starts."""
    previous = jnp.pad(hidden[:, :-1], ((0, 0), (1, 0), (0, 0)))
    same = jnp.pad(segments[:, 1:] == segments[:, :-1], ((0, 0), (1, 0)))
    return jnp.where(same[..., None], previous, jnp.zeros_like(previous))


def mixture_nll(
    native_nll: jax.Array, log_partition: jax.Array, copy_logit: jax.Array, sink_logp: jax.Array, target_logp: jax.Array
) -> jax.Array:
    """Mix the vocabulary head, the copy target mass and the sink fallback in log space; all ``[B, S]``.

    The copy gate competes with the whole vocabulary, ``sigmoid(copy_logit - log Z)``; log Z stays attached to
    autodiff (detaching it changes the vocabulary head's objective)."""
    log_copy = -jax.nn.softplus(log_partition - copy_logit)
    log_native = -jax.nn.softplus(copy_logit - log_partition)
    native_mass = jnp.logaddexp(log_native, log_copy + sink_logp)
    return -jnp.logaddexp(native_mass - native_nll, log_copy + target_logp)


def local_plus_nll(
    hidden: jax.Array,
    lm_head: jax.Array,
    head: CopyHead,
    tokens: jax.Array,
    labels: jax.Array,
    segments: jax.Array,
    native_nll: jax.Array,
    classes: jax.Array,
    members: jax.Array,
    *,
    tile: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One batch shard with the full context and vocabulary. ``native_nll`` is the vocabulary head's per-token NLL.
    Returns per-token mixture NLL, vocabulary log partition, copy gate and effective copy mass, each ``[B, S]``."""
    batch, length, dim = hidden.shape
    if length % tile:
        raise ValueError(f"copy query tile {tile} must divide the sequence length {length}")
    f32 = jnp.float32
    head = jax.tree.map(lambda x: x.astype(f32), head)
    head_rows = lm_head.T  # [V, D]: a token's vocabulary column as a row
    target_logit = jnp.einsum("bsd,bsd->bs", hidden, head_rows[labels].astype(hidden.dtype), preferred_element_type=f32)
    log_partition = native_nll + target_logit
    q = jnp.einsum("bsd,dh->bsh", hidden, head.query.astype(hidden.dtype), preferred_element_type=f32) / math.sqrt(dim)
    k = jnp.einsum("bsd,dh->bsh", hidden, head.key.astype(hidden.dtype), preferred_element_type=f32)
    previous = jnp.einsum(
        "bsd,dh->bsh",
        predecessor_states(hidden, segments),
        head.prev_key.astype(hidden.dtype),
        preferred_element_type=f32,
    )
    k = k + head.prev_gain * previous
    copy_logit = _GATE_CAP * jnp.tanh(jnp.einsum("bsd,d->bs", hidden.astype(f32), head.gate) / _GATE_CAP)
    variant_logit = jnp.einsum("bsd,d->bs", hidden.astype(f32), head.variant) + head.variant_bias
    token_classes, label_classes = classes[tokens], classes[labels]
    positions = jnp.arange(length)

    @jax.checkpoint
    def chunk(_, index):
        start = index * tile

        def rows(x):
            return jax.lax.dynamic_slice_in_dim(x, start, tile, axis=1)

        query, segment, target, target_class, x = (rows(a) for a in (q, segments, labels, label_classes, hidden))
        scores = jnp.einsum("bqd,bkd->bqk", query, k)
        allowed = (positions[None, None, :] <= (start + jnp.arange(tile))[None, :, None]) & (
            segments[:, None, :] == segment[:, :, None]
        )
        scores = jnp.where(allowed, scores, _MASKED)
        sink_score = jnp.einsum("bqd,d->bq", query, head.sink)
        normalizer = jnp.logaddexp(jax.nn.logsumexp(scores, -1), sink_score)
        exact = allowed & (tokens[:, None, :] == target[:, :, None])
        same_class = allowed & (token_classes[:, None, :] == target_class[:, :, None])
        exact_lse = jax.nn.logsumexp(jnp.where(exact, scores, _MASKED), -1)
        class_lse = jax.nn.logsumexp(jnp.where(same_class, scores, _MASKED), -1)
        # The target's share of its class under the vocabulary logits restricted to the class.
        ids = members[target]  # [B, tile, W]
        columns = head_rows[jnp.maximum(ids, 0)].astype(x.dtype)  # [B, tile, W, D]
        class_logits = jnp.einsum("bqd,bqwd->bqw", x, columns, preferred_element_type=f32)
        class_logits = jnp.where(ids >= 0, class_logits, _MASKED)
        own = jnp.sum(jnp.where(ids == target[..., None], class_logits, 0.0), -1)
        in_class = own - jax.nn.logsumexp(class_logits, -1)
        return None, (sink_score - normalizer, exact_lse - normalizer, class_lse - normalizer, in_class)

    _, outputs = jax.lax.scan(chunk, None, jnp.arange(length // tile))
    sink_logp, exact_logp, class_logp, in_class = (x.transpose(1, 0, 2).reshape(batch, length) for x in outputs)
    target_logp = jnp.logaddexp(
        exact_logp - jax.nn.softplus(variant_logit), in_class + class_logp - jax.nn.softplus(-variant_logit)
    )
    nll = mixture_nll(native_nll, log_partition, copy_logit, sink_logp, target_logp)
    gate = jax.nn.sigmoid(copy_logit - log_partition)
    return nll, log_partition, gate, gate * -jnp.expm1(sink_logp)


def plus_nll(
    hidden: jax.Array,
    lm_head: jax.Array,
    head: CopyHead,
    tokens: jax.Array,
    labels: jax.Array,
    segments: jax.Array,
    native_nll: jax.Array,
    tables: CaseTables,
    batch_axes: tuple[str, ...],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """``local_plus_nll`` over the batch-sharded ``[B, S]`` inputs, with the heads replicated on every shard."""
    tile = math.gcd(hidden.shape[1], _QUERY_TILE)
    hs, ts = P(batch_axes, None, None), P(batch_axes, None)
    replicated = [reshard(x, P()) for x in (lm_head, jnp.asarray(tables.classes), jnp.asarray(tables.members))]
    lm_head, classes, members = replicated
    head = jax.tree.map(lambda x: reshard(x, P()), head)
    hidden = reshard(hidden, hs)
    tokens, labels, segments, native_nll = (reshard(x, ts) for x in (tokens, labels, segments, native_nll))

    def run(*local):
        return local_plus_nll(*local, tile=tile)

    return jax.shard_map(
        run,
        mesh=get_abstract_mesh(),
        in_specs=(hs, P(), P(), ts, ts, ts, ts, P(), P()),
        out_specs=(ts,) * 4,
        check_vma=False,
    )(hidden, lm_head, head, tokens, labels, segments, native_nll, classes, members)
