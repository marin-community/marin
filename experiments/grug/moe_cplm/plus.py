# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Belay-CPLM-plus: predecessor keys and normalized case-class emissions.

Composition of the frozen mechanisms in astra-d8-research issues 230 and 220.
The predecessor is masked at packed-document boundaries. Class tables depend
only on the frozen tokenizer; they are constants, never learned parameters.
"""

import dataclasses

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from experiments.grug.fast_track import model
from experiments.grug.moe_cplm.belay import CopyPasteBelay, capped, vocabulary_statistics
from experiments.grug.moe_cplm.head import mixture_nll

CASE_TABLES = None


def predecessor_states(hidden, segments):
    previous = jnp.pad(hidden[:, :-1], ((0, 0), (1, 0), (0, 0)))
    same = jnp.pad(segments[:, 1:] == segments[:, :-1], ((0, 0), (1, 0)))
    return jnp.where(same[..., None], previous, jnp.zeros_like(previous))


def class_copy_statistics(q, k, sink, tokens, labels, classes, segments, *, tile=128):
    batch, length, dim = q.shape
    if length % tile:
        raise ValueError("Copy query tile must divide sequence length")
    positions = jnp.arange(length)
    q = q.astype(jnp.float32) / dim**0.5
    k, sink = k.astype(jnp.float32), sink.astype(jnp.float32)
    token_classes, label_classes = classes[tokens], classes[labels]

    @jax.checkpoint
    def chunk(_, index):
        start = index * tile
        query = jax.lax.dynamic_slice_in_dim(q, start, tile, axis=1)
        segment = jax.lax.dynamic_slice_in_dim(segments, start, tile, axis=1)
        target = jax.lax.dynamic_slice_in_dim(labels, start, tile, axis=1)
        target_class = jax.lax.dynamic_slice_in_dim(label_classes, start, tile, axis=1)
        scores = jnp.einsum("bqd,bkd->bqk", query, k)
        allowed = (positions[None, None, :] <= (start + jnp.arange(tile))[None, :, None]) & (
            segments[:, None, :] == segment[:, :, None]
        )
        scores = jnp.where(allowed, scores, -1e30)
        sink_score = jnp.einsum("bqd,d->bq", query, sink)
        normalizer = jnp.logaddexp(jax.nn.logsumexp(scores, -1), sink_score)
        exact = allowed & (tokens[:, None, :] == target[:, :, None])
        same_class = allowed & (token_classes[:, None, :] == target_class[:, :, None])
        exact_lse = jax.nn.logsumexp(jnp.where(exact, scores, -1e30), -1)
        class_lse = jax.nn.logsumexp(jnp.where(same_class, scores, -1e30), -1)
        return None, (sink_score - normalizer, exact_lse - normalizer, class_lse - normalizer)

    _, outputs = jax.lax.scan(chunk, None, jnp.arange(length // tile))
    return tuple(x.transpose(1, 0, 2).reshape(batch, length) for x in outputs)


def class_target_logp(x, head, labels, members):
    ids = members[labels]
    columns = jnp.take(head, jnp.maximum(ids, 0), axis=1)
    logits = capped(jnp.einsum("nd,dnk->nk", x, columns, preferred_element_type=jnp.float32))
    logits = jnp.where(ids >= 0, logits, -1e30)
    own = jnp.sum(jnp.where(ids == labels[:, None], logits, 0.0), -1)
    return own - jax.nn.logsumexp(logits, -1)


def local_head(
    hidden,
    head,
    qw,
    kw,
    gw,
    sink,
    tokens,
    labels,
    segments,
    prev_weight,
    prev_gain,
    variant_weight,
    variant_bias,
    classes,
    members,
    *,
    tile=128,
    reference=False,
):
    batch, length, dim = hidden.shape
    ce, z = vocabulary_statistics(hidden, head, labels, reference=reference)
    q = jnp.einsum("bsd,dh->bsh", hidden, qw, preferred_element_type=jnp.float32)
    k = jnp.einsum("bsd,dh->bsh", hidden, kw, preferred_element_type=jnp.float32)
    previous = jnp.einsum(
        "bsd,dh->bsh", predecessor_states(hidden, segments), prev_weight, preferred_element_type=jnp.float32
    )
    k = k + prev_gain.astype(jnp.float32) * previous
    g = capped(jnp.sum(hidden.astype(jnp.float32) * gw.astype(jnp.float32), -1))
    sink_logp, exact_logp, class_logp = class_copy_statistics(q, k, sink, tokens, labels, classes, segments, tile=tile)
    u = jnp.sum(hidden.astype(jnp.float32) * variant_weight.astype(jnp.float32), -1) + variant_bias
    pi = class_target_logp(hidden.reshape(-1, dim), head, labels.reshape(-1), members).reshape(batch, length)
    target_logp = jnp.logaddexp(exact_logp - jax.nn.softplus(u), pi + class_logp - jax.nn.softplus(-u))
    nll = mixture_nll(ce, z, g, sink_logp, target_logp)
    gate = jax.nn.sigmoid(g - z)
    return nll, ce, gate, gate * -jnp.expm1(sink_logp)


class BelayCPLMPlus(CopyPasteBelay):
    cp_prev_key: jax.Array
    cp_prev_gain: jax.Array
    cp_variant: jax.Array
    cp_variant_bias: jax.Array

    @staticmethod
    def init(cfg, *, key):
        base = CopyPasteBelay.init(cfg, key=key)
        d = cfg.hidden_dim
        weight = 0.02 * jax.random.normal(jax.random.fold_in(key, 261019248), (d, d))
        return BelayCPLMPlus(
            **{f.name: getattr(base, f.name) for f in dataclasses.fields(base)},
            cp_prev_key=jax.sharding.reshard(weight, P()),
            cp_prev_gain=jax.sharding.reshard(jnp.zeros(d), P()),
            cp_variant=jax.sharding.reshard(jnp.zeros(d), P()),
            cp_variant_bias=jax.sharding.reshard(jnp.full((), -3.0), P()),
        )

    def copy_head(self, hidden, token_ids, labels, segments):
        if CASE_TABLES is None:
            raise ValueError("Belay-CPLM-plus requires frozen tokenizer class tables")
        classes, members = (jnp.asarray(x, jnp.int32) for x in CASE_TABLES)
        ts, hs = P(model._BATCH_AXES, None), P(model._BATCH_AXES, None, None)
        specs = (hs, P(), P(), P(), P(), P(), ts, ts, ts, P(), P(), P(), P(), P(), P())
        args = (
            hidden,
            self.output_proj,
            self.cp_query,
            self.cp_key,
            self.cp_gate,
            self.cp_sink,
            token_ids,
            labels,
            segments,
            self.cp_prev_key,
            self.cp_prev_gain,
            self.cp_variant,
            self.cp_variant_bias,
            classes,
            members,
        )
        args = tuple(jax.sharding.reshard(a, s) for a, s in zip(args, specs, strict=True))
        return jax.shard_map(local_head, in_specs=specs, out_specs=(ts, ts, ts, ts), check_vma=False)(*args)
