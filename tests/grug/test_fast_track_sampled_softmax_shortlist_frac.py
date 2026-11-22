# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``sampled_softmax_shortlist_frac``: the shortlist loss against dense logits, the estimator's mean over
every draw against the full-vocabulary loss (values and gradients), and a tiny model trained through it."""

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh

from experiments.grug.fast_track import sampled_softmax
from experiments.grug.fast_track.model import GrugModelConfig, Transformer

_BATCH_AXES = ("replica_dcn", "data", "expert")
_Z_LOSS = 1e-2


def _mesh() -> Mesh:
    return Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def _dense_loss(logits, labels):
    log_partition = jax.nn.logsumexp(logits, -1)
    target = jnp.take_along_axis(logits, labels[..., None], -1)[..., 0]
    return log_partition - target + _Z_LOSS * jnp.square(log_partition)


def _problem(seq: int):
    batch, dim, vocab, size = 2, 6, 24, 5
    keys = iter(jax.random.split(jax.random.key(0), 4))
    h = jax.random.normal(next(keys), (batch, seq, dim))
    head = jax.random.normal(next(keys), (dim, vocab))
    # Skewed labels so the shortlist covers some but not all of them.
    labels = jnp.minimum(jax.random.geometric(next(keys), 0.15, (batch, seq)) - 1, vocab - 1).astype(jnp.int32)
    weight = (jax.random.uniform(next(keys), (batch, seq)) > 0.2).astype(jnp.float32)
    with jax.set_mesh(_mesh()):
        shortlist = sampled_softmax.batch_shortlist(labels, weight, vocab, size)
        columns = sampled_softmax.shortlist_positions(shortlist, labels, vocab, _BATCH_AXES)
    shortlist, columns = jnp.asarray(np.asarray(shortlist)), jnp.asarray(np.asarray(columns))
    return h, head, labels, weight, shortlist, columns


def _short(h, head, labels, shortlist, columns):
    in_shortlist = columns < shortlist.shape[0]
    logits = h @ head[:, shortlist]
    pivot = jnp.where(in_shortlist, columns, 0)
    pivot_logit = jnp.take_along_axis(logits, pivot[..., None], -1)[..., 0]
    shortlist_ce = jax.nn.logsumexp(logits, -1) - pivot_logit
    target_logit = jnp.take_along_axis(h @ head, labels[..., None], -1)[..., 0]
    return sampled_softmax.shortlist_loss(shortlist_ce, pivot_logit, target_logit, in_shortlist, _Z_LOSS)


def test_shortlist_loss_scores_the_shortlist_plus_the_target():
    h, head, labels, _, shortlist, columns = _problem(seq=20)
    in_shortlist = np.asarray(columns) < shortlist.shape[0]
    assert in_shortlist.any() and not in_shortlist.all()
    members = jnp.zeros(head.shape[1], bool).at[shortlist].set(True)
    keep = members[None, None, :] | (jnp.arange(head.shape[1]) == labels[..., None])
    expected = _dense_loss(jnp.where(keep, h @ head, -jnp.inf), labels)
    np.testing.assert_allclose(_short(h, head, labels, shortlist, columns), expected, rtol=1e-5, atol=1e-5)
    # The frequency shortlist holds the most frequent label.
    assert int(jnp.bincount(labels.reshape(-1)).argmax()) in np.asarray(shortlist)


def test_mean_over_every_draw_is_the_full_loss_and_gradient():
    # 40 positions: blocks of 16, 16 and 8, so the short last block is drawn with probability 1/8.
    h, head, labels, weight, shortlist, columns = _problem(seq=40)
    starts = jnp.arange(0, 40, sampled_softmax.SAMPLE_BLOCK)
    lengths = jnp.minimum(sampled_softmax.SAMPLE_BLOCK, 40 - starts)

    def estimate(h, head, offset):
        positions = jnp.broadcast_to(starts + offset % lengths, (h.shape[0], starts.shape[0]))
        sampled_h = jnp.take_along_axis(h, positions[..., None], 1)
        full = _dense_loss(sampled_h @ head, jnp.take_along_axis(labels, positions, 1))
        short = _short(h, head, labels, shortlist, columns)
        return sampled_softmax.debiased_loss(short, full, positions, lengths.astype(jnp.float32), weight)[0]

    def mean_estimate(h, head):
        return sum(estimate(h, head, o) for o in range(sampled_softmax.SAMPLE_BLOCK)) / sampled_softmax.SAMPLE_BLOCK

    def full_loss(h, head):
        return jnp.sum(weight * _dense_loss(h @ head, labels)) / jnp.sum(weight)

    # A single draw differs from the full loss; the mean over draws is exact.
    assert abs(float(estimate(h, head, 3)) - float(full_loss(h, head))) > 1e-3
    np.testing.assert_allclose(mean_estimate(h, head), full_loss(h, head), rtol=1e-5)
    got = jax.grad(mean_estimate, argnums=(0, 1))(h, head)
    want = jax.grad(full_loss, argnums=(0, 1))(h, head)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, w, rtol=1e-4, atol=1e-6)


def test_sampled_positions_draw_one_position_inside_each_block():
    tokens = jax.random.randint(jax.random.key(1), (64, 40), 0, 100)
    positions, lengths = sampled_softmax.sampled_positions(tokens, sampled_softmax.SAMPLE_BLOCK)
    np.testing.assert_array_equal(lengths, [16, 16, 8])
    offsets = np.asarray(positions) - np.arange(0, 40, 16)
    assert (offsets >= 0).all() and (offsets < np.asarray(lengths)).all()
    # Across rows every offset of a full block is drawn.
    assert len(np.unique(offsets[:, 0])) == 16


def test_tiny_model_trains_on_the_estimate_and_evaluates_the_full_vocabulary():
    hidden, seq, vocab = 64, 64, 128
    config = GrugModelConfig(
        vocab_size=vocab,
        hidden_dim=hidden,
        intermediate_dim=hidden // 2,
        shared_expert_intermediate_dim=hidden // 2,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=hidden // 2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        local_kv_heads=None,
        global_kv_heads=None,
        head_dim=16,
        max_seq_len=seq,
        sliding_window=seq,
        global_every=2,
        capacity_factor=8.0,
        pooled_transport_capacity_factor=8.0,
        initializer_std=0.5 / math.sqrt(hidden),
        qk_norm=False,
        sconv_sites=("attn", "mlp"),
        sampled_softmax_shortlist_frac=0.125,
    )
    tokens = jax.random.randint(jax.random.key(1), (2, seq), 0, vocab)
    weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0)
    with jax.set_mesh(_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        plain = Transformer.init(dataclasses.replace(config, sampled_softmax_shortlist_frac=0.0), key=jax.random.key(0))

        def train_loss(m):
            loss, metrics = m.next_token_loss(
                tokens, weight, logsumexp_weight=1e-4, return_router_metrics=True, train_terms=True
            )
            return loss, metrics

        def eval_loss(m):
            return m.next_token_loss(tokens, weight, logsumexp_weight=1e-4)

        (loss, metrics), grads = eqx.filter_jit(eqx.filter_value_and_grad(train_loss, has_aux=True))(model)
        full, full_grads = eqx.filter_jit(eqx.filter_value_and_grad(eval_loss))(model)
        plain_full = eqx.filter_jit(eval_loss)(plain)

    # Evaluation is the plain full-vocabulary loss.
    np.testing.assert_array_equal(np.asarray(full), np.asarray(plain_full))
    # The shortlist misses some labels, so the estimate is a noisy draw around the full loss.
    assert 0 < float(metrics["train/aux/sampled_softmax/label_coverage"]) < 1
    assert float(metrics["train/aux/sampled_softmax/shortlist_loss"]) < float(full)
    assert abs(float(loss) - float(full)) < 0.5 * float(full)
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    assert all(np.isfinite(np.asarray(g)).all() for g in leaves)
    # Every output row gets a gradient, including the labels outside the shortlist.
    sampled, exact = np.asarray(grads.output_proj).ravel(), np.asarray(full_grads.output_proj).ravel()
    assert np.linalg.norm(sampled) > 0
    assert np.dot(sampled, exact) / (np.linalg.norm(sampled) * np.linalg.norm(exact)) > 0
