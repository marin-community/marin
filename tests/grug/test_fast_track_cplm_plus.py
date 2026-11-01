# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``cplm_plus``: the tiled target-only CPLM Plus loss against a dense full-vocabulary mixture, its
causality and document isolation, the case-class tables, and a tiny model training through the head."""

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

from experiments.grug.fast_track import cplm
from experiments.grug.fast_track.model import GrugModelConfig, LocalMixer, Transformer

_VOCAB = 12
_DIM = 4


def _dense_distribution(params, tokens, segments, classes):
    """The full normalized next-token mixture, written out over the vocabulary."""
    h, output, head = params
    logits = h @ output
    native = jax.nn.softmax(logits, -1)
    copy_logit = 10 * jnp.tanh((h @ head.gate) / 10)
    gate = jax.nn.sigmoid(copy_logit - jax.nn.logsumexp(logits, -1))
    previous = jnp.concatenate([jnp.zeros_like(h[:, :1]), h[:, :-1]], 1)
    starts = jnp.concatenate([jnp.ones_like(segments[:, :1], dtype=bool), segments[:, 1:] != segments[:, :-1]], 1)
    previous = previous * (~starts)[..., None]
    q, k = h @ head.query, h @ head.key + (previous @ head.prev_key) * head.prev_gain
    scale = math.sqrt(h.shape[-1])
    scores = jnp.einsum("btd,bsd->bts", q, k) / scale
    positions = jnp.arange(h.shape[1])
    allowed = (positions[:, None] >= positions[None, :])[None] & (segments[:, :, None] == segments[:, None, :])
    scores = jnp.where(allowed, scores, -jnp.inf)
    attention = jax.nn.softmax(jnp.concatenate([scores, ((q @ head.sink) / scale)[..., None]], -1), -1)
    in_class = classes[tokens][..., None] == classes[None, None, :]
    class_emission = jax.nn.softmax(jnp.where(in_class[:, None, :, :], logits[:, :, None, :], -jnp.inf), -1)
    alpha = jax.nn.sigmoid(h @ head.variant + head.variant_bias)[..., None, None]
    emission = (1 - alpha) * jax.nn.one_hot(tokens, output.shape[-1])[:, None] + alpha * class_emission
    copied = jnp.einsum("bts,btsv->btv", attention[..., :-1], emission)
    return (1 - gate + gate * attention[..., -1])[..., None] * native + gate[..., None] * copied


@pytest.fixture
def example():
    keys = iter(jax.random.split(jax.random.key(51), 16))
    shapes = {"query": (_DIM, _DIM), "key": (_DIM, _DIM), "prev_key": (_DIM, _DIM), "variant_bias": ()}
    head = cplm.CopyHead(
        **{
            f.name: 0.4 * jax.random.normal(next(keys), shapes.get(f.name, (_DIM,)))
            for f in dataclasses.fields(cplm.CopyHead)
        }
    )
    h = 0.4 * jax.random.normal(next(keys), (2, 8, _DIM))
    output = 0.4 * jax.random.normal(next(keys), (_DIM, _VOCAB))
    tokens = jnp.array([[0, 2, 2, 4, 3, 3, 1, 6], [1, 1, 6, 0, 4, 4, 9, 2]])
    labels = jnp.roll(tokens, -1, 1)
    segments = jnp.broadcast_to(jnp.arange(8) // 4, tokens.shape)
    # Classes of three consecutive ids.
    classes = jnp.arange(_VOCAB) // 3
    members = classes[:, None] * 3 + jnp.arange(3)
    return (h, output, head), tokens, labels, segments, classes, members


def _tiled_nll(params, tokens, labels, segments, classes, members, tile):
    h, output, head = params
    logits = h @ output
    native_nll = jax.nn.logsumexp(logits, -1) - jnp.take_along_axis(logits, labels[..., None], -1)[..., 0]
    return cplm.local_plus_nll(h, output, head, tokens, labels, segments, native_nll, classes, members, tile=tile)[0]


@pytest.mark.parametrize("tile", [1, 4, 8])
def test_tiled_loss_matches_dense_mixture_values_and_gradients(example, tile):
    params, tokens, labels, segments, classes, members = example
    np.testing.assert_allclose(_dense_distribution(params, tokens, segments, classes).sum(-1), 1, atol=2e-6)

    def expected(p):
        distribution = _dense_distribution(p, tokens, segments, classes)
        return -jnp.log(jnp.take_along_axis(distribution, labels[..., None], -1)[..., 0])

    def actual(p):
        return _tiled_nll(p, tokens, labels, segments, classes, members, tile)

    np.testing.assert_allclose(actual(params), expected(params), rtol=3e-6, atol=3e-6)
    got = jax.tree.leaves(jax.grad(lambda p: actual(p).mean())(params))
    want = jax.tree.leaves(jax.grad(lambda p: expected(p).mean())(params))
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, w, rtol=5e-5, atol=3e-6)


def test_no_future_or_cross_document_leak(example):
    (h, output, head), tokens, labels, segments, classes, members = example

    def nll(h, tokens):
        return _tiled_nll((h, output, head), tokens, labels, segments, classes, members, 4)

    base = nll(h, tokens)
    future = nll(h.at[:, 2:].add(5), tokens.at[:, 2:].set(11))
    np.testing.assert_allclose(future[:, :2], base[:, :2], atol=2e-6)
    other = nll(h.at[:, :4].add(5), tokens.at[:, :4].set(11))
    np.testing.assert_allclose(other[:, 4:], base[:, 4:], atol=2e-6)


def test_case_classes_group_spacing_underscore_and_case_only():
    tables = cplm.case_tables_from_pieces(["View", " view", "_VIEW", "42", "view-", "é", ""])
    np.testing.assert_array_equal(tables.classes, [0, 0, 0, 3, 4, 5, 6])
    np.testing.assert_array_equal(tables.members[:3], [[0, 1, 2]] * 3)
    np.testing.assert_array_equal(tables.members[3:], [[i, -1, -1] for i in range(3, 7)])


def test_tiny_model_trains_through_the_copy_head():
    hidden, seq, vocab = 64, 128, 128
    config = GrugModelConfig(
        vocab_size=vocab,
        hidden_dim=hidden,
        intermediate_dim=hidden // 2,
        shared_expert_intermediate_dim=hidden // 2,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=hidden // 2,
        num_layers=4,
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
        mla=True,
        mla_kv_latent_dim=32,
        inkling_relpos=True,
        rel_dim=4,
        rel_extent=8,
        attn_res=True,
        attn_res_num_blocks=2,
        local_mixer=LocalMixer.KDA,
        cplm_plus=True,
    )
    cplm.bind_case_tables(cplm.case_tables_from_pieces([f"w{i // 4}" if i % 2 else f" W{i // 4}" for i in range(vocab)]))
    mesh = Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    with jax.set_mesh(mesh):
        model = Transformer.init(config, key=jax.random.key(0))
        plain = Transformer.init(dataclasses.replace(config, cplm_plus=False), key=jax.random.key(0))
        # Repeated spans give the copy head something to find.
        tokens = jnp.tile(jax.random.randint(jax.random.key(1), (2, seq // 4), 0, vocab), (1, 4))
        weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0)

        def loss_fn(m):
            return m.next_token_loss(tokens, weight, logsumexp_weight=1e-4)

        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))(model)

        def per_token_fn(m):
            return m.next_token_loss(tokens, weight, reduction="none", logsumexp_weight=1e-4)

        per_token = eqx.filter_jit(per_token_fn)(model)
        hidden_on = eqx.filter_jit(lambda m: m(tokens)[0])(model)
        hidden_off = eqx.filter_jit(lambda m: m(tokens)[0])(plain)

    # The flag adds the head and leaves the backbone's initialization alone.
    np.testing.assert_array_equal(hidden_on, hidden_off)
    assert np.isfinite(float(loss))
    np.testing.assert_allclose(float(jnp.sum(per_token)) / float(jnp.sum(weight)), float(loss), rtol=1e-5)
    head_grads = grads.copy_head
    # prev_key's gradient passes through prev_gain, which starts at zero.
    for name in ("query", "key", "gate", "sink", "prev_gain", "variant", "variant_bias"):
        assert float(jnp.linalg.norm(getattr(head_grads, name))) > 0, name
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(eqx.filter(grads, eqx.is_array)))
