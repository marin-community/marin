# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full normalized emission oracle; forward/VJP checks only, no training."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.moe_cplm import plus


def distribution(params, tokens, segments, classes):
    h, output, qw, kw, gw, sink, pw, gain, vw, vb = params
    logits = 10 * jnp.tanh((h @ output) / 10)
    native = jax.nn.softmax(logits, -1)
    gate = jax.nn.softmax(jnp.concatenate([logits, (10 * jnp.tanh((h @ gw) / 10))[..., None]], -1), -1)[..., -1]
    previous = jnp.concatenate([jnp.zeros_like(h[:, :1]), h[:, :-1]], 1)
    starts = jnp.concatenate([jnp.ones_like(segments[:, :1], dtype=bool), segments[:, 1:] != segments[:, :-1]], 1)
    previous = previous * (~starts)[..., None]
    q, k = h @ qw, h @ kw + (previous @ pw) * gain
    scores = jnp.einsum("btd,bsd->bts", q, k) / h.shape[-1] ** 0.5
    positions = jnp.arange(h.shape[1])
    allowed = (positions[:, None] >= positions[None, :])[None] & (segments[:, :, None] == segments[:, None, :])
    scores = jnp.where(allowed, scores, -jnp.inf)
    attention = jax.nn.softmax(jnp.concatenate([scores, ((q @ sink) / h.shape[-1] ** 0.5)[..., None]], -1), -1)
    matches = classes[tokens][..., None] == classes[None, None, :]
    class_logits = jnp.where(matches[:, None, :, :], logits[:, :, None, :], -jnp.inf)
    class_emission = jax.nn.softmax(class_logits, -1)
    alpha = jax.nn.sigmoid(h @ vw + vb)[..., None, None]
    emission = (1 - alpha) * jax.nn.one_hot(tokens, output.shape[-1])[:, None] + alpha * class_emission
    copied = jnp.einsum("bts,btsv->btv", attention[..., :-1], emission)
    return (1 - gate + gate * attention[..., -1])[..., None] * native + gate[..., None] * copied


@pytest.fixture
def example():
    shapes = [(2, 8, 4), (4, 12), (4, 4), (4, 4), (4,), (4,), (4, 4), (4,), (4,), ()]
    params = tuple(
        jax.random.normal(k, s) * 0.2
        for k, s in zip(jax.random.split(jax.random.key(51), len(shapes)), shapes, strict=True)
    )
    tokens = jnp.array([[0, 2, 2, 4, 3, 3, 1, 6], [1, 1, 6, 0, 4, 4, 9, 2]])
    labels = jnp.roll(tokens, -1, 1)
    segments = jnp.broadcast_to(jnp.arange(8) // 4, tokens.shape)
    classes = jnp.arange(12) // 3
    members = (jnp.arange(12) // 3)[:, None] * 3 + jnp.arange(3)
    return params, tokens, labels, segments, classes, members


def losses(params, tokens, labels, segments, classes, members, tile):
    h, output, qw, kw, gw, sink, *extra = params
    return plus.local_head(
        h, output, qw, kw, gw, sink, tokens, labels, segments, *extra, classes, members, tile=tile, reference=True
    )[0]


@pytest.mark.parametrize("tile", [1, 4, 8])
def test_normalized_mixture_and_all_parameter_gradients(example, tile):
    p, t, y, s, c, m = example
    dense = distribution(p, t, s, c)
    np.testing.assert_allclose(dense.sum(-1), 1, atol=2e-6)

    def expected(q):
        return -jnp.log(jnp.take_along_axis(distribution(q, t, s, c), y[..., None], -1)[..., 0])

    def actual(q):
        return losses(q, t, y, s, c, m, tile)

    np.testing.assert_allclose(actual(p), expected(p), rtol=3e-6, atol=3e-6)
    for got, want in zip(
        jax.grad(lambda q: actual(q).mean())(p), jax.grad(lambda q: expected(q).mean())(p), strict=True
    ):
        np.testing.assert_allclose(got, want, rtol=5e-5, atol=3e-6)


def test_no_future_or_cross_document_predecessor_leak(example):
    p, t, y, s, c, m = example
    base = losses(p, t, y, s, c, m, 4)
    future = (p[0].at[:, 2:].add(5), *p[1:])
    np.testing.assert_allclose(losses(future, t.at[:, 2:].set(11), y, s, c, m, 4)[:, :2], base[:, :2], atol=2e-6)
    other = (p[0].at[:, :4].add(5), *p[1:])
    np.testing.assert_allclose(losses(other, t.at[:, :4].set(11), y, s, c, m, 4)[:, 4:], base[:, 4:], atol=2e-6)


def test_zero_gain_blocks_only_predecessor_matrix_gradient(example):
    p, t, y, s, c, m = example
    p = (*p[:7], jnp.zeros_like(p[7]), *p[8:])
    gradients = jax.grad(lambda q: losses(q, t, y, s, c, m, 4).mean())(p)
    np.testing.assert_array_equal(gradients[6], 0)
    assert np.linalg.norm(gradients[7]) > 0
    assert np.linalg.norm(gradients[8]) > 0
