# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent dense vocabulary-mixture oracle; no optimizer updates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.moe_cplm.head import copy_paste_nll, mixture_nll


def dense_reference(params, tokens, labels, segments):
    hidden, vocabulary, query, key, gate, sink = params
    logits = hidden @ vocabulary
    native = jax.nn.softmax(logits, -1)
    copy_gate = jax.nn.softmax(jnp.concatenate([logits, (hidden @ gate)[..., None]], -1), -1)[..., -1]
    queries, keys = hidden @ query, hidden @ key
    scores = jnp.einsum("bqd,bkd->bqk", queries, keys) / hidden.shape[-1] ** 0.5
    length = tokens.shape[1]
    allowed = (jnp.arange(length)[None, :] <= jnp.arange(length)[:, None])[None] & (
        segments[:, :, None] == segments[:, None, :]
    )
    scores = jnp.where(allowed, scores, -jnp.inf)
    sink_score = (queries @ sink) / hidden.shape[-1] ** 0.5
    attention = jax.nn.softmax(jnp.concatenate([scores, sink_score[..., None]], -1), -1)
    copied = jnp.einsum("bqk,bkv->bqv", attention[..., :-1], jax.nn.one_hot(tokens, vocabulary.shape[-1]))
    distribution = (1 - copy_gate + copy_gate * attention[..., -1])[..., None] * native
    distribution += copy_gate[..., None] * copied
    return -jnp.log(jnp.take_along_axis(distribution, labels[..., None], -1)[..., 0])


def tiled_loss(params, tokens, labels, segments, tile):
    hidden, vocabulary, query, key, gate, sink = params
    logits = hidden @ vocabulary
    log_partition = jax.nn.logsumexp(logits, -1)
    native_nll = log_partition - jnp.take_along_axis(logits, labels[..., None], -1)[..., 0]
    return copy_paste_nll(
        hidden, query, key, gate, sink, tokens, labels, segments, native_nll, log_partition, query_tile=tile
    )


@pytest.fixture
def example():
    keys = jax.random.split(jax.random.key(151), 6)
    shapes = ((2, 16, 8), (8, 17), (8, 8), (8, 8), (8,), (8,))
    params = tuple(0.2 * jax.random.normal(key, shape) for key, shape in zip(keys, shapes, strict=True))
    tokens = jnp.broadcast_to((jnp.arange(16) % 5)[None], (2, 16))
    # Includes duplicates, currently observed targets and targets absent from
    # the context. Documents split at position 8, including across query tiles.
    labels = jnp.broadcast_to((jnp.arange(16) % 7)[None], tokens.shape)
    segments = jnp.broadcast_to((jnp.arange(16) // 8)[None], tokens.shape)
    return params, tokens, labels, segments


@pytest.mark.parametrize("tile", [1, 4, 16])
def test_tiled_loss_matches_dense_mixture_values_and_gradients(example, tile):
    params, tokens, labels, segments = example
    weights = jnp.linspace(0, 1, tokens.size).reshape(tokens.shape)
    def actual(p):
        return tiled_loss(p, tokens, labels, segments, tile)

    def expected(p):
        return dense_reference(p, tokens, labels, segments)

    np.testing.assert_allclose(jax.jit(actual)(params), jax.jit(expected)(params), atol=3e-6, rtol=3e-6)
    actual_grad = jax.jit(jax.grad(lambda p: jnp.mean(actual(p) * weights)))(params)
    expected_grad = jax.jit(jax.grad(lambda p: jnp.mean(expected(p) * weights)))(params)
    for got, want in zip(actual_grad, expected_grad, strict=True):
        assert np.isfinite(np.asarray(got)).all()
        np.testing.assert_allclose(got, want, atol=3e-6, rtol=3e-5)


def test_copy_head_excludes_future_and_other_documents(example):
    params, tokens, labels, segments = example
    original = tiled_loss(params, tokens, labels, segments, 4)
    future_params = (params[0].at[:, 4:].add(2), *params[1:])
    future = tiled_loss(future_params, tokens.at[:, 4:].set(9), labels, segments, 4)
    np.testing.assert_allclose(original[:, :4], future[:, :4], atol=3e-6, rtol=3e-6)
    other_params = (params[0].at[:, :8].add(2), *params[1:])
    other = tiled_loss(other_params, tokens.at[:, :8].set(9), labels, segments, 4)
    np.testing.assert_allclose(original[:, 8:], other[:, 8:], atol=3e-6, rtol=3e-6)


def test_disabled_gate_recovers_native_loss_and_zero_weights_give_zero_gradients(example):
    params, tokens, labels, segments = example
    native = jnp.linspace(0, 10, tokens.size).reshape(tokens.shape)
    disabled = mixture_nll(
        native, jnp.full_like(native, 1e10), jnp.zeros_like(native), jnp.zeros_like(native), jnp.zeros_like(native)
    )
    np.testing.assert_array_equal(disabled, native)
    gradients = jax.grad(lambda p: jnp.sum(tiled_loss(p, tokens, labels, segments, 4) * 0))(params)
    for grad in gradients:
        np.testing.assert_array_equal(grad, jnp.zeros_like(grad))
