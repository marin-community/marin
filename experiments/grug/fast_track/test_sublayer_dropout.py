# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``sublayer_dropout``: training (with a route key) drops sublayer outputs; evaluation is the plain model."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_ngram_stat as t


def _loss(model, tokens, key):
    return model.next_token_loss(tokens, jnp.ones(tokens.shape), train_terms=True, route_key=key)


def test_dropout_changes_training_only():
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    mesh, dropped = t._model(ngram_stat_rows=0, sublayer_dropout=0.3)
    _, plain = t._model(ngram_stat_rows=0)
    with jax.set_mesh(mesh):
        train_a = float(eqx.filter_jit(_loss)(dropped, tokens, jax.random.PRNGKey(5)))
        train_b = float(eqx.filter_jit(_loss)(dropped, tokens, jax.random.PRNGKey(6)))
        evaluate = float(eqx.filter_jit(_loss)(dropped, tokens, None))
        base = float(eqx.filter_jit(_loss)(plain, tokens, None))
    np.testing.assert_allclose(evaluate, base, rtol=1e-6)
    assert train_a != evaluate and train_a != train_b  # a fresh mask per key


def test_dropout_rate_is_validated():
    with pytest.raises(ValueError, match="sublayer_dropout"):
        t._config(sublayer_dropout=1.0)
