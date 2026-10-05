# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``future_aux``: cheap multi-token prediction heads (hashed future n-grams, a low-dim bottleneck, embedding
regression) that train from the final hidden, mask futures crossing documents, and leave evals untouched."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.model import FutureAux, _future_targets
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig


def test_targets_hash_the_right_future_tokens_and_stop_at_document_ends():
    tokens = jnp.array([[5, 6, 7, 8, 9, 10]])
    weight = jnp.ones(tokens.shape)
    segments = jnp.array([[0, 0, 0, 0, 1, 1]])
    futures, w, buckets = _future_targets(FutureAux.HASH_BIGRAM, tokens, weight, segments, 97)
    np.testing.assert_array_equal(futures[0][0, :4], [6, 7, 8, 9])
    np.testing.assert_array_equal(futures[1][0, :4], [7, 8, 9, 10])
    # Position 2's bigram (8, 9) crosses into document 1, and the last two positions run past the end.
    np.testing.assert_array_equal(np.asarray(w[0]), [1, 1, 0, 0, 0, 0])
    assert int(buckets.min()) >= 0 and int(buckets.max()) < 97
    # Same future tokens give the same bucket; a different bigram (almost always) a different one.
    _, _, again = _future_targets(FutureAux.HASH_BIGRAM, tokens, weight, None, 97)
    np.testing.assert_array_equal(np.asarray(again), np.asarray(buckets))


@pytest.mark.parametrize("mode", [m for m in FutureAux if m != FutureAux.NONE])
def test_every_mode_trains_its_head_and_skips_evals(mode):
    mesh, model = t._model(ngram_stat_rows=0, future_aux=mode, future_aux_buckets=64, future_aux_lowdim=16)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    weight = jnp.ones(tokens.shape)

    def loss(m, w):
        out, metrics = m.next_token_loss(
            tokens, weight, return_router_metrics=True, train_terms=True, future_aux_weight=w
        )
        return out, metrics

    with jax.set_mesh(mesh):
        (with_aux, metrics), grads = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))(
            model, jnp.float32(0.5)
        )
        plain = eqx.filter_jit(lambda m: m.next_token_loss(tokens, weight))(model)
    future = float(metrics["train/aux/future_loss"])
    assert future > 0
    np.testing.assert_allclose(float(with_aux), float(plain) + 0.5 * future, rtol=1e-5)
    assert float(jnp.abs(grads.future_head).max()) > 0
    mask = GrugMoeMuonHConfig().create_mask(eqx.filter(model, eqx.is_inexact_array))
    assert mask.future_head == "adamh"
