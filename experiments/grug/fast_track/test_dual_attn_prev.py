# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``dual_attn_prev``: odd rows run MLA attention at the previous step's weights in training only, and their
gradient reaches the current weights, never the frozen copy."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.model import refresh_attn_prev
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_OVERRIDES = dict(ngram_stat_rows=0, mla=True)


def _loss(model, tokens, route_key):
    return model.next_token_loss(tokens, jnp.ones(tokens.shape), train_terms=True, route_key=route_key)


def _perturb_prev(model):
    """Move the frozen copy away from the current weights, as one optimizer step would."""
    leaves = lambda m: m.stacked_blocks.stacked.attn_prev.w_q  # noqa: E731
    return eqx.tree_at(leaves, model, leaves(model) * 0.5)


def test_identical_copies_match_the_single_weight_model():
    mesh, dual = t._model(dual_attn_prev=True, **_OVERRIDES)
    _, plain = t._model(**_OVERRIDES)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (4, t._SEQ), 0, t._VOCAB)
    key = jax.random.PRNGKey(3)
    with jax.set_mesh(mesh):
        a = float(eqx.filter_jit(_loss)(dual, tokens, key))
        b = float(eqx.filter_jit(_loss)(plain, tokens, key))
    np.testing.assert_allclose(a, b, rtol=1e-5)


def test_old_rows_change_training_only_and_their_gradient_reaches_the_current_weights():
    mesh, dual = t._model(dual_attn_prev=True, **_OVERRIDES)
    dual = _perturb_prev(dual)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (4, t._SEQ), 0, t._VOCAB)
    key = jax.random.PRNGKey(3)
    with jax.set_mesh(mesh):
        train = float(eqx.filter_jit(_loss)(dual, tokens, key))
        evaluate = float(eqx.filter_jit(_loss)(dual, tokens, None))
        clean = float(
            eqx.filter_jit(_loss)(
                eqx.tree_at(lambda m: m.stacked_blocks.stacked.attn_prev, dual, dual.stacked_blocks.stacked.attn),
                tokens,
                None,
            )
        )
        grads = eqx.filter_jit(eqx.filter_grad(_loss))(dual, tokens, key)
    assert train != evaluate
    np.testing.assert_allclose(evaluate, clean, rtol=1e-6)  # evals ignore the old copy
    assert float(jnp.abs(grads.stacked_blocks.stacked.attn.w_q).max()) > 0
    assert float(jnp.abs(grads.stacked_blocks.stacked.attn_prev.w_q).max()) == 0


def test_refresh_copies_the_pre_update_attention_into_the_frozen_copy():
    mesh, old = t._model(dual_attn_prev=True, **_OVERRIDES)
    new = eqx.tree_at(lambda m: m.stacked_blocks.stacked.attn.w_q, old, old.stacked_blocks.stacked.attn.w_q + 1.0)
    with jax.set_mesh(mesh):
        refreshed = refresh_attn_prev(new, old)
    np.testing.assert_array_equal(
        np.asarray(refreshed.stacked_blocks.stacked.attn_prev.w_q), np.asarray(old.stacked_blocks.stacked.attn.w_q)
    )
    np.testing.assert_array_equal(
        np.asarray(refreshed.stacked_blocks.stacked.attn.w_q), np.asarray(new.stacked_blocks.stacked.attn.w_q)
    )


def test_the_frozen_copy_gets_no_optimizer_updates():
    _, model = t._model(dual_attn_prev=True, **_OVERRIDES)
    mask = GrugMoeMuonHConfig().create_mask(eqx.filter(model, eqx.is_inexact_array))
    assert {label for label in jax.tree.leaves(mask.stacked_blocks.stacked.attn_prev)} == {"frozen"}
    assert mask.stacked_blocks.stacked.attn.w_q == "muonh"
