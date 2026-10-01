# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``expert_visit_bias``: each visited routed expert writes its own learned row into the residual."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.model import _visit_counts
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig


def _tokens():
    return jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)


def test_visit_counts_mark_selected_experts_and_ignore_null_ids():
    selected = jnp.array([[0, 2], [3, 5]])
    weights = jnp.array([[0.25, 0.75], [1.0, 1.0]])
    np.testing.assert_allclose(
        np.asarray(_visit_counts(selected, weights, 4)), [[0.25, 0.0, 0.75, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )


@pytest.mark.parametrize("mode", ["sum", "weighted"])
def test_a_no_op_at_init_then_the_rows_get_a_gradient_and_change_the_output(mode):
    mesh, plain = t._model()
    _, model = t._model(expert_visit_bias=mode)
    tokens = _tokens()
    loss = lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape, jnp.float32))  # noqa: E731
    with jax.set_mesh(mesh):
        forward = eqx.filter_jit(lambda m, x: m(x)[0])
        np.testing.assert_allclose(np.asarray(forward(model, tokens)), np.asarray(forward(plain, tokens)), atol=1e-5)
        grads = eqx.filter_jit(eqx.filter_grad(loss))(model)
        bias_grad = np.asarray(grads.kda_blocks.stacked.mlp.expert_visit_bias)
        assert np.abs(bias_grad).sum() > 0
        moved = eqx.tree_at(lambda m: m.kda_blocks.stacked.mlp.expert_visit_bias, model, jnp.asarray(-bias_grad))
        assert not np.allclose(np.asarray(forward(moved, tokens)), np.asarray(forward(model, tokens)), atol=1e-6)


def test_bias_trains_with_adam():
    _, model = t._model(expert_visit_bias="sum")
    params = eqx.filter(model, eqx.is_array)
    mask = GrugMoeMuonHConfig().create_mask(params)
    assert mask.kda_blocks.stacked.mlp.expert_visit_bias == "adam"
    assert mask.stacked_blocks.stacked.mlp.expert_visit_bias == "adam"
