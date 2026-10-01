# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``attn_res_token_query``: a token-embedding term added to each AttnRes gate's static query."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.model import _attn_res_num_gates
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig
from experiments.grug.fast_track.train import _dump_final_params


def _tokens():
    return jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)


@pytest.mark.parametrize(("mode", "tables"), [("shared", 1), ("per_gate", None)])
def test_a_no_op_at_init_then_the_table_gets_a_gradient_and_changes_the_output(mode, tables):
    mesh, plain = t._model()
    _, model = t._model(attn_res_token_query=mode)
    table = model.attn_res_query_token
    assert table.shape[1:] == (t._VOCAB, plain.config.hidden_dim)
    cfg = plain.config
    assert table.shape[0] == (tables or _attn_res_num_gates(cfg) + cfg.num_layers * int(cfg.attn_res_v_gate))
    tokens = _tokens()
    loss = lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape, jnp.float32))  # noqa: E731
    with jax.set_mesh(mesh):
        forward = eqx.filter_jit(lambda m, x: m(x)[0])
        np.testing.assert_allclose(np.asarray(forward(model, tokens)), np.asarray(forward(plain, tokens)), atol=1e-5)
        grad = np.asarray(eqx.filter_jit(eqx.filter_grad(loss))(model).attn_res_query_token)
        # Only the rows of tokens in the batch get a gradient.
        seen = np.unique(np.asarray(tokens))
        assert np.abs(grad[:, seen]).sum() > 0
        assert not np.abs(np.delete(grad, seen, axis=1)).any()
        moved = eqx.tree_at(lambda m: m.attn_res_query_token, model, jnp.asarray(-grad))
        assert not np.allclose(np.asarray(forward(moved, tokens)), np.asarray(forward(model, tokens)), atol=1e-6)


def test_table_trains_with_its_own_group():
    _, model = t._model(attn_res_token_query="shared")
    mask = GrugMoeMuonHConfig().create_mask(eqx.filter(model, eqx.is_array))
    assert mask.attn_res_query_token == "attn_res_token_query"


def test_final_param_dump_writes_the_matching_tables(tmp_path):
    mesh, model = t._model(attn_res_token_query="shared")
    path = str(tmp_path / "final_params.npz")
    with jax.set_mesh(mesh):
        _dump_final_params(model, ("attn_res_query_token$", "attn_res_query_final$"), path)
    dumped = np.load(path)
    assert sorted(dumped.files) == ["attn_res_query_final", "attn_res_query_token"]
    np.testing.assert_array_equal(dumped["attn_res_query_token"], np.asarray(model.attn_res_query_token))
