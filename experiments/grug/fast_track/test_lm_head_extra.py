# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``lm_head_extra_dim``: the final layer's experts write an lm_head-only slice beside the residual stream."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_ngram_stat as t


def test_final_layer_writes_a_head_only_slice_read_by_a_wider_lm_head():
    mesh, model = t._model(ngram_stat_rows=0, mla=True, num_layers=4, latent_out_full_layers=(3,), lm_head_extra_dim=16)
    cfg = model.config
    assert model.output_proj.shape == (cfg.hidden_dim + 16, cfg.vocab_size)
    tail_mlp = model.stacked_blocks_tail.stacked.mlp
    assert tail_mlp.expert_mlp.w_down.shape[-1] == cfg.hidden_dim + 16 and tail_mlp.w_latent_up is None
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        hidden, _ = eqx.filter_jit(lambda m: m(tokens))(model)
        loss, grads = eqx.filter_jit(
            eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape)))
        )(model)
    assert hidden.shape[-1] == cfg.hidden_dim + 16 and np.isfinite(float(loss))
    # The head-only columns of the final experts and the extra norm both learn from the loss.
    extra_cols = grads.stacked_blocks_tail.stacked.mlp.expert_mlp.w_down[..., cfg.hidden_dim :]
    assert float(jnp.abs(extra_cols).max()) > 0
    assert float(jnp.abs(grads.lm_head_extra_norm.weight).max()) > 0


def test_shared_source_widens_only_the_shared_expert():
    mesh, model = t._model(
        ngram_stat_rows=0,
        mla=True,
        num_layers=4,
        latent_out_full_layers=(3,),
        lm_head_extra_dim=16,
        lm_head_extra_source="shared",
    )
    cfg = model.config
    tail = model.stacked_blocks_tail.stacked
    assert tail.mlp.expert_mlp.w_down.shape[-1] == cfg.hidden_dim
    assert tail.shared[0].w_down.shape[-1] == cfg.hidden_dim + 16
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        loss, grads = eqx.filter_jit(
            eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape)))
        )(model)
    assert np.isfinite(float(loss))
    extra_cols = grads.stacked_blocks_tail.stacked.shared[0].w_down[..., cfg.hidden_dim :]
    assert float(jnp.abs(extra_cols).max()) > 0


@pytest.mark.parametrize("extra", [0, 16])
def test_final_shared_only_layer_runs_no_routed_experts(extra):
    settings = dict(lm_head_extra_dim=extra, lm_head_extra_source="shared") if extra else {}
    mesh, model = t._model(
        ngram_stat_rows=0,
        mla=True,
        num_layers=4,
        num_shared_experts=1,
        latent_out_full_layers=(3,),
        final_shared_only=True,
        final_shared_intermediate_dim=48,
        **settings,
    )
    tail = model.stacked_blocks_tail.stacked
    assert tail.mlp.cfg.routed_off and tail.shared[0].w_up.shape[-1] == 48
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        loss, grads = eqx.filter_jit(
            eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape)))
        )(model)
    assert np.isfinite(float(loss))
    g = grads.stacked_blocks_tail.stacked
    assert float(jnp.abs(g.mlp.expert_mlp.w_up).max()) == 0.0  # the routed path never runs
    assert float(jnp.abs(g.shared[0].w_up).max()) > 0


def test_final_intermediate_dim_widens_only_the_final_layers_experts():
    mesh, model = t._model(
        ngram_stat_rows=0, mla=True, num_layers=4, latent_out_full_layers=(3,), final_intermediate_dim=32
    )
    assert model.stacked_blocks_tail.stacked.mlp.expert_mlp.w_up.shape[-1] == 32
    assert model.stacked_blocks.stacked.mlp.expert_mlp.w_up.shape[-1] == model.config.intermediate_dim
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        loss = eqx.filter_jit(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape)))(model)
    assert np.isfinite(float(loss))


def test_final_experts_per_token_changes_only_the_final_layers_top_k():
    mesh, model = t._model(
        ngram_stat_rows=0, mla=True, num_layers=4, latent_out_full_layers=(3,), final_experts_per_token=3
    )
    assert model.stacked_blocks_tail.stacked.mlp.cfg.num_experts_per_token == 3
    assert model.stacked_blocks.stacked.mlp.cfg.num_experts_per_token == model.config.num_experts_per_token
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        loss = eqx.filter_jit(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape)))(model)
    assert np.isfinite(float(loss))
