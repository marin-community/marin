# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``mla_num_heads`` widens only the global (MLA) layers' attention; the KDA layers keep ``num_heads``."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import experiments.grug.fast_track.test_ngram_stat as t


def test_mla_layers_get_their_own_head_count_and_train():
    mesh, model = t._model(ngram_stat_rows=0, mla=True, mla_kv_latent_dim=16, mla_num_heads=3)
    head_dim = model.config.inferred_head_dim
    assert model.stacked_blocks.stacked.attn.w_q.shape[-1] == 3 * head_dim
    assert model.stacked_blocks.stacked.attn.w_o.shape[-2] == 3 * head_dim
    assert model.kda_blocks.stacked.attn.w_o.shape[-2] == model.config.num_heads * head_dim
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        loss, grads = eqx.filter_jit(
            eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape)))
        )(model)
    assert np.isfinite(float(loss))
    assert float(jnp.abs(grads.stacked_blocks.stacked.attn.w_q[..., -head_dim:]).max()) > 0
