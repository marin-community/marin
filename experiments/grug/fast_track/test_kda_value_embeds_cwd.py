# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``kda_value_embeds`` and ``muonh_cautious_wd``."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig, _scale_invariant_hyperball_updates


def _loss(model, tokens):
    return model.next_token_loss(tokens, jnp.ones(tokens.shape, jnp.float32))


def test_kda_value_embeddings_train_with_the_value_embedding_rules():
    mesh, plain = t._model()
    _, model = t._model(kda_value_embeds="gated")
    attn = model.kda_blocks.stacked.attn
    num_kda = attn.w_q.shape[0]
    assert attn.value_embed.shape == (num_kda, t._VOCAB, attn.w_q.shape[-1])
    assert attn.ve_gate.shape == (num_kda, model.config.hidden_dim, model.config.num_heads)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        grads = eqx.filter_jit(eqx.filter_grad(_loss))(model, tokens)
        assert float(eqx.filter_jit(_loss)(model, tokens)) != float(eqx.filter_jit(_loss)(plain, tokens))
    for leaf in (grads.kda_blocks.stacked.attn.value_embed, grads.kda_blocks.stacked.attn.ve_gate):
        assert np.abs(np.asarray(leaf)).sum() > 0
    mask = GrugMoeMuonHConfig().create_mask(eqx.filter(model, eqx.is_array))
    assert mask.kda_blocks.stacked.attn.value_embed == "adam"
    assert mask.kda_blocks.stacked.attn.ve_gate == "adam"


def test_cautious_wd_keeps_the_norm_and_shrinks_sign_agreeing_coordinates():
    param = jnp.array([[1.0, -2.0], [3.0, 0.5]])
    # The update agrees in sign with the weight at (0, 0) and (1, 0) only.
    update = jnp.array([[0.1, 0.1], [0.1, -0.1]])
    plain = param + _scale_invariant_hyperball_updates(param, update, 0.1)
    decayed = param + _scale_invariant_hyperball_updates(param, update, 0.1, cautious_wd=2.0)
    norm = float(jnp.linalg.norm(param))
    np.testing.assert_allclose(float(jnp.linalg.norm(decayed)), norm, rtol=1e-6)
    ratio = np.asarray(decayed) / np.asarray(plain)
    # Agreeing coordinates shrink relative to the plain step; the others grow to keep the norm.
    assert ratio[0, 0] < 1 and ratio[1, 0] < 1 and ratio[0, 1] > 1 and ratio[1, 1] > 1
    np.testing.assert_allclose(
        np.asarray(param + _scale_invariant_hyperball_updates(param, update, 0.1, cautious_wd=0.0)), np.asarray(plain)
    )
