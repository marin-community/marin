# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``value_embeds=gated_lambda``: the gated value embedding times a learned per-layer scalar (init 1), so it starts
as ``gated`` and the scalar trains."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import experiments.grug.fast_track.test_ngram_stat as t


def _loss_and_grads(mode: str, tokens):
    mesh, model = t._model(ngram_stat_rows=0, mla=True, mla_kv_latent_dim=16, value_embeds=mode)
    with jax.set_mesh(mesh):
        return eqx.filter_jit(eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape))))(
            model
        )


def test_gated_lambda_starts_as_gated_and_its_scalar_trains():
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    loss_gated, grads_gated = _loss_and_grads("gated", tokens)
    loss_lam, grads_lam = _loss_and_grads("gated_lambda", tokens)
    np.testing.assert_allclose(float(loss_lam), float(loss_gated), rtol=1e-6)
    # Under `gated` the scalar is unused; under `gated_lambda` it gets a gradient.
    assert float(jnp.abs(grads_gated.stacked_blocks.stacked.attn.ve_lambda[..., 1]).max()) == 0
    assert float(jnp.abs(grads_lam.stacked_blocks.stacked.attn.ve_lambda[..., 1]).max()) > 0
