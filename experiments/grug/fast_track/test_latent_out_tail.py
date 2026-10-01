# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``latent_out_full_layers``: chosen layers' routed experts write the full residual stream (no latent write)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_kda_local as t
import experiments.grug.fast_track.test_optimizer_group_knobs as knobs
from experiments.grug.fast_track.model import CausalSelfAttention, KimiDeltaAttention
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_HIDDEN, _LATENT = 32, 16


def _tokens():
    return jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)


@pytest.mark.parametrize("tail", [(5,), (4, 5)])
def test_only_the_tail_layers_write_the_full_stream(tail):
    _, model = t._model(latent_out_full_layers=tail)
    layers = model.layers()
    kda, softmax = KimiDeltaAttention, CausalSelfAttention
    # Layer order and mixer kinds are unchanged by moving layers into the tail stacks.
    assert [type(layer.attn) for layer in layers] == [kda, kda, kda, softmax, kda, softmax]
    for i, layer in enumerate(layers):
        out_dim = layer.mlp.expert_mlp.w_down.shape[-1]
        if i in tail:
            assert out_dim == _HIDDEN and layer.mlp.w_latent_up is None
        else:
            assert out_dim == _LATENT and layer.mlp.w_latent_up is not None
    assert (model.kda_blocks_tail is None) == (4 not in tail)


def test_tail_model_trains_end_to_end():
    mesh, model = t._model(latent_out_full_layers=(4, 5))
    tokens = _tokens()
    with jax.set_mesh(mesh):
        loss, grads = eqx.filter_jit(
            eqx.filter_value_and_grad(lambda m, x: m.next_token_loss(x, jnp.ones(x.shape, jnp.float32)))
        )(model, tokens)
    assert np.isfinite(float(loss))
    tail_grad = np.asarray(grads.stacked_blocks_tail.stacked.mlp.expert_mlp.w_down)
    assert np.all(np.isfinite(tail_grad)) and np.abs(tail_grad).sum() > 0


def test_tail_layers_join_the_same_optimizer_groups():
    mesh, model = t._model(latent_out_full_layers=(4, 5))
    params = eqx.filter(model, eqx.is_inexact_array)
    mask = GrugMoeMuonHConfig(eig_families=("kda",), eig_mode="muon").create_mask(params)
    assert mask.kda_blocks_tail.stacked.attn.w_q == mask.kda_blocks.stacked.attn.w_q == "eig"
    assert mask.stacked_blocks_tail.stacked.mlp.expert_mlp.w_down == "muonh"
    with jax.set_mesh(mesh):
        updates = knobs._two_steps(GrugMoeMuonHConfig(), params)
    assert np.all(np.isfinite(np.asarray(updates.stacked_blocks_tail.stacked.mlp.expert_mlp.w_down)))


def test_tail_settings_are_validated():
    with pytest.raises(ValueError):
        t._config(latent_out_full_layers=(5, 5))
    with pytest.raises(ValueError):
        t._config(latent_out_full_layers=(5,), attn_res=False)
