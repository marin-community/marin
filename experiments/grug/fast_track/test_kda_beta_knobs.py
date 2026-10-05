# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""KDA write-strength knobs: ``init_std_mult_beta`` scales only ``w_beta``, ``kda_beta_scale`` adds a trained
per-head logit scale (Adam), and ``kda_beta_group`` moves ``w_beta`` to AdamH or Adam."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

import experiments.grug.fast_track.test_ngram_stat as t
import experiments.grug.fast_track.test_optimizer_group_knobs as knobs
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig


def test_beta_init_mult_scales_only_w_beta():
    _, base = t._model(ngram_stat_rows=0, init_std_mult_gates=2.0)
    _, wide = t._model(ngram_stat_rows=0, init_std_mult_gates=2.0, init_std_mult_beta=4.0)
    attn, wide_attn = base.kda_blocks.stacked.attn, wide.kda_blocks.stacked.attn
    np.testing.assert_allclose(np.asarray(wide_attn.w_beta), 2 * np.asarray(attn.w_beta), rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(wide_attn.w_g), np.asarray(attn.w_g))


def test_beta_scale_starts_neutral_trains_and_is_adam():
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    losses = {}
    for flag in (False, True):
        mesh, model = t._model(ngram_stat_rows=0, kda_beta_scale=flag)
        with jax.set_mesh(mesh):
            loss, grads = eqx.filter_jit(
                eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape)))
            )(model)
        losses[flag] = float(loss)
    np.testing.assert_allclose(losses[True], losses[False], rtol=1e-6)
    assert float(jnp.abs(grads.kda_blocks.stacked.attn.beta_scale).max()) > 0
    mask = GrugMoeMuonHConfig().create_mask(eqx.filter(model, eqx.is_inexact_array))
    assert mask.kda_blocks.stacked.attn.beta_scale == "kda_decay"


@pytest.mark.parametrize("group", ["kda_beta", "adamh", "adam"])
def test_beta_group_routes_and_updates_w_beta(group):
    mesh, model = t._model(ngram_stat_rows=0)
    params = eqx.filter(model, eqx.is_inexact_array)
    config = GrugMoeMuonHConfig(kda_beta_group=group)
    expected = "kda_beta_adam" if group == "adam" else group
    assert config.create_mask(params).kda_blocks.stacked.attn.w_beta == expected
    with jax.set_mesh(mesh):
        updates = knobs._two_steps(config, params)
    w_beta = np.asarray(params.kda_blocks.stacked.attn.w_beta)
    step = np.asarray(updates.kda_blocks.stacked.attn.w_beta)
    assert np.all(np.isfinite(step)) and np.abs(step).max() > 0
    if group != "adam":
        # The hyperball groups keep each layer's norm; plain Adam lets it move.
        new_norm = np.linalg.norm((w_beta + step).reshape(w_beta.shape[0], -1), axis=1)
        np.testing.assert_allclose(new_norm, np.linalg.norm(w_beta.reshape(w_beta.shape[0], -1), axis=1), rtol=1e-4)


def test_adam_warmup_stretches_only_the_adam_schedule():
    mesh, model = t._model(ngram_stat_rows=0)
    params = eqx.filter(model, eqx.is_inexact_array)

    def lrs_at(config, step):
        opt = config.build(1000)
        with jax.set_mesh(mesh):
            # Replicated leaves, as in test_optimizer_group_knobs: the tiny model's layouts are ambiguous for NS.
            p = jax.tree.map(lambda x: reshard(x, P(*(None,) * x.ndim)), params)
            state = opt.init(p)
            state = state._replace(count=jnp.asarray(step, jnp.int32))
            grads = jax.tree.map(jnp.zeros_like, p)
            _, state = eqx.filter_jit(opt.update)(grads, state, p)
        return float(state.hyperparams["learning_rate"]), float(state.hyperparams["adam_lr"])

    base, long = GrugMoeMuonHConfig(warmup=10), GrugMoeMuonHConfig(warmup=10, adam_warmup=100)
    muon_b, adam_b = lrs_at(base, 50)
    muon_l, adam_l = lrs_at(long, 50)
    np.testing.assert_allclose(muon_l, muon_b, rtol=1e-6)
    np.testing.assert_allclose(adam_l, 0.5 * adam_b, rtol=0.05)
