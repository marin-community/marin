# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gain parameterizations (``norm_gain_fn``, ``sublayer_scale_fn``): every option starts as the linear model, the
exponential gain moves relatively, and ``norm_gain_lr_mult`` reaches only the norm gains."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.model import NormGainFn, RMSNorm, apply_gain_fn
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig


def _loss(**overrides):
    mesh, model = t._model(ngram_stat_rows=0, **overrides)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        return float(eqx.filter_jit(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape)))(model))


@pytest.mark.parametrize("fn", [NormGainFn.EXP, NormGainFn.SOFTPLUS, NormGainFn.SIGMOID2])
def test_every_gain_fn_starts_as_the_linear_model(fn):
    np.testing.assert_allclose(_loss(norm_gain_fn=fn), _loss(), rtol=1e-5)


def test_sigmoid2_sublayer_scales_start_at_one():
    np.testing.assert_allclose(
        _loss(sublayer_scales=True, sublayer_scale_fn=NormGainFn.SIGMOID2), _loss(sublayer_scales=True), rtol=1e-5
    )


def test_exp_gain_moves_by_the_same_fraction_at_any_size():
    small, large = jnp.log(0.5), jnp.log(4.0)
    step = 0.01
    for w in (small, large):
        ratio = apply_gain_fn(NormGainFn.EXP, w + step) / apply_gain_fn(NormGainFn.EXP, w)
        np.testing.assert_allclose(float(ratio), np.exp(step), rtol=1e-6)
    with jax.set_mesh(t._mesh()):
        assert float(RMSNorm.init(4, 1e-6, NormGainFn.SOFTPLUS).gain()[0]) == pytest.approx(1.0, rel=1e-6)


def test_norm_gain_lr_mult_scales_only_the_norm_gains():
    mesh, model = t._model(ngram_stat_rows=0)
    params = eqx.filter(model, eqx.is_inexact_array)
    mask = GrugMoeMuonHConfig(norm_gain_lr_mult=2.0).create_mask(params)
    assert mask.stacked_blocks.stacked.rms_mlp.weight == "norm_gain"
    assert mask.final_norm.weight == "norm_gain"
    assert mask.stacked_blocks.stacked.attn.w_q == "muonh"

    def first_update(config):
        opt = config.build(10)
        with jax.set_mesh(mesh):
            p = jax.tree.map(lambda x: reshard(x, P(*(None,) * x.ndim)), params)
            grads = jax.tree.map(lambda x: jnp.full(x.shape, 0.01), p)
            updates, _ = eqx.filter_jit(opt.update)(grads, opt.init(p), p)
        return updates

    base, fast = first_update(GrugMoeMuonHConfig()), first_update(GrugMoeMuonHConfig(norm_gain_lr_mult=2.0))
    np.testing.assert_allclose(
        np.asarray(fast.final_norm.weight), 2.0 * np.asarray(base.final_norm.weight), rtol=1e-4, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(fast.stacked_blocks.stacked.attn.w_q), np.asarray(base.stacked_blocks.stacked.attn.w_q), rtol=1e-5
    )
