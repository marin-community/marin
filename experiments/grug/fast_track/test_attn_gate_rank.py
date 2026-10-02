# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Low-rank per-channel attention gate (``attn_gate_rank``) and the MLA-only q/k optimizer scope."""

import equinox as eqx
import jax
import numpy as np

import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig, QkScope


def _forward(model, tokens):
    return eqx.filter_jit(lambda m, x: m(x)[0])(model, tokens)


def test_low_rank_gate_starts_as_the_identity_and_learns():
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    mesh, base = t._model(mla=True)
    _, gated = t._model(mla=True, attn_gate_rank=4)
    with jax.set_mesh(mesh):
        np.testing.assert_allclose(_forward(gated, tokens), _forward(base, tokens), rtol=1e-5, atol=1e-5)
        grads = eqx.filter_jit(eqx.filter_grad(lambda m: m.next_token_loss(tokens, jax.numpy.ones(tokens.shape))))(gated)
    up = [
        leaf
        for path, leaf in jax.tree_util.tree_leaves_with_path(grads)
        if jax.tree_util.keystr(path).endswith("attn_gate_up")
    ]
    assert up and all(float(jax.numpy.abs(g).max()) > 0 for g in up)


def test_mla_qk_scope_covers_only_the_mla_query_and_key_projections():
    _, model = t._model(mla=True)
    params = eqx.filter(model, eqx.is_inexact_array)
    mask = GrugMoeMuonHConfig(muonh_qk_lr_mult=0.5, muonh_qk_scope=QkScope.MLA).create_mask(params)
    mla = mask.stacked_blocks.stacked.attn
    assert mla.w_q == mla.w_dkv == mla.w_uk == "muonh_qk"
    assert mla.w_uv == "muonh" and mla.w_o == "muonh"
    kda = mask.kda_blocks.stacked.attn
    assert kda.w_q == kda.w_k == "muonh"
