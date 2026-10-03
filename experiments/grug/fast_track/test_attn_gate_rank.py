# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Low-rank per-channel attention gate (``attn_gate_rank``) and the MLA-only q/k optimizer scope."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

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


def test_rel_pos_lr_mult_scales_only_the_relative_position_updates():
    mesh, model = t._model(mla=True, inkling_relpos=True, max_seq_len=128, sliding_window=128)
    params = eqx.filter(model, eqx.is_inexact_array)
    mask = GrugMoeMuonHConfig(rel_pos_lr_mult=0.3).create_mask(params)
    rel = mask.stacked_blocks.stacked.attn.rel_pos
    assert {label for label in jax.tree.leaves(rel)} == {"rel_pos"}
    assert mask.stacked_blocks.stacked.attn.w_q == "muonh"

    def first_update(config):
        opt = config.build(10)
        with jax.set_mesh(mesh):
            p = jax.tree.map(lambda x: reshard(x, P(*(None,) * x.ndim)), params)
            grads = jax.tree.map(lambda x: jnp.full(x.shape, 0.01), p)
            updates, _ = eqx.filter_jit(opt.update)(grads, opt.init(p), p)
        return updates

    base, slow = first_update(GrugMoeMuonHConfig()), first_update(GrugMoeMuonHConfig(rel_pos_lr_mult=0.3))
    for b, s in zip(
        jax.tree.leaves(base.stacked_blocks.stacked.attn.rel_pos),
        jax.tree.leaves(slow.stacked_blocks.stacked.attn.rel_pos),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(s), 0.3 * np.asarray(b), rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(slow.stacked_blocks.stacked.attn.w_q), np.asarray(base.stacked_blocks.stacked.attn.w_q), rtol=1e-5
    )
