# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-group optimizer knobs: a separate MuonH momentum for the routed experts, and Sinkhorn momentum on the bigram
table."""

import equinox as eqx
import jax
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_BIGRAM = dict(ngram_stat_rows=0, second_embed=True, second_embed_bigram=True, embed2_rows=64)


def _two_steps(config: GrugMoeMuonHConfig, params):
    """The second step's updates after two independent random gradients (momentum sets how much of step one stays)."""
    opt = config.build(10)

    @eqx.filter_jit
    def run(params):
        # Replicated leaves: the tiny test model's layouts are ambiguous for the stacked Newton-Schulz contraction.
        params = jax.tree.map(lambda p: reshard(p, P(*(None,) * p.ndim)), params)
        leaves, treedef = jax.tree.flatten(params)
        keys = jax.random.split(jax.random.PRNGKey(0), 2 * len(leaves))
        first = jax.tree.unflatten(
            treedef, [0.01 * jax.random.normal(k, p.shape) for k, p in zip(keys[::2], leaves, strict=True)]
        )
        second = jax.tree.unflatten(
            treedef, [0.01 * jax.random.normal(k, p.shape) for k, p in zip(keys[1::2], leaves, strict=True)]
        )
        state = opt.init(params)
        _, state = opt.update(first, state, params)
        updates, _ = opt.update(second, state, params)
        return updates

    return run(params)


def test_routed_momentum_changes_only_the_routed_experts():
    mesh, model = t._model(**_BIGRAM)
    params = eqx.filter(model, eqx.is_inexact_array)
    mask = GrugMoeMuonHConfig(muonh_routed_momentum=0.5).create_mask(params)
    mlp_mask = mask.kda_blocks.stacked.mlp
    assert mlp_mask.expert_mlp.w_up == "muonh_routed"
    assert mask.kda_blocks.stacked.attn.w_q == "muonh"
    with jax.set_mesh(mesh):
        base = _two_steps(GrugMoeMuonHConfig(), params)
        low = _two_steps(GrugMoeMuonHConfig(muonh_routed_momentum=0.0), params)
    np.testing.assert_allclose(
        np.asarray(low.kda_blocks.stacked.attn.w_q), np.asarray(base.kda_blocks.stacked.attn.w_q), rtol=1e-6
    )
    assert not np.allclose(
        np.asarray(low.kda_blocks.stacked.mlp.expert_mlp.w_up), np.asarray(base.kda_blocks.stacked.mlp.expert_mlp.w_up)
    )


def test_bigram_table_can_train_with_sinkhorn_momentum():
    mesh, model = t._model(**_BIGRAM)
    params = eqx.filter(model, eqx.is_inexact_array)
    config = GrugMoeMuonHConfig(embed2_update="sinkhorn")
    assert config.create_mask(params).token_embed2 == "embed2"
    with jax.set_mesh(mesh):
        sinkhorn = _two_steps(config, params)
        adam = _two_steps(GrugMoeMuonHConfig(), params)
    table = np.asarray(sinkhorn.token_embed2)
    assert np.isfinite(table).all() and np.abs(table).sum() > 0
    assert not np.allclose(table, np.asarray(adam.token_embed2))
    # Sinkhorn keeps one momentum buffer for the table; AdEMAMix keeps three (m, v and the slow m).
    with jax.set_mesh(mesh):
        floats = lambda c: sum(x.size for x in jax.tree.leaves(c.build(10).init(params)))  # noqa: E731
        assert floats(GrugMoeMuonHConfig(adam_ademamix_alpha=5.0)) - floats(config) >= model.token_embed2.size


def test_embed2_update_is_validated():
    with pytest.raises(ValueError):
        GrugMoeMuonHConfig(embed2_update="lion")
    with pytest.raises(ValueError):
        GrugMoeMuonHConfig(embed2_update="sinkhorn", embed2_row_sparse_adam=True)


def test_qk_group_takes_its_own_lr_and_leaves_other_matrices_alone():
    mesh, model = t._model(**_BIGRAM)
    params = eqx.filter(model, eqx.is_inexact_array)
    mask = GrugMoeMuonHConfig(muonh_qk_lr_mult=2.0).create_mask(params)
    attn = mask.kda_blocks.stacked.attn
    assert attn.w_q == "muonh_qk" and attn.w_k == "muonh_qk"
    assert attn.w_v == "muonh" and attn.w_o == "muonh"
    with jax.set_mesh(mesh):
        base = _two_steps(GrugMoeMuonHConfig(), params)
        bold = _two_steps(GrugMoeMuonHConfig(muonh_qk_lr_mult=2.0), params)
    np.testing.assert_allclose(
        np.asarray(bold.kda_blocks.stacked.attn.w_v), np.asarray(base.kda_blocks.stacked.attn.w_v), rtol=1e-6
    )
    ratio = np.linalg.norm(np.asarray(bold.kda_blocks.stacked.attn.w_q)) / np.linalg.norm(
        np.asarray(base.kda_blocks.stacked.attn.w_q)
    )
    assert 1.5 < ratio < 2.5
