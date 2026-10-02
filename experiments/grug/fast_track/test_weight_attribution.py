# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig
from experiments.grug.fast_track.weight_attribution import AttributionWriter, leaf_name, per_layer_sum, rails_by_name


def test_per_layer_sum_splits_stacked_tensors_by_layer():
    x = jnp.arange(24.0).reshape(2, 3, 4)
    np.testing.assert_allclose(per_layer_sum("kda_blocks.stacked.attn.w_q", x), [66.0, 210.0])
    np.testing.assert_allclose(per_layer_sum("token_embed", x), [276.0])


def test_rails_are_found_by_parameter_path_after_the_switch():
    mesh, model = t._model(ngram_stat_rows=0)
    params = eqx.filter(model, eqx.is_inexact_array)
    opt = GrugMoeMuonHConfig(muon_bimaxwell=True).build(6)  # rails engage after step 2
    with jax.set_mesh(mesh):
        params = jax.tree.map(lambda p: reshard(p, P(*(None,) * p.ndim)), params)
        state = opt.init(params)
        leaves, treedef = jax.tree.flatten(params)
        for i in range(4):
            keys = jax.random.split(jax.random.PRNGKey(i), len(leaves))
            grads = jax.tree.unflatten(
                treedef, [0.01 * jax.random.normal(k, p.shape) for k, p in zip(keys, leaves, strict=True)]
            )
            _, state = eqx.filter_jit(opt.update)(grads, state, params)
    fast, slow = rails_by_name(state)
    names = {
        leaf_name(path) for path, leaf in jax.tree_util.tree_leaves_with_path(params) if isinstance(leaf, jax.Array)
    }
    assert fast and set(fast) == set(slow) and set(fast) <= names
    assert any(".attn.w_q" in n for n in fast)
    # Both rails moved off zero once the rails engaged.
    assert all(float(jnp.abs(v).max()) > 0 for v in fast.values())


def test_writer_writes_chunks_with_logp(tmp_path):
    writer = AttributionWriter(str(tmp_path), chunk_size=2)
    dots = {"a": {k: np.ones(1) for k in ("Gd", "Gdp", "ddp", "dd", "GG", "Gf", "Gfp", "Gs")}}
    writer.add(5, -0.5, dots)
    writer.add(6, -0.4, dots)
    out = np.load(tmp_path / "weight_attribution_0000.npz")
    np.testing.assert_array_equal(out["steps"], [5, 6])
    np.testing.assert_allclose(out["logp"], [-0.5, -0.4])
    assert out["Gd/a"].shape == (2, 1)


def test_without_bimaxwell_the_momentum_buffer_stands_in_for_the_fast_rail():
    mesh, model = t._model(ngram_stat_rows=0)
    params = eqx.filter(model, eqx.is_inexact_array)
    opt = GrugMoeMuonHConfig().build(6)
    with jax.set_mesh(mesh):
        params = jax.tree.map(lambda p: reshard(p, P(*(None,) * p.ndim)), params)
        state = opt.init(params)
        leaves, treedef = jax.tree.flatten(params)
        grads = jax.tree.unflatten(treedef, [jnp.full(p.shape, 0.01) for p in leaves])
        _, state = eqx.filter_jit(opt.update)(grads, state, params)
    buf, slow = rails_by_name(state)
    assert buf and not slow
    assert any(".attn.w_q" in n for n in buf)
