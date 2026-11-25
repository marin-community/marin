# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The copy's per-shard routing statistics, reduced once after the layers, against Levanter's per-layer QB router."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.grug_moe import QBRoutedMoE
from levanter.grug.sharding import compact_grug_mesh
from levanter.testing.devices import use_test_devices

from experiments.amd.june_67b_a2b import model as amd
from experiments.amd.june_67b_a2b.test_model import CONFIG

_LAYERS = 2
_BATCH, _SEQ = 4, 16
# Capacities small enough that both the senders and the receivers drop assignments.
_CONFIG = dataclasses.replace(CONFIG, hidden_dim=8, capacity_factor=0.4, pooled_transport_capacity_factor=0.6)


def _run(layers, x, valid, *, per_shard):
    mesh = jax.sharding.get_abstract_mesh()

    def loss(x, layers):
        layer_stats = []
        for layer in layers:
            if per_shard:
                out, stats = layer(x, valid)
            else:
                out, stats = QBRoutedMoE(
                    num_experts_per_token=_CONFIG.num_experts_per_token, batch_axes=amd._BATCH_AXES
                )(
                    x.reshape(-1, x.shape[-1]),
                    valid.reshape(-1),
                    router=layer.router,
                    router_bias=layer.router_bias,
                    expert_mlp=layer.expert_mlp,
                    mesh=mesh,
                )
                out = jax.sharding.reshard(out.reshape(x.shape), jax.typeof(x).sharding.spec)
            x = x + out
            layer_stats.append(stats)
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *layer_stats)
        if per_shard:
            stacked = amd._reduce_routing_stats(stacked)
        return jnp.sum(x**2) + jnp.sum(stacked["router_z_loss"]), stacked

    (value, stats), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(x, layers)
    return value, jax.tree.map(np.asarray, stats), [np.asarray(g) for g in jax.tree.leaves(grads)]


def _compare(mesh):
    rng = np.random.default_rng(0)
    batch = NamedSharding(mesh, P(amd._BATCH_AXES))
    x = jax.device_put(rng.standard_normal((_BATCH, _SEQ, _CONFIG.hidden_dim)).astype(np.float32), batch)
    valid = jax.device_put(rng.random((_BATCH, _SEQ)) < 0.8, batch)
    layers = [amd.MoEMLP.init(_CONFIG, key=jax.random.key(layer)) for layer in range(_LAYERS)]
    return (*_run(layers, x, valid, per_shard=False), *_run(layers, x, valid, per_shard=True))


# Without expert parallelism the copy takes Levanter's local path, which drops nothing.
@pytest.mark.parametrize("expert_axis_size", [pytest.param(2, id="expert_parallel"), pytest.param(1, id="local")])
def test_per_shard_stats_reduced_after_the_layers_match_per_layer_stats(expert_axis_size):
    with use_test_devices(4):
        mesh = compact_grug_mesh(expert_axis_size=expert_axis_size)
        with jax.set_mesh(mesh):
            loss, stats, grads, p_loss, p_stats, p_grads = _compare(mesh)

    np.testing.assert_allclose(p_loss, loss, rtol=1e-6)
    # The statistics do not feed the layers, and a sum's gradient does not depend on its order: exact.
    for got, want in zip(p_grads, grads, strict=True):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(p_stats["capacity_overflow"], stats["capacity_overflow"])
    if expert_axis_size > 1:
        assert stats["routing_sender_drops"].min() > 0 and stats["routing_receiver_drops"].min() > 0, stats
    # Sums over the shards taken in another order than the per-layer reductions.
    for name in ("qb_beta", "router_z_loss"):
        np.testing.assert_allclose(p_stats[name], stats[name], rtol=1e-6, atol=1e-7)
