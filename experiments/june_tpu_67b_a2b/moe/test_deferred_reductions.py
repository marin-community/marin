# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reductions the June model can move out of its layer scan, on four CPU devices (data=2 x expert=2).

- ``replicated_grad_reduction="after_scan"``: the small replicated weights' gradients stay unreduced through the scan.
- ``router_stats_reduction="after_scan"``: routing statistics and QB thresholds are reduced once after it.
- ``qb_balance="per_node"``: one QB threshold per group of data shards, each group with its own router bias.
"""

import collections
import dataclasses
import re

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from levanter.grug.sharding import compact_grug_mesh
from levanter.testing.devices import use_cpu_devices

from experiments.june_tpu_67b_a2b.moe import train as june_train
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer

BASE = GrugModelConfig(
    vocab_size=32,
    hidden_dim=16,
    intermediate_dim=8,
    shared_expert_intermediate_dim=8,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=3,
    num_heads=2,
    num_kv_heads=1,
    head_dim=8,
    max_seq_len=8,
    sliding_window=4,
    capacity_factor=4.0,
    attention_implementation="reference",
    moe_implementation="ring",
    use_array_stacked_blocks=True,
    disable_pko=True,
)


def _mesh():
    return compact_grug_mesh(expert_axis_size=2, replica_axis_size=1)


# Weights replicated per node: replica_dcn=2, data=1, so the attention and shared-expert weights are replicated too.
def _replica_mesh():
    return compact_grug_mesh(expert_axis_size=2, replica_axis_size=2)


TOKENS = np.asarray(jax.random.randint(jax.random.key(1), (8, 8), 0, BASE.vocab_size))
WEIGHT = np.ones((8, 8), dtype=jnp.float32)


def run(cfg, mesh=None):
    if mesh is None:
        mesh = _mesh()
    with jax.set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.key(0))

        def loss_fn(m):
            return m.next_token_loss(TOKENS, WEIGHT, reduction="mean", return_router_metrics=True)

        step = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
        (loss, metrics), grads = step(model)
        hlo = step.lower(model).compile().as_text()
    return loss, metrics, grads, hlo


def all_reduces_in_loops(hlo):
    # All-reduces inside while-loop bodies (the layer scans and anything nested in them).
    bodies = set(re.findall(r"body=%?([\w.\-]+)", hlo))
    count, current = 0, None
    for line in hlo.splitlines():
        header = re.match(r"^(?:ENTRY )?%?([\w.\-]+) .*\{$", line)
        if header and not line.startswith(" "):
            current = header.group(1)
        elif current in bodies and re.search(r"\ball-reduce(-start)?\(", line):
            count += 1
    return count


def leaves(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


@pytest.mark.timeout(600)
@pytest.mark.multidevice
def test_deferred_replicated_gradients_match_and_leave_the_layer_scan():
    with use_cpu_devices(4):
        loss, _metrics, grads, hlo = run(BASE)
        d_loss, _d_metrics, d_grads, d_hlo = run(dataclasses.replace(BASE, replicated_grad_reduction="after_scan"))
        np.testing.assert_allclose(d_loss, loss, rtol=1e-6)
        for got, want in zip(leaves(d_grads), leaves(grads), strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)
        base, deferred = all_reduces_in_loops(hlo), all_reduces_in_loops(d_hlo)
        assert deferred < base, (deferred, base)
        # A zero size limit defers nothing.
        *_, z_hlo = run(dataclasses.replace(BASE, replicated_grad_reduction="after_scan", replicated_grad_max_bytes=0))
        assert all_reduces_in_loops(z_hlo) == base


@pytest.mark.timeout(600)
@pytest.mark.multidevice
def test_deferred_gradients_cover_dense_weights_replicated_per_node():
    with use_cpu_devices(4):
        loss, _metrics, grads, hlo = run(BASE, _replica_mesh())
        deferred = dataclasses.replace(BASE, replicated_grad_reduction="after_scan")
        d_loss, _, d_grads, d_hlo = run(deferred, _replica_mesh())
        np.testing.assert_allclose(d_loss, loss, rtol=1e-6)
        for got, want in zip(leaves(d_grads), leaves(grads), strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)
        # Per layer only the expert weights' all-reduce over replica_dcn stays in the loop.
        base, deferred = all_reduces_in_loops(hlo), all_reduces_in_loops(d_hlo)
        assert deferred < base, (deferred, base)
        print("in-loop all-reduces", base, deferred)


@pytest.mark.timeout(600)
@pytest.mark.multidevice
def test_router_stats_after_scan_match_per_layer():
    with use_cpu_devices(4):
        loss, metrics, grads, hlo = run(BASE)
        a_loss, a_metrics, a_grads, a_hlo = run(dataclasses.replace(BASE, router_stats_reduction="after_scan"))
        np.testing.assert_array_equal(a_loss, loss)
        for name in (
            "qb_beta_per_layer",
            "train/router/routing_counts_per_layer",
            "train/router/router_z_loss",
            "train/router/load_balancing_loss",
            "train/router/routing_entropy_mean",
        ):
            np.testing.assert_allclose(np.asarray(a_metrics[name]), np.asarray(metrics[name]), rtol=1e-5, atol=1e-6)
        for got, want in zip(leaves(a_grads), leaves(grads), strict=True):
            np.testing.assert_array_equal(got, want)
        assert all_reduces_in_loops(a_hlo) < all_reduces_in_loops(hlo)


@pytest.mark.timeout(600)
@pytest.mark.multidevice
def test_per_node_qb_thresholds_and_bias():
    with use_cpu_devices(4):
        cfg = dataclasses.replace(BASE, router_stats_reduction="after_scan")
        _, global_metrics, _, _ = run(cfg)
        _, node_metrics, _, _ = run(dataclasses.replace(cfg, qb_balance="per_node"))
        per_node = np.asarray(node_metrics["qb_beta_per_layer"])
        overall = np.asarray(global_metrics["qb_beta_per_layer"])
        assert per_node.shape == (BASE.num_layers, 2, BASE.num_experts), per_node.shape
        # Every token is valid and both groups hold the same number, so the global threshold is their mean.
        np.testing.assert_allclose(per_node.mean(axis=1), overall, rtol=1e-5, atol=1e-6)
        assert not np.allclose(per_node[:, 0], per_node[:, 1]), "groups saw different tokens; thresholds must differ"

        with jax.set_mesh(_mesh()):
            # Group g owns the g-th half of the tokens (batch axes are ordered replica_dcn, data, expert).
            layer = Transformer.init(dataclasses.replace(cfg, qb_balance="per_node"), key=jax.random.key(0))
            routed = jax.tree_util.tree_map(lambda x: x[0], layer.stacked_blocks.stacked).mlp.routed_moe
            logits = reshard(jnp.zeros((8, BASE.num_experts)), P(routed.batch_axes, None))
            bias = jnp.arange(2 * BASE.num_experts, dtype=jnp.float32).reshape(2, BASE.num_experts)
            got = np.asarray(jax.jit(lambda l, b: routed._add_router_bias(l, b, _mesh()))(logits, bias))
        np.testing.assert_array_equal(got, np.repeat(np.asarray(bias), 4, axis=0))

        with jax.set_mesh(_mesh()):
            betas = reshard(jnp.asarray(per_node), P(None, ("replica_dcn", "data"), None))
            synced = np.asarray(jax.jit(june_train._average_qb_betas_over_groups)(betas))
        np.testing.assert_allclose(synced, np.broadcast_to(per_node.mean(axis=1, keepdims=True), per_node.shape))


@pytest.mark.timeout(600)
@pytest.mark.multidevice
def test_per_node_train_step_syncs_every_interval():
    with use_cpu_devices(4):
        cfg = dataclasses.replace(
            BASE, router_stats_reduction="after_scan", qb_balance="per_node", replicated_grad_reduction="after_scan"
        )
        batch = collections.namedtuple("Batch", "tokens loss_weight attn_mask")(TOKENS, WEIGHT, None)
        policy = jmp.get_policy("f32")
        optimizer = optax.adam(1e-3)
        with jax.set_mesh(_mesh()):
            state = june_train.initial_state(cfg, optimizer=optimizer, mp=policy, key=jax.random.key(0), ema_beta=None)
            assert state.pending_qb_betas.shape == (BASE.num_layers, 2, BASE.num_experts)
            step = june_train._make_train_step(optimizer, policy, z_loss_weight=0.0, ema_beta=None, qb_sync_interval=2)
            state, _, _ = step(state, batch)  # step 1: per-node thresholds kept
            first = np.asarray(state.pending_qb_betas)
            state, _, _ = step(state, batch)  # step 2: averaged over the nodes
            second = np.asarray(state.pending_qb_betas)
        assert not np.allclose(first[:, 0], first[:, 1])
        np.testing.assert_allclose(second[:, 0], second[:, 1])
