# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train-step changes that must not change the numbers, on four CPU devices (data=2 x expert=2, or replica_dcn=2 x
expert=2 for weights replicated per node).

- ``carry_compute_params``: the compute-dtype copy carried in the train state gives the same losses and parameters as
  casting the parameters inside each step, also with the layer scan on a replicated mesh.
- ``replicated_grad_reduction``: reducing the replicated weights' gradients in every layer or after the layer scan
  trains the same on a replicated mesh.
- The ring_filtered routing state saved for backward (``grug_moe_routing``) gives the same gradients as recomputing it.
"""

import collections
import dataclasses

import jax
import jmp
import numpy as np
import optax
import pytest
from levanter.grug.sharding import compact_grug_mesh
from levanter.testing.devices import use_cpu_devices

from experiments.june_tpu_67b_a2b.moe import train as june_train
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig

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
    moe_implementation="ring_filtered",
    pooled_transport_capacity_factor=1.1,
    use_array_stacked_blocks=True,
    disable_pko=True,
)


def _mesh():
    return compact_grug_mesh(expert_axis_size=2, replica_axis_size=1)


# Weights replicated per node: replica_dcn=2, data=1, so every weight but the experts' is replicated.
def _replica_mesh():
    return compact_grug_mesh(expert_axis_size=2, replica_axis_size=2)


TOKENS = np.asarray(jax.random.randint(jax.random.key(1), (8, 8), 0, BASE.vocab_size))
BATCH = collections.namedtuple("Batch", "tokens loss_weight attn_mask")(TOKENS, np.ones((8, 8), np.float32), None)
POLICY = jmp.get_policy("p=f32,c=bfloat16")


def train(cfg, *, carry_compute_params, mesh=None, steps=3):
    if mesh is None:
        mesh = _mesh()
    optimizer = optax.adam(1e-2)
    with jax.set_mesh(mesh):
        state = june_train.initial_state(
            cfg,
            optimizer=optimizer,
            mp=POLICY,
            key=jax.random.key(0),
            ema_beta=None,
            carry_compute_params=carry_compute_params,
        )
        step = june_train._make_train_step(optimizer, POLICY, z_loss_weight=0.0, ema_beta=None)
        losses = []
        for _ in range(steps):
            state, metrics, _ = step(state, BATCH)
            losses.append(float(metrics["train/loss"]))
    return losses, state


def leaves(tree):
    return [np.asarray(x, dtype=np.float32) for x in jax.tree_util.tree_leaves(tree)]


@pytest.mark.timeout(600)
@pytest.mark.multidevice
def test_carried_compute_params_match_casting_in_the_step():
    with use_cpu_devices(4):
        cfg = dataclasses.replace(BASE, remat_mode="save_moe")
        losses, state = train(cfg, carry_compute_params=False)
        carried_losses, carried = train(cfg, carry_compute_params=True)
        assert state.compute_params is None
        np.testing.assert_allclose(carried_losses, losses, rtol=1e-6)
        for a, b in zip(leaves(carried.params), leaves(state.params), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
        # The carried copy is the cast of the updated parameters, router biases included.
        for a, b in zip(leaves(carried.compute_params), leaves(POLICY.cast_to_compute(carried.params)), strict=True):
            np.testing.assert_array_equal(a, b)


@pytest.mark.timeout(600)
@pytest.mark.multidevice
def test_saving_routing_state_matches_recomputing_it():
    with use_cpu_devices(4):
        losses, state = train(dataclasses.replace(BASE, remat_mode="recompute_all"), carry_compute_params=False)
        saved_losses, saved = train(dataclasses.replace(BASE, remat_mode="save_moe"), carry_compute_params=False)
        np.testing.assert_allclose(saved_losses, losses, rtol=1e-6)
        for a, b in zip(leaves(saved.params), leaves(state.params), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)


@pytest.mark.timeout(900)
@pytest.mark.multidevice
def test_carried_compute_params_match_on_a_replicated_mesh():
    with use_cpu_devices(4):
        for reduction in ("per_layer", "after_scan"):
            cfg = dataclasses.replace(
                BASE, remat_mode="save_moe", replicated_grad_reduction=reduction, router_stats_reduction=reduction
            )
            losses, state = train(cfg, carry_compute_params=False, mesh=_replica_mesh())
            carried_losses, carried = train(cfg, carry_compute_params=True, mesh=_replica_mesh())
            np.testing.assert_allclose(carried_losses, losses, rtol=1e-6, err_msg=reduction)
            for a, b in zip(leaves(carried.params), leaves(state.params), strict=True):
                np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7, err_msg=reduction)


@pytest.mark.timeout(900)
@pytest.mark.multidevice
def test_replicated_gradients_per_layer_train_like_after_scan_on_a_replicated_mesh():
    with use_cpu_devices(4):
        cfg = dataclasses.replace(BASE, remat_mode="save_moe", router_stats_reduction="after_scan")
        losses, _ = train(cfg, carry_compute_params=False, mesh=_replica_mesh(), steps=4)
        deferred_losses, _ = train(
            dataclasses.replace(cfg, replicated_grad_reduction="after_scan"),
            carry_compute_params=False,
            mesh=_replica_mesh(),
            steps=4,
        )
        assert losses[-1] < losses[0], losses
        # The bfloat16 gradients are summed in a different order. Adam turns that rounding into visible differences in
        # parameters with near-zero gradients, so compare the losses, which stay within a few 1e-6.
        np.testing.assert_allclose(deferred_losses, losses, rtol=3e-5)
