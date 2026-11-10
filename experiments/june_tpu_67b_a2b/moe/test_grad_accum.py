# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gradient accumulation over microbatches in the June train step, on eight CPU devices with the ring_filtered MoE.

A train step with ``microbatches=K`` gives the loss, gradients and routing counts of the whole batch, on a mesh with
replicas and expert parallelism (replica_dcn=4 x expert=2) and on one that also shards the parameters over ``data``
(replica_dcn=2 x data=2 x expert=2). The capacity factors are high enough that no expert overflows, since capacity
applies per microbatch. This holds with the logit and router z-losses on, with padding that leaves the microbatches
unequal valid tokens, and with a microbatch that carries no loss weight at all. The next step's QB thresholds are the
mean of those of the microbatches.
"""

import collections
import dataclasses

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from levanter.grug.attention import AttentionMask
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
    capacity_factor=8.0,
    attention_implementation="reference",
    moe_implementation="ring_filtered",
    pooled_transport_capacity_factor=4.0,
    use_array_stacked_blocks=True,
    disable_pko=True,
    replicated_grad_reduction="after_scan",
    router_stats_reduction="after_scan",
)


def _meshes():
    return {
        "replica4_expert2": compact_grug_mesh(expert_axis_size=2, replica_axis_size=4),
        "replica2_data2_expert2": compact_grug_mesh(expert_axis_size=2, replica_axis_size=2),
    }


# 32 rows: four microbatches of one row per device.
TOKENS = np.asarray(jax.random.randint(jax.random.key(1), (32, 8), 0, BASE.vocab_size))
LOSS_WEIGHT = np.ones((32, 8), np.float32)
LOSS_WEIGHT[:, -1] = 0
LOSS_WEIGHT[::3, :4] = 0  # Microbatches with unequal loss weight.
Batch = collections.namedtuple("Batch", "tokens loss_weight attn_mask")
BATCH = Batch(TOKENS, LOSS_WEIGHT, None)
# Rows 0, 3, 6, ... end in three padding positions, so the microbatches route unequal numbers of tokens.
SEGMENT_IDS = np.zeros((32, 8), np.int32)
SEGMENT_IDS[::3, -3:] = -1
PADDED_BATCH = Batch(
    TOKENS,
    np.where(SEGMENT_IDS >= 0, LOSS_WEIGHT, 0),
    AttentionMask(is_causal=True, segment_ids=(SEGMENT_IDS, SEGMENT_IDS)),
)
# Rows 0, 4, 8, ... carry no loss weight: microbatch 0 of four has none.
UNWEIGHTED_ROWS_WEIGHT = LOSS_WEIGHT.copy()
UNWEIGHTED_ROWS_WEIGHT[::4] = 0
UNWEIGHTED_MICROBATCH = Batch(TOKENS, UNWEIGHTED_ROWS_WEIGHT, None)


def keep_gradients():
    # Leaves the parameters unchanged and stores the step's gradients as the optimizer state.
    return optax.GradientTransformation(
        init=lambda params: jax.tree.map(jnp.zeros_like, params),
        update=lambda grads, state, params=None: (jax.tree.map(jnp.zeros_like, grads), grads),
    )


def step(cfg, mesh, policy, *, microbatches, carry_compute_params=False, batch=BATCH, z_loss_weight=0.0):
    optimizer = keep_gradients()
    with jax.set_mesh(mesh):
        state = june_train.initial_state(
            cfg,
            optimizer=optimizer,
            mp=policy,
            key=jax.random.key(0),
            ema_beta=None,
            carry_compute_params=carry_compute_params,
        )
        train_step = june_train._make_train_step(
            optimizer, policy, z_loss_weight=z_loss_weight, ema_beta=None, microbatches=microbatches
        )
        state, metrics, _ = train_step(state, batch)
    return state, jax.device_get(metrics)


def leaves(tree):
    return [np.asarray(x, dtype=np.float32) for x in jax.tree_util.tree_leaves(tree)]


def grad_errors(state, reference):
    # Relative L2 error of every gradient leaf against the reference state's.
    return [
        np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
        for a, b in zip(leaves(state.opt_state), leaves(reference.opt_state), strict=True)
    ]


def assert_matches_whole_batch(cfg, policy, check_grads, *, carry_compute_params=False, **step_args):
    for name, mesh in _meshes().items():
        whole, whole_metrics = step(
            cfg, mesh, policy, microbatches=1, carry_compute_params=carry_compute_params, **step_args
        )
        assert np.sum(whole_metrics["moe/dropped_assignments"]) == 0, name
        for k in (2, 4):
            msg = f"{name} K={k}"
            accumulated, metrics = step(
                cfg, mesh, policy, microbatches=k, carry_compute_params=carry_compute_params, **step_args
            )
            assert metrics.keys() == whole_metrics.keys(), msg
            for key in (
                "train/loss",
                "train/cross_entropy_loss",
                "train/router/aux_loss_weighted",
                "train/router/router_z_loss",
            ):
                np.testing.assert_allclose(metrics[key], whole_metrics[key], rtol=1e-6, err_msg=f"{msg} {key}")
            # Same routing for every token, and no drops: capacity applies per microbatch.
            np.testing.assert_array_equal(
                metrics["train/router/routing_counts_per_layer"],
                whole_metrics["train/router/routing_counts_per_layer"],
                err_msg=msg,
            )
            assert np.sum(metrics["moe/dropped_assignments"]) == 0, msg
            check_grads(accumulated, whole, name, msg)


@pytest.mark.multidevice
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    ("router_z_loss_coef", "step_args"),
    [
        (None, {"batch": BATCH}),
        # The logit z-loss is weighted by loss weight, the router z-loss averages over valid tokens.
        (0.1, {"batch": PADDED_BATCH, "z_loss_weight": 1e-2}),
        (0.1, {"batch": UNWEIGHTED_MICROBATCH}),
    ],
    ids=["unequal_loss_weight", "z_losses_and_padding", "unweighted_microbatch"],
)
def test_microbatches_match_the_whole_batch_in_float32(router_z_loss_coef, step_args):
    # Only the order of the float32 sums changes: a few ulps (2**-23 each); 4e-7 observed.
    cfg = BASE if router_z_loss_coef is None else dataclasses.replace(BASE, router_z_loss_coef=router_z_loss_coef)

    def check_grads(accumulated, whole, name, msg):
        assert max(grad_errors(accumulated, whole)) <= 1e-5, msg

    with use_cpu_devices(8):
        assert_matches_whole_batch(cfg, policy=jmp.get_policy("f32"), check_grads=check_grads, **step_args)


@pytest.mark.timeout(900)
@pytest.mark.multidevice
def test_microbatches_match_the_whole_batch_with_bfloat16_compute():
    # The forward is the same for every token, so the loss matches to float32 rounding. The gradients do not: each
    # microbatch rounds its bfloat16 weight gradients and their cross-device sums on its own. The whole batch's
    # bfloat16 gradients are themselves off the float32 ones, by up to 2e-2 (the norm weights, whose gradient sums
    # many terms), so require every microbatched gradient leaf to be no further from its float32 counterpart than the
    # whole batch's, with a margin and a floor of one bfloat16 rounding (2**-8).
    with use_cpu_devices(8):
        policy = jmp.get_policy("p=f32,c=bfloat16")
        cfg = dataclasses.replace(BASE, remat_mode="save_moe")
        exact = {name: step(cfg, mesh, jmp.get_policy("f32"), microbatches=1)[0] for name, mesh in _meshes().items()}

        def check_grads(accumulated, whole, name, msg):
            bounds = 1.5 * np.asarray(grad_errors(whole, exact[name])) + 2**-8
            errors = np.asarray(grad_errors(accumulated, exact[name]))
            assert np.all(errors <= bounds), (msg, errors, bounds)

        assert_matches_whole_batch(cfg, policy, check_grads)
        assert_matches_whole_batch(cfg, policy, check_grads, carry_compute_params=True)


@pytest.mark.timeout(900)
@pytest.mark.multidevice
def test_qb_thresholds_are_the_mean_over_microbatches():
    with use_cpu_devices(8):
        policy = jmp.get_policy("f32")
        for qb_balance in ("global", "per_node"):
            cfg = dataclasses.replace(BASE, qb_balance=qb_balance)
            mesh = _meshes()["replica2_data2_expert2"]
            k = 4
            state, _ = step(cfg, mesh, policy, microbatches=k)
            # Microbatch i holds rows i, i + k, ...: run each through a whole-batch step on its own.
            expected = np.mean(
                [
                    step(cfg, mesh, policy, microbatches=1, batch=Batch(TOKENS[i::k], LOSS_WEIGHT[i::k], None))[
                        0
                    ].pending_qb_betas
                    for i in range(k)
                ],
                axis=0,
            )
            assert jax.typeof(state.pending_qb_betas).sharding.spec == june_train.qb_betas_spec(cfg), qb_balance
            np.testing.assert_allclose(state.pending_qb_betas, expected, rtol=1e-6, atol=1e-7, err_msg=qb_balance)


@pytest.mark.timeout(900)
@pytest.mark.multidevice
def test_per_node_qb_thresholds_weight_each_group_by_its_own_valid_tokens():
    # Padding spread unevenly over the groups: microbatch 1 of 4 holds rows 1, 5, 9, ..., two per group of the
    # replica_dcn=2 x data=2 groups, and group 0's (rows 1 and 5) are all padding. Each group's thresholds are the mean
    # of the microbatches' weighted by that group's valid tokens, so microbatch 1 adds nothing to group 0's.
    with use_cpu_devices(8):
        policy = jmp.get_policy("f32")
        cfg = dataclasses.replace(BASE, qb_balance="per_node")
        mesh = _meshes()["replica2_data2_expert2"]
        k, groups = 4, 4
        segment_ids = SEGMENT_IDS.copy()
        segment_ids[[1, 5]] = -1

        def batch(rows):
            ids = segment_ids[rows]
            return Batch(
                TOKENS[rows],
                np.where(ids >= 0, LOSS_WEIGHT[rows], 0),
                AttentionMask(is_causal=True, segment_ids=(ids, ids)),
            )

        state, _ = step(cfg, mesh, policy, microbatches=k, batch=batch(slice(None)))
        betas, valid = [], []
        for i in range(k):
            rows = slice(i, None, k)
            betas.append(step(cfg, mesh, policy, microbatches=1, batch=batch(rows))[0].pending_qb_betas)
            valid.append(np.sum(segment_ids[rows] >= 0, axis=1).reshape(groups, -1).sum(axis=1))
        betas, valid = np.stack(betas), np.stack(valid).astype(np.float64)
        assert valid[1, 0] == 0 and np.all(np.abs(betas[1, :, 0]) == 0)
        shares = valid / valid.sum(axis=0)
        expected = np.sum(shares[:, None, :, None] * betas, axis=0)
        np.testing.assert_allclose(state.pending_qb_betas, expected, rtol=1e-6, atol=1e-7)
