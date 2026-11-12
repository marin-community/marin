# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gradient accumulation over microbatches in the June train step, on eight CPU devices with the ring_filtered MoE.

A train step with ``microbatches=K`` gives the loss, gradients and routing counts of the whole batch, on a mesh with
replicas and expert parallelism (replica_dcn=4 x expert=2) and on one that also shards the parameters over ``data``
(replica_dcn=2 x data=2 x expert=2). The capacity factors are high enough that no expert overflows, since capacity
applies per microbatch. This holds with the logit and router z-losses on, with padding that leaves the microbatches
unequal valid tokens, and with a microbatch that carries no loss weight at all. The next step's QB thresholds are the
mean of those of the microbatches. Every weight's gradient is reduced across the replicas once per step, after the
microbatch loop. With ``overlap_last_microbatch`` the step gives the same loss, metrics and gradients, bit for bit, and
the expert weights' gradients are reduced instead one layer at a time in the last microbatch's backward layer scan.
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


def step(
    cfg,
    mesh,
    policy,
    *,
    microbatches,
    carry_compute_params=False,
    batch=BATCH,
    z_loss_weight=0.0,
    overlap_last_microbatch=False,
):
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
            optimizer,
            policy,
            z_loss_weight=z_loss_weight,
            ema_beta=None,
            microbatches=microbatches,
            overlap_last_microbatch=overlap_last_microbatch,
        )
        state, metrics, _ = train_step(state, batch)
    return state, jax.device_get(metrics)


def compiled_step_hlo(cfg, mesh, policy, *, microbatches, overlap_last_microbatch=False, combine_all_reduces=True):
    optimizer = optax.adam(1e-2)
    # Without the combiner every reduction keeps its own instruction and op_name.
    options = {} if combine_all_reduces else {"xla_disable_hlo_passes": "cpu-all-reduce-combiner"}
    with jax.set_mesh(mesh):
        state = june_train.initial_state(cfg, optimizer=optimizer, mp=policy, key=jax.random.key(0), ema_beta=None)
        train_step = june_train._make_train_step(
            optimizer,
            policy,
            z_loss_weight=0.0,
            ema_beta=None,
            microbatches=microbatches,
            overlap_last_microbatch=overlap_last_microbatch,
        )
        return train_step.lower(state, BATCH).compile(compiler_options=options).as_text()


Reduction = collections.namedtuple("Reduction", "in_loop elements scalar op_name")
_REDUCTIONS = ("all-reduce", "all-reduce-start", "reduce-scatter")


def replica_reductions(hlo, mesh, ops=_REDUCTIONS):
    # The all-reduces and reduce-scatters (or other collectives ``ops``) of a compiled CPU module that cross
    # replica_dcn: whether each runs inside a while loop (the microbatch loop, the layer scans), how many elements it
    # reduces (its operands', so a reduce-scatter counts its output times its group size; an all-gather its output),
    # whether every operand is a scalar, and its source op.
    computations, current = {}, None
    for line in hlo.splitlines():
        header = re.match(r"(ENTRY )?%?([\w.\-]+) .*\{$", line)
        if header and not line.startswith(" "):
            current = header.group(2)
            computations[current] = (bool(header.group(1)), [])
        elif current and line.startswith(" "):
            computations[current][1].append(line.strip())
    # A computation runs in a loop if it is a while body or is called from one.
    in_loop, stack = {}, [(name, False) for name, (entry, _) in computations.items() if entry]
    while stack:
        name, looped = stack.pop()
        if in_loop.get(name, False) >= looped and name in in_loop:
            continue
        in_loop[name] = looped
        for line in computations[name][1]:
            body = re.search(r"\bbody=%?([\w.\-]+)", line)
            for callee in re.findall(r"%([\w.\-]+)", line):
                if callee in computations:
                    stack.append((callee, looped or (body is not None and callee == body.group(1))))
    coords = np.stack(np.unravel_index(np.arange(mesh.devices.size), mesh.devices.shape), axis=1)
    replica = mesh.axis_names.index("replica_dcn")
    # The step's own mesh as the HLO prints it.
    mesh_text = ",".join(f"'{axis}'={size}" for axis, size in mesh.shape.items())
    out = []
    for name, (_, lines) in computations.items():
        for line in lines:
            m = re.match(r"(?:ROOT )?%?[\w.\-]+ = (.+?) ([\w\-]+)\((.*)$", line)
            if not m or m.group(2) not in ops:
                continue
            shape, _, attrs = m.groups()
            named = re.search(
                r"replica_groups=mesh\[([^\]]*)\]" r"(?:, device_ids=\(\[([\d,]+)\](?:T\(([\d,]+)\))?\))?\s*\{([^}]*)\}",
                attrs,
            )
            # Iota groups: [groups, size]<=[dims], optionally T(permutation) of the dims.
            iota = re.search(r"replica_groups=\[(\d+),(\d+)\]<=\[([\d,]+)\](?:T\(([\d,]+)\))?", attrs)
            if named and named.group(1) == mesh_text:
                sizes = {axis: int(size) for axis, size in re.findall(r"'(\w+)'=(\d+)", named.group(1))}
                axes = re.findall(r"'(\w+)'", named.group(4))
                crosses = "replica_dcn" in axes
                group_size = int(np.prod([sizes[axis] for axis in axes]))
            elif named:
                # A mesh of XLA's own (axis_0, ...), with the device order in device_ids when it is not the iota.
                sizes = [(axis, int(size)) for axis, size in re.findall(r"'(\w+)'=(\d+)", named.group(1))]
                names = [axis for axis, _ in sizes]
                axes = re.findall(r"'(\w+)'", named.group(4))
                group_size = int(np.prod([size for axis, size in sizes if axis in axes]))
                ids = np.arange(mesh.devices.size)
                if named.group(2):
                    ids = ids.reshape([int(d) for d in named.group(2).split(",")])
                    if named.group(3):
                        ids = ids.transpose([int(d) for d in named.group(3).split(",")])
                ids = ids.reshape([size for _, size in sizes])
                ids = np.moveaxis(ids, [names.index(axis) for axis in axes], list(range(-len(axes), 0)))
                groups = ids.reshape(-1, group_size).tolist()
                crosses = any(len({coords[int(d)][replica] for d in group}) > 1 for group in groups)
            else:
                if iota:
                    count, size, dims, permutation = iota.groups()
                    ids = np.arange(mesh.devices.size).reshape([int(d) for d in dims.split(",")])
                    if permutation:
                        ids = ids.transpose([int(d) for d in permutation.split(",")])
                    groups = ids.reshape(int(count), int(size)).tolist()
                else:
                    groups = re.search(r"replica_groups=\{(.*?)\}\}", attrs)
                    assert groups, line
                    groups = [group.split(",") for group in re.findall(r"\{([\d,]+)", groups.group(0))]
                crosses = any(len({coords[int(d)][replica] for d in group}) > 1 for group in groups)
                group_size = len(groups[0])
            if crosses:
                dims = [[int(d) for d in each.split(",") if d] for each in re.findall(r"\[([\d,]*)\]", shape)]
                elements = sum(int(np.prod(each)) for each in dims)
                if m.group(2) == "reduce-scatter":
                    elements *= group_size
                op_name = re.search(r'op_name="([^"]*)"', attrs)
                scalar = all(not each for each in dims)
                out.append(Reduction(in_loop[name], elements, scalar, op_name.group(1) if op_name else ""))
    return out


def leaves(tree):
    return [np.asarray(x, dtype=np.float32) for x in jax.tree_util.tree_leaves(tree)]


def grad_errors(state, reference):
    # Relative L2 error of every gradient leaf against the reference state's.
    return [
        np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
        for a, b in zip(leaves(state.opt_state), leaves(reference.opt_state), strict=True)
    ]


def assert_matches_whole_batch(cfg, policy, check_grads, mesh_name, *, carry_compute_params=False, **step_args):
    # Four microbatches of one row per device, against the whole batch.
    mesh = _meshes()[mesh_name]
    whole, whole_metrics = step(
        cfg, mesh, policy, microbatches=1, carry_compute_params=carry_compute_params, **step_args
    )
    assert np.sum(whole_metrics["moe/dropped_assignments"]) == 0, mesh_name
    msg = f"{mesh_name} K=4"
    accumulated, metrics = step(
        cfg, mesh, policy, microbatches=4, carry_compute_params=carry_compute_params, **step_args
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
    check_grads(accumulated, whole, mesh_name, msg)


@pytest.mark.multidevice
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    ("router_z_loss_coef", "step_args", "mesh_name"),
    [
        (None, {"batch": BATCH}, "replica4_expert2"),
        # The logit z-loss is weighted by loss weight, the router z-loss averages over valid tokens.
        (0.1, {"batch": PADDED_BATCH, "z_loss_weight": 1e-2}, "replica2_data2_expert2"),
        (0.1, {"batch": UNWEIGHTED_MICROBATCH}, "replica2_data2_expert2"),
    ],
    ids=["unequal_loss_weight", "z_losses_and_padding", "unweighted_microbatch"],
)
def test_microbatches_match_the_whole_batch_in_float32(router_z_loss_coef, step_args, mesh_name):
    # Only the order of the float32 sums changes: a few ulps (2**-23 each); 4e-7 observed.
    cfg = BASE if router_z_loss_coef is None else dataclasses.replace(BASE, router_z_loss_coef=router_z_loss_coef)

    def check_grads(accumulated, whole, name, msg):
        assert max(grad_errors(accumulated, whole)) <= 1e-5, msg

    with use_cpu_devices(8):
        assert_matches_whole_batch(cfg, jmp.get_policy("f32"), check_grads, mesh_name, **step_args)


@pytest.mark.timeout(900)
@pytest.mark.multidevice
def test_microbatches_match_the_whole_batch_with_bfloat16_compute():
    # The forward is the same for every token, so the loss matches to float32 rounding. The gradients do not: each
    # microbatch rounds its own bfloat16 weight gradients, and each device's float32 sum of them is rounded to
    # bfloat16 once more before the single cross-device sum after the loop. The whole batch's bfloat16 gradients are
    # themselves off the float32 ones, by up to 2e-2 (the norm weights, whose gradient sums many terms), so require
    # every microbatched gradient leaf to be no further from its float32 counterpart than the whole batch's, with a
    # margin and a floor of one bfloat16 rounding (2**-8).
    with use_cpu_devices(8):
        policy = jmp.get_policy("p=f32,c=bfloat16")
        cfg = dataclasses.replace(BASE, remat_mode="save_moe")
        # The mesh that also shards the parameters over ``data``.
        mesh_name = "replica2_data2_expert2"
        exact, _ = step(cfg, _meshes()[mesh_name], jmp.get_policy("f32"), microbatches=1)

        def check_grads(accumulated, whole, name, msg):
            bounds = 1.5 * np.asarray(grad_errors(whole, exact)) + 2**-8
            errors = np.asarray(grad_errors(accumulated, exact))
            assert np.all(errors <= bounds), (msg, errors, bounds)

        assert_matches_whole_batch(cfg, policy, check_grads, mesh_name)
        assert_matches_whole_batch(cfg, policy, check_grads, mesh_name, carry_compute_params=True)


@pytest.mark.timeout(900)
@pytest.mark.multidevice
def test_qb_thresholds_are_the_mean_over_microbatches():
    with use_cpu_devices(8):
        policy = jmp.get_policy("f32")
        for qb_balance in ("global", "per_node"):
            cfg = dataclasses.replace(BASE, qb_balance=qb_balance)
            mesh = _meshes()["replica2_data2_expert2"]
            k = 2
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


@pytest.mark.timeout(900)
@pytest.mark.multidevice
def test_gradients_are_reduced_across_replicas_once_after_the_microbatch_loop():
    # Backward ops carry "transpose(" in their op name. In the whole-batch step the expert weights' gradients are
    # reduced across the replicas in every layer of the backward layer scan; with microbatches, no backward reduction
    # of an array across the replicas runs in any loop, only the forward's loss and routing statistics. The scalar
    # cotangent of the loss weight is a backward reduction that may stay in the loop (it does on GPU), so it is not
    # counted as a weight gradient. After the loop, the reductions across the replicas cover every weight's per-device
    # gradient exactly once: in total its element count, plus a few scalars, fewer than the smallest weight has.
    with use_cpu_devices(8):
        policy = jmp.get_policy("p=f32,c=bfloat16")
        for name, mesh in _meshes().items():
            whole = replica_reductions(compiled_step_hlo(BASE, mesh, policy, microbatches=1), mesh)
            assert any(r.in_loop and "transpose(" in r.op_name for r in whole), (name, whole)
            with jax.set_mesh(mesh):
                params = june_train.initial_state(
                    BASE, optimizer=optax.adam(1e-2), mp=policy, key=jax.random.key(0), ema_beta=None
                ).params
                marked = jax.eval_shape(june_train._mark_batch_reduced, params)
            sizes = [
                int(np.prod(x.sharding.shard_shape(x.shape)))
                for path, x in jax.tree_util.tree_leaves_with_path(marked)
                if getattr(path[-1], "name", None) != "router_bias"
            ]
            reductions = replica_reductions(compiled_step_hlo(BASE, mesh, policy, microbatches=4), mesh)
            in_loop = [r for r in reductions if r.in_loop]
            assert in_loop and not any("transpose(" in r.op_name and not r.scalar for r in in_loop), (name, in_loop)
            after_loop = sum(r.elements for r in reductions if not r.in_loop)
            assert sum(sizes) <= after_loop < sum(sizes) + min(sizes), (name, after_loop, sum(sizes), reductions)


@pytest.mark.multidevice
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    ("cfg_changes", "policy", "step_args", "mesh_name", "k"),
    [
        ({}, "f32", {"batch": PADDED_BATCH, "z_loss_weight": 1e-2}, "replica2_data2_expert2", 4),
        ({"router_z_loss_coef": 0.1}, "f32", {"batch": UNWEIGHTED_MICROBATCH}, "replica4_expert2", 4),
        # K=2: a microbatch loop of one iteration before the last microbatch.
        ({"remat_mode": "save_moe"}, "p=f32,c=bfloat16", {}, "replica4_expert2", 2),
        (
            {"remat_mode": "save_moe"},
            "p=f32,c=bfloat16",
            {"carry_compute_params": True},
            "replica2_data2_expert2",
            4,
        ),
    ],
    ids=["f32_z_loss_and_padding", "f32_unweighted_microbatch", "bfloat16", "bfloat16_carry_compute_params"],
)
def test_overlapping_the_last_microbatch_gives_the_same_step(cfg_changes, policy, step_args, mesh_name, k):
    # The per-layer reduction in the last microbatch's backward adds the same float32 accumulator and gradient, casts
    # the sum the same way and reduces it over the same devices as the single reduction after the loop: on CPU the
    # gradients, the loss, the metrics and the next QB thresholds are bit-identical.
    cfg = dataclasses.replace(BASE, **cfg_changes)
    policy = jmp.get_policy(policy)
    with use_cpu_devices(8):
        mesh = _meshes()[mesh_name]
        msg = f"{mesh_name} K={k}"
        hoisted, hoisted_metrics = step(cfg, mesh, policy, microbatches=k, **step_args)
        overlapped, metrics = step(cfg, mesh, policy, microbatches=k, overlap_last_microbatch=True, **step_args)
        assert metrics.keys() == hoisted_metrics.keys(), msg
        for key in metrics:
            np.testing.assert_array_equal(metrics[key], hoisted_metrics[key], err_msg=f"{msg} {key}")
        for a, b in zip(
            jax.tree_util.tree_leaves((overlapped.opt_state, overlapped.pending_qb_betas)),
            jax.tree_util.tree_leaves((hoisted.opt_state, hoisted.pending_qb_betas)),
            strict=True,
        ):
            assert a.dtype == b.dtype and a.sharding == b.sharding, msg
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=msg)


@pytest.mark.multidevice
@pytest.mark.timeout(900)
@pytest.mark.parametrize("replicated_grad_max_bytes", [None, 0], ids=["after_scan", "per_layer"])
def test_overlapping_the_last_microbatch_with_one_expert_shard(replicated_grad_max_bytes):
    # With a size-1 expert axis the expert weights are replicated: small enough, the layer scan itself marks them
    # reduced after the scan (replicated_grad_reduction="after_scan"), and they then take the same path as the other
    # weights; larger ones are reduced per layer. Either way the step is bit-identical to the hoisted one.
    cfg = BASE
    if replicated_grad_max_bytes is not None:
        cfg = dataclasses.replace(BASE, replicated_grad_max_bytes=replicated_grad_max_bytes)
    with use_cpu_devices(8):
        mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=8)
        hoisted, hoisted_metrics = step(cfg, mesh, jmp.get_policy("f32"), microbatches=2)
        overlapped, metrics = step(cfg, mesh, jmp.get_policy("f32"), microbatches=2, overlap_last_microbatch=True)
        for key in metrics:
            np.testing.assert_array_equal(metrics[key], hoisted_metrics[key], err_msg=key)
        overlapped_grads, hoisted_grads = (jax.tree_util.tree_leaves(s.opt_state) for s in (overlapped, hoisted))
        for a, b in zip(overlapped_grads, hoisted_grads, strict=True):
            assert a.dtype == b.dtype and a.sharding == b.sharding
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.timeout(900)
@pytest.mark.multidevice
def test_overlapping_the_last_microbatch_reduces_the_expert_gradients_per_layer():
    # With overlap_last_microbatch, the expert weights' gradients are reduced across the replicas inside the backward
    # layer scan of the last microbatch, which runs after the microbatch loop: one layer's slice of each per
    # iteration, so over the scan each expert weight once. No other backward reduction of an array runs in a loop.
    # After it, there is exactly one reduction for every other weight's gradient. Compiled without the all-reduce
    # combiner, which would merge them with each other and with the forward's routing statistics.
    with use_cpu_devices(8):
        policy = jmp.get_policy("p=f32,c=bfloat16")
        layer_scan = "jit(train_step)/transpose(jvp(Transformer))/while/body/"
        # K=2 on one mesh and K=4 on the other.
        for (name, mesh), k in zip(_meshes().items(), (2, 4), strict=True):
            with jax.set_mesh(mesh):
                params = june_train.initial_state(
                    BASE, optimizer=optax.adam(1e-2), mp=policy, key=jax.random.key(0), ema_beta=None
                ).params
                marked = jax.eval_shape(june_train._mark_batch_reduced, params)
                per_layer = june_train._per_layer_reduced(params)
            expert, other = [], []
            for (path, x), is_expert in zip(
                jax.tree_util.tree_leaves_with_path(marked), jax.tree_util.tree_leaves(per_layer), strict=True
            ):
                if getattr(path[-1], "name", None) != "router_bias":
                    (expert if is_expert else other).append(int(np.prod(x.sharding.shard_shape(x.shape))))
            assert len(expert) == 3, (name, expert)
            layer_slices = [size // BASE.num_layers for size in expert]
            msg = f"{name} K={k}"
            reductions = replica_reductions(
                compiled_step_hlo(
                    BASE, mesh, policy, microbatches=k, overlap_last_microbatch=True, combine_all_reduces=False
                ),
                mesh,
            )
            backward_in_loop = [r for r in reductions if r.in_loop and "transpose(" in r.op_name and not r.scalar]
            assert backward_in_loop and all(r.op_name.startswith(layer_scan) for r in backward_in_loop), (
                msg,
                backward_in_loop,
            )
            assert sum(r.elements for r in backward_in_loop) == sum(layer_slices), (msg, backward_in_loop)
            forward = [r for r in reductions if "jvp(" in r.op_name and "transpose(" not in r.op_name]
            assert forward, (msg, reductions)  # The routing statistics, so the op_name test means something.
            after_loop = [r.elements for r in reductions if not r.in_loop and not r.scalar and r not in forward]
            assert sorted(after_loop) == sorted(other), (msg, reductions)


def test_overlapping_the_last_microbatch_needs_two_microbatches():
    with pytest.raises(ValueError, match="at least 2 microbatches"):
        june_train._make_train_step(
            optax.sgd(1e-2),
            jmp.get_policy("f32"),
            z_loss_weight=0.0,
            ema_beta=None,
            microbatches=1,
            overlap_last_microbatch=True,
        )


@pytest.mark.multidevice
def test_overlapping_the_last_microbatch_needs_stacked_blocks():
    cfg = dataclasses.replace(
        BASE,
        use_array_stacked_blocks=False,
        replicated_grad_reduction=GrugModelConfig.replicated_grad_reduction,
        router_stats_reduction=GrugModelConfig.router_stats_reduction,
    )
    policy = jmp.get_policy("f32")
    optimizer = optax.sgd(1e-2)
    with use_cpu_devices(8), jax.set_mesh(_meshes()["replica4_expert2"]):
        state = jax.eval_shape(
            lambda key: june_train.initial_state(cfg, optimizer=optimizer, mp=policy, key=key, ema_beta=None),
            jax.random.key(0),
        )
        train_step = june_train._make_train_step(
            optimizer, policy, z_loss_weight=0.0, ema_beta=None, microbatches=2, overlap_last_microbatch=True
        )
        with pytest.raises(ValueError, match="stacked layer layout"):
            jax.eval_shape(train_step, state, BATCH)
