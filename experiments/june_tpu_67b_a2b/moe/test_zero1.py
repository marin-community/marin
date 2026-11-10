# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ZeRO-1 over the replicas (``GrugTrainerConfig.zero1``) in the June train step, on eight CPU devices.

With ``zero1`` the expert weights' float32 parameters, optimizer state and EMA are sharded over ``replica_dcn`` as well
as ``expert``; the step reduces each expert gradient into that shard and all-gathers the updated compute-dtype
parameters back to the model's sharding after the update. On the meshes of the gradient accumulation tests
(replica_dcn=4 x expert=2, replica_dcn=2 x data=2 x expert=2), with AdamW and with the production MuonH, the losses and
parameters match the replicated update; the state is 1/replica_dcn per device; the expert weights cross the replicas
once per step each way, after the microbatch loop; and a checkpoint saved with ZeRO-1 restores with it off and on.
With ``gather_in_first_microbatch`` each expert weight is all-gathered once per step, layer by layer in the first
microbatch's forward layer scan, into the compute copy the scan carries, with the same numbers as without it and no
memory for the gathered stack beyond that copy.

On CPU XLA lowers the reduce-scatter to an all-reduce and a slice, so the HLO test counts the elements each
reduction covers rather than requiring a reduce-scatter.
"""

import collections
import dataclasses
import re
import tempfile

import jax
import jmp
import levanter.eval
import numpy as np
import optax
import pytest
from jax.sharding import PartitionSpec as P
from levanter.callbacks import StepInfo
from levanter.callbacks.state_adapter import CallbackStateView
from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.optim.config import AdamConfig
from levanter.testing.devices import use_cpu_devices

from experiments.june_tpu_67b_a2b.moe import train as june_train
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeMuonHConfig
from experiments.june_tpu_67b_a2b.moe.train_test_support import (
    BASE,
    BATCH,
    PADDED_BATCH,
    UNWEIGHTED_MICROBATCH,
    _meshes,
    leaves,
    replica_reductions,
)

# Eight experts, so that they split over expert=2 x replica_dcn=4.
ZERO1_CFG = dataclasses.replace(BASE, num_experts=8, remat_mode="save_moe")
LAYOUTS = {
    "stacked": ZERO1_CFG,
    # QB balanced per node: router_bias and the QB betas are sharded over replica_dcn already, and stay as they are.
    "per_node_qb": dataclasses.replace(ZERO1_CFG, qb_balance="per_node"),
    # One [E, ...] expert weight per layer, which MuonH orthogonalizes per expert on the sharded stack.
    "unstacked": dataclasses.replace(
        ZERO1_CFG,
        use_array_stacked_blocks=False,
        replicated_grad_reduction=GrugModelConfig.replicated_grad_reduction,
        router_stats_reduction=GrugModelConfig.router_stats_reduction,
    ),
}
POLICY = jmp.get_policy("p=f32,c=bfloat16")
# The test meshes each layout's update test runs on: both for the stacked layout.
LAYOUT_MESHES = {
    "stacked": ("replica4_expert2", "replica2_data2_expert2"),
    "per_node_qb": ("replica2_data2_expert2",),
    "unstacked": ("replica4_expert2",),
}
OPTIMIZERS = {
    "adamw": lambda: AdamConfig(learning_rate=1e-2, warmup=0.0, weight_decay=0.1, max_grad_norm=1.0).build(10),
    "muonh": lambda: GrugMoeMuonHConfig(learning_rate=2e-2, adam_lr=1e-2, warmup=0.0).build(10),
}
EMA_BETA = 0.9
GATHERS = ("all-gather", "all-gather-start")


def zero1_state(mesh, optimizer, *, zero1, cfg=ZERO1_CFG):
    with jax.set_mesh(mesh):
        return jax.jit(
            lambda key: june_train.initial_state(
                cfg,
                optimizer=optimizer,
                mp=POLICY,
                key=key,
                ema_beta=EMA_BETA,
                carry_compute_params=True,
                zero1=zero1,
            )
        )(jax.random.key(0))


def zero1_step(optimizer, *, zero1, microbatches=4, overlap=False, gather=False):
    return june_train._make_train_step(
        optimizer,
        POLICY,
        z_loss_weight=0.0,
        ema_beta=EMA_BETA,
        microbatches=microbatches,
        zero1=zero1,
        overlap_last_microbatch=overlap,
        gather_in_first_microbatch=gather,
    )


# (zero1, gather_in_first_microbatch) of the three ways to run the step; the last two share the state's layout.
MODES = {"replicated": (False, False), "zero1": (True, False), "gather_in_first_microbatch": (True, True)}


def is_expert(path):
    return "expert_mlp" in jax.tree_util.keystr(path)


def specs(tree):
    return {jax.tree_util.keystr(path): x.sharding.spec for path, x in jax.tree_util.tree_leaves_with_path(tree)}


def zero1_spec(spec):
    # The expert dimension of an expert weight also split over the replicas.
    entries = list(spec)
    i = entries.index("expert")
    entries[i] = ("expert", "replica_dcn")
    return P(*entries)


def relative_errors(tree, reference):
    return {
        jax.tree_util.keystr(path): np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
        for (path, a), b in zip(
            jax.tree_util.tree_leaves_with_path(jax.tree.map(lambda x: np.asarray(x, np.float32), tree)),
            leaves(reference),
            strict=True,
        )
    }


@pytest.mark.timeout(900)
@pytest.mark.parametrize("optimizer_name", ["adamw", "muonh"])
@pytest.mark.parametrize("layout", ["stacked", "per_node_qb", "unstacked"])
@pytest.mark.multidevice
def test_zero1_matches_the_replicated_update(layout, optimizer_name):
    # After one step only the order of the optimizer's norm sums differs (the global clip norm, MuonH's per-layer
    # hyperball norm), a few float32 ulps: below 1e-6 relative, 4.6e-7 observed. Later steps start from compute copies
    # that can round differently to bfloat16 and train on, so after three steps every leaf need only be within one
    # bfloat16 rounding (2**-8) of the replicated run's; MuonH, which normalizes each update, shows 4.2e-4. The losses,
    # means over the batch, move far less: they matched exactly. Only the expert weights' parameters move to the
    # replicas: with per-node QB, router_bias and the betas keep their sharding.
    with use_cpu_devices(8):
        for name in LAYOUT_MESHES[layout]:
            mesh = _meshes()[name]
            optimizer = OPTIMIZERS[optimizer_name]()
            runs, replicated_specs, replicated_compute_specs = {}, {}, {}
            for zero1 in (False, True):
                state = zero1_state(mesh, optimizer, zero1=zero1, cfg=LAYOUTS[layout])
                train_step = zero1_step(optimizer, zero1=zero1)
                losses, after_one_step = [], None
                with jax.set_mesh(mesh):
                    for _ in range(3):
                        state, metrics, _ = train_step(state, BATCH)
                        losses.append(float(metrics["train/loss"]))
                        if after_one_step is None:
                            after_one_step = jax.tree.map(np.asarray, (state.params, state.ema_params))
                runs[zero1] = (losses, after_one_step, jax.tree.map(np.asarray, (state.params, state.ema_params)))
                if zero1:
                    for path, spec in specs(state.params).items():
                        expected = zero1_spec(replicated_specs[path]) if "expert_mlp" in path else replicated_specs[path]
                        assert spec == expected, (name, path, spec)
                    assert specs(state.compute_params) == replicated_compute_specs, name
                else:
                    replicated_specs, replicated_compute_specs = specs(state.params), specs(state.compute_params)
            (losses, one, three), (zero1_losses, zero1_one, zero1_three) = runs[False], runs[True]
            np.testing.assert_allclose(zero1_losses, losses, rtol=1e-5, err_msg=name)
            errors = relative_errors(zero1_one, one)
            assert max(errors.values()) <= 1e-6, (name, errors)
            errors = relative_errors(zero1_three, three)
            assert max(errors.values()) <= 2**-8, (name, errors)


@pytest.mark.timeout(900)
@pytest.mark.multidevice
def test_zero1_shards_the_expert_state_over_the_replicas():
    # The expert weights' parameters, EMA and every optimizer state leaf of their shape are sharded over the replicas
    # as well, so each device keeps 1/replica_dcn of them; everything else keeps the replicated run's sharding, and
    # the compute copy keeps the model's sharding after the update.
    with use_cpu_devices(8):
        for optimizer_name, make_optimizer in OPTIMIZERS.items():
            optimizer = make_optimizer()
            for name, mesh in _meshes().items():
                msg = f"{name} {optimizer_name}"
                replicas = mesh.shape["replica_dcn"]
                with jax.set_mesh(mesh):
                    states = {}
                    for zero1 in (False, True):
                        initial = jax.eval_shape(
                            lambda key, optimizer=optimizer, zero1=zero1: june_train.initial_state(
                                ZERO1_CFG,
                                optimizer=optimizer,
                                mp=POLICY,
                                key=key,
                                ema_beta=EMA_BETA,
                                carry_compute_params=True,
                                zero1=zero1,
                            ),
                            jax.random.key(0),
                        )
                        states[zero1] = jax.eval_shape(zero1_step(optimizer, zero1=zero1), initial, BATCH)[0]
                replicated, sharded = states[False], states[True]
                expert_shapes = {
                    x.shape for path, x in jax.tree_util.tree_leaves_with_path(replicated.params) if is_expert(path)
                }
                assert len(expert_shapes) == 2, expert_shapes
                for tree in ("params", "ema_params"):
                    for path, before, after in zip(
                        specs(getattr(replicated, tree)),
                        specs(getattr(replicated, tree)).values(),
                        specs(getattr(sharded, tree)).values(),
                        strict=True,
                    ):
                        expected = zero1_spec(before) if "expert_mlp" in path else before
                        assert after == expected, (msg, tree, path, before, after)
                assert specs(sharded.compute_params) == specs(replicated.compute_params), msg

                def state_bytes(tree, expert, expert_shapes=expert_shapes):
                    return sum(
                        int(np.prod(x.sharding.shard_shape(x.shape))) * x.dtype.itemsize
                        for x in jax.tree.leaves(tree)
                        if hasattr(x, "sharding") and (x.shape in expert_shapes) == expert
                    )

                for expert in (False, True):
                    before = state_bytes((replicated.params, replicated.opt_state, replicated.ema_params), expert)
                    after = state_bytes((sharded.params, sharded.opt_state, sharded.ema_params), expert)
                    assert after * (replicas if expert else 1) == before, (msg, expert, before, after)


@pytest.mark.timeout(900)
@pytest.mark.multidevice
def test_zero1_crosses_the_replicas_once_per_step_after_the_microbatch_loop():
    # On top of the replicated run's collectives across the replicas, each expert weight is all-gathered once per step
    # after the microbatch loop, its compute copy's shard in full; no gather runs in a loop that the replicated run does
    # not have. The gradient reductions across the replicas still cover every weight's gradient once, after the loop.
    # The step is compiled for the state one step has written, so a compute copy left sharded over the replicas would
    # show up as gathers in the layer loop.
    with use_cpu_devices(8):
        # K=2 on one mesh and K=4 on the other.
        for (name, mesh), k in zip(_meshes().items(), (2, 4), strict=True):
            optimizer = optax.adam(1e-2)
            with jax.set_mesh(mesh):
                marked = jax.eval_shape(
                    june_train._mark_batch_reduced, zero1_state(mesh, optimizer, zero1=False).compute_params
                )
            sizes = [
                int(np.prod(x.sharding.shard_shape(x.shape)))
                for path, x in jax.tree_util.tree_leaves_with_path(marked)
                if getattr(path[-1], "name", None) != "router_bias"
            ]
            msg = f"{name} K={k}"
            gathers, reductions = {}, {}
            for zero1 in (False, True):
                train_step = zero1_step(optimizer, zero1=zero1, microbatches=k)
                with jax.set_mesh(mesh):
                    state, _, _ = train_step(zero1_state(mesh, optimizer, zero1=zero1), BATCH)
                    hlo = train_step.lower(state, BATCH).compile().as_text()
                gathers[zero1] = replica_reductions(hlo, mesh, GATHERS)
                reductions[zero1] = replica_reductions(hlo, mesh)
            in_loop = [sorted((r.elements, r.op_name) for r in gathers[z] if r.in_loop) for z in (False, True)]
            assert in_loop[1] == in_loop[0], (msg, in_loop)
            added = collections.Counter(r.elements for r in gathers[True] if not r.in_loop)
            added.subtract(r.elements for r in gathers[False] if not r.in_loop)
            expert_shards = collections.Counter(
                int(np.prod(x.sharding.shard_shape(x.shape)))
                for path, x in jax.tree_util.tree_leaves_with_path(state.compute_params)
                if is_expert(path)
            )
            assert +added == expert_shards and -added == collections.Counter(), (msg, added, expert_shards)
            in_loop = [r for r in reductions[True] if r.in_loop]
            assert not any("transpose(" in r.op_name and not r.scalar for r in in_loop), (msg, in_loop)
            after_loop = sum(r.elements for r in reductions[True] if not r.in_loop)
            assert sum(sizes) <= after_loop < sum(sizes) + min(sizes), (msg, after_loop, sum(sizes))


@pytest.mark.timeout(900)
@pytest.mark.parametrize("saved_with", ["replicated", "zero1", "gather_in_first_microbatch"])
@pytest.mark.multidevice
def test_zero1_checkpoint_restores_with_zero1_off_and_on(saved_with):
    # The checkpoint keeps the tree and the global shapes, so the exemplar's sharding decides what each device reads:
    # a run can turn ZeRO-1 or gather_in_first_microbatch on or off when it resumes. The compute copy rebuilt from the
    # restored parameters has the exemplar's sharding, the model's, and the parameters' values, also after a
    # gather_in_first_microbatch step, whose compute copy holds the weights that step ran with.
    with use_cpu_devices(8):
        mesh = _meshes()["replica4_expert2"]
        optimizer = OPTIMIZERS["muonh"]()
        saved_zero1, saved_gather = MODES[saved_with]
        with jax.set_mesh(mesh):
            state, _, _ = zero1_step(optimizer, zero1=saved_zero1, gather=saved_gather)(
                zero1_state(mesh, optimizer, zero1=saved_zero1), BATCH
            )
        saved = jax.tree.map(np.asarray, state)
        next_losses = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            with jax.set_mesh(mesh):
                save_checkpoint(june_train._checkpointed(state), step=1, checkpoint_path=tmpdir)
            for mode, (zero1, gather) in MODES.items():
                exemplar = zero1_state(mesh, optimizer, zero1=zero1)
                with jax.set_mesh(mesh):
                    # As _run_grug_local restores: without the compute copy, then rebuilt in the initial copy's sharding.
                    restored = load_checkpoint(june_train._checkpointed(exemplar), tmpdir, mesh=mesh)
                    rebuilt = june_train.rebuilt_compute_params(
                        dataclasses.replace(restored, compute_params=exemplar.compute_params), POLICY
                    )
                assert specs(restored) == specs(june_train._checkpointed(exemplar)), mode
                assert specs(rebuilt.compute_params) == specs(exemplar.compute_params), mode
                for a, b in zip(
                    jax.tree.leaves(restored), jax.tree.leaves(june_train._checkpointed(saved)), strict=True
                ):
                    np.testing.assert_array_equal(np.asarray(a), b, err_msg=mode)
                for a, b in zip(
                    jax.tree.leaves(rebuilt.compute_params),
                    jax.tree.leaves(POLICY.cast_to_compute(saved.params)),
                    strict=True,
                ):
                    np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=mode)
                with jax.set_mesh(mesh):
                    _, metrics, _ = zero1_step(optimizer, zero1=zero1, gather=gather)(rebuilt, BATCH)
                next_losses[mode] = float(metrics["train/loss"])
            for mode, loss in next_losses.items():
                np.testing.assert_allclose(loss, next_losses["replicated"], rtol=1e-6, err_msg=mode)


@pytest.mark.timeout(900)
@pytest.mark.multidevice
def test_zero1_eval_hooks_gather_each_model_once_per_step(monkeypatch):
    # With ZeRO-1 the eval hooks evaluate float32 models in the model's own sharding, one model per hook, so a sharded
    # master never reaches the evaluator (which would gather it in every layer of every eval batch) and only one
    # gathered model is live at a time. A step evaluated already gathers nothing.
    with use_cpu_devices(8):
        evaluated = []

        def fake_eval_model(evaluator, model, prefix=""):
            evaluated.append((prefix, specs(model), {x.dtype for x in jax.tree.leaves(model)}))
            return {}

        monkeypatch.setattr(levanter.eval, "eval_model", fake_eval_model)
        mesh = _meshes()["replica4_expert2"]
        state = zero1_state(mesh, optax.adam(1e-2), zero1=True)
        model_specs = specs(zero1_state(mesh, optax.adam(1e-2), zero1=False).compute_params)
        trainer = june_train.GrugTrainerConfig(carry_compute_params=True, zero1=True, gather_in_first_microbatch=True)
        with jax.set_mesh(mesh):
            # The shardings _run_grug_local gives the eval hooks.
            model_shardings = june_train._eval_model_shardings(trainer, ZERO1_CFG, jax.random.key(0))
            unsharded = dataclasses.replace(trainer, zero1=False, gather_in_first_microbatch=False)
            assert june_train._eval_model_shardings(unsharded, ZERO1_CFG, jax.random.key(0)) is None
        sharding_specs = {
            jax.tree_util.keystr(path): s.spec for path, s in jax.tree_util.tree_leaves_with_path(model_shardings)
        }
        assert sharding_specs == model_specs
        assert specs(state.params) != model_specs
        for current, ema in ((True, True), (True, False), (False, True)):
            hooks = june_train._eval_hooks(
                None, prefix="eval", eval_current=current, eval_ema=ema, model_shardings=model_shardings
            )
            assert len(hooks) == current + ema, (current, ema, hooks)
            evaluated.clear()
            view = CallbackStateView(
                step=state.step + 1, model=state.params, eval_model=state.ema_params, opt_state=state.opt_state
            )
            with jax.set_mesh(mesh):
                for hook in hooks:
                    hook(StepInfo(state=view, loss=0.0, step_duration=0.0))
                # The same step again, forced, with models that cannot be gathered: nothing runs.
                unusable = dataclasses.replace(view, model=object(), eval_model=object())
                for hook in hooks:
                    hook(StepInfo(state=unusable, loss=0.0, step_duration=0.0), force=True)
            assert [prefix for prefix, _, _ in evaluated] == ["eval"] * current + ["eval/ema"] * ema, evaluated
            for prefix, evaluated_specs, dtypes in evaluated:
                assert evaluated_specs == model_specs, prefix
                assert dtypes == {np.dtype(np.float32)}, (prefix, dtypes)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("optimizer_name", ["adamw", "muonh"])
@pytest.mark.multidevice
def test_zero1_with_the_overlapped_last_microbatch_matches_the_replicated_update(optimizer_name):
    # With overlap_last_microbatch the expert gradients are reduce-scattered layer by layer in the last microbatch's
    # backward, through the gradient sinks; the step matches the replicated overlapped step as without the overlap.
    with use_cpu_devices(8):
        name = {"adamw": "replica4_expert2", "muonh": "replica2_data2_expert2"}[optimizer_name]
        mesh = _meshes()[name]
        optimizer = OPTIMIZERS[optimizer_name]()
        runs = {}
        for zero1 in (False, True):
            state = zero1_state(mesh, optimizer, zero1=zero1)
            train_step = zero1_step(optimizer, zero1=zero1, overlap=True)
            losses, after_one_step = [], None
            with jax.set_mesh(mesh):
                for _ in range(3):
                    state, metrics, _ = train_step(state, BATCH)
                    losses.append(float(metrics["train/loss"]))
                    if after_one_step is None:
                        after_one_step = jax.tree.map(np.asarray, (state.params, state.ema_params))
            runs[zero1] = (losses, after_one_step, jax.tree.map(np.asarray, (state.params, state.ema_params)))
        (losses, one, three), (zero1_losses, zero1_one, zero1_three) = runs[False], runs[True]
        np.testing.assert_allclose(zero1_losses, losses, rtol=1e-5, err_msg=name)
        errors = relative_errors(zero1_one, one)
        assert max(errors.values()) <= 1e-6, (name, errors)
        errors = relative_errors(zero1_three, three)
        assert max(errors.values()) <= 2**-8, (name, errors)


@pytest.mark.timeout(900)
@pytest.mark.multidevice
def test_zero1_with_the_overlapped_last_microbatch_reduces_into_the_shard_per_layer():
    # The per-layer reductions stay in the last microbatch's backward layer scan, one layer's slice of each expert
    # weight per iteration, and that scan already outputs the expert gradients in their ZeRO-1 shard: the reduction is
    # a reduce-scatter (on CPU an all-reduce and a slice inside the loop), not an all-reduce sliced after the scan.
    # The compute copy is all-gathered once per expert weight after the loops, as without the overlap. Compiled
    # without the all-reduce combiner, so every reduction keeps its own op_name.
    with use_cpu_devices(8):
        layer_scan = "jit(train_step)/transpose(jvp(Transformer))/while/body/"
        optimizer = optax.adam(1e-2)
        options = {"xla_disable_hlo_passes": "cpu-all-reduce-combiner"}
        # K=2 on one mesh and K=4 on the other.
        for (name, mesh), k in zip(_meshes().items(), (2, 4), strict=True):
            msg = f"{name} K={k}"
            hlos = {}
            for zero1 in (False, True):
                train_step = zero1_step(optimizer, zero1=zero1, microbatches=k, overlap=True)
                with jax.set_mesh(mesh):
                    state, _, _ = train_step(zero1_state(mesh, optimizer, zero1=zero1), BATCH)
                    hlos[zero1] = train_step.lower(state, BATCH).compile(compiler_options=options).as_text()
            expert_compute = {
                jax.tree_util.keystr(path): x.sharding.shard_shape(x.shape)
                for path, x in jax.tree_util.tree_leaves_with_path(state.compute_params)
                if is_expert(path)
            }
            expert_zero1 = {
                jax.tree_util.keystr(path): x.sharding.shard_shape(x.shape)
                for path, x in jax.tree_util.tree_leaves_with_path(state.params)
                if is_expert(path)
            }
            # A layer's slice of each expert gradient as the backward reduces it: all-gathered over ``data``
            # (FSDP) by the marking, like the other weights' gradients.
            with jax.set_mesh(mesh):
                marked = jax.eval_shape(june_train._mark_batch_reduced, state.compute_params)
            layer_slices = sorted(
                int(np.prod(x.sharding.shard_shape(x.shape)[1:]))
                for path, x in jax.tree_util.tree_leaves_with_path(marked)
                if is_expert(path)
            )
            reductions = {z: replica_reductions(hlos[z], mesh) for z in (False, True)}
            per_layer = [r for r in reductions[True] if r.in_loop and "transpose(" in r.op_name and not r.scalar]
            assert per_layer and all(r.op_name.startswith(layer_scan) for r in per_layer), (msg, per_layer)
            assert sorted(r.elements for r in per_layer) == layer_slices, (msg, per_layer)
            after = [sorted(r.elements for r in reductions[z] if not r.in_loop and not r.scalar) for z in (0, 1)]
            assert after[1] == after[0], (msg, after)
            # The while loops whose bodies run the per-layer reductions output the ZeRO-1 shard of every expert
            # gradient (bfloat16, the compute dtype) and never the compute shard.
            computations = dict(re.findall(r"(?m)^%?([\w.\-]+) .*\{$\n((?:[ ].*\n?)*)", hlos[True]))
            loop_outputs = []
            for shape, body in re.findall(r"= (.*?) while\(.*?body=%?([\w.\-]+)", hlos[True]):
                if layer_scan in computations.get(body, "") and "all-reduce" in computations.get(body, ""):
                    loop_outputs.append(shape)
            assert loop_outputs, msg
            for path, shard in expert_zero1.items():
                zero1_shape = "bf16[" + ",".join(map(str, shard)) + "]"
                assert any(zero1_shape in out for out in loop_outputs), (msg, path, zero1_shape, loop_outputs)
            gathers = {z: replica_reductions(hlos[z], mesh, GATHERS) for z in (False, True)}
            in_loop = [sorted(r.elements for r in gathers[z] if r.in_loop) for z in (False, True)]
            assert in_loop[1] == in_loop[0], (msg, in_loop)
            added = collections.Counter(r.elements for r in gathers[True] if not r.in_loop)
            added.subtract(r.elements for r in gathers[False] if not r.in_loop)
            expert_shards = collections.Counter(int(np.prod(shape)) for shape in expert_compute.values())
            assert +added == expert_shards and -added == collections.Counter(), (msg, added, expert_shards)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("optimizer_name", ["adamw", "muonh"])
@pytest.mark.parametrize("layout", ["stacked", "per_node_qb"])
@pytest.mark.multidevice
def test_gather_in_first_microbatch_matches_zero1(layout, optimizer_name):
    # Gathering the expert weights in the first microbatch's forward rather than at the end of the step does the same
    # arithmetic in the same order: three steps match ZeRO-1 without it bit for bit, with and without the overlapped
    # last microbatch, and with K=2, where no microbatch loop runs between the first and the overlapped last
    # microbatch. Also with padding, and with a first microbatch without loss weight, whose gradients start the
    # accumulators. The step gathers into the compute copy it carries, in the model's sharding: afterwards its expert
    # weights are those the step ran with, the cast of the parameters it started from, bit for bit, and its other
    # weights the cast of the updated parameters, as without it.
    with use_cpu_devices(8):
        cases = [(name, k, overlap, "BATCH") for name in _meshes() for k, overlap in ((4, False), (4, True), (2, True))]
        cases += [
            ("replica4_expert2", 4, True, "UNWEIGHTED_MICROBATCH"),
            ("replica2_data2_expert2", 4, False, "PADDED_BATCH"),
        ]
        for name, k, overlap, batch_name in cases:
            batches = {"BATCH": BATCH, "PADDED_BATCH": PADDED_BATCH, "UNWEIGHTED_MICROBATCH": UNWEIGHTED_MICROBATCH}
            mesh, batch = _meshes()[name], batches[batch_name]
            msg = f"{name} K={k} overlap={overlap} {batch_name}"
            optimizer = OPTIMIZERS[optimizer_name]()
            cfg = LAYOUTS[layout]
            runs, specs_by_mode = {}, {}
            for gather in (False, True):
                state = zero1_state(mesh, optimizer, zero1=True, cfg=cfg)
                train_step = zero1_step(optimizer, zero1=True, microbatches=k, overlap=overlap, gather=gather)
                losses = []
                with jax.set_mesh(mesh):
                    for _ in range(3):
                        started_from = POLICY.cast_to_compute(jax.tree.map(np.asarray, state.params))
                        state, metrics, _ = train_step(state, batch)
                        losses.append(float(metrics["train/loss"]))
                        updated = POLICY.cast_to_compute(jax.tree.map(np.asarray, state.params))
                        for (path, copy), before, after in zip(
                            jax.tree_util.tree_leaves_with_path(state.compute_params),
                            jax.tree.leaves(started_from),
                            jax.tree.leaves(updated),
                            strict=True,
                        ):
                            expected = before if gather and is_expert(path) else after
                            np.testing.assert_array_equal(
                                np.asarray(copy), np.asarray(expected), err_msg=f"{msg} {jax.tree_util.keystr(path)}"
                            )
                runs[gather] = (losses, jax.tree.map(np.asarray, (state.params, state.ema_params, state.opt_state)))
                specs_by_mode[gather] = specs(state.compute_params)
            assert specs_by_mode[True] == specs_by_mode[False], msg
            assert runs[True][0] == runs[False][0], (msg, runs[True][0], runs[False][0])
            for a, b in zip(jax.tree.leaves(runs[True][1]), jax.tree.leaves(runs[False][1]), strict=True):
                np.testing.assert_array_equal(a, b, err_msg=msg)


@pytest.mark.timeout(900)
@pytest.mark.multidevice
@pytest.mark.skipif(
    jax.__version_info__ < (0, 12),
    reason="JAX 0.11's CPU buffer assignment keeps part of the gathered stack live next to the compute copy",
)
def test_gather_in_first_microbatch_keeps_the_gathered_stack_in_the_carried_compute_copy():
    # The first microbatch's layer scan writes the gathered expert weights into the compute copy it carries, which the
    # state donates, rather than returning them as a separate stacked output that would stay live through the other
    # microbatches. So the step's temporaries are those of ZeRO-1 without it, up to a few KiB that do not grow with the
    # weights (3.3 KiB here); that bound is the real check. The arguments are equal by construction (both build the
    # same state) and are kept as a regression pin: returning the gathered stack instead adds about the whole
    # stack to the temporaries (577 of 576 KiB on replica4_expert2), and carrying the compute copy in the ZeRO-1 shard
    # takes the gathered stack's bytes less one shard off the arguments. The first microbatch's backward still reads
    # its layers' gathered weights from its own residuals, but it runs without the accumulators the later microbatches
    # hold. Wide experts make the expert weights most of the memory.
    with use_cpu_devices(8):
        cfg = dataclasses.replace(ZERO1_CFG, intermediate_dim=512)
        optimizer = optax.adam(1e-2)
        for name, mesh in _meshes().items():
            stack = sum(
                int(np.prod(x.sharding.shard_shape(x.shape))) * x.dtype.itemsize
                for path, x in jax.tree_util.tree_leaves_with_path(
                    zero1_state(mesh, optimizer, zero1=True, cfg=cfg).compute_params
                )
                if is_expert(path)
            )
            for k, overlap in ((4, False), (4, True), (2, True)):
                msg = f"{name} K={k} overlap={overlap}"
                memory = {}
                for gather in (False, True):
                    train_step = zero1_step(optimizer, zero1=True, microbatches=k, overlap=overlap, gather=gather)
                    with jax.set_mesh(mesh):
                        state, _, _ = train_step(zero1_state(mesh, optimizer, zero1=True, cfg=cfg), BATCH)
                        memory[gather] = train_step.lower(state, BATCH).compile().memory_analysis()
                assert memory[True].argument_size_in_bytes == memory[False].argument_size_in_bytes, msg
                added = memory[True].temp_size_in_bytes - memory[False].temp_size_in_bytes
                assert added < stack // 16, (msg, added, stack)


@pytest.mark.timeout(900)
@pytest.mark.multidevice
@pytest.mark.skipif(
    jax.__version_info__ < (0, 12),
    reason="JAX 0.11 gathers expert weight slices again in the first microbatch's backward",
)
def test_gather_in_first_microbatch_gathers_each_expert_weight_once_in_the_first_forward():
    # Each expert weight is all-gathered once per step, one layer's slice per iteration of the first microbatch's
    # forward layer scan, and never in a backward or after the loops, where ZeRO-1 without it gathers the updated
    # compute copy. Gathers are matched to ZeRO-1's by place and op_name, with the microbatch loop's prefix dropped
    # (the first microbatch now runs outside the loop): the new ones are the expert weights' layer slices, and the
    # only ones gone are ZeRO-1's end-of-step gathers. The gradients are reduced as without it. Compiled without the
    # all-reduce combiner, so every reduction keeps its own op_name.
    with use_cpu_devices(8):
        first_forward = "jit(train_step)/jvp(Transformer)/while/body/"
        microbatch_loop = "jit(train_step)/while/body/"
        optimizer = optax.adam(1e-2)
        options = {"xla_disable_hlo_passes": "cpu-all-reduce-combiner"}

        def unlooped(r):
            name = r.op_name.replace(microbatch_loop, "jit(train_step)/", 1)
            return (r.in_loop, r.elements, name)

        def gradient_reductions(hlo, mesh):
            # Not the forward's statistics: the reductions in a backward or after the loops.
            return collections.Counter(
                (r.in_loop, r.elements)
                for r in replica_reductions(hlo, mesh)
                if not r.scalar and ("transpose(" in r.op_name or "jvp(Transformer)" not in r.op_name)
            )

        for name, mesh in _meshes().items():
            for k, overlap in ((4, False), (4, True), (2, True)):
                msg = f"{name} K={k} overlap={overlap}"
                hlos, states = {}, {}
                for gather in (False, True):
                    train_step = zero1_step(optimizer, zero1=True, microbatches=k, overlap=overlap, gather=gather)
                    with jax.set_mesh(mesh):
                        state, _, _ = train_step(zero1_state(mesh, optimizer, zero1=True), BATCH)
                        hlos[gather] = train_step.lower(state, BATCH).compile(compiler_options=options).as_text()
                    states[gather] = state
                # The expert weights' compute shards, which ZeRO-1 gathers after the loops.
                expert_shards = [
                    x.sharding.shard_shape(x.shape)
                    for path, x in jax.tree_util.tree_leaves_with_path(states[False].compute_params)
                    if is_expert(path)
                ]
                assert expert_shards, msg
                listed = {z: [unlooped(r) for r in replica_reductions(hlos[z], mesh, GATHERS)] for z in (False, True)}
                gathers = {z: set(listed[z]) for z in (False, True)}
                added = [g for g in listed[True] if g not in gathers[False]]
                assert all(
                    in_loop and op_name.startswith(first_forward) and "transpose(" not in op_name
                    for in_loop, _, op_name in added
                ), (msg, added)
                # A layer's slice of each expert weight as the forward uses it: marked, so gathered over ``data`` too.
                with jax.set_mesh(mesh):
                    marked = jax.eval_shape(june_train._mark_batch_reduced, states[False].compute_params)
                layer_slices = sorted(
                    int(np.prod(x.sharding.shard_shape(x.shape)[1:]))
                    for path, x in jax.tree_util.tree_leaves_with_path(marked)
                    if is_expert(path)
                )
                assert sorted(elements for _, elements, _ in added) == layer_slices, (msg, added, layer_slices)
                end_of_step = {
                    (False, int(np.prod(shard)), "jit(train_step)/convert_element_type") for shard in expert_shards
                }
                assert end_of_step <= gathers[False], (msg, gathers[False])
                assert gathers[False] - gathers[True] == end_of_step, (msg, gathers[False] - gathers[True])
                assert gradient_reductions(hlos[True], mesh) == gradient_reductions(hlos[False], mesh), msg
