# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reduced multi-process checkpoint parity test; run save then resume in fresh gangs."""

import argparse
import json
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from iris.runtime.jax_init import initialize_jax
from jax.experimental import multihost_utils
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.checkpoint import save_checkpoint as save_levanter_checkpoint
from levanter.data.text.examples import GrugLmExample
from levanter.mpmd_checkpoint import checkpoint_arrays
from levanter.pipeline import reshape_batch_into_microbatches
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.moe_pipeline.checkpoint import GrugMoeCheckpointState, restore_checkpoint, save_checkpoint
from experiments.grug.moe_pipeline.model import BATCH_AXES, GrugModelConfig, Transformer
from experiments.grug.moe_pipeline.pipeline import (
    TRAIN_LOSS_KEY,
    AutomaticPipelineSchedule,
    GrugMoePipelineConfig,
    automatic_stage_to_mpmd_indices,
    initialize_mpmd_automatic_pipeline_state,
    make_automatic_pipeline_step,
    make_pipeline_mesh,
    prepare_automatic_mpmd_step,
)

_RELATIVE_L2_TOLERANCE = 0.002


def _assert_optimizer_counts(state, completed_steps: int) -> None:
    optimizers = (state.opt_state,) if isinstance(state, GrugMoeCheckpointState) else state.opt_state
    for stage_optimizer in optimizers:
        count = stage_optimizer[0].count
        if not isinstance(count, jax.Array):
            count = count.to_mpmd_local_array
        if count is not None:
            assert int(count) == completed_steps, (int(count), completed_steps)


def _compare(actual, expected) -> float:
    """Assert continuation parity and return the largest floating-leaf relative L2 error."""
    actual_leaves, actual_tree = jax.tree.flatten(checkpoint_arrays(actual))
    expected_leaves, expected_tree = jax.tree.flatten(checkpoint_arrays(expected))
    assert actual_tree == expected_tree
    errors = []
    for value, reference in zip(actual_leaves, expected_leaves, strict=True):
        error = np.zeros(3, dtype=np.float64)
        for shard, expected_shard in zip(value.addressable_shards, reference.addressable_shards, strict=True):
            observed = np.asarray(shard.data)
            wanted = np.asarray(expected_shard.data)
            if np.issubdtype(wanted.dtype, np.integer):
                error[2] += np.count_nonzero(observed != wanted)
            else:
                difference = observed.astype(np.float64) - wanted.astype(np.float64)
                error[0] += np.sum(difference**2)
                error[1] += np.sum(wanted.astype(np.float64) ** 2)
        errors.append(error)
    totals = np.asarray(multihost_utils.process_allgather(np.asarray(errors))).reshape(-1, len(errors), 3).sum(axis=0)
    assert np.all(totals[:, 2] == 0), "integer checkpoint state differs"
    relative = np.sqrt(totals[:, 0] / np.maximum(totals[:, 1], np.finfo(np.float64).tiny))
    assert np.all(relative <= _RELATIVE_L2_TOLERANCE), f"per-leaf relative L2 errors: {relative.tolist()}"
    return float(relative.max())


def _fsdp_step(optimizer, policy, mesh):
    @jax.jit
    def step(state, batches, denominator):
        def loss_fn(params, batch):
            for index, beta in enumerate(state.pending_qb_betas):
                params = eqx.tree_at(
                    lambda model, index=index: model.blocks[index].mlp.router_bias,
                    params,
                    -beta + jnp.mean(beta),
                    is_leaf=lambda value: value is None,
                )
            return policy.cast_to_compute(params).next_token_loss(
                batch.tokens, batch.loss_weight, return_router_metrics=True
            )

        gradients = jax.tree.map(jnp.zeros_like, state.params)
        pending = tuple(jnp.zeros_like(beta) for beta in state.pending_qb_betas)
        loss = jnp.array(0.0)
        for index in range(batches.tokens.shape[0]):
            batch = jax.tree.map(lambda value, index=index: value[index], batches)
            (batch_loss, metrics), batch_gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch)
            gradients = jax.tree.map(jnp.add, gradients, batch_gradients)
            pending = tuple(beta + update for beta, update in zip(pending, metrics["qb_beta_per_layer"], strict=True))
            loss += batch_loss
        microbatches = batches.tokens.shape[0]
        gradients = jax.tree.map(lambda value: value / microbatches, gradients)
        updates, opt_state = optimizer.update(gradients, state.opt_state, state.params)
        return replace(
            state,
            params=eqx.apply_updates(state.params, updates),
            opt_state=opt_state,
            pending_qb_betas=tuple(beta / microbatches for beta in pending),
        ), {TRAIN_LOSS_KEY: loss / microbatches}

    def run(state, batches, denominator):
        with jax.set_mesh(mesh):
            return step(state, batches, denominator)

    return run


def _global_loss(value) -> float:
    if not isinstance(value, jax.Array):
        value = value.to_mpmd_local_array
    local = np.nan if value is None else float(value)
    losses = np.asarray(multihost_utils.process_allgather(np.array(local)))
    return float(losses[np.isfinite(losses)][0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--mode", choices=("fsdp", "pp"), default="pp")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--schedule", type=AutomaticPipelineSchedule, choices=list(AutomaticPipelineSchedule), required=True
    )
    parser.add_argument("--phase", choices=("save", "resume"), required=True)
    args = parser.parse_args()
    initialize_jax()
    assert jax.process_count() == 4 and jax.local_device_count() == 8, "requires four H100x8 processes"
    physical_stages = 2
    stages = 4 if args.schedule == AutomaticPipelineSchedule.DUALPIPE_V else 2
    config = GrugMoePipelineConfig(
        stages=stages, physical_stages=physical_stages if stages == 4 else None, microbatches=4
    )
    if args.mode == "pp":
        mesh, mpmd_mesh = make_pipeline_mesh(config, expert_axis_size=8, replica_axis_size=2)
    else:
        mesh = Mesh(
            np.array(jax.devices()).reshape(1, 4, 8, 1),
            (*BATCH_AXES, "model"),
            axis_types=(AxisType.Explicit,) * 4,
        )
    model_config = GrugModelConfig(
        vocab_size=256,
        hidden_dim=128,
        intermediate_dim=128,
        shared_expert_intermediate_dim=128,
        num_layers=4,
        num_experts=16,
        num_experts_per_token=2,
        num_heads=1,
        num_kv_heads=1,
        max_seq_len=16,
        sliding_window=16,
        attention_implementation="reference",
        moe_implementation="ring",
    )
    optimizer = optax.adamw(1e-4, b1=0.9, b2=0.95, weight_decay=0.1)
    policy = jmp.get_policy(f"params={args.dtype},compute={args.dtype},output={args.dtype}")
    with jax.set_mesh(mesh):
        model = policy.cast_to_param(Transformer.init(model_config, key=jax.random.PRNGKey(0)))
        if args.mode == "pp":
            state, static = initialize_mpmd_automatic_pipeline_state(
                model,
                optimizer,
                mpmd_mesh,
                num_stages=stages,
                stage_to_mpmd_index=automatic_stage_to_mpmd_indices(config, args.schedule),
            )
        else:
            for index in range(model_config.num_layers):
                model = eqx.tree_at(lambda value, index=index: value.blocks[index].mlp.router_bias, model, None)
            state = GrugMoeCheckpointState(
                model,
                optimizer.init(model),
                tuple(jnp.zeros((model_config.num_experts,)) for _ in model.blocks),
            )
        tokens = np.arange(128 * 16, dtype=np.int32).reshape(128, 16) % 256
        weights = np.ones((128, 16), dtype=np.float32)
        weights[:, -1] = 0
        sharding = NamedSharding(mesh, P(BATCH_AXES, None))
        batch = GrugLmExample(tokens=jax.device_put(tokens, sharding), loss_weight=jax.device_put(weights, sharding))
        denominator = jnp.sum(batch.loss_weight)
        batches = reshape_batch_into_microbatches(batch, config.microbatches)
    if args.mode == "pp":
        step = make_automatic_pipeline_step(
            optimizer,
            policy,
            static,
            state,
            batches,
            config=config,
            mpmd_mesh=mpmd_mesh,
            schedule_name=args.schedule,
        )
        prepared = prepare_automatic_mpmd_step(step, state, batches, denominator, mpmd_mesh)
        step, state = prepared.step, prepared.state
        batches, denominator = prepared.batches, prepared.loss_denominator
        shardings = step.in_shardings[0][0]
    else:
        step = _fsdp_step(optimizer, policy, mesh)
    contract = {"shape": "reduced-h128-l4-e16-v1"}
    resume_root = prefix_join(args.checkpoint_root, "resume")
    expected_root = prefix_join(args.checkpoint_root, "expected")
    loss_path = StoragePath(prefix_join(args.checkpoint_root, "loss.json"))

    def save(root, value, completed):
        if args.mode == "pp":
            return save_checkpoint(root, value, step=completed, contract=contract)
        return save_levanter_checkpoint(value, completed, prefix_join(root, f"step-{completed}"))

    def restore(root, value):
        if args.mode == "pp":
            return restore_checkpoint(root, value, shardings, contract=contract)
        path = latest_checkpoint_path(root)
        metadata = json.loads((StoragePath(path) / "metadata.json").read_text())
        return load_checkpoint(value, path), metadata["step"]

    if args.phase == "save":
        state, _ = step(state, batches, denominator)
        _assert_optimizer_counts(state, 1)
        save(resume_root, state, 1)
        state, metrics = step(state, batches, denominator)
        _assert_optimizer_counts(state, 2)
        save(expected_root, state, 2)
        loss = _global_loss(metrics[TRAIN_LOSS_KEY])
        if jax.process_index() == 0:
            loss_path.write_text(json.dumps(loss))
        print("CHECKPOINT_SMOKE_SAVED", flush=True)
        return
    state, completed = restore(resume_root, state)
    assert completed == 1, f"expected checkpoint step 1, got {completed}"
    _assert_optimizer_counts(state, completed)
    state, metrics = step(state, batches, denominator)
    expected, expected_step = restore(expected_root, state)
    assert completed + 1 == expected_step == 2
    _assert_optimizer_counts(state, expected_step)
    _assert_optimizer_counts(expected, expected_step)
    maximum_error = _compare(state, expected)
    loss = _global_loss(metrics[TRAIN_LOSS_KEY])
    expected_loss = json.loads(loss_path.read_text())
    loss_error = abs(loss - expected_loss) / max(abs(expected_loss), np.finfo(float).tiny)
    assert loss_error <= _RELATIVE_L2_TOLERANCE, (loss, expected_loss, loss_error)
    print(f"CHECKPOINT_SMOKE_PASSED schedule={args.schedule} max_relative_l2={maximum_error}", flush=True)


if __name__ == "__main__":
    main()
