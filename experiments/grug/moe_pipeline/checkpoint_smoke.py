# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reduced multi-process checkpoint parity test; run save then resume in fresh gangs."""

import argparse
import json

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from iris.runtime.jax_init import initialize_jax
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample
from levanter.pipeline import reshape_batch_into_microbatches
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.moe_pipeline.checkpoint import checkpoint_arrays, restore_checkpoint, save_checkpoint
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


def _compare(actual, expected) -> float:
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
    assert np.all(relative <= 0.002), f"per-leaf relative L2 errors: {relative.tolist()}"
    return float(relative.max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
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
    mesh, mpmd_mesh = make_pipeline_mesh(config, expert_axis_size=8, replica_axis_size=2)
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
    policy = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    with jax.set_mesh(mesh):
        model = policy.cast_to_param(Transformer.init(model_config, key=jax.random.PRNGKey(0)))
        state, static = initialize_mpmd_automatic_pipeline_state(
            model,
            optimizer,
            mpmd_mesh,
            num_stages=stages,
            stage_to_mpmd_index=automatic_stage_to_mpmd_indices(config, args.schedule),
        )
        tokens = np.arange(64 * 16, dtype=np.int32).reshape(64, 16) % 256
        weights = np.ones((64, 16), dtype=np.float32)
        weights[:, -1] = 0
        sharding = NamedSharding(mesh, P(BATCH_AXES, None))
        batch = GrugLmExample(tokens=jax.device_put(tokens, sharding), loss_weight=jax.device_put(weights, sharding))
        denominator = jnp.sum(batch.loss_weight)
        batches = reshape_batch_into_microbatches(batch, config.microbatches)
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
    shardings = step.in_shardings[0][0]
    contract = {"schedule": args.schedule, "shape": "reduced-h128-l4-e16-pp2-ep8-replica2-v1"}
    resume_root = prefix_join(args.checkpoint_root, "resume")
    expected_root = prefix_join(args.checkpoint_root, "expected")
    loss_path = StoragePath(prefix_join(args.checkpoint_root, f"loss-{jax.process_index()}.json"))
    if args.phase == "save":
        state, _ = step(state, prepared.batches, prepared.loss_denominator)
        save_checkpoint(resume_root, state, step=1, contract=contract)
        state, metrics = step(state, prepared.batches, prepared.loss_denominator)
        save_checkpoint(expected_root, state, step=2, contract=contract)
        loss = metrics[TRAIN_LOSS_KEY].to_mpmd_local_array
        if loss is not None:
            loss_path.write_text(json.dumps(float(loss)))
        print("CHECKPOINT_SMOKE_SAVED", flush=True)
        return
    state, completed = restore_checkpoint(resume_root, state, shardings, contract=contract)
    assert completed == 1, f"expected checkpoint step 1, got {completed}"
    state, metrics = step(state, prepared.batches, prepared.loss_denominator)
    expected, expected_step = restore_checkpoint(expected_root, state, shardings, contract=contract)
    assert completed + 1 == expected_step == 2
    maximum_error = _compare(state, expected)
    loss = metrics[TRAIN_LOSS_KEY].to_mpmd_local_array
    if loss is not None:
        expected_loss = json.loads(loss_path.read_text())
        loss_error = abs(float(loss) - expected_loss) / max(abs(expected_loss), np.finfo(float).tiny)
        assert loss_error <= 0.002, (float(loss), expected_loss, loss_error)
    print(f"CHECKPOINT_SMOKE_PASSED schedule={args.schedule} max_relative_l2={maximum_error}", flush=True)


if __name__ == "__main__":
    main()
