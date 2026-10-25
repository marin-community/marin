# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check full Ckpt 1 updates and checkpoint continuation on TPU."""

import argparse
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path

import draccus
import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np
from haliax.partitioning import set_mesh
from jax.experimental import multihost_utils
from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.grug.sharding import compact_grug_mesh
from levanter.trainer import _initialize_global_tracker
from levanter.utils.jax_utils import parameter_count

from experiments.grug.fast_track import train
from experiments.grug.fast_track.belay_ckpt1_tpu import DATA_ROOT, build_config, validate_recipe, verified_tokenizer
from experiments.grug.fast_track.belay_ckpt1_tpu_preflight import emit


@jax.jit
def equal_leaves(left, right):
    return jnp.all(jnp.stack([jnp.all(x == y) for x, y in zip(left, right, strict=True)]))


def require_equal(left, right, label):
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if left_tree != right_tree or not bool(equal_leaves(left_leaves, right_leaves)):
        raise AssertionError(f"Checkpoint continuation mismatch: {label}")
    emit("exact_match", label=label, rank=jax.process_index())


def batch_digest(batch):
    """Gather only a small diagnostic batch, never model or optimizer arrays."""
    values = {
        "tokens": batch.tokens,
        "loss_weight": batch.loss_weight,
        "segment_ids": batch.attn_mask.segment_ids[0],
    }
    result = {}
    for name, value in values.items():
        array = np.ascontiguousarray(multihost_utils.process_allgather(value, tiled=True))
        result[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logging-only", action="store_true")
    args = parser.parse_args()
    if not args.logging_only and os.environ.get("JAX_PLATFORMS") != "tpu":
        raise RuntimeError("Optimizer checks require explicit JAX_PLATFORMS=tpu")
    if os.environ.get("GRUG_SKIP_OPTIMIZER"):
        raise RuntimeError("The checkpoint diagnostic requires real optimizer updates")
    run_id = os.environ["BELAY_RUN_ID"]
    output_root = f"{DATA_ROOT}/kaiyuew/belay-ckpt1-20261025/{run_id}"
    tpu = os.environ.get("BELAY_TPU", "v4-64")
    config = build_config(run_id, output_root, verified_tokenizer(), tpu, 0)
    validate_recipe(config)
    trainer = config.trainer.trainer
    if args.logging_only:
        _initialize_global_tracker(trainer.tracker, run_id)
        tracker.log_configuration(config)
        tracker.log_summary({"diagnostic/logging_only_pass": True, "diagnostic/optimizer_updates": 0})
        tracker.current_tracker().finish()
        print(json.dumps({"logging_only_pass": True, "optimizer_updates": 0}), flush=True)
        return

    trainer.initialize()
    tracker.log_configuration(config)
    tracker.log_summary({"diagnostic_only": True, "diagnostic/original_schedule_steps": 8821})
    chips = {"v4-32": 16, "v4-64": 32}[tpu]
    if jax.device_count() != chips or jax.default_backend() != "tpu":
        raise AssertionError("Unexpected TPU topology")
    nonce = bytes(multihost_utils.broadcast_one_to_all(np.frombuffer(uuid.uuid4().bytes, dtype=np.uint8))).hex()
    checkpoint_path = f"{output_root}/attempt-{nonce}/checkpoint-step1"
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    (output / "resolved-config.json").write_text(json.dumps(draccus.encode(config), indent=2) + "\n")
    optimizer = config.optimizer.build(trainer.num_train_steps)
    step_fn, _ = train._build_train_step(config, optimizer)
    data_key, model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)
    if config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(config.trainer.data_seed)
    mesh = compact_grug_mesh(expert_axis_size=16, replica_axis_size=1, model_axis_size=chips // 16)
    emit("full_update_check_start", rank=jax.process_index(), checkpoint_path=checkpoint_path)

    with set_mesh(mesh):

        @jax.jit
        def initialize(key):
            return train.initial_state(
                config.model,
                optimizer=optimizer,
                mp=trainer.mp,
                key=key,
                ema_beta=config.trainer.ema_beta,
                offload_opt_state=config.trainer.offload_opt_state,
                master_param_mode=config.trainer.master_param_mode,
            )

        state = initialize(model_key)
        jax.block_until_ready(state)
        count = parameter_count(state.params)
        if count != 501789462:
            raise AssertionError(f"Unexpected parameter count: {count}")
        emit("full_state_initialized", rank=jax.process_index(), parameter_count=count)
        dataset = train.build_train_dataset(
            config.data, max_seq_len=config.model.max_seq_len, batch_schedule=trainer.batch_schedule, key=data_key
        )
        loader = train.build_train_loader(
            dataset, batch_schedule=trainer.batch_schedule, mesh=mesh, max_buffered_batches=2
        )
        iterator = loader.iter_from_step(0)
        batch0 = next(iterator)
        batch0_hash = batch_digest(batch0)
        emit("batch_frozen", rank=jax.process_index(), step=0, digest=batch0_hash)
        if batch0_hash["tokens"]["sha256"] != "60c5f688c118e647ab71e35481bb67852533e319e0cf1cf157f2fffb9df7cc61":
            raise AssertionError("First native TPU batch differs from frozen data-order probe")
        leaves, tree = jax.tree.flatten(state)
        step = train._flat_train_step(step_fn, tree)
        started = time.perf_counter()
        emit("full_update_compile_start", rank=jax.process_index())
        compiled = step.lower(leaves, batch0).compile()
        emit("full_update_compiled", rank=jax.process_index(), seconds=time.perf_counter() - started)

        def advance(current, batch, index, restored=False):
            started = time.perf_counter()
            next_leaves, metrics, _ = compiled(jax.tree.leaves(current), batch)
            jax.block_until_ready(next_leaves)
            loss = float(metrics["train/loss"])
            if not math.isfinite(loss):
                raise AssertionError(f"Non-finite full-shape loss: {loss}")
            metric_name = "diagnostic/restored_loss" if restored else "train/loss"
            tracker.log({metric_name: loss}, step=index)
            emit(
                "full_update_complete",
                rank=jax.process_index(),
                index=index,
                restored=restored,
                loss=loss,
                seconds=time.perf_counter() - started,
            )
            return jax.tree.unflatten(tree, next_leaves), metrics

        state1, _ = advance(state, batch0, 0)
        save_checkpoint(
            state1,
            step=1,
            checkpoint_path=checkpoint_path,
            is_temporary=False,
            write_config=trainer.checkpointer.write,
            metadata={"diagnostic_only": True, "original_schedule_steps": 8821},
        )
        multihost_utils.sync_global_devices("belay_update_check_checkpoint_committed")
        restored1 = load_checkpoint(state1, checkpoint_path, mesh=mesh, allow_partial=False)
        require_equal(state1, restored1, "all model, optimizer, step and pending-router state after restore")
        batch1 = next(iterator)
        restored_batch1 = next(loader.iter_from_step(int(restored1.step)))
        require_equal(batch1, restored_batch1, "next native data batch after restore")
        emit("batch_frozen", rank=jax.process_index(), step=1, digest=batch_digest(restored_batch1))
        state2, metrics2 = advance(state1, batch1, 1)
        restored2, restored_metrics2 = advance(restored1, restored_batch1, 2, restored=True)
        require_equal(state2, restored2, "complete next-step state after restore")
        require_equal(metrics2, restored_metrics2, "complete next-step metrics after restore")
        tracker.log_summary(
            {
                "diagnostic/full_update_restore_pass": True,
                "diagnostic/optimizer_evaluations": 3,
                "diagnostic/unique_updates": 2,
                "diagnostic/checkpoint_path": checkpoint_path,
            }
        )
        emit("full_update_restore_pass", rank=jax.process_index(), optimizer_evaluations=3, unique_updates=2)
    tracker.current_tracker().finish()


if __name__ == "__main__":
    main()
