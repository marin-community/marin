# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check full Ckpt 1 updates and checkpoint continuation on TPU."""

import argparse
import dataclasses
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

from experiments.grug.belay_tpu import qb_balancing, train
from experiments.grug.belay_tpu.belay_ckpt1_tpu import (
    DATA_ROOT,
    build_config,
    require_tpu_rounding,
    validate_recipe,
    verified_tokenizer,
)
from experiments.grug.belay_tpu.belay_ckpt1_tpu_preflight import emit
from experiments.grug.belay_tpu.grugmuon_stacked import _newtonschulz_batched_syrk
from experiments.grug.belay_tpu.packed_kda_check import check_packed_kda
from experiments.grug.belay_tpu.remat_check import check_attention_remat
from experiments.grug.fast_track.grugmuon_stacked import _zeropower_via_newtonschulz_local
from experiments.grug.fast_track.qb_balancing import bincount_upper_quantile as source_bincount_upper_quantile


def check_muon_gram_port():
    """Compare the batched TPU port with the source per-matrix Newton-Schulz."""
    rng = np.random.RandomState(17)
    for rows, columns in ((256, 384), (384, 256), (128, 128)):
        values = rng.randn(3, rows, columns).astype(np.float32)
        values[0] = 0
        matrices = jnp.asarray(values, jnp.bfloat16)
        reference = jax.jit(jax.vmap(lambda x: _zeropower_via_newtonschulz_local(x, 5, 1e-8, "quintic")))(matrices)
        actual = jax.jit(lambda x: _newtonschulz_batched_syrk(x, 5, 1e-8, "quintic"))(matrices)
        expected, observed = (np.asarray(x, np.float32) for x in (reference, actual))
        # Same BF16 bound as tests/grug/test_fast_track_muon_bucketing.py.
        tolerance = 2e-3 * float(np.abs(expected).max())
        np.testing.assert_allclose(observed, expected, rtol=0, atol=tolerance)
        emit(
            "muon_gram_port_pass",
            rank=jax.process_index(),
            shape=list(values.shape),
            max_absolute_error=float(np.abs(observed - expected).max()),
            absolute_tolerance=tolerance,
        )


@jax.jit
def equal_leaves(left, right):
    return jnp.all(jnp.stack([jnp.all(x == y) for x, y in zip(left, right, strict=True)]))


@jax.jit
def leaf_differences(left, right):
    counts, maxima, means = [], [], []
    for x, y in zip(left, right, strict=True):
        x, y = jnp.asarray(x), jnp.asarray(y)
        delta = jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32))
        counts.append(jnp.count_nonzero(x != y))
        maxima.append(jnp.max(delta, initial=0))
        means.append(jnp.sum(delta) / max(delta.size, 1))
    return jnp.stack(counts), jnp.stack(maxima), jnp.stack(means)


def require_equal(left, right, label):
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if left_tree != right_tree:
        raise AssertionError(f"Checkpoint tree mismatch: {label}")
    if not bool(equal_leaves(left_leaves, right_leaves)):
        counts, maxima, means = (np.asarray(x) for x in leaf_differences(left_leaves, right_leaves))
        paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_flatten_with_path(left)[0]]
        differences = [
            {
                "path": paths[i],
                "shape": list(left_leaves[i].shape),
                "dtype": str(left_leaves[i].dtype),
                "unequal_elements": int(counts[i]),
                "max_absolute_error": float(maxima[i]),
                "mean_absolute_error": float(means[i]),
            }
            for i in np.flatnonzero(counts)
        ]
        differences.sort(key=lambda row: row["max_absolute_error"], reverse=True)
        (Path(os.environ["IRIS_OUTPUT_DIR"]) / f"state-mismatch-rank{jax.process_index()}.json").write_text(
            json.dumps({"label": label, "leaves": differences}, indent=2) + "\n"
        )
        emit(
            "state_mismatch",
            label=label,
            rank=jax.process_index(),
            mismatched_leaves=len(differences),
            largest_leaves=differences[:12],
        )
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
    parser.add_argument(
        "--compile-only", action="store_true", help="Compile the original saved-residual step; no updates"
    )
    parser.add_argument(
        "--reference-step1", help="Require the first update to exactly match this frozen step-1 checkpoint"
    )
    parser.add_argument(
        "--source-qb", action="store_true", help="Use the unchanged source histogram as a cross-run control"
    )
    args = parser.parse_args()
    if args.source_qb:
        qb_balancing.bincount_upper_quantile = source_bincount_upper_quantile
    if not args.logging_only and os.environ.get("JAX_PLATFORMS") != "tpu,cpu":
        raise RuntimeError("Optimizer checks require JAX_PLATFORMS=tpu,cpu; CPU is needed for data shuffling")
    if os.environ.get("GRUG_SKIP_OPTIMIZER"):
        raise RuntimeError("The checkpoint diagnostic requires real optimizer updates")
    run_id = os.environ["BELAY_RUN_ID"]
    output_root = f"{DATA_ROOT}/kaiyuew/belay-ckpt1-20261025/{run_id}"
    tpu = os.environ.get("BELAY_TPU", "v4-64")
    config = build_config(run_id, output_root, verified_tokenizer(), tpu, 0)
    if args.compile_only:
        config = dataclasses.replace(config, model=dataclasses.replace(config.model, attn_res_remat_attention=False))
    validate_recipe(config)
    trainer = config.trainer.trainer
    if args.logging_only:
        _initialize_global_tracker(trainer.tracker, run_id)
        tracker.log_configuration(config)
        tracker.log_summary({"diagnostic/logging_only_pass": True, "diagnostic/optimizer_updates": 0})
        tracker.current_tracker().finish()
        print(json.dumps({"logging_only_pass": True, "optimizer_updates": 0}), flush=True)
        return

    require_tpu_rounding()
    trainer.initialize()
    tracker.log_configuration(config)
    tracker.log_summary(
        {
            "diagnostic_only": True,
            "diagnostic/original_schedule_steps": 8821,
            "diagnostic/source_qb_control": args.source_qb,
            "diagnostic/reference_step1": args.reference_step1,
        }
    )
    chips = {"v4-32": 16, "v4-64": 32}[tpu]
    if jax.device_count() != chips or jax.default_backend() != "tpu":
        raise AssertionError("Unexpected TPU topology")
    check_muon_gram_port()
    check_packed_kda()
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
        if config.model.attn_res_remat_attention:
            emit("attention_remat_oracle_pass", **check_attention_remat(config, mesh))

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
        lowered = step.lower(leaves, batch0)
        if jax.process_index() == 0:
            (output / "train-step-before-optimization.mlir").write_text(str(lowered.compiler_ir()))
        compiled = lowered.compile()
        emit("full_update_memory_analysis", rank=jax.process_index(), analysis=str(compiled.memory_analysis()))
        emit("full_update_compiled", rank=jax.process_index(), seconds=time.perf_counter() - started)
        if args.compile_only:
            tracker.log_summary({"diagnostic/compile_only_pass": True, "diagnostic/optimizer_updates": 0})
            tracker.current_tracker().finish()
            return

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
        if args.reference_step1:
            reference1 = load_checkpoint(state1, args.reference_step1, mesh=mesh, allow_partial=False)
            require_equal(state1, reference1, "first update against frozen reference checkpoint")
            del reference1
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
