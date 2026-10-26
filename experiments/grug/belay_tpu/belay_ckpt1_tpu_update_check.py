# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check full Ckpt 1 updates and checkpoint continuation on TPU."""

import argparse
import dataclasses
import functools
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
from levanter.grug._moe import ep_ragged_all_to_all
from levanter.grug.sharding import compact_grug_mesh
from levanter.trainer import _initialize_global_tracker
from levanter.utils.jax_utils import parameter_count

from experiments.grug.belay_tpu import local_experts, model, qb_balancing, routing_tpu, train
from experiments.grug.belay_tpu.belay_ckpt1_tpu import (
    DATA_ROOT,
    build_config,
    require_tpu_rounding,
    validate_recipe,
    verified_tokenizer,
)
from experiments.grug.belay_tpu.belay_ckpt1_tpu_preflight import emit
from experiments.grug.belay_tpu.expert_collectives import all_to_all_reduce_scatter, butterfly_reduce_scatter
from experiments.grug.belay_tpu.expert_gradient_adapter import partitioned_grouped_dot, whole_width_tiling
from experiments.grug.belay_tpu.grugmuon_stacked import _newtonschulz_batched_syrk
from experiments.grug.belay_tpu.local_movement import combine_expert_order
from experiments.grug.belay_tpu.moe_saved_output import saved_output_routed_experts
from experiments.grug.belay_tpu.packed_kda_check import check_packed_kda
from experiments.grug.belay_tpu.remat_check import check_attention_remat
from experiments.grug.belay_tpu.saved_relu2_adapter import saved_relu2_mlp, select_saved_relu2
from experiments.grug.belay_tpu.streaming_combine import combine_streaming, dispatch_streaming
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


def compare_state(left, right, label):
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
        return False
    emit("exact_match", label=label, rank=jax.process_index())
    return True


def require_equal(left, right, label):
    if not compare_state(left, right, label):
        raise AssertionError(f"Checkpoint continuation mismatch: {label}")


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
    parser.add_argument("--partitioned-expert-gradients", action="store_true")
    parser.add_argument("--whole-width-expert-tiles", action="store_true")
    parser.add_argument("--expert-row-partitions", type=int, choices=(0, 2), default=0)
    parser.add_argument("--kda-state-precision", choices=("highest", "high"), default="highest")
    parser.add_argument(
        "--streaming-movement", action="store_true", help="Accumulate expert routes without full restores"
    )
    parser.add_argument("--local-movement", action="store_true", help="Form combine gradients in expert order")
    parser.add_argument("--local-experts", action="store_true", help="Gather expert weights and keep tokens local")
    parser.add_argument("--saved-relu2", action="store_true", help="Fuse expert ReLU2 and save preactivations")
    parser.add_argument("--expert-reduction", choices=("native", "butterfly", "all-to-all"), default="native")
    parser.add_argument("--initial-checkpoint", help="Profile continuation from a complete checkpoint leaf")
    parser.add_argument(
        "--update-steps", type=int, default=2, help="Consecutive diagnostic updates under the original schedule"
    )
    parser.add_argument(
        "--profile-steps", type=int, default=0, help="Additional updates to trace after the timing window"
    )
    parser.add_argument(
        "--compile-only", action="store_true", help="Compile the original saved-residual step; no updates"
    )
    parser.add_argument(
        "--reference-step1", help="Record first-update differences from this frozen checkpoint without rejecting them"
    )
    parser.add_argument(
        "--source-qb", action="store_true", help="Use the unchanged source histogram as a cross-run control"
    )
    parser.add_argument("--routing-ops", action="store_true", help="Use the selected Base16 routing lookup and sorts")
    parser.add_argument(
        "--saved-output", action="store_true", help="Compute routing gradients from saved expert returns"
    )
    args = parser.parse_args()
    state_precision = jax.lax.Precision[args.kda_state_precision.upper()]
    model.chunk_kda_tpu = functools.partial(model.chunk_kda_tpu, state_precision=state_precision)
    if args.initial_checkpoint and args.reference_step1:
        parser.error("A continuation cannot compare its first update to the initialization reference")
    if args.expert_reduction != "native" and not args.local_experts:
        parser.error("Expert-gradient reduction diagnostics require --local-experts")
    if args.whole_width_expert_tiles and not args.partitioned_expert_gradients:
        parser.error("Whole-width expert tiles require --partitioned-expert-gradients")
    if args.expert_row_partitions and (not args.partitioned_expert_gradients or args.whole_width_expert_tiles):
        parser.error("Row partitions require --partitioned-expert-gradients without --whole-width-expert-tiles")
    if args.partitioned_expert_gradients:
        if not args.local_experts or args.saved_relu2:
            parser.error("Partitioned expert gradients require --local-experts without --saved-relu2")
        local_experts.grouped_dot = (
            functools.partial(partitioned_grouped_dot, tiling=whole_width_tiling)
            if args.whole_width_expert_tiles
            else partitioned_grouped_dot
        )
    if args.expert_row_partitions:
        local_experts.grouped_dot = functools.partial(partitioned_grouped_dot, row_partitions=args.expert_row_partitions)
    if args.local_movement:
        if not args.local_experts:
            parser.error("Expert-order combine requires --local-experts")
        local_experts.combine_local = combine_expert_order
    if args.streaming_movement:
        if not args.local_experts or args.local_movement:
            parser.error("Streaming movement requires --local-experts without --local-movement")
        local_experts.combine_local = functools.partial(combine_streaming, implementation="scan")
        local_experts.dispatch_local = dispatch_streaming
    if args.expert_reduction == "butterfly":
        local_experts.reduce_expert_gradients = butterfly_reduce_scatter
    elif args.expert_reduction == "all-to-all":
        local_experts.reduce_expert_gradients = all_to_all_reduce_scatter
    if args.update_steps < 2:
        parser.error("Checkpoint continuation requires at least two consecutive updates")
    if args.profile_steps < 0:
        parser.error("Profile steps cannot be negative")
    total_updates = args.update_steps + args.profile_steps
    if args.source_qb:
        qb_balancing.bincount_upper_quantile = source_bincount_upper_quantile
    if args.routing_ops:
        model._selected_router_logits = routing_tpu.unrolled_router_logits
        ep_ragged_all_to_all._accepted_assignments = routing_tpu.accepted_assignments_cutoff
        ep_ragged_all_to_all._restore_assignment_scalars = routing_tpu.restore_assignment_scalars_sort
        ep_ragged_all_to_all._permute_assignment_weights = routing_tpu.permute_assignment_weights_sort
    if args.saved_output:
        ep_ragged_all_to_all._routed_experts = saved_output_routed_experts
    if args.saved_relu2:
        ep_ragged_all_to_all._select_expert_mlp = select_saved_relu2
        local_experts.expert_mlp = saved_relu2_mlp
    if not args.logging_only and os.environ.get("JAX_PLATFORMS") != "tpu,cpu":
        raise RuntimeError("Optimizer checks require JAX_PLATFORMS=tpu,cpu; CPU is needed for data shuffling")
    if os.environ.get("GRUG_SKIP_OPTIMIZER"):
        raise RuntimeError("The checkpoint diagnostic requires real optimizer updates")
    run_id = os.environ["BELAY_RUN_ID"]
    output_root = f"{DATA_ROOT}/kaiyuew/belay-ckpt1-20261025/{run_id}"
    tpu = os.environ.get("BELAY_TPU", "v4-64")
    config = build_config(run_id, output_root, verified_tokenizer(), tpu, 0, local_experts=args.local_experts)
    if args.saved_output and config.model.moe_shared_placement == model.SharedExpertPlacement.OVERLAP:
        raise ValueError("Saved-output diagnostic requires the non-overlapped Ckpt 1 expert path")
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
            "diagnostic/routing_ops": args.routing_ops,
            "diagnostic/expert_reduction": args.expert_reduction,
            "diagnostic/saved_output": args.saved_output,
            "diagnostic/local_experts": args.local_experts,
            "diagnostic/local_movement": args.local_movement,
            "diagnostic/streaming_movement": args.streaming_movement,
            "diagnostic/partitioned_expert_gradients": args.partitioned_expert_gradients,
            "diagnostic/whole_width_expert_tiles": args.whole_width_expert_tiles,
            "diagnostic/expert_row_partitions": args.expert_row_partitions,
            "diagnostic/kda_state_precision": args.kda_state_precision,
            "diagnostic/saved_relu2": args.saved_relu2,
            "diagnostic/requested_unique_updates": total_updates,
            "diagnostic/profile_updates": args.profile_steps,
            "diagnostic/reference_step1": args.reference_step1,
        }
    )
    chips = {"v4-32": 16, "v4-64": 32}[tpu]
    if jax.device_count() != chips or jax.default_backend() != "tpu":
        raise AssertionError("Unexpected TPU topology")
    check_muon_gram_port()
    check_packed_kda(state_precision=state_precision)
    nonce = bytes(multihost_utils.broadcast_one_to_all(np.frombuffer(uuid.uuid4().bytes, dtype=np.uint8))).hex()
    checkpoint_path = f"{output_root}/attempt-{nonce}/checkpoint-after-first-update"
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    (output / "resolved-config.json").write_text(json.dumps(draccus.encode(config), indent=2) + "\n")
    optimizer = config.optimizer.build(trainer.num_train_steps)
    step_fn, _ = train._build_train_step(config, optimizer)
    data_key, model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)
    if config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(config.trainer.data_seed)
    mesh = compact_grug_mesh(
        expert_axis_size=config.trainer.expert_axis_size,
        replica_axis_size=1,
        model_axis_size=config.trainer.model_axis_size,
    )
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
        if args.initial_checkpoint:
            state = train.restore_template_from(state)
            state = load_checkpoint(state, args.initial_checkpoint, mesh=mesh, allow_partial=False)
        initial_step = int(state.step)
        if initial_step + total_updates > trainer.num_train_steps:
            raise ValueError("Diagnostic updates exceed the original optimizer schedule")
        tracker.log_summary(
            {"diagnostic/initial_step": initial_step, "diagnostic/initial_checkpoint": args.initial_checkpoint}
        )
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
        iterator = loader.iter_from_step(initial_step)
        batch0 = next(iterator)
        batch0_hash = batch_digest(batch0)
        emit("batch_frozen", rank=jax.process_index(), step=initial_step, digest=batch0_hash)
        if (
            initial_step == 0
            and batch0_hash["tokens"]["sha256"] != "60c5f688c118e647ab71e35481bb67852533e319e0cf1cf157f2fffb9df7cc61"
        ):
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

        measurements = []

        def advance(current, batch, index, restored=False):
            started = time.perf_counter()
            next_leaves, metrics, _ = compiled(jax.tree.leaves(current), batch)
            jax.block_until_ready(next_leaves)
            loss = float(metrics["train/loss"])
            if not math.isfinite(loss):
                raise AssertionError(f"Non-finite full-shape loss: {loss}")
            metric_name = "diagnostic/restored_loss" if restored else "train/loss"
            tracker.log({metric_name: loss}, step=initial_step + index)
            elapsed = time.perf_counter() - started
            if not restored:
                measurements.append({"index": index, "seconds": elapsed, "loss": loss})
            emit(
                "full_update_complete",
                rank=jax.process_index(),
                index=index,
                global_step=initial_step + index,
                restored=restored,
                loss=loss,
                seconds=elapsed,
            )
            return jax.tree.unflatten(tree, next_leaves), metrics

        state1, _ = advance(state, batch0, 0)
        save_checkpoint(
            state1,
            step=initial_step + 1,
            checkpoint_path=checkpoint_path,
            is_temporary=False,
            write_config=trainer.checkpointer.write,
            metadata={"diagnostic_only": True, "original_schedule_steps": 8821},
        )
        multihost_utils.sync_global_devices("belay_update_check_checkpoint_committed")
        if args.reference_step1:
            reference1 = load_checkpoint(state1, args.reference_step1, mesh=mesh, allow_partial=False)
            reference_equal = compare_state(state1, reference1, "first update against frozen reference checkpoint")
            tracker.log_summary({"diagnostic/reference_state_exact": reference_equal})
            del reference1
        restored1 = load_checkpoint(state1, checkpoint_path, mesh=mesh, allow_partial=False)
        require_equal(state1, restored1, "all model, optimizer, step and pending-router state after restore")
        batch1 = next(iterator)
        restored_batch1 = next(loader.iter_from_step(int(restored1.step)))
        require_equal(batch1, restored_batch1, "next native data batch after restore")
        emit("batch_frozen", rank=jax.process_index(), step=initial_step + 1, digest=batch_digest(restored_batch1))
        state2, metrics2 = advance(state1, batch1, 1)
        restored2, restored_metrics2 = advance(restored1, restored_batch1, 1, restored=True)
        require_equal(state2, restored2, "complete next-step state after restore")
        require_equal(metrics2, restored_metrics2, "complete next-step metrics after restore")
        del restored1, restored2
        for index in range(2, args.update_steps):
            state2, _ = advance(state2, next(iterator), index)
        if args.update_steps > 5:
            timings = [row["seconds"] for row in measurements if row["index"] >= 5]
            emit(
                "full_update_timing_window",
                rank=jax.process_index(),
                first_step=initial_step + 5,
                last_step=initial_step + args.update_steps - 1,
                median_seconds=float(np.median(timings)),
                mean_seconds=float(np.mean(timings)),
                scope=(
                    "Synchronized update plus loss logging; excludes batch loading and checkpoint I/O; "
                    "not end-to-end MFU"
                ),
            )
        if args.profile_steps:
            profile_path = str(output / "profile")
            with jax.profiler.trace(profile_path):
                for index in range(args.update_steps, total_updates):
                    with jax.profiler.StepTraceAnnotation("train", step_num=initial_step + index):
                        state2, _ = advance(state2, next(iterator), index)
            emit(
                "full_update_profile_saved",
                rank=jax.process_index(),
                path=profile_path,
                first_step=initial_step + args.update_steps,
                last_step=initial_step + total_updates - 1,
            )
        tracker.log_summary(
            {
                "diagnostic/full_update_restore_pass": True,
                "diagnostic/optimizer_evaluations": total_updates + 1,
                "diagnostic/unique_updates": total_updates,
                "diagnostic/checkpoint_path": checkpoint_path,
            }
        )
        emit(
            "full_update_restore_pass",
            rank=jax.process_index(),
            optimizer_evaluations=total_updates + 1,
            unique_updates=total_updates,
        )
    tracker.current_tracker().finish()


if __name__ == "__main__":
    main()
