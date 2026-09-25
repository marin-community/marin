# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded synthetic hero pipeline trial; no checkpoints or production state."""

import argparse
import dataclasses
import importlib
import itertools
import json
import time

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import wandb
from finestore.cache import PersistentKvCache
from fray.device_flops import device_flops
from iris.runtime.jax_init import initialize_jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxpp import dime2, env_vars
from levanter.cutlass_kernel_cache import install as install_cutlass_cache
from levanter.data.text.examples import GrugLmExample
from levanter.pipeline import reshape_batch_into_microbatches
from levanter.utils.jax_utils import barrier_sync_named, multihost_allgather_sync

from experiments.grug.moe_hero_ep.hero_recipe import HERO_MODEL_CONFIG
from experiments.grug.moe_hero_ep.model import GrugModelConfig, QbEstimator
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_hero_ep.train import _compute_flops
from experiments.grug.moe_hero_pipeline.pipeline import (
    BATCH_AXES,
    TRAIN_LOSS_KEY,
    AutomaticPipelineSchedule,
    GrugMoePipelineConfig,
    automatic_stage_to_mpmd_indices,
    initialize_stage_local_pipeline_state,
    make_automatic_pipeline_step,
    make_pipeline_mesh,
    park_pipeline_state,
    precompile_automatic_mpmd_step,
    prepare_automatic_mpmd_step,
    restore_pipeline_state,
)

_MULTIHOST_TIMEOUT = 600


def _log(event: str, **fields) -> None:
    if jax.process_index() == 0:
        print(json.dumps({"event": event, **fields}, default=str), flush=True)
        if wandb.run is not None:
            wandb.run.summary["phase"] = event
            wandb.run.summary[event] = json.loads(json.dumps(fields, default=str))


def _global_loss(value) -> float:
    local = value if isinstance(value, jax.Array) else value.to_mpmd_local_array
    report = np.array([local is not None, 0.0 if local is None else float(local)], dtype=np.float64)
    # Finished stages must not launch GPU collectives while other stages still execute.
    reports = np.asarray(multihost_allgather_sync(report.tolist(), timeout=_MULTIHOST_TIMEOUT)).reshape(-1, 2)
    losses = reports[reports[:, 0] != 0, 1]
    if not len(losses) or not np.all(np.isfinite(losses)):
        raise FloatingPointError(f"Pipeline loss absent or nonfinite: {reports.tolist()}")
    return float(losses[0])


def _checked_cuda_result(result: tuple, operation: str) -> tuple:
    status, *values = result
    if int(status) != 0:
        raise RuntimeError(f"{operation} failed: {status}")
    return tuple(values)


def _synchronize_local_cuda_devices(completed_steps: int) -> None:
    """Wait on every local CUDA device without executing numerical diagnostics."""
    runtime = importlib.import_module("cuda.bindings.runtime")
    driver = importlib.import_module("cuda.bindings.driver")
    devices = jax.local_devices()
    if any(device.platform != "gpu" for device in devices):
        raise ValueError("synchronize-devices-after-step requires CUDA devices")
    started = time.monotonic()
    # Preserve the calling thread's exact context, including a null/non-primary
    # context, as well as its CUDA runtime device selection.
    (original_context,) = _checked_cuda_result(driver.cuCtxGetCurrent(), "cuCtxGetCurrent")
    (original_device,) = _checked_cuda_result(runtime.cudaGetDevice(), "cudaGetDevice")
    try:
        for device in devices:
            ordinal = device.local_hardware_id
            _checked_cuda_result(runtime.cudaSetDevice(ordinal), f"cudaSetDevice({ordinal})")
            _checked_cuda_result(runtime.cudaDeviceSynchronize(), f"cudaDeviceSynchronize({ordinal})")
    finally:
        try:
            _checked_cuda_result(runtime.cudaSetDevice(original_device), "cudaSetDevice(restore)")
        finally:
            _checked_cuda_result(driver.cuCtxSetCurrent(original_context), "cuCtxSetCurrent(restore)")
    print(
        "HERO_DEVICE_SYNC "
        + json.dumps(
            {
                "event": "pipeline_devices_synchronized",
                "rank": jax.process_index(),
                "step": completed_steps,
                "devices": len(devices),
                "elapsed_seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )


def _initialize_pipeline_communicators(mpmd_mesh, placements: tuple[int, ...]) -> None:
    # Create every neighboring channel before a blocking NCCL group can prevent
    # a stage from reaching another stage's communicator initialization.
    edges = sorted({tuple(sorted((a, b))) for a, b in itertools.pairwise(placements) if a != b})
    for left, right in edges:
        for source, target in zip(
            mpmd_mesh.unstack[left].devices.flat, mpmd_mesh.unstack[right].devices.flat, strict=True
        ):
            if jax.process_index() not in (source.process_index, target.process_index):
                continue
            if env_vars.jaxpp_directional_communicators.value:
                dime2.get_or_create_comm(dime2.UniqueDevices(source, target))
                dime2.get_or_create_comm(dime2.UniqueDevices(target, source))
            else:
                dime2.get_or_create_comm(dime2.UniqueSortedDevices(source, target))
    barrier_sync_named("hero_pipeline_communicators_initialized", timeout=_MULTIHOST_TIMEOUT)

    # NCCL establishes P2P transports on the first send/recv, even after
    # communicator initialization. Connect both directions in a common edge
    # order before pipeline scheduling can introduce a blocking setup cycle.
    for left, right in edges:
        left_mesh, right_mesh = mpmd_mesh.unstack[left], mpmd_mesh.unstack[right]
        if any(device.process_index == jax.process_index() for device in left_mesh.devices.flat):
            local_stage, remote_stage = left, right
        elif any(device.process_index == jax.process_index() for device in right_mesh.devices.flat):
            local_stage, remote_stage = right, left
        else:
            continue
        local_sharding = NamedSharding(mpmd_mesh.unstack[local_stage], P())
        remote_sharding = NamedSharding(mpmd_mesh.unstack[remote_stage], P())
        payload = jax.make_array_from_callback(
            (1,), local_sharding, lambda _, stage=local_stage: np.full((1,), stage, dtype=np.float32)
        )
        recv_buffer = jax.make_array_from_callback((1,), local_sharding, lambda _: np.zeros((1,), dtype=np.float32))
        transfer = dime2.start_transfer([payload], [remote_sharding], [recv_buffer], [remote_sharding])
        (received,) = transfer.done()
        received.block_until_ready()
        for shard in received.addressable_shards:
            actual = np.asarray(shard.data)
            if not np.all(actual == remote_stage):
                raise RuntimeError(f"Pipeline channel {remote_stage}->{local_stage} received {actual}")
    barrier_sync_named("hero_pipeline_channels_connected", timeout=_MULTIHOST_TIMEOUT)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-hero", action="store_true")
    parser.add_argument("--diagnostic-layers", type=int, help="Use fewer full-width hero layers for fault isolation")
    parser.add_argument(
        "--schedule",
        type=AutomaticPipelineSchedule,
        choices=list(AutomaticPipelineSchedule),
        default=AutomaticPipelineSchedule.STANDARD_1F1B,
    )
    parser.add_argument("--physical-stages", type=int)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--expert-axis-size", type=int, default=1)
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--expert-waves", type=int, default=3)
    parser.add_argument("--offload-opt-state", action="store_true")
    parser.add_argument("--offload-activations", action="store_true")
    parser.add_argument(
        "--park-state-during-warmup",
        action="store_true",
        help="Park real device state on pinned host during disposable warmup, then restore it",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--synchronize-devices-after-step",
        action="store_true",
        help="Diagnostic CUDA synchronization on every local device after each optimizer step",
    )
    parser.add_argument("--optimizer", choices=("adamw", "muonh"), default="adamw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", help="Optional rank-zero W&B console capture in marin-community/marin_moe")
    parser.add_argument("--compilation-cache", default="/tmp/hero-pipeline-jax-cache")
    args = parser.parse_args()
    if args.diagnostic_layers is not None and (not args.full_hero or args.diagnostic_layers < 1):
        parser.error("diagnostic-layers requires full-hero and a positive layer count")
    if args.sequence_length is not None and args.sequence_length < 1:
        parser.error("sequence-length must be positive")
    if args.expert_waves < 1:
        parser.error("expert-waves must be positive")
    if args.steps < 1 or args.expert_axis_size < 1:
        parser.error("steps and expert-axis-size must be positive")

    if (args.schedule == AutomaticPipelineSchedule.DUALPIPE_V) != (args.physical_stages is not None):
        raise ValueError("DualPipeV requires physical-stages; other schedules use one logical stage per physical stage")
    return args


def _model_config(args: argparse.Namespace) -> GrugModelConfig:
    if args.full_hero:
        model_config = dataclasses.replace(
            HERO_MODEL_CONFIG,
            attention_implementation="gpu_fa4_cute",
            moe_implementation="fixed_pooled_wave_all_to_all",
            remat_mode="recompute_all",
            num_expert_waves=args.expert_waves,
        )
    else:
        model_config = GrugModelConfig(
            vocab_size=1024,
            hidden_dim=256,
            intermediate_dim=128,
            shared_expert_intermediate_dim=128,
            num_shared_experts=2,
            num_experts=12 * args.expert_axis_size,
            num_experts_per_token=2,
            latent_dim=128,
            num_layers=max(4, args.stages),
            num_heads=4,
            num_kv_heads=2,
            local_kv_heads=2,
            global_kv_heads=1,
            head_dim=64,
            max_seq_len=256,
            sliding_window=128,
            sconv=True,
            rope_fused=True,
            qb_estimator=QbEstimator.HIST,
            attention_implementation="gpu_fa4_cute",
            moe_implementation="fixed_pooled_wave_all_to_all",
            num_expert_waves=args.expert_waves,
            capacity_factor=1.15,
            pooled_transport_capacity_factor=1.15,
            remat_mode="recompute_all",
        )
    if args.diagnostic_layers is not None:
        model_config = dataclasses.replace(model_config, num_layers=args.diagnostic_layers)
    if args.sequence_length is not None:
        model_config = dataclasses.replace(model_config, max_seq_len=args.sequence_length)
    if args.offload_activations:
        model_config = dataclasses.replace(model_config, remat_mode="offload_carry")
    return model_config


def main() -> None:
    args = _parse_args()
    # Configure both caches before Iris initialization can select object storage.
    jax.config.update("jax_compilation_cache_dir", args.compilation_cache)
    install_cutlass_cache(PersistentKvCache.in_memory())
    initialize_jax()
    if args.run_id and jax.process_index() == 0:
        wandb.init(
            entity="marin-community",
            project="marin_moe",
            id=args.run_id,
            name=args.run_id,
            group="hero-h100-pipeline",
            resume="never",
            config=vars(args),
            settings=wandb.Settings(save_code=False, console="redirect"),
        )
    config = GrugMoePipelineConfig(
        stages=args.stages, microbatches=args.microbatches, physical_stages=args.physical_stages
    )
    placements = automatic_stage_to_mpmd_indices(config, args.schedule)
    mesh, mpmd_mesh = make_pipeline_mesh(config, expert_axis_size=args.expert_axis_size, replica_axis_size=1)
    model_config = _model_config(args)
    full_hero = args.full_hero and args.diagnostic_layers is None
    batch_multiple = args.microbatches * jax.device_count() // config.mpmd_stages
    batch_size = args.batch_size if args.batch_size is not None else batch_multiple
    if batch_size < 1 or batch_size % batch_multiple:
        raise ValueError(f"batch-size must be a positive multiple of {batch_multiple}")
    policy = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    flops_per_example, _ = _compute_flops(model_config=model_config)
    peak_flops = device_flops("h100")
    assert peak_flops is not None
    if args.optimizer == "muonh":
        optimizer = GrugMoeMuonHConfig(
            learning_rate=13 / 3 * 1e-4, adam_lr=1e-4, warmup=0, lr_schedule="constant"
        ).build(args.steps)
    else:
        optimizer = optax.adamw(1e-4, b1=0.9, b2=0.95, mu_dtype=jnp.bfloat16, weight_decay=0.1)
    _log(
        "pipeline_init",
        model=dataclasses.asdict(model_config),
        batch_size=batch_size,
        stages=args.stages,
        physical_stages=config.mpmd_stages,
        schedule=args.schedule,
        placements=placements,
        experts=args.expert_axis_size,
        microbatches=args.microbatches,
        devices=jax.device_count(),
        optimizer=f"{args.optimizer}_bf16",
        offload_opt_state=args.offload_opt_state,
        offload_activations=args.offload_activations,
        full_hero=full_hero,
        diagnostic_layers=args.diagnostic_layers,
    )
    started = time.monotonic()
    state, static_stages = initialize_stage_local_pipeline_state(
        model_config,
        optimizer,
        policy,
        mpmd_mesh,
        num_stages=args.stages,
        seed=args.seed,
        stage_to_mpmd_index=placements,
        offload_opt_state=args.offload_opt_state,
    )
    _log("pipeline_initialized", elapsed_seconds=time.monotonic() - started)
    if args.offload_opt_state:
        assert all(value.sharding.memory_kind == "pinned_host" for value in jax.tree.leaves(state.opt_state))
        _log("optimizer_state_offloaded", memory_kind="pinned_host")
    tokens = np.random.default_rng(args.seed).integers(
        model_config.vocab_size, size=(batch_size, model_config.max_seq_len), dtype=np.int32
    )
    weights = np.ones_like(tokens, dtype=np.float32)
    weights[:, -1] = 0
    with jax.set_mesh(mesh):
        sharding = NamedSharding(mesh, P(BATCH_AXES, None))
        batch = GrugLmExample(tokens=jax.device_put(tokens, sharding), loss_weight=jax.device_put(weights, sharding))
        denominator = jnp.sum(batch.loss_weight)
        batches = reshape_batch_into_microbatches(batch, args.microbatches)
    step = make_automatic_pipeline_step(
        optimizer,
        policy,
        static_stages,
        state,
        batches,
        config=config,
        mpmd_mesh=mpmd_mesh,
        schedule_name=args.schedule,
        offload_opt_state=args.offload_opt_state,
    )
    started = time.monotonic()
    prepared = prepare_automatic_mpmd_step(step, state, batches, denominator, mpmd_mesh)
    _log("pipeline_compiled", elapsed_seconds=time.monotonic() - started)
    state = prepared.state
    compiled_step = prepared.step
    batches = prepared.batches
    denominator = prepared.loss_denominator
    del prepared
    # The compiled function takes state as an argument; no real arrays are
    # donated to disposable warmup. Delete old aliases before parking buffers.
    del step
    parked = park_pipeline_state(state) if args.park_state_during_warmup else None
    if parked is not None:
        del state
        _log("pipeline_state_parked", local_device_bytes=parked.local_device_bytes)
    started = time.monotonic()
    task_count = precompile_automatic_mpmd_step(compiled_step)
    if parked is not None:
        state = restore_pipeline_state(parked)
        del parked
        _log("pipeline_state_restored")
    barrier_sync_named("hero_pipeline_tasks_warmed", timeout=_MULTIHOST_TIMEOUT)
    _log("pipeline_tasks_warmed", task_count=task_count, elapsed_seconds=time.monotonic() - started)
    started = time.monotonic()
    _initialize_pipeline_communicators(mpmd_mesh, placements)
    _log("pipeline_communicators_initialized", elapsed_seconds=time.monotonic() - started)
    for completed_steps in range(1, args.steps + 1):
        started = time.monotonic()
        state, metrics = compiled_step(state, batches, denominator)
        jax.block_until_ready((state, metrics))
        if args.synchronize_devices_after_step:
            _synchronize_local_cuda_devices(completed_steps)
        if args.offload_opt_state:
            assert all(value.sharding.memory_kind == "pinned_host" for value in jax.tree.leaves(state.opt_state))
        elapsed = time.monotonic() - started
        loss = _global_loss(metrics[TRAIN_LOSS_KEY])
        mfu_percent = 100 * batch_size * flops_per_example / (elapsed * jax.device_count() * peak_flops)
        _log(
            "pipeline_step",
            step=completed_steps,
            loss=loss,
            elapsed_seconds=elapsed,
            tokens_per_second=batch_size * model_config.max_seq_len / elapsed,
            mfu_percent=mfu_percent,
        )
        if args.run_id and jax.process_index() == 0:
            wandb.log({"train/loss": loss, "step_seconds": elapsed, "throughput/mfu": mfu_percent}, step=completed_steps)
    barrier_sync_named("hero_pipeline_smoke_complete", timeout=_MULTIHOST_TIMEOUT)
    _log("pipeline_complete", steps=args.steps, full_hero=full_hero)
    if args.run_id and jax.process_index() == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
