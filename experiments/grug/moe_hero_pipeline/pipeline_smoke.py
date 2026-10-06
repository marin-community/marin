# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded synthetic Hero pipeline trial with recipe controls and checkpoint resume."""

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
from fray.device_flops import device_flops_for_jax_device
from iris.jax.init import initialize_jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxpp import dime2, env_vars
from levanter.cutlass_kernel_cache import install as install_cutlass_cache
from levanter.data.text.examples import GrugLmExample
from levanter.pipeline import reshape_batch_into_microbatches
from levanter.utils.jax_utils import barrier_sync_named, multihost_allgather_sync

from experiments.grug.moe_hero_ep.hero_recipe import HERO_MODEL_CONFIG
from experiments.grug.moe_hero_ep.heuristic import MoeHeuristic
from experiments.grug.moe_hero_ep.model import OFFLOAD_CARRY_REMAT_MODE, GrugModelConfig, QbEstimator
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_hero_ep.train import (
    RAGGED_MOE_IMPLEMENTATION,
    _apply_hero_ep_runtime_defaults,
    _compute_flops,
    verify_ragged_pjrt,
)
from experiments.grug.moe_hero_pipeline.arguments import parse_args
from experiments.grug.moe_hero_pipeline.checkpoint import restore_checkpoint, save_checkpoint
from experiments.grug.moe_hero_pipeline.pipeline import (
    _HOST_MEMORY_KIND,
    BATCH_AXES,
    TRAIN_LOSS_KEY,
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
_MP_POLICY = "params=bfloat16,compute=bfloat16,output=bfloat16"
_ADAMW_LEARNING_RATE = 1e-4
_ADAMW_BETA1 = 0.9
_ADAMW_BETA2 = 0.95
_ADAMW_MOMENTUM_DTYPE = jnp.bfloat16
_ADAMW_WEIGHT_DECAY = 0.1


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


def _model_config(args: argparse.Namespace) -> GrugModelConfig:
    if args.main_hero_recipe:
        model_config = HERO_MODEL_CONFIG
    elif args.full_hero:
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
    if args.attention_implementation is not None:
        model_config = dataclasses.replace(model_config, attention_implementation=args.attention_implementation)
    if args.moe_implementation is not None:
        model_config = dataclasses.replace(
            model_config, moe_implementation=args.moe_implementation, num_expert_waves=args.expert_waves
        )
    if args.diagnostic_layers is not None:
        model_config = dataclasses.replace(model_config, num_layers=args.diagnostic_layers)
    if args.sequence_length is not None:
        model_config = dataclasses.replace(model_config, max_seq_len=args.sequence_length)
    if args.offload_activations:
        model_config = dataclasses.replace(model_config, remat_mode=OFFLOAD_CARRY_REMAT_MODE)
    return model_config


def _optimizer_and_contract(
    args: argparse.Namespace, model_config: GrugModelConfig, batch_size: int
) -> tuple[optax.GradientTransformation, dict[str, object]]:
    if args.main_hero_recipe:
        optimizer_config = dataclasses.replace(
            MoeHeuristic().build_optimizer_config(
                num_train_steps=args.steps,
                batch_size=batch_size,
                hidden_dim=model_config.hidden_dim,
                seq_len=model_config.max_seq_len,
            ),
            use_syrk=all("H100" not in device.device_kind for device in jax.local_devices()),
            gate_router_weight_decay=0.02,
            expert_normalization=args.expert_normalization,
        )
        optimizer = optimizer_config.build(args.steps)
        optimizer_contract = {"type": "muonh", **dataclasses.asdict(optimizer_config)}
    elif args.optimizer == "muonh":
        optimizer_config = GrugMoeMuonHConfig(
            learning_rate=13 / 3 * 1e-4,
            adam_lr=1e-4,
            warmup=0,
            lr_schedule="constant",
            expert_normalization=args.expert_normalization,
        )
        optimizer = optimizer_config.build(args.steps)
        optimizer_contract = {"type": "muonh", **dataclasses.asdict(optimizer_config)}
    else:
        optimizer = optax.adamw(
            _ADAMW_LEARNING_RATE,
            b1=_ADAMW_BETA1,
            b2=_ADAMW_BETA2,
            mu_dtype=_ADAMW_MOMENTUM_DTYPE,
            weight_decay=_ADAMW_WEIGHT_DECAY,
        )
        optimizer_contract = {
            "type": "adamw",
            "learning_rate": _ADAMW_LEARNING_RATE,
            "b1": _ADAMW_BETA1,
            "b2": _ADAMW_BETA2,
            "mu_dtype": np.dtype(_ADAMW_MOMENTUM_DTYPE).name,
            "weight_decay": _ADAMW_WEIGHT_DECAY,
        }
    return optimizer, optimizer_contract


def main() -> None:
    args = parse_args()
    model_config = _model_config(args)
    if args.main_hero_recipe:
        _apply_hero_ep_runtime_defaults(
            inline_watch_enabled=False,
            moe_implementation=model_config.moe_implementation,
            remat_mode=model_config.remat_mode,
            processes_per_task=args.processes_per_task,
        )
        if model_config.moe_implementation == RAGGED_MOE_IMPLEMENTATION:
            verify_ragged_pjrt()
    # Configure both caches before Iris initialization can select object storage.
    jax.config.update("jax_compilation_cache_dir", args.compilation_cache)
    install_cutlass_cache(PersistentKvCache.in_memory())
    if args.coordinator_address:
        if args.num_processes is None or args.process_id is None or args.local_device_id is None:
            raise ValueError("External bootstrap requires num-processes, process-id, and local-device-id")
        jax.distributed.initialize(
            coordinator_address=args.coordinator_address,
            num_processes=args.num_processes,
            process_id=args.process_id,
            local_device_ids=[args.local_device_id],
        )
    initialize_jax()
    if args.run_id and jax.process_index() == 0:
        wandb.init(
            entity="marin-community",
            project="marin_moe",
            id=args.run_id,
            name=args.run_id,
            group="pipeline-gpu-validation",
            resume="never",
            config=vars(args),
            settings=wandb.Settings(save_code=False, console="redirect"),
        )
    config = GrugMoePipelineConfig(
        stages=args.stages, microbatches=args.microbatches, physical_stages=args.physical_stages
    )
    placements = automatic_stage_to_mpmd_indices(config, args.schedule)
    mesh, mpmd_mesh = make_pipeline_mesh(config, expert_axis_size=args.expert_axis_size, replica_axis_size=1)
    full_hero = (args.full_hero or args.main_hero_recipe) and args.diagnostic_layers is None
    batch_multiple = args.microbatches * jax.device_count() // config.mpmd_stages
    batch_size = args.batch_size if args.batch_size is not None else batch_multiple
    if batch_size < 1 or batch_size % batch_multiple:
        raise ValueError(f"batch-size must be a positive multiple of {batch_multiple}")
    mp_policy = "params=float32,compute=bfloat16,output=bfloat16" if args.main_hero_recipe else _MP_POLICY
    policy = jmp.get_policy(mp_policy)
    flops_per_example, _ = _compute_flops(model_config=model_config)
    peak_flops = device_flops_for_jax_device(jax.local_devices()[0].device_kind)
    assert peak_flops is not None
    optimizer, optimizer_contract = _optimizer_and_contract(args, model_config, batch_size)
    checkpoint_contract = {
        "model": dataclasses.asdict(model_config),
        "mp_policy": mp_policy,
        "optimizer": optimizer_contract,
        "training_steps": args.steps,
        "qb_bias_mode": args.qb_bias_mode,
    }
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
        optimizer=f"{args.optimizer}",
        mp_policy=mp_policy,
        optimizer_config=optimizer_contract,
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
        assert all(value.sharding.memory_kind == _HOST_MEMORY_KIND for value in jax.tree.leaves(state.opt_state))
        _log("optimizer_state_offloaded", memory_kind=_HOST_MEMORY_KIND)
    tokens = np.random.default_rng(args.seed).integers(
        model_config.vocab_size, size=(batch_size, model_config.max_seq_len), dtype=np.int32
    )
    weights = np.ones_like(tokens, dtype=np.float32)
    weights[:, -1] = 0
    with jax.set_mesh(mesh):
        sharding = NamedSharding(mesh, P(BATCH_AXES, None))
        batch = GrugLmExample(tokens=jax.device_put(tokens, sharding), loss_weight=jax.device_put(weights, sharding))
    with jax.set_mesh(mesh):
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
        logsumexp_weight=1e-4 if args.main_hero_recipe else None,
        offload_opt_state=args.offload_opt_state,
        qb_bias_mode=args.qb_bias_mode,
    )
    started = time.monotonic()
    prepared = prepare_automatic_mpmd_step(step, state, batches, denominator, mpmd_mesh)
    _log("pipeline_compiled", elapsed_seconds=time.monotonic() - started)
    state = prepared.state
    compiled_step = prepared.step
    batches = prepared.batches
    denominator = prepared.loss_denominator
    del prepared
    start_step = 0
    if args.checkpoint_root:
        state, start_step = restore_checkpoint(
            args.checkpoint_root, state, compiled_step.in_shardings[0][0], contract=checkpoint_contract
        )
        if args.offload_opt_state:
            assert all(value.sharding.memory_kind == _HOST_MEMORY_KIND for value in jax.tree.leaves(state.opt_state))
        _log("pipeline_checkpoint_restored", step=start_step, checkpoint_root=args.checkpoint_root)
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
    last_step = args.stop_after_step or args.steps
    for completed_steps in range(start_step + 1, last_step + 1):
        started = time.monotonic()
        state, metrics = compiled_step(state, batches, denominator)
        jax.block_until_ready((state, metrics))
        if args.synchronize_devices_after_step:
            _synchronize_local_cuda_devices(completed_steps)
        if args.offload_opt_state:
            assert all(value.sharding.memory_kind == _HOST_MEMORY_KIND for value in jax.tree.leaves(state.opt_state))
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
            wandb.log(
                {TRAIN_LOSS_KEY: loss, "step_seconds": elapsed, "throughput/mfu": mfu_percent}, step=completed_steps
            )
        if args.checkpoint_root and args.checkpoint_every_steps:
            if completed_steps % args.checkpoint_every_steps == 0 or completed_steps == last_step:
                path = save_checkpoint(args.checkpoint_root, state, step=completed_steps, contract=checkpoint_contract)
                _log("pipeline_checkpoint_saved", step=completed_steps, path=path)
    barrier_sync_named("hero_pipeline_smoke_complete", timeout=_MULTIHOST_TIMEOUT)
    _log("pipeline_complete", steps=last_step, full_hero=full_hero)
    if args.run_id and jax.process_index() == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
