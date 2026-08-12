# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Four-GB200 matched training launcher for Grug MoE backends."""

import dataclasses
import os
from enum import StrEnum

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.kernels.mixture_of_kittens import (
    MokLikeBackwardPeerStorage,
    MokLikeBuildConfig,
    MokLikeConfig,
    MokLikeForwardXStorage,
)
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.data import mixture, tokenized
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.grug.moe_hero_ep.heuristic import build_hero_configs
from experiments.grug.moe_hero_ep.train import (
    GpuAllocator,
    GpuDefaultPoolPreallocation,
    GpuTempBufferPool,
    GrugRunConfig,
    GrugTrainerConfig,
    XlaAutotuneCacheMode,
    run_grug,
)
from experiments.llama import llama3_tokenizer

DEFAULT_STEPS = 25
DEFAULT_WANDB_PROJECT = "marin_moe"
BATCH_SIZE_PER_NODE = 64
GPUS_PER_NODE = 4
CPUS_PER_NODE = 96
RAM_PER_NODE = "900g"
MATCHED_CAPACITY_FACTOR = 1.1
PRODUCTION_MOK_LIKE_WORKSPACE_SLOTS = 1
DEFAULT_GPU_DEVICE_MEMORY_FRACTION = 0.85
STRICT_WORST_CASE_FOUR_RANK_SCHEDULE_CAPACITY_FACTOR = 4.0
FSDP_EXPERT_CHUNKS = 4
RAGGED_EP_XLA_FLAGS = (
    "--xla_gpu_enable_latency_hiding_scheduler=false",
    "--xla_gpu_experimental_parallel_collective_overlap_limit=1",
)
MIXED_PRECISION = "params=float32,compute=bfloat16,output=bfloat16"
MOK_LIKE_BUILD_PACKAGES = (
    "nvidia-cuda-cccl==13.0.85",
    "nvidia-cuda-crt==13.0.88",
    "nvidia-cuda-nvcc==13.0.88",
    "nvidia-cuda-runtime==13.0.88",
    "nvidia-nvvm==13.0.88",
)
MOK_LIKE_SOURCE_ROOT = "/tmp/marin-mok-like/source"
MOK_LIKE_BUILD_ROOT = "/tmp/marin-mok-like/build"

_SLIMPAJAMA_TOKENIZE_RESOURCES = ResourceConfig(ram="64g", disk="64g")
_SLIMPAJAMA_SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel")
_SCALE_METADATA = {
    1: ("four-gb200", "moe-backend-comparison-4gb200"),
    2: ("two-node", "moe-backend-comparison-2node"),
    16: ("one-rack", "moe-backend-comparison-1rack"),
    32: ("two-rack", "moe-backend-comparison-2rack"),
}
SUPPORTED_NUM_NODES = tuple(_SCALE_METADATA)


class MoeBackend(StrEnum):
    """Matched Grug MoE execution strategies."""

    MOK_LIKE = "mok_like"
    EP = "ep"
    FSDP = "fsdp"


class MokLikeExperimentPreset(StrEnum):
    """Reviewed v12 configurations for the native mok_like scale gates."""

    PROMOTED_DROPLESS_V12 = "promoted_dropless_v12"
    CAPACITY_LIMITED_V12 = "capacity_limited_v12"


@dataclasses.dataclass(frozen=True)
class MokLikeExperimentConfig:
    """Complete runtime identity for one reviewed mok_like experiment arm."""

    schedule_capacity_factor: float
    workspace_slots: int
    forward_x_storage: MokLikeForwardXStorage
    backward_peer_storage: MokLikeBackwardPeerStorage
    gpu_allocator: GpuAllocator
    gpu_temp_buffer_pool: GpuTempBufferPool
    gpu_default_pool_preallocation: GpuDefaultPoolPreallocation
    gpu_default_pool_trim_interval_updates: int | None
    xla_autotune_cache_mode: XlaAutotuneCacheMode
    gpu_device_memory_fraction: float
    max_retries_failure: int
    max_retries_preemption: int
    max_task_failures: int


_PROMOTED_DROPLESS_V12 = MokLikeExperimentConfig(
    schedule_capacity_factor=STRICT_WORST_CASE_FOUR_RANK_SCHEDULE_CAPACITY_FACTOR,
    workspace_slots=PRODUCTION_MOK_LIKE_WORKSPACE_SLOTS,
    forward_x_storage=MokLikeForwardXStorage.XLA_PEER_EXPERIMENTAL,
    backward_peer_storage=MokLikeBackwardPeerStorage.RUNTIME_STAGED,
    gpu_allocator=GpuAllocator.CUDA_ASYNC,
    gpu_temp_buffer_pool=GpuTempBufferPool.SHARED,
    gpu_default_pool_preallocation=GpuDefaultPoolPreallocation.EAGER,
    gpu_default_pool_trim_interval_updates=None,
    xla_autotune_cache_mode=XlaAutotuneCacheMode.LOCAL_ONLY,
    gpu_device_memory_fraction=0.80,
    max_retries_failure=0,
    max_retries_preemption=0,
    max_task_failures=0,
)
_MOK_LIKE_PRESETS = {
    MokLikeExperimentPreset.PROMOTED_DROPLESS_V12: _PROMOTED_DROPLESS_V12,
    MokLikeExperimentPreset.CAPACITY_LIMITED_V12: dataclasses.replace(
        _PROMOTED_DROPLESS_V12,
        schedule_capacity_factor=MATCHED_CAPACITY_FACTOR,
    ),
}
_NON_MOK_LIKE_DEFAULTS = MokLikeExperimentConfig(
    schedule_capacity_factor=MATCHED_CAPACITY_FACTOR,
    workspace_slots=PRODUCTION_MOK_LIKE_WORKSPACE_SLOTS,
    forward_x_storage=MokLikeForwardXStorage.RUNTIME_STAGED,
    backward_peer_storage=MokLikeBackwardPeerStorage.RUNTIME_STAGED,
    gpu_allocator=GpuAllocator.CUDA_ASYNC,
    gpu_temp_buffer_pool=GpuTempBufferPool.SHARED,
    gpu_default_pool_preallocation=GpuDefaultPoolPreallocation.EAGER,
    gpu_default_pool_trim_interval_updates=None,
    xla_autotune_cache_mode=XlaAutotuneCacheMode.REMOTE_SYNC,
    gpu_device_memory_fraction=DEFAULT_GPU_DEVICE_MEMORY_FRACTION,
    max_retries_failure=0,
    max_retries_preemption=0,
    max_task_failures=0,
)


class MoeBackendComparisonResult(Artifact):
    """Metrics-only result from one matched MoE backend arm."""


def _slimpajama_6b_dataset() -> ArtifactStep[TokenizedCache]:
    return tokenized(
        "slimpajama-6b",
        source="DKYoon/SlimPajama-6B",
        tokenizer=llama3_tokenizer,
        resources=_SLIMPAJAMA_TOKENIZE_RESOURCES,
        version="2026.06.28",
    )


def build_backend_comparison_run(
    *,
    run_id: str,
    num_steps: int,
    backend: MoeBackend,
    num_nodes: int = 1,
    num_layers: int | None = None,
    watch_interval: int = 0,
    mok_like_preset: MokLikeExperimentPreset | None = None,
    mok_like_schedule_capacity_factor: float | None = None,
    mok_like_workspace_slots: int | None = None,
    forward_x_storage: MokLikeForwardXStorage | None = None,
    backward_peer_storage: MokLikeBackwardPeerStorage | None = None,
    gpu_allocator: GpuAllocator | None = None,
    gpu_temp_buffer_pool: GpuTempBufferPool | None = None,
    gpu_default_pool_preallocation: GpuDefaultPoolPreallocation | None = None,
    gpu_default_pool_trim_interval_updates: int | None = None,
    xla_autotune_cache_mode: XlaAutotuneCacheMode | None = None,
    gpu_device_memory_fraction: float | None = None,
    version: str | None = None,
) -> ArtifactStep[MoeBackendComparisonResult]:
    """Build one arm of the weak-scaled matched MoE backend comparison."""

    if backend is MoeBackend.MOK_LIKE:
        mok_like_preset = mok_like_preset or MokLikeExperimentPreset.PROMOTED_DROPLESS_V12
        preset = _MOK_LIKE_PRESETS[mok_like_preset]
    else:
        if mok_like_preset is not None:
            raise ValueError("mok_like_preset is only supported by the mok_like backend")
        preset = _NON_MOK_LIKE_DEFAULTS
    mok_like_schedule_capacity_factor = (
        preset.schedule_capacity_factor
        if mok_like_schedule_capacity_factor is None
        else mok_like_schedule_capacity_factor
    )
    mok_like_workspace_slots = preset.workspace_slots if mok_like_workspace_slots is None else mok_like_workspace_slots
    forward_x_storage = preset.forward_x_storage if forward_x_storage is None else forward_x_storage
    backward_peer_storage = preset.backward_peer_storage if backward_peer_storage is None else backward_peer_storage
    gpu_allocator = preset.gpu_allocator if gpu_allocator is None else gpu_allocator
    gpu_temp_buffer_pool = preset.gpu_temp_buffer_pool if gpu_temp_buffer_pool is None else gpu_temp_buffer_pool
    gpu_default_pool_preallocation = (
        preset.gpu_default_pool_preallocation
        if gpu_default_pool_preallocation is None
        else gpu_default_pool_preallocation
    )
    gpu_default_pool_trim_interval_updates = (
        preset.gpu_default_pool_trim_interval_updates
        if gpu_default_pool_trim_interval_updates is None
        else gpu_default_pool_trim_interval_updates
    )
    xla_autotune_cache_mode = (
        preset.xla_autotune_cache_mode if xla_autotune_cache_mode is None else xla_autotune_cache_mode
    )
    gpu_device_memory_fraction = (
        preset.gpu_device_memory_fraction if gpu_device_memory_fraction is None else gpu_device_memory_fraction
    )

    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if watch_interval < 0:
        raise ValueError(f"watch_interval must be non-negative, got {watch_interval}")
    if num_nodes not in SUPPORTED_NUM_NODES:
        raise ValueError(f"num_nodes must be one of {SUPPORTED_NUM_NODES}, got {num_nodes}")
    if not 0.0 < gpu_device_memory_fraction <= 1.0:
        raise ValueError("gpu_device_memory_fraction must be in (0, 1]")
    if gpu_temp_buffer_pool is GpuTempBufferPool.SEPARATE and gpu_allocator is not GpuAllocator.CUDA_ASYNC:
        raise ValueError("a separate GPU temp-buffer pool requires the cuda_async allocator")
    if gpu_temp_buffer_pool is GpuTempBufferPool.SEPARATE:
        gpu_default_pool_preallocation = GpuDefaultPoolPreallocation.ON_DEMAND
    if (
        gpu_allocator is not GpuAllocator.CUDA_ASYNC
        and gpu_default_pool_preallocation is GpuDefaultPoolPreallocation.ON_DEMAND
    ):
        raise ValueError("on-demand default-pool preallocation requires the cuda_async allocator")
    if backend is not MoeBackend.MOK_LIKE and xla_autotune_cache_mode is not XlaAutotuneCacheMode.REMOTE_SYNC:
        raise ValueError("local-only XLA autotune caching is only supported by the mok_like backend")
    if gpu_default_pool_trim_interval_updates is not None:
        if type(gpu_default_pool_trim_interval_updates) is not int or gpu_default_pool_trim_interval_updates <= 0:
            raise ValueError("gpu_default_pool_trim_interval_updates must be positive")
        if gpu_default_pool_trim_interval_updates > num_steps:
            raise ValueError("gpu_default_pool_trim_interval_updates must not exceed num_steps")
        if backend is not MoeBackend.MOK_LIKE:
            raise ValueError("default GPU pool trimming is only supported by the mok_like backend")
        if gpu_allocator is not GpuAllocator.CUDA_ASYNC:
            raise ValueError("default GPU pool trimming requires the cuda_async allocator")
        if gpu_temp_buffer_pool is not GpuTempBufferPool.SHARED:
            raise ValueError("default GPU pool trimming requires the shared temp-buffer pool")
    if type(mok_like_workspace_slots) is not int or not 1 <= mok_like_workspace_slots <= 2:
        raise ValueError("mok_like_workspace_slots must be an integer from 1 through 2")
    if backend is not MoeBackend.MOK_LIKE and mok_like_schedule_capacity_factor != MATCHED_CAPACITY_FACTOR:
        raise ValueError("mok_like_schedule_capacity_factor is only supported by the mok_like backend")
    if backend is not MoeBackend.MOK_LIKE and forward_x_storage is not MokLikeForwardXStorage.RUNTIME_STAGED:
        raise ValueError("forward_x_storage is only supported by the mok_like backend")
    if backend is not MoeBackend.MOK_LIKE and backward_peer_storage is not MokLikeBackwardPeerStorage.RUNTIME_STAGED:
        raise ValueError("backward_peer_storage is only supported by the mok_like backend")
    if backend is not MoeBackend.MOK_LIKE and mok_like_workspace_slots != PRODUCTION_MOK_LIKE_WORKSPACE_SLOTS:
        raise ValueError("mok_like_workspace_slots is only supported by the mok_like backend")
    global_batch_size = BATCH_SIZE_PER_NODE * num_nodes
    scale_tag, wandb_group = _SCALE_METADATA[num_nodes]
    model, optimizer = build_hero_configs(num_train_steps=num_steps, batch_size=global_batch_size)
    common_model = dataclasses.replace(
        model,
        num_experts=8,
        num_shared_experts=1,
        capacity_factor=MATCHED_CAPACITY_FACTOR,
        **({"num_layers": num_layers} if num_layers is not None else {}),
    )
    if backend is MoeBackend.MOK_LIKE:
        model = dataclasses.replace(
            common_model,
            moe_implementation="fixed_all_to_all",
            mok_like=MokLikeConfig(
                schedule_capacity_factor=mok_like_schedule_capacity_factor,
                workspace_slots=mok_like_workspace_slots,
                forward_x_storage=forward_x_storage,
                backward_peer_storage=backward_peer_storage,
            ),
            expert_chunks=1,
            remat_mode="offload_moe",
        )
        expert_axis_size = GPUS_PER_NODE
        remat_tag = "offload-moe"
        device_memory_fraction = gpu_device_memory_fraction
    elif backend is MoeBackend.EP:
        model = dataclasses.replace(
            common_model,
            moe_implementation="ragged_all_to_all",
            mok_like=None,
            expert_chunks=1,
            remat_mode="recompute_all",
        )
        expert_axis_size = GPUS_PER_NODE
        remat_tag = "recompute-all"
        device_memory_fraction = gpu_device_memory_fraction
    else:
        model = dataclasses.replace(
            common_model,
            moe_implementation="sonic_cute",
            mok_like=None,
            expert_chunks=FSDP_EXPERT_CHUNKS,
            remat_mode="recompute_all",
        )
        expert_axis_size = 1
        remat_tag = "recompute-all"
        device_memory_fraction = gpu_device_memory_fraction
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT
    train_resources = ResourceConfig.with_gpu(
        "GB200",
        count=GPUS_PER_NODE,
        cpu=CPUS_PER_NODE,
        ram=RAM_PER_NODE,
        disk="1t",
        replicas=num_nodes,
    )
    grug_trainer = GrugTrainerConfig(
        data_seed=0,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=True,
        expert_axis_size=expert_axis_size,
        replica_axis_size=num_nodes,
        sharding_dump_path=None,
    )
    name = f"grug/moe-backend-comparison/{backend.value}/{run_id}"
    version = resolve_version(name, version)
    slim = _slimpajama_6b_dataset()

    def build_config(ctx: StepContext) -> GrugRunConfig:
        profiler_enabled = num_steps >= 10
        profiler_start_step = 80 if num_steps >= 100 else 5
        trainer = TrainerConfig(
            id=run_id,
            seed=0,
            train_batch_size=global_batch_size,
            num_train_steps=num_steps,
            profiler=ProfilerConfig(
                enabled=profiler_enabled,
                start_step=profiler_start_step,
                num_steps=5,
                process_index=0,
                profile_options=ProfileOptionsConfig(
                    host_tracer_level=1,
                    python_tracer_level=0,
                    enable_hlo_proto=True,
                ),
            ),
            mp=jmp.get_policy(MIXED_PRECISION),
            tracker=WandbConfig(
                entity="marin-community",
                project=wandb_project,
                tags=[
                    "grug",
                    "moe",
                    "moe-backend-comparison",
                    backend.value,
                    model.moe_implementation.replace("_", "-"),
                    remat_tag,
                    *(
                        [f"mok-like-preset-{mok_like_preset.value.replace('_', '-')}"]
                        if mok_like_preset is not None
                        else []
                    ),
                    f"allocator-{gpu_allocator.value}",
                    f"temp-buffer-pool-{gpu_temp_buffer_pool.value}",
                    f"default-pool-preallocation-{gpu_default_pool_preallocation.value.replace('_', '-')}",
                    f"xla-autotune-cache-{xla_autotune_cache_mode.value.replace('_', '-')}",
                    *(
                        [f"default-pool-trim-interval-updates-{gpu_default_pool_trim_interval_updates}"]
                        if gpu_default_pool_trim_interval_updates is not None
                        else ["default-pool-trim-disabled"]
                    ),
                    f"device-memory-{device_memory_fraction:g}",
                    "contiguous-expert-placement",
                    "pgle-off",
                    scale_tag,
                    f"nodes-{num_nodes}",
                    f"gpus-{num_nodes * GPUS_PER_NODE}",
                    f"dp-{num_nodes}",
                    "one-process-per-node",
                    f"steps-{num_steps}",
                    *(
                        [f"mok-like-schedule-capacity-{mok_like_schedule_capacity_factor:g}"]
                        if backend is MoeBackend.MOK_LIKE
                        else []
                    ),
                    *(
                        [f"mok-like-workspace-slots-{mok_like_workspace_slots}"]
                        if backend is MoeBackend.MOK_LIKE
                        else []
                    ),
                    *(
                        ["strict-dropless-four-rank-capacity"]
                        if backend is MoeBackend.MOK_LIKE
                        and mok_like_schedule_capacity_factor == STRICT_WORST_CASE_FOUR_RANK_SCHEDULE_CAPACITY_FACTOR
                        else []
                    ),
                    *(
                        ["forward-x-zero-copy"]
                        if forward_x_storage is MokLikeForwardXStorage.XLA_PEER_EXPERIMENTAL
                        else []
                    ),
                    *(
                        [f"forward-x-storage-{forward_x_storage.value.replace('_', '-')}"]
                        if backend is MoeBackend.MOK_LIKE
                        else []
                    ),
                    *(
                        ["backward-peer-zero-copy"]
                        if backward_peer_storage is MokLikeBackwardPeerStorage.XLA_PEER_EXPERIMENTAL
                        else []
                    ),
                    *(
                        ["backward-inputs-zero-copy"]
                        if backward_peer_storage is MokLikeBackwardPeerStorage.XLA_PEER_INPUTS_EXPERIMENTAL
                        else []
                    ),
                    *(
                        [f"backward-peer-storage-{backward_peer_storage.value.replace('_', '-')}"]
                        if backend is MoeBackend.MOK_LIKE
                        else []
                    ),
                    *(["latency-hiding-off", "collective-overlap-1"] if backend is MoeBackend.EP else []),
                ],
                group=wandb_group,
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(
                watch_targets=["grads", "updates", "params"],
                interval=watch_interval,
            ),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
        )
        return GrugRunConfig(
            model=model,
            data=mixture(ctx, {slim: 1.0}, shuffle=_SLIMPAJAMA_SHUFFLE),
            resources=ctx.runtime_arg("train_resources"),
            optimizer=optimizer,
            trainer=dataclasses.replace(grug_trainer, trainer=trainer),
            eval=None,
            mok_like_build=(
                MokLikeBuildConfig(
                    source_root=MOK_LIKE_SOURCE_ROOT,
                    cache_root=MOK_LIKE_BUILD_ROOT,
                    cuda_arch="sm_100a",
                    clone_if_missing=True,
                )
                if backend is MoeBackend.MOK_LIKE
                else None
            ),
            mok_like_pinned_host_memory_limit_gb=192 if backend is MoeBackend.MOK_LIKE else None,
            gpu_allocator=gpu_allocator,
            gpu_temp_buffer_pool=gpu_temp_buffer_pool,
            gpu_default_pool_preallocation=gpu_default_pool_preallocation,
            gpu_default_pool_trim_interval_updates=gpu_default_pool_trim_interval_updates,
            xla_autotune_cache_mode=xla_autotune_cache_mode,
            gpu_device_memory_fraction=device_memory_fraction,
            xla_flag_overrides=RAGGED_EP_XLA_FLAGS if backend is MoeBackend.EP else (),
            processes_per_task=1,
            pip_packages=MOK_LIKE_BUILD_PACKAGES if backend is MoeBackend.MOK_LIKE else (),
            max_retries_failure=preset.max_retries_failure,
            max_retries_preemption=preset.max_retries_preemption,
            max_task_failures=preset.max_task_failures,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=MoeBackendComparisonResult,
        run=run_grug,
        build_config=build_config,
        deps=(slim,),
        runtime_args={"train_resources": train_resources},
    )


def build_mok_like_run(
    *,
    run_id: str,
    num_steps: int,
    num_nodes: int = 1,
    num_layers: int | None = None,
    watch_interval: int = 0,
    mok_like_preset: MokLikeExperimentPreset = MokLikeExperimentPreset.PROMOTED_DROPLESS_V12,
    mok_like_schedule_capacity_factor: float | None = None,
    mok_like_workspace_slots: int | None = None,
    forward_x_storage: MokLikeForwardXStorage | None = None,
    backward_peer_storage: MokLikeBackwardPeerStorage | None = None,
    gpu_allocator: GpuAllocator | None = None,
    gpu_temp_buffer_pool: GpuTempBufferPool | None = None,
    gpu_default_pool_preallocation: GpuDefaultPoolPreallocation | None = None,
    gpu_default_pool_trim_interval_updates: int | None = None,
    xla_autotune_cache_mode: XlaAutotuneCacheMode | None = None,
    gpu_device_memory_fraction: float | None = None,
    version: str | None = None,
) -> ArtifactStep[MoeBackendComparisonResult]:
    """Build the supported Marin-native arm of the matched comparison."""
    return build_backend_comparison_run(
        run_id=run_id,
        num_steps=num_steps,
        backend=MoeBackend.MOK_LIKE,
        mok_like_preset=mok_like_preset,
        num_nodes=num_nodes,
        num_layers=num_layers,
        watch_interval=watch_interval,
        mok_like_schedule_capacity_factor=mok_like_schedule_capacity_factor,
        mok_like_workspace_slots=mok_like_workspace_slots,
        forward_x_storage=forward_x_storage,
        backward_peer_storage=backward_peer_storage,
        gpu_allocator=gpu_allocator,
        gpu_temp_buffer_pool=gpu_temp_buffer_pool,
        gpu_default_pool_preallocation=gpu_default_pool_preallocation,
        gpu_default_pool_trim_interval_updates=gpu_default_pool_trim_interval_updates,
        xla_autotune_cache_mode=xla_autotune_cache_mode,
        gpu_device_memory_fraction=gpu_device_memory_fraction,
        version=version,
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@click.option(
    "--backend",
    type=click.Choice([backend.value for backend in MoeBackend]),
    default=MoeBackend.MOK_LIKE.value,
    show_default=True,
    help="Matched MoE execution strategy.",
)
@click.option(
    "--num-steps",
    type=click.IntRange(min=1),
    default=DEFAULT_STEPS,
    show_default=True,
    help="Use 25 for a failure screen or 100 for a stable profile-free comparison window.",
)
@click.option(
    "--num-nodes",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Weak-scale over 1, 2, 16, or 32 four-GPU GB200 nodes.",
)
@click.option(
    "--num-layers",
    type=click.IntRange(min=1),
    default=None,
    help="Override the 48-layer production shape for an early integration gate.",
)
@click.option(
    "--watch-interval",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Log parameter, update, and gradient norms at this interval; zero disables it.",
)
@click.option(
    "--mok-like-preset",
    type=click.Choice(MokLikeExperimentPreset, case_sensitive=False),
    default=None,
    help=(
        "Reviewed mok_like configuration. The mok_like backend defaults to promoted_dropless_v12; "
        "capacity_limited_v12 retains the matched factor-1.1 control."
    ),
)
@click.option(
    "--mok-like-schedule-capacity-factor",
    type=click.FloatRange(min=1.0),
    default=None,
    help=(
        "Override the preset's static schedule headroom for mok_like only. Factor 4 is dropless for "
        "four expert ranks; factor 1.1 is the capacity-limited matched control."
    ),
)
@click.option(
    "--mok-like-workspace-slots",
    type=click.IntRange(min=1, max=2),
    default=None,
    help="Override the preset with one production workspace slot or two concurrent-call stress slots.",
)
@click.option(
    "--forward-x-storage",
    type=click.Choice(MokLikeForwardXStorage, case_sensitive=False),
    default=None,
    help="Override the preset's forward activation storage.",
)
@click.option(
    "--backward-peer-storage",
    type=click.Choice(MokLikeBackwardPeerStorage, case_sensitive=False),
    default=None,
    help="Override the preset's backward peer-buffer storage.",
)
@click.option(
    "--gpu-allocator",
    type=click.Choice(GpuAllocator, case_sensitive=False),
    default=None,
    help="Override the preset's GPU allocator.",
)
@click.option(
    "--gpu-temp-buffer-pool",
    type=click.Choice(GpuTempBufferPool, case_sensitive=False),
    default=None,
    help="Override the preset's XLA temporary-buffer pool.",
)
@click.option(
    "--gpu-default-pool-preallocation",
    type=click.Choice(GpuDefaultPoolPreallocation, case_sensitive=False),
    default=None,
    help="Override the preset's default CUDA-pool preallocation; separate temp pools use on-demand.",
)
@click.option(
    "--gpu-default-pool-trim-interval-updates",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Trim each local GPU's default CUDA pool after every N completed updates; each trim's telemetry is logged "
        "at that update's zero-based W&B step. Disabled when omitted."
    ),
)
@click.option(
    "--xla-autotune-cache-mode",
    type=click.Choice(XlaAutotuneCacheMode, case_sensitive=False),
    default=None,
    help="Override the preset's per-fusion autotune cache persistence mode.",
)
@click.option(
    "--gpu-device-memory-fraction",
    type=click.FloatRange(min=0.0, max=1.0, min_open=True),
    default=None,
    help="Override the preset's XLA device-memory fraction.",
)
@build_options
def main(
    run_id: str,
    backend: str,
    num_steps: int,
    num_nodes: int,
    num_layers: int | None,
    watch_interval: int,
    mok_like_preset: MokLikeExperimentPreset | None,
    mok_like_schedule_capacity_factor: float | None,
    mok_like_workspace_slots: int | None,
    forward_x_storage: MokLikeForwardXStorage | None,
    backward_peer_storage: MokLikeBackwardPeerStorage | None,
    gpu_allocator: GpuAllocator | None,
    gpu_temp_buffer_pool: GpuTempBufferPool | None,
    gpu_default_pool_preallocation: GpuDefaultPoolPreallocation | None,
    gpu_default_pool_trim_interval_updates: int | None,
    xla_autotune_cache_mode: XlaAutotuneCacheMode | None,
    gpu_device_memory_fraction: float | None,
) -> ArtifactStep[MoeBackendComparisonResult]:
    return build_backend_comparison_run(
        run_id=run_id,
        num_steps=num_steps,
        backend=MoeBackend(backend),
        mok_like_preset=mok_like_preset,
        num_nodes=num_nodes,
        num_layers=num_layers,
        watch_interval=watch_interval,
        mok_like_schedule_capacity_factor=mok_like_schedule_capacity_factor,
        mok_like_workspace_slots=mok_like_workspace_slots,
        forward_x_storage=forward_x_storage,
        backward_peer_storage=backward_peer_storage,
        gpu_allocator=gpu_allocator,
        gpu_temp_buffer_pool=gpu_temp_buffer_pool,
        gpu_default_pool_preallocation=gpu_default_pool_preallocation,
        gpu_default_pool_trim_interval_updates=gpu_default_pool_trim_interval_updates,
        xla_autotune_cache_mode=xla_autotune_cache_mode,
        gpu_device_memory_fraction=gpu_device_memory_fraction,
    )


if __name__ == "__main__":
    main()
