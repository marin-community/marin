# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Four-GB200 matched training launcher for Grug MoE backends."""

import dataclasses
import logging
import os
from enum import StrEnum

import click
import jmp
from fray.cluster import ResourceConfig
from iris.hooks.multigpu import IRIS_MULTIGPU_CHILD_WRAPPER_ENV
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.kernels.mixture_of_kittens import (
    MokLikeBackwardPeerStorage,
    MokLikeBuildConfig,
    MokLikeConfig,
    MokLikeForwardXStorage,
    MokLikeWorkspaceTransport,
)
from levanter.kernels.mixture_of_kittens.source import SUPPORTED_NUM_DEVICES
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
from experiments.grug.moe_hero_ep.launch import JAX_NIGHTLY_WHEELS_20260809, pjrt_wheel_install_script
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

logger = logging.getLogger(__name__)

DEFAULT_STEPS = 25
DEFAULT_WANDB_PROJECT = "marin_moe"
BATCH_SIZE_PER_NODE = 64
GPUS_PER_NODE = 4
CPUS_PER_NODE = 96
RAM_PER_NODE = "900g"
MATCHED_CAPACITY_FACTOR = 1.1
# The sealed v15 expert width (#8108). Both arms of the comparison are pinned to it because the
# fused kernel cannot express the hero's asymmetric routed/shared widths.
MOK_LIKE_EXPERT_INTERMEDIATE_DIM = 3072
# The matched comparison shape from #8108. An expert group wider than this needs more experts:
# the axis has to divide them.
MOK_LIKE_MATCHED_NUM_EXPERTS = 8
# The hero routes top-4 of its bank and carries two shared experts. The matched comparison ran a
# single shared expert because the fused kernel could only express one shared width; both are knobs
# now so the proposed 8-of-384 architecture is reachable.
MATCHED_NUM_EXPERTS_PER_TOKEN = 4
MATCHED_NUM_SHARED_EXPERTS = 1
PRODUCTION_MOK_LIKE_WORKSPACE_SLOTS = 1
DEFAULT_GPU_DEVICE_MEMORY_FRACTION = 0.85
PROMOTED_MOK_LIKE_PINNED_HOST_MEMORY_LIMIT_GB = 176
STRICT_WORST_CASE_FOUR_RANK_SCHEDULE_CAPACITY_FACTOR = 4.0
FSDP_EXPERT_CHUNKS = 4
RAGGED_EP_XLA_FLAGS = (
    "--xla_gpu_enable_latency_hiding_scheduler=false",
    "--xla_gpu_experimental_parallel_collective_overlap_limit=1",
)
# Sharded autotuning splits the search across processes and exchanges results through the
# coordination service. Under the fabric transport every rank is its own process and all four
# park in backend_compile_and_load at zero CPU, so the step never compiles.
CROSS_PROCESS_XLA_FLAGS = ("--xla_gpu_shard_autotuning=false",)
MIXED_PRECISION = "params=float32,compute=bfloat16,output=bfloat16"
MOK_LIKE_BUILD_PACKAGES = (
    "nvidia-cuda-cccl==13.0.85",
    "nvidia-cuda-crt==13.0.88",
    "nvidia-cuda-nvcc==13.0.88",
    "nvidia-cuda-runtime==13.0.88",
    "nvidia-nvvm==13.0.88",
)
# The image carries CUDA as wheels rather than a toolkit, so compute-sanitizer is absent unless
# asked for. Install it only when a run is actually wrapping its ranks in it, since it is a large
# download that every ordinary run would otherwise pay for.
MOK_LIKE_SANITIZER_PACKAGES = ("nvidia-cuda-sanitizer-api==13.0.85",)


def _mok_like_build_packages() -> tuple[str, ...]:
    wrapper = os.environ.get(IRIS_MULTIGPU_CHILD_WRAPPER_ENV, "")
    if "compute-sanitizer" in wrapper:
        return MOK_LIKE_BUILD_PACKAGES + MOK_LIKE_SANITIZER_PACKAGES
    return MOK_LIKE_BUILD_PACKAGES


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
    pinned_host_memory_limit_gb: int | None
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
    pinned_host_memory_limit_gb=PROMOTED_MOK_LIKE_PINNED_HOST_MEMORY_LIMIT_GB,
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
    pinned_host_memory_limit_gb=None,
    max_retries_failure=0,
    max_retries_preemption=0,
    max_task_failures=0,
)


def _xla_flag_overrides(backend: MoeBackend, workspace_transport: MokLikeWorkspaceTransport) -> tuple[str, ...]:
    overrides = RAGGED_EP_XLA_FLAGS if backend is MoeBackend.EP else ()
    if workspace_transport.crosses_processes:
        overrides += CROSS_PROCESS_XLA_FLAGS
    return overrides


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
    mok_like_num_devices: int = 4,
    mok_like_workspace_transport: str = MokLikeWorkspaceTransport.IN_PROCESS_PEER.value,
    # A latent projection halves the bytes each comm CTA moves, so an SM split tuned for
    # full-width traffic over-provisions comms. Left unset these keep the kernel defaults.
    mok_like_num_comm_sms: int | None = None,
    mok_like_bwd_num_comm_sms: int | None = None,
    mok_like_minibatch_size: int | None = None,
    mok_like_macrobatch_size: int | None = None,
    # Zero disables the hero's latent projection on both arms, which is the control the fused
    # backend needs to attribute a regression to the two-width path rather than to the shape.
    latent_dim: int | None = None,
    routed_intermediate_dim: int | None = None,
    pinned_host_memory_limit_gb: int | None = None,
    batch_size_per_node: int | None = None,
    shared_intermediate_dim: int = MOK_LIKE_EXPERT_INTERMEDIATE_DIM,
    num_experts: int = MOK_LIKE_MATCHED_NUM_EXPERTS,
    num_experts_per_token: int = MATCHED_NUM_EXPERTS_PER_TOKEN,
    num_shared_experts: int = MATCHED_NUM_SHARED_EXPERTS,
    mok_like_remat_mode: str = "offload_moe",
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
    max_retries_preemption: int | None = None,
    max_retries_failure: int | None = None,
    max_task_failures: int | None = None,
    jax_nightly: bool = False,
    pjrt_wheel: str | None = None,
    version: str | None = None,
) -> ArtifactStep[MoeBackendComparisonResult]:
    """Build one arm of the weak-scaled matched MoE backend comparison."""

    workspace_transport = MokLikeWorkspaceTransport(mok_like_workspace_transport)
    if backend is MoeBackend.MOK_LIKE:
        mok_like_preset = mok_like_preset or MokLikeExperimentPreset.PROMOTED_DROPLESS_V12
        preset = _MOK_LIKE_PRESETS[mok_like_preset]
        if pinned_host_memory_limit_gb is not None:
            # Widening the routed experts widens the offloaded context with them, so the reviewed
            # pin is not a property of the backend -- it is a property of one shape.
            preset = dataclasses.replace(preset, pinned_host_memory_limit_gb=pinned_host_memory_limit_gb)
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
    # The presets tolerate zero preemption retries because a seal wants a run that either completes
    # untouched or reports why it did not. Iteration on a contended cluster wants the opposite: a
    # rack run pays twenty minutes of compile before its first update, so one preemption discards
    # the entire attempt. Buying restarts here keeps a measurement reachable without moving to a
    # priority band that displaces other users' work.
    max_retries_preemption = preset.max_retries_preemption if max_retries_preemption is None else max_retries_preemption
    # A preemption often surfaces on the surviving ranks as a coordination-service connection
    # failure rather than as a preemption, and that is charged here instead. At rack scale the
    # exposure grows with the task count, so an iteration run needs both budgets to be non-zero
    # or one lost task discards every other rank's compile.
    max_retries_failure = preset.max_retries_failure if max_retries_failure is None else max_retries_failure
    # A gang tolerates zero task failures by default, so one rank losing the coordination service
    # kills every other rank's compile. That is the right contract for a seal and the wrong one
    # for a thirty-two node iteration run, where a transient connection failure is likely enough
    # to prevent any measurement from ever completing.
    max_task_failures = preset.max_task_failures if max_task_failures is None else max_task_failures
    forward_x_storage = preset.forward_x_storage if forward_x_storage is None else forward_x_storage
    backward_peer_storage = preset.backward_peer_storage if backward_peer_storage is None else backward_peer_storage
    # The direct-read storage modes reach a peer's XLA buffer through a process-local peer mapping.
    # Under a cross-process transport that address belongs to another process and only the arena is
    # mapped across them, so the presets' choice is unavailable here. Report the substitution rather
    # than letting MokLikeConfig reject every fabric run that uses a preset.
    if workspace_transport.crosses_processes and forward_x_storage.reads_xla_buffers_directly:
        logger.info(
            "%s cannot read peer XLA buffers; using %s for forward x",
            workspace_transport,
            MokLikeForwardXStorage.RUNTIME_STAGED,
        )
        forward_x_storage = MokLikeForwardXStorage.RUNTIME_STAGED
    if workspace_transport.crosses_processes and backward_peer_storage.reads_xla_buffers_directly:
        logger.info(
            "%s cannot read peer XLA buffers; using %s for backward peers",
            workspace_transport,
            MokLikeBackwardPeerStorage.RUNTIME_STAGED,
        )
        backward_peer_storage = MokLikeBackwardPeerStorage.RUNTIME_STAGED
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
    # Data-parallel reduction costs the same per step whatever the batch, so tokens per node per
    # step set how much compute that fixed cost is amortised over. It is the lever on the gap
    # between a single node's MFU and a rack's.
    global_batch_size = (batch_size_per_node or BATCH_SIZE_PER_NODE) * num_nodes
    scale_tag, wandb_group = _SCALE_METADATA[num_nodes]
    model, optimizer = build_hero_configs(num_train_steps=num_steps, batch_size=global_batch_size)
    common_model = dataclasses.replace(
        model,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        num_shared_experts=num_shared_experts,
        capacity_factor=MATCHED_CAPACITY_FACTOR,
        # The fused kernel now carries separate routed and shared widths on both axes -- the token
        # width a latent projection narrows, and the intermediate width the hero widens -- so the
        # whole hero shape comes through unpinned. `--routed-intermediate-dim` still overrides it,
        # and moves both arms together so the comparison stays matched.
        **(
            {}
            if routed_intermediate_dim is None
            else {"intermediate_dim": routed_intermediate_dim, "shared_expert_intermediate_dim": shared_intermediate_dim}
        ),
        **({} if latent_dim is None else {"latent_dim": latent_dim or None}),
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
                num_devices=mok_like_num_devices,
                workspace_transport=workspace_transport,
                **({} if mok_like_minibatch_size is None else {"minibatch_size": mok_like_minibatch_size}),
                **({} if mok_like_macrobatch_size is None else {"macrobatch_size": mok_like_macrobatch_size}),
                **({} if mok_like_num_comm_sms is None else {"num_comm_sms": mok_like_num_comm_sms}),
                **({} if mok_like_bwd_num_comm_sms is None else {"bwd_num_comm_sms": mok_like_bwd_num_comm_sms}),
            ),
            expert_chunks=1,
            remat_mode=mok_like_remat_mode,
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
    replica_axis_size = num_nodes
    # An expert group wider than a node spans hosts, so the expert axis takes those GPUs and the
    # replica axis keeps what is left. Every other arm keeps the sealed within-node group, where
    # this reproduces the previous `replica_axis_size=num_nodes` exactly.
    if backend is MoeBackend.MOK_LIKE and workspace_transport.crosses_processes and mok_like_num_devices > GPUS_PER_NODE:
        expert_axis_size = mok_like_num_devices
        total_devices = num_nodes * GPUS_PER_NODE
        if total_devices % expert_axis_size != 0:
            raise ValueError(
                f"expert group of {expert_axis_size} does not divide the {total_devices} devices "
                f"on {num_nodes} nodes"
            )
        replica_axis_size = total_devices // expert_axis_size
    if model.num_experts % expert_axis_size != 0:
        raise ValueError(
            f"num_experts={model.num_experts} must divide over an expert axis of {expert_axis_size}; "
            f"pass --num-experts with a multiple of {expert_axis_size}"
        )
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
        replica_axis_size=replica_axis_size,
        sharding_dump_path=None,
    )
    name = f"grug/moe-backend-comparison/{backend.value}/{run_id}"
    version = resolve_version(name, version)
    slim = _slimpajama_6b_dataset()

    # A self-built PJRT wheel carries the stock nightly alongside it, so asking for both would
    # install the nightly twice and let the stock plugin win the second write.
    runtime_packages = _mok_like_build_packages() if backend is MoeBackend.MOK_LIKE else ()
    if jax_nightly and pjrt_wheel is None:
        runtime_packages += JAX_NIGHTLY_WHEELS_20260809
    runtime_setup_scripts = (pjrt_wheel_install_script(pjrt_wheel),) if pjrt_wheel is not None else ()

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
                        [f"mok-like-pinned-host-memory-{preset.pinned_host_memory_limit_gb}gb"]
                        if preset.pinned_host_memory_limit_gb is not None
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
                    # The adapter is compiled for a fixed rank count and the fabric rendezvous
                    # checks this one, not the model config's. Leaving it at the default asked a
                    # 64-process job to rendezvous a four-rank group.
                    num_devices=mok_like_num_devices,
                )
                if backend is MoeBackend.MOK_LIKE
                else None
            ),
            mok_like_pinned_host_memory_limit_gb=preset.pinned_host_memory_limit_gb,
            gpu_allocator=gpu_allocator,
            gpu_temp_buffer_pool=gpu_temp_buffer_pool,
            gpu_default_pool_preallocation=gpu_default_pool_preallocation,
            gpu_default_pool_trim_interval_updates=gpu_default_pool_trim_interval_updates,
            xla_autotune_cache_mode=xla_autotune_cache_mode,
            gpu_device_memory_fraction=device_memory_fraction,
            xla_flag_overrides=_xla_flag_overrides(backend, workspace_transport),
            # Fabric transport builds its peer table from imported handles rather
            # than by switching devices inside one process, so each rank must be its
            # own process with a single visible GPU. The in-process path keeps the
            # sealed one-process-per-node layout.
            processes_per_task=(
                GPUS_PER_NODE if workspace_transport is MokLikeWorkspaceTransport.FABRIC_SYMMETRIC else 1
            ),
            pip_packages=runtime_packages,
            extra_setup_scripts=runtime_setup_scripts,
            max_retries_failure=max_retries_failure,
            max_retries_preemption=max_retries_preemption,
            max_task_failures=max_task_failures,
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
    mok_like_num_devices: int = 4,
    mok_like_num_comm_sms: int | None = None,
    mok_like_bwd_num_comm_sms: int | None = None,
    mok_like_minibatch_size: int | None = None,
    mok_like_macrobatch_size: int | None = None,
    latent_dim: int | None = None,
    routed_intermediate_dim: int | None = None,
    pinned_host_memory_limit_gb: int | None = None,
    batch_size_per_node: int | None = None,
    shared_intermediate_dim: int = MOK_LIKE_EXPERT_INTERMEDIATE_DIM,
    num_experts: int = MOK_LIKE_MATCHED_NUM_EXPERTS,
    num_experts_per_token: int = MATCHED_NUM_EXPERTS_PER_TOKEN,
    num_shared_experts: int = MATCHED_NUM_SHARED_EXPERTS,
    mok_like_remat_mode: str = "offload_moe",
    mok_like_workspace_transport: str = MokLikeWorkspaceTransport.IN_PROCESS_PEER.value,
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
    max_retries_preemption: int | None = None,
    max_retries_failure: int | None = None,
    max_task_failures: int | None = None,
    jax_nightly: bool = False,
    pjrt_wheel: str | None = None,
    version: str | None = None,
) -> ArtifactStep[MoeBackendComparisonResult]:
    """Build the supported Marin-native arm of the matched comparison."""
    return build_backend_comparison_run(
        run_id=run_id,
        num_steps=num_steps,
        backend=MoeBackend.MOK_LIKE,
        mok_like_preset=mok_like_preset,
        mok_like_num_devices=mok_like_num_devices,
        mok_like_num_comm_sms=mok_like_num_comm_sms,
        mok_like_bwd_num_comm_sms=mok_like_bwd_num_comm_sms,
        mok_like_minibatch_size=mok_like_minibatch_size,
        mok_like_macrobatch_size=mok_like_macrobatch_size,
        latent_dim=latent_dim,
        routed_intermediate_dim=routed_intermediate_dim,
        shared_intermediate_dim=shared_intermediate_dim,
        pinned_host_memory_limit_gb=pinned_host_memory_limit_gb,
        batch_size_per_node=batch_size_per_node,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        num_shared_experts=num_shared_experts,
        mok_like_remat_mode=mok_like_remat_mode,
        mok_like_workspace_transport=mok_like_workspace_transport,
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
        max_retries_preemption=max_retries_preemption,
        max_retries_failure=max_retries_failure,
        max_task_failures=max_task_failures,
        jax_nightly=jax_nightly,
        pjrt_wheel=pjrt_wheel,
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
    "--mok-like-num-devices",
    type=click.Choice([str(value) for value in SUPPORTED_NUM_DEVICES]),
    default="4",
    show_default=True,
    callback=lambda ctx, param, value: int(value),
    help="Expert-group size. Above four ranks the group spans processes and requires fabric transport.",
)
@click.option(
    "--mok-like-remat-mode",
    type=click.Choice(["offload_moe", "save_moe"]),
    default="offload_moe",
    show_default=True,
    help="Rematerialization for the mok_like arm. offload_moe adds a host transfer per layer; "
    "save_moe keeps the saved context on device.",
)
@click.option(
    "--batch-size-per-node",
    type=click.IntRange(min=1),
    default=None,
    help="Sequences per node per step. Raising it amortises the data-parallel reduction.",
)
@click.option(
    "--pinned-host-memory-limit-gb",
    type=click.IntRange(min=1),
    default=None,
    help="Override the reviewed pinned-host cap that bounds the offloaded MoE context.",
)
@click.option(
    "--routed-intermediate-dim",
    type=click.IntRange(min=256),
    default=None,
    help="Pin the routed expert intermediate width on both arms. Unset keeps the hero's own.",
)
@click.option(
    "--shared-intermediate-dim",
    type=click.IntRange(min=256),
    default=MOK_LIKE_EXPERT_INTERMEDIATE_DIM,
    show_default=True,
    help="Shared expert intermediate width used when --routed-intermediate-dim pins the routed one.",
)
@click.option(
    "--latent-dim",
    type=click.IntRange(min=0),
    default=None,
    help="Override the hero latent width on both arms. Zero removes the projection entirely.",
)
@click.option(
    "--mok-like-minibatch-size",
    type=click.IntRange(min=256),
    default=None,
    help="Override the routed minibatch size, which sets how finely comms overlap compute.",
)
@click.option(
    "--mok-like-macrobatch-size",
    type=click.IntRange(min=256),
    default=None,
    help="Override the routed macrobatch size.",
)
@click.option(
    "--mok-like-num-comm-sms",
    type=click.IntRange(min=2),
    default=None,
    help="Override the forward comm-SM count. Latent narrows the wire, so fewer may suffice.",
)
@click.option(
    "--mok-like-bwd-num-comm-sms",
    type=click.IntRange(min=2),
    default=None,
    help="Override the backward comm-SM count.",
)
@click.option(
    "--num-experts",
    type=int,
    default=MOK_LIKE_MATCHED_NUM_EXPERTS,
    show_default=True,
    help="Routed experts on the matched shape. Must divide over the expert axis, so an expert "
    "group wider than this needs a larger count.",
)
@click.option(
    "--num-experts-per-token",
    type=click.IntRange(min=1),
    default=MATCHED_NUM_EXPERTS_PER_TOKEN,
    show_default=True,
    help="Routed experts each token selects. Per-token routed FLOPs scale with this, so it moves "
    "the MFU denominator as well as the shape.",
)
@click.option(
    "--num-shared-experts",
    type=click.IntRange(min=0),
    default=MATCHED_NUM_SHARED_EXPERTS,
    show_default=True,
    help="Shared experts every token passes through. The hero carries two of them.",
)
@click.option(
    "--mok-like-workspace-transport",
    type=click.Choice([transport.value for transport in MokLikeWorkspaceTransport]),
    default=MokLikeWorkspaceTransport.IN_PROCESS_PEER.value,
    show_default=True,
    help=(
        "How the peer workspace is made visible. in_process_peer is the sealed EP4 path; "
        "fabric_symmetric exchanges CUDA VMM fabric handles and is required above four ranks."
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
@click.option(
    "--max-retries-preemption",
    type=click.IntRange(min=0),
    default=None,
    help=(
        "Override the preset's zero preemption retries. A rack run pays its compile before the "
        "first update, so one preemption discards the whole attempt; restarts keep a measurement "
        "reachable on a contended cluster without taking a higher priority band."
    ),
)
@click.option(
    "--max-retries-failure",
    type=click.IntRange(min=0),
    default=None,
    help=(
        "Override the preset's zero failure retries. A preemption often reaches the surviving "
        "ranks as a coordination-service connection failure, which is charged here rather than to "
        "the preemption budget."
    ),
)
@click.option(
    "--max-task-failures",
    type=click.IntRange(min=0),
    default=None,
    help=(
        "Override the preset's zero tolerated task failures. One rank losing the coordination "
        "service otherwise destroys every other rank's compile, which at thirty-two nodes can "
        "prevent a measurement from ever completing."
    ),
)
@click.option(
    "--jax-nightly",
    is_flag=True,
    default=False,
    help=(
        "Install the pinned dev20260809 JAX nightly instead of the workspace pin. Stock 0.11.0 "
        "deadlocks in cross-process clique init when one process owns several local GPUs "
        "(marin#8081, MEP-024); the nightly does not."
    ),
)
@click.option(
    "--pjrt-wheel",
    type=str,
    default=None,
    help=(
        "Object-store URL of a self-built jax-cuda13-pjrt wheel. Installs the pinned nightly with "
        "this PJRT substituted, so it implies --jax-nightly. Use it to pick up the kMaxPeers-128 "
        "and 4 KiB window-alignment patches that sixty-four-rank collectives need."
    ),
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
    mok_like_num_devices: int,
    mok_like_num_comm_sms: int | None,
    mok_like_bwd_num_comm_sms: int | None,
    mok_like_minibatch_size: int | None,
    mok_like_macrobatch_size: int | None,
    latent_dim: int | None,
    routed_intermediate_dim: int | None,
    shared_intermediate_dim: int,
    pinned_host_memory_limit_gb: int | None,
    batch_size_per_node: int | None,
    num_experts: int,
    num_experts_per_token: int,
    num_shared_experts: int,
    mok_like_remat_mode: str,
    mok_like_workspace_transport: str,
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
    max_retries_preemption: int | None,
    max_retries_failure: int | None,
    max_task_failures: int | None,
    jax_nightly: bool,
    pjrt_wheel: str | None,
) -> ArtifactStep[MoeBackendComparisonResult]:
    return build_backend_comparison_run(
        run_id=run_id,
        num_steps=num_steps,
        backend=MoeBackend(backend),
        mok_like_preset=mok_like_preset,
        mok_like_num_devices=mok_like_num_devices,
        mok_like_num_comm_sms=mok_like_num_comm_sms,
        mok_like_bwd_num_comm_sms=mok_like_bwd_num_comm_sms,
        mok_like_minibatch_size=mok_like_minibatch_size,
        mok_like_macrobatch_size=mok_like_macrobatch_size,
        latent_dim=latent_dim,
        routed_intermediate_dim=routed_intermediate_dim,
        shared_intermediate_dim=shared_intermediate_dim,
        pinned_host_memory_limit_gb=pinned_host_memory_limit_gb,
        batch_size_per_node=batch_size_per_node,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        num_shared_experts=num_shared_experts,
        mok_like_remat_mode=mok_like_remat_mode,
        mok_like_workspace_transport=mok_like_workspace_transport,
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
        max_retries_preemption=max_retries_preemption,
        max_retries_failure=max_retries_failure,
        max_task_failures=max_task_failures,
        jax_nightly=jax_nightly,
        pjrt_wheel=pjrt_wheel,
    )


if __name__ == "__main__":
    main()
