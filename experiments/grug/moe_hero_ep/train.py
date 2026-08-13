# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import dataclasses
import functools
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import levanter.callbacks as callbacks
import levanter.tracker
import numpy as np
import optax
from fray.cluster import ResourceConfig
from haliax import Axis
from haliax.partitioning import set_mesh
from iris.runtime.jax_init import XLA_AUTOTUNE_CACHE_MODE_ENV, XlaAutotuneCacheMode
from jax._src import config as jax_config
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_dataclass
from jaxtyping import PRNGKeyArray
from levanter.callbacks.state_adapter import StateCallbackRunner
from levanter.callbacks.watch import WatchConfig, compute_watch_stats
from levanter.data.dataset import AsyncDataset
from levanter.data.loader import DataLoader
from levanter.data.mixture import MixtureDataset, rescale_mixture_schedule_for_batch_schedule
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.examples import GrugLmExample, grug_lm_example_from_named
from levanter.eval import TaggedEvaluator, cb_tagged_evaluate, eval_model
from levanter.grug.sharding import compact_grug_mesh
from levanter.kernels.mixture_of_kittens import (
    MokLikeBackwardPeerStorage,
    MokLikeBuildConfig,
    MokLikeDebugCounters,
    MokLikeForwardXStorage,
    MokLikeMemoryPoolTrimTelemetry,
    MokLikeRuntimeHandle,
    initialize_mok_like_runtime,
)
from levanter.models.lm_model import LmExample
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.schedule import BatchSchedule
from levanter.trainer import TrainerConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.jax_utils import parameter_count
from levanter.utils.logging import LoadingTimeTrackerIterator

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.sharding_dump import dump_grug_state_sharding_run_artifact

# This file intentionally mirrors `experiments/grug/base/train.py` with
# variant-specific model/loss/FLOP wiring, per the grug copy-first workflow in
# `.agents/skills/change-grug/`.

logger = logging.getLogger(__name__)

_MOK_LIKE_PEER_WAIT_PHASES = ("forward_pre", "forward_post", "backward_pre", "backward_post")
_MOK_LIKE_STAGING_COPY_PHASES = ("forward", "backward")
_BF16_BYTES = 2
_FLOAT32_BYTES = 4

HERO_EP_RUNTIME_ENV = {
    "JAX_ENABLE_PGLE": "true",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
}
_XLA_FLAG_DEFAULTS = ("--xla_gpu_enable_latency_hiding_scheduler=true",)
XLA_COLLECTIVE_OVERLAP_FLAG = "--xla_gpu_experimental_parallel_collective_overlap_limit"
DEFAULT_COLLECTIVE_OVERLAP_LIMIT = 4
# Full inline norm watch failed with overlap 4. Overlap 1 completed the selected full-watch gate.
INLINE_WATCH_COLLECTIVE_OVERLAP_LIMIT = 1
# TODO(https://github.com/marin-community/marin/issues/5675): Re-enable XLA GPU
# command buffers after the CUDA graph failure is fixed.
XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="
XLA_SEPARATE_TEMP_BUFFER_FLAG = "--xla_gpu_temp_buffer_use_separate_color"
XLA_FLAGS_ENV = "XLA_FLAGS"


class GpuAllocator(StrEnum):
    """Pinned JAX GPU allocators supported by the Grug hero launcher."""

    CUDA_ASYNC = "cuda_async"
    VMM = "vmm"


class GpuTempBufferPool(StrEnum):
    """Allocation pool used for XLA's compiled executable temporary heap."""

    SHARED = "shared"
    SEPARATE = "separate"


class GpuDefaultPoolPreallocation(StrEnum):
    """Allocation policy for XLA's default CUDA memory pool."""

    EAGER = "eager"
    ON_DEMAND = "on_demand"


class WatchMode(StrEnum):
    """Where a watched training step computes gradient and parameter statistics."""

    INLINE = "inline"
    DIAGNOSTIC = "diagnostic"


def _apply_hero_ep_runtime_defaults(*, inline_watch_enabled: bool, processes_per_task: int = 1) -> None:
    env_defaults = dict(HERO_EP_RUNTIME_ENV)
    if processes_per_task > 1:
        # With one process per GPU, the per-process CUPTI sessions collide with each
        # other and with CoreWeave's DCGM, so PGLE cannot profile and its recompile
        # machinery only adds failure modes. Default it off; an explicit env wins.
        env_defaults["JAX_ENABLE_PGLE"] = "false"
    for name, value in env_defaults.items():
        os.environ.setdefault(name, value)
    xla_flags = os.environ.get(XLA_FLAGS_ENV, "").split()
    overlap_limit = INLINE_WATCH_COLLECTIVE_OVERLAP_LIMIT if inline_watch_enabled else DEFAULT_COLLECTIVE_OVERLAP_LIMIT
    flag_defaults = (
        f"{XLA_COLLECTIVE_OVERLAP_FLAG}={overlap_limit}",
        *_XLA_FLAG_DEFAULTS,
        XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG,
    )
    explicit_names = {flag.partition("=")[0] for flag in xla_flags}
    xla_flags.extend(flag for flag in flag_defaults if flag.partition("=")[0] not in explicit_names)
    os.environ[XLA_FLAGS_ENV] = " ".join(xla_flags)


@dataclass(frozen=True)
class GrugTrainerConfig:
    """Runtime knobs for grug training."""

    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(use_explicit_mesh_axes=True))
    data_seed: int | None = None
    log_every: int = 1
    ema_beta: float | None = None  # EMA coefficient for eval/checkpoint model; None disables EMA.
    z_loss_weight: float = 1e-4  # Weight on final-logit logsumexp z-loss stabilization term.
    # Keep disabled except on model sizes where Grace-Blackwell host offload has been measured.
    # The d6144 EP64 runs used it; d5120 required a 135 GiB pinned-host arena and regressed.
    offload_opt_state: bool = False
    # Inline watch computes statistics on every step and uses the watch interval only for logging.
    # This keeps one training executable resident. A diagnostic watch repeats forward and backward
    # in a separate executable, which costs compute but shortens gradient liveness.
    watch_mode: WatchMode = WatchMode.INLINE
    # A short throughput gate leaves this off. A compute-optimal run needs it: the loop already
    # restores from the latest committed checkpoint, so without a writer an interrupted run
    # restarts at step 0.
    save_checkpoints: bool = False

    # Grug builds its own compact (replica_dcn, data, expert, model) mesh instead of using
    # the Trainer's logical axis mapping; `data` absorbs whatever these two leave free.
    # Defaults reproduce the historical layout: no expert parallelism and full replication
    # across slices (replica_axis_size=None -> jax.process_count()), i.e. parameters
    # replicated per slice and sharded only over the intra-slice `data` axis. For a model
    # too large to replicate within one slice, set replica_axis_size=1 (FSDP across every
    # slice) and expert_axis_size>1 (expert parallelism over the intra-slice devices).
    expert_axis_size: int = 1
    replica_axis_size: int | None = None
    sharding_dump_path: str | None = None


@dataclass(frozen=True)
class GrugEvalConfig:
    """Perplexity eval settings for grug training."""

    eval_batch_size: int = 512
    steps_per_eval: int | None = 1000
    max_eval_batches: int | None = None
    prefix: str = "eval"
    eval_current: bool = True
    eval_ema: bool = True
    compute_bpb: bool = True
    # For expert-parallel runs, also evaluate under the dropless local backend on an
    # expert-collapsed mesh, logging a separate `eval_dropless` macro loss alongside the
    # as-trained (with-drop) eval. No-op when the mesh has no expert parallelism.
    dropless_eval: bool = False


@dataclass(frozen=True)
class GrugRunConfig:
    """Top-level config for grug training."""

    model: GrugModelConfig
    data: LmDataConfig
    resources: ResourceConfig
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    mok_like_build: MokLikeBuildConfig | None = None
    mok_like_pinned_host_memory_limit_gb: int | None = None
    gpu_allocator: GpuAllocator = GpuAllocator.CUDA_ASYNC
    gpu_temp_buffer_pool: GpuTempBufferPool = GpuTempBufferPool.SHARED
    gpu_default_pool_preallocation: GpuDefaultPoolPreallocation = GpuDefaultPoolPreallocation.EAGER
    gpu_default_pool_trim_interval_updates: int | None = None
    xla_autotune_cache_mode: XlaAutotuneCacheMode = XlaAutotuneCacheMode.REMOTE_SYNC
    gpu_device_memory_fraction: float | None = None
    xla_flag_overrides: tuple[str, ...] = ()
    pip_packages: tuple[str, ...] = ()
    max_retries_failure: int = 3
    max_retries_preemption: int = 100
    max_task_failures: int = 10
    # Stop after this many steps while `trainer.num_train_steps` still sizes the learning-rate
    # schedule. Warmup and decay are fractions of `num_train_steps`, so training the head of a
    # long schedule requires the two to differ. None runs the whole schedule.
    stop_after_steps: int | None = None
    # GPU processes per task: > 1 runs one JAX process per GPU (multi-controller)
    # via the iris.hooks.multigpu_main supervisor instead of one process per node.
    processes_per_task: int = 1

    def __post_init__(self) -> None:
        for field_name in ("max_retries_failure", "max_retries_preemption", "max_task_failures"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")


def _initialize_mok_like_for_config(
    config: GrugRunConfig,
    mesh: Mesh,
    *,
    batch_size: int,
) -> MokLikeRuntimeHandle | None:
    if config.model.mok_like is None:
        if config.mok_like_build is not None:
            raise ValueError("mok_like_build was provided but the model does not select mok_like")
        return None
    batch_sizes = config.trainer.trainer.batch_schedule.unique_batch_sizes()
    if len(batch_sizes) != 1:
        raise ValueError(
            "mok_like requires a fixed batch size because its peer workspace has a static token shape; "
            f"configured sizes are {sorted(batch_sizes)}"
        )
    if config.mok_like_build is None:
        raise ValueError("a mok_like model requires explicit mok_like_build configuration")
    if config.eval is not None and config.eval.eval_batch_size != batch_size:
        raise ValueError(
            "mok_like evaluation must use the training batch size because its peer workspace has a static token shape; "
            f"training batch is {batch_size}, eval batch is {config.eval.eval_batch_size}"
        )

    num_tokens = _mok_like_tokens_per_rank(batch_size=batch_size, sequence_length=config.model.max_seq_len, mesh=mesh)
    return initialize_mok_like_runtime(
        build_config=config.mok_like_build,
        num_tokens=num_tokens,
        hidden_dim=config.model.hidden_dim,
        top_k=config.model.num_experts_per_token,
        workspace_slots=config.model.mok_like.workspace_slots,
        mesh=mesh,
        workspace_transport=config.model.mok_like.workspace_transport,
    )


def _mok_like_tokens_per_rank(*, batch_size: int, sequence_length: int, mesh: Mesh) -> int:
    batch_axis_size = 1
    for axis_name in _BATCH_AXES:
        batch_axis_size *= int(mesh.shape[axis_name])
    global_tokens = batch_size * sequence_length
    if global_tokens % batch_axis_size != 0:
        raise ValueError(
            f"global token count {global_tokens} must divide evenly over mok_like batch axes {batch_axis_size}"
        )
    return global_tokens // batch_axis_size


def _apply_dispatch_environment(config: GrugRunConfig) -> None:
    xla_flags = os.environ.get(XLA_FLAGS_ENV, "").split()
    for override in config.xla_flag_overrides:
        if not override.startswith("--") or "=" not in override:
            raise ValueError(f"XLA flag overrides must have the form --name=value, got {override!r}")
        name = override.partition("=")[0]
        xla_flags = [flag for flag in xla_flags if flag.partition("=")[0] != name]
        xla_flags.append(override)
    if not isinstance(config.gpu_temp_buffer_pool, GpuTempBufferPool):
        raise TypeError("gpu_temp_buffer_pool must be a GpuTempBufferPool")
    if not isinstance(config.gpu_default_pool_preallocation, GpuDefaultPoolPreallocation):
        raise TypeError("gpu_default_pool_preallocation must be a GpuDefaultPoolPreallocation")
    if not isinstance(config.xla_autotune_cache_mode, XlaAutotuneCacheMode):
        raise TypeError("xla_autotune_cache_mode must be an XlaAutotuneCacheMode")
    os.environ[XLA_AUTOTUNE_CACHE_MODE_ENV] = config.xla_autotune_cache_mode.value
    if config.gpu_temp_buffer_pool is GpuTempBufferPool.SEPARATE and config.gpu_allocator is not GpuAllocator.CUDA_ASYNC:
        raise ValueError("a separate GPU temp-buffer pool requires the cuda_async allocator")
    if (
        config.gpu_temp_buffer_pool is GpuTempBufferPool.SEPARATE
        and config.gpu_default_pool_preallocation is not GpuDefaultPoolPreallocation.ON_DEMAND
    ):
        raise ValueError("a separate GPU temp-buffer pool requires on-demand default-pool preallocation")
    if (
        config.gpu_allocator is not GpuAllocator.CUDA_ASYNC
        and config.gpu_default_pool_preallocation is GpuDefaultPoolPreallocation.ON_DEMAND
    ):
        raise ValueError("on-demand default-pool preallocation requires the cuda_async allocator")
    trim_interval_updates = config.gpu_default_pool_trim_interval_updates
    if trim_interval_updates is not None:
        if type(trim_interval_updates) is not int or trim_interval_updates <= 0:
            raise ValueError("gpu_default_pool_trim_interval_updates must be positive")
        if config.model.mok_like is None:
            raise ValueError("default GPU pool trimming requires a mok_like model")
        if config.gpu_allocator is not GpuAllocator.CUDA_ASYNC:
            raise ValueError("default GPU pool trimming requires the cuda_async allocator")
        if config.gpu_temp_buffer_pool is not GpuTempBufferPool.SHARED:
            raise ValueError("default GPU pool trimming requires the shared temp-buffer pool")
        if trim_interval_updates > config.trainer.trainer.num_train_steps:
            raise ValueError("gpu_default_pool_trim_interval_updates must not exceed num_train_steps")
    xla_flags = [flag for flag in xla_flags if flag.partition("=")[0] != XLA_SEPARATE_TEMP_BUFFER_FLAG]
    use_separate_temp_pool = config.gpu_temp_buffer_pool is GpuTempBufferPool.SEPARATE
    xla_flags.append(f"{XLA_SEPARATE_TEMP_BUFFER_FLAG}={'true' if use_separate_temp_pool else 'false'}")
    os.environ[XLA_FLAGS_ENV] = " ".join(xla_flags)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
        "true" if config.gpu_default_pool_preallocation is GpuDefaultPoolPreallocation.EAGER else "false"
    )

    host_memory_limit = config.mok_like_pinned_host_memory_limit_gb
    device_memory_fraction = config.gpu_device_memory_fraction
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = config.gpu_allocator.value
    if device_memory_fraction is not None:
        if not 0.0 < device_memory_fraction <= 1.0:
            raise ValueError("gpu_device_memory_fraction must be in (0, 1]")
        os.environ.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
        os.environ["XLA_CLIENT_MEM_FRACTION"] = str(device_memory_fraction)
    if config.model.mok_like is None:
        if host_memory_limit is not None:
            raise ValueError("mok_like pinned-host memory configuration requires a mok_like model")
        return
    if host_memory_limit is None:
        if config.model.remat_mode == "offload_moe":
            raise ValueError("mok_like with offload_moe requires an explicit pinned-host memory limit")
    elif host_memory_limit <= 0:
        raise ValueError("mok_like_pinned_host_memory_limit_gb must be positive")
    else:
        os.environ["XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB"] = str(host_memory_limit)
    if device_memory_fraction is None:
        raise ValueError("mok_like requires an explicit device memory fraction")


def _mok_like_debug_metrics(counters: MokLikeDebugCounters) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {
        "mok_like/runtime/total_peer_wait_events": sum(counters.peer_ready_waits) + sum(counters.completion_waits),
        "mok_like/runtime/staging_copy_calls_total": sum(sum(rank) for rank in counters.staging_copy_calls),
        "mok_like/runtime/staging_copy_bytes_total": sum(sum(rank) for rank in counters.staging_copy_bytes),
    }
    phase_peer_fields = {
        "peer_wait_events": counters.peer_wait_events,
        "peer_wait_cycles": counters.peer_wait_cycles,
        "peer_wait_max_cycles": counters.peer_wait_max_cycles,
    }
    for metric_name, ranks in phase_peer_fields.items():
        for rank, phases in enumerate(ranks):
            for phase, peers in zip(_MOK_LIKE_PEER_WAIT_PHASES, phases, strict=True):
                for peer, value in enumerate(peers):
                    metrics[f"mok_like/runtime/rank_{rank}/{phase}/peer_{peer}/{metric_name}"] = value
    for phase_index, phase in enumerate(_MOK_LIKE_PEER_WAIT_PHASES):
        events = sum(
            counters.peer_wait_events[rank][phase_index][peer]
            for rank in range(len(counters.peer_wait_events))
            for peer in range(len(counters.peer_wait_events[rank][phase_index]))
        )
        cycles = sum(
            counters.peer_wait_cycles[rank][phase_index][peer]
            for rank in range(len(counters.peer_wait_cycles))
            for peer in range(len(counters.peer_wait_cycles[rank][phase_index]))
        )
        maximum = max(
            counters.peer_wait_max_cycles[rank][phase_index][peer]
            for rank in range(len(counters.peer_wait_max_cycles))
            for peer in range(len(counters.peer_wait_max_cycles[rank][phase_index]))
        )
        metrics[f"mok_like/runtime/{phase}/peer_wait_events_total"] = events
        metrics[f"mok_like/runtime/{phase}/peer_wait_cycles_total"] = cycles
        metrics[f"mok_like/runtime/{phase}/peer_wait_max_cycles"] = maximum
        metrics[f"mok_like/runtime/{phase}/peer_wait_mean_cycles"] = cycles / events if events else 0.0
    rank_fields = {
        "peer_ready_waits": counters.peer_ready_waits,
        "completion_waits": counters.completion_waits,
        "generation_mismatches": counters.generation_mismatches,
        "slot_reuse_failures": counters.slot_reuse_failures,
        "max_active_slots": counters.max_active_slots,
    }
    for metric_name, ranks in rank_fields.items():
        for rank, value in enumerate(ranks):
            metrics[f"mok_like/runtime/rank_{rank}/{metric_name}"] = value
    for rank, acquisitions in enumerate(counters.slot_acquisitions):
        for slot, value in enumerate(acquisitions):
            metrics[f"mok_like/runtime/rank_{rank}/slot_{slot}/acquisitions"] = value
    for rank, (calls, bytes_by_phase) in enumerate(
        zip(counters.staging_copy_calls, counters.staging_copy_bytes, strict=True)
    ):
        for phase, call_count, byte_count in zip(_MOK_LIKE_STAGING_COPY_PHASES, calls, bytes_by_phase, strict=True):
            metrics[f"mok_like/runtime/rank_{rank}/{phase}/staging_copy_calls"] = call_count
            metrics[f"mok_like/runtime/rank_{rank}/{phase}/staging_copy_bytes"] = byte_count
    return metrics


def _memory_pool_trim_metrics(
    telemetry: MokLikeMemoryPoolTrimTelemetry,
    *,
    completed_update: int,
    trim_ordinal: int,
) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {
        "mok_like/runtime/default_pool_trim/completed_update": completed_update,
        "mok_like/runtime/default_pool_trim/trim_ordinal": trim_ordinal,
        "mok_like/runtime/default_pool_trim/active_reservations": telemetry.active_reservations,
        "mok_like/runtime/default_pool_trim/active_workspace_slots": telemetry.active_workspace_slots,
        "mok_like/runtime/default_pool_trim/wall_time_seconds": telemetry.wall_time_seconds,
    }
    for rank in telemetry.ranks:
        prefix = f"mok_like/runtime/default_pool_trim/rank_{rank.rank}"
        metrics[f"{prefix}/reserved_bytes_before"] = rank.reserved_bytes_before
        metrics[f"{prefix}/used_bytes_before"] = rank.used_bytes_before
        metrics[f"{prefix}/reserved_bytes_after"] = rank.reserved_bytes_after
        metrics[f"{prefix}/used_bytes_after"] = rank.used_bytes_after
        metrics[f"{prefix}/device_free_bytes_before"] = rank.device_free_bytes_before
        metrics[f"{prefix}/device_total_bytes_before"] = rank.device_total_bytes_before
        metrics[f"{prefix}/device_free_bytes_after"] = rank.device_free_bytes_after
        metrics[f"{prefix}/device_total_bytes_after"] = rank.device_total_bytes_after
        metrics[f"{prefix}/graph_reserved_bytes_after"] = rank.graph_reserved_bytes_after
        metrics[f"{prefix}/graph_used_bytes_after"] = rank.graph_used_bytes_after
        metrics[f"{prefix}/device_bytes_outside_default_pool_after"] = max(
            0,
            rank.device_total_bytes_after - rank.device_free_bytes_after - rank.reserved_bytes_after,
        )
        metrics[f"{prefix}/device_bytes_outside_default_and_graph_pools_after"] = max(
            0,
            rank.device_total_bytes_after
            - rank.device_free_bytes_after
            - rank.reserved_bytes_after
            - rank.graph_reserved_bytes_after,
        )
    return metrics


@dataclass
class _MokLikeHostTrimAudit:
    """Fixed-size host state accumulated across quiescent pool trims."""

    trim_count: int = 0
    active_reservation_anomalies: int = 0
    active_workspace_slot_anomalies: int = 0
    reserved_bytes_before: int = 0
    reserved_bytes_after: int = 0
    released_bytes: int = 0

    def record(self, telemetry: MokLikeMemoryPoolTrimTelemetry) -> None:
        self.trim_count += 1
        self.active_reservation_anomalies += int(telemetry.active_reservations != 0)
        self.active_workspace_slot_anomalies += int(telemetry.active_workspace_slots != 0)
        for rank in telemetry.ranks:
            self.reserved_bytes_before += rank.reserved_bytes_before
            self.reserved_bytes_after += rank.reserved_bytes_after
            self.released_bytes += max(0, rank.reserved_bytes_before - rank.reserved_bytes_after)


_MOK_LIKE_PROCESS_SUMMARY_FIELDS = 17


def _pack_mok_like_process_summary(values: tuple[int, ...]) -> np.ndarray:
    if len(values) != _MOK_LIKE_PROCESS_SUMMARY_FIELDS:
        raise ValueError(f"expected {_MOK_LIKE_PROCESS_SUMMARY_FIELDS} process-summary fields, got {len(values)}")
    unsigned = np.asarray(values, dtype=np.uint64)
    packed = np.empty((unsigned.size, 2), dtype=np.uint32)
    packed[:, 0] = unsigned.astype(np.uint32)
    packed[:, 1] = (unsigned >> np.uint64(32)).astype(np.uint32)
    return packed.reshape(-1)


def _unpack_mok_like_process_summaries(packed: np.ndarray) -> np.ndarray:
    words = np.asarray(packed, dtype=np.uint32).reshape(-1, _MOK_LIKE_PROCESS_SUMMARY_FIELDS, 2)
    return words[:, :, 0].astype(np.uint64) | (words[:, :, 1].astype(np.uint64) << np.uint64(32))


def _maybe_trim_default_memory_pools(
    config: GrugRunConfig,
    runtime: MokLikeRuntimeHandle | None,
    *,
    completed_update: int,
    train_step_result: tuple[GrugTrainState, dict[str, jax.Array], dict[str, jax.Array] | None],
) -> MokLikeMemoryPoolTrimTelemetry | None:
    """Run the configured cadence trim after a fully completed one-based update."""

    interval_updates = config.gpu_default_pool_trim_interval_updates
    if interval_updates is None or completed_update % interval_updates != 0:
        return None
    if runtime is None:
        raise RuntimeError("default GPU pool trimming requires an initialized mok_like runtime")

    jax.block_until_ready(train_step_result)
    telemetry = runtime.trim_default_memory_pools()
    trim_ordinal = completed_update // interval_updates
    trim_metrics = _memory_pool_trim_metrics(
        telemetry,
        completed_update=completed_update,
        trim_ordinal=trim_ordinal,
    )
    levanter.tracker.log(trim_metrics, step=completed_update - 1)
    for rank in telemetry.ranks:
        logger.info(
            "mok_like default-pool trim rank=%d reserved_before=%d used_before=%d reserved_after=%d used_after=%d "
            "device_free_before=%d device_total_before=%d device_free_after=%d device_total_after=%d "
            "graph_reserved_after=%d graph_used_after=%d",
            rank.rank,
            rank.reserved_bytes_before,
            rank.used_bytes_before,
            rank.reserved_bytes_after,
            rank.used_bytes_after,
            rank.device_free_bytes_before,
            rank.device_total_bytes_before,
            rank.device_free_bytes_after,
            rank.device_total_bytes_after,
            rank.graph_reserved_bytes_after,
            rank.graph_used_bytes_after,
        )
    logger.info(
        "mok_like default-pool trim completed_update=%d trim_ordinal=%d active_reservations=%d "
        "active_workspace_slots=%d wall_time_seconds=%.6f",
        completed_update,
        trim_ordinal,
        telemetry.active_reservations,
        telemetry.active_workspace_slots,
        telemetry.wall_time_seconds,
    )
    return telemetry


def _mok_like_process_metrics(
    counters: MokLikeDebugCounters,
    call_counts: tuple[int, int],
    trim_audit: _MokLikeHostTrimAudit,
    *,
    expected_handler_calls: int,
    expected_trim_count: int,
    forward_x_storage: MokLikeForwardXStorage,
    backward_peer_storage: MokLikeBackwardPeerStorage,
    num_tokens: int,
    hidden_dim: int,
    top_k: int,
    workspace_slots: int,
) -> dict[str, int]:
    """Gather the small process-level correctness contract after training quiesces."""

    forward_staging_calls = sum(rank[0] for rank in counters.staging_copy_calls)
    forward_staging_bytes = sum(rank[0] for rank in counters.staging_copy_bytes)
    backward_staging_calls = sum(rank[1] for rank in counters.staging_copy_calls)
    backward_staging_bytes = sum(rank[1] for rank in counters.staging_copy_bytes)
    expected_backward_staging_calls, expected_backward_staging_bytes = _expected_mok_like_backward_staging(
        backward_peer_storage,
        expected_handler_calls=expected_handler_calls,
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        top_k=top_k,
    )
    slot1_acquisitions = sum(rank[1] for rank in counters.slot_acquisitions)
    max_active_slots = max(counters.max_active_slots, default=0)
    local_summary = _pack_mok_like_process_summary(
        (
            jax.process_index(),
            call_counts[0],
            call_counts[1],
            sum(counters.generation_mismatches),
            sum(counters.slot_reuse_failures),
            forward_staging_calls,
            forward_staging_bytes,
            backward_staging_calls,
            backward_staging_bytes,
            slot1_acquisitions,
            max_active_slots,
            trim_audit.trim_count,
            trim_audit.active_reservation_anomalies,
            trim_audit.active_workspace_slot_anomalies,
            trim_audit.reserved_bytes_before,
            trim_audit.reserved_bytes_after,
            trim_audit.released_bytes,
        )
    )
    packed_summaries = np.asarray(multihost_utils.process_allgather(local_summary, tiled=False), dtype=np.uint32)
    gathered = _unpack_mok_like_process_summaries(packed_summaries)
    gathered = gathered.reshape(jax.process_count(), _MOK_LIKE_PROCESS_SUMMARY_FIELDS)
    expected_process_indices = np.arange(jax.process_count(), dtype=np.uint64)
    if not np.array_equal(gathered[:, 0], expected_process_indices):
        raise RuntimeError(
            f"mok_like runtime summaries have unexpected process indices {gathered[:, 0].tolist()}, "
            f"expected {expected_process_indices.tolist()}"
        )

    metrics: dict[str, int] = {
        "mok_like/runtime/process_count": jax.process_count(),
        "mok_like/runtime/expected_handler_calls_per_process": expected_handler_calls,
        "mok_like/runtime/processes_with_protocol_errors": int(np.count_nonzero(gathered[:, 3:5].sum(axis=1))),
        "mok_like/runtime/processes_with_forward_staging": int(
            np.count_nonzero((gathered[:, 5] != 0) | (gathered[:, 6] != 0))
        ),
        "mok_like/runtime/total_forward_staging_calls": int(gathered[:, 5].sum()),
        "mok_like/runtime/total_forward_staging_bytes": int(gathered[:, 6].sum()),
        "mok_like/runtime/processes_with_backward_staging": int(
            np.count_nonzero((gathered[:, 7] != 0) | (gathered[:, 8] != 0))
        ),
        "mok_like/runtime/expected_backward_staging_calls_per_process": expected_backward_staging_calls,
        "mok_like/runtime/expected_backward_staging_bytes_per_process": expected_backward_staging_bytes,
        "mok_like/runtime/total_backward_staging_calls": int(gathered[:, 7].sum()),
        "mok_like/runtime/total_backward_staging_bytes": int(gathered[:, 8].sum()),
        "mok_like/runtime/processes_using_slot1": int(np.count_nonzero(gathered[:, 9])),
        "mok_like/runtime/max_active_slots_across_processes": int(gathered[:, 10].max(initial=0)),
        "mok_like/runtime/expected_trim_count_per_process": expected_trim_count,
        "mok_like/runtime/expected_trim_count_across_processes": expected_trim_count * jax.process_count(),
        "mok_like/runtime/actual_trim_count_across_processes": int(gathered[:, 11].sum()),
        "mok_like/runtime/processes_with_trim_anomalies": int(np.count_nonzero(gathered[:, 12:14].sum(axis=1))),
        "mok_like/runtime/total_trim_reserved_bytes_before": int(gathered[:, 14].sum()),
        "mok_like/runtime/total_trim_reserved_bytes_after": int(gathered[:, 15].sum()),
        "mok_like/runtime/total_trimmed_bytes_across_processes": int(gathered[:, 16].sum()),
    }
    for summary in gathered:
        (
            process_index,
            forward_calls,
            backward_calls,
            generation_mismatches,
            slot_reuse_failures,
            process_forward_staging_calls,
            process_forward_staging_bytes,
            process_backward_staging_calls,
            process_backward_staging_bytes,
            process_slot1_acquisitions,
            process_max_active_slots,
            process_trim_count,
            active_reservation_anomalies,
            active_workspace_slot_anomalies,
            reserved_bytes_before,
            reserved_bytes_after,
            released_bytes,
        ) = summary
        prefix = f"mok_like/runtime/process_{process_index}"
        metrics[f"{prefix}/forward_calls"] = int(forward_calls)
        metrics[f"{prefix}/backward_calls"] = int(backward_calls)
        metrics[f"{prefix}/generation_mismatches"] = int(generation_mismatches)
        metrics[f"{prefix}/slot_reuse_failures"] = int(slot_reuse_failures)
        metrics[f"{prefix}/forward_staging_calls"] = int(process_forward_staging_calls)
        metrics[f"{prefix}/forward_staging_bytes"] = int(process_forward_staging_bytes)
        metrics[f"{prefix}/backward_staging_calls"] = int(process_backward_staging_calls)
        metrics[f"{prefix}/backward_staging_bytes"] = int(process_backward_staging_bytes)
        metrics[f"{prefix}/slot1_acquisitions"] = int(process_slot1_acquisitions)
        metrics[f"{prefix}/max_active_slots"] = int(process_max_active_slots)
        metrics[f"{prefix}/trim_count"] = int(process_trim_count)
        metrics[f"{prefix}/trim_active_reservation_anomalies"] = int(active_reservation_anomalies)
        metrics[f"{prefix}/trim_active_workspace_slot_anomalies"] = int(active_workspace_slot_anomalies)
        metrics[f"{prefix}/trim_reserved_bytes_before"] = int(reserved_bytes_before)
        metrics[f"{prefix}/trim_reserved_bytes_after"] = int(reserved_bytes_after)
        metrics[f"{prefix}/trimmed_bytes"] = int(released_bytes)

    bad_call_counts = gathered[(gathered[:, 1] != expected_handler_calls) | (gathered[:, 2] != expected_handler_calls)]
    protocol_errors = gathered[(gathered[:, 3] != 0) | (gathered[:, 4] != 0)]
    unexpected_forward_staging = (
        gathered[(gathered[:, 5] != 0) | (gathered[:, 6] != 0)]
        if forward_x_storage is MokLikeForwardXStorage.XLA_PEER_EXPERIMENTAL
        else np.empty((0, gathered.shape[1]), dtype=gathered.dtype)
    )
    bad_backward_staging = gathered[
        (gathered[:, 7] != expected_backward_staging_calls) | (gathered[:, 8] != expected_backward_staging_bytes)
    ]
    invalid_slot_usage = (
        gathered[(gathered[:, 9] != 0) | (gathered[:, 10] > 1)]
        if workspace_slots == 1
        else np.empty((0, gathered.shape[1]), dtype=gathered.dtype)
    )
    bad_trim_counts = gathered[gathered[:, 11] != expected_trim_count]
    trim_anomalies = gathered[(gathered[:, 12] != 0) | (gathered[:, 13] != 0)]
    if (
        bad_call_counts.size
        or protocol_errors.size
        or unexpected_forward_staging.size
        or bad_backward_staging.size
        or invalid_slot_usage.size
        or bad_trim_counts.size
        or trim_anomalies.size
    ):
        raise RuntimeError(
            "mok_like distributed runtime contract failed: "
            f"expected_handler_calls={expected_handler_calls}, expected_trim_count={expected_trim_count}, "
            f"forward_x_storage={forward_x_storage.value}, backward_peer_storage={backward_peer_storage.value}, "
            f"expected_backward_staging=({expected_backward_staging_calls}, {expected_backward_staging_bytes}), "
            f"workspace_slots={workspace_slots}, "
            f"summaries={gathered.tolist()}"
        )
    return metrics


def _expected_mok_like_backward_staging(
    storage: MokLikeBackwardPeerStorage,
    *,
    expected_handler_calls: int,
    num_tokens: int,
    hidden_dim: int,
    top_k: int,
) -> tuple[int, int]:
    activation_bytes = num_tokens * hidden_dim * _BF16_BYTES
    router_bytes = num_tokens * top_k * _FLOAT32_BYTES
    if storage is MokLikeBackwardPeerStorage.RUNTIME_STAGED:
        return expected_handler_calls * 4, expected_handler_calls * (2 * activation_bytes + 2 * router_bytes)
    if storage is MokLikeBackwardPeerStorage.XLA_PEER_INPUTS_EXPERIMENTAL:
        return expected_handler_calls, expected_handler_calls * router_bytes
    if storage is MokLikeBackwardPeerStorage.XLA_PEER_EXPERIMENTAL:
        return 0, 0
    raise AssertionError(f"unhandled backward peer storage {storage}")


def build_train_dataset(
    data_config: LmDataConfig,
    *,
    max_seq_len: int,
    batch_schedule: BatchSchedule,
    key: PRNGKeyArray,
) -> MixtureDataset[GrugLmExample]:
    pos = Axis("position", max_seq_len)
    mix_key, shuffle_key = jax.random.split(key)
    weights = data_config.train_weights
    if isinstance(weights, list):
        weights = rescale_mixture_schedule_for_batch_schedule(weights, batch_schedule)

    initial_batch_size = batch_schedule.batch_size_at_step(0)
    datasets = data_config.train_sets(pos, key=shuffle_key, initial_batch_size=initial_batch_size)
    return MixtureDataset(
        datasets=datasets,
        weights=weights,
        stop_strategy=data_config.stop_strategy,
        key=mix_key,
        block_size=data_config.mixture_block_size,
    )


_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")


def build_train_loader(
    dataset: AsyncDataset[GrugLmExample],
    *,
    batch_schedule: BatchSchedule,
    mesh: Mesh,
) -> DataLoader[GrugLmExample]:
    # DataLoader uses this batch axis mapping to shard batches across the distributed mesh.
    # `compact_grug_mesh` always carries (replica_dcn, data, expert, model); length-1 axes
    # are kept so we can name "expert" unconditionally.
    return DataLoader(
        dataset,
        batch_schedule.schedule,
        mesh=mesh,
        axis_resources={"__BATCH__": _BATCH_AXES},
        batch_axis_name="__BATCH__",
        allow_nondivisible_batch_size=False,
    )


def _reshard_tree_to_mesh(tree, mesh: Mesh):
    """Move each array leaf onto ``mesh``, preserving its PartitionSpec.

    The train and eval meshes name the same axes (only the ``expert``/``data`` sizes differ), so a
    leaf's PartitionSpec is valid on both; ``jax.device_put`` performs the cross-mesh transfer. The
    model's own ``reshard`` calls fix the exact layout inside the forward, so any valid placement on
    the target mesh suffices here. Non-array leaves pass through.
    """

    def move(leaf):
        if not isinstance(leaf, jax.Array):
            return leaf
        spec = leaf.sharding.spec if isinstance(leaf.sharding, NamedSharding) else P()
        return jax.device_put(leaf, NamedSharding(mesh, spec))

    return jax.tree.map(move, tree)


def _to_dropless_local(model: Transformer) -> Transformer:
    """Swap the scanned block's MoE expert backend to the dropless local ``sonic_cute`` path.

    ``implementation``/``expert_chunks`` are static fields shared across the whole stacked block,
    so one replacement covers every layer. The forward reads ``self.expert_mlp.implementation``
    (not the model config), so this alone routes the eval dropless. Must run on an expert-collapsed
    mesh: the local backend raises when the mesh expert axis is larger than one.
    """
    expert_mlp = model.stacked_blocks.stacked.mlp.expert_mlp
    dropless = dataclasses.replace(expert_mlp, implementation="sonic_cute", expert_chunks=1)
    return eqx.tree_at(lambda m: m.stacked_blocks.stacked.mlp.expert_mlp, model, dropless)


def build_tagged_evaluator(
    *,
    data_config: LmDataConfig,
    max_seq_len: int,
    mesh: Mesh,
    eval_cfg: GrugEvalConfig,
    mp: jmp.Policy,
    model_transform: Callable[[Transformer], Transformer] | None = None,
) -> TaggedEvaluator[LmExample | GrugLmExample, Transformer] | None:
    pos = Axis("position", max_seq_len)
    tagged_eval_sets = data_config.tagged_eval_sets(pos)
    if len(tagged_eval_sets) == 0:
        logger.warning("No evaluation datasets provided.")
        return None

    max_examples_per_dataset = None
    if eval_cfg.max_eval_batches is not None:
        max_examples_per_dataset = eval_cfg.max_eval_batches * eval_cfg.eval_batch_size

    tokenizer = data_config.the_tokenizer if eval_cfg.compute_bpb else None
    # `compact_grug_mesh` always carries (replica_dcn, data, expert, model); length-1 axes
    # are kept so we can name "expert" unconditionally.
    eval_axis_mapping = {"batch": _BATCH_AXES}
    eval_batch = Axis("batch", eval_cfg.eval_batch_size)
    eval_array_sharding = NamedSharding(mesh, P(_BATCH_AXES, None))

    def eval_loss_fn(model: Transformer, batch: LmExample | GrugLmExample) -> tuple[jax.Array, jax.Array, jax.Array]:
        # Evaluate at the compute dtype, as the train step does at `mp.cast_to_compute(params)`.
        # Parameters are stored float32, and `gpu_fa4_cute` accepts only bf16/fp16, so without this
        # every eval raises `TypeError: ... supports only bf16/fp16, got float32` on Blackwell. The
        # reference attention path takes float32, which hid this on H100.
        model = mp.cast_to_compute(model)
        if model_transform is not None:
            model = model_transform(model)
        if isinstance(batch, LmExample):
            batch = grug_lm_example_from_named(batch)
        per_pos_loss = model.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="none",
            logsumexp_weight=None,
        )
        per_pos_loss = jax.sharding.reshard(per_pos_loss, eval_array_sharding)
        per_pos_weight = jax.sharding.reshard(batch.loss_weight, eval_array_sharding)
        per_pos_token_id = jnp.pad(batch.tokens[:, 1:], ((0, 0), (0, 1)))
        return per_pos_loss, per_pos_weight, per_pos_token_id

    return TaggedEvaluator(
        EvalBatch=eval_batch,
        tagged_eval_sets=tagged_eval_sets,
        loss_fn=eval_loss_fn,
        tokenizer=tokenizer,
        device_mesh=mesh,
        axis_mapping=eval_axis_mapping,
        max_examples_per_dataset=max_examples_per_dataset,
    )


def _compute_flops(
    *,
    model_config: GrugModelConfig,
) -> tuple[float, dict[str, float]]:
    flops_per_token = lm_flops_per_token(
        hidden_dim=model_config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        shared_intermediate_dim=model_config.shared_expert_intermediate_dim,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        num_heads=model_config.num_heads,
        seq_len=model_config.max_seq_len,
        vocab_size=model_config.vocab_size,
        glu=True,
        num_experts=model_config.num_experts,
        num_shared_experts=model_config.num_shared_experts if model_config.shared_expert_intermediate_dim > 0 else 0,
        num_experts_per_tok=model_config.num_experts_per_token,
        sliding_window=model_config.sliding_window,
        global_every=model_config.global_every,
        local_kv_heads=model_config.local_kv_heads,
        global_kv_heads=model_config.global_kv_heads,
    )
    # `lm_flops_per_token` prices every matmul at `hidden_dim`. Under LatentMoE the routed experts
    # live at `latent_dim` instead, and two projections are added per layer, so correct both terms
    # or MFU is overstated by roughly the compression ratio.
    if model_config.latent_dim is not None:
        latent, hidden = model_config.latent_dim, model_config.hidden_dim
        # Matches the routed term in `lm_flops_per_token`: 2 * 3 * width * intermediate * top_k.
        routed_delta = 2 * 3 * model_config.intermediate_dim * model_config.num_experts_per_token * (latent - hidden)
        # W_down (hidden -> latent) and W_up (latent -> hidden), once per token each.
        projection = 2 * 2 * hidden * latent
        flops_per_token += model_config.num_layers * (routed_delta + projection)

    flops_per_example = 3 * flops_per_token * model_config.max_seq_len

    flops_summary: dict[str, float] = {
        "throughput/flops_per_token_analytic": flops_per_token,
        "throughput/flops_per_example_analytic": flops_per_example,
    }

    return flops_per_example, flops_summary


def _make_mixture_stage_callback(train_dataset: MixtureDataset, batch_schedule: BatchSchedule):
    last_mixture_stage = -1

    def log_mixture_stage(step_info):
        nonlocal last_mixture_stage
        seq_index = batch_schedule.global_data_offset_by_step(step_info.step)
        block_id = seq_index // train_dataset.block_size
        stage = train_dataset._get_stage_for_block(block_id)
        if stage == last_mixture_stage:
            return

        weights = train_dataset.weight_stages[stage][1]
        mixture_log = {f"mixture/weight/{name}": weight for name, weight in weights.items()}
        mixture_log["mixture/stage"] = stage
        levanter.tracker.log(mixture_log, step=step_info.step)
        last_mixture_stage = stage

    return log_mixture_stage


@register_dataclass
@dataclass(frozen=True)
class GrugTrainState:
    step: jax.Array
    params: Transformer
    opt_state: optax.OptState
    ema_params: Transformer | None
    pending_qb_betas: jax.Array


def _apply_qb_betas(model: Transformer, qb_betas: jax.Array) -> Transformer:
    """Set router biases from QB betas (computed on previous step)."""
    new_bias = -qb_betas
    new_bias = new_bias - jnp.mean(new_bias, axis=-1, keepdims=True)
    return eqx.tree_at(lambda t: t.stacked_blocks.stacked.mlp.router_bias, model, new_bias)


def _optimizer_state_to_memory_kind(tree, memory_kind: str):
    """Move named-sharded optimizer arrays to a JAX memory kind."""

    def _move(leaf):
        if not isinstance(leaf, jax.Array):
            return leaf
        sharding = jax.typeof(leaf).sharding
        mesh = getattr(sharding, "mesh", None)
        if mesh is None or len(getattr(mesh, "axis_names", ())) == 0:
            # Scalar optimizer metadata carries no named mesh and is negligible in HBM.
            return leaf
        return jax.device_put(leaf, sharding.with_memory_kind(memory_kind))

    return jax.tree.map(_move, tree)


def initial_state(
    model_config: GrugModelConfig,
    *,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    key: PRNGKeyArray,
    ema_beta: float | None,
    offload_opt_state: bool = False,
    mok_like_runtime: MokLikeRuntimeHandle | None = None,
) -> GrugTrainState:
    params = mp.cast_to_param(Transformer.init(model_config, key=key))
    if mok_like_runtime is not None:
        params = params.bind_mok_like_runtime(mok_like_runtime)
    num_moe_layers = model_config.num_layers
    opt_state = optimizer.init(params)
    if offload_opt_state:
        opt_state = _optimizer_state_to_memory_kind(opt_state, "pinned_host")
    return GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        opt_state=opt_state,
        ema_params=params if ema_beta is not None else None,
        pending_qb_betas=jnp.zeros((num_moe_layers, model_config.num_experts)),
    )


def _drop_metrics(
    dropped_assignments: jax.Array,
    *,
    batch_size: int,
    sequence_length: int,
    top_k: int,
    num_layers: int,
) -> dict[str, int | float]:
    # Global assignment totals can exceed int32; float32 would also round large drop counts.
    dropped_assignments_host = int(dropped_assignments)
    total_assignments = batch_size * sequence_length * top_k * num_layers
    return {
        "moe/dropped_assignments": dropped_assignments_host,
        "moe/drop_fraction": dropped_assignments_host / total_assignments,
    }


def _loss_and_grads(params, batch, mp: jmp.Policy, z_loss: float | None):
    def loss_fn(model):
        compute_params = mp.cast_to_compute(model)
        return compute_params.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="mean",
            logsumexp_weight=z_loss,
            return_router_metrics=True,
        )

    return jax.value_and_grad(loss_fn, has_aux=True)(params)


def _compute_diagnostic_watch_stats(params, batch, mp: jmp.Policy, z_loss: float | None, watch_config: WatchConfig):
    (_, _), grads = _loss_and_grads(params, batch, mp, z_loss)
    return compute_watch_stats(
        watch_targets=watch_config.watch_targets,
        include_norms=watch_config.include_norms,
        include_per_parameter_norms=watch_config.include_per_parameter_norms,
        include_histogram=watch_config.include_histograms,
        split_scan_layers=watch_config.split_scan_layers,
        params=params,
        grads=grads,
        model_tree_type=type(params),
    )


def _make_diagnostic_watch_step(mp: jmp.Policy, *, z_loss_weight: float, watch_config: WatchConfig):
    watch_targets = (
        tuple(t.strip() for t in watch_config.watch_targets.split(","))
        if isinstance(watch_config.watch_targets, str)
        else tuple(watch_config.watch_targets)
    )
    unsupported_targets = set(watch_targets) - {"grads", "params"}
    if unsupported_targets:
        raise ValueError(f"diagnostic watch does not support targets {sorted(unsupported_targets)}")
    diagnostic_watch_config = replace(watch_config, watch_targets=list(watch_targets))
    z_loss = z_loss_weight if z_loss_weight > 0 else None

    @jax.jit
    def diagnostic_watch_step(params: Transformer, batch, pending_qb_betas: jax.Array):
        params = _apply_qb_betas(params, pending_qb_betas)
        return _compute_diagnostic_watch_stats(params, batch, mp, z_loss, diagnostic_watch_config)

    return diagnostic_watch_step


def _make_train_step(
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    z_loss_weight: float,
    ema_beta: float | None,
    watch_config: WatchConfig | None = None,
    offload_opt_state: bool = False,
):
    one = jnp.array(1, dtype=jnp.int32)
    z_loss = z_loss_weight if z_loss_weight > 0 else None
    if watch_config is not None:
        if isinstance(watch_config.watch_targets, str):
            watch_targets = tuple(t.strip() for t in watch_config.watch_targets.split(","))
        else:
            watch_targets = tuple(watch_config.watch_targets)
    else:
        watch_targets = ()

    @functools.partial(jax.jit, donate_argnums=(0,))
    def train_step(state: GrugTrainState, batch):
        # Apply pending QB betas to router biases inside JIT (avoids eager
        # host-side TPU kernel launches that can cause SPMD sync issues).
        qb_params = _apply_qb_betas(state.params, state.pending_qb_betas)
        if ema_beta is not None:
            qb_ema_params = _apply_qb_betas(state.ema_params, state.pending_qb_betas)
        else:
            qb_ema_params = None

        (loss, summarized_metrics), grads = _loss_and_grads(qb_params, batch, mp, z_loss)
        metrics = {"train/loss": loss, **summarized_metrics}
        opt_state_in = (
            _optimizer_state_to_memory_kind(state.opt_state, "device") if offload_opt_state else state.opt_state
        )
        updates, opt_state = optimizer.update(grads, opt_state_in, qb_params)
        params = optax.apply_updates(qb_params, updates)

        if ema_beta is None:
            ema_params = None
        else:
            if qb_ema_params is None:
                raise ValueError("ema_params must be initialized when ema_beta is set.")
            ema_params = jax.tree_util.tree_map(
                lambda old, new: ema_beta * old + (1.0 - ema_beta) * new,
                qb_ema_params,
                params,
            )

        watch_stats = None
        if watch_config is not None:
            watch_stats = compute_watch_stats(
                watch_targets=watch_targets,
                include_norms=watch_config.include_norms,
                include_per_parameter_norms=watch_config.include_per_parameter_norms,
                include_histogram=watch_config.include_histograms,
                split_scan_layers=watch_config.split_scan_layers,
                params=qb_params,
                grads=grads,
                updates=updates,
                opt_state=opt_state_in,
                model_tree_type=type(state.params),
            )

        if offload_opt_state:
            opt_state = _optimizer_state_to_memory_kind(opt_state, "pinned_host")

        next_state = dataclasses.replace(
            state,
            step=state.step + one,
            params=params,
            opt_state=opt_state,
            ema_params=ema_params,
            pending_qb_betas=metrics["qb_beta_per_layer"],
        )

        return next_state, metrics, watch_stats

    return train_step


def _run_grug_local(config: GrugRunConfig) -> None:
    """Entry point for the grug template training loop."""
    trainer = config.trainer.trainer
    trainer.initialize()
    levanter.tracker.log_configuration(config)

    run_id = trainer.id
    if run_id is None:
        raise ValueError("trainer.id was not initialized")

    optimizer = config.optimizer.build(trainer.num_train_steps)
    watch_config = trainer.watch
    diagnostic_watch_step = None
    inline_watch_config = watch_config if watch_config.is_enabled else None
    if watch_config.is_enabled and config.trainer.watch_mode == WatchMode.DIAGNOSTIC:
        diagnostic_watch_step = _make_diagnostic_watch_step(
            trainer.mp,
            z_loss_weight=config.trainer.z_loss_weight,
            watch_config=watch_config,
        )
        inline_watch_config = None
    train_step = _make_train_step(
        optimizer,
        trainer.mp,
        z_loss_weight=config.trainer.z_loss_weight,
        ema_beta=config.trainer.ema_beta,
        watch_config=inline_watch_config,
        offload_opt_state=config.trainer.offload_opt_state,
    )

    data_key, model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)
    if config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(config.trainer.data_seed)

    # Grug uses raw PartitionSpecs rather than Trainer's logical axis mapping.
    # Keep the mesh compact so the batch pspec derived by `_batch_spec(mesh)` spans slices directly.
    # replica_axis_size=None lets compact_grug_mesh default to jax.process_count() (full
    # cross-slice replication); set it to 1 on GrugTrainerConfig for cross-slice FSDP.
    mesh = compact_grug_mesh(
        expert_axis_size=config.trainer.expert_axis_size,
        replica_axis_size=config.trainer.replica_axis_size,
    )
    with set_mesh(mesh), contextlib.ExitStack() as runtime_stack:
        batch_schedule = trainer.batch_schedule

        train_dataset = build_train_dataset(
            config.data,
            max_seq_len=config.model.max_seq_len,
            batch_schedule=batch_schedule,
            key=data_key,
        )
        train_loader = build_train_loader(
            train_dataset,
            batch_schedule=batch_schedule,
            mesh=mesh,
        )

        initial_batch_size = batch_schedule.batch_size_at_step(0)
        mok_like_runtime = _initialize_mok_like_for_config(
            config,
            mesh,
            batch_size=initial_batch_size,
        )
        if mok_like_runtime is not None:
            runtime_stack.callback(mok_like_runtime.close)

        @jax.jit
        def _init_state(model_rng):
            return initial_state(
                config.model,
                optimizer=optimizer,
                mp=trainer.mp,
                key=model_rng,
                ema_beta=config.trainer.ema_beta,
                offload_opt_state=config.trainer.offload_opt_state,
                mok_like_runtime=mok_like_runtime,
            )

        state = _init_state(model_key)

        checkpointer = trainer.checkpointer.create(run_id) if config.trainer.save_checkpoints else None
        state = restore_grug_state_from_checkpoint(
            state,
            checkpoint_search_paths=trainer.checkpoint_search_paths(run_id),
            load_checkpoint_setting=trainer.load_checkpoint,
            mesh=mesh,
            allow_partial=trainer.allow_partial_checkpoint,
        )
        dump_grug_state_sharding_run_artifact(
            state,
            log_dir=trainer.log_dir,
            run_id=run_id,
            path_override=config.trainer.sharding_dump_path,
        )

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.params)})

        flops_per_example, flops_summary = _compute_flops(model_config=config.model)
        levanter.tracker.log_summary(flops_summary)

        eval_cfg = config.eval
        evaluator = None
        dropless_evaluator = None
        dropless_eval_mesh = None
        if eval_cfg is not None:
            evaluator = build_tagged_evaluator(
                data_config=config.data,
                max_seq_len=config.model.max_seq_len,
                mesh=mesh,
                eval_cfg=eval_cfg,
                mp=trainer.mp,
            )
            # Expert-parallel runs drop tokens over capacity; a second evaluator scores the same
            # weights dropless under the local backend on an expert-collapsed mesh (expert folded
            # into `data`), which the local backend requires. FSDP runs already have expert=1.
            if eval_cfg.dropless_eval and mesh.shape["expert"] > 1:
                dropless_eval_mesh = compact_grug_mesh(
                    expert_axis_size=1,
                    replica_axis_size=mesh.shape["replica_dcn"],
                    model_axis_size=mesh.shape["model"],
                )
                # Build under the eval mesh so every constant the evaluator captures at construction
                # (e.g. `log2e`, the byte-per-token table, output shardings) is bound to the eval mesh
                # rather than the ambient train mesh; otherwise those leak a train-mesh aval into the
                # eval jit and fail the explicit-mesh check.
                with set_mesh(dropless_eval_mesh):
                    dropless_evaluator = build_tagged_evaluator(
                        data_config=config.data,
                        max_seq_len=config.model.max_seq_len,
                        mesh=dropless_eval_mesh,
                        eval_cfg=eval_cfg,
                        mp=trainer.mp,
                        model_transform=_to_dropless_local,
                    )

        # `trainer.num_train_steps` sizes the schedule; this bounds the run. Progress and the loop
        # both use it so a head-of-schedule run reports against the steps it will actually take.
        requested_stop_step = trainer.num_train_steps if config.stop_after_steps is None else config.stop_after_steps
        stop_step = min(requested_stop_step, trainer.num_train_steps)

        profiler_cfg = trainer.profiler
        profiler_num_steps = profiler_cfg.resolve_num_profile_steps(num_train_steps=stop_step)
        profiler_enabled = profiler_cfg.is_enabled and profiler_num_steps > 0

        log_every = max(1, config.trainer.log_every)
        iterator = LoadingTimeTrackerIterator(train_loader.iter_from_step(int(state.step)))

        state_callbacks = StateCallbackRunner[GrugTrainState](
            step_getter=lambda s: s.step,
            model_getter=lambda s: s.params,
            eval_model_getter=lambda s: s.ema_params if s.ema_params is not None else s.params,
            opt_state_getter=lambda s: s.opt_state,
        )
        state_callbacks.add_hook(
            callbacks.log_performance_stats(config.model.max_seq_len, batch_schedule, flops_per_example),
            every=log_every,
        )
        state_callbacks.add_hook(callbacks.pbar_logger(total=stop_step), every=log_every)
        state_callbacks.add_hook(callbacks.log_step_info(stop_step), every=log_every)
        if profiler_enabled:
            state_callbacks.add_hook(
                profiler_cfg.build(
                    str(trainer.log_dir / run_id / "profiler"),
                    run_id=run_id,
                    num_steps=profiler_num_steps,
                ),
                every=1,
            )
        state_callbacks.add_hook(_make_mixture_stage_callback(train_dataset, batch_schedule), every=1)
        if evaluator is not None and eval_cfg is not None:
            interval = eval_cfg.steps_per_eval
            eval_ema = eval_cfg.eval_ema and config.trainer.ema_beta is not None
            if interval is not None and interval > 0 and (eval_cfg.eval_current or eval_ema):
                state_callbacks.add_hook(
                    cb_tagged_evaluate(
                        evaluator,
                        prefix=eval_cfg.prefix,
                        eval_current=eval_cfg.eval_current,
                        eval_ema=eval_ema,
                    ),
                    every=interval,
                )
                if dropless_evaluator is not None and dropless_eval_mesh is not None:
                    # The training loop runs under `set_mesh(mesh)` (expert-parallel). The dropless
                    # evaluator runs under the expert-collapsed mesh, so the model params -- sharded on
                    # the train mesh -- must be resharded onto the eval mesh before its eval jit (JAX
                    # does not auto-reshard across explicit meshes), then the local backend sees
                    # expert=1. PGLE is disabled for the eval module as in `cb_tagged_evaluate`.
                    dropless_prefix = f"{eval_cfg.prefix}_dropless"

                    def dropless_eval_hook(
                        step, *args, _mesh=dropless_eval_mesh, _ev=dropless_evaluator, _prefix=dropless_prefix, **kwargs
                    ):
                        step_count = int(step.step)
                        if step_count < 0:
                            return
                        with set_mesh(_mesh):
                            model = _reshard_tree_to_mesh(step.model, _mesh)
                            with jax_config.enable_pgle(False):
                                log_dict = eval_model(_ev, model, prefix=_prefix)
                            levanter.tracker.log(log_dict, step=step_count)

                    state_callbacks.add_hook(dropless_eval_hook, every=interval)

        last_loss: float | jax.Array = 0.0
        last_step_duration = 0.0
        training_start_step = int(state.step)
        trim_audit = _MokLikeHostTrimAudit()
        if mok_like_runtime is not None:
            mok_like_runtime.reset_call_counts()
            mok_like_runtime.reset_debug_counters()

        # Main optimization loop.
        try:
            while int(state.step) < stop_step:
                with jax.profiler.TraceAnnotation("load_batch"):
                    batch = next(iterator)
                current_step = int(state.step)
                watch_due = (
                    watch_config.is_enabled and watch_config.interval > 0 and current_step % watch_config.interval == 0
                )
                if watch_due and diagnostic_watch_step is not None:
                    watch_stats = diagnostic_watch_step(state.params, batch, state.pending_qb_betas)
                    jax.block_until_ready(watch_stats)
                else:
                    watch_stats = None
                step_start = time.perf_counter()
                state, metrics, inline_watch_stats = train_step(state, batch)
                if inline_watch_stats is not None and watch_due:
                    watch_stats = inline_watch_stats
                step = int(state.step) - 1

                trim_telemetry = _maybe_trim_default_memory_pools(
                    config,
                    mok_like_runtime,
                    completed_update=step + 1,
                    train_step_result=train_step_result,
                )
                if trim_telemetry is not None:
                    trim_audit.record(trim_telemetry)

                jax.block_until_ready(metrics["train/loss"])

                if not jnp.isfinite(metrics["train/loss"]):
                    raise RuntimeError(f"Non-finite loss ({float(metrics['train/loss'])}) at step {int(state.step)}.")
                duration = time.perf_counter() - step_start
                hook_start = time.perf_counter()
                with jax.profiler.TraceAnnotation("callbacks"):
                    state_callbacks.run(state, loss=metrics["train/loss"], step_duration=duration)
                    last_loss = metrics["train/loss"]
                    last_step_duration = duration
                    levanter.tracker.log({"throughput/hook_time": time.perf_counter() - hook_start}, step=step)
                    levanter.tracker.log({"throughput/loading_time": iterator.this_load_time}, step=step)
                    router_metrics = {
                        key: value
                        for key, value in metrics.items()
                        if (key.startswith("train/router/") or key.startswith("moe_bias/"))
                        and key not in ("train/router/routing_counts_per_layer", "qb_beta_per_layer")
                    }
                    if router_metrics:
                        levanter.tracker.log(router_metrics, step=step)
                    if "train/cross_entropy_loss" in metrics:
                        levanter.tracker.log(
                            {"train/cross_entropy_loss": metrics["train/cross_entropy_loss"]},
                            step=step,
                        )
                    if "moe/dropped_assignments" in metrics:
                        drop_metrics = _drop_metrics(
                            metrics["moe/dropped_assignments"],
                            batch_size=batch.tokens.shape[0],
                            sequence_length=batch.tokens.shape[1],
                            top_k=config.model.num_experts_per_token,
                            num_layers=config.model.num_layers,
                        )
                        levanter.tracker.log(drop_metrics, step=step)

                    if watch_stats is not None:
                        levanter.tracker.log(watch_stats, step=step)

                if checkpointer is not None:
                    with callbacks.progress_event_scope(
                        state_callbacks.emit_event,
                        callbacks.ProgressEvent.CHECKPOINT_STARTED,
                        callbacks.ProgressEvent.CHECKPOINT_FINISHED,
                    ):
                        checkpointer.on_step(tree=state, step=int(state.step))

        except BaseException:
            logger.exception(
                "Fatal error in grug training loop; skipping final callbacks/checkpoint to preserve root cause"
            )
            raise
        else:
            # Mirror classic trainer behavior: force callbacks on the last completed step.
            state_callbacks.run(state, loss=last_loss, step_duration=last_step_duration, force=True)
            if mok_like_runtime is not None:
                counters = mok_like_runtime.debug_counters()
                debug_metrics = _mok_like_debug_metrics(counters)
                expected_handler_calls = (trainer.num_train_steps - training_start_step) * config.model.num_layers * 4
                trim_interval_updates = config.gpu_default_pool_trim_interval_updates
                expected_trim_count = (
                    trainer.num_train_steps // trim_interval_updates - training_start_step // trim_interval_updates
                    if trim_interval_updates is not None
                    else 0
                )
                process_metrics = _mok_like_process_metrics(
                    counters,
                    mok_like_runtime.call_counts(),
                    trim_audit,
                    expected_handler_calls=expected_handler_calls,
                    expected_trim_count=expected_trim_count,
                    forward_x_storage=config.model.mok_like.forward_x_storage,
                    backward_peer_storage=config.model.mok_like.backward_peer_storage,
                    num_tokens=_mok_like_tokens_per_rank(
                        batch_size=initial_batch_size,
                        sequence_length=config.model.max_seq_len,
                        mesh=mesh,
                    ),
                    hidden_dim=config.model.hidden_dim,
                    top_k=config.model.num_experts_per_token,
                    workspace_slots=config.model.mok_like.workspace_slots,
                )
                levanter.tracker.log_summary({**debug_metrics, **process_metrics})
                logger.info(
                    "mok_like runtime completed with %d recorded peer waits and %d handlers per phase/process",
                    debug_metrics["mok_like/runtime/total_peer_wait_events"],
                    expected_handler_calls,
                )
            if checkpointer is not None:
                with callbacks.progress_event_scope(
                    state_callbacks.emit_event,
                    callbacks.ProgressEvent.CHECKPOINT_STARTED,
                    callbacks.ProgressEvent.CHECKPOINT_FINISHED,
                ):
                    checkpointer.on_step(tree=state, step=int(state.step), force=True)
                    checkpointer.wait_until_finished()
        finally:
            state_callbacks.emit_event(callbacks.ProgressEvent.TRAINING_FINISHED)

    levanter.tracker.current_tracker().finish()


def run_grug(config: GrugRunConfig) -> None:
    """Dispatch grug training through Fray jobs."""
    trainer = config.trainer.trainer
    if trainer.id is None:
        raise ValueError("trainer.id must be set before dispatching grug training.")

    # Dispatch snapshots os.environ for the child task, so apply the hero defaults first.
    inline_watch_enabled = trainer.watch.is_enabled and config.trainer.watch_mode == WatchMode.INLINE
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=inline_watch_enabled, processes_per_task=config.processes_per_task
    )
    _apply_dispatch_environment(config)
    dispatch_grug_training_run(
        run_id=trainer.id,
        config=config,
        local_entrypoint=_run_grug_local,
        resources=config.resources,
        max_retries_failure=config.max_retries_failure,
        max_retries_preemption=config.max_retries_preemption,
        max_task_failures=config.max_task_failures,
        processes_per_task=config.processes_per_task,
        pip_packages=config.pip_packages,
    )


__all__ = [
    "GpuAllocator",
    "GpuDefaultPoolPreallocation",
    "GpuTempBufferPool",
    "GrugEvalConfig",
    "GrugRunConfig",
    "GrugTrainState",
    "GrugTrainerConfig",
    "XlaAutotuneCacheMode",
    "initial_state",
    "run_grug",
]
