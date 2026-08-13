# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit lifecycle for the peer-visible workspaces used by mok_like."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import jax
import numpy as np

from levanter.kernels.mixture_of_kittens.availability import require_mok_like_available
from levanter.kernels.mixture_of_kittens.build import build_native_library
from levanter.kernels.mixture_of_kittens.config import MokLikeTopology
from levanter.kernels.mixture_of_kittens.ffi import (
    BACKWARD_TARGET,
    BACKWARD_TARGET_EP64,
    FAILURE_FENCE_TARGET,
    FORWARD_TARGET,
    FORWARD_TARGET_EP64,
)
from levanter.kernels.mixture_of_kittens.source import MokLikeBuildConfig, mok_source_root
from levanter.kernels.mixture_of_kittens.symmetric_memory import (
    MOK_LIKE_EP64_SIZE,
    MokLikeSymmetricWorkspace,
    initialize_mok_like_symmetric_workspace,
)

_INIT_SYMBOL = "levanter_mok_init_runtime"
_INIT_EP64_SYMBOL = "levanter_mok_init_runtime_ep64"
_SHUTDOWN_SYMBOL = "levanter_mok_shutdown_runtime"
_LAST_ERROR_SYMBOL = "levanter_mok_last_error"
_ARM_TEST_FAILURE_SYMBOL = "levanter_mok_arm_test_failure"
_RESET_CALL_COUNTS_SYMBOL = "levanter_mok_reset_call_counts"
_FORWARD_CALL_COUNT_SYMBOL = "levanter_mok_forward_call_count"
_BACKWARD_CALL_COUNT_SYMBOL = "levanter_mok_backward_call_count"
_DEBUG_COUNTER_COUNT_SYMBOL = "levanter_mok_debug_counter_count"
_RESET_DEBUG_COUNTERS_SYMBOL = "levanter_mok_reset_debug_counters"
_READ_DEBUG_COUNTERS_SYMBOL = "levanter_mok_read_debug_counters"
_TRIM_DEFAULT_MEMORY_POOLS_SYMBOL = "levanter_mok_trim_default_memory_pools"
_NUM_DEVICES = 4
_PEER_WAIT_PHASE_COUNT = 4
_DEBUG_COUNTERS_PER_RANK = 59
_MEMORY_POOL_STATS_PER_RANK = 10
_MEMORY_POOL_TRIM_TRAILER_SIZE = 3
_ACTIVE_RUNTIME: MokLikeRuntimeHandle | None = None
_REGISTERED_LIBRARY_PATH: Path | None = None


class MokLikeTestFailurePoint(IntEnum):
    # Values are part of the native failure-injection ABI.
    BEFORE_INPUT_READY = 0
    BEFORE_COMPLETION = 1


class MokLikeTestFailurePhase(IntEnum):
    # Values are part of the native failure-injection ABI.
    FORWARD = 0
    BACKWARD = 1


@dataclass(frozen=True)
class MokLikeDebugCounters:
    """Quiescent snapshot of native synchronization diagnostics, grouped by rank."""

    peer_ready_waits: tuple[int, ...]
    completion_waits: tuple[int, ...]
    generation_mismatches: tuple[int, ...]
    slot_reuse_failures: tuple[int, ...]
    slot_acquisitions: tuple[tuple[int, int], ...]
    max_active_slots: tuple[int, ...]
    peer_wait_events: tuple[tuple[tuple[int, ...], ...], ...]
    peer_wait_cycles: tuple[tuple[tuple[int, ...], ...], ...]
    peer_wait_max_cycles: tuple[tuple[tuple[int, ...], ...], ...]
    staging_copy_calls: tuple[tuple[int, int], ...]
    staging_copy_bytes: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class MokLikeMemoryPoolRankTelemetry:
    """One device's default CUDA memory-pool state around a trim."""

    rank: int
    reserved_bytes_before: int
    used_bytes_before: int
    reserved_bytes_after: int
    used_bytes_after: int
    device_free_bytes_before: int
    device_total_bytes_before: int
    device_free_bytes_after: int
    device_total_bytes_after: int
    graph_reserved_bytes_after: int
    graph_used_bytes_after: int


@dataclass(frozen=True)
class MokLikeMemoryPoolTrimTelemetry:
    """Quiescent native snapshot from one trim of all process-local default pools."""

    ranks: tuple[MokLikeMemoryPoolRankTelemetry, ...]
    active_reservations: int
    active_workspace_slots: int
    wall_time_seconds: float


def load_native_library(
    build_config: MokLikeBuildConfig, *, expert_parallel_size: int = _NUM_DEVICES
) -> tuple[ctypes.CDLL, ctypes.CDLL, Path]:
    """Build and load the adapter without initializing peer workspaces."""

    global_mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    cuda_driver = ctypes.CDLL("libcuda.so.1", mode=global_mode)
    library_path = build_native_library(build_config, expert_parallel_size=expert_parallel_size)
    library = ctypes.CDLL(str(library_path), mode=global_mode)
    return cuda_driver, library, library_path


def _native_last_error(library: ctypes.CDLL, default: str) -> str:
    function = getattr(library, _LAST_ERROR_SYMBOL)
    function.argtypes = []
    function.restype = ctypes.c_char_p
    message = function()
    return message.decode() if message else default


def register_ffi_targets(library: ctypes.CDLL, *, topology: MokLikeTopology = MokLikeTopology.LOCAL_EP4) -> None:
    """Register the loaded forward/backward handlers with JAX."""

    compute_targets = (
        (FORWARD_TARGET_EP64, BACKWARD_TARGET_EP64)
        if topology is MokLikeTopology.NVLINK_EP64
        else (FORWARD_TARGET, BACKWARD_TARGET)
    )
    for target in (*compute_targets, FAILURE_FENCE_TARGET):
        handler = getattr(library, target)
        handler.restype = ctypes.c_void_p
        jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(handler), platform="CUDA", api_version=1)
        jax.ffi.register_ffi_target_as_batch_partitionable(target)


def runtime_signature(
    num_tokens: int,
    hidden_dim: int,
    top_k: int,
    workspace_slots: int,
    *,
    topology: MokLikeTopology = MokLikeTopology.LOCAL_EP4,
) -> tuple[int, int, int, int, int]:
    return topology.expert_axis_size, num_tokens, hidden_dim, top_k, workspace_slots


def initialize_native_runtime(library: ctypes.CDLL, signature: tuple[int, int, int, int, int]) -> None:
    """Allocate the peer-visible workspaces owned by a runtime handle."""

    function = getattr(library, _INIT_SYMBOL)
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    function.restype = ctypes.c_int
    if function(*signature) != 0:
        raise RuntimeError(_native_last_error(library, "MoK-like runtime initialization failed"))


def initialize_native_runtime_ep64(library: ctypes.CDLL, workspace: MokLikeSymmetricWorkspace) -> None:
    """Initialize the one-rank EP64 native runtime from symmetric arena aliases."""

    arguments = workspace.native_arguments
    peer_pointers = (ctypes.c_uint64 * len(arguments.peer_arena_pointers))(*arguments.peer_arena_pointers)
    arena_offsets = (ctypes.c_uint64 * len(arguments.arena_offsets))(*arguments.arena_offsets)
    function = getattr(library, _INIT_EP64_SYMBOL)
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int64,
    ]
    function.restype = ctypes.c_int
    result = function(
        arguments.rank,
        arguments.world_size,
        arguments.num_tokens,
        arguments.hidden_dim,
        arguments.top_k,
        arguments.workspace_slots,
        peer_pointers,
        len(peer_pointers),
        arena_offsets,
        len(arena_offsets),
    )
    if result != 0:
        raise RuntimeError(_native_last_error(library, "MoK-like EP64 runtime initialization failed"))


def shutdown_native_runtime(library: ctypes.CDLL) -> None:
    """Free the peer-visible workspaces owned by a runtime handle."""

    function = getattr(library, _SHUTDOWN_SYMBOL)
    function.argtypes = []
    function.restype = ctypes.c_int
    if function() != 0:
        raise RuntimeError(_native_last_error(library, "MoK-like runtime shutdown failed"))


def _rollback_failed_ep64_initialization(
    *,
    library: ctypes.CDLL,
    workspace: MokLikeSymmetricWorkspace,
    native_error: Exception | None,
    initialization_errors: tuple[str | None, ...],
) -> None:
    """Collectively unwind a mixed-success EP64 initialization and raise its primary error."""

    primary_error = native_error or RuntimeError(
        f"MoK-like EP64 native initialization failed on peer ranks: {initialization_errors}"
    )
    rollback_failures: list[str] = []
    try:
        workspace.quiesce()
    except Exception as error:
        rollback_failures.append(f"workspace quiesce failed: {type(error).__name__}: {error}")
    if native_error is None:
        try:
            shutdown_native_runtime(library)
        except Exception as error:
            rollback_failures.append(f"native shutdown failed: {type(error).__name__}: {error}")

    local_rollback_error = "; ".join(rollback_failures) or None
    try:
        rollback_errors = workspace.gather_initialization_errors(
            RuntimeError(local_rollback_error) if local_rollback_error is not None else None
        )
    except Exception as error:
        primary_error.add_note(f"EP64 rollback agreement failed: {type(error).__name__}: {error}")
        raise primary_error
    if any(error is not None for error in rollback_errors):
        primary_error.add_note(f"EP64 initialization rollback was incomplete: {rollback_errors}")
        raise primary_error

    try:
        workspace.close()
    except Exception as error:
        primary_error.add_note(f"EP64 symmetric workspace release failed: {type(error).__name__}: {error}")
    raise primary_error


@dataclass(eq=False)
class MokLikeRuntimeHandle:
    """Idempotent owner for one local-EP4 or symmetric-EP64 native runtime."""

    build_config: MokLikeBuildConfig
    signature: tuple[int, int, int, int, int]
    library_path: Path
    _cuda_driver: ctypes.CDLL = field(repr=False)
    _library: ctypes.CDLL = field(repr=False)
    expert_parallel_size: int = _NUM_DEVICES
    topology: MokLikeTopology = MokLikeTopology.LOCAL_EP4
    _symmetric_workspace: MokLikeSymmetricWorkspace | None = field(default=None, repr=False)
    _closed: bool = False

    def require_compatible(self, *, num_tokens: int, hidden_dim: int, top_k: int, workspace_slots: int) -> None:
        """Fail before lowering when this handle cannot serve the requested shape."""

        if self._closed:
            raise RuntimeError("mok_like runtime handle is closed")
        expected = runtime_signature(num_tokens, hidden_dim, top_k, workspace_slots, topology=self.topology)
        if self.signature != expected:
            raise RuntimeError(f"mok_like runtime has signature {self.signature}, requested {expected}")
        if _ACTIVE_RUNTIME is not self:
            raise RuntimeError("mok_like runtime handle is not the active native workspace owner")

    def close(self) -> None:
        """Free the workspace once; subsequent calls have no effect."""

        global _ACTIVE_RUNTIME
        if self._closed:
            return
        if _ACTIVE_RUNTIME is not self:
            raise RuntimeError("mok_like runtime handle is not active")
        if self._symmetric_workspace is not None:
            self._symmetric_workspace.quiesce()
        shutdown_native_runtime(self._library)
        if self._symmetric_workspace is not None:
            self._symmetric_workspace.close()
        self._closed = True
        _ACTIVE_RUNTIME = None

    def reset_call_counts(self) -> None:
        """Reset host-side FFI invocation counters used by rematerialization gates."""

        if self._closed:
            raise RuntimeError("mok_like runtime handle is closed")
        function = getattr(self._library, _RESET_CALL_COUNTS_SYMBOL)
        function.argtypes = []
        function.restype = None
        function()

    def arm_test_failure(
        self,
        *,
        rank: int,
        phase: MokLikeTestFailurePhase,
        point: MokLikeTestFailurePoint,
        require_two_active_slots: bool = False,
    ) -> None:
        """Arm one handler failure for the next matching invocation."""
        if self._closed:
            raise RuntimeError("mok_like runtime handle is closed")
        function = getattr(self._library, _ARM_TEST_FAILURE_SYMBOL)
        function.argtypes = [ctypes.c_int] * 4
        function.restype = ctypes.c_int
        if function(rank, int(phase), int(point), int(require_two_active_slots)) != 0:
            raise RuntimeError(_native_last_error(self._library, "MoK-like test failure injection failed"))

    def call_counts(self) -> tuple[int, int]:
        """Return process-local forward and backward FFI handler invocations."""

        if self._closed:
            raise RuntimeError("mok_like runtime handle is closed")

        def read(symbol: str) -> int:
            function = getattr(self._library, symbol)
            function.argtypes = []
            function.restype = ctypes.c_int64
            return int(function())

        return read(_FORWARD_CALL_COUNT_SYMBOL), read(_BACKWARD_CALL_COUNT_SYMBOL)

    def reset_debug_counters(self) -> None:
        """Reset synchronization counters while no mok_like call is in flight."""

        if self._closed:
            raise RuntimeError("mok_like runtime handle is closed")
        function = getattr(self._library, _RESET_DEBUG_COUNTERS_SYMBOL)
        function.argtypes = []
        function.restype = ctypes.c_int
        if function() != 0:
            raise RuntimeError(_native_last_error(self._library, "MoK-like debug counter reset failed"))

    def debug_counters(self) -> MokLikeDebugCounters:
        """Synchronize local devices and return native synchronization counters."""

        if self._closed:
            raise RuntimeError("mok_like runtime handle is closed")
        count_function = getattr(self._library, _DEBUG_COUNTER_COUNT_SYMBOL)
        count_function.argtypes = []
        count_function.restype = ctypes.c_int64
        count = int(count_function())
        local_device_count = 1 if self.topology is MokLikeTopology.NVLINK_EP64 else _NUM_DEVICES
        counters_per_rank = (
            11 + 12 * self.expert_parallel_size
            if self.topology is MokLikeTopology.NVLINK_EP64
            else _DEBUG_COUNTERS_PER_RANK
        )
        expected = local_device_count * counters_per_rank
        if count != expected:
            raise RuntimeError(f"mok_like native debug counter ABI returned {count} values, expected {expected}")

        output = (ctypes.c_uint64 * count)()
        read_function = getattr(self._library, _READ_DEBUG_COUNTERS_SYMBOL)
        read_function.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64]
        read_function.restype = ctypes.c_int
        if read_function(output, count) != 0:
            raise RuntimeError(_native_last_error(self._library, "MoK-like debug counter read failed"))
        values = tuple(int(value) for value in output)
        ranks = tuple(
            values[rank * counters_per_rank : (rank + 1) * counters_per_rank] for rank in range(local_device_count)
        )

        def phase_peer_values(offset: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
            return tuple(
                tuple(
                    (
                        rank[offset + phase * _NUM_DEVICES : offset + (phase + 1) * _NUM_DEVICES]
                        if self.topology is MokLikeTopology.LOCAL_EP4
                        else rank[
                            offset
                            + phase * self.expert_parallel_size : offset
                            + (phase + 1) * self.expert_parallel_size
                        ]
                    )
                    for phase in range(_PEER_WAIT_PHASE_COUNT)
                )
                for rank in ranks
            )

        return MokLikeDebugCounters(
            peer_ready_waits=tuple(rank[0] for rank in ranks),
            completion_waits=tuple(rank[1] for rank in ranks),
            generation_mismatches=tuple(rank[2] for rank in ranks),
            slot_reuse_failures=tuple(rank[3] for rank in ranks),
            slot_acquisitions=tuple((rank[4], rank[5]) for rank in ranks),
            max_active_slots=tuple(rank[6] for rank in ranks),
            peer_wait_events=phase_peer_values(7),
            peer_wait_cycles=phase_peer_values(7 + 4 * self.expert_parallel_size),
            peer_wait_max_cycles=phase_peer_values(7 + 8 * self.expert_parallel_size),
            staging_copy_calls=tuple(
                (rank[7 + 12 * self.expert_parallel_size], rank[9 + 12 * self.expert_parallel_size]) for rank in ranks
            ),
            staging_copy_bytes=tuple(
                (rank[8 + 12 * self.expert_parallel_size], rank[10 + 12 * self.expert_parallel_size]) for rank in ranks
            ),
        )

    def trim_default_memory_pools(self) -> MokLikeMemoryPoolTrimTelemetry:
        """Synchronize all local GPUs and trim their quiescent default CUDA pools once."""

        if self._closed:
            raise RuntimeError("mok_like runtime handle is closed")
        local_device_count = 1 if self.topology is MokLikeTopology.NVLINK_EP64 else _NUM_DEVICES
        count = local_device_count * _MEMORY_POOL_STATS_PER_RANK + _MEMORY_POOL_TRIM_TRAILER_SIZE
        output = (ctypes.c_uint64 * count)()
        function = getattr(self._library, _TRIM_DEFAULT_MEMORY_POOLS_SYMBOL)
        function.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64]
        function.restype = ctypes.c_int
        if function(output, count) != 0:
            raise RuntimeError(_native_last_error(self._library, "MoK-like default memory-pool trim failed"))
        values = tuple(int(value) for value in output)
        ranks = tuple(
            MokLikeMemoryPoolRankTelemetry(
                rank=(self._symmetric_workspace.rank if self._symmetric_workspace is not None else rank),
                reserved_bytes_before=values[rank * _MEMORY_POOL_STATS_PER_RANK],
                used_bytes_before=values[rank * _MEMORY_POOL_STATS_PER_RANK + 1],
                reserved_bytes_after=values[rank * _MEMORY_POOL_STATS_PER_RANK + 2],
                used_bytes_after=values[rank * _MEMORY_POOL_STATS_PER_RANK + 3],
                device_free_bytes_before=values[rank * _MEMORY_POOL_STATS_PER_RANK + 4],
                device_total_bytes_before=values[rank * _MEMORY_POOL_STATS_PER_RANK + 5],
                device_free_bytes_after=values[rank * _MEMORY_POOL_STATS_PER_RANK + 6],
                device_total_bytes_after=values[rank * _MEMORY_POOL_STATS_PER_RANK + 7],
                graph_reserved_bytes_after=values[rank * _MEMORY_POOL_STATS_PER_RANK + 8],
                graph_used_bytes_after=values[rank * _MEMORY_POOL_STATS_PER_RANK + 9],
            )
            for rank in range(local_device_count)
        )
        trailer = local_device_count * _MEMORY_POOL_STATS_PER_RANK
        return MokLikeMemoryPoolTrimTelemetry(
            ranks=ranks,
            active_reservations=values[trailer],
            active_workspace_slots=values[trailer + 1],
            wall_time_seconds=values[trailer + 2] / 1e9,
        )

    def __enter__(self) -> MokLikeRuntimeHandle:
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        if isinstance(_exc_value, BaseException) and self.topology is MokLikeTopology.NVLINK_EP64:
            return
        try:
            self.close()
        except BaseException as close_error:
            if not isinstance(_exc_value, BaseException):
                raise
            _exc_value.add_note(f"MoK-like runtime close failed: {type(close_error).__name__}: {close_error}")


def validate_mok_like_expert_groups(
    devices: np.ndarray,
    axis_names: Sequence[str],
    *,
    world_devices: Sequence[jax.Device] | None = None,
    topology: MokLikeTopology = MokLikeTopology.LOCAL_EP4,
) -> None:
    """Validate expert rank order for the requested runtime topology."""

    expert_parallel_size = topology.expert_axis_size
    if expert_parallel_size not in devices.shape:
        raise ValueError(f"mok_like requires an expert axis of size {expert_parallel_size}, got shape={devices.shape}")
    try:
        expert_axis = tuple(axis_names).index("expert")
    except ValueError as error:
        raise ValueError("mok_like requires a named expert mesh axis") from error
    if devices.shape[expert_axis] != expert_parallel_size:
        raise ValueError(f"mok_like requires an expert axis of size {expert_parallel_size}, got shape={devices.shape}")

    groups = np.moveaxis(devices, expert_axis, -1).reshape(-1, expert_parallel_size)
    canonical_devices = tuple(jax.devices() if world_devices is None else world_devices)
    seen_processes: set[int] = set()
    for group_index, group in enumerate(groups):
        process_indices = tuple(int(device.process_index) for device in group)
        if topology is MokLikeTopology.NVLINK_EP64:
            expected_process_indices = tuple(range(MOK_LIKE_EP64_SIZE))
            if process_indices != expected_process_indices:
                raise ValueError(
                    f"mok_like EP64 expert group {group_index} must follow JAX process order; "
                    f"got process_indices={process_indices}"
                )
            platforms = tuple(device.platform for device in group)
            if any(platform != "gpu" for platform in platforms):
                raise ValueError(f"mok_like expert group {group_index} requires CUDA devices, got {platforms}")
            continue
        if len(set(process_indices)) != 1:
            raise ValueError(
                f"mok_like expert group {group_index} crosses JAX processes: process_indices={process_indices}"
            )
        process_index = process_indices[0]
        if process_index in seen_processes:
            raise ValueError(f"mok_like process {process_index} appears in more than one expert group")
        seen_processes.add(process_index)

        expected_group = tuple(device for device in canonical_devices if int(device.process_index) == process_index)
        if tuple(group) != expected_group:
            raise ValueError(
                f"mok_like expert group {group_index} must match JAX's process-local device order; "
                f"got {[int(device.id) for device in group]}, "
                f"expected {[int(device.id) for device in expected_group]}"
            )
        platforms = tuple(device.platform for device in group)
        if any(platform != "gpu" for platform in platforms):
            raise ValueError(f"mok_like expert group {group_index} requires CUDA devices, got platforms={platforms}")


def validate_mok_like_mesh_topology(
    mesh: jax.sharding.Mesh, *, topology: MokLikeTopology = MokLikeTopology.LOCAL_EP4
) -> None:
    """Validate a concrete global mesh before allocating process-local workspaces."""

    if not isinstance(mesh, jax.sharding.Mesh):
        raise TypeError("mok_like runtime initialization requires a concrete jax.sharding.Mesh")
    validate_mok_like_expert_groups(mesh.devices, mesh.axis_names, topology=topology)
    mesh_processes = {int(device.process_index) for device in mesh.devices.flat}
    expected_processes = set(range(jax.process_count()))
    if mesh_processes != expected_processes:
        raise ValueError(
            f"mok_like mesh process indices must match the JAX world; got {sorted(mesh_processes)}, "
            f"expected {sorted(expected_processes)}"
        )


def _validate_topology(mesh: jax.sharding.Mesh, topology: MokLikeTopology) -> None:
    devices = jax.local_devices()
    expected_local_devices = 1 if topology is MokLikeTopology.NVLINK_EP64 else _NUM_DEVICES
    if len(devices) != expected_local_devices:
        raise RuntimeError(
            f"mok_like {topology.value} requires exactly {expected_local_devices} visible local GPU(s), "
            f"found {len(devices)}"
        )
    non_cuda = tuple(device.platform for device in devices if device.platform != "gpu")
    if non_cuda:
        raise RuntimeError(f"mok_like requires CUDA devices, found platforms={non_cuda}")
    if topology is MokLikeTopology.LOCAL_EP4:
        validate_mok_like_mesh_topology(mesh)
    else:
        validate_mok_like_mesh_topology(mesh, topology=topology)


def initialize_mok_like_runtime(
    *,
    build_config: MokLikeBuildConfig,
    num_tokens: int,
    hidden_dim: int,
    top_k: int,
    workspace_slots: int = 2,
    mesh: jax.sharding.Mesh,
    topology: MokLikeTopology = MokLikeTopology.LOCAL_EP4,
) -> MokLikeRuntimeHandle:
    """Build/register the adapter and allocate one caller-owned workspace."""

    global _ACTIVE_RUNTIME, _REGISTERED_LIBRARY_PATH
    if _ACTIVE_RUNTIME is not None:
        raise RuntimeError("a mok_like runtime is already active in this process")
    if num_tokens <= 0 or hidden_dim <= 0 or top_k <= 0:
        raise ValueError("num_tokens, hidden_dim, and top_k must be positive")
    if not isinstance(topology, MokLikeTopology):
        raise TypeError("topology must be a MokLikeTopology")
    if type(workspace_slots) is not int or not 1 <= workspace_slots <= 2:
        raise ValueError("workspace_slots must be an integer from 1 through 2")
    if topology is MokLikeTopology.NVLINK_EP64 and workspace_slots != 1:
        raise ValueError("NVLink EP64 requires exactly one workspace slot")
    if num_tokens % 256 != 0 or hidden_dim % 256 != 0:
        raise ValueError("num_tokens and hidden_dim must be divisible by 256")

    _validate_topology(mesh, topology)
    mok_source_root(build_config)
    require_mok_like_available(build_config)
    if topology is MokLikeTopology.LOCAL_EP4:
        cuda_driver, library, library_path = load_native_library(build_config)
    else:
        cuda_driver, library, library_path = load_native_library(
            build_config, expert_parallel_size=topology.expert_axis_size
        )
    if _REGISTERED_LIBRARY_PATH is None:
        if topology is MokLikeTopology.LOCAL_EP4:
            register_ffi_targets(library)
        else:
            register_ffi_targets(library, topology=topology)
        _REGISTERED_LIBRARY_PATH = library_path
    elif _REGISTERED_LIBRARY_PATH != library_path:
        raise RuntimeError(
            f"mok_like FFI targets already use {_REGISTERED_LIBRARY_PATH}; cannot register {library_path}"
        )

    signature = runtime_signature(num_tokens, hidden_dim, top_k, workspace_slots, topology=topology)
    symmetric_workspace = None
    if topology is MokLikeTopology.NVLINK_EP64:
        symmetric_workspace = initialize_mok_like_symmetric_workspace(
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            top_k=top_k,
            workspace_slots=workspace_slots,
        )
        native_error: Exception | None = None
        try:
            initialize_native_runtime_ep64(library, symmetric_workspace)
        except Exception as error:
            native_error = error
        initialization_errors = symmetric_workspace.gather_initialization_errors(native_error)
        if any(error is not None for error in initialization_errors):
            _rollback_failed_ep64_initialization(
                library=library,
                workspace=symmetric_workspace,
                native_error=native_error,
                initialization_errors=initialization_errors,
            )
    else:
        initialize_native_runtime(library, signature)
    handle = MokLikeRuntimeHandle(
        build_config=build_config,
        signature=signature,
        library_path=library_path,
        _cuda_driver=cuda_driver,
        _library=library,
        expert_parallel_size=topology.expert_axis_size,
        topology=topology,
        _symmetric_workspace=symmetric_workspace,
    )
    _ACTIVE_RUNTIME = handle
    return handle


def mok_like_runtime_initialized(handle: MokLikeRuntimeHandle | None = None) -> bool:
    """Return whether a live native runtime exists, optionally for one handle."""

    if handle is None:
        return _ACTIVE_RUNTIME is not None and not _ACTIVE_RUNTIME._closed
    return _ACTIVE_RUNTIME is handle and not handle._closed
