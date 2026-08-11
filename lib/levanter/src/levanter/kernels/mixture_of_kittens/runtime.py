# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit lifecycle for the peer-visible workspaces used by mok_like."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import jax
import numpy as np

from levanter.kernels.mixture_of_kittens.availability import require_mok_like_available
from levanter.kernels.mixture_of_kittens.build import build_native_library
from levanter.kernels.mixture_of_kittens.ffi import BACKWARD_TARGET, FORWARD_TARGET
from levanter.kernels.mixture_of_kittens.source import MokLikeBuildConfig, mok_source_root

_INIT_SYMBOL = "levanter_mok_init_runtime"
_SHUTDOWN_SYMBOL = "levanter_mok_shutdown_runtime"
_LAST_ERROR_SYMBOL = "levanter_mok_last_error"
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


def load_native_library(build_config: MokLikeBuildConfig) -> tuple[ctypes.CDLL, ctypes.CDLL, Path]:
    """Build and load the adapter without initializing peer workspaces."""

    global_mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    cuda_driver = ctypes.CDLL("libcuda.so.1", mode=global_mode)
    library_path = build_native_library(build_config)
    library = ctypes.CDLL(str(library_path), mode=global_mode)
    return cuda_driver, library, library_path


def _native_last_error(library: ctypes.CDLL, default: str) -> str:
    function = getattr(library, _LAST_ERROR_SYMBOL)
    function.argtypes = []
    function.restype = ctypes.c_char_p
    message = function()
    return message.decode() if message else default


def register_ffi_targets(library: ctypes.CDLL) -> None:
    """Register the loaded forward/backward handlers with JAX."""

    for target in (FORWARD_TARGET, BACKWARD_TARGET):
        handler = getattr(library, target)
        handler.restype = ctypes.c_void_p
        jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(handler), platform="CUDA", api_version=1)
        jax.ffi.register_ffi_target_as_batch_partitionable(target)


def runtime_signature(
    num_tokens: int, hidden_dim: int, top_k: int, workspace_slots: int
) -> tuple[int, int, int, int, int]:
    return _NUM_DEVICES, num_tokens, hidden_dim, top_k, workspace_slots


def initialize_native_runtime(library: ctypes.CDLL, signature: tuple[int, int, int, int, int]) -> None:
    """Allocate the peer-visible workspaces owned by a runtime handle."""

    function = getattr(library, _INIT_SYMBOL)
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    function.restype = ctypes.c_int
    if function(*signature) != 0:
        raise RuntimeError(_native_last_error(library, "MoK-like runtime initialization failed"))


def shutdown_native_runtime(library: ctypes.CDLL) -> None:
    """Free the peer-visible workspaces owned by a runtime handle."""

    function = getattr(library, _SHUTDOWN_SYMBOL)
    function.argtypes = []
    function.restype = ctypes.c_int
    if function() != 0:
        raise RuntimeError(_native_last_error(library, "MoK-like runtime shutdown failed"))


@dataclass(eq=False)
class MokLikeRuntimeHandle:
    """Idempotent owner for one four-GPU native workspace allocation."""

    build_config: MokLikeBuildConfig
    signature: tuple[int, int, int, int, int]
    library_path: Path
    _cuda_driver: ctypes.CDLL = field(repr=False)
    _library: ctypes.CDLL = field(repr=False)
    _closed: bool = False

    def require_compatible(self, *, num_tokens: int, hidden_dim: int, top_k: int, workspace_slots: int) -> None:
        """Fail before lowering when this handle cannot serve the requested shape."""

        if self._closed:
            raise RuntimeError("mok_like runtime handle is closed")
        expected = runtime_signature(num_tokens, hidden_dim, top_k, workspace_slots)
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
        shutdown_native_runtime(self._library)
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
        """Synchronize all four devices and return native synchronization counters."""

        if self._closed:
            raise RuntimeError("mok_like runtime handle is closed")
        count_function = getattr(self._library, _DEBUG_COUNTER_COUNT_SYMBOL)
        count_function.argtypes = []
        count_function.restype = ctypes.c_int64
        count = int(count_function())
        expected = _NUM_DEVICES * _DEBUG_COUNTERS_PER_RANK
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
            values[rank * _DEBUG_COUNTERS_PER_RANK : (rank + 1) * _DEBUG_COUNTERS_PER_RANK]
            for rank in range(_NUM_DEVICES)
        )

        def phase_peer_values(offset: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
            return tuple(
                tuple(
                    rank[offset + phase * _NUM_DEVICES : offset + (phase + 1) * _NUM_DEVICES]
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
            peer_wait_cycles=phase_peer_values(23),
            peer_wait_max_cycles=phase_peer_values(39),
            staging_copy_calls=tuple((rank[55], rank[57]) for rank in ranks),
            staging_copy_bytes=tuple((rank[56], rank[58]) for rank in ranks),
        )

    def trim_default_memory_pools(self) -> MokLikeMemoryPoolTrimTelemetry:
        """Synchronize all local GPUs and trim their quiescent default CUDA pools once."""

        if self._closed:
            raise RuntimeError("mok_like runtime handle is closed")
        count = _NUM_DEVICES * _MEMORY_POOL_STATS_PER_RANK + _MEMORY_POOL_TRIM_TRAILER_SIZE
        output = (ctypes.c_uint64 * count)()
        function = getattr(self._library, _TRIM_DEFAULT_MEMORY_POOLS_SYMBOL)
        function.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64]
        function.restype = ctypes.c_int
        if function(output, count) != 0:
            raise RuntimeError(_native_last_error(self._library, "MoK-like default memory-pool trim failed"))
        values = tuple(int(value) for value in output)
        ranks = tuple(
            MokLikeMemoryPoolRankTelemetry(
                rank=rank,
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
            for rank in range(_NUM_DEVICES)
        )
        trailer = _NUM_DEVICES * _MEMORY_POOL_STATS_PER_RANK
        return MokLikeMemoryPoolTrimTelemetry(
            ranks=ranks,
            active_reservations=values[trailer],
            active_workspace_slots=values[trailer + 1],
            wall_time_seconds=values[trailer + 2] / 1e9,
        )

    def __enter__(self) -> MokLikeRuntimeHandle:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def validate_mok_like_expert_groups(
    devices: np.ndarray,
    axis_names: Sequence[str],
    *,
    world_devices: Sequence[jax.Device] | None = None,
) -> None:
    """Validate that each four-device expert group is contained in one process."""

    if _NUM_DEVICES not in devices.shape:
        raise ValueError(f"mok_like requires an expert axis of size {_NUM_DEVICES}, got shape={devices.shape}")
    try:
        expert_axis = tuple(axis_names).index("expert")
    except ValueError as error:
        raise ValueError("mok_like requires a named expert mesh axis") from error
    if devices.shape[expert_axis] != _NUM_DEVICES:
        raise ValueError(f"mok_like requires an expert axis of size {_NUM_DEVICES}, got shape={devices.shape}")

    groups = np.moveaxis(devices, expert_axis, -1).reshape(-1, _NUM_DEVICES)
    canonical_devices = tuple(jax.devices() if world_devices is None else world_devices)
    seen_processes: set[int] = set()
    for group_index, group in enumerate(groups):
        process_indices = tuple(int(device.process_index) for device in group)
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


def validate_mok_like_mesh_topology(mesh: jax.sharding.Mesh) -> None:
    """Validate a concrete global mesh before allocating process-local workspaces."""

    if not isinstance(mesh, jax.sharding.Mesh):
        raise TypeError("mok_like runtime initialization requires a concrete jax.sharding.Mesh")
    validate_mok_like_expert_groups(mesh.devices, mesh.axis_names)
    mesh_processes = {int(device.process_index) for device in mesh.devices.flat}
    expected_processes = set(range(jax.process_count()))
    if mesh_processes != expected_processes:
        raise ValueError(
            f"mok_like mesh process indices must match the JAX world; got {sorted(mesh_processes)}, "
            f"expected {sorted(expected_processes)}"
        )


def _validate_topology(mesh: jax.sharding.Mesh) -> None:
    devices = jax.local_devices()
    if len(devices) != 4:
        raise RuntimeError(f"mok_like requires exactly four visible local GPUs, found {len(devices)} devices")
    non_cuda = tuple(device.platform for device in devices if device.platform != "gpu")
    if non_cuda:
        raise RuntimeError(f"mok_like requires four CUDA devices, found platforms={non_cuda}")
    validate_mok_like_mesh_topology(mesh)


def initialize_mok_like_runtime(
    *,
    build_config: MokLikeBuildConfig,
    num_tokens: int,
    hidden_dim: int,
    top_k: int,
    workspace_slots: int = 2,
    mesh: jax.sharding.Mesh,
) -> MokLikeRuntimeHandle:
    """Build/register the adapter and allocate one caller-owned workspace."""

    global _ACTIVE_RUNTIME, _REGISTERED_LIBRARY_PATH
    if _ACTIVE_RUNTIME is not None:
        raise RuntimeError("a mok_like runtime is already active in this process")
    if num_tokens <= 0 or hidden_dim <= 0 or top_k <= 0:
        raise ValueError("num_tokens, hidden_dim, and top_k must be positive")
    if type(workspace_slots) is not int or not 1 <= workspace_slots <= 2:
        raise ValueError("workspace_slots must be an integer from 1 through 2")
    if num_tokens % 256 != 0 or hidden_dim % 256 != 0:
        raise ValueError("num_tokens and hidden_dim must be divisible by 256")

    _validate_topology(mesh)
    mok_source_root(build_config)
    require_mok_like_available(build_config)
    cuda_driver, library, library_path = load_native_library(build_config)
    if _REGISTERED_LIBRARY_PATH is None:
        register_ffi_targets(library)
        _REGISTERED_LIBRARY_PATH = library_path
    elif _REGISTERED_LIBRARY_PATH != library_path:
        raise RuntimeError(
            f"mok_like FFI targets already use {_REGISTERED_LIBRARY_PATH}; cannot register {library_path}"
        )

    signature = runtime_signature(num_tokens, hidden_dim, top_k, workspace_slots)
    initialize_native_runtime(library, signature)
    handle = MokLikeRuntimeHandle(
        build_config=build_config,
        signature=signature,
        library_path=library_path,
        _cuda_driver=cuda_driver,
        _library=library,
    )
    _ACTIVE_RUNTIME = handle
    return handle


def mok_like_runtime_initialized(handle: MokLikeRuntimeHandle | None = None) -> bool:
    """Return whether a live native runtime exists, optionally for one handle."""

    if handle is None:
        return _ACTIVE_RUNTIME is not None and not _ACTIVE_RUNTIME._closed
    return _ACTIVE_RUNTIME is handle and not handle._closed
