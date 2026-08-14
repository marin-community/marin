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
from levanter.kernels.mixture_of_kittens.config import EXPERT_AXIS as _EXPERT_AXIS
from levanter.kernels.mixture_of_kittens.config import _DEVICES_PER_NODE, MokLikeWorkspaceTransport
from levanter.kernels.mixture_of_kittens.ffi import FAILURE_FENCE_TARGET, backward_target, forward_target
from levanter.kernels.mixture_of_kittens.source import MokLikeBuildConfig, mok_source_root

_INIT_SYMBOL = "levanter_mok_init_runtime"
_INIT_LOCAL_ARENA_SYMBOL = "levanter_mok_init_local_arena"
_IMPORT_ARENA_PEERS_SYMBOL = "levanter_mok_import_arena_peers"
_FABRIC_HANDLE_BYTES_SYMBOL = "levanter_mok_fabric_handle_bytes"
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


def register_ffi_targets(library: ctypes.CDLL, num_devices: int = _NUM_DEVICES) -> None:
    """Register the loaded forward/backward handlers with JAX.

    ``num_devices`` must match the rank count the library was compiled for; the
    handler symbols are rank-suffixed, so a mismatch raises here rather than
    resolving to the wrong object.
    """

    targets = (forward_target(num_devices), backward_target(num_devices), FAILURE_FENCE_TARGET)
    for target in targets:
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


def fabric_handle_bytes(library: ctypes.CDLL) -> int:
    """Size of one exported fabric handle, read from the native adapter."""

    function = getattr(library, _FABRIC_HANDLE_BYTES_SYMBOL)
    function.argtypes = []
    function.restype = ctypes.c_int
    return int(function())


def initialize_local_arena(
    library: ctypes.CDLL,
    *,
    rank: int,
    num_devices: int,
    num_tokens: int,
    hidden_dim: int,
    top_k: int,
    workspace_slots: int,
    device_ordinal: int,
) -> np.ndarray:
    """Allocate this process's symmetric arena and return its exported handles.

    Returns a ``[workspace_slots, handle_bytes]`` uint8 array. The caller must
    gather these across the expert axis and pass the result to
    :func:`import_arena_peers`; until then the runtime has no peer mappings and
    is not usable.
    """

    handle_bytes = fabric_handle_bytes(library)
    out = np.zeros((workspace_slots, handle_bytes), dtype=np.uint8)
    function = getattr(library, _INIT_LOCAL_ARENA_SYMBOL)
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ubyte),
    ]
    function.restype = ctypes.c_int
    buffer = out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    status = function(
        rank, num_devices, num_tokens, hidden_dim, top_k, workspace_slots, device_ordinal, buffer
    )
    if status != 0:
        raise RuntimeError(_native_last_error(library, "MoK-like local arena allocation failed"))
    return out


def import_arena_peers(library: ctypes.CDLL, handles: np.ndarray) -> None:
    """Import every rank's exported arena and bind this rank's runtimes.

    ``handles`` must be ``[workspace_slots, num_devices, handle_bytes]`` uint8 and
    C-contiguous: the native side reads it slot-major, then by rank.
    """

    if handles.dtype != np.uint8 or handles.ndim != 3:
        raise ValueError(f"handles must be a rank-three uint8 array, got {handles.dtype} {handles.shape}")
    contiguous = np.ascontiguousarray(handles)
    function = getattr(library, _IMPORT_ARENA_PEERS_SYMBOL)
    function.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
    function.restype = ctypes.c_int
    buffer = contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    if function(buffer) != 0:
        raise RuntimeError(_native_last_error(library, "MoK-like peer arena import failed"))


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

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
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


def _validate_topology(
    mesh: jax.sharding.Mesh,
    num_devices: int = _NUM_DEVICES,
    workspace_transport: MokLikeWorkspaceTransport = MokLikeWorkspaceTransport.IN_PROCESS_PEER,
) -> None:
    """Check local devices against the expert-group size the adapter was built for.

    The expected local device count follows the transport, not the group size. Under the
    process-local peer transport the whole group is reached through one process's peer-access
    table, so the process has to own every rank in it. Under the fabric transport each process
    allocates and exports one rank's arena, so a group of ``num_devices`` ranks is assembled from
    that many processes and each one sees a single device.
    """

    devices = jax.local_devices()
    if workspace_transport.crosses_processes:
        if len(devices) != 1:
            raise RuntimeError(
                f"mok_like under {workspace_transport.value} places one rank per process, so each "
                f"process must see exactly 1 GPU, found {len(devices)} devices"
            )
    else:
        expected_local = min(num_devices, _DEVICES_PER_NODE)
        if len(devices) != expected_local:
            raise RuntimeError(
                f"mok_like at {num_devices} ranks requires exactly {expected_local} "
                f"visible local GPUs, found {len(devices)} devices"
            )
    non_cuda = tuple(device.platform for device in devices if device.platform != "gpu")
    if non_cuda:
        raise RuntimeError(f"mok_like requires CUDA devices, found platforms={non_cuda}")
    if num_devices <= _DEVICES_PER_NODE and not workspace_transport.crosses_processes:
        validate_mok_like_mesh_topology(mesh)


def _expert_axis_index(mesh: jax.sharding.Mesh) -> int:
    """This process's coordinate along the mesh's expert axis.

    The fabric transport places one rank per process, so the process owns exactly one device and
    that device has a single position on the expert axis.
    """
    if _EXPERT_AXIS not in mesh.axis_names:
        raise RuntimeError(f"mesh has no {_EXPERT_AXIS!r} axis; got {mesh.axis_names}")
    local_device = jax.local_devices()[0]
    positions = np.argwhere(np.asarray(mesh.devices, dtype=object) == local_device)
    if positions.shape[0] != 1:
        raise RuntimeError(
            f"expected exactly one mesh position for {local_device}, found {positions.shape[0]}"
        )
    return int(positions[0][mesh.axis_names.index(_EXPERT_AXIS)])


def _initialize_fabric_arena(
    library: ctypes.CDLL,
    *,
    mesh: jax.sharding.Mesh,
    num_devices: int,
    num_tokens: int,
    hidden_dim: int,
    top_k: int,
    workspace_slots: int,
) -> None:
    """Run the two-phase fabric rendezvous across the expert group.

    Each process allocates its own arena, the exported handles are gathered across
    processes, and every process imports the full set. The gather is the only step
    that needs a collective, and it moves 64 bytes per rank per slot.

    This uses a whole-job gather, so it is correct when the expert group spans
    every process in the job -- one rack at EP64 with one process per GPU. A job
    that also replicates across racks needs a per-group gather instead; that case
    is rejected below rather than silently mixing handles from different groups.
    """

    from jax.experimental import multihost_utils

    process_count = jax.process_count()
    if process_count != num_devices:
        raise RuntimeError(
            f"fabric transport currently requires one process per expert-group rank with the "
            f"group spanning the whole job: num_devices={num_devices} but process_count={process_count}"
        )

    # A rank here must mean the same thing it means to the kernel: a position on the mesh's expert
    # axis, which is what the dispatch schedule's peer_rank indexes. The arena table and the handle
    # gather are both ordered by process index, so the two have to agree. They are not guaranteed
    # to -- the mesh is built over jax.devices() in its own order -- and a silent disagreement
    # hands the kernel a peer pointer belonging to a different rank than the schedule intends.
    rank = _expert_axis_index(mesh)
    if rank != jax.process_index():
        raise RuntimeError(
            f"fabric transport requires this process's expert-axis position ({rank}) to match its "
            f"process index ({jax.process_index()}); the handle gather is ordered by process index, "
            f"so a mismatch binds each rank's peers to the wrong arenas"
        )
    # The CUDA runtime enumerates every GPU on the node regardless of this process's JAX slice,
    # so the arena has to be told which ordinal it owns rather than inferring it from a count.
    local_device = jax.local_devices()[0]
    local_handles = initialize_local_arena(
        library,
        rank=rank,
        num_devices=num_devices,
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        top_k=top_k,
        workspace_slots=workspace_slots,
        device_ordinal=local_device.local_hardware_id,
    )

    # process_allgather stacks along a new leading axis ordered by process index,
    # giving [num_devices, workspace_slots, handle_bytes]. The native side reads
    # slot-major, so swap the first two axes before handing the buffer over.
    gathered = np.asarray(multihost_utils.process_allgather(local_handles, tiled=False))
    expected = (num_devices, workspace_slots, local_handles.shape[-1])
    if gathered.shape != expected:
        raise RuntimeError(f"gathered fabric handles have shape {gathered.shape}, expected {expected}")
    slot_major = np.ascontiguousarray(np.swapaxes(gathered, 0, 1))
    import_arena_peers(library, slot_major)


def initialize_mok_like_runtime(
    *,
    build_config: MokLikeBuildConfig,
    num_tokens: int,
    hidden_dim: int,
    top_k: int,
    workspace_slots: int = 2,
    mesh: jax.sharding.Mesh,
    workspace_transport: MokLikeWorkspaceTransport = MokLikeWorkspaceTransport.IN_PROCESS_PEER,
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
    _validate_topology(mesh, build_config.num_devices, workspace_transport)
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
    if workspace_transport is MokLikeWorkspaceTransport.FABRIC_SYMMETRIC:
        _initialize_fabric_arena(
            library,
            mesh=mesh,
            num_devices=build_config.num_devices,
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            top_k=top_k,
            workspace_slots=workspace_slots,
        )
    else:
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
