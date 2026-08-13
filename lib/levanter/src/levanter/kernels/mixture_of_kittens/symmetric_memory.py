# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""CUDA symmetric workspace ownership for the NVLink EP64 MoK runtime."""

from __future__ import annotations

import gc
import os
import socket
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import jax

MOK_LIKE_EP64_ARENA_SCHEMA_VERSION = 1
MOK_LIKE_EP64_ARENA_ALIGNMENT = 256
MOK_LIKE_EP64_SIZE = 64
MOK_LIKE_EP64_ARENA_OFFSET_FIELDS = (
    "x",
    "combine",
    "d_y",
    "d_x_routed",
    "router_weights",
    "d_router_weights",
    "generation",
    "forward_input_ready",
    "backward_input_ready",
    "forward_completions",
    "backward_completions",
    "last_forward_completion",
    "cancellation",
    "debug_counters",
)

_SIGNAL_PAD_BYTES = 1024 * 1024
_IMEX_DEVICE_ROOT = Path("/dev/nvidia-caps-imex-channels")


class MokLikeTopologyError(RuntimeError):
    """The JAX process/device ordering does not match the EP64 runtime group."""


class MokLikeSymmetricMemoryUnavailable(RuntimeError):
    """The CUDA symmetric-memory workspace could not be created or mapped."""


def _align_up(value: int) -> int:
    return (value + MOK_LIKE_EP64_ARENA_ALIGNMENT - 1) // MOK_LIKE_EP64_ARENA_ALIGNMENT * MOK_LIKE_EP64_ARENA_ALIGNMENT


@dataclass(frozen=True)
class MokLikeSymmetricArenaLayout:
    """Versioned, identically-sized allocation layout shared by every EP64 rank."""

    total_bytes: int
    offsets: Mapping[str, int]
    sizes: Mapping[str, int]
    schema_version: int = MOK_LIKE_EP64_ARENA_SCHEMA_VERSION

    @property
    def native_offset_table(self) -> tuple[int, ...]:
        """Return the stable native ABI table: schema, total bytes, then field offsets."""

        return (
            self.schema_version,
            self.total_bytes,
            *(self.offsets[field_name] for field_name in MOK_LIKE_EP64_ARENA_OFFSET_FIELDS),
        )


def mok_like_symmetric_arena_layout(
    *, num_tokens: int, hidden_dim: int, top_k: int, world_size: int = MOK_LIKE_EP64_SIZE
) -> MokLikeSymmetricArenaLayout:
    """Compute the canonical single-slot EP64 arena layout."""

    if num_tokens <= 0 or hidden_dim <= 0 or top_k <= 0:
        raise ValueError("num_tokens, hidden_dim, and top_k must be positive")
    if world_size != MOK_LIKE_EP64_SIZE:
        raise ValueError(f"symmetric MoK workspace requires world size {MOK_LIKE_EP64_SIZE}, got {world_size}")

    activation_bytes = num_tokens * hidden_dim * 2
    routed_activation_bytes = num_tokens * top_k * hidden_dim * 2
    router_bytes = num_tokens * top_k * 4
    peer_stamp_bytes = world_size * 8
    field_sizes = {
        "x": activation_bytes,
        "combine": routed_activation_bytes,
        "d_y": activation_bytes,
        "d_x_routed": routed_activation_bytes,
        "router_weights": router_bytes,
        "d_router_weights": router_bytes,
        "generation": peer_stamp_bytes,
        "forward_input_ready": peer_stamp_bytes,
        "backward_input_ready": peer_stamp_bytes,
        "forward_completions": peer_stamp_bytes,
        "backward_completions": peer_stamp_bytes,
        "last_forward_completion": peer_stamp_bytes,
        "cancellation": peer_stamp_bytes,
        # Seven leading scalars, three four-phase peer arrays, and four trailers.
        "debug_counters": (11 + 12 * world_size) * 8,
    }
    offsets: dict[str, int] = {}
    cursor = 0
    for field_name in MOK_LIKE_EP64_ARENA_OFFSET_FIELDS:
        cursor = _align_up(cursor)
        offsets[field_name] = cursor
        cursor += field_sizes[field_name]

    return MokLikeSymmetricArenaLayout(
        total_bytes=_align_up(cursor),
        offsets=MappingProxyType(offsets),
        sizes=MappingProxyType(field_sizes),
    )


@dataclass(frozen=True)
class MokLikeSymmetricNativeArguments:
    """Typed values passed to ``levanter_mok_init_runtime_ep64``."""

    rank: int
    world_size: int
    num_tokens: int
    hidden_dim: int
    top_k: int
    workspace_slots: int
    peer_arena_pointers: tuple[int, ...]
    arena_offsets: tuple[int, ...]


@dataclass(eq=False)
class MokLikeSymmetricWorkspace:
    """Collective owner of one PyTorch CUDA symmetric-memory arena."""

    rank: int
    world_size: int
    local_pointer: int
    peer_pointers: tuple[int, ...]
    layout: MokLikeSymmetricArenaLayout
    num_tokens: int
    hidden_dim: int
    top_k: int
    workspace_slots: int
    backend: str
    _torch: Any = field(repr=False)
    _distributed: Any = field(repr=False)
    _device: Any = field(repr=False)
    _group: Any = field(repr=False)
    _arena: Any = field(repr=False)
    _handle: Any = field(repr=False)
    _timeout: float = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _quiesced: bool = field(default=False, init=False, repr=False)

    @property
    def native_arguments(self) -> MokLikeSymmetricNativeArguments:
        """Return the complete, immutable native initialization record."""

        if self._closed:
            raise RuntimeError("MoK-like symmetric workspace is closed")
        return MokLikeSymmetricNativeArguments(
            rank=self.rank,
            world_size=self.world_size,
            num_tokens=self.num_tokens,
            hidden_dim=self.hidden_dim,
            top_k=self.top_k,
            workspace_slots=self.workspace_slots,
            peer_arena_pointers=self.peer_pointers,
            arena_offsets=self.layout.native_offset_table,
        )

    @property
    def is_closed(self) -> bool:
        return self._closed

    def _device_phase_barrier(self, channel: int) -> None:
        """Host-align ranks around one bounded device-side fabric barrier."""

        self._torch.cuda.synchronize(self._device)
        self._distributed.barrier(group=self._group)
        self._handle.barrier(channel=channel, timeout_ms=int(self._timeout * 1000))
        self._torch.cuda.synchronize(self._device)

    def quiesce(self) -> None:
        """Synchronize CUDA and rendezvous all ranks before native shutdown."""

        if self._closed:
            raise RuntimeError("MoK-like symmetric workspace is closed")
        if self._quiesced:
            return
        self._device_phase_barrier(channel=0)
        self._quiesced = True

    def gather_initialization_errors(self, error: BaseException | None) -> tuple[str | None, ...]:
        """Collect native initialization outcomes before any rank enters training."""

        if self._closed:
            raise RuntimeError("MoK-like symmetric workspace is closed")
        local_error = None if error is None else f"{type(error).__name__}: {error}"
        errors: list[str | None] = [None] * self.world_size
        self._distributed.all_gather_object(errors, local_error, group=self._group)
        return tuple(errors)

    def close(self) -> None:
        """Collectively release the workspace after all native calls are quiescent."""

        if self._closed:
            return
        self.quiesce()
        self._device_phase_barrier(channel=1)
        self._handle = None
        self._arena = None
        gc.collect()
        self._distributed.barrier(group=self._group)
        self._distributed.destroy_process_group(self._group)
        self._closed = True

    def __enter__(self) -> MokLikeSymmetricWorkspace:
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        self.close()


def _validated_rank() -> tuple[int, int, int]:
    from iris.hooks.multigpu import (  # noqa: PLC0415
        IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV,
        IRIS_MULTIGPU_PROCESS_COUNT_ENV,
        IRIS_MULTIGPU_PROCESS_INDEX_ENV,
    )

    if not jax.distributed.is_initialized():
        raise MokLikeTopologyError("JAX distributed initialization must complete before the EP64 workspace")
    required = (
        IRIS_MULTIGPU_PROCESS_INDEX_ENV,
        IRIS_MULTIGPU_PROCESS_COUNT_ENV,
        IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV,
    )
    missing = tuple(name for name in required if name not in os.environ)
    if missing:
        raise MokLikeTopologyError(f"EP64 requires the Iris one-GPU process supervisor; missing {missing}")
    try:
        rank = int(os.environ[IRIS_MULTIGPU_PROCESS_INDEX_ENV])
        world_size = int(os.environ[IRIS_MULTIGPU_PROCESS_COUNT_ENV])
        device_ids = tuple(int(value) for value in os.environ[IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV].split(","))
    except ValueError as error:
        raise MokLikeTopologyError("Iris supplied an invalid EP64 rank, world size, or device id") from error
    if not 0 <= rank < world_size:
        raise MokLikeTopologyError(f"invalid supervised EP64 rank {rank} for world size {world_size}")
    if world_size != MOK_LIKE_EP64_SIZE or jax.process_count() != world_size or jax.process_index() != rank:
        raise MokLikeTopologyError(
            f"EP64 rank/world mismatch: Iris={rank}/{world_size}, JAX={jax.process_index()}/{jax.process_count()}"
        )
    if len(device_ids) != 1:
        raise MokLikeTopologyError(f"EP64 requires exactly one supervised GPU per process, got {device_ids}")
    devices = jax.local_devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise MokLikeTopologyError(f"EP64 requires exactly one local JAX GPU device, got {devices}")
    hardware_id = getattr(devices[0], "local_hardware_id", None)
    if hardware_id is not None and int(hardware_id) != device_ids[0]:
        raise MokLikeTopologyError(
            f"JAX local hardware id {hardware_id} does not match supervised GPU {device_ids[0]}"
        )
    return rank, world_size, device_ids[0]


def _coordinator_endpoint(rank: int, timeout: float) -> tuple[str, str]:
    from iris.cluster.client.job_info import get_job_info  # noqa: PLC0415
    from levanter.utils.jax_utils import multihost_broadcast_sync  # noqa: PLC0415
    from rigging.network import interface_for_ipv4  # noqa: PLC0415

    job_info = get_job_info()
    if job_info is None:
        raise MokLikeTopologyError("EP64 symmetric memory requires Iris job metadata")
    endpoint: str | None = None
    if rank == 0:
        family = socket.AF_INET6 if ":" in job_info.advertise_host else socket.AF_INET
        with socket.socket(family) as listener:
            listener.bind(("", 0))
            port = listener.getsockname()[1]
        endpoint = (
            f"[{job_info.advertise_host}]:{port}" if family == socket.AF_INET6 else f"{job_info.advertise_host}:{port}"
        )
    resolved = multihost_broadcast_sync(endpoint, is_source=rank == 0, timeout=timeout)
    if not isinstance(resolved, str):
        raise MokLikeTopologyError(f"invalid Torch Store endpoint {resolved!r}")
    return resolved, interface_for_ipv4(job_info.advertise_host)


def initialize_mok_like_symmetric_workspace(
    *,
    num_tokens: int,
    hidden_dim: int,
    top_k: int,
    workspace_slots: int,
    timeout: float = 300.0,
) -> MokLikeSymmetricWorkspace:
    """Create one identical symmetric arena per EP64 rank and expose every peer alias.

    JAX distributed initialization must already be complete. All ranks must call
    this function once, in identical order and with identical shape arguments.
    """

    if workspace_slots != 1:
        raise ValueError("NVLink EP64 requires exactly one symmetric workspace slot")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    rank, world_size, device_id = _validated_rank()
    layout = mok_like_symmetric_arena_layout(
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        top_k=top_k,
        world_size=world_size,
    )
    imex_channels = tuple(_IMEX_DEVICE_ROOT.glob("channel*"))
    if not imex_channels:
        raise MokLikeSymmetricMemoryUnavailable(f"no IMEX channel devices are visible under {_IMEX_DEVICE_ROOT}")
    endpoint, gloo_interface = _coordinator_endpoint(rank, timeout)
    os.environ["TORCH_SYMMMEM_IMPLICIT_POOL"] = "0"
    os.environ["TORCH_SYMM_MEM_DISABLE_MULTICAST"] = "1"
    os.environ["LOCAL_RANK"] = str(device_id)
    os.environ["GLOO_SOCKET_IFNAME"] = gloo_interface

    try:
        import torch  # noqa: PLC0415
        import torch.distributed as dist  # noqa: PLC0415
        import torch.distributed._symmetric_memory as symm_mem  # noqa: PLC0415
    except ImportError as error:
        raise MokLikeSymmetricMemoryUnavailable("PyTorch 2.11 symmetric memory is unavailable") from error

    if not torch.cuda.is_available() or not dist.is_gloo_available():
        raise MokLikeSymmetricMemoryUnavailable("EP64 requires CUDA and the PyTorch Gloo backend")
    if dist.is_initialized():
        raise MokLikeSymmetricMemoryUnavailable("a PyTorch process group is already initialized")

    group = None
    arena = None
    handle = None
    try:
        torch.cuda.set_device(device_id)
        device = torch.device("cuda", device_id)
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{endpoint}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout),
        )
        group = dist.group.WORLD
        local_contract = (
            rank,
            world_size,
            num_tokens,
            hidden_dim,
            top_k,
            workspace_slots,
            layout.schema_version,
            layout.total_bytes,
        )
        contracts: list[tuple[int, ...] | None] = [None] * world_size
        dist.all_gather_object(contracts, local_contract, group=group)
        expected_contracts = tuple((peer, *local_contract[1:]) for peer in range(world_size))
        if tuple(contracts) != expected_contracts:
            raise MokLikeTopologyError(f"EP64 ranks disagree on symmetric workspace contract: {contracts}")
        symm_mem.set_signal_pad_size(_SIGNAL_PAD_BYTES)
        backend = str(symm_mem.get_backend(device))
        if backend.upper() != "CUDA":
            raise MokLikeSymmetricMemoryUnavailable(f"expected CUDA symmetric memory, got {backend!r}")
        arena = symm_mem.empty(layout.total_bytes, dtype=torch.uint8, device=device)
        handle = symm_mem.rendezvous(arena, group)
        if int(handle.rank) != rank or int(handle.world_size) != world_size:
            raise MokLikeSymmetricMemoryUnavailable(
                f"symmetric-memory identity is {handle.rank}/{handle.world_size}, expected {rank}/{world_size}"
            )
        peer_pointers = tuple(int(pointer) for pointer in handle.buffer_ptrs)
        if len(peer_pointers) != world_size or not all(peer_pointers):
            raise MokLikeSymmetricMemoryUnavailable(f"invalid peer pointer table of length {len(peer_pointers)}")
        local_pointer = int(arena.data_ptr())
        if peer_pointers[rank] != local_pointer:
            raise MokLikeSymmetricMemoryUnavailable(
                f"local arena pointer {local_pointer:#x} differs from rank-{rank} peer alias {peer_pointers[rank]:#x}"
            )
        arena.zero_()
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        handle.barrier(channel=0, timeout_ms=int(timeout * 1000))
        torch.cuda.synchronize(device)
        return MokLikeSymmetricWorkspace(
            rank=rank,
            world_size=world_size,
            local_pointer=local_pointer,
            peer_pointers=peer_pointers,
            layout=layout,
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            top_k=top_k,
            workspace_slots=workspace_slots,
            backend=backend,
            _torch=torch,
            _distributed=dist,
            _device=device,
            _group=group,
            _arena=arena,
            _handle=handle,
            _timeout=timeout,
        )
    except Exception as error:
        handle = None
        arena = None
        gc.collect()
        if group is not None and dist.is_initialized():
            dist.destroy_process_group(group)
        if isinstance(error, (MokLikeTopologyError, MokLikeSymmetricMemoryUnavailable)):
            raise
        raise MokLikeSymmetricMemoryUnavailable("failed to initialize the EP64 symmetric workspace") from error
