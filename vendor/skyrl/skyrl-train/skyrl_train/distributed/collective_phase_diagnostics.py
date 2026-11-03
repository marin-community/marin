"""Low-overhead process-group sequence snapshots at policy phase boundaries."""

from __future__ import annotations

import itertools
import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Protocol, TextIO
from pathlib import Path

import torch.distributed as dist
from loguru import logger
from skyrl_train.env_vars import (
    COLLECTIVE_PHASE_DIAGNOSTICS_ENV,
    DEBUG_ARTIFACT_DIR_ENV,
    ensure_debug_artifact_directories,
)

LOG_PREFIX = "COLLECTIVE_PHASE_DIAGNOSTICS "
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
WORLD_GROUP = "world"
_region_ids = itertools.count(1)
_region_ids_lock = threading.Lock()
_artifact_lock = threading.Lock()
_artifact_files: dict[Path, TextIO] = {}


class ProcessGroupLike(Protocol):
    def _get_sequence_number_for_group(self) -> int: ...


class DeviceMeshLike(Protocol):
    mesh_dim_names: tuple[str, ...]
    shape: tuple[int, ...]

    def get_coordinate(self) -> list[int]: ...

    def get_group(self, name: str) -> ProcessGroupLike: ...


class CollectiveRegionKind(StrEnum):
    POLICY_TRAINING_STEP = "policy_training_step"
    POLICY_INFERENCE_FORWARD = "policy_inference_forward"


class CollectivePhase(StrEnum):
    TRAINING_STEP_ENTER = "training_step_enter"
    MODEL_FORWARD_ENTER = "model_forward_enter"
    MODEL_FORWARD_EXIT = "model_forward_exit"
    BACKWARD_ENTER = "backward_enter"
    BACKWARD_EXIT = "backward_exit"
    TRAINING_STEP_EXIT = "training_step_exit"
    FORWARD_ENTER = "forward_enter"
    FORWARD_IMPL_ENTER = "forward_impl_enter"
    MOE_EP_A2A_FIRST = "moe_ep_a2a_first"
    FORWARD_IMPL_EXIT = "forward_impl_exit"
    FORWARD_EXIT = "forward_exit"


@dataclass(frozen=True)
class CollectiveRegionMetadata:
    global_step: int | None = None
    local_step: int | None = None


@dataclass(frozen=True)
class MeshCollectiveSnapshot:
    mesh_dim_names: tuple[str, ...]
    mesh_shape: tuple[int, ...]
    mesh_coordinate: tuple[int, ...]
    sequence_numbers: dict[str, int]


@dataclass(frozen=True)
class CollectivePhaseRecord:
    region_id: int
    event_index: int
    kind: CollectiveRegionKind
    rank: int
    phase: CollectivePhase
    metadata: CollectiveRegionMetadata
    snapshot: MeshCollectiveSnapshot


@dataclass
class _RegionContext:
    region_id: int
    kind: CollectiveRegionKind
    rank: int
    metadata: CollectiveRegionMetadata
    device_mesh: DeviceMeshLike
    moe_boundary_logged: bool = False
    next_event_index: int = 0


_region: ContextVar[_RegionContext | None] = ContextVar("collective_phase_region", default=None)


def enabled() -> bool:
    # Read the manager-projected worker contract; phase instrumentation never defines it.
    return os.environ.get(COLLECTIVE_PHASE_DIAGNOSTICS_ENV, "0") == "1"


def _default_process_group() -> ProcessGroupLike:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")
    return dist.distributed_c10d._get_default_group()


def capture_mesh_snapshot(
    device_mesh: DeviceMeshLike,
) -> MeshCollectiveSnapshot:
    """Read mesh geometry and existing process-group counters without synchronization."""
    dim_names = tuple(device_mesh.mesh_dim_names)
    sequence_numbers = {WORLD_GROUP: int(_default_process_group()._get_sequence_number_for_group())}
    for name in dim_names:
        sequence_numbers[name] = int(device_mesh.get_group(name)._get_sequence_number_for_group())
    return MeshCollectiveSnapshot(
        mesh_dim_names=dim_names,
        mesh_shape=tuple(int(size) for size in device_mesh.shape),
        mesh_coordinate=tuple(int(coordinate) for coordinate in device_mesh.get_coordinate()),
        sequence_numbers=sequence_numbers,
    )


@contextmanager
def region(
    device_mesh: DeviceMeshLike | None,
    *,
    kind: CollectiveRegionKind,
    rank: int,
    metadata: CollectiveRegionMetadata | None = None,
) -> Iterator[None]:
    """Scope one policy operation when diagnostics are enabled."""
    if not enabled():
        yield
        return
    if device_mesh is None:
        raise ValueError("enabled collective phase diagnostics require a device mesh")
    with _region_ids_lock:
        region_id = next(_region_ids)
    token = _region.set(
        _RegionContext(
            region_id=region_id,
            kind=kind,
            rank=rank,
            metadata=metadata or CollectiveRegionMetadata(),
            device_mesh=device_mesh,
        )
    )
    try:
        yield
    finally:
        _region.reset(token)


def _capture_record(context: _RegionContext, phase: CollectivePhase) -> CollectivePhaseRecord:
    snapshot = capture_mesh_snapshot(context.device_mesh)
    event_index = context.next_event_index
    context.next_event_index += 1
    return CollectivePhaseRecord(
        region_id=context.region_id,
        event_index=event_index,
        kind=context.kind,
        rank=context.rank,
        phase=phase,
        metadata=context.metadata,
        snapshot=snapshot,
    )


def _active_region() -> _RegionContext | None:
    if not enabled():
        return None
    return _region.get()


def _log_phase(context: _RegionContext, phase: CollectivePhase) -> None:
    record = _capture_record(context, phase)
    payload = json.dumps(asdict(record), sort_keys=True, separators=(",", ":"))
    # The manager owns the path; each rank only consumes its projected value.
    artifact_root = os.environ.get(DEBUG_ARTIFACT_DIR_ENV)
    if not artifact_root:
        logger.info(LOG_PREFIX + payload)
        return
    ensure_debug_artifact_directories(artifact_root)
    path = Path(artifact_root) / "collective_phases" / f"{os.uname().nodename}.{os.getpid()}.rank{record.rank}.jsonl"
    encoded_size = len(payload.encode()) + 1
    with _artifact_lock:
        artifact_file = _artifact_files.get(path)
        if artifact_file is None:
            artifact_file = path.open("a", buffering=1)
            _artifact_files[path] = artifact_file
        if artifact_file.tell() > 0 and artifact_file.tell() + encoded_size > MAX_ARTIFACT_BYTES:
            artifact_file.close()
            previous = path.with_suffix(".previous.jsonl")
            previous.unlink(missing_ok=True)
            path.replace(previous)
            artifact_file = path.open("a", buffering=1)
            _artifact_files[path] = artifact_file
        artifact_file.write(payload + "\n")


def log_phase(phase: CollectivePhase) -> None:
    """Log counters when diagnostics are enabled inside a region."""
    context = _active_region()
    if context is None:
        return
    _log_phase(context, phase)


def start_phase(phase: CollectivePhase) -> None:
    """Start a phase, reset its first-MoE guard, and log its entry."""
    context = _active_region()
    if context is None:
        return
    context.moe_boundary_logged = False
    _log_phase(context, phase)


def log_moe_ep_boundary_once() -> None:
    """Log the first MoE boundary when a diagnostic phase is active."""
    context = _active_region()
    if context is None or context.moe_boundary_logged:
        return
    context.moe_boundary_logged = True
    _log_phase(context, CollectivePhase.MOE_EP_A2A_FIRST)
