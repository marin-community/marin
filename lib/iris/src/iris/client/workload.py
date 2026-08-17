# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable workload snapshots returned by the public client."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from rigging.timing import Timestamp

from iris.cluster.types import JobName
from iris.resources.state import FederationState, JobState, TaskState


class DeviceKind(StrEnum):
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


@dataclass(frozen=True, slots=True)
class Device:
    kind: DeviceKind
    variant: str
    count: int
    topology: str = ""


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """Normalized resources requested by a workload."""

    cpu_millicores: int
    memory_bytes: int
    disk_bytes: int
    device: Device | None = None


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    memory_mb: int
    disk_mb: int
    cpu_millicores: int
    memory_peak_mb: int
    process_count: int


@dataclass(frozen=True, slots=True)
class BuildMetrics:
    started_at: Timestamp | None
    finished_at: Timestamp | None
    from_cache: bool
    image_tag: str


@dataclass(frozen=True, slots=True)
class AttemptStatus:
    """One execution attempt reported for the current logical Task."""

    attempt_number: int
    attempt_uid: str
    state: TaskState
    worker_id: str
    exit_code: int
    error_message: str
    started_at: Timestamp | None
    finished_at: Timestamp | None
    is_worker_failure: bool
    pod_name: str
    pod_uid: str
    node_name: str
    terminal_reason: str


@dataclass(frozen=True, slots=True)
class TaskStatus:
    """Current snapshot of a logical Task name."""

    task_id: JobName
    state: TaskState
    worker_id: str
    worker_address: str
    exit_code: int
    error_message: str
    submitted_at: Timestamp | None
    started_at: Timestamp | None
    finished_at: Timestamp | None
    ports: Mapping[str, int]
    resource_usage: ResourceUsage | None
    build_metrics: BuildMetrics | None
    current_attempt_number: int
    attempts: tuple[AttemptStatus, ...]
    pending_reason: str
    can_be_scheduled: bool
    container_id: str
    backend_id: str
    execution_cluster_id: str
    status_message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports", MappingProxyType(dict(self.ports)))
        object.__setattr__(self, "attempts", tuple(self.attempts))


@dataclass(frozen=True, slots=True)
class JobStatus:
    """Current snapshot of a logical Job name.

    The legacy controller wire does not expose a Job incarnation UID. A value
    returned here therefore describes whichever Job currently owns ``job_id``.
    """

    job_id: JobName
    state: JobState
    exit_code: int
    error_message: str
    submitted_at: Timestamp | None
    started_at: Timestamp | None
    finished_at: Timestamp | None
    ports: Mapping[str, int]
    status_message: str
    build_metrics: BuildMetrics | None
    failure_count: int
    preemption_count: int
    tasks: tuple[TaskStatus, ...]
    name: str
    resources: ResourceRequest
    task_state_counts: Mapping[TaskState, int]
    task_count: int
    completed_count: int
    pending_reason: str
    has_children: bool
    parent_job_id: JobName | None
    backend_id: str
    execution_cluster_id: str
    federation_state: FederationState
    submitting_user: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports", MappingProxyType(dict(self.ports)))
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "task_state_counts", MappingProxyType(dict(self.task_state_counts)))
