# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native records exchanged between a worker and its transport boundary."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType

from rigging.provenance import Provenance
from rigging.timing import Timestamp

from iris.cluster.constraints import AttributeValue
from iris.cluster.types import AttemptUid, JobName
from iris.resources.attempt import AttemptLaunch, AttemptObservation
from iris.resources.execution import Device
from iris.resources.state import TaskState


@dataclass(frozen=True, slots=True)
class WorkerMetadata:
    hostname: str = ""
    ip_address: str = ""
    cpu_count: int = 0
    memory_bytes: int = 0
    disk_bytes: int = 0
    device: Device | None = None
    tpu_name: str = ""
    tpu_worker_hostnames: str = ""
    tpu_worker_id: str = ""
    tpu_chips_per_host_bounds: str = ""
    gpu_count: int = 0
    gpu_name: str = ""
    gpu_memory_mb: int = 0
    gce_instance_name: str = ""
    gce_zone: str = ""
    attributes: Mapping[str, AttributeValue] = field(default_factory=dict)
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))


@dataclass(frozen=True, slots=True)
class WorkerResourceSnapshot:
    timestamp: Timestamp | None = None
    host_cpu_percent: int = 0
    memory_used_bytes: int = 0
    memory_total_bytes: int = 0
    disk_used_bytes: int = 0
    disk_total_bytes: int = 0
    running_task_count: int = 0
    total_process_count: int = 0
    net_recv_bytes: int = 0
    net_sent_bytes: int = 0


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    memory_mb: int = 0
    disk_mb: int = 0
    cpu_millicores: int = 0
    memory_peak_mb: int = 0
    process_count: int = 0


@dataclass(frozen=True, slots=True)
class BuildMetrics:
    build_started: Timestamp | None = None
    build_finished: Timestamp | None = None
    from_cache: bool = False
    image_tag: str = ""


@dataclass(frozen=True, slots=True)
class WorkerTaskStatus:
    task_id: JobName
    state: TaskState
    worker_id: str = ""
    worker_address: str = ""
    exit_code: int | None = None
    error: str = ""
    started_at: Timestamp | None = None
    finished_at: Timestamp | None = None
    ports: Mapping[str, int] = field(default_factory=dict)
    resource_usage: ResourceUsage | None = None
    build_metrics: BuildMetrics | None = None
    attempt_id: int = 0
    container_id: str = ""
    status_message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports", MappingProxyType(dict(self.ports)))


class StopReason(IntEnum):
    UNSPECIFIED = 0
    CANCELLED = 1
    PREEMPTED = 2
    SUPERSEDED = 3
    JOB_TERMINATED = 4
    TASK_TIMEOUT = 5
    WORKER_DRAIN = 6


@dataclass(frozen=True, slots=True)
class DesiredAttempt:
    attempt_uid: AttemptUid
    launch: AttemptLaunch | None = None
    stop_reason: StopReason | None = None

    def __post_init__(self) -> None:
        if (self.stop_reason is None) == (self.launch is None):
            # A run directive may intentionally omit its one-shot launch spec.
            if self.stop_reason is None:
                return
            raise ValueError("DesiredAttempt must select run or stop")

    @property
    def is_run(self) -> bool:
        return self.stop_reason is None


@dataclass(frozen=True, slots=True)
class AttemptStatus:
    observation: AttemptObservation
    finished_at: Timestamp | None = None
    resource_usage: ResourceUsage | None = None


@dataclass(frozen=True, slots=True)
class WorkerHealth:
    healthy: bool
    health_error: str = ""
    resources: WorkerResourceSnapshot | None = None


@dataclass(frozen=True, slots=True)
class WorkerReconcileRequest:
    worker_id: str
    desired: tuple[DesiredAttempt, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "desired", tuple(self.desired))


@dataclass(frozen=True, slots=True)
class WorkerReconcileResponse:
    worker_id: str
    observed: tuple[AttemptStatus, ...]
    health: WorkerHealth

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed", tuple(self.observed))
