# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed Job specifications, queries, and read records."""

from dataclasses import dataclass
from enum import IntEnum, StrEnum

from rigging.timing import Duration, Timestamp

from iris.cluster.constraints import Constraint
from iris.cluster.resources.execution import Environment, ResourceSpec, RuntimeEntrypoint
from iris.cluster.resources.identity import JobIdentity, ResourceKey
from iris.cluster.resources.state import JobState, PriorityBand, TaskState


class JobPreemptionPolicy(IntEnum):
    UNSPECIFIED = 0
    TERMINATE_CHILDREN = 1
    PRESERVE_CHILDREN = 2


class ExistingJobPolicy(IntEnum):
    UNSPECIFIED = 0
    ERROR = 1
    KEEP = 2
    RECREATE = 3


class ContainerProfile(IntEnum):
    UNSPECIFIED = 0
    RESTRICTED = 1
    DEFAULT = 2
    DOCKER_ACCESS = 3
    PRIVILEGED = 4
    GVISOR = 5


@dataclass(frozen=True, slots=True)
class CoschedulingConfig:
    group_by: str


@dataclass(frozen=True, slots=True)
class JobSpec:
    version: int
    name: str
    entrypoint: RuntimeEntrypoint
    resources: ResourceSpec
    environment: Environment
    bundle_id: str
    scheduling_timeout: Duration | None
    ports: tuple[str, ...]
    max_task_failures: int
    max_retries_failure: int
    max_retries_preemption: int
    constraints: tuple[Constraint, ...]
    coscheduling: CoschedulingConfig | None
    replicas: int
    timeout: Duration | None
    fail_if_exists: bool
    preemption_policy: JobPreemptionPolicy
    existing_job_policy: ExistingJobPolicy
    priority_band: PriorityBand
    task_image: str
    submit_argv: tuple[str, ...]
    client_revision_date: str
    container_profile: ContainerProfile

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports", tuple(self.ports))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "submit_argv", tuple(self.submit_argv))


@dataclass(frozen=True, slots=True)
class JobQuery:
    owner_id: str | None = None
    parent: ResourceKey | None = None
    job_id_prefix: str | None = None
    states: frozenset[JobState] = frozenset()
    backend_id: str | None = None
    execution_cluster_id: str | None = None
    page_size: int = 50
    page_token: str | None = None


@dataclass(frozen=True, slots=True)
class JobSummary:
    identity: JobIdentity
    owner_id: str
    parent: JobIdentity | None
    state: JobState
    execution_cluster_id: str
    backend_id: str
    num_tasks: int
    submitted_at: Timestamp
    started_at: Timestamp | None
    finished_at: Timestamp | None
    error_message: str
    pending_reason: str


class FederationPosture(StrEnum):
    """A Job's durable handoff posture, independent of the legacy RPC enum."""

    LOCAL = "local"
    QUEUED = "queued"
    PENDING_ACCEPTANCE = "pending_acceptance"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class TaskStateCount:
    state: TaskState
    count: int


@dataclass(frozen=True, slots=True)
class JobTaskAggregate:
    task_count: int
    completed_count: int
    failure_count: int
    preemption_count: int
    state_counts: tuple[TaskStateCount, ...]


@dataclass(frozen=True, slots=True)
class JobObservation:
    """A bounded persisted Job observation with its Task and child aggregates."""

    summary: JobSummary
    tasks: JobTaskAggregate
    has_children: bool
    federation_posture: FederationPosture


@dataclass(frozen=True, slots=True)
class JobDetail:
    summary: JobSummary
    spec: JobSpec
