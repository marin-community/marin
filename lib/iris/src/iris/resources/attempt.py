# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed Attempt launch, lifecycle, retry-count, and provider-runtime records."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from rigging.timing import Duration, Timestamp

from iris.cluster.constraints import Constraint
from iris.resources.execution import Environment, ResourceSpec, RuntimeEntrypoint
from iris.resources.identity import (
    AttemptIdentity as AttemptIdentity,
)
from iris.resources.identity import (
    AttemptLocator as AttemptLocator,
)
from iris.resources.identity import (
    NodeIdentity,
)
from iris.resources.job import ContainerProfile, CoschedulingConfig, PriorityBand
from iris.resources.names import (
    AttemptUid,
    JobName,
)
from iris.resources.source import ResourceSourceStatus
from iris.resources.state import TaskState

PREEMPTION_ATTEMPT_STATES: frozenset[int] = frozenset(
    {
        TaskState.WORKER_FAILED,
        TaskState.KILLED,
        TaskState.PREEMPTED,
    }
)


@dataclass(frozen=True, slots=True)
class AttemptCounts:
    """Retry counters derived from a Task's Attempt history."""

    failure_count: int = 0
    preemption_count: int = 0


class AttemptCountRecord(Protocol):
    """Attempt fields needed to derive retry counters."""

    state: int
    started_at_ms: object | None


def counts_from_attempts(attempts: Iterable[AttemptCountRecord]) -> AttemptCounts:
    """Derive retry counters from an iterable of Attempt lifecycle records."""
    failure = 0
    preemption = 0
    for attempt in attempts:
        state = int(attempt.state)
        if state == TaskState.FAILED:
            failure += 1
        elif state in PREEMPTION_ATTEMPT_STATES and attempt.started_at_ms is not None:
            preemption += 1
    return AttemptCounts(failure_count=failure, preemption_count=preemption)


@dataclass(frozen=True, slots=True)
class AttemptLaunchTemplate:
    """Immutable per-Job execution fields shared by all of its Attempts."""

    num_tasks: int
    entrypoint: RuntimeEntrypoint
    environment: Environment
    bundle_id: str
    resources: ResourceSpec
    timeout: Duration | None
    ports: tuple[str, ...]
    constraints: tuple[Constraint, ...]
    task_image: str
    coscheduling: CoschedulingConfig | None
    priority_band: PriorityBand
    container_profile: ContainerProfile

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports", tuple(self.ports))
        object.__setattr__(self, "constraints", tuple(self.constraints))


@dataclass(frozen=True, slots=True)
class AttemptLaunch:
    """Exact Attempt incarnation and the execution fields needed to start it."""

    task_id: JobName
    attempt_id: int
    attempt_uid: AttemptUid
    template: AttemptLaunchTemplate


@dataclass(frozen=True, slots=True)
class AttemptObservation:
    """Runtime state reported for one Attempt by an execution backend."""

    attempt_uid: AttemptUid
    state: TaskState
    exit_code: int | None = None
    error: str | None = None
    container_id: str | None = None


@dataclass(frozen=True, slots=True)
class AttemptSummary:
    identity: AttemptIdentity
    state: TaskState
    execution_cluster_id: str
    backend_id: str
    node: NodeIdentity | None
    created_at: Timestamp
    started_at: Timestamp | None
    finished_at: Timestamp | None
    exit_code: int | None
    error_message: str
    terminal_reason: str


@dataclass(frozen=True, slots=True)
class AttemptRuntimeObject:
    provider_kind: str
    namespace: str
    name: str
    provider_uid: str
    provider_node_id: str
    provider_node_uid: str
    container_id: str
    observed_at: Timestamp


@dataclass(frozen=True, slots=True)
class AttemptDetail:
    summary: AttemptSummary
    runtime: AttemptRuntimeObject | None
    source_statuses: tuple[ResourceSourceStatus, ...]
