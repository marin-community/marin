# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed Attempt launch, lifecycle, and provider-runtime records."""

from dataclasses import dataclass

from rigging.timing import Duration, Timestamp

from iris.cluster.constraints import Constraint
from iris.cluster.resources.execution import Environment, ResourceSpec, RuntimeEntrypoint
from iris.cluster.resources.identity import (
    AttemptIdentity as AttemptIdentity,
)
from iris.cluster.resources.identity import (
    AttemptLocator as AttemptLocator,
)
from iris.cluster.resources.identity import (
    NodeIdentity,
)
from iris.cluster.resources.job import ContainerProfile, CoschedulingConfig, PriorityBand
from iris.cluster.resources.source import ResourceSourceStatus
from iris.cluster.resources.state import TaskState
from iris.cluster.types import AttemptUid, JobName


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
