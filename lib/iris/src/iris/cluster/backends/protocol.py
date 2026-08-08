# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed, persistence-free contracts for Iris execution backends."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from rigging.timing import Deadline, Duration, Timestamp

from iris.cluster.resources.action import ActionKind
from iris.cluster.resources.activity import ActivityEntry
from iris.cluster.resources.attempt import AttemptDetail, AttemptRuntimeObject
from iris.cluster.resources.endpoint import ExecResult, ProfileResult
from iris.cluster.resources.identity import (
    AttemptIdentity,
    JobIdentity,
    NodeIdentity,
    SliceIdentity,
    TaskIdentity,
)
from iris.cluster.resources.job import JobSpec
from iris.cluster.resources.node import NodeDetail, NodeSummary
from iris.cluster.resources.slice import SliceDetail, SliceSummary
from iris.cluster.resources.source import ResourceSourceStatus
from iris.cluster.types import TaskState
from iris.rpc import job_pb2


class BackendCapability(StrEnum):
    """A backend feature used for routing and resource presentation."""

    WORKER_DAEMON = "workers"
    IRIS_AUTOSCALER = "autoscaler"
    CLUSTER_VIEW = "cluster"


@dataclass(frozen=True, slots=True)
class SourceSnapshot[T]:
    items: tuple[T, ...]
    status: ResourceSourceStatus


@dataclass(frozen=True, slots=True)
class ExactAttemptTarget:
    identity: AttemptIdentity
    execution_cluster_id: str
    backend_id: str
    node_uid: str | None
    node_address: str | None
    runtime: AttemptRuntimeObject | None


@dataclass(frozen=True, slots=True)
class PendingTaskInput:
    identity: TaskIdentity
    job: JobIdentity
    task_index: int
    owner_id: str
    spec: JobSpec
    current_attempt: AttemptIdentity | None
    failure_count: int
    preemption_count: int
    submitted_at: Timestamp


@dataclass(frozen=True, slots=True)
class BudgetInput:
    owner_id: str
    priority_band: job_pb2.PriorityBand
    accelerator_kind: str
    limit: float
    usage: float


@dataclass(frozen=True, slots=True)
class ScheduleRequest:
    backend_id: str
    pending_tasks: tuple[PendingTaskInput, ...]
    nodes: SourceSnapshot[NodeSummary]
    slices: SourceSnapshot[SliceSummary]
    budgets: tuple[BudgetInput, ...]
    now: Timestamp
    trace: bool


@dataclass(frozen=True, slots=True)
class PlacementDecision:
    task: TaskIdentity
    backend_id: str
    scaling_group_id: str | None
    node: NodeIdentity | None
    slice: SliceIdentity | None


@dataclass(frozen=True, slots=True)
class PreemptionDecision:
    attempt: AttemptIdentity
    reason: str


@dataclass(frozen=True, slots=True)
class UnschedulableDecision:
    task: TaskIdentity
    reason: str


@dataclass(frozen=True, slots=True)
class CapacityDemandInput:
    accelerator_kind: str
    accelerator_variant: str
    accelerator_count: int
    priority_band: job_pb2.PriorityBand
    task_count: int


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    placements: tuple[PlacementDecision, ...] = ()
    preemptions: tuple[PreemptionDecision, ...] = ()
    unschedulable: tuple[UnschedulableDecision, ...] = ()
    residual_demand: tuple[CapacityDemandInput, ...] = ()


@dataclass(frozen=True, slots=True)
class DesiredAttemptInput:
    identity: AttemptIdentity
    job: JobIdentity
    spec: JobSpec
    backend_id: str
    node: NodeIdentity | None
    desired_state: TaskState


@dataclass(frozen=True, slots=True)
class ActionTargetInput:
    action_id: str
    kind: ActionKind
    attempt: ExactAttemptTarget


@dataclass(frozen=True, slots=True)
class ReconcileRequest:
    backend_id: str
    desired_attempts: tuple[DesiredAttemptInput, ...]
    action_targets: tuple[ActionTargetInput, ...]
    nodes: SourceSnapshot[NodeSummary]
    slices: SourceSnapshot[SliceSummary]
    now: Timestamp


@dataclass(frozen=True, slots=True)
class AttemptObservation:
    identity: AttemptIdentity
    state: TaskState
    node: NodeIdentity | None
    runtime: AttemptRuntimeObject | None
    started_at: Timestamp | None
    finished_at: Timestamp | None
    exit_code: int | None
    status_message: str
    error_message: str
    terminal_reason: str


class RuntimeTargetState(StrEnum):
    ACTIVE = "active"
    ABSENT = "absent"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ActionObservation:
    action_id: str
    attempt: AttemptIdentity
    state: RuntimeTargetState
    reason: str


@dataclass(frozen=True, slots=True)
class NodeRetirementDecision:
    node: NodeIdentity
    reason: str


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    attempts: tuple[AttemptObservation, ...] = ()
    actions: tuple[ActionObservation, ...] = ()
    retired_nodes: tuple[NodeRetirementDecision, ...] = ()
    activity: tuple[ActivityEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class AutoscaleRequest:
    backend_id: str
    demand: tuple[CapacityDemandInput, ...]
    nodes: SourceSnapshot[NodeSummary]
    slices: SourceSnapshot[SliceSummary]
    now: Timestamp


@dataclass(frozen=True, slots=True)
class AutoscaleResult:
    nodes: SourceSnapshot[NodeSummary]
    slices: SourceSnapshot[SliceSummary]
    retired_nodes: tuple[NodeRetirementDecision, ...] = ()
    activity: tuple[ActivityEntry, ...] = ()


class NodeReader(Protocol):
    def snapshot_nodes(self) -> SourceSnapshot[NodeSummary]:
        """Return the last complete bounded snapshot without provider I/O."""

    def describe_node(self, identity: NodeIdentity, *, deadline: Deadline) -> NodeDetail:
        """Return cached detail with at most one deadline-bounded enrichment."""


class SliceReader(Protocol):
    def snapshot_slices(self) -> SourceSnapshot[SliceSummary]:
        """Return the last complete bounded projection without provider I/O."""

    def describe_slice(self, identity: SliceIdentity, *, deadline: Deadline) -> SliceDetail:
        """Return cached membership with at most one deadline-bounded refresh."""


class AttemptRuntime(Protocol):
    def describe_attempt(self, target: ExactAttemptTarget, *, deadline: Deadline) -> AttemptDetail:
        """Describe only the exact runtime identity carried by target."""

    def exec_attempt(
        self,
        target: ExactAttemptTarget,
        command: Sequence[str],
        *,
        deadline: Deadline,
    ) -> ExecResult: ...

    def profile_attempt(
        self,
        target: ExactAttemptTarget,
        profile: job_pb2.ProfileType,
        *,
        duration: Duration,
        deadline: Deadline,
    ) -> ProfileResult: ...


class TaskBackend(Protocol):
    backend_id: str
    name: str
    capabilities: frozenset[BackendCapability]
    attempts: AttemptRuntime

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        """Make a pure placement decision from controller-owned input."""

    def reconcile(self, request: ReconcileRequest) -> ReconcileResult:
        """Perform bounded effects and return observations for controller commit."""

    def autoscale(self, request: AutoscaleRequest) -> AutoscaleResult:
        """Perform one bounded capacity cycle and return its projection."""

    def close(self) -> None:
        """Release backend-owned resources without writing controller state."""


@dataclass(frozen=True, slots=True)
class BackendBinding:
    tasks: TaskBackend
    nodes: NodeReader
    slices: SliceReader | None
