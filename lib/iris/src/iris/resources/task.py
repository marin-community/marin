# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed Task queries and read records."""

from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.resources.attempt import AttemptSummary
from iris.resources.identity import AttemptIdentity, JobIdentity, NodeIdentity, ResourceKey, TaskIdentity
from iris.resources.source import (
    MAX_ROOT_CAUSE_HIGHLIGHT,
    MAX_ROOT_CAUSE_HIGHLIGHTS,
    ResourceSourceStatus,
)
from iris.resources.state import TaskState


@dataclass(frozen=True, slots=True)
class TaskQuery:
    job: ResourceKey | None = None
    job_id_prefix: str | None = None
    states: frozenset[TaskState] = frozenset()
    backend_id: str | None = None
    authority_cluster_id: str | None = None
    execution_cluster_id: str | None = None
    page_size: int = 100
    page_token: str | None = None


@dataclass(frozen=True, slots=True)
class TaskSummary:
    identity: TaskIdentity
    job: JobIdentity
    task_index: int
    state: TaskState
    execution_cluster_id: str
    backend_id: str
    current_attempt: AttemptIdentity | None
    current_node: NodeIdentity | None
    failure_count: int
    preemption_count: int
    submitted_at: Timestamp
    started_at: Timestamp | None
    finished_at: Timestamp | None
    status_message: str
    error_message: str


@dataclass(frozen=True, slots=True)
class TaskDetail:
    summary: TaskSummary
    attempts: tuple[AttemptSummary, ...]
    source_statuses: tuple[ResourceSourceStatus, ...]
    root_cause_highlights: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.root_cause_highlights) > MAX_ROOT_CAUSE_HIGHLIGHTS:
            raise ValueError(f"root_cause_highlights exceeds {MAX_ROOT_CAUSE_HIGHLIGHTS} entries")
        if any(len(highlight) > MAX_ROOT_CAUSE_HIGHLIGHT for highlight in self.root_cause_highlights):
            raise ValueError(f"root-cause highlight exceeds {MAX_ROOT_CAUSE_HIGHLIGHT} characters")
