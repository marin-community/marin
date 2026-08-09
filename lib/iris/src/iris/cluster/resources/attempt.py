# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed Attempt lifecycle and provider-runtime records."""

from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.cluster.resources.identity import (
    AttemptIdentity as AttemptIdentity,
)
from iris.cluster.resources.identity import (
    AttemptLocator as AttemptLocator,
)
from iris.cluster.resources.identity import (
    NodeIdentity,
)
from iris.cluster.resources.source import ResourceSourceStatus
from iris.cluster.resources.state import TaskState


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
