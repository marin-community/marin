# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed Node inventory and detail records."""

from dataclasses import dataclass
from enum import StrEnum

from rigging.timing import Timestamp

from iris.cluster.resources.attempt import AttemptSummary
from iris.cluster.resources.identity import (
    NodeIdentity as NodeIdentity,
)
from iris.cluster.resources.identity import (
    NodeLocator as NodeLocator,
)
from iris.cluster.resources.identity import (
    SliceIdentity,
)
from iris.cluster.resources.source import ResourceSourceStatus


class NodeHealth(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class NodeQuery:
    backend_id: str | None = None
    contains: str | None = None
    health: frozenset[NodeHealth] = frozenset()
    page_size: int = 100
    page_token: str | None = None


@dataclass(frozen=True, slots=True)
class NodeCapacity:
    cpu_millicores: int
    memory_bytes: int
    disk_bytes: int
    accelerator_kind: str
    accelerator_variant: str
    accelerator_count: int


class NodeAttributeKind(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"


@dataclass(frozen=True, slots=True)
class NodeAttribute:
    key: str
    kind: NodeAttributeKind
    string_value: str | None = None
    integer_value: int | None = None
    float_value: float | None = None

    def __post_init__(self) -> None:
        selected = {
            NodeAttributeKind.STRING: self.string_value is not None,
            NodeAttributeKind.INTEGER: self.integer_value is not None,
            NodeAttributeKind.FLOAT: self.float_value is not None,
        }
        if not isinstance(self.kind, NodeAttributeKind) or not selected[self.kind]:
            raise ValueError(f"{self.kind!r} requires its matching attribute value")
        populated = sum(value is not None for value in (self.string_value, self.integer_value, self.float_value))
        if populated != 1:
            raise ValueError("NodeAttribute requires exactly one typed value")


@dataclass(frozen=True, slots=True)
class NodeSummary:
    identity: NodeIdentity
    health: NodeHealth
    schedulable: bool
    capacity: NodeCapacity
    scaling_group_id: str | None
    slice: SliceIdentity | None
    running_task_count: int
    observed_at: Timestamp


@dataclass(frozen=True, slots=True)
class NodeDetail:
    summary: NodeSummary
    address: str | None
    attributes: tuple[NodeAttribute, ...]
    recent_attempts: tuple[AttemptSummary, ...]
    bootstrap_log_key: str | None
    source_statuses: tuple[ResourceSourceStatus, ...]
