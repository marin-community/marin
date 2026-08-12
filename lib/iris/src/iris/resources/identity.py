# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Logical resource keys and exact resource incarnations."""

from dataclasses import dataclass
from enum import StrEnum

from iris.resources.errors import InvalidResourceKey


class ResourceKind(StrEnum):
    JOB = "job"
    TASK = "task"
    ATTEMPT = "attempt"
    ENDPOINT = "endpoint"
    NODE = "node"
    SLICE = "slice"


def _require_identifier(value: str, field_name: str) -> None:
    if not value.strip():
        raise InvalidResourceKey(f"{field_name} must be non-empty")


def _require_kind(key: "ResourceKey", expected: ResourceKind) -> None:
    if key.kind is not expected:
        raise InvalidResourceKey(f"expected a {expected.value} key, got {key.kind.value}")


def _require_attempt_number(attempt_number: int) -> None:
    if not isinstance(attempt_number, int) or isinstance(attempt_number, bool) or attempt_number < 0:
        raise InvalidResourceKey("attempt_number must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ResourceKey:
    cluster_id: str
    kind: ResourceKind
    resource_id: str

    def __post_init__(self) -> None:
        _require_identifier(self.cluster_id, "cluster_id")
        if not isinstance(self.kind, ResourceKind):
            raise InvalidResourceKey("kind must be a ResourceKind")
        _require_identifier(self.resource_id, "resource_id")
        if self.kind is ResourceKind.ATTEMPT:
            task_id, separator, attempt_number = self.resource_id.rpartition(":")
            if (
                not separator
                or not task_id
                or not attempt_number.isdecimal()
                or str(int(attempt_number)) != attempt_number
            ):
                raise InvalidResourceKey("attempt resource_id must end in a canonical non-negative attempt number")


@dataclass(frozen=True, slots=True)
class JobIdentity:
    key: ResourceKey
    job_uid: str

    def __post_init__(self) -> None:
        _require_kind(self.key, ResourceKind.JOB)
        _require_identifier(self.job_uid, "job_uid")


@dataclass(frozen=True, slots=True)
class TaskIdentity:
    key: ResourceKey
    task_uid: str

    def __post_init__(self) -> None:
        _require_kind(self.key, ResourceKind.TASK)
        _require_identifier(self.task_uid, "task_uid")


@dataclass(frozen=True, slots=True)
class AttemptLocator:
    task: ResourceKey
    attempt_number: int | None

    def __post_init__(self) -> None:
        _require_kind(self.task, ResourceKind.TASK)
        if self.attempt_number is not None:
            _require_attempt_number(self.attempt_number)


@dataclass(frozen=True, slots=True)
class AttemptIdentity:
    task: ResourceKey
    attempt_number: int
    attempt_uid: str

    def __post_init__(self) -> None:
        _require_kind(self.task, ResourceKind.TASK)
        _require_attempt_number(self.attempt_number)
        _require_identifier(self.attempt_uid, "attempt_uid")


@dataclass(frozen=True, slots=True)
class NodeIdentity:
    key: ResourceKey
    backend_id: str
    node_uid: str

    def __post_init__(self) -> None:
        _require_kind(self.key, ResourceKind.NODE)
        _require_identifier(self.backend_id, "backend_id")
        _require_identifier(self.node_uid, "node_uid")


@dataclass(frozen=True, slots=True)
class NodeLocator:
    key: ResourceKey
    backend_id: str
    node_uid: str | None = None

    def __post_init__(self) -> None:
        _require_kind(self.key, ResourceKind.NODE)
        _require_identifier(self.backend_id, "backend_id")
        if self.node_uid is not None:
            _require_identifier(self.node_uid, "node_uid")


@dataclass(frozen=True, slots=True)
class SliceIdentity:
    key: ResourceKey
    backend_id: str
    slice_uid: str

    def __post_init__(self) -> None:
        _require_kind(self.key, ResourceKind.SLICE)
        _require_identifier(self.backend_id, "backend_id")
        _require_identifier(self.slice_uid, "slice_uid")


@dataclass(frozen=True, slots=True)
class SliceLocator:
    key: ResourceKey
    backend_id: str
    slice_uid: str | None = None

    def __post_init__(self) -> None:
        _require_kind(self.key, ResourceKind.SLICE)
        _require_identifier(self.backend_id, "backend_id")
        if self.slice_uid is not None:
            _require_identifier(self.slice_uid, "slice_uid")
