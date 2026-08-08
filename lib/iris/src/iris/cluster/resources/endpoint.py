# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed Endpoint queries and read records."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from rigging.timing import Duration, Timestamp

from iris.cluster.resources.errors import InvalidResourceKey
from iris.cluster.resources.identity import AttemptIdentity, ResourceKey, ResourceKind


class EndpointAccess(StrEnum):
    PRIVATE = "private"
    LINK = "link"


@dataclass(frozen=True, slots=True)
class EndpointQuery:
    name_prefix: str | None = None
    task: ResourceKey | None = None
    page_size: int = 100
    page_token: str | None = None


@dataclass(frozen=True, slots=True)
class EndpointSummary:
    key: ResourceKey
    endpoint_id: str
    name: str
    task: ResourceKey | None
    execution_cluster_id: str
    access: EndpointAccess
    lease_deadline: Timestamp | None

    def __post_init__(self) -> None:
        if self.key.kind is not ResourceKind.ENDPOINT:
            raise InvalidResourceKey("EndpointSummary requires an endpoint key")
        if not self.endpoint_id.strip() or self.key.resource_id != self.endpoint_id:
            raise InvalidResourceKey("endpoint_id must equal the endpoint key resource_id")
        if self.task is not None and self.task.kind is not ResourceKind.TASK:
            raise InvalidResourceKey("EndpointSummary task must be a task key")


@dataclass(frozen=True, slots=True)
class EndpointDetail:
    summary: EndpointSummary
    address: str
    metadata: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class EndpointToken:
    token: str
    expires_at: Timestamp
    capability_url: str


class CpuProfileFormat(StrEnum):
    UNSPECIFIED = "unspecified"
    FLAMEGRAPH = "flamegraph"
    SPEEDSCOPE = "speedscope"
    RAW = "raw"


class MemoryProfileFormat(StrEnum):
    UNSPECIFIED = "unspecified"
    FLAMEGRAPH = "flamegraph"
    TABLE = "table"
    STATS = "stats"
    RAW = "raw"


@dataclass(frozen=True, slots=True)
class CpuProfileConfiguration:
    format: CpuProfileFormat
    rate_hz: int
    native: bool | None


@dataclass(frozen=True, slots=True)
class MemoryProfileConfiguration:
    format: MemoryProfileFormat
    leaks: bool


@dataclass(frozen=True, slots=True)
class ThreadsProfileConfiguration:
    include_locals: bool


type ProfileConfiguration = CpuProfileConfiguration | MemoryProfileConfiguration | ThreadsProfileConfiguration


@dataclass(frozen=True, slots=True)
class ExecRequest:
    attempt: AttemptIdentity
    command: tuple[str, ...]
    timeout: Duration | None


@dataclass(frozen=True, slots=True)
class ProfileRequest:
    attempt: AttemptIdentity | None
    profile: ProfileConfiguration | None
    duration: Duration | None


@dataclass(frozen=True, slots=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    error_message: str


@dataclass(frozen=True, slots=True)
class ProfileResult:
    profile_data: bytes
    error_message: str
