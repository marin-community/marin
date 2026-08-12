# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed Endpoint queries and read records."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from rigging.timing import Duration, Timestamp

from iris.resources.errors import InvalidResourceKey
from iris.resources.identity import AttemptIdentity, ResourceKey, ResourceKind

PROXY_TIMEOUT_METADATA_KEY = "proxy_timeout_seconds"
"""Endpoint metadata key overriding the controller proxy timeout in seconds."""


class EndpointAccess(StrEnum):
    """Who may use an Endpoint through the controller proxy."""

    PRIVATE = "private"
    LINK = "link"

    @classmethod
    def from_storage(cls, value: int | None) -> "EndpointAccess":
        """Decode the stable persistence representation."""
        if value in (None, 0):
            return cls.PRIVATE
        if value == 2:
            return cls.LINK
        raise ValueError(f"Unknown stored endpoint access value: {value!r}")

    def to_storage(self) -> int:
        """Encode the stable persistence representation."""
        return 0 if self is EndpointAccess.PRIVATE else 2


@dataclass(frozen=True, slots=True)
class EndpointQuery:
    name_prefix: str | None = None
    task: ResourceKey | None = None
    owner_id: str | None = None
    page_size: int = 100
    page_token: str | None = None
    system_only: bool = False


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
    include_native: bool = False


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
