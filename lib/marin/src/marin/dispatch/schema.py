# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Schema for the event-driven monitoring dispatcher."""

from dataclasses import dataclass
from enum import StrEnum


class RunTrack(StrEnum):
    RAY = "ray"
    IRIS = "iris"


class RunStatus(StrEnum):
    UNKNOWN = "unknown"
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"


class TickEventKind(StrEnum):
    SCHEDULED_POLL = "scheduled_poll"
    MANUAL = "manual"
    FAILURE_ALERT = "failure_alert"


@dataclass(frozen=True)
class RayRunConfig:
    job_id: str
    cluster: str
    experiment: str


@dataclass(frozen=True)
class IrisRunConfig:
    job_id: str
    config: str
    resubmit_command: str


@dataclass(frozen=True)
class RunPointer:
    """A pointer to a single run, either Ray or Iris.

    Exactly one of `ray` or `iris` must be set, matching `track`.
    """

    track: RunTrack
    ray: RayRunConfig | None = None
    iris: IrisRunConfig | None = None

    def __post_init__(self) -> None:
        if self.track == RunTrack.RAY and self.ray is None:
            raise ValueError("RunPointer with track=ray must have ray config set")
        if self.track == RunTrack.IRIS and self.iris is None:
            raise ValueError("RunPointer with track=iris must have iris config set")


@dataclass(frozen=True)
class MonitoringCollection:
    """A collection binding a research thread to its monitoring context."""

    name: str
    prompt: str
    logbook: str
    branch: str
    issue: int
    runs: tuple[RunPointer, ...] = ()
    created_at: str = ""
    paused: bool = False


@dataclass(frozen=True)
class TickEvent:
    """Payload delivered to an agent session for one monitoring tick."""

    kind: TickEventKind
    collection_name: str
    run_index: int
    run_pointer: RunPointer
    prompt: str
    logbook: str
    branch: str
    issue: int
    timestamp: str


@dataclass
class RunState:
    """Mutable per-run state tracked alongside a collection."""

    last_status: RunStatus = RunStatus.UNKNOWN
    last_check: str = ""
    restart_count: int = 0
    last_error: str = ""
    consecutive_failures: int = 0
