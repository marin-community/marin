# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Internal worker types for task tracking."""

from datetime import datetime, timezone
from typing import Protocol

from finelog.rpc import logging_pb2
from pydantic import BaseModel
from rigging.timing import Timestamp

from iris.rpc import job_pb2
from iris.rpc.job_pb2 import TaskState


class LogLine(BaseModel):
    timestamp: datetime
    source: str  # "build", "stdout", "stderr"
    data: str

    @classmethod
    def now(cls, source: str, data: str) -> "LogLine":
        return cls(timestamp=datetime.now(timezone.utc), source=source, data=data)

    @classmethod
    def at(cls, timestamp: Timestamp, source: str, data: str) -> "LogLine":
        """Create a log line with an explicit timestamp."""
        return cls(
            timestamp=datetime.fromtimestamp(timestamp.epoch_seconds(), tz=timezone.utc),
            source=source,
            data=data,
        )

    def to_proto(self) -> logging_pb2.LogEntry:
        proto = logging_pb2.LogEntry(
            source=self.source,
            data=self.data,
        )
        # finelog.logging.LogEntry uses finelog.logging.Timestamp; assign directly.
        proto.timestamp.epoch_ms = Timestamp.from_seconds(self.timestamp.timestamp()).epoch_ms()
        return proto


class TaskLogs(BaseModel):
    lines: list[LogLine] = []

    def add(self, source: str, data: str, timestamp: Timestamp | None = None) -> None:
        if timestamp:
            self.lines.append(LogLine.at(timestamp, source, data))
        else:
            self.lines.append(LogLine.now(source, data))


class TaskInfo(Protocol):
    """Read-only view of task state used by RPC handlers and the reconcile path.

    Decouples the service layer from TaskAttempt's execution internals (thread,
    runtime, providers, etc.) while exposing the state the worker needs to
    report back to the controller (status, exit_code, error,
    platform_container_id, finished_at).
    """

    @property
    def status(self) -> TaskState:
        """Current task state (PENDING, RUNNING, SUCCEEDED, etc.)."""
        ...

    @property
    def exit_code(self) -> int | None:
        """Process exit code once the container has stopped, else ``None``."""
        ...

    @property
    def error(self) -> str | None:
        """Human-readable error string for failed/worker-failed attempts."""
        ...

    @property
    def platform_container_id(self) -> str | None:
        """Platform-specific container ID (docker hash, k8s pod name, etc.)."""
        ...

    @property
    def finished_at(self) -> Timestamp | None:
        """Terminal-state timestamp; ``None`` while the attempt is still active."""
        ...

    def to_proto(self) -> job_pb2.TaskStatus:
        """Convert to protobuf TaskStatus message."""
        ...
