# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Internal worker types for task tracking."""

from datetime import datetime, timezone
from typing import Protocol

from pydantic import BaseModel

from finelog.rpc import logging_pb2
from iris.rpc import job_pb2
from iris.rpc.job_pb2 import TaskState
from rigging.timing import Timestamp


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
        # finelog.logging.LogEntry uses finelog.time.Timestamp; assign directly.
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
    """Read-only view of task state for RPC handlers.

    This protocol decouples the service layer from TaskAttempt's execution internals
    (thread, runtime, providers, etc.) while providing access to state needed for
    RPC responses.
    """

    @property
    def status(self) -> TaskState:
        """Current task state (PENDING, RUNNING, SUCCEEDED, etc.)."""
        ...

    def to_proto(self) -> job_pb2.TaskStatus:
        """Convert to protobuf TaskStatus message."""
        ...
