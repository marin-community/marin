# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Internal worker types for task tracking."""

from datetime import datetime, timezone
from typing import Protocol

from pydantic import BaseModel

from iris.rpc import cluster_pb2
from iris.rpc.cluster_pb2 import TaskState
from iris.time_utils import Timestamp


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

    def to_proto(self) -> cluster_pb2.Worker.LogEntry:
        proto = cluster_pb2.Worker.LogEntry(
            source=self.source,
            data=self.data,
        )
        proto.timestamp.CopyFrom(Timestamp.from_seconds(self.timestamp.timestamp()).to_proto())
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

    @property
    def result(self) -> bytes | None:
        """Serialized task result (cloudpickle), if available."""
        ...

    def to_proto(self) -> cluster_pb2.TaskStatus:
        """Convert to protobuf TaskStatus message."""
        ...
