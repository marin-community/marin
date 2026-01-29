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

from pydantic import BaseModel

from iris.rpc import cluster_pb2


class LogLine(BaseModel):
    timestamp: datetime
    source: str  # "build", "stdout", "stderr"
    data: str

    @classmethod
    def now(cls, source: str, data: str) -> "LogLine":
        return cls(timestamp=datetime.now(timezone.utc), source=source, data=data)

    def to_proto(self) -> cluster_pb2.Worker.LogEntry:
        return cluster_pb2.Worker.LogEntry(
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
            source=self.source,
            data=self.data,
        )


class TaskLogs(BaseModel):
    lines: list[LogLine] = []

    def add(self, source: str, data: str) -> None:
        self.lines.append(LogLine.now(source, data))
