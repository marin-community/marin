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

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from iris.cluster.types import is_task_finished
from iris.rpc import cluster_pb2
from iris.rpc.cluster_pb2 import TaskState
from iris.time_utils import now_ms


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


@dataclass(kw_only=True)
class Task:
    """Internal worker representation of a running task."""

    task_id: str  # Full task ID: "{job_id}/task-{index}"
    job_id: str  # Parent job ID
    task_index: int = 0  # 0-indexed
    num_tasks: int = 1  # Total tasks in job
    attempt_id: int = 0
    request: cluster_pb2.Worker.RunTaskRequest
    status: TaskState = cluster_pb2.TASK_STATE_PENDING
    exit_code: int | None = None
    error: str | None = None
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    ports: dict[str, int] = field(default_factory=dict)
    status_message: str = ""

    # Resource tracking
    current_memory_mb: int = 0
    peak_memory_mb: int = 0
    current_cpu_percent: int = 0
    process_count: int = 0
    disk_mb: int = 0

    # Build tracking
    build_started_ms: int | None = None
    build_finished_ms: int | None = None
    build_from_cache: bool = False
    image_tag: str = ""

    # Internals
    container_id: str | None = None
    workdir: Path | None = None  # Task working directory with logs
    thread: threading.Thread | None = None
    cleanup_done: bool = False
    should_stop: bool = False

    # Structured logs (build logs stored here, container logs fetched from Docker)
    logs: TaskLogs = field(default_factory=TaskLogs)

    result: bytes | None = None  # cloudpickle serialized return value from container

    def transition_to(
        self,
        state: TaskState,
        *,
        message: str = "",
        error: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        self.status = state
        self.status_message = message
        if is_task_finished(state):
            self.finished_at_ms = now_ms()
            if error:
                self.error = error
            if exit_code is not None:
                self.exit_code = exit_code

    def to_proto(self) -> cluster_pb2.TaskStatus:
        return cluster_pb2.TaskStatus(
            task_id=self.task_id,
            job_id=self.job_id,
            task_index=self.task_index,
            state=self.status,
            exit_code=self.exit_code or 0,
            error=self.error or "",
            started_at_ms=self.started_at_ms or 0,
            finished_at_ms=self.finished_at_ms or 0,
            ports=self.ports,
            current_attempt_id=self.attempt_id,
            resource_usage=cluster_pb2.ResourceUsage(
                memory_mb=self.current_memory_mb,
                memory_peak_mb=self.peak_memory_mb,
                disk_mb=self.disk_mb,
                cpu_millicores=self.current_cpu_percent * 10,
                cpu_percent=self.current_cpu_percent,
                process_count=self.process_count,
            ),
            build_metrics=cluster_pb2.BuildMetrics(
                build_started_ms=self.build_started_ms or 0,
                build_finished_ms=self.build_finished_ms or 0,
                from_cache=self.build_from_cache,
                image_tag=self.image_tag,
            ),
        )
