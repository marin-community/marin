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

"""WorkerService RPC implementation using Connect RPC."""

import re
import time
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from iris.cluster.worker.worker_types import Task
from iris.rpc import cluster_pb2
from iris.rpc.errors import rpc_error_handler


class TaskProvider(Protocol):
    """Protocol for task management operations."""

    def submit_task(self, request: cluster_pb2.Worker.RunTaskRequest) -> str: ...
    def get_task(self, task_id: str) -> Task | None: ...
    def list_tasks(self) -> list[Task]: ...
    def kill_task(self, task_id: str, term_timeout_ms: int = 5000) -> bool: ...
    def get_logs(self, task_id: str, start_line: int = 0) -> list[cluster_pb2.Worker.LogEntry]: ...


class WorkerServiceImpl:
    """Implementation of WorkerService RPC interface."""

    def __init__(self, provider: TaskProvider):
        self._provider = provider
        self._start_time = time.time()

    def run_task(
        self,
        request: cluster_pb2.Worker.RunTaskRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.Worker.RunTaskResponse:
        """Start execution of a task."""
        with rpc_error_handler("running task"):
            task_id = self._provider.submit_task(request)
            task = self._provider.get_task(task_id)

            if not task:
                raise ConnectError(Code.INTERNAL, f"Task {task_id} not found after submission")

            return cluster_pb2.Worker.RunTaskResponse(
                task_id=task_id,
                state=task.to_proto().state,
            )

    def get_task_status(
        self,
        request: cluster_pb2.Worker.GetTaskStatusRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.TaskStatus:
        """Get status of a task."""
        task = self._provider.get_task(request.task_id)
        if not task:
            raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found")

        status = task.to_proto()
        if request.include_result and task.result:
            # TaskStatus doesn't have serialized_result field, but we could add it
            # For now, result is available via the task object
            pass
        return status

    def list_tasks(
        self,
        _request: cluster_pb2.Worker.ListTasksRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.Worker.ListTasksResponse:
        """List all tasks on this worker."""
        tasks = self._provider.list_tasks()
        return cluster_pb2.Worker.ListTasksResponse(
            tasks=[task.to_proto() for task in tasks],
        )

    def fetch_task_logs(
        self,
        request: cluster_pb2.Worker.FetchTaskLogsRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.Worker.FetchTaskLogsResponse:
        """Fetch logs for a task."""
        start_line = request.filter.start_line if request.filter.start_line else 0
        logs = self._provider.get_logs(request.task_id, start_line=start_line)

        # Apply additional filters
        result = []
        for entry in logs:
            # Time range filter (start_ms is exclusive for incremental polling)
            if request.filter.start_ms and entry.timestamp_ms <= request.filter.start_ms:
                continue
            if request.filter.end_ms and entry.timestamp_ms > request.filter.end_ms:
                continue
            # TODO: Regex filter is vulnerable to DoS via catastrophic backtracking.
            # Malicious regex like (a+)+ can cause minutes of CPU time. Consider using
            # the re2 library or adding timeout/complexity limits.
            # Regex filter
            if request.filter.regex:
                if not re.search(request.filter.regex, entry.data):
                    continue

            result.append(entry)

            # Max lines limit
            if request.filter.max_lines and len(result) >= request.filter.max_lines:
                break

        return cluster_pb2.Worker.FetchTaskLogsResponse(logs=result)

    def kill_task(
        self,
        request: cluster_pb2.Worker.KillTaskRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.Empty:
        """Kill a running task."""
        task = self._provider.get_task(request.task_id)
        if not task:
            raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found")

        success = self._provider.kill_task(
            request.task_id,
            term_timeout_ms=request.term_timeout_ms or 5000,
        )
        if not success:
            # Task exists but is already in terminal state
            state_name = cluster_pb2.TaskState.Name(task.status)
            raise ConnectError(
                Code.FAILED_PRECONDITION,
                f"Task {request.task_id} already completed with state {state_name}",
            )
        return cluster_pb2.Empty()

    def health_check(
        self,
        _request: cluster_pb2.Empty,
        _ctx: RequestContext,
    ) -> cluster_pb2.Worker.HealthResponse:
        """Report worker health."""
        tasks = self._provider.list_tasks()
        running = sum(1 for t in tasks if t.status == cluster_pb2.TASK_STATE_RUNNING)

        return cluster_pb2.Worker.HealthResponse(
            healthy=True,
            uptime_ms=int((time.time() - self._start_time) * 1000),
            running_tasks=running,
        )
