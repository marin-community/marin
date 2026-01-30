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

import logging
import re
import time
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from iris.chaos import chaos
from iris.cluster.worker.worker_types import TaskInfo
from iris.logging import LogBuffer
from iris.rpc import cluster_pb2
from iris.rpc.errors import rpc_error_handler

logger = logging.getLogger(__name__)


class TaskProvider(Protocol):
    """Protocol for task management operations.

    Returns TaskInfo (read-only view) to decouple service layer from TaskAttempt internals.
    """

    def submit_task(self, request: cluster_pb2.Worker.RunTaskRequest) -> str: ...
    def get_task(self, task_id: str) -> TaskInfo | None: ...
    def list_tasks(self) -> list[TaskInfo]: ...
    def kill_task(self, task_id: str, term_timeout_ms: int = 5000) -> bool: ...
    def get_logs(self, task_id: str, start_line: int = 0) -> list[cluster_pb2.Worker.LogEntry]: ...
    def pop_completed_tasks(self) -> list[cluster_pb2.Controller.CompletedTaskEntry]: ...
    def on_heartbeat_received(self) -> None: ...


class WorkerServiceImpl:
    """Implementation of WorkerService RPC interface."""

    def __init__(self, provider: TaskProvider, log_buffer: LogBuffer | None = None):
        self._provider = provider
        self._log_buffer = log_buffer
        self._start_time = time.time()

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

    def get_process_logs(
        self,
        request: cluster_pb2.Worker.GetProcessLogsRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.Worker.GetProcessLogsResponse:
        """Get worker process logs from the in-memory ring buffer."""
        if not self._log_buffer:
            return cluster_pb2.Worker.GetProcessLogsResponse(records=[])
        prefix = request.prefix or None
        limit = request.limit if request.limit > 0 else 200
        records = self._log_buffer.query(prefix=prefix, limit=limit)
        return cluster_pb2.Worker.GetProcessLogsResponse(
            records=[
                cluster_pb2.ProcessLogRecord(
                    timestamp=r.timestamp,
                    level=r.level,
                    logger_name=r.logger_name,
                    message=r.message,
                )
                for r in records
            ]
        )

    def heartbeat(
        self,
        request: cluster_pb2.HeartbeatRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.HeartbeatResponse:
        """Handle controller-initiated heartbeat.

        Processes tasks_to_run and tasks_to_kill, then returns current state.
        """
        with rpc_error_handler("heartbeat"):
            # Chaos injection for testing heartbeat failures and delays
            if rule := chaos("worker.heartbeat"):
                if rule.delay_seconds > 0:
                    time.sleep(rule.delay_seconds)
                if rule.error:
                    raise rule.error
                # If no error specified, raise generic RuntimeError
                if not rule.delay_seconds:
                    raise RuntimeError("chaos: worker.heartbeat")

            # Update heartbeat timestamp first
            self._provider.on_heartbeat_received()

            # Start new tasks
            for run_req in request.tasks_to_run:
                try:
                    self._provider.submit_task(run_req)
                    logger.info(f"Heartbeat: submitted task {run_req.task_id}")
                except Exception as e:
                    logger.warning(f"Heartbeat: failed to submit task {run_req.task_id}: {e}")

            # Kill requested tasks
            for task_id in request.tasks_to_kill:
                try:
                    self._provider.kill_task(task_id)
                    logger.info(f"Heartbeat: killed task {task_id}")
                except Exception as e:
                    logger.warning(f"Heartbeat: failed to kill task {task_id}: {e}")

            # Build response with current running tasks
            tasks = self._provider.list_tasks()
            running = []
            for t in tasks:
                proto = t.to_proto()
                if proto.state in (cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_BUILDING):
                    running.append(
                        cluster_pb2.Controller.RunningTaskEntry(
                            task_id=proto.task_id,
                            attempt_id=proto.current_attempt_id,
                        )
                    )

            # Completed tasks come from the worker's unreported buffer
            completed = self._provider.pop_completed_tasks()

            return cluster_pb2.HeartbeatResponse(
                running_tasks=running,
                completed_tasks=completed,
            )
