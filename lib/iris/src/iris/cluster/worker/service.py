# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""WorkerService RPC implementation using Connect RPC."""

import logging
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
from iris.time_utils import Timer

logger = logging.getLogger(__name__)


class TaskProvider(Protocol):
    """Protocol for task management operations.

    Returns TaskInfo (read-only view) to decouple service layer from TaskAttempt internals.
    """

    def submit_task(self, request: cluster_pb2.Worker.RunTaskRequest) -> str: ...
    def get_task(self, task_id: str, attempt_id: int = -1) -> TaskInfo | None: ...
    def list_tasks(self) -> list[TaskInfo]: ...
    def kill_task(self, task_id: str, term_timeout_ms: int = 5000) -> bool: ...
    def handle_heartbeat(self, request: cluster_pb2.HeartbeatRequest) -> cluster_pb2.HeartbeatResponse: ...
    def profile_task(self, task_id: str, duration_seconds: int, profile_type: cluster_pb2.ProfileType) -> bytes: ...


class WorkerServiceImpl:
    """Implementation of WorkerService RPC interface."""

    def __init__(self, provider: TaskProvider, log_buffer: LogBuffer | None = None):
        self._provider = provider
        self._log_buffer = log_buffer
        self._timer = Timer()

    def get_task_status(
        self,
        request: cluster_pb2.Worker.GetTaskStatusRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.TaskStatus:
        """Get status of a task."""
        task = self._provider.get_task(request.task_id)
        if not task:
            raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found")

        return task.to_proto()

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

    def health_check(
        self,
        _request: cluster_pb2.Empty,
        _ctx: RequestContext,
    ) -> cluster_pb2.Worker.HealthResponse:
        """Report worker health."""
        tasks = self._provider.list_tasks()
        running = sum(1 for t in tasks if t.status == cluster_pb2.TASK_STATE_RUNNING)

        response = cluster_pb2.Worker.HealthResponse(
            healthy=True,
            running_tasks=running,
        )
        response.uptime.milliseconds = self._timer.elapsed_ms()
        return response

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

            # Delegate to worker for reconciliation
            return self._provider.handle_heartbeat(request)

    def profile_task(
        self,
        request: cluster_pb2.ProfileTaskRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.ProfileTaskResponse:
        """Profile a running task using py-spy (CPU) or memray (memory)."""
        with rpc_error_handler("profile_task"):
            try:
                # Validate profile_type
                if not request.HasField("profile_type"):
                    raise ValueError("profile_type is required")

                data = self._provider.profile_task(
                    request.task_id,
                    duration_seconds=request.duration_seconds or 10,
                    profile_type=request.profile_type,
                )
                return cluster_pb2.ProfileTaskResponse(profile_data=data)
            except Exception as e:
                return cluster_pb2.ProfileTaskResponse(error=str(e))
