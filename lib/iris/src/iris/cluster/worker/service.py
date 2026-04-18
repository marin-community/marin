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
from iris.cluster.log_store import LogStore
from iris.cluster.process_status import get_process_status as _get_process_status
from iris.cluster.runtime.profile import is_system_target, parse_profile_target, profile_local_process
from iris.cluster.worker.worker_types import TaskInfo
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
    def profile_task(
        self, task_id: str, duration_seconds: int, profile_type: cluster_pb2.ProfileType, attempt_id: int | None = None
    ) -> bytes: ...


class WorkerServiceImpl:
    """Implementation of WorkerService RPC interface."""

    def __init__(
        self,
        provider: TaskProvider,
        log_store: LogStore | None = None,
    ):
        self._provider = provider
        self._log_store = log_store
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

    def fetch_logs(
        self,
        request: cluster_pb2.FetchLogsRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.FetchLogsResponse:
        """Fetch logs from the worker's LogStore by key with filtering."""
        if not self._log_store:
            return cluster_pb2.FetchLogsResponse(entries=[], cursor=0)

        max_lines = request.max_lines if request.max_lines > 0 else 1000
        result = self._log_store.get_logs(
            request.source,
            since_ms=request.since_ms,
            cursor=request.cursor,
            substring_filter=request.substring,
            max_lines=max_lines,
            tail=request.tail,
            min_level=request.min_level,
        )
        return cluster_pb2.FetchLogsResponse(entries=result.entries, cursor=result.cursor)

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

    def get_process_status(
        self,
        request: cluster_pb2.GetProcessStatusRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.GetProcessStatusResponse:
        """Return local process info and recent process logs."""
        return _get_process_status(request, self._log_store, self._timer)

    def profile_task(
        self,
        request: cluster_pb2.ProfileTaskRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.ProfileTaskResponse:
        """Profile a running task or the worker process itself.

        The target field determines what to profile:
        - /system/process: the worker process itself
        - /job/.../task/N:A: a specific task attempt (delegated to TaskProvider)
        """
        with rpc_error_handler("profile_task"):
            try:
                if not request.HasField("profile_type"):
                    raise ValueError("profile_type is required")

                duration = request.duration_seconds or 10

                # /system/process: profile the worker process itself using py-spy/memray
                if is_system_target(request.target):
                    data = profile_local_process(duration, request.profile_type)
                    return cluster_pb2.ProfileTaskResponse(profile_data=data)

                # Task target: parse optional :attempt_id and delegate to the container handle
                target = parse_profile_target(request.target)
                data = self._provider.profile_task(
                    target.task_id.to_wire(),
                    duration_seconds=duration,
                    profile_type=request.profile_type,
                    attempt_id=target.attempt_id,
                )
                return cluster_pb2.ProfileTaskResponse(profile_data=data)
            except Exception as e:
                return cluster_pb2.ProfileTaskResponse(error=str(e))
