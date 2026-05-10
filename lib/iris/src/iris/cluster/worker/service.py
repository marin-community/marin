# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""WorkerService RPC implementation using Connect RPC."""

import logging
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from rigging.timing import Timer

from iris.cluster.process_status import get_process_status as _get_process_status
from iris.cluster.runtime.profile import ProfileTrigger
from iris.cluster.worker.worker_types import TaskInfo
from iris.rpc import job_pb2, worker_pb2
from iris.rpc.errors import rpc_error_handler

logger = logging.getLogger(__name__)


class TaskProvider(Protocol):
    """Protocol for task management operations.

    Returns TaskInfo (read-only view) to decouple service layer from TaskAttempt internals.
    """

    def submit_task(self, request: job_pb2.RunTaskRequest) -> str: ...
    def get_task(self, task_id: str, attempt_id: int = -1) -> TaskInfo | None: ...
    def list_tasks(self) -> list[TaskInfo]: ...
    def kill_task(self, task_id: str, term_timeout_ms: int = 5000) -> bool: ...
    def handle_ping(self, request: worker_pb2.Worker.PingRequest) -> worker_pb2.Worker.PingResponse: ...
    def handle_start_tasks(
        self, request: worker_pb2.Worker.StartTasksRequest
    ) -> worker_pb2.Worker.StartTasksResponse: ...
    def handle_stop_tasks(self, request: worker_pb2.Worker.StopTasksRequest) -> worker_pb2.Worker.StopTasksResponse: ...
    def handle_poll_tasks(self, request: worker_pb2.Worker.PollTasksRequest) -> worker_pb2.Worker.PollTasksResponse: ...
    def capture_and_log_profile(
        self,
        *,
        target: str,
        request: job_pb2.ProfileTaskRequest,
        trigger: ProfileTrigger,
    ) -> bytes: ...
    def exec_in_container(
        self, task_id: str, command: list[str], timeout_seconds: int = 60
    ) -> worker_pb2.Worker.ExecInContainerResponse: ...


class WorkerServiceImpl:
    """Implementation of WorkerService RPC interface."""

    def __init__(
        self,
        provider: TaskProvider,
    ):
        self._provider = provider
        self._timer = Timer()

    def get_task_status(
        self,
        request: worker_pb2.Worker.GetTaskStatusRequest,
        _ctx: RequestContext,
    ) -> job_pb2.TaskStatus:
        """Get status of a task."""
        task = self._provider.get_task(request.task_id)
        if not task:
            raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found")

        return task.to_proto()

    def list_tasks(
        self,
        _request: worker_pb2.Worker.ListTasksRequest,
        _ctx: RequestContext,
    ) -> worker_pb2.Worker.ListTasksResponse:
        """List all tasks on this worker."""
        tasks = self._provider.list_tasks()
        return worker_pb2.Worker.ListTasksResponse(
            tasks=[task.to_proto() for task in tasks],
        )

    def health_check(
        self,
        _request: job_pb2.Empty,
        _ctx: RequestContext,
    ) -> worker_pb2.Worker.HealthResponse:
        """Report worker health."""
        tasks = self._provider.list_tasks()
        running = sum(1 for t in tasks if t.status == job_pb2.TASK_STATE_RUNNING)

        response = worker_pb2.Worker.HealthResponse(
            healthy=True,
            running_tasks=running,
        )
        response.uptime.milliseconds = self._timer.elapsed_ms()
        return response

    def get_process_status(
        self,
        request: job_pb2.GetProcessStatusRequest,
        _ctx: RequestContext,
    ) -> job_pb2.GetProcessStatusResponse:
        """Return local process info (logs are in the central LogService)."""
        return _get_process_status(self._timer)

    def profile_task(
        self,
        request: job_pb2.ProfileTaskRequest,
        _ctx: RequestContext,
    ) -> job_pb2.ProfileTaskResponse:
        """Profile a running task or the worker process itself.

        The target field determines what to profile:
        - ``/system/process``: the worker process itself. The persisted row's
          ``source`` is rewritten to ``/system/worker/<id>``.
        - ``/job/.../task/N[:attempt_id]``: a specific task attempt.

        All captures (CPU, memory, threads) persist to ``iris.profile`` with
        ``trigger="on_demand"`` and the bytes are returned inline.
        """
        with rpc_error_handler("profile_task"):
            try:
                if not request.HasField("profile_type"):
                    raise ValueError("profile_type is required")
                data = self._provider.capture_and_log_profile(
                    target=request.target,
                    request=request,
                    trigger=ProfileTrigger.ON_DEMAND,
                )
                return job_pb2.ProfileTaskResponse(profile_data=data)
            except Exception as e:
                return job_pb2.ProfileTaskResponse(error=str(e))

    def exec_in_container(
        self,
        request: worker_pb2.Worker.ExecInContainerRequest,
        _ctx: RequestContext,
    ) -> worker_pb2.Worker.ExecInContainerResponse:
        """Execute a command in a running task's container."""
        with rpc_error_handler("exec_in_container"):
            if not request.command:
                raise ConnectError(Code.INVALID_ARGUMENT, "command is required")
            timeout_seconds = request.timeout_seconds if request.timeout_seconds != 0 else 60
            return self._provider.exec_in_container(request.task_id, list(request.command), timeout_seconds)

    def ping(self, request: worker_pb2.Worker.PingRequest, _ctx: RequestContext) -> worker_pb2.Worker.PingResponse:
        with rpc_error_handler("ping"):
            return self._provider.handle_ping(request)

    def start_tasks(
        self, request: worker_pb2.Worker.StartTasksRequest, _ctx: RequestContext
    ) -> worker_pb2.Worker.StartTasksResponse:
        with rpc_error_handler("start_tasks"):
            return self._provider.handle_start_tasks(request)

    def stop_tasks(
        self, request: worker_pb2.Worker.StopTasksRequest, _ctx: RequestContext
    ) -> worker_pb2.Worker.StopTasksResponse:
        with rpc_error_handler("stop_tasks"):
            return self._provider.handle_stop_tasks(request)

    def poll_tasks(
        self, request: worker_pb2.Worker.PollTasksRequest, _ctx: RequestContext
    ) -> worker_pb2.Worker.PollTasksResponse:
        with rpc_error_handler("poll_tasks"):
            return self._provider.handle_poll_tasks(request)
