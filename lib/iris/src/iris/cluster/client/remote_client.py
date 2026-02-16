# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RPC-based cluster client implementation."""

import logging
import time
from typing import TypeVar
from collections.abc import Callable

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.cluster.runtime.entrypoint import build_runtime_entrypoint
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, is_job_finished
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.time_utils import Deadline, Duration, ExponentialBackoff

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RemoteClusterClient:
    """Cluster client via RPC to controller.

    All parameters are explicit, no context magic. Takes full job IDs, full endpoint names, etc.
    """

    def __init__(
        self,
        controller_address: str,
        bundle_gcs_path: str | None = None,
        bundle_blob: bytes | None = None,
        timeout_ms: int = 30000,
    ):
        """Initialize RPC cluster operations.

        Args:
            controller_address: Controller URL (e.g., "http://localhost:8080")
            bundle_gcs_path: GCS path to workspace bundle for job inheritance
            bundle_blob: Workspace bundle as bytes (for initial job submission)
            timeout_ms: RPC timeout in milliseconds
        """
        self._address = controller_address
        self._bundle_gcs_path = bundle_gcs_path
        self._bundle_blob = bundle_blob
        self._timeout_ms = timeout_ms
        self._client = ControllerServiceClientSync(
            address=controller_address,
            timeout_ms=timeout_ms,
        )

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Check if an error should trigger retry.

        Retries on:
        - ConnectError with Code.UNAVAILABLE (controller temporarily down)
        - ConnectError with Code.INTERNAL (network errors bubble up as INTERNAL)

        Does not retry on:
        - Application errors (NOT_FOUND, INVALID_ARGUMENT, ALREADY_EXISTS, etc.)
        - These indicate issues with the request itself, not transient failures
        """
        if isinstance(exc, ConnectError):
            return exc.code in (Code.UNAVAILABLE, Code.INTERNAL)
        return False

    def _call_with_retry(
        self,
        operation: str,
        call_fn: Callable[[], T],
        *,
        max_attempts: int = 5,
        initial_backoff: float = 0.1,
        max_backoff: float = 5.0,
        backoff_factor: float = 2.0,
    ) -> T:
        """Execute an RPC call with exponential backoff retry.

        Args:
            operation: Description of the operation for logging
            call_fn: Callable that performs the RPC
            max_attempts: Maximum number of attempts (default: 5)
            initial_backoff: Initial retry delay in seconds (default: 0.1)
            max_backoff: Maximum delay between retries (default: 5.0)
            backoff_factor: Exponential backoff multiplier (default: 2.0)

        Returns:
            Result from call_fn

        Raises:
            Exception from call_fn if all retries exhausted or error is not retryable
        """
        backoff = ExponentialBackoff(
            initial=initial_backoff,
            maximum=max_backoff,
            factor=backoff_factor,
        )
        last_exception = None

        for attempt in range(max_attempts):
            try:
                return call_fn()
            except Exception as e:
                last_exception = e
                if not self._is_retryable_error(e):
                    # Non-retryable error, fail immediately
                    raise

                if attempt + 1 >= max_attempts:
                    # Final attempt failed, raise
                    logger.warning(
                        "Operation %s failed after %d attempts: %s",
                        operation,
                        max_attempts,
                        e,
                    )
                    raise

                # Log and retry
                delay = backoff.next_interval()
                if attempt == 0:
                    # First retry: log at INFO to make it visible
                    logger.info(
                        "Operation %s failed (attempt %d/%d), retrying in %.2fs: %s",
                        operation,
                        attempt + 1,
                        max_attempts,
                        delay,
                        e,
                    )
                else:
                    # Subsequent retries: log at DEBUG to reduce noise
                    logger.debug(
                        "Operation %s failed (attempt %d/%d), retrying in %.2fs: %s",
                        operation,
                        attempt + 1,
                        max_attempts,
                        delay,
                        e,
                    )
                time.sleep(delay)

        # Should not reach here due to raise in loop, but satisfy type checker
        assert last_exception is not None
        raise last_exception

    def submit_job(
        self,
        job_id: JobName,
        entrypoint: Entrypoint,
        resources: cluster_pb2.ResourceSpecProto,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        ports: list[str] | None = None,
        scheduling_timeout: Duration | None = None,
        constraints: list[cluster_pb2.Constraint] | None = None,
        coscheduling: cluster_pb2.CoschedulingConfig | None = None,
        replicas: int = 1,
        max_retries_failure: int = 0,
        max_retries_preemption: int = 100,
        timeout: Duration | None = None,
        fail_if_exists: bool = False,
    ) -> None:
        if replicas < 1:
            raise ValueError(f"replicas must be >= 1, got {replicas}")

        if environment is None:
            environment = EnvironmentSpec().to_proto()
        env_config = environment

        runtime_ep = build_runtime_entrypoint(entrypoint, env_config)

        # Use bundle_gcs_path if available, otherwise use bundle_blob
        if self._bundle_gcs_path:
            request = cluster_pb2.Controller.LaunchJobRequest(
                name=job_id.to_wire(),
                entrypoint=runtime_ep,
                resources=resources,
                environment=env_config,
                bundle_gcs_path=self._bundle_gcs_path,
                ports=ports or [],
                constraints=constraints or [],
                replicas=replicas,
                max_retries_failure=max_retries_failure,
                max_retries_preemption=max_retries_preemption,
                fail_if_exists=fail_if_exists,
            )
        else:
            request = cluster_pb2.Controller.LaunchJobRequest(
                name=job_id.to_wire(),
                entrypoint=runtime_ep,
                resources=resources,
                environment=env_config,
                bundle_blob=self._bundle_blob or b"",
                ports=ports or [],
                constraints=constraints or [],
                replicas=replicas,
                max_retries_failure=max_retries_failure,
                max_retries_preemption=max_retries_preemption,
                fail_if_exists=fail_if_exists,
            )

        if scheduling_timeout is not None:
            request.scheduling_timeout.CopyFrom(scheduling_timeout.to_proto())
        if timeout is not None:
            request.timeout.CopyFrom(timeout.to_proto())
        if coscheduling is not None:
            request.coscheduling.CopyFrom(coscheduling)
        self._client.launch_job(request)

    def get_job_status(self, job_id: JobName) -> cluster_pb2.JobStatus:
        def _call():
            request = cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id.to_wire())
            response = self._client.get_job_status(request)
            return response.job

        return self._call_with_retry(f"get_job_status({job_id})", _call)

    def wait_for_job(
        self,
        job_id: JobName,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> cluster_pb2.JobStatus:
        """Wait for job to complete with exponential backoff polling.

        Args:
            job_id: Full job ID
            timeout: Maximum time to wait in seconds
            poll_interval: Maximum time between status checks

        Returns:
            Final JobStatus

        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        deadline = Deadline.from_seconds(timeout)
        backoff = ExponentialBackoff(initial=0.1, maximum=poll_interval)

        while True:
            job_info = self.get_job_status(job_id)
            if is_job_finished(job_info.state):
                return job_info

            if deadline.expired():
                raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")

            interval = backoff.next_interval()
            time.sleep(min(interval, deadline.remaining_seconds()))

    def terminate_job(self, job_id: JobName) -> None:
        request = cluster_pb2.Controller.TerminateJobRequest(job_id=job_id.to_wire())
        self._client.terminate_job(request)

    def register_endpoint(
        self,
        name: str,
        address: str,
        job_id: JobName,
        metadata: dict[str, str] | None = None,
    ) -> str:
        request = cluster_pb2.Controller.RegisterEndpointRequest(
            name=name,
            address=address,
            job_id=job_id.to_wire(),
            metadata=metadata or {},
        )
        response = self._client.register_endpoint(request)
        return response.endpoint_id

    def unregister_endpoint(self, endpoint_id: str) -> None:
        """Unregister an endpoint via RPC.

        This is a no-op for the RPC implementation. The controller automatically
        cleans up endpoints when jobs terminate, so explicit unregistration
        is not required.

        Args:
            endpoint_id: Endpoint ID (ignored)
        """
        # No-op: controller auto-cleans endpoints on job termination
        del endpoint_id

    def list_endpoints(self, prefix: str) -> list[cluster_pb2.Controller.Endpoint]:
        request = cluster_pb2.Controller.ListEndpointsRequest(prefix=prefix)
        response = self._client.list_endpoints(request)
        return list(response.endpoints)

    def list_jobs(self) -> list[cluster_pb2.JobStatus]:
        def _call():
            request = cluster_pb2.Controller.ListJobsRequest()
            response = self._client.list_jobs(request)
            return list(response.jobs)

        return self._call_with_retry("list_jobs", _call)

    def shutdown(self, wait: bool = True) -> None:
        del wait
        # No cleanup needed - controller client is managed separately

    def get_task_status(self, task_name: JobName) -> cluster_pb2.TaskStatus:
        """Get status of a specific task within a job.

        Args:
            task_name: Full task name (/job/.../index)

        Returns:
            TaskStatus proto for the requested task
        """
        task_name.require_task()

        def _call():
            request = cluster_pb2.Controller.GetTaskStatusRequest(task_id=task_name.to_wire())
            response = self._client.get_task_status(request)
            return response.task

        return self._call_with_retry(f"get_task_status({task_name})", _call)

    def list_tasks(self, job_id: JobName) -> list[cluster_pb2.TaskStatus]:
        """List all tasks for a job.

        Args:
            job_id: Job ID to query tasks for

        Returns:
            List of TaskStatus protos, one per task in the job
        """

        def _call():
            request = cluster_pb2.Controller.ListTasksRequest(job_id=job_id.to_wire())
            response = self._client.list_tasks(request)
            return list(response.tasks)

        return self._call_with_retry(f"list_tasks({job_id})", _call)

    def fetch_task_logs(
        self,
        target: JobName,
        *,
        include_children: bool = False,
        since_ms: int = 0,
        max_total_lines: int = 0,
        regex: str | None = None,
        attempt_id: int = -1,
    ) -> cluster_pb2.Controller.GetTaskLogsResponse:
        """Fetch logs for a task or all tasks in a job.

        Args:
            target: Task ID or Job ID (detected by trailing numeric)
            include_children: Include logs from child jobs (job ID only)
            since_ms: Only return logs after this timestamp (exclusive)
            max_total_lines: Maximum total lines (0 = default 10000)
            regex: Regex filter for log content
            attempt_id: Filter to specific attempt (-1 = all attempts)
        """

        def _call():
            request = cluster_pb2.Controller.GetTaskLogsRequest(
                id=target.to_wire(),
                include_children=include_children,
                since_ms=since_ms,
                max_total_lines=max_total_lines,
                regex=regex or "",
                attempt_id=attempt_id,
            )
            return self._client.get_task_logs(request)

        return self._call_with_retry(f"fetch_task_logs({target})", _call)

    def get_autoscaler_status(self) -> cluster_pb2.Controller.GetAutoscalerStatusResponse:
        """Get autoscaler status including recent actions and group states.

        Returns:
            GetAutoscalerStatusResponse proto with autoscaler status and recent actions
        """
        request = cluster_pb2.Controller.GetAutoscalerStatusRequest()
        return self._client.get_autoscaler_status(request)
