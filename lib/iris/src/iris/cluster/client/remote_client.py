# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RPC-based cluster client implementation."""

import logging
import time
import uuid
from pathlib import Path

from collections.abc import Iterable

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.interceptor import InterceptorSync

from iris.cluster.client.bundle import BundleCreator
from iris.cluster.log_store._types import build_log_source
from iris.cluster.runtime.entrypoint import build_runtime_entrypoint
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, TaskAttempt, adjust_tpu_replicas, is_job_finished
from iris.rpc import logging_pb2
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.rpc.logging_connect import LogServiceClientSync
from iris.rpc.errors import call_with_retry, format_connect_error, poll_with_retries
from iris.time_proto import duration_to_proto
from iris.version import client_revision_date
from rigging.timing import Deadline, Duration, ExponentialBackoff

logger = logging.getLogger(__name__)

# How long to tolerate controller unavailability before giving up on monitoring.
# The job itself keeps running server-side; this only affects the client's ability
# to poll status. One hour gives ample time for controller restarts/upgrades.
CONTROLLER_UNAVAILABLE_TOLERANCE = 3600.0

# Upper bound on GetJobState polling cadence for long-running jobs. The loop
# ramps 100ms -> 1s within a handful of polls (factor=1.5 in ExponentialBackoff)
# and then caps here, so long jobs cost ~1 state RPC / 30s instead of hammering
# the controller at the old ~2s ceiling.
MAX_STATE_POLL_INTERVAL = 30.0

# Floor on the backoff cap. ``ExponentialBackoff`` requires ``maximum >= initial``
# (currently 100ms), so we clamp the caller-supplied ``poll_interval`` up to this
# value before handing it to the backoff. Callers asking for a sub-100ms cap end
# up polling at 100ms instead of crashing with ValueError.
MIN_STATE_POLL_INTERVAL = 0.1


class RemoteClusterClient:
    """Cluster client via RPC to controller.

    All parameters are explicit, no context magic. Takes full job IDs, full endpoint names, etc.
    """

    def __init__(
        self,
        controller_address: str,
        bundle_id: str | None = None,
        workspace: Path | None = None,
        timeout_ms: int = 30000,
        interceptors: Iterable[InterceptorSync] = (),
    ):
        """Initialize RPC cluster operations.

        Args:
            controller_address: Controller URL (e.g., "http://localhost:8080")
            bundle_id: Workspace bundle identifier for job inheritance
            workspace: Path to workspace directory. Bundle is created lazily on first job submission.
            timeout_ms: RPC timeout in milliseconds
            interceptors: Client-side interceptors (e.g. AuthTokenInjector for token auth)
        """
        self._address = controller_address
        self._bundle_id = bundle_id
        self._workspace = workspace.resolve() if workspace is not None else None
        self._bundle_blob: bytes | None = None
        self._timeout_ms = timeout_ms
        self._client = ControllerServiceClientSync(
            address=controller_address,
            timeout_ms=timeout_ms,
            interceptors=interceptors,
        )
        self._log_client = LogServiceClientSync(
            address=controller_address,
            timeout_ms=timeout_ms,
            interceptors=interceptors,
        )

    def submit_job(
        self,
        job_id: JobName,
        entrypoint: Entrypoint,
        resources: job_pb2.ResourceSpecProto,
        environment: job_pb2.EnvironmentConfig | None = None,
        ports: list[str] | None = None,
        scheduling_timeout: Duration | None = None,
        constraints: list[job_pb2.Constraint] | None = None,
        coscheduling: job_pb2.CoschedulingConfig | None = None,
        replicas: int = 1,
        max_retries_failure: int = 0,
        max_retries_preemption: int = 1000,
        timeout: Duration | None = None,
        reservation: job_pb2.ReservationConfig | None = None,
        preemption_policy: job_pb2.JobPreemptionPolicy = job_pb2.JOB_PREEMPTION_POLICY_UNSPECIFIED,
        existing_job_policy: job_pb2.ExistingJobPolicy = job_pb2.EXISTING_JOB_POLICY_UNSPECIFIED,
        task_image: str | None = None,
        priority_band: job_pb2.PriorityBand = job_pb2.PRIORITY_BAND_UNSPECIFIED,
        submit_argv: list[str] | None = None,
    ) -> JobName:
        if replicas < 1:
            raise ValueError(f"replicas must be >= 1, got {replicas}")
        replicas = adjust_tpu_replicas(resources.device if resources.HasField("device") else None, replicas)

        if environment is None:
            environment = EnvironmentSpec().to_proto()
        env_config = environment

        runtime_ep = build_runtime_entrypoint(entrypoint, env_config)

        request = controller_pb2.Controller.LaunchJobRequest(
            name=job_id.to_wire(),
            entrypoint=runtime_ep,
            resources=resources,
            environment=env_config,
            ports=ports or [],
            constraints=constraints or [],
            replicas=replicas,
            max_retries_failure=max_retries_failure,
            max_retries_preemption=max_retries_preemption,
            preemption_policy=preemption_policy,
            existing_job_policy=existing_job_policy,
            task_image=task_image or "",
            priority_band=priority_band,
            submit_argv=submit_argv or [],
            client_revision_date=client_revision_date(),
        )
        if self._bundle_id:
            request.bundle_id = self._bundle_id
        else:
            if self._bundle_blob is None and self._workspace is not None:
                creator = BundleCreator(self._workspace)
                self._bundle_blob = creator.create_bundle()
                logger.info(f"Workspace bundle size: {len(self._bundle_blob) / 1024 / 1024:.1f} MB")
            request.bundle_blob = self._bundle_blob or b""

        if scheduling_timeout is not None:
            request.scheduling_timeout.CopyFrom(duration_to_proto(scheduling_timeout))
        if timeout is not None:
            request.timeout.CopyFrom(duration_to_proto(timeout))
        if coscheduling is not None:
            request.coscheduling.CopyFrom(coscheduling)
        if reservation is not None:
            request.reservation.CopyFrom(reservation)

        def _call():
            return self._client.launch_job(request)

        response = call_with_retry(f"launch_job({job_id})", _call)
        return JobName.from_wire(response.job_id)

    def get_job_status(self, job_id: JobName) -> job_pb2.JobStatus:
        def _call():
            request = controller_pb2.Controller.GetJobStatusRequest(job_id=job_id.to_wire())
            response = self._client.get_job_status(request)
            return response.job

        return call_with_retry(f"get_job_status({job_id})", _call)

    def get_job_states(self, job_ids: list[JobName]) -> dict[str, int]:
        """Lightweight batch query returning only the state enum per job."""

        def _call():
            request = controller_pb2.Controller.GetJobStateRequest(
                job_ids=[jid.to_wire() for jid in job_ids],
            )
            response = self._client.get_job_state(request)
            return dict(response.states)

        return call_with_retry(f"get_job_states({len(job_ids)} jobs)", _call)

    def _poll_job_state(self, job_id: JobName) -> int:
        """Fetch only the state enum for a single job via the lightweight RPC."""
        states = self.get_job_states([job_id])
        wire_id = job_id.to_wire()
        if wire_id not in states:
            raise ConnectError(Code.NOT_FOUND, f"Job {wire_id} not found")
        return states[wire_id]

    def wait_for_job(
        self,
        job_id: JobName,
        timeout: float = 300.0,
        poll_interval: float = MAX_STATE_POLL_INTERVAL,
    ) -> job_pb2.JobStatus:
        """Wait for job to complete with exponential backoff polling.

        If the controller becomes unavailable, retries with backoff for up to
        ``CONTROLLER_UNAVAILABLE_TOLERANCE`` seconds or until the caller's
        *timeout* expires — whichever comes first. The unavailability timer
        resets each time a status check succeeds.

        Args:
            job_id: Full job ID
            timeout: Maximum time to wait in seconds
            poll_interval: Upper bound on the state-poll backoff. The loop
                starts at 100ms and grows exponentially until reaching this
                cap.

        Returns:
            Final JobStatus

        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        deadline = Deadline.from_seconds(timeout)
        backoff = ExponentialBackoff(
            initial=MIN_STATE_POLL_INTERVAL,
            maximum=max(poll_interval, MIN_STATE_POLL_INTERVAL),
        )

        while True:
            # Poll with lightweight state-only RPC during the loop.
            state = poll_with_retries(
                str(job_id),
                lambda: self._poll_job_state(job_id),
                deadline=deadline,
                unavailable_tolerance=CONTROLLER_UNAVAILABLE_TOLERANCE,
            )

            if is_job_finished(state):
                # Fetch full status once at the end for error details.
                return poll_with_retries(
                    str(job_id),
                    lambda: self.get_job_status(job_id),
                    deadline=deadline,
                    unavailable_tolerance=CONTROLLER_UNAVAILABLE_TOLERANCE,
                )

            if deadline.expired():
                raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")

            interval = backoff.next_interval()
            time.sleep(min(interval, deadline.remaining_seconds()))

    def wait_for_job_with_streaming(
        self,
        job_id: JobName,
        *,
        timeout: float,
        poll_interval: float = MAX_STATE_POLL_INTERVAL,
        since_ms: int = 0,
        min_level: str = "",
    ) -> job_pb2.JobStatus:
        """Wait for job completion while streaming task logs via the controller RPC.

        Delegates log reading to the controller (which has the correct storage
        credentials and endpoint configuration), avoiding client-side S3 access.

        ``poll_interval`` caps the state-poll backoff; the loop starts at 100ms
        and grows exponentially until reaching that bound, matching
        :py:meth:`wait_for_job`.

        If the controller becomes unavailable, retries with backoff for up to
        ``CONTROLLER_UNAVAILABLE_TOLERANCE`` seconds or until the caller's
        *timeout* expires -- whichever comes first. Log fetch failures are
        non-fatal -- they log a warning but never abort monitoring.
        """
        deadline = Deadline.from_seconds(timeout)
        terminal_status: job_pb2.JobStatus | None = None
        source = build_log_source(job_id)
        cursor: int = 0
        backoff = ExponentialBackoff(
            initial=MIN_STATE_POLL_INTERVAL,
            maximum=max(poll_interval, MIN_STATE_POLL_INTERVAL),
        )

        while True:
            # Poll with lightweight state-only RPC during the loop.
            state = poll_with_retries(
                str(job_id),
                lambda: self._poll_job_state(job_id),
                deadline=deadline,
                unavailable_tolerance=CONTROLLER_UNAVAILABLE_TOLERANCE,
            )

            state_name = job_pb2.JobState.Name(state)

            try:
                log_response = self.fetch_logs(source, since_ms=since_ms, cursor=cursor, min_level=min_level)
            except Exception as e:
                msg = format_connect_error(e) if isinstance(e, ConnectError) else str(e)
                logger.warning("Failed to fetch logs for %s, will retry: %s", job_id, msg)
                log_response = None

            if log_response is not None:
                for entry in log_response.entries:
                    key = entry.key or source
                    logger.info("task=%s | %s", key, entry.data)

                if log_response.cursor > cursor:
                    cursor = log_response.cursor

            if is_job_finished(state):
                total_lines = len(log_response.entries) if log_response else 0
                logger.info(
                    "job=%s finished with state=%s, draining logs (total_lines=%d)",
                    job_id,
                    state_name,
                    total_lines,
                )
                if terminal_status is not None:
                    return terminal_status
                # Fetch full status for error details on the final return.
                terminal_status = poll_with_retries(
                    str(job_id),
                    lambda: self.get_job_status(job_id),
                    deadline=deadline,
                    unavailable_tolerance=CONTROLLER_UNAVAILABLE_TOLERANCE,
                )
                time.sleep(1)
                continue

            deadline.raise_if_expired(f"Job {job_id} did not complete in {timeout}s")
            interval = backoff.next_interval()
            time.sleep(min(interval, deadline.remaining_seconds()))

    def terminate_job(self, job_id: JobName) -> None:
        request = controller_pb2.Controller.TerminateJobRequest(job_id=job_id.to_wire())
        self._client.terminate_job(request)

    def register_endpoint(
        self,
        name: str,
        address: str,
        task_attempt: TaskAttempt,
        metadata: dict[str, str] | None = None,
    ) -> str:
        endpoint_id = str(uuid.uuid4())
        request = controller_pb2.Controller.RegisterEndpointRequest(
            name=name,
            address=address,
            task_id=task_attempt.task_id.to_wire(),
            attempt_id=task_attempt.attempt_id if task_attempt.attempt_id is not None else 0,
            metadata=metadata or {},
            endpoint_id=endpoint_id,
        )

        def _call():
            return self._client.register_endpoint(request)

        response = call_with_retry("register_endpoint", _call)
        return response.endpoint_id

    def unregister_endpoint(self, endpoint_id: str) -> None:
        """Unregister an endpoint via RPC."""
        request = controller_pb2.Controller.UnregisterEndpointRequest(endpoint_id=endpoint_id)
        self._client.unregister_endpoint(request)

    def list_endpoints(self, prefix: str, *, exact: bool = False) -> list[controller_pb2.Controller.Endpoint]:
        def _call():
            request = controller_pb2.Controller.ListEndpointsRequest(prefix=prefix, exact=exact)
            response = self._client.list_endpoints(request, timeout_ms=10_000)
            return list(response.endpoints)

        return call_with_retry("list_endpoints", _call)

    def list_workers(self) -> list[controller_pb2.Controller.WorkerHealthStatus]:
        """List all workers registered with the controller."""

        def _call():
            request = controller_pb2.Controller.ListWorkersRequest()
            response = self._client.list_workers(request)
            return list(response.workers)

        return call_with_retry("list_workers", _call)

    def list_jobs(
        self,
        *,
        query: controller_pb2.Controller.JobQuery | None = None,
        page_size: int = 500,
    ) -> list[job_pb2.JobStatus]:
        """Fetch all jobs matching ``query`` by paging through ``ListJobs``.

        The server caps each page at ``MAX_LIST_JOBS_LIMIT``; this helper walks
        ``has_more`` / ``total_count`` to return the full result set without
        asking the controller for an unbounded scan.
        """
        if query is None:
            query = controller_pb2.Controller.JobQuery()

        jobs: list[job_pb2.JobStatus] = []
        offset = query.offset or 0
        while True:
            page_query = controller_pb2.Controller.JobQuery()
            page_query.CopyFrom(query)
            page_query.offset = offset
            page_query.limit = page_size

            def _call(q=page_query):
                request = controller_pb2.Controller.ListJobsRequest(query=q)
                return self._client.list_jobs(request)

            response = call_with_retry("list_jobs", _call)
            jobs.extend(response.jobs)
            if not response.has_more or not response.jobs:
                return jobs
            offset += len(response.jobs)

    def shutdown(self, wait: bool = True) -> None:
        del wait
        self._log_client.close()
        self._client.close()

    def get_task_status(self, task_name: JobName) -> job_pb2.TaskStatus:
        """Get status of a specific task within a job.

        Args:
            task_name: Full task name (/job/.../index)

        Returns:
            TaskStatus proto for the requested task
        """
        task_name.require_task()

        def _call():
            request = controller_pb2.Controller.GetTaskStatusRequest(task_id=task_name.to_wire())
            response = self._client.get_task_status(request)
            return response.task

        return call_with_retry(f"get_task_status({task_name})", _call)

    def list_tasks(self, job_id: JobName) -> list[job_pb2.TaskStatus]:
        """List all tasks for a job.

        Args:
            job_id: Job ID to query tasks for

        Returns:
            List of TaskStatus protos, one per task in the job
        """

        def _call():
            request = controller_pb2.Controller.ListTasksRequest(job_id=job_id.to_wire())
            response = self._client.list_tasks(request)
            return list(response.tasks)

        return call_with_retry(f"list_tasks({job_id})", _call)

    def fetch_logs(
        self,
        source: str,
        *,
        since_ms: int = 0,
        cursor: int = 0,
        max_lines: int = 0,
        substring: str = "",
        min_level: str = "",
        tail: bool = False,
    ) -> logging_pb2.FetchLogsResponse:
        request = logging_pb2.FetchLogsRequest(
            source=source,
            since_ms=since_ms,
            cursor=cursor,
            max_lines=max_lines,
            substring=substring,
            min_level=min_level,
            tail=tail,
        )

        def _call():
            return self._log_client.fetch_logs(request)

        return call_with_retry(f"fetch_logs({source})", _call)

    def get_autoscaler_status(self) -> controller_pb2.Controller.GetAutoscalerStatusResponse:
        """Get autoscaler status including recent actions and group states.

        Returns:
            GetAutoscalerStatusResponse proto with autoscaler status and recent actions
        """

        def _call():
            request = controller_pb2.Controller.GetAutoscalerStatusRequest()
            return self._client.get_autoscaler_status(request)

        return call_with_retry("get_autoscaler_status", _call)
