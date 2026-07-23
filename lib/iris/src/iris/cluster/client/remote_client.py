# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RPC-based cluster client implementation."""

import logging
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.interceptor import InterceptorSync
from finelog.client import LogClient
from finelog.rpc import logging_pb2
from rigging.connect import proxy_path
from rigging.timing import Deadline, Duration, ExponentialBackoff

from iris.cluster.client.bundle import create_workspace_zip
from iris.cluster.client.endpoint_client import EndpointClient
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.log_keys import build_log_source
from iris.cluster.runtime.entrypoint import build_runtime_entrypoint
from iris.cluster.runtime.env import with_slice_topology_env
from iris.cluster.stats.tables import TASK_STATUS_NAMESPACE, TASK_STATUS_STORAGE_POLICY, TaskStatusRow
from iris.cluster.types import (
    EndpointAccess,
    Entrypoint,
    EnvironmentSpec,
    JobName,
    TaskAttempt,
    adjust_tpu_replicas,
    is_job_finished,
)
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.controller_connect import ControllerServiceClientSync, EndpointServiceClientSync
from iris.rpc.errors import call_with_retry, format_connect_error, poll_with_retries
from iris.time_proto import duration_to_proto
from iris.version import client_revision_date

logger = logging.getLogger(__name__)


# How long to tolerate controller unavailability before giving up on monitoring.
# The job itself keeps running server-side; this only affects the client's ability
# to poll status. One hour gives ample time for controller restarts/upgrades.
CONTROLLER_UNAVAILABLE_TOLERANCE = 3600.0

# Upper bound on GetJobState polling cadence for long-running jobs. The loop
# ramps 100ms -> 1s within a handful of polls (factor=1.5 in ExponentialBackoff)
# and then caps here, so long jobs cost ~1 state RPC / 30s.
MAX_STATE_POLL_INTERVAL = 30.0

# Floor on the backoff cap. ``ExponentialBackoff`` requires ``maximum >= initial``
# (currently 100ms), so callers asking for a sub-100ms cap are clamped to this
# value before being handed to the backoff.
MIN_STATE_POLL_INTERVAL = 0.1

# Floor on the per-call deadline for LaunchJob. The handler can legitimately
# run well past the default 30s client timeout when replacing a still-draining
# predecessor (_JOB_REPLACEMENT_DRAIN_WAIT is 120s on the controller) or when
# uploading a large workspace bundle. Setting this below the worst-case server
# budget guarantees a deadline timeout + retry, which then races the original
# in-flight INSERT and trips a UNIQUE constraint. Pad past 120s for bundle
# upload, autoscaler feasibility, and connection overhead. Bump alongside
# _JOB_REPLACEMENT_DRAIN_WAIT if that grows. Callers that configured a larger
# RemoteClusterClient timeout keep theirs — this is a floor, not a ceiling.
LAUNCH_JOB_TIMEOUT_FLOOR_MS = 180_000


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
        use_controller_proxy: bool = True,
        extra_bundle_includes: Sequence[str] = (),
    ):
        """Initialize RPC cluster operations.

        Args:
            controller_address: Controller URL (e.g., "http://localhost:8080")
            bundle_id: Workspace bundle identifier for job inheritance
            workspace: Path to workspace directory. Bundle is created lazily on first job submission.
            timeout_ms: RPC timeout in milliseconds
            interceptors: Client-side interceptors (e.g. AuthTokenInjector for token auth)
            use_controller_proxy: Route service RPCs (currently finelog
                logs/stats) through the controller instead of resolving the
                backing service's address from the controller's endpoint
                registry and connecting straight to it. The direct path is only
                safe for clients running *inside* the cluster, which can reach
                internal service addresses; external clients (CLI over a
                tunnel) cannot and must keep the default proxied path.
        """
        self._address = controller_address
        self._bundle_id = bundle_id
        self._workspace = workspace.resolve() if workspace is not None else None
        self._extra_bundle_includes = extra_bundle_includes
        self._bundle_blob: bytes | None = None
        self._timeout_ms = timeout_ms
        self._use_controller_proxy = use_controller_proxy
        self._client = ControllerServiceClientSync(
            address=controller_address,
            timeout_ms=timeout_ms,
            interceptors=interceptors,
            accept_compression=IRIS_RPC_COMPRESSIONS,
            send_compression=None,
        )
        # Endpoint registry on its own service. EndpointClient owns the RPC stub
        # and the background lease renewal: register() keeps the endpoint alive
        # until unregister()/close(), so the controller keeps serving it while
        # the task runs.
        self._endpoint_client = EndpointClient(
            EndpointServiceClientSync(
                address=controller_address,
                timeout_ms=timeout_ms,
                interceptors=interceptors,
                accept_compression=IRIS_RPC_COMPRESSIONS,
                send_compression=None,
            )
        )
        # In-cluster clients resolve the finelog endpoint and write direct so
        # task-status pushes don't pile up on the controller's RPC thread pool;
        # external clients route through the controller, the only ingress they
        # can reach. The resolver fires lazily on first table/log use, so this
        # adds no RPC for CLI calls that never touch logs.
        self._log_client = LogClient.connect(
            LOG_SERVER_ENDPOINT_NAME,
            resolver=self.resolve_endpoint,
            timeout_ms=timeout_ms,
            interceptors=interceptors,
        )
        # Deferred so CLI calls that never push status (submit_job, list_jobs,
        # get_status) don't spawn a finelog flush thread.
        self._task_status_table = None

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
        max_task_failures: int = 0,
        timeout: Duration | None = None,
        preemption_policy: job_pb2.JobPreemptionPolicy = job_pb2.JOB_PREEMPTION_POLICY_UNSPECIFIED,
        existing_job_policy: job_pb2.ExistingJobPolicy = job_pb2.EXISTING_JOB_POLICY_UNSPECIFIED,
        task_image: str | None = None,
        priority_band: job_pb2.PriorityBand = job_pb2.PRIORITY_BAND_UNSPECIFIED,
        container_profile: job_pb2.ContainerProfile = job_pb2.CONTAINER_PROFILE_UNSPECIFIED,
        submit_argv: list[str] | None = None,
    ) -> JobName:
        if replicas < 1:
            raise ValueError(f"replicas must be >= 1, got {replicas}")
        replicas = adjust_tpu_replicas(resources.device if resources.HasField("device") else None, replicas)

        if environment is None:
            environment = EnvironmentSpec().to_proto()
        env_config = with_slice_topology_env(environment, resources, replicas)

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
            max_task_failures=max_task_failures,
            preemption_policy=preemption_policy,
            existing_job_policy=existing_job_policy,
            task_image=task_image or "",
            priority_band=priority_band,
            container_profile=container_profile,
            submit_argv=submit_argv or [],
            client_revision_date=client_revision_date(),
        )
        if self._bundle_id:
            request.bundle_id = self._bundle_id
        else:
            if self._bundle_blob is None and self._workspace is not None:
                self._bundle_blob = create_workspace_zip(self._workspace, extra_includes=self._extra_bundle_includes)
                logger.info(f"Workspace bundle size: {len(self._bundle_blob) / 1024 / 1024:.1f} MB")
            request.bundle_blob = self._bundle_blob or b""

        if scheduling_timeout is not None:
            request.scheduling_timeout.CopyFrom(duration_to_proto(scheduling_timeout))
        if timeout is not None:
            request.timeout.CopyFrom(duration_to_proto(timeout))
        if coscheduling is not None:
            request.coscheduling.CopyFrom(coscheduling)

        launch_timeout_ms = max(self._timeout_ms, LAUNCH_JOB_TIMEOUT_FLOOR_MS)

        def _call():
            return self._client.launch_job(request, timeout_ms=launch_timeout_ms)

        response = call_with_retry(f"launch_job({job_id})", _call)
        return JobName.from_wire(response.job_id)

    def get_job_status(self, job_id: JobName) -> job_pb2.JobStatus:
        def _call():
            request = controller_pb2.Controller.GetJobStatusRequest(job_id=job_id.to_wire())
            response = self._client.get_job_status(request)
            return response.job

        return call_with_retry(f"get_job_status({job_id})", _call)

    def get_job_states(self, job_ids: list[JobName]) -> dict[str, job_pb2.JobState]:
        """Lightweight batch query returning only the state enum per job."""

        def _call():
            request = controller_pb2.Controller.GetJobStateRequest(
                job_ids=[jid.to_wire() for jid in job_ids],
            )
            response = self._client.get_job_state(request)
            return dict(response.states)

        return call_with_retry(f"get_job_states({len(job_ids)} jobs)", _call)

    def _poll_job_state(self, job_id: JobName) -> job_pb2.JobState:
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
        source, match_scope = build_log_source(job_id)
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
                log_response = self.fetch_logs(
                    source,
                    match_scope=match_scope,
                    since_ms=since_ms,
                    cursor=cursor,
                    min_level=min_level,
                )
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
        access: int = EndpointAccess.ENDPOINT_ACCESS_PRIVATE,
    ) -> str:
        return self._endpoint_client.register(name, address, task_attempt, metadata, access)

    def unregister_endpoint(self, endpoint_id: str) -> None:
        """Unregister an endpoint via RPC."""
        self._endpoint_client.unregister(endpoint_id)

    def mint_endpoint_token(
        self, endpoint_name: str, ttl: Duration | None = None
    ) -> controller_pb2.Controller.MintEndpointTokenResponse:
        """Mint a scoped bearer token for ``endpoint_name``'s /proxy path.

        Authorized to the endpoint's owning user (or admin); the CLI holds that
        identity. ``ttl`` is clamped server-side to the controller's maximum.
        """
        request = controller_pb2.Controller.MintEndpointTokenRequest(
            endpoint_name=endpoint_name,
            ttl=duration_to_proto(ttl) if ttl is not None else None,
        )
        return call_with_retry(
            f"mint_endpoint_token({endpoint_name})", lambda: self._client.mint_endpoint_token(request)
        )

    def list_endpoints(self, prefix: str, *, exact: bool = False) -> list[controller_pb2.Controller.Endpoint]:
        return self._endpoint_client.list_endpoints(prefix, exact=exact)

    def resolve_endpoint(self, endpoint_name: str) -> str:
        """Resolve ``endpoint_name`` to a service address.

        When ``use_controller_proxy`` is set (external clients), returns the
        endpoint's path under the controller's generic proxy; otherwise looks
        the name up in the controller's endpoint registry and returns the
        backing service's direct address.
        """
        if self._use_controller_proxy:
            return f"{self._address.rstrip('/')}{proxy_path(endpoint_name)}"
        endpoints = self.list_endpoints(endpoint_name, exact=True)
        if not endpoints:
            raise ConnectionError(f"No {endpoint_name!r} endpoint registered on controller")
        return endpoints[0].address

    def list_workers(
        self,
        query: controller_pb2.Controller.WorkerQuery | None = None,
    ) -> list[controller_pb2.Controller.WorkerHealthStatus]:
        """List all workers registered with the controller."""

        def _call():
            request = controller_pb2.Controller.ListWorkersRequest()
            if query is not None:
                request.query.CopyFrom(query)
            response = self._client.list_workers(request)
            return list(response.workers)

        return call_with_retry("list_workers", _call)

    def list_jobs(
        self,
        *,
        query: controller_pb2.Controller.JobQuery,
        limit: int | None = None,
        page_size: int = 500,
    ) -> list[job_pb2.JobStatus]:
        """Fetch jobs matching ``query`` by paging through ``ListJobs``.

        ``limit`` caps the total number of jobs returned; paging stops as soon
        as that many are collected (or the result set is exhausted). Because the
        server defaults to sorting by submission date descending, a small
        ``limit`` yields the most recent jobs without scanning the whole table.

        ``limit=None`` walks every matching job. That requires a query narrow
        enough to stay under ``MAX_LIST_JOBS_OFFSET`` (via ``state_filter`` /
        ``name_filter`` / ``job_id_prefix`` / ``parent_job_id``); otherwise the
        walk fails once it reaches the offset cap.
        """
        jobs: list[job_pb2.JobStatus] = []
        offset = query.offset or 0
        while limit is None or len(jobs) < limit:
            this_page = page_size if limit is None else min(page_size, limit - len(jobs))
            page_query = controller_pb2.Controller.JobQuery()
            page_query.CopyFrom(query)
            page_query.offset = offset
            page_query.limit = this_page

            def _call(q=page_query):
                request = controller_pb2.Controller.ListJobsRequest(query=q)
                return self._client.list_jobs(request)

            response = call_with_retry("list_jobs", _call)
            jobs.extend(response.jobs)
            if not response.has_more or not response.jobs:
                break
            offset += len(response.jobs)
        return jobs

    def shutdown(self, wait: bool = True) -> None:
        del wait
        self._endpoint_client.close()
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

    def kick_tasks(
        self,
        targets: list[str],
        desired_state: job_pb2.TaskState,
        reason: str,
    ) -> list[controller_pb2.Controller.KickResult]:
        """Force task attempts into a terminal state out-of-band (emergency override)."""

        def _call():
            request = controller_pb2.Controller.KickTasksRequest(
                targets=targets,
                desired_state=desired_state,
                reason=reason,
            )
            response = self._client.kick_tasks(request)
            return list(response.results)

        return call_with_retry(f"kick_tasks({', '.join(targets)})", _call)

    def fetch_logs(
        self,
        source: str,
        *,
        match_scope: int = logging_pb2.MATCH_SCOPE_UNSPECIFIED,
        since_ms: int = 0,
        cursor: int = 0,
        max_lines: int = 0,
        substring: str = "",
        min_level: str = "",
        tail: bool = False,
    ) -> logging_pb2.FetchLogsResponse:
        request = logging_pb2.FetchLogsRequest(
            source=source,
            match_scope=match_scope,
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

    def list_backends(self) -> list[controller_pb2.Controller.BackendSummary]:
        """Fetch this controller's backends: their topology and current state."""
        request = controller_pb2.Controller.ListBackendsRequest()
        # Single attempt, no retry (unlike the sibling calls): a failed federation
        # heartbeat marks the peer unreachable until the next one, so retrying here
        # would only delay that signal.
        response = self._client.list_backends(request)
        return list(response.backends)

    def get_autoscaler_status(self) -> controller_pb2.Controller.GetAutoscalerStatusResponse:
        """Get autoscaler status including recent actions and group states.

        Returns:
            GetAutoscalerStatusResponse proto with autoscaler status and recent actions
        """

        def _call():
            request = controller_pb2.Controller.GetAutoscalerStatusRequest()
            return self._client.get_autoscaler_status(request)

        return call_with_retry("get_autoscaler_status", _call)

    def report_task_status_text(
        self,
        task_id: JobName,
        attempt_id: int,
        detail_md: str,
        summary_md: str,
    ) -> None:
        """Push one row of markdown status text to the ``iris.task_status`` namespace.

        Buffered and fire-and-forget; ``LogClient`` swallows flush failures
        so a finelog outage cannot abort task execution.
        """
        if self._task_status_table is None:
            self._task_status_table = self._log_client.get_table(
                TASK_STATUS_NAMESPACE,
                TaskStatusRow,
                storage_policy=TASK_STATUS_STORAGE_POLICY,
            )
        self._task_status_table.write(
            [
                TaskStatusRow(
                    ts=datetime.now(UTC).replace(tzinfo=None),
                    task_id=task_id.to_wire(),
                    attempt_id=attempt_id,
                    status_text_detail_md=detail_md,
                    status_text_summary_md=summary_md,
                )
            ]
        )
