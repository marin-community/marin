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

"""RPC-based cluster client implementation."""

import os
import sys
import time

from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import Entrypoint, JobName, generate_dockerfile, is_job_finished
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.time_utils import Deadline, Duration, ExponentialBackoff


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
    ) -> None:
        if replicas < 1:
            raise ValueError(f"replicas must be >= 1, got {replicas}")

        entrypoint_proto = entrypoint.to_proto()

        # If no dockerfile is provided, inherit from the parent job or generate a default.
        dockerfile = environment.dockerfile if environment and environment.dockerfile else ""
        if not dockerfile:
            job_info = get_job_info()
            if job_info is not None and job_info.dockerfile:
                dockerfile = job_info.dockerfile
            else:
                dockerfile = generate_dockerfile(python_version=f"{sys.version_info.major}.{sys.version_info.minor}")

        env_vars = dict(environment.env_vars) if environment else {}
        job_info = get_job_info()
        if job_info is not None:
            for key, value in os.environ.items():
                if key.startswith("IRIS_"):
                    continue
                env_vars.setdefault(key, value)

        env_config = cluster_pb2.EnvironmentConfig(
            pip_packages=list(environment.pip_packages) if environment else [],
            env_vars=env_vars,
            extras=list(environment.extras) if environment else [],
            python_version=environment.python_version if environment else "",
            dockerfile=dockerfile,
        )

        # Use bundle_gcs_path if available, otherwise use bundle_blob
        if self._bundle_gcs_path:
            request = cluster_pb2.Controller.LaunchJobRequest(
                name=job_id.to_wire(),
                entrypoint=entrypoint_proto,
                resources=resources,
                environment=env_config,
                bundle_gcs_path=self._bundle_gcs_path,
                ports=ports or [],
                constraints=constraints or [],
                replicas=replicas,
                max_retries_failure=max_retries_failure,
                max_retries_preemption=max_retries_preemption,
            )
        else:
            request = cluster_pb2.Controller.LaunchJobRequest(
                name=job_id.to_wire(),
                entrypoint=entrypoint_proto,
                resources=resources,
                environment=env_config,
                bundle_blob=self._bundle_blob or b"",
                ports=ports or [],
                constraints=constraints or [],
                replicas=replicas,
                max_retries_failure=max_retries_failure,
                max_retries_preemption=max_retries_preemption,
            )

        if scheduling_timeout is not None:
            request.scheduling_timeout.CopyFrom(scheduling_timeout.to_proto())
        if timeout is not None:
            request.timeout.CopyFrom(timeout.to_proto())
        if coscheduling is not None:
            request.coscheduling.CopyFrom(coscheduling)
        self._client.launch_job(request)

    def get_job_status(self, job_id: JobName) -> cluster_pb2.JobStatus:
        request = cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id.to_wire())
        response = self._client.get_job_status(request)
        return response.job

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
        request = cluster_pb2.Controller.ListJobsRequest()
        response = self._client.list_jobs(request)
        return list(response.jobs)

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
        request = cluster_pb2.Controller.GetTaskStatusRequest(task_id=task_name.to_wire())
        response = self._client.get_task_status(request)
        return response.task

    def list_tasks(self, job_id: JobName) -> list[cluster_pb2.TaskStatus]:
        """List all tasks for a job.

        Args:
            job_id: Job ID to query tasks for

        Returns:
            List of TaskStatus protos, one per task in the job
        """
        request = cluster_pb2.Controller.ListTasksRequest(job_id=job_id.to_wire())
        response = self._client.list_tasks(request)
        return list(response.tasks)

    def fetch_task_logs(
        self,
        target: JobName,
        *,
        include_children: bool = False,
        since_ms: int = 0,
        max_total_lines: int = 0,
        regex: str | None = None,
    ) -> cluster_pb2.Controller.GetTaskLogsResponse:
        """Fetch logs for a task or all tasks in a job.

        Args:
            target: Task ID or Job ID (detected by trailing numeric)
            include_children: Include logs from child jobs (job ID only)
            since_ms: Only return logs after this timestamp (exclusive)
            max_total_lines: Maximum total lines (0 = default 10000)
            regex: Regex filter for log content
        """
        request = cluster_pb2.Controller.GetTaskLogsRequest(
            id=target.to_wire(),
            include_children=include_children,
            since_ms=since_ms,
            max_total_lines=max_total_lines,
            regex=regex or "",
        )
        return self._client.get_task_logs(request)

    def get_autoscaler_status(self) -> cluster_pb2.Controller.GetAutoscalerStatusResponse:
        """Get autoscaler status including recent actions and group states.

        Returns:
            GetAutoscalerStatusResponse proto with autoscaler status and recent actions
        """
        request = cluster_pb2.Controller.GetAutoscalerStatusRequest()
        return self._client.get_autoscaler_status(request)
