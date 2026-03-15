# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ClusterClient protocol defining the interface for cluster client implementations."""

from typing import Protocol

from iris.cluster.types import Entrypoint, JobName, TaskAttempt
from iris.rpc import cluster_pb2
from iris.time_utils import Duration


class TaskStateLogger(Protocol):
    """Observer for task state changes during streaming job wait.

    Implement this protocol to customize how task lifecycle events and log
    output are presented.  The streaming loop in ``wait_for_job_with_streaming``
    calls these methods as child-job states transition and log batches arrive.
    """

    def task_started(self, job_id: str, status: cluster_pb2.JobStatus) -> None:
        """Called when a child job is first observed in the status list."""
        ...

    def task_finished(self, job_id: str, status: cluster_pb2.JobStatus) -> None:
        """Called when a child job reaches a terminal state (success or failure)."""
        ...

    def task_logging(self, response: cluster_pb2.Controller.GetTaskLogsResponse) -> None:
        """Called with each batch of fetched task logs."""
        ...


class ClusterClient(Protocol):
    """Protocol for cluster client implementations.

    RemoteClusterClient satisfies this protocol, enabling callers to depend
    on the interface rather than concrete types.
    """

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
        reservation: cluster_pb2.ReservationConfig | None = None,
        preemption_policy: cluster_pb2.JobPreemptionPolicy = cluster_pb2.JOB_PREEMPTION_POLICY_UNSPECIFIED,
        existing_job_policy: cluster_pb2.ExistingJobPolicy = cluster_pb2.EXISTING_JOB_POLICY_UNSPECIFIED,
    ) -> None: ...

    def get_job_status(self, job_id: JobName) -> cluster_pb2.JobStatus: ...

    def wait_for_job(
        self,
        job_id: JobName,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> cluster_pb2.JobStatus: ...

    def wait_for_job_with_streaming(
        self,
        job_id: JobName,
        *,
        timeout: float,
        poll_interval: float,
        include_children: bool,
        since_ms: int = 0,
        state_logger: TaskStateLogger | None = None,
        min_level: str = "",
    ) -> cluster_pb2.JobStatus: ...

    def terminate_job(self, job_id: JobName) -> None: ...

    def register_endpoint(
        self,
        name: str,
        address: str,
        task_attempt: TaskAttempt,
        metadata: dict[str, str] | None = None,
    ) -> str: ...

    def unregister_endpoint(self, endpoint_id: str) -> None: ...

    def list_endpoints(self, prefix: str, *, exact: bool = False) -> list[cluster_pb2.Controller.Endpoint]: ...

    def list_workers(self) -> list[cluster_pb2.Controller.WorkerHealthStatus]: ...

    def list_jobs(self) -> list[cluster_pb2.JobStatus]: ...

    def get_task_status(self, task_name: JobName) -> cluster_pb2.TaskStatus: ...

    def list_tasks(self, job_id: JobName) -> list[cluster_pb2.TaskStatus]: ...

    def fetch_task_logs(
        self,
        target: JobName,
        *,
        include_children: bool = False,
        since_ms: int = 0,
        max_total_lines: int = 0,
        substring: str | None = None,
        attempt_id: int = -1,
        cursor: int = 0,
        min_level: str = "",
    ) -> cluster_pb2.Controller.GetTaskLogsResponse: ...

    def get_autoscaler_status(self) -> cluster_pb2.Controller.GetAutoscalerStatusResponse: ...

    def shutdown(self, wait: bool = True) -> None: ...
