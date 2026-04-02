# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ClusterClient protocol defining the interface for cluster client implementations."""

from typing import Protocol

from iris.cluster.types import Entrypoint, JobName, TaskAttempt
from iris.rpc import cluster_pb2
from rigging.timing import Duration


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
        max_retries_preemption: int = 1000,
        timeout: Duration | None = None,
        reservation: cluster_pb2.ReservationConfig | None = None,
        preemption_policy: cluster_pb2.JobPreemptionPolicy = cluster_pb2.JOB_PREEMPTION_POLICY_UNSPECIFIED,
        existing_job_policy: cluster_pb2.ExistingJobPolicy = cluster_pb2.EXISTING_JOB_POLICY_UNSPECIFIED,
    ) -> JobName: ...

    def get_job_status(self, job_id: JobName) -> cluster_pb2.JobStatus: ...

    def get_job_states(self, job_ids: list[JobName]) -> dict[str, int]:
        """Lightweight batch query returning only the state enum per job."""
        ...

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
        since_ms: int = 0,
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
    ) -> cluster_pb2.FetchLogsResponse: ...

    def get_autoscaler_status(self) -> cluster_pb2.Controller.GetAutoscalerStatusResponse: ...

    def shutdown(self, wait: bool = True) -> None: ...
