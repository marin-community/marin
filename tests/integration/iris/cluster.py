# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extracted cluster helper for Iris integration tests."""

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass

from iris.client.client import IrisClient, Job
from iris.cluster.constraints import Constraint
from iris.cluster.types import (
    CoschedulingConfig,
    Entrypoint,
    EnvironmentSpec,
    ReservationEntry,
    ResourceSpec,
    is_job_finished,
)
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from rigging.timing import Duration


@dataclass
class IrisIntegrationCluster:
    """Wraps a cluster connection with convenience methods for integration tests.

    Unlike the E2E IrisTestCluster, this is designed for tests that exercise
    job submission and lifecycle without dashboard/screenshot concerns.
    """

    url: str
    client: IrisClient
    controller_client: ControllerServiceClientSync
    job_timeout: float = 60.0

    def submit(
        self,
        fn,
        name: str,
        *args,
        cpu: float = 1,
        memory: str = "4g",
        ports: list[str] | None = None,
        scheduling_timeout: Duration | None = None,
        replicas: int = 1,
        max_retries_failure: int = 0,
        max_retries_preemption: int = 1000,
        timeout: Duration | None = None,
        coscheduling: CoschedulingConfig | None = None,
        constraints: list[Constraint] | None = None,
        reservation: list[ReservationEntry] | None = None,
    ) -> Job:
        """Submit a callable as a job."""
        return self.client.submit(
            entrypoint=Entrypoint.from_callable(fn, *args),
            name=name,
            resources=ResourceSpec(cpu=cpu, memory=memory),
            environment=EnvironmentSpec(),
            ports=ports,
            scheduling_timeout=scheduling_timeout,
            replicas=replicas,
            max_retries_failure=max_retries_failure,
            max_retries_preemption=max_retries_preemption,
            timeout=timeout,
            coscheduling=coscheduling,
            constraints=constraints,
            reservation=reservation,
        )

    def status(self, job: Job) -> cluster_pb2.JobStatus:
        job_id = job.job_id.to_wire()
        request = cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        response = self.controller_client.get_job_status(request)
        return response.job

    def task_status(self, job: Job, task_index: int = 0) -> cluster_pb2.TaskStatus:
        task_id = job.job_id.task(task_index).to_wire()
        request = cluster_pb2.Controller.GetTaskStatusRequest(task_id=task_id)
        response = self.controller_client.get_task_status(request)
        return response.task

    def wait(self, job: Job, timeout: float = 60.0, poll_interval: float = 0.5) -> cluster_pb2.JobStatus:
        """Poll until a job reaches a terminal state."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.status(job)
            if is_job_finished(status.state):
                return status
            time.sleep(poll_interval)
        raise TimeoutError(f"Job {job.job_id} did not complete in {timeout}s")

    def wait_for_state(
        self,
        job: Job,
        state: int,
        timeout: float = 10.0,
        poll_interval: float = 0.1,
    ) -> cluster_pb2.JobStatus:
        deadline = time.monotonic() + timeout
        status = self.status(job)
        while time.monotonic() < deadline:
            status = self.status(job)
            if status.state == state:
                return status
            time.sleep(poll_interval)
        raise TimeoutError(f"Job {job.job_id} did not reach state {state} in {timeout}s (current: {status.state})")

    def wait_for_task_state(
        self,
        job: Job,
        state: int,
        task_index: int = 0,
        timeout: float = 60.0,
        poll_interval: float = 0.5,
    ) -> cluster_pb2.TaskStatus:
        deadline = time.monotonic() + timeout
        task = self.task_status(job, task_index)
        while time.monotonic() < deadline:
            task = self.task_status(job, task_index)
            if task.state == state:
                return task
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Task {task_index} of {job.job_id} did not reach state {state} " f"in {timeout}s (current: {task.state})"
        )

    @contextmanager
    def launched_job(self, fn, name: str, *args, **kwargs):
        """Submit a job and guarantee it's killed on exit."""
        job = self.submit(fn, name, *args, **kwargs)
        try:
            yield job
        finally:
            self.kill(job)

    def kill(self, job: Job) -> None:
        job_id = job.job_id.to_wire()
        request = cluster_pb2.Controller.TerminateJobRequest(job_id=job_id)
        self.controller_client.terminate_job(request)

    def wait_for_workers(self, min_workers: int, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        healthy = []
        while time.monotonic() < deadline:
            request = cluster_pb2.Controller.ListWorkersRequest()
            response = self.controller_client.list_workers(request)
            healthy = [w for w in response.workers if w.healthy]
            if len(healthy) >= min_workers:
                return
            time.sleep(0.5)
        raise TimeoutError(f"Only {len(healthy)} of {min_workers} workers registered in {timeout}s")

    def get_task_logs(self, job: Job, task_index: int = 0) -> list[str]:
        task_id = job.job_id.task(task_index).to_wire()
        request = cluster_pb2.FetchLogsRequest(source=re.escape(task_id) + ":.*")
        response = self.controller_client.fetch_logs(request)
        return [f"{e.source}: {e.data}" for e in response.entries]
