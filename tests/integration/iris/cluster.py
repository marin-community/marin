# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extracted cluster helper for Iris integration tests."""

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass

from finelog.rpc import logging_pb2
from iris.client.client import IrisClient, Job
from iris.cluster.constraints import Constraint
from iris.cluster.types import (
    CoschedulingConfig,
    Entrypoint,
    EnvironmentSpec,
    ReservationEntry,
    ResourceSpec,
)
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Duration


@dataclass
class IrisIntegrationCluster:
    """Wraps an IrisClient with convenience methods for integration tests.

    All RPCs go through RemoteClusterClient which has built-in retry logic,
    making tests resilient to transient port-forward drops.
    """

    url: str
    client: IrisClient
    job_timeout: float = 60.0

    @property
    def _cluster(self):
        return self.client._cluster_client

    @property
    def controller_client(self):
        """Raw controller stub for RPCs not yet on RemoteClusterClient (profile, exec)."""
        return self._cluster._client

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

    def status(self, job: Job) -> job_pb2.JobStatus:
        return self.client.status(job.job_id)

    def task_status(self, job: Job, task_index: int = 0) -> job_pb2.TaskStatus:
        return self.client.task_status(job.job_id.task(task_index))

    def wait(self, job: Job, timeout: float = 60.0, poll_interval: float = 0.5) -> job_pb2.JobStatus:
        """Poll until a job reaches a terminal state."""
        return job.wait(timeout=timeout, poll_interval=poll_interval, raise_on_failure=False)

    def wait_for_state(
        self,
        job: Job,
        state: int,
        timeout: float = 10.0,
        poll_interval: float = 0.1,
    ) -> job_pb2.JobStatus:
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
    ) -> job_pb2.TaskStatus:
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
        self.client.terminate(job.job_id)

    def wait_for_workers(self, min_workers: int, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        healthy = []
        while time.monotonic() < deadline:
            workers = self._cluster.list_workers()
            healthy = [w for w in workers if w.healthy]
            if len(healthy) >= min_workers:
                return
            time.sleep(0.5)
        raise TimeoutError(f"Only {len(healthy)} of {min_workers} workers registered in {timeout}s")

    def list_workers(self) -> list[controller_pb2.Controller.WorkerHealthStatus]:
        """List workers with retry logic via RemoteClusterClient."""
        return self._cluster.list_workers()

    def fetch_logs(self, source: str, **kwargs) -> logging_pb2.FetchLogsResponse:
        """Fetch logs with retry logic via RemoteClusterClient."""
        return self._cluster.fetch_logs(source, **kwargs)

    def get_task_logs(self, job: Job, task_index: int = 0) -> list[str]:
        task_id = job.job_id.task(task_index).to_wire()
        response = self._cluster.fetch_logs(re.escape(task_id) + ":.*")
        return [f"{e.source}: {e.data}" for e in response.entries]
