# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extracted cluster helper for Iris integration tests."""

import time
from contextlib import contextmanager
from dataclasses import dataclass

from iris.client.client import IrisClient, Job, Task
from iris.cluster.constraints import Constraint
from iris.cluster.resources.identity import NodeLocator
from iris.cluster.resources.job import JobSummary
from iris.cluster.resources.log import LogPage, LogQuery
from iris.cluster.resources.node import NodeDetail, NodeHealth, NodeQuery
from iris.cluster.resources.task import TaskSummary
from iris.cluster.types import CoschedulingConfig, Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import iris_logging_pb2
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
        )

    def status(self, job: Job) -> JobSummary:
        return job.status()

    def task(self, job: Job, task_index: int = 0) -> Task:
        match = next((task for task in job.tasks() if task.task_index == task_index), None)
        if match is None:
            raise LookupError(f"Job {job.job_id} has no Task {task_index}")
        return match

    def task_status(self, job: Job, task_index: int = 0) -> TaskSummary:
        return self.task(job, task_index).status()

    def wait(self, job: Job, timeout: float = 60.0, poll_interval: float = 0.5) -> JobSummary:
        """Poll until a job reaches a terminal state."""
        return job.wait(timeout=timeout, poll_interval=poll_interval, raise_on_failure=False)

    def wait_for_state(
        self,
        job: Job,
        state: int,
        timeout: float = 10.0,
        poll_interval: float = 0.1,
    ) -> JobSummary:
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
    ) -> TaskSummary:
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
        job.cancel(idempotency_key=f"integration-cancel:{job.identity.job_uid}")

    def wait_for_workers(self, min_workers: int, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        ready: list[NodeDetail] = []
        while time.monotonic() < deadline:
            ready = [node for node in self.list_nodes() if node.summary.health is NodeHealth.READY]
            if len(ready) >= min_workers:
                return
            time.sleep(0.5)
        raise TimeoutError(f"Only {len(ready)} of {min_workers} Nodes ready in {timeout}s")

    def list_nodes(self) -> list[NodeDetail]:
        """List and describe the public Node inventory."""
        summaries = []
        query = NodeQuery(page_size=500)
        while True:
            page = self.client.list_nodes(query)
            summaries.extend(page.items)
            if page.next_page_token is None:
                break
            query = NodeQuery(page_size=500, page_token=page.next_page_token)
        return [
            self.client.describe_node(
                NodeLocator(summary.identity.key, summary.identity.backend_id, summary.identity.node_uid)
            )
            for summary in summaries
        ]

    def task_logs(
        self,
        job: Job,
        task_index: int = 0,
        *,
        minimum_level: iris_logging_pb2.LogLevel = iris_logging_pb2.LOG_LEVEL_UNKNOWN,
    ) -> LogPage:
        task = self.task(job, task_index)
        return self.client.fetch_task_logs(
            task.identity,
            LogQuery(max_lines=1_000, minimum_level=minimum_level),
        )

    def get_task_logs(self, job: Job, task_index: int = 0) -> list[str]:
        return [f"{entry.source}: {entry.data}" for entry in self.task_logs(job, task_index).entries]
