# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for controller unit tests."""

import tempfile
from pathlib import Path

import pytest

from iris.cluster.controller.db import ATTEMPTS, TASKS, ControllerDB
from iris.cluster.controller.provider import ProviderSyncOutcome, ProviderUnsupportedError
from iris.cluster.controller.transitions import ControllerTransitions
from iris.cluster.log_store import LogStore
from iris.cluster.types import JobName
from iris.rpc import cluster_pb2
from iris.time_utils import Timestamp


class FakeProvider:
    """Minimal TaskProvider for tests that only exercise transitions, not RPCs.

    Implements the unified TaskProvider protocol. sync() always returns an
    empty outcome (no-op) since these tests focus on transition logic.
    """

    @property
    def has_workers(self) -> bool:
        return True

    def sync(self, transitions):
        return ProviderSyncOutcome()

    def kill_unmapped_tasks(self, task_ids, transitions):
        return False

    def fetch_live_logs(self, task_id, attempt_id, cursor, max_lines, worker_id=None, address=None):
        raise ProviderUnsupportedError("fake")

    def fetch_process_logs(self, worker_id, address, request):
        raise ProviderUnsupportedError("fake")

    def get_process_status(self, worker_id, address, request):
        raise ProviderUnsupportedError("fake")

    def on_worker_failed(self, worker_id, address):
        pass

    def profile_task(self, task_id, attempt_id, address, request, timeout_ms):
        raise ProviderUnsupportedError("fake")

    def exec_in_container(self, address, request, timeout_seconds):
        raise ProviderUnsupportedError("fake")

    def get_cluster_status(self):
        return cluster_pb2.Controller.GetKubernetesClusterStatusResponse()

    def close(self):
        pass


class FakeDirectProvider:
    """Minimal direct TaskProvider for tests that exercise direct provider paths.

    has_workers=False, so the controller treats it as a direct provider.
    """

    @property
    def has_workers(self) -> bool:
        return False

    def sync(self, transitions):
        return ProviderSyncOutcome()

    def kill_unmapped_tasks(self, task_ids, transitions):
        for task_id in task_ids:
            transitions.buffer_direct_kill(task_id.to_wire())
        return bool(task_ids)

    def fetch_live_logs(self, task_id, attempt_id, cursor, max_lines, worker_id=None, address=None):
        return [], cursor

    def fetch_process_logs(self, worker_id, address, request):
        raise ProviderUnsupportedError("direct provider has no worker processes")

    def get_process_status(self, worker_id, address, request):
        raise ProviderUnsupportedError("direct provider has no worker processes")

    def on_worker_failed(self, worker_id, address):
        pass

    def profile_task(self, task_id, attempt_id, address, request, timeout_ms):
        raise ProviderUnsupportedError("fake")

    def exec_in_container(self, address, request, timeout_seconds):
        raise ProviderUnsupportedError("direct provider does not support exec_in_container")

    def get_cluster_status(self):
        return cluster_pb2.Controller.GetKubernetesClusterStatusResponse()

    def close(self):
        pass


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


def make_controller_state(**kwargs) -> ControllerTransitions:
    """Create a ControllerTransitions with a fresh temp DB and log store."""
    tmp = Path(tempfile.mkdtemp(prefix="iris_test_"))
    db = ControllerDB(db_dir=tmp)
    log_store = LogStore(log_dir=tmp / "logs")
    return ControllerTransitions(db=db, log_store=log_store, **kwargs)


def make_test_entrypoint() -> cluster_pb2.RuntimeEntrypoint:
    """Create a minimal RuntimeEntrypoint proto for testing."""
    entrypoint = cluster_pb2.RuntimeEntrypoint()
    entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
    return entrypoint


def make_direct_job_request(name: str = "test-job", replicas: int = 1) -> cluster_pb2.Controller.LaunchJobRequest:
    """Create a LaunchJobRequest suitable for direct provider tests."""
    job_name = JobName.root("test-user", name)
    return cluster_pb2.Controller.LaunchJobRequest(
        name=job_name.to_wire(),
        entrypoint=make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=replicas,
    )


def submit_direct_job(state: ControllerTransitions, name: str, replicas: int = 1) -> list[JobName]:
    """Submit a job for direct provider testing and return the created task IDs."""
    jid = JobName.root("test-user", name)
    req = make_direct_job_request(name, replicas)
    state.submit_job(jid, req, Timestamp.now())
    with state._db.snapshot() as q:
        tasks = q.select(TASKS, where=TASKS.c.job_id == jid.to_wire())
    return [t.task_id for t in tasks]


def query_task(state: ControllerTransitions, task_id: JobName):
    """Query a single task by ID."""
    with state._db.snapshot() as q:
        return q.one(TASKS, where=TASKS.c.task_id == task_id.to_wire())


def query_attempt(state: ControllerTransitions, task_id: JobName, attempt_id: int):
    """Query a single attempt row."""
    with state._db.snapshot() as q:
        rows = q.select(
            ATTEMPTS,
            where=(ATTEMPTS.c.task_id == task_id.to_wire()) & (ATTEMPTS.c.attempt_id == attempt_id),
        )
    return rows[0] if rows else None
