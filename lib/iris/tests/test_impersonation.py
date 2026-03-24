# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GCP impersonation based on the logged-in user's GCP email.

Covers: env injection, DB roundtrip for gcp_email, and task dispatch
using gcp_email as the impersonation identity.
"""

import pytest

from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.transitions import Assignment, ControllerTransitions
from iris.cluster.log_store import LogStore
from iris.cluster.runtime.env import build_common_iris_env
from iris.cluster.types import JobName, WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import Timestamp


def _make_test_entrypoint() -> cluster_pb2.RuntimeEntrypoint:
    entrypoint = cluster_pb2.RuntimeEntrypoint()
    entrypoint.run_command.argv[:] = ["python", "train.py"]
    return entrypoint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    db = ControllerDB(db_dir=tmp_path)
    yield db
    db.close()


@pytest.fixture
def state(tmp_path):
    db = ControllerDB(db_dir=tmp_path)
    log_store = LogStore(log_dir=tmp_path / "logs")
    s = ControllerTransitions(db=db, log_store=log_store, enable_task_impersonation=True)
    yield s
    log_store.close()
    db.close()


# ---------------------------------------------------------------------------
# 1. Env injection
# ---------------------------------------------------------------------------


def test_build_common_iris_env_includes_impersonation():
    """CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT is set when impersonation identity provided."""
    env = build_common_iris_env(
        task_id="test/task/0",
        attempt_id=0,
        num_tasks=1,
        bundle_id="test-bundle",
        controller_address=None,
        environment=cluster_pb2.EnvironmentConfig(),
        constraints=[],
        ports=[],
        resources=None,
        impersonate_service_account="alice@example.com",
    )
    assert env["CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT"] == "alice@example.com"


def test_build_common_iris_env_no_impersonation_when_empty():
    """CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT not set when identity is empty."""
    env = build_common_iris_env(
        task_id="test/task/0",
        attempt_id=0,
        num_tasks=1,
        bundle_id="test-bundle",
        controller_address=None,
        environment=cluster_pb2.EnvironmentConfig(),
        constraints=[],
        ports=[],
        resources=None,
    )
    assert "CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT" not in env


def test_build_common_iris_env_no_impersonation_when_explicitly_empty():
    """Passing empty string for impersonate_service_account omits the env var."""
    env = build_common_iris_env(
        task_id="test/task/0",
        attempt_id=0,
        num_tasks=1,
        bundle_id="test-bundle",
        controller_address=None,
        environment=cluster_pb2.EnvironmentConfig(),
        constraints=[],
        ports=[],
        resources=None,
        impersonate_service_account="",
    )
    assert "CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT" not in env


# ---------------------------------------------------------------------------
# 2. DB roundtrip for gcp_email
# ---------------------------------------------------------------------------


def test_user_gcp_email_roundtrip(db):
    """gcp_email is stored and retrieved correctly."""
    db.ensure_user("alice@example.com", Timestamp.now())
    db.set_user_gcp_email("alice@example.com", "alice@example.com")
    assert db.get_user_gcp_email("alice@example.com") == "alice@example.com"


def test_user_gcp_email_none_by_default(db):
    """gcp_email is None for a user that hasn't logged in via GCP."""
    db.ensure_user("bob@example.com", Timestamp.now())
    assert db.get_user_gcp_email("bob@example.com") is None


# ---------------------------------------------------------------------------
# 3. Task dispatch: impersonation uses gcp_email
# ---------------------------------------------------------------------------


def _register_worker(state, worker_id: WorkerId, now: Timestamp):
    state.register_or_refresh_worker(
        worker_id=worker_id,
        address=f"{worker_id}:8080",
        metadata=cluster_pb2.WorkerMetadata(
            hostname=str(worker_id),
            ip_address="127.0.0.1",
            cpu_count=8,
            memory_bytes=16 * 1024**3,
            disk_bytes=100 * 1024**3,
        ),
        ts=now,
    )


def _submit_and_dispatch(state, user: str, worker_id: WorkerId, now: Timestamp):
    job_id = JobName.root(user, "test-job")
    req = cluster_pb2.Controller.LaunchJobRequest(
        name=job_id.to_wire(),
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    state.submit_job(job_id, req, now)
    task_id = job_id.task(0)
    state.queue_assignments([Assignment(task_id=task_id, worker_id=worker_id)])
    return state.drain_dispatch(worker_id)


def test_dispatch_impersonates_gcp_email_for_non_admin(state):
    """Non-admin user's gcp_email is used as impersonate_service_account in RunTaskRequest."""
    now = Timestamp.now()
    user = "alice@example.com"
    worker_id = WorkerId("w1")

    state._db.ensure_user(user, now, role="user")
    state._db.set_user_gcp_email(user, user)
    _register_worker(state, worker_id, now)

    batch = _submit_and_dispatch(state, user, worker_id, now)
    assert batch is not None
    assert len(batch.tasks_to_run) == 1
    assert batch.tasks_to_run[0].impersonate_service_account == user


def test_dispatch_skips_impersonation_for_admin(state):
    """Admin user does not get impersonation even if gcp_email is set."""
    now = Timestamp.now()
    user = "admin@example.com"
    worker_id = WorkerId("w1")

    state._db.ensure_user(user, now, role="admin")
    state._db.set_user_role(user, "admin")
    state._db.set_user_gcp_email(user, user)
    _register_worker(state, worker_id, now)

    batch = _submit_and_dispatch(state, user, worker_id, now)
    assert batch is not None
    assert len(batch.tasks_to_run) == 1
    assert batch.tasks_to_run[0].impersonate_service_account == ""


def test_dispatch_skips_impersonation_when_no_gcp_email(state):
    """Non-admin user without gcp_email gets no impersonation."""
    now = Timestamp.now()
    user = "bob@example.com"
    worker_id = WorkerId("w1")

    state._db.ensure_user(user, now, role="user")
    _register_worker(state, worker_id, now)

    batch = _submit_and_dispatch(state, user, worker_id, now)
    assert batch is not None
    assert len(batch.tasks_to_run) == 1
    assert batch.tasks_to_run[0].impersonate_service_account == ""
