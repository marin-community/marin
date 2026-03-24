# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GCP service account impersonation across the stack.

Covers: env injection, DB roundtrip, service RPC authorization, task dispatch,
and K8s pod manifest.
"""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.transitions import Assignment, ControllerTransitions
from iris.cluster.log_store import LogStore
from iris.cluster.runtime.env import build_common_iris_env
from iris.cluster.types import JobName, WorkerId
from iris.rpc import cluster_pb2
from iris.rpc.auth import AuthzAction, VerifiedIdentity, _verified_identity, authorize
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
    """CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT is set when impersonation SA provided."""
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
        impersonate_service_account="user@project.iam.gserviceaccount.com",
    )
    assert env["CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT"] == "user@project.iam.gserviceaccount.com"


def test_build_common_iris_env_no_impersonation_when_empty():
    """CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT not set when SA is empty."""
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
# 2. DB roundtrip
# ---------------------------------------------------------------------------


def test_user_gcp_email_roundtrip(db):
    """gcp_email is stored and retrieved correctly."""
    db.ensure_user("alice@example.com", Timestamp.now())
    db.set_user_gcp_email("alice@example.com", "alice@example.com")
    assert db.get_user_gcp_email("alice@example.com") == "alice@example.com"


def test_user_service_account_roundtrip(db):
    """gcp_service_account is stored and retrieved correctly."""
    db.ensure_user("alice@example.com", Timestamp.now())
    db.set_user_service_account("alice@example.com", "sa@project.iam.gserviceaccount.com")
    assert db.get_user_service_account("alice@example.com") == "sa@project.iam.gserviceaccount.com"


def test_user_service_account_clear(db):
    """Empty string clears gcp_service_account."""
    db.ensure_user("alice@example.com", Timestamp.now())
    db.set_user_service_account("alice@example.com", "sa@project.iam.gserviceaccount.com")
    db.set_user_service_account("alice@example.com", "")
    assert db.get_user_service_account("alice@example.com") is None


def test_user_gcp_email_none_by_default(db):
    """gcp_email is None for a user that hasn't set it."""
    db.ensure_user("bob@example.com", Timestamp.now())
    assert db.get_user_gcp_email("bob@example.com") is None


def test_user_service_account_none_by_default(db):
    """gcp_service_account is None for a user that hasn't set it."""
    db.ensure_user("bob@example.com", Timestamp.now())
    assert db.get_user_service_account("bob@example.com") is None


# ---------------------------------------------------------------------------
# 3. Authorization: SetUserServiceAccount requires admin
# ---------------------------------------------------------------------------


def test_authorize_manage_other_keys_requires_admin():
    """MANAGE_OTHER_KEYS (used by SetUserServiceAccount) rejects non-admin users."""
    reset = _verified_identity.set(VerifiedIdentity(user_id="regular-user", role="user"))
    try:
        with pytest.raises(ConnectError) as exc_info:
            authorize(AuthzAction.MANAGE_OTHER_KEYS)
        assert exc_info.value.code == Code.PERMISSION_DENIED
    finally:
        _verified_identity.reset(reset)


def test_authorize_manage_other_keys_allows_admin():
    """MANAGE_OTHER_KEYS passes for admin users."""
    reset = _verified_identity.set(VerifiedIdentity(user_id="admin-user", role="admin"))
    try:
        identity = authorize(AuthzAction.MANAGE_OTHER_KEYS)
        assert identity.user_id == "admin-user"
    finally:
        _verified_identity.reset(reset)


# ---------------------------------------------------------------------------
# 4. Task dispatch: impersonation flows into RunTaskRequest
# ---------------------------------------------------------------------------


def test_dispatch_includes_impersonation_for_non_admin(state):
    """Non-admin user with a service account gets impersonate_service_account in RunTaskRequest."""
    now = Timestamp.now()
    user = "alice@example.com"
    sa = "alice-sa@project.iam.gserviceaccount.com"

    state._db.ensure_user(user, now, role="user")
    state._db.set_user_service_account(user, sa)

    worker_id = WorkerId("w1")
    state.register_or_refresh_worker(
        worker_id=worker_id,
        address="w1:8080",
        metadata=cluster_pb2.WorkerMetadata(
            hostname="w1",
            ip_address="127.0.0.1",
            cpu_count=8,
            memory_bytes=16 * 1024**3,
            disk_bytes=100 * 1024**3,
        ),
        ts=now,
    )

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

    batch = state.drain_dispatch(worker_id)
    assert batch is not None
    assert len(batch.tasks_to_run) == 1
    assert batch.tasks_to_run[0].impersonate_service_account == sa


def test_dispatch_skips_impersonation_for_admin(state):
    """Admin user does not get impersonate_service_account even if SA is set."""
    now = Timestamp.now()
    user = "admin@example.com"
    sa = "admin-sa@project.iam.gserviceaccount.com"

    state._db.ensure_user(user, now, role="admin")
    state._db.set_user_role(user, "admin")
    state._db.set_user_service_account(user, sa)

    worker_id = WorkerId("w1")
    state.register_or_refresh_worker(
        worker_id=worker_id,
        address="w1:8080",
        metadata=cluster_pb2.WorkerMetadata(
            hostname="w1",
            ip_address="127.0.0.1",
            cpu_count=8,
            memory_bytes=16 * 1024**3,
            disk_bytes=100 * 1024**3,
        ),
        ts=now,
    )

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

    batch = state.drain_dispatch(worker_id)
    assert batch is not None
    assert len(batch.tasks_to_run) == 1
    assert batch.tasks_to_run[0].impersonate_service_account == ""


def test_dispatch_skips_impersonation_when_no_sa(state):
    """Non-admin user without a service account gets no impersonation."""
    now = Timestamp.now()
    user = "bob@example.com"

    state._db.ensure_user(user, now, role="user")

    worker_id = WorkerId("w1")
    state.register_or_refresh_worker(
        worker_id=worker_id,
        address="w1:8080",
        metadata=cluster_pb2.WorkerMetadata(
            hostname="w1",
            ip_address="127.0.0.1",
            cpu_count=8,
            memory_bytes=16 * 1024**3,
            disk_bytes=100 * 1024**3,
        ),
        ts=now,
    )

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

    batch = state.drain_dispatch(worker_id)
    assert batch is not None
    assert len(batch.tasks_to_run) == 1
    assert batch.tasks_to_run[0].impersonate_service_account == ""
