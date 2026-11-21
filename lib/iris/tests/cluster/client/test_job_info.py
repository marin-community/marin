# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iris.cluster.client.job_info import JobInfo, get_job_info, resolve_job_user, set_job_info
from iris.cluster.runtime.env import build_common_iris_env
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from rigging.timing import Duration


@pytest.fixture(autouse=True)
def _reset_job_info():
    """Clear the JobInfo contextvar between tests so state doesn't leak."""
    set_job_info(None)
    yield
    set_job_info(None)


def test_job_info_user_derives_from_task_id():
    info = JobInfo(task_id=JobName.from_wire("/alice/train/0"))
    assert info.user == "alice"


def test_resolve_job_user_prefers_explicit_value():
    assert resolve_job_user("alice") == "alice"


def test_resolve_job_user_uses_current_job_info_before_os_user(monkeypatch):
    set_job_info(JobInfo(task_id=JobName.from_wire("/alice/train/0")))
    monkeypatch.setattr("getpass.getuser", lambda: "local-user")
    assert resolve_job_user() == "alice"


def test_resolve_job_user_falls_back_to_os_user(monkeypatch):
    monkeypatch.setattr("getpass.getuser", lambda: "local-user")
    assert resolve_job_user() == "local-user"


def test_resolve_job_user_falls_back_to_root_when_os_user_lookup_fails(monkeypatch):
    def _raise():
        raise OSError("no passwd entry")

    monkeypatch.setattr("getpass.getuser", _raise)
    assert resolve_job_user() == "root"


def test_runtime_metadata_round_trip_preserves_resolved_termination_grace(monkeypatch):
    environment = build_common_iris_env(
        task_id="/alice/train/0",
        attempt_id=2,
        attempt_uid="attempt-2",
        num_tasks=1,
        bundle_id="bundle-id",
        controller_address=None,
        task_token=None,
        environment=job_pb2.EnvironmentConfig(),
        constraints=(),
        ports=(),
        resources=None,
        termination_grace=Duration.from_ms(60123),
    )
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    info = get_job_info()

    assert info is not None
    assert info.termination_grace == Duration.from_ms(60123)
    assert info.task_id == JobName.from_wire("/alice/train/0")
    assert info.attempt_id == 2
