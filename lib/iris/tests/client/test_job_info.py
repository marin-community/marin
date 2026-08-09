# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from iris.client.job_info import JobInfo, get_job_info, resolve_job_user, set_job_info
from iris.cluster.constraints import ConstraintMode, ConstraintOp
from iris.cluster.types import JobName


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


def test_worker_region_from_env(monkeypatch):
    """IRIS_WORKER_REGION is read into JobInfo.worker_region (regression for #5541)."""
    monkeypatch.setenv("IRIS_TASK_ID", "/test-user/my-job/0:1")
    monkeypatch.setenv("IRIS_WORKER_REGION", "us-central1")
    info = get_job_info()
    assert info is not None
    assert info.worker_region == "us-central1"


def test_worker_region_absent_when_env_not_set(monkeypatch):
    """worker_region is None when IRIS_WORKER_REGION is not set."""
    monkeypatch.setenv("IRIS_TASK_ID", "/test-user/my-job/0:1")
    monkeypatch.delenv("IRIS_WORKER_REGION", raising=False)
    info = get_job_info()
    assert info is not None
    assert info.worker_region is None


def test_constraints_from_task_environment(monkeypatch):
    monkeypatch.setenv("IRIS_TASK_ID", "/test-user/my-job/0:1")
    monkeypatch.setenv(
        "IRIS_JOB_CONSTRAINTS",
        json.dumps(
            [
                {
                    "key": "region",
                    "op": "in",
                    "values": ["us-east1", "us-west1"],
                    "mode": "preferred",
                }
            ]
        ),
    )

    info = get_job_info()

    assert info is not None
    assert len(info.constraints) == 1
    constraint = info.constraints[0]
    assert constraint.key == "region"
    assert constraint.op is ConstraintOp.IN
    assert constraint.mode is ConstraintMode.PREFERRED
    assert tuple(item.value for item in constraint.values) == ("us-east1", "us-west1")
