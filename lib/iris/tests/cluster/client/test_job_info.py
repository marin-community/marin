# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.client.job_info import JobInfo, get_job_info, resolve_job_user, set_job_info
from iris.cluster.types import JobName


def test_job_info_user_derives_from_task_id():
    info = JobInfo(task_id=JobName.from_wire("/alice/train/0"))
    assert info.user == "alice"


def test_resolve_job_user_prefers_explicit_value():
    assert resolve_job_user("alice") == "alice"


def test_resolve_job_user_uses_current_job_info_before_os_user(monkeypatch):
    set_job_info(JobInfo(task_id=JobName.from_wire("/alice/train/0")))
    monkeypatch.setattr("getpass.getuser", lambda: "local-user")
    assert resolve_job_user() == "alice"
    set_job_info(None)


def test_resolve_job_user_falls_back_to_os_user(monkeypatch):
    set_job_info(None)
    monkeypatch.setattr("getpass.getuser", lambda: "local-user")
    assert resolve_job_user() == "local-user"


def test_resolve_job_user_falls_back_to_root_when_os_user_lookup_fails(monkeypatch):
    set_job_info(None)

    def _raise():
        raise OSError("no passwd entry")

    monkeypatch.setattr("getpass.getuser", _raise)
    assert resolve_job_user() == "root"


def test_get_job_info_accepts_iris_task_id_env(monkeypatch):
    set_job_info(None)
    monkeypatch.delenv("IRIS_JOB_ID", raising=False)
    monkeypatch.setenv("IRIS_TASK_ID", "/alice/train/0:0")
    monkeypatch.setenv("IRIS_NUM_TASKS", "2")
    monkeypatch.setenv("IRIS_ATTEMPT_ID", "3")
    monkeypatch.setenv("IRIS_WORKER_ID", "worker-7")
    monkeypatch.setenv("IRIS_CONTROLLER_ADDRESS", "http://10.0.0.1:10000")

    info = get_job_info()

    assert info is not None
    assert info.task_id == JobName.from_wire("/alice/train/0")
    assert info.job_id == JobName.from_wire("/alice/train")
    assert info.num_tasks == 2
    assert info.attempt_id == 3
    assert info.worker_id == "worker-7"
    assert info.controller_address == "http://10.0.0.1:10000"

    set_job_info(None)
