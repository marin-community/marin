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


def test_worker_region_from_env(monkeypatch):
    """IRIS_WORKER_REGION is read into JobInfo.worker_region."""
    set_job_info(None)
    monkeypatch.setenv("IRIS_TASK_ID", "/test-user/my-job/0:1")
    monkeypatch.setenv("IRIS_WORKER_REGION", "us-central1")
    info = get_job_info()
    assert info is not None
    assert info.worker_region == "us-central1"
    set_job_info(None)


def test_worker_region_absent_when_env_not_set(monkeypatch):
    """worker_region is None when IRIS_WORKER_REGION is not set."""
    set_job_info(None)
    monkeypatch.setenv("IRIS_TASK_ID", "/test-user/my-job/0:1")
    monkeypatch.delenv("IRIS_WORKER_REGION", raising=False)
    info = get_job_info()
    assert info is not None
    assert info.worker_region is None
    set_job_info(None)
