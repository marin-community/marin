# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.client.job_info import JobInfo, resolve_job_user, set_job_info
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
