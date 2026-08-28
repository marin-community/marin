# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest
from iris.cluster.client.job_info import JobInfo, get_job_info, resolve_job_user, set_job_info
from iris.cluster.runtime.env import IRIS_JOB_SETUP_LAYERS_ENV, serialize_setup_layers
from iris.cluster.setup_scripts import EnvironmentLayer
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


def test_job_info_reads_inherited_environment_layers(monkeypatch):
    monkeypatch.setenv("IRIS_TASK_ID", "/alice/train/0:0")
    layers = [EnvironmentLayer.job_tree(setup="install profiler", activate="export PROFILE=1")]
    monkeypatch.setenv(IRIS_JOB_SETUP_LAYERS_ENV, serialize_setup_layers(layers))

    info = cast(JobInfo, get_job_info())

    assert info.setup_layers == layers


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
