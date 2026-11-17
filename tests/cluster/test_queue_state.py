# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from iris.cluster.types import JobName
from iris.resources.state import TaskState

from tests.cluster import conftest
from tests.cluster.conftest import _task_left_queue


@pytest.mark.parametrize(
    "state,started",
    [(TaskState.PENDING, False), (TaskState.BUILDING, False), (TaskState.ASSIGNED, False), (TaskState.RUNNING, True)],
)
def test_typed_task_states_preserve_the_queue_runtime_boundary(state, started):
    client = SimpleNamespace(task_status=lambda task: SimpleNamespace(state=state))
    assert _task_left_queue(client, JobName.from_wire("/test/job/0")) is started


@pytest.mark.parametrize("in_job", [False, True])
def test_cluster_client_uses_task_identity_only_inside_iris(monkeypatch, in_job):
    task_client = object()
    desktop_client = object()
    info = SimpleNamespace(controller_address="http://in-cluster", bundle_id="bundle") if in_job else None
    monkeypatch.setattr(conftest, "get_job_info", lambda: info)
    monkeypatch.setattr(conftest.IrisClient, "in_cluster", lambda *a, **kw: nullcontext(task_client))
    monkeypatch.setattr(conftest.IrisClient, "remote", lambda *a, **kw: nullcontext(desktop_client))
    monkeypatch.setattr(
        conftest,
        "connect_controller",
        lambda **kw: nullcontext(SimpleNamespace(url="http://desktop", credentials=SimpleNamespace(iap_provider=None))),
    )
    with conftest.open_cluster_client(conftest.MARIN_TPU_CLUSTER) as client:
        assert client is (task_client if in_job else desktop_client)
