# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Endpoint naming and finelog-forwarding wiring for the telltale server."""

import pytest
from iris.cluster.client.job_info import JobInfo
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.hooks.multigpu import IRIS_MULTIGPU_PROCESS_INDEX_ENV
from iris.cluster.types import JobName
from iris.runtime import telltale


@pytest.mark.parametrize(
    ("task_id", "process_index", "expected"),
    [
        ("/alice/train/worker/3", None, "telltale/worker/3"),
        ("/alice/train/0", None, "telltale/0"),
        ("/alice/train/0", "2", "telltale/0/2"),
    ],
)
def test_endpoint_name_identifies_the_process_within_its_namespace(monkeypatch, task_id, process_index, expected):
    monkeypatch.setattr(telltale, "get_job_info", lambda: JobInfo(task_id=JobName.from_wire(task_id)))
    if process_index is not None:
        monkeypatch.setenv(IRIS_MULTIGPU_PROCESS_INDEX_ENV, "2")

    assert telltale._endpoint_name() == expected


def test_identity_carries_the_job_root_worker_and_process_index(monkeypatch):
    monkeypatch.setenv(IRIS_MULTIGPU_PROCESS_INDEX_ENV, "2")
    info = JobInfo(
        task_id=JobName.from_wire("/alice/train/worker/3"),
        worker_id="w-7",
        attempt_id=1,
        worker_region="us-east5",
    )

    identity = telltale._identity(info)

    # job_id is the job root, not the task's immediate parent (.../worker).
    assert identity.job_id == "/alice/train"
    assert identity.task_index == 3
    assert identity.worker == "w-7"
    assert identity.attempt == 1
    assert identity.region == "us-east5"
    assert identity.process_index == 2


class _FakeClient:
    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint

    def resolve_endpoint(self, name: str) -> str:
        assert name == LOG_SERVER_ENDPOINT_NAME
        return self._endpoint


class _FakeCtx:
    def __init__(self, endpoint: str) -> None:
        self.client = _FakeClient(endpoint)


def test_start_forwarding_resolves_the_endpoint_and_hands_a_sink_to_telltale(monkeypatch):
    built: dict[str, object] = {}

    class _Sink:
        def __init__(self, endpoint: str) -> None:
            built["endpoint"] = endpoint

    def _capture(sink, *, identity):
        built["sink"] = sink
        built["identity"] = identity

    monkeypatch.setattr(telltale, "FinelogMetricSink", _Sink)
    monkeypatch.setattr(telltale.telltale, "start_forwarding", _capture)

    info = JobInfo(task_id=JobName.from_wire("/alice/train/0"), worker_id="w-1", attempt_id=0)
    telltale._start_forwarding(info, _FakeCtx("http://finelog:10001"))

    assert built["endpoint"] == "http://finelog:10001"
    assert isinstance(built["sink"], _Sink)
    assert built["identity"].job_id == "/alice/train"
    assert built["identity"].task_index == 0
