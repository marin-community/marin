# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
from iris.cluster.client.job_info import JobInfo
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.types import JobName
from iris.hooks.multigpu import IRIS_MULTIGPU_PROCESS_INDEX_ENV
from iris.runtime import telemetry
from rigging import telemetry as rigging_telemetry
from rigging.testing import RecordingTelemetryTransport


class _FakeClient:
    def resolve_endpoint(self, name: str) -> str:
        assert name == LOG_SERVER_ENDPOINT_NAME
        return "http://finelog:10001/"


class _FakeCtx:
    client = _FakeClient()


@pytest.fixture(autouse=True)
def reset_telemetry() -> Iterator[None]:
    rigging_telemetry.shutdown(0.01)
    yield
    rigging_telemetry.shutdown(0.1)


def _transport(monkeypatch: pytest.MonkeyPatch) -> RecordingTelemetryTransport:
    transport = RecordingTelemetryTransport()
    monkeypatch.setattr(rigging_telemetry, "_RequestsTransport", lambda: transport)
    return transport


def test_configure_stamps_canonical_resource_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(IRIS_MULTIGPU_PROCESS_INDEX_ENV, "2")
    monkeypatch.setenv("IRIS_NODE_NAME", "g83d142")
    info = JobInfo(
        task_id=JobName.from_wire("/alice/train/worker/3"),
        worker_id="w-7",
        attempt_id=1,
        worker_region="us-east5",
    )
    monkeypatch.setattr(telemetry, "get_job_info", lambda: info)
    monkeypatch.setattr(telemetry, "get_iris_ctx", lambda: _FakeCtx())
    transport = _transport(monkeypatch)

    telemetry.configure(
        "levanter",
        root_run_uid="pretrain-42",
        attributes={"model_revision": "abc123", "role": rigging_telemetry.TelemetryRole.TRAINER.value},
    )
    rigging_telemetry.gauge("identity_probe").set(1)
    transport.record("identity_probe", {})

    assert transport.resources[-1] == {
        "service": "levanter",
        "attributes": {
            "job_id": "/alice/train/worker",
            "task_id": "/alice/train/worker/3",
            "attempt": "1",
            "worker": "w-7",
            "region": "us-east5",
            "process_index": "2",
            "node_name": "g83d142",
            "model_revision": "abc123",
            "role": "trainer",
            "root_run_uid": "pretrain-42",
            "execution_uid": "iris:/alice/train/worker/3:attempt:1",
        },
    }


def test_vllm_resource_exposes_serving_job_join(monkeypatch: pytest.MonkeyPatch) -> None:
    info = JobInfo(task_id=JobName.from_wire("/alice/serve/0"), worker_id="w-1", attempt_id=0)
    monkeypatch.setattr(telemetry, "get_job_info", lambda: info)
    monkeypatch.setattr(telemetry, "get_iris_ctx", lambda: _FakeCtx())
    transport = _transport(monkeypatch)

    telemetry.configure("vllm", attributes={"role": rigging_telemetry.TelemetryRole.INFERENCE.value})
    rigging_telemetry.gauge("identity_probe").set(1)
    transport.record("identity_probe", {})

    attributes = transport.resources[-1]["attributes"]
    assert attributes["serving_job_id"] == "/alice/serve"
    assert attributes["root_run_uid"] == "/alice/serve"


def test_configure_stamps_explicit_distributed_process_index(monkeypatch: pytest.MonkeyPatch) -> None:
    info = JobInfo(task_id=JobName.from_wire("/alice/train/7"), worker_id="w-1", attempt_id=2)
    monkeypatch.setattr(telemetry, "get_job_info", lambda: info)
    monkeypatch.setattr(telemetry, "get_iris_ctx", lambda: _FakeCtx())
    transport = _transport(monkeypatch)

    telemetry.configure("levanter", root_run_uid="run-42", process_index=0)
    rigging_telemetry.gauge("identity_probe").set(1)
    transport.record("identity_probe", {})

    attributes = transport.resources[-1]["attributes"]
    assert attributes["task_id"] == "/alice/train/7"
    assert attributes["attempt"] == "2"
    assert attributes["root_run_uid"] == "run-42"
    assert attributes["process_index"] == "0"


@pytest.mark.parametrize("ambiguous_name", ["run", "run_id"])
def test_configure_rejects_ambiguous_identity(monkeypatch: pytest.MonkeyPatch, ambiguous_name: str) -> None:
    info = JobInfo(task_id=JobName.from_wire("/alice/train/0"), worker_id="w-1", attempt_id=0)
    monkeypatch.setattr(telemetry, "get_job_info", lambda: info)
    monkeypatch.setattr(telemetry, "get_iris_ctx", lambda: _FakeCtx())
    transport = _transport(monkeypatch)

    telemetry.configure("levanter", attributes={ambiguous_name: "ambiguous"})
    rigging_telemetry.gauge("identity_probe").set(1)

    assert rigging_telemetry.runtime_status().configured is False
    assert transport.records == []


@pytest.mark.parametrize("accessor", ["get_job_info", "get_iris_ctx"])
def test_configure_contains_malformed_iris_metadata(monkeypatch: pytest.MonkeyPatch, accessor: str) -> None:
    info = JobInfo(task_id=JobName.from_wire("/alice/train/0"), worker_id="w-1", attempt_id=0)
    monkeypatch.setattr(telemetry, "get_job_info", lambda: info)
    monkeypatch.setattr(telemetry, "get_iris_ctx", lambda: _FakeCtx())

    def malformed_metadata() -> None:
        raise ValueError("malformed Iris metadata")

    monkeypatch.setattr(telemetry, accessor, malformed_metadata)

    # Telemetry metadata is best-effort and must not interrupt application startup.
    telemetry.configure("levanter")

    assert rigging_telemetry.runtime_status().configured is False
