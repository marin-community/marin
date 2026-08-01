# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iris.cluster.client.job_info import JobInfo
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.types import JobName
from iris.hooks.multigpu import IRIS_MULTIGPU_PROCESS_INDEX_ENV
from iris.runtime import telemetry


class _FakeClient:
    def resolve_endpoint(self, name: str) -> str:
        assert name == LOG_SERVER_ENDPOINT_NAME
        return "http://finelog:10001/"


class _FakeCtx:
    client = _FakeClient()


def test_configure_resolves_direct_endpoint_and_stamps_resource_identity(monkeypatch):
    calls = []
    monkeypatch.setenv(IRIS_MULTIGPU_PROCESS_INDEX_ENV, "2")
    info = JobInfo(
        task_id=JobName.from_wire("/alice/train/worker/3"),
        worker_id="w-7",
        attempt_id=1,
        worker_region="us-east5",
    )
    monkeypatch.setattr(telemetry, "get_job_info", lambda: info)
    monkeypatch.setattr(telemetry, "get_iris_ctx", lambda: _FakeCtx())
    monkeypatch.setattr(telemetry.telemetry, "configure", lambda **kwargs: calls.append(kwargs))

    telemetry.configure(
        "levanter",
        role=telemetry.TelemetryRole.TRAINER,
        root_run_uid="pretrain-42",
        attributes={"model_revision": "abc123"},
    )

    assert calls == [
        {
            "endpoint": "http://finelog:10001/v1/telemetry",
            "service": "levanter",
            "attributes": {
                "job_id": "/alice/train/worker",
                "task_id": "/alice/train/worker/3",
                "attempt": "1",
                "worker": "w-7",
                "region": "us-east5",
                "process_index": "2",
                "model_revision": "abc123",
                "role": "trainer",
                "root_run_uid": "pretrain-42",
                "execution_uid": "iris:/alice/train/worker/3:attempt:1",
            },
        }
    ]


def test_vllm_resource_exposes_serving_job_join(monkeypatch):
    calls = []
    info = JobInfo(task_id=JobName.from_wire("/alice/serve/0"), worker_id="w-1", attempt_id=0)
    monkeypatch.setattr(telemetry, "get_job_info", lambda: info)
    monkeypatch.setattr(telemetry, "get_iris_ctx", lambda: _FakeCtx())
    monkeypatch.setattr(telemetry.telemetry, "configure", lambda **kwargs: calls.append(kwargs))

    telemetry.configure("vllm", role=telemetry.TelemetryRole.INFERENCE)

    assert calls[0]["attributes"]["serving_job_id"] == "/alice/serve"
    assert calls[0]["attributes"]["root_run_uid"] == "/alice/serve"


@pytest.mark.parametrize("ambiguous_name", ["run", "run_id"])
def test_configure_rejects_ambiguous_or_overridden_identity(monkeypatch, ambiguous_name):
    calls = []
    info = JobInfo(task_id=JobName.from_wire("/alice/train/0"), worker_id="w-1", attempt_id=0)
    monkeypatch.setattr(telemetry, "get_job_info", lambda: info)
    monkeypatch.setattr(telemetry, "get_iris_ctx", lambda: _FakeCtx())
    monkeypatch.setattr(telemetry.telemetry, "configure", lambda **kwargs: calls.append(kwargs))

    telemetry.configure(
        "levanter",
        role=telemetry.TelemetryRole.TRAINER,
        attributes={ambiguous_name: "ambiguous"},
    )

    assert calls == []


@pytest.mark.parametrize("accessor", ["get_job_info", "get_iris_ctx"])
def test_configure_contains_malformed_iris_metadata(monkeypatch, accessor):
    calls = []
    info = JobInfo(task_id=JobName.from_wire("/alice/train/0"), worker_id="w-1", attempt_id=0)
    monkeypatch.setattr(telemetry, "get_job_info", lambda: info)
    monkeypatch.setattr(telemetry, "get_iris_ctx", lambda: _FakeCtx())

    def malformed_metadata():
        raise ValueError("malformed Iris metadata")

    monkeypatch.setattr(telemetry, accessor, malformed_metadata)
    monkeypatch.setattr(telemetry.telemetry, "configure", lambda **kwargs: calls.append(kwargs))

    telemetry.configure("levanter", role=telemetry.TelemetryRole.TRAINER)

    # Telemetry metadata is best-effort and must not interrupt application startup.
    assert calls == []
