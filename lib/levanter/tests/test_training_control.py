# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass
import json
import re
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest
from jax import numpy as jnp

import levanter.training_control as training_control
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import EndpointAccess, JobName
from levanter.checkpoint import Checkpointer
from levanter.training_control import TrainingDashboard
from levanter.trainer import TrainerConfig


@dataclass
class _TrainingConfig:
    trainer: TrainerConfig
    model_name: str = "tiny-model"
    api_key: str = "config-secret"


class _Registry:
    def __init__(self):
        self.name: str | None = None
        self.address: str | None = None
        self.access: int | None = None
        self.active = False

    @contextmanager
    def registered(self, name: str, address: str, metadata=None, access=None):
        self.name = name
        self.address = address
        self.access = access
        self.active = True
        try:
            yield "endpoint-id"
        finally:
            self.active = False


class _FailingRegistry:
    def __init__(self):
        self.called = False

    def registered(self, name: str, address: str, metadata=None, access=None):
        self.called = True
        raise RuntimeError("controller unavailable")


def test_training_dashboard_registers_redacted_status_page(monkeypatch, tmp_path):
    config = _TrainingConfig(trainer=TrainerConfig(id="config-run"))
    registry = _Registry()
    job_info = JobInfo(
        task_id=JobName.from_wire("/alice/parent/train/0"),
        advertise_host="127.0.0.1",
    )

    monkeypatch.setattr(training_control.jax, "process_index", lambda: 0)
    monkeypatch.setattr(training_control, "get_iris_ctx", lambda: SimpleNamespace(registry=registry))
    monkeypatch.setattr(training_control, "get_job_info", lambda: job_info)
    monkeypatch.setattr(
        training_control.os,
        "environ",
        {
            "VISIBLE_VALUE": "value <script>alert(1)</script>",
            "HF_TOKEN": "environment-secret",
            "IRIS_JOB_ENV": '{"HF_TOKEN": "nested-environment-secret"}',
            "MARIN_PROVENANCE": '{"argv": ["--token", "nested-provenance-secret"]}',
        },
    )
    checkpointer = Checkpointer(tmp_path / "checkpoints", None, [])

    with TrainingDashboard(config, checkpointer.request_checkpoint, "dashboard-run"):
        assert registry.active
        assert registry.address is not None
        with urlopen(registry.address, timeout=2) as response:
            body = response.read().decode()
            headers = response.headers

        assert registry.name == "/alice/parent/train/training-control"
        assert registry.access == EndpointAccess.ENDPOINT_ACCESS_LINK
        assert headers["Cache-Control"] == "no-store"
        assert "dashboard-run" in body
        assert "tiny-model" in body
        assert "VISIBLE_VALUE" in body
        assert "value &lt;script&gt;alert(1)&lt;/script&gt;" in body
        assert "HF_TOKEN" in body
        assert "config-secret" not in body
        assert "environment-secret" not in body
        assert "nested-environment-secret" not in body
        assert "nested-provenance-secret" not in body

    assert not registry.active


def test_training_dashboard_requests_persist_temporary_checkpoints(monkeypatch, tmp_path):
    config = _TrainingConfig(trainer=TrainerConfig(id="config-run"))
    registry = _Registry()
    job_info = JobInfo(
        task_id=JobName.from_wire("/alice/parent/train/0"),
        advertise_host="127.0.0.1",
    )

    monkeypatch.setattr(training_control.jax, "process_index", lambda: 0)
    monkeypatch.setattr(training_control, "get_iris_ctx", lambda: SimpleNamespace(registry=registry))
    monkeypatch.setattr(training_control, "get_job_info", lambda: job_info)
    checkpointer = Checkpointer(tmp_path / "checkpoints", None, [])

    with TrainingDashboard(config, checkpointer.request_checkpoint, "dashboard-run"):
        assert registry.address is not None
        with urlopen(registry.address, timeout=2) as response:
            body = response.read().decode()

        token_match = re.search(r'name="token" value="([^"]+)"', body)
        assert token_match is not None
        request = Request(registry.address, data=urlencode({"token": token_match.group(1)}).encode(), method="POST")
        with urlopen(request, timeout=2) as response:
            assert response.status == 202
        checkpointer.on_step(tree={"value": jnp.array(1)}, step=1)
        checkpointer.wait_until_finished()

        metadata = json.loads((tmp_path / "checkpoints" / "step-1" / "metadata.json").read_text())
        assert metadata["step"] == 1
        assert metadata["is_temporary"] is True

        programmatic_request = Request(
            registry.address + "/checkpoint",
            data=b"",
            headers={"X-Levanter-Training-Control": "request-checkpoint"},
            method="POST",
        )
        with urlopen(programmatic_request, timeout=2) as response:
            assert response.status == 202
        checkpointer.on_step(tree={"value": jnp.array(2)}, step=2)
        checkpointer.wait_until_finished()
        programmatic_metadata = json.loads((tmp_path / "checkpoints" / "step-2" / "metadata.json").read_text())
        assert programmatic_metadata["step"] == 2
        assert programmatic_metadata["is_temporary"] is True

        with pytest.raises(HTTPError) as error:
            urlopen(Request(registry.address, data=b"token=invalid", method="POST"), timeout=2)
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            urlopen(Request(registry.address + "/checkpoint", data=b"", method="POST"), timeout=2)
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            urlopen(
                Request(registry.address + "/unknown", data=urlencode({"token": token_match.group(1)}).encode()),
                timeout=2,
            )
        assert error.value.code == 404
        checkpointer.on_step(tree={"value": jnp.array(3)}, step=3)
        checkpointer.wait_until_finished()
        assert not (tmp_path / "checkpoints" / "step-3").exists()
    assert not registry.active


def test_training_dashboard_failure_does_not_stop_training(monkeypatch):
    config = _TrainingConfig(trainer=TrainerConfig(id="test-run"))
    registry = _FailingRegistry()
    job_info = JobInfo(
        task_id=JobName.from_wire("/alice/parent/train/0"),
        advertise_host="127.0.0.1",
    )

    monkeypatch.setattr(training_control.jax, "process_index", lambda: 0)
    monkeypatch.setattr(training_control, "get_iris_ctx", lambda: SimpleNamespace(registry=registry))
    monkeypatch.setattr(training_control, "get_job_info", lambda: job_info)

    with TrainingDashboard(config, lambda: None, "test-run"):
        pass

    assert registry.called
