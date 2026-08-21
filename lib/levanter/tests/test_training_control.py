# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from urllib.request import urlopen

import levanter.training_control as training_control
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import EndpointAccess, JobName
from levanter.training_control import TrainingControl
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


def test_training_control_registers_redacted_status_page(monkeypatch):
    config = _TrainingConfig(trainer=TrainerConfig(id="test-run"))
    registry = _Registry()
    job_info = JobInfo(
        task_id=JobName.from_wire("/alice/parent/train/0"),
        advertise_host="127.0.0.1",
        ports={training_control.TRAINING_CONTROL_PORT: 0},
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
    with TrainingControl(config):
        assert registry.active
        assert registry.address is not None
        with urlopen(registry.address, timeout=2) as response:
            body = response.read().decode()
            headers = response.headers

        assert registry.name == "/alice/parent/train/training-control"
        assert registry.access == EndpointAccess.ENDPOINT_ACCESS_PRIVATE
        assert headers["Cache-Control"] == "no-store"
        assert "test-run" in body
        assert "tiny-model" in body
        assert "VISIBLE_VALUE" in body
        assert "value &lt;script&gt;alert(1)&lt;/script&gt;" in body
        assert "HF_TOKEN" in body
        assert "config-secret" not in body
        assert "environment-secret" not in body
        assert "nested-environment-secret" not in body
        assert "nested-provenance-secret" not in body
    assert not registry.active


def test_training_control_failure_does_not_stop_training(monkeypatch):
    config = _TrainingConfig(trainer=TrainerConfig(id="test-run"))
    registry = _FailingRegistry()
    job_info = JobInfo(
        task_id=JobName.from_wire("/alice/parent/train/0"),
        advertise_host="127.0.0.1",
        ports={training_control.TRAINING_CONTROL_PORT: 0},
    )

    monkeypatch.setattr(training_control.jax, "process_index", lambda: 0)
    monkeypatch.setattr(training_control, "get_iris_ctx", lambda: SimpleNamespace(registry=registry))
    monkeypatch.setattr(training_control, "get_job_info", lambda: job_info)

    with TrainingControl(config):
        pass

    assert registry.called
