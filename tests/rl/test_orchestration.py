# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from types import SimpleNamespace

import pytest

from marin.rl.orchestration import _HostedRuntime, _run_rl_coordinator
from marin.rl.rl_job import RunConfig


class _FakeHostedActor:
    def __init__(self, name: str, shutdown_calls: list[str]):
        self.handle = name
        self._name = name
        self._shutdown_calls = shutdown_calls

    def shutdown(self) -> None:
        self._shutdown_calls.append(self._name)


class _FakeJobHandle:
    def __init__(self, job_id: str):
        self.job_id = job_id


class _FakeClient:
    def __init__(self):
        self.submissions = []

    def submit(self, request):
        self.submissions.append(request)
        return _FakeJobHandle(request.name)


class _FakeRLJob:
    def __init__(self, _config):
        pass

    def to_worker_configs(self):
        return _FakeWorkerConfig(seed=7, run_id="train"), _FakeWorkerConfig(seed=11, run_id="rollout")


@dataclasses.dataclass(frozen=True)
class _FakeWorkerConfig:
    seed: int
    run_id: str
    worker_index: int = 0


def test_run_rl_coordinator_shuts_down_hosted_actors_when_child_job_fails(monkeypatch):
    shutdown_calls: list[str] = []
    client = _FakeClient()
    hosted_actors = [
        _FakeHostedActor("curriculum", shutdown_calls),
        _FakeHostedActor("run-state", shutdown_calls),
        _FakeHostedActor("weight-transfer", shutdown_calls),
    ]
    config = SimpleNamespace(
        run_id="rl-test",
        pip_dependency_groups=["math"],
        run_config=RunConfig(
            train_tpu_type="v5p-8",
            inference_tpu_type="v5p-8",
            num_rollout_workers=1,
            regions=["us-central1"],
        ),
    )

    monkeypatch.setattr("marin.rl.orchestration.current_client", lambda: client)
    monkeypatch.setattr("marin.rl.orchestration.RLJob", _FakeRLJob)
    monkeypatch.setattr(
        "marin.rl.orchestration._create_runtime_handles",
        lambda _client, _config: _HostedRuntime(runtime=SimpleNamespace(), hosted_actors=hosted_actors),
    )
    monkeypatch.setattr(
        "marin.rl.orchestration.wait_all",
        lambda _jobs, raise_on_failure: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        _run_rl_coordinator(config)

    assert len(client.submissions) == 2
    assert shutdown_calls == ["weight-transfer", "run-state", "curriculum"]


def test_run_rl_coordinator_uses_run_config_ram_overrides(monkeypatch):
    client = _FakeClient()
    hosted_runtime = _HostedRuntime(runtime=SimpleNamespace(), hosted_actors=[])
    config = SimpleNamespace(
        run_id="rl-test",
        pip_dependency_groups=["math"],
        run_config=RunConfig(
            train_tpu_type="v5p-8",
            inference_tpu_type="v5p-8",
            num_rollout_workers=1,
            train_ram="300g",
            inference_ram="300g",
            regions=["us-central1"],
        ),
    )

    monkeypatch.setattr("marin.rl.orchestration.current_client", lambda: client)
    monkeypatch.setattr("marin.rl.orchestration.RLJob", _FakeRLJob)
    monkeypatch.setattr(
        "marin.rl.orchestration._create_runtime_handles",
        lambda _client, _config: hosted_runtime,
    )
    monkeypatch.setattr("marin.rl.orchestration.wait_all", lambda _jobs, raise_on_failure: None)

    _run_rl_coordinator(config)

    assert len(client.submissions) == 2
    assert client.submissions[0].resources.ram == "300g"
    assert client.submissions[1].resources.ram == "300g"
