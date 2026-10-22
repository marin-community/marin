# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import threading
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fray.types import JobRequest, ResourceConfig, create_environment
from marin.inference import iris as serving
from marin.inference import iris_tpu_vllm
from marin.inference.config import (
    BrokerConfig,
    IrisConfig,
    LevanterEngineConfig,
    RemoteInferenceConfig,
    ServedModelConfig,
    VllmEngineConfig,
    VllmLauncherType,
)
from marin.inference import tpu_ray_runtime
from marin.inference.tpu_ray_runtime import ray_start_command


def _config():
    return RemoteInferenceConfig(
        model=ServedModelConfig(weights="gs://models/model", tensor_parallel_size=16),
        engine=VllmEngineConfig(launcher=VllmLauncherType.TPU),
        iris=IrisConfig(
            worker_resources=ResourceConfig.with_tpu("v4-32"), worker_environment=create_environment(extras=["tpu"])
        ),
    )


def test_multihost_requires_direct_tpu_vllm():
    config = _config()
    assert config.iris.worker_resources.replicas == 4
    with pytest.raises(ValueError, match="isolated TPU vLLM"):
        replace(config, engine=LevanterEngineConfig())
    with pytest.raises(ValueError, match="one instance and no broker"):
        replace(config, broker=BrokerConfig())


@pytest.mark.parametrize("leader", [False, True])
def test_ray_listeners_use_allocated_ports_and_random_worker_ports(leader):
    ports = dict(zip(iris_tpu_vllm.RAY_PORTS, [12000, 12001, 12002, 12003], strict=True))
    command = ray_start_command(host="10.0.0.2", address="10.0.0.1:12000", leader=leader, chips=4, ports=ports)
    assert command[command.index("--min-worker-port") + 1] == "0"
    assert command[command.index("--max-worker-port") + 1] == "0"
    assert command[command.index("--dashboard-agent-listen-port") + 1] == "12003"
    if leader:
        for flag, expected in [
            ("--port", "12000"),
            ("--ray-client-server-port", "12001"),
            ("--dashboard-port", "12002"),
        ]:
            assert command[command.index(flag) + 1] == expected
    else:
        assert command[command.index("--address") + 1] == "10.0.0.1:12000"
        assert "--head" not in command


def test_failed_ray_supervisor_reports_its_log(tmp_path):
    log = tmp_path / "supervisor.log"
    log.write_text("Ray Client port is occupied")
    process = MagicMock(returncode=1)
    process.poll.return_value = 1
    gang = iris_tpu_vllm.TpuRayGang(True, "10.0.0.1:12000", process, log)
    with pytest.raises(RuntimeError, match="Ray Client port is occupied"):
        gang.check_alive()


def test_evaluation_gang_requests_all_hosts_on_one_slice(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(iris_tpu_vllm, "iris_ctx", lambda: SimpleNamespace(client=client))
    monkeypatch.setattr(iris_tpu_vllm, "convert_entrypoint", lambda value: value)
    config = _config()
    request = JobRequest(
        name="serve-test",
        entrypoint=MagicMock(),
        resources=config.iris.worker_resources,
        environment=config.iris.worker_environment,
        max_retries_preemption=0,
    )
    iris_tpu_vllm.submit_tpu_gang(request)
    submission = client.submit.call_args.kwargs
    assert submission["replicas"] == 4
    assert submission["coscheduling"].group_by == "tpu-name"
    assert set(submission["ports"]) == set(iris_tpu_vllm.RAY_PORTS)
    assert submission["max_retries_preemption"] == 0


def test_followers_never_load_a_model_or_publish_an_endpoint(monkeypatch):
    config = _config()
    service = serving.IrisServiceConfig(
        model=config.model, engine=config.engine, iris=config.iris, endpoint_name="/serve/test"
    )

    @contextmanager
    def follower(*_args, **_kwargs):
        yield SimpleNamespace(leader=False)

    monkeypatch.setattr(serving, "tpu_ray_gang", follower)
    resolved = MagicMock(side_effect=AssertionError("follower must not load model"))
    register = MagicMock(side_effect=AssertionError("follower must not publish endpoint"))
    monkeypatch.setattr(serving, "_resolved_model", resolved)
    monkeypatch.setattr(serving, "_register_dashboard", register)
    serving._run_tpu_service(service)
    resolved.assert_not_called()
    register.assert_not_called()


def test_ray_head_failure_interrupts_blocked_initialization(monkeypatch, tmp_path):
    blocked = threading.Event()
    fake_ray = SimpleNamespace(init=lambda **_: blocked.wait(), shutdown=blocked.set)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(tpu_ray_runtime.signal, "signal", lambda *_: None)
    monkeypatch.setattr(
        tpu_ray_runtime,
        "ray_start_command",
        lambda **_: [
            sys.executable,
            "-c",
            "print('GCS port collision', flush=True); raise SystemExit(1)",
        ],
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "host": "127.0.0.1",
                "address": "127.0.0.1:12000",
                "leader": True,
                "chips": 4,
                "ports": {},
                "hosts": 4,
                "log_path": str(tmp_path / "ray.log"),
                "ready_path": str(tmp_path / "ready.json"),
                "timeout_seconds": 60,
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["tpu_ray_runtime.py", "--config", str(config)])
    with pytest.raises(RuntimeError, match="GCS port collision"):
        tpu_ray_runtime.main()
    assert blocked.is_set()
    assert not (tmp_path / "ready.json").exists()


def test_default_multihost_tp_counts_the_whole_slice(monkeypatch):
    monkeypatch.setattr("marin.inference.model_preparation.resolve_model_path", lambda *args: args[0])
    config = _config()
    model, chips = serving._resolved_model(replace(config.model, tensor_parallel_size=None), config.iris)
    assert chips == 16
    assert model.tensor_parallel_size == 16
