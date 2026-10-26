# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from fray.types import ResourceConfig, create_environment
from marin.inference import iris as serving
from marin.inference import iris_tpu_vllm, iris_vllm, tpu_ray_runtime
from marin.inference.config import (
    IrisConfig,
    RemoteInferenceConfig,
    ServedModelConfig,
    VllmEngineConfig,
    VllmLauncherType,
)


def _config():
    return RemoteInferenceConfig(
        model=ServedModelConfig(weights="gs://models/model", tensor_parallel_size=16),
        engine=VllmEngineConfig(launcher=VllmLauncherType.TPU),
        iris=IrisConfig(
            worker_resources=ResourceConfig.with_tpu("v4-32"), worker_environment=create_environment(extras=["tpu"])
        ),
    )


def test_failed_ray_supervisor_reports_its_log(tmp_path):
    log = tmp_path / "supervisor.log"
    log.write_text("Ray Client port is occupied")
    process = MagicMock(returncode=1)
    process.poll.return_value = 1
    gang = iris_tpu_vllm.TpuRayGang(True, "10.0.0.1:12000", process, log)
    with pytest.raises(RuntimeError, match="Ray Client port is occupied"):
        gang.check_alive()


@pytest.mark.parametrize("transient", [False, True])
def test_iris_follower_only_starts_ray_and_does_not_publish_endpoint(monkeypatch, transient):
    config = _config()
    service = serving.IrisServiceConfig(
        model=config.model, engine=config.engine, iris=config.iris, endpoint_name="/serve/test"
    )
    job_info = SimpleNamespace(
        num_tasks=4,
        task_index=1,
        job_id="serve-test",
        advertise_host="127.0.0.1",
        ports=dict(zip(iris_tpu_vllm.RAY_PORTS, [12000, 12001, 12002, 12003], strict=True)),
    )
    registry = MagicMock()
    context = SimpleNamespace(registry=registry, resolver=object())
    coordinator = MagicMock()
    coordinator.vllm_primary_address.return_value = "127.0.0.1:12000"
    coordinator.shutdown_requested.side_effect = (
        [ConnectError(Code.UNAVAILABLE, "temporary outage"), True] if transient else [True]
    )
    coordinator.follower_stopped.side_effect = (
        [ConnectError(Code.NOT_FOUND, "temporary resolver gap"), None] if transient else [None]
    )
    monkeypatch.setattr(iris_vllm, "_POLL_SECONDS", 0.001)
    process = MagicMock()
    process.poll.return_value = None
    commands = []
    real_popen = subprocess.Popen

    def start_process(command, **kwargs):
        # The vLLM backend imports GitPython lazily, which probes the git executable.
        if Path(command[0]).name == "git":
            return real_popen(command, **kwargs)
        commands.append(command)
        launch = json.loads(Path(command[-1]).read_text())
        Path(launch["ready_path"]).write_text("{}")
        return process

    monkeypatch.setattr(serving, "get_job_info", lambda: job_info)
    monkeypatch.setattr(iris_tpu_vllm, "get_job_info", lambda: job_info)
    monkeypatch.setattr(serving, "iris_ctx", lambda: context)
    monkeypatch.setattr(iris_vllm, "iris_ctx", lambda: context)
    monkeypatch.setattr(iris_vllm, "ActorClient", lambda *_args, **_kwargs: coordinator)
    monkeypatch.setattr(iris_tpu_vllm.subprocess, "Popen", start_process)

    serving.run_iris_service(service)

    assert len(commands) == 1
    registry.registered.assert_not_called()
    assert coordinator.follower_stopped.call_count == (2 if transient else 1)
    coordinator.follower_stopped.assert_called_with(1)
    process.terminate.assert_called()


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
