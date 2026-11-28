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
from marin.inference.vllm_cache import VllmCompilationCacheReference


def _config(cpu: float = 8):
    return RemoteInferenceConfig(
        model=ServedModelConfig(weights="gs://models/model", tensor_parallel_size=16),
        engine=VllmEngineConfig(launcher=VllmLauncherType.TPU),
        iris=IrisConfig(
            worker_resources=ResourceConfig.with_tpu("v4-32", cpu=cpu),
            worker_environment=create_environment(extras=["tpu"]),
        ),
    )


def test_failed_ray_supervisor_reports_its_log(tmp_path):
    log = tmp_path / "supervisor.log"
    log.write_text("Ray Client port is occupied")
    process = MagicMock(returncode=1)
    process.poll.return_value = 1
    gang = iris_tpu_vllm.TpuRayGang(iris_vllm.IrisGangCoordinator("test", 0, 4), "10.0.0.1:12000", process, log)
    with pytest.raises(RuntimeError, match="Ray Client port is occupied"):
        gang.check_alive()


def _run_tpu_follower(monkeypatch, coordinator: MagicMock) -> tuple[list[dict], MagicMock]:
    """Run a follower ``run_iris_service`` with a fake coordinator and no real Ray."""
    config = _config(cpu=6.5)
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
    coordinator.vllm_primary_address.return_value = "127.0.0.1:12000"
    monkeypatch.setattr(iris_vllm, "_POLL_SECONDS", 0.001)
    process = MagicMock()
    process.poll.return_value = None
    commands = []
    real_popen = subprocess.Popen

    def start_process(command, **kwargs):
        # The vLLM backend imports GitPython lazily, which probes the git executable.
        if Path(command[0]).name == "git":
            return real_popen(command, **kwargs)
        # Exercise the actual constructed script invocation through argparse.
        # Without safe-path mode, sibling types.py shadows stdlib types before
        # the supervisor can start. Stop at --help so no Ray/model runs locally.
        with real_popen(
            [sys.executable, *command[command.index("python") + 1 : -2], "--help"],
            env=kwargs["env"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as probe:
            _, error = probe.communicate(timeout=10)
            assert probe.returncode == 0, error
        launch = json.loads(Path(command[-1]).read_text())
        commands.append(launch)
        Path(launch["ready_path"]).write_text("{}")
        return process

    monkeypatch.setattr(serving, "get_job_info", lambda: job_info)
    monkeypatch.setattr(iris_tpu_vllm, "get_job_info", lambda: job_info)
    monkeypatch.setattr(serving, "iris_ctx", lambda: context)
    monkeypatch.setattr(iris_vllm, "iris_ctx", lambda: context)
    monkeypatch.setattr(iris_vllm, "ActorClient", lambda *_args, **_kwargs: coordinator)
    monkeypatch.setattr(iris_tpu_vllm.subprocess, "Popen", start_process)

    serving.run_iris_service(service)
    registry.registered.assert_not_called()
    return commands, process


@pytest.mark.parametrize("transient", [False, True])
def test_iris_follower_only_starts_ray_and_does_not_publish_endpoint(monkeypatch, transient):
    coordinator = MagicMock()
    # The leader has no managed compilation cache, so the follower skips restore.
    coordinator.compilation_cache.return_value = (True, None)
    stopping = iris_vllm._GangState.STOPPING
    coordinator.state.side_effect = (
        [ConnectError(Code.UNAVAILABLE, "temporary outage"), stopping, stopping] if transient else [stopping, stopping]
    )
    coordinator.follower_stopped.side_effect = (
        [ConnectError(Code.NOT_FOUND, "temporary resolver gap"), None] if transient else [None]
    )

    commands, process = _run_tpu_follower(monkeypatch, coordinator)

    assert len(commands) == 1
    ray_start = tpu_ray_runtime.ray_start_command(
        **{key: commands[0][key] for key in ("host", "address", "leader", "cpus", "chips", "ports")}
    )
    # Ray gets the CPUs Iris granted (6.5 rounded down), not a fixed count.
    assert ray_start[ray_start.index("--num-cpus") + 1] == "6"
    coordinator.compilation_cache_restored.assert_not_called()
    assert coordinator.follower_stopped.call_count == (2 if transient else 1)
    coordinator.follower_stopped.assert_called_with(1)
    process.terminate.assert_called()


def test_iris_follower_restores_leader_cache_before_acknowledging(monkeypatch):
    events: list[str] = []
    reference = VllmCompilationCacheReference(
        cache_key="0" * 64,
        remote_cache_dir="gs://bucket/cache",
        local_cache_dir="/tmp/cache",
        identity={"launcher": "tpu", "compile": {"model_name_or_path": "m", "extra_cli_args": []}},
    )
    coordinator = MagicMock()
    # Not shared on the first poll, then shared: the follower keeps polling.
    coordinator.compilation_cache.side_effect = [(False, None), (True, reference.model_dump_json())]
    coordinator.compilation_cache_restored.side_effect = lambda *_: events.append("ack")

    def state():
        # The first two state reads happen while polling for the cache reference.
        events.append("state")
        return iris_vllm._GangState.WAITING if len(events) < 2 else iris_vllm._GangState.STOPPING

    coordinator.state.side_effect = state
    coordinator.follower_stopped.side_effect = lambda *_: events.append("stopped")
    cache = MagicMock(restored=True)
    cache.close.side_effect = lambda: events.append("close")

    def restore_reference(received):
        assert received == reference
        events.append("restore")
        return cache

    monkeypatch.setattr(iris_vllm.VllmCompilationCache, "restore_reference", restore_reference)
    monkeypatch.setattr(iris_vllm.time, "sleep", lambda _: None)

    _run_tpu_follower(monkeypatch, coordinator)

    coordinator.compilation_cache_restored.assert_called_once_with(1, True)
    assert events == ["state", "state", "restore", "ack", "state", "close", "stopped"]


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
                "cpus": 8,
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
