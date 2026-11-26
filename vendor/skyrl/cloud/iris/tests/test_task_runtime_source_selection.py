"""Behavior tests for runtime source selection inside an Iris GPU task."""

import contextlib
import signal
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from cloud.iris import task_runtime
from marinskyrl.environment_contract import DEBUG_ARTIFACT_DIR_ENV

pytestmark = pytest.mark.usefixtures("iris_ray_ports")


def test_training_driver_starts_from_the_immutable_runtime_checkout(monkeypatch) -> None:
    observed: dict[str, object] = {}
    process = object()

    def fake_popen(argv, **kwargs):
        observed["argv"] = argv
        observed.update(kwargs)
        return process

    monkeypatch.setattr(task_runtime.subprocess, "Popen", fake_popen)
    environment = {"SKYRL_HOME": "/app/marinskyrl", "PYTHONPATH": "/app/marinskyrl:/app"}

    config_path = Path("/tmp/launch.yaml")
    launched = task_runtime.launch_training_driver(config_path, environment)

    assert launched is process
    assert observed == {
        "argv": [sys.executable, "-m", "cloud.iris.training_driver", "--config", str(config_path)],
        "env": environment,
        "cwd": "/app/marinskyrl",
        "start_new_session": True,
    }


def test_head_returns_driver_abort_when_failure_diagnostics_block(tmp_path, monkeypatch) -> None:
    diagnostics_started = threading.Event()
    release_diagnostics = threading.Event()

    def block_diagnostics(*_args, **_kwargs):
        diagnostics_started.set()
        release_diagnostics.wait()

    args = _runtime_args(tmp_path)
    _isolate_head_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(task_runtime, "FAILURE_DIAGNOSTICS_TIMEOUT", 0.01)
    monkeypatch.setattr(task_runtime.subprocess, "run", block_diagnostics)

    result = []

    def run_head():
        result.append(_run_head_script(args, "import os; os.abort()", monkeypatch))

    runtime_thread = threading.Thread(target=run_head)
    runtime_thread.start()
    try:
        assert diagnostics_started.wait(timeout=5)
        runtime_thread.join(timeout=1)
        assert result == [128 + signal.SIGABRT]
    finally:
        release_diagnostics.set()
        runtime_thread.join(timeout=5)


def _runtime_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        rendezvous_dir=None,
        ray_spill_backend=task_runtime.RaySpillBackend.LOCAL,
        ray_spill_dir=str(tmp_path / "spill"),
        cluster_join_timeout=1,
        ray_labels={},
        role_topology=None,
    )


def _isolate_head_runtime(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(DEBUG_ARTIFACT_DIR_ENV, str(tmp_path / "debug"))
    monkeypatch.setattr(task_runtime, "_num_tasks", lambda: 1)
    monkeypatch.setattr(task_runtime, "_own_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(task_runtime, "ray_start_head", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(task_runtime, "ray_stop", lambda: None)
    monkeypatch.setattr(task_runtime.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(task_runtime, "training_driver_env", lambda _ifname: {"SKYRL_HOME": str(tmp_path)})
    monkeypatch.setattr(task_runtime, "ray_metrics_telemetry", lambda *_args: contextlib.nullcontext())


def _run_head_script(args: SimpleNamespace, script: str, monkeypatch) -> int:
    def launch(_config_path, environment):
        return task_runtime.subprocess.Popen([sys.executable, "-c", script], env=environment, start_new_session=True)

    monkeypatch.setattr(task_runtime, "launch_training_driver", launch)
    return task_runtime.run_head(args, Path("/tmp/unused-launch-config.yaml"))
