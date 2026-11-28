"""Behavior tests for runtime source selection inside an Iris GPU task."""

import contextlib
import os
import signal
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from cloud.iris import task_runtime
from marinskyrl.environment_contract import DEBUG_ARTIFACT_DIR_ENV

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
        "stdout": task_runtime.subprocess.PIPE,
        "stderr": task_runtime.subprocess.STDOUT,
    }


def test_head_returns_driver_abort_when_failure_artifact_upload_blocks(tmp_path, monkeypatch, iris_ray_ports) -> None:
    upload_started = threading.Event()
    release_upload = threading.Event()

    class BlockingWrite:
        def __enter__(self):
            upload_started.set()
            release_upload.wait()
            return self

        def __exit__(self, *_args):
            return False

        def write(self, _payload):
            return None

    class BlockingFilesystem:
        def open(self, _path, _mode):
            return BlockingWrite()

    args = _runtime_args(tmp_path, rendezvous_dir="s3://incident/rendezvous")
    _isolate_head_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(task_runtime, "FAILURE_ARTIFACT_TIMEOUT", 0.01)
    monkeypatch.setattr(task_runtime, "fs_and_path", lambda _uri: (BlockingFilesystem(), "incident/report"))

    result = []

    def run_head():
        result.append(_run_head_driver(args, "abort", monkeypatch, iris_ray_ports))

    runtime_thread = threading.Thread(target=run_head)
    runtime_thread.start()
    try:
        assert upload_started.wait(timeout=5)
        runtime_thread.join(timeout=1)
        assert result == [128 + signal.SIGABRT]
    finally:
        release_upload.set()
        runtime_thread.join(timeout=5)


def _runtime_args(
    tmp_path: Path,
    *,
    liveness_timeout: float = 0,
    rendezvous_dir: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        ray_log_dir=None,
        rendezvous_dir=rendezvous_dir,
        ray_spill_backend=task_runtime.RaySpillBackend.LOCAL,
        ray_spill_dir=str(tmp_path / "spill"),
        cluster_join_timeout=1,
        ray_labels={},
        role_topology=None,
        driver_liveness_timeout=liveness_timeout,
    )


def _isolate_head_runtime(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(DEBUG_ARTIFACT_DIR_ENV, str(tmp_path / "debug"))
    monkeypatch.setattr(task_runtime, "DRIVER_WATCHDOG_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(task_runtime, "FAILURE_ARTIFACT_TIMEOUT", 2)
    monkeypatch.setattr(task_runtime, "_num_tasks", lambda: 1)
    monkeypatch.setattr(task_runtime, "_own_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(task_runtime, "ray_start_head", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(task_runtime, "ray_stop", lambda: None)
    monkeypatch.setattr(task_runtime.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(task_runtime, "training_driver_env", lambda _ifname: {"SKYRL_HOME": str(tmp_path)})
    monkeypatch.setattr(task_runtime, "ray_metrics_telemetry", lambda *_args: contextlib.nullcontext())
    # Do not let host-specific /tmp, GPU, process, or dmesg scans consume the fixture's
    # bounded artifact-persistence window.
    monkeypatch.setattr(
        task_runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="diagnostic output"),
    )


def _run_head_driver(args: SimpleNamespace, behavior: str, monkeypatch, ports) -> int:
    def launch(_config_path, environment):
        # run_head starts the liveness clock when launch returns. Hold the return until the
        # child interpreter has started so its startup time, which is unbounded on a loaded
        # host, is not charged against the test's short liveness windows.
        ready_read, ready_write = os.pipe()
        try:
            process = task_runtime.subprocess.Popen(
                [sys.executable, "-I", "-m", "cloud.iris.tests.watchdog_driver", behavior, str(ready_write)],
                env=environment,
                start_new_session=True,
                stdout=task_runtime.subprocess.PIPE,
                stderr=task_runtime.subprocess.STDOUT,
                pass_fds=(ready_write,),
            )
        finally:
            os.close(ready_write)
        with os.fdopen(ready_read, "rb") as ready:
            ready.read(1)
        return process

    monkeypatch.setattr(task_runtime, "launch_training_driver", launch)
    return task_runtime.run_head(args, Path("/tmp/unused-launch-config.yaml"), ports=ports)


def test_head_kills_a_silent_driver_and_records_the_stall_reason(tmp_path, monkeypatch, iris_ray_ports) -> None:
    _isolate_head_runtime(tmp_path, monkeypatch)
    rendezvous_dir = tmp_path / "rendezvous"
    (rendezvous_dir / "term_artifacts").mkdir(parents=True)
    exit_code = _run_head_driver(
        _runtime_args(tmp_path, liveness_timeout=0.1, rendezvous_dir=str(rendezvous_dir)),
        "stall",
        monkeypatch,
        iris_ray_ports,
    )

    artifacts = list((rendezvous_dir / "term_artifacts").glob("*.txt"))
    assert exit_code == task_runtime.DRIVER_STALLED_EXIT_CODE
    assert len(artifacts) == 1
    assert "driver stalled" in artifacts[0].read_text()


def test_head_keeps_a_driver_alive_while_it_emits_progress(tmp_path, monkeypatch, iris_ray_ports) -> None:
    _isolate_head_runtime(tmp_path, monkeypatch)
    # The driver runs well past the liveness window, but no single silence exceeds it.
    exit_code = _run_head_driver(_runtime_args(tmp_path, liveness_timeout=0.2), "progress", monkeypatch, iris_ray_ports)

    assert exit_code == 0


def test_head_allows_a_silent_driver_when_the_watchdog_is_disabled(tmp_path, monkeypatch, iris_ray_ports) -> None:
    _isolate_head_runtime(tmp_path, monkeypatch)
    exit_code = _run_head_driver(_runtime_args(tmp_path, liveness_timeout=0), "silent", monkeypatch, iris_ray_ports)

    assert exit_code == 0
