"""Behavior tests for runtime source selection inside an Iris GPU task."""

import contextlib
import json
import os
import signal
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from cloud.iris import task_runtime
from marinskyrl.driver_status import DRIVER_PROGRESS_PATH_ENV, DRIVER_TIMING_FILENAME
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

    def block_diagnostics(reason):
        diagnostics_started.set()
        release_diagnostics.wait()

    args = _runtime_args(tmp_path)
    _isolate_head_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(task_runtime, "FAILURE_DIAGNOSTICS_TIMEOUT", 0.01)
    monkeypatch.setattr(task_runtime, "log_termination_diagnostics", block_diagnostics)

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


def _runtime_args(
    tmp_path: Path,
    *,
    progress_timeout: float = 0,
    rendezvous_dir: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        rendezvous_dir=rendezvous_dir,
        ray_spill_backend=task_runtime.RaySpillBackend.LOCAL,
        ray_spill_dir=str(tmp_path / "spill"),
        cluster_join_timeout=1,
        ray_labels={},
        role_topology=None,
        driver_progress_timeout=progress_timeout,
    )


def _isolate_head_runtime(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(DEBUG_ARTIFACT_DIR_ENV, str(tmp_path / "debug"))
    monkeypatch.setattr(task_runtime, "DRIVER_WATCHDOG_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(task_runtime, "FAILURE_DIAGNOSTICS_TIMEOUT", 2)
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


def _run_head_script(args: SimpleNamespace, script: str, monkeypatch, *, trainer_timing_uri: str | None = None) -> int:
    def launch(_config_path, environment):
        # run_head starts the liveness clock when launch returns. Hold the return until the
        # child interpreter has started so its startup time, which is unbounded on a loaded
        # host, is not charged against the test's short liveness windows.
        ready_read, ready_write = os.pipe()
        handshake = f"import os\nos.write({ready_write}, b'r')\nos.close({ready_write})\n"
        try:
            process = task_runtime.subprocess.Popen(
                [sys.executable, "-c", handshake + script],
                env=environment,
                start_new_session=True,
                pass_fds=(ready_write,),
            )
        finally:
            os.close(ready_write)
        with os.fdopen(ready_read, "rb") as ready:
            ready.read(1)
        return process

    monkeypatch.setattr(task_runtime, "launch_training_driver", launch)
    return task_runtime.run_head(
        args, Path("/tmp/unused-launch-config.yaml"), trainer_timing_uri=trainer_timing_uri
    )


def test_head_persists_completed_trainer_timing(tmp_path, monkeypatch) -> None:
    _isolate_head_runtime(tmp_path, monkeypatch)
    timing = tmp_path / "trainer_timing.json"
    script = f"""
import os
from pathlib import Path

status = Path(os.environ[{DRIVER_PROGRESS_PATH_ENV!r}])
status.with_name({DRIVER_TIMING_FILENAME!r}).write_text('{{"elapsed_excluding_eval_seconds": 203.125}}')
print('task output: Training run completed: elapsed_excluding_eval_seconds=999.000')
"""

    exit_code = _run_head_script(_runtime_args(tmp_path), script, monkeypatch, trainer_timing_uri=str(timing))

    assert exit_code == 0
    assert json.loads(timing.read_text()) == {"elapsed_excluding_eval_seconds": 203.125}


def test_head_kills_a_stalled_driver_that_keeps_logging(tmp_path, monkeypatch, capfd) -> None:
    _isolate_head_runtime(tmp_path, monkeypatch)
    script = """
import os
import signal
import threading

def block_abort(*_args):
    while True:
        print("worker still logging", flush=True)
        threading.Event().wait(0.01)

signal.signal(signal.SIGABRT, block_abort)
print("training started", flush=True)
os.kill(os.getpid(), signal.SIGABRT)
"""

    exit_code = _run_head_script(
        _runtime_args(tmp_path, progress_timeout=0.1), script, monkeypatch
    )

    assert exit_code == task_runtime.DRIVER_STALLED_EXIT_CODE
    assert "TERMINATION ARTIFACT" in capfd.readouterr().out


def test_head_keeps_a_quiet_driver_alive_while_work_advances(tmp_path, monkeypatch) -> None:
    _isolate_head_runtime(tmp_path, monkeypatch)
    # The driver runs well past the liveness window, but no single silence exceeds it.
    script = f"""
import os
from pathlib import Path
import threading

status = Path(os.environ[{DRIVER_PROGRESS_PATH_ENV!r}])
for _ in range(6):
    status.touch()
    threading.Event().wait(0.05)
"""

    exit_code = _run_head_script(_runtime_args(tmp_path, progress_timeout=0.2), script, monkeypatch)

    assert exit_code == 0


def test_head_allows_a_silent_driver_when_the_watchdog_is_disabled(tmp_path, monkeypatch) -> None:
    _isolate_head_runtime(tmp_path, monkeypatch)
    script = """
import threading

threading.Event().wait(0.1)
"""

    exit_code = _run_head_script(_runtime_args(tmp_path, progress_timeout=0), script, monkeypatch)

    assert exit_code == 0
