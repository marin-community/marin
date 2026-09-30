"""Controller launch and timeout behavior; no generated code executes here."""

import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

from capability_pipeline import grading_diagnostics as diagnostics


def test_default_runner_is_thread_safe_and_uses_pinned_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPABILITY_REMOTE_REGRADE", "1")
    monkeypatch.setenv("DAYTONA_API_KEY", "fixture-only")
    monkeypatch.setenv("CAPABILITY_REGRADE_INNER", "1")
    calls = {}

    class Child:
        pid = 812345

        def wait(self, timeout=None):
            return 2

    def spawn(command, **kwargs):
        calls.update(command=command, **kwargs)
        return Child()

    monkeypatch.setattr(diagnostics.subprocess, "Popen", spawn)
    signals = []
    monkeypatch.setattr(
        diagnostics.os, "killpg", lambda pid, sig: signals.append((pid, sig))
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(
            diagnostics._default_runner,
            plan_bundle=tmp_path / "plan-bundle",
            plan_sha256="a" * 64,
            output=tmp_path / "regrade",
            taskcompendium_source=tmp_path / "base",
            timeout_seconds=7,
        ).result()
    assert result == 2
    assert calls["command"][1:4] == ["-m", "capability_pipeline.cli", "regrade"]
    assert calls["start_new_session"] is True
    assert "CAPABILITY_REGRADE_INNER" not in calls["env"]
    assert signals == [(812345, signal.SIGTERM), (812345, signal.SIGKILL)]


def test_runner_timeout_kills_entire_process_group(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPABILITY_REMOTE_REGRADE", "1")
    monkeypatch.setenv("DAYTONA_API_KEY", "fixture-only")

    class Child:
        pid = 812346
        waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return -15

    child = Child()
    monkeypatch.setattr(diagnostics.subprocess, "Popen", lambda *a, **k: child)
    signals = []
    monkeypatch.setattr(
        diagnostics.os, "killpg", lambda pid, sig: signals.append((pid, sig))
    )
    with pytest.raises(subprocess.TimeoutExpired):
        diagnostics._default_runner(
            plan_bundle=tmp_path / "plan",
            plan_sha256="a" * 64,
            output=tmp_path / "regrade",
            taskcompendium_source=tmp_path / "base",
            timeout_seconds=1,
        )
    assert signals == [(child.pid, signal.SIGTERM), (child.pid, signal.SIGKILL)]
    assert child.waits == 3


def test_default_runner_refuses_local_task_execution(tmp_path, monkeypatch):
    monkeypatch.delenv("CAPABILITY_REMOTE_REGRADE", raising=False)
    with pytest.raises(RuntimeError, match="remote worker"):
        diagnostics._default_runner(
            plan_bundle=tmp_path / "plan",
            plan_sha256="a" * 64,
            output=tmp_path / "regrade",
            taskcompendium_source=tmp_path / "base",
            timeout_seconds=1,
        )
    assert not (tmp_path / "controller-run.log").exists()
